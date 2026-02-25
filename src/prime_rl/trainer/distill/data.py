from typing import TypedDict, cast

import torch
from datasets import Dataset, load_dataset
from jaxtyping import Bool, Int
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.trainer.distill.config import DistillDataConfig
from prime_rl.trainer.sft.config import LossMaskConfig
from prime_rl.trainer.sft.data import StatefulIterableDataset
from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import get_logger


class DistillBatch(TypedDict):
    # Student (without context)
    student_input_ids: Int[Tensor, "batch seq"]
    student_position_ids: Int[Tensor, "batch seq"]
    student_target_ids: Int[Tensor, "batch seq"]
    student_loss_mask: Bool[Tensor, "batch seq"]
    # Teacher (with context)
    teacher_input_ids: Int[Tensor, "batch seq"]
    teacher_position_ids: Int[Tensor, "batch seq"]
    teacher_target_ids: Int[Tensor, "batch seq"]
    teacher_loss_mask: Bool[Tensor, "batch seq"]


def _build_loss_mask(
    messages: list[dict],
    tokenizer: PreTrainedTokenizer,
    loss_mask_config: LossMaskConfig,
) -> list[bool]:
    """Build per-token loss mask using incremental tokenization."""
    loss_mask: list[bool] = []
    prev_ids, prev_len = [], 0
    for i, message in enumerate(messages):
        should_mask = {
            "system": loss_mask_config.system,
            "user": loss_mask_config.user,
            "assistant": loss_mask_config.assistant,
            "tool": loss_mask_config.tool,
        }.get(message["role"], False)

        cur_ids = tokenizer.apply_chat_template(
            messages[: i + 1],
            add_generation_prompt=(
                message["role"] in ["user", "tool"]
                and i + 1 < len(messages)
                and messages[i + 1]["role"] == "assistant"
            ),
        )
        assert prev_ids == cur_ids[:prev_len], (
            f"Mismatch in incremental tokenization at message {i}"
        )
        loss_mask.extend([should_mask] * (len(cur_ids) - prev_len))
        prev_ids, prev_len = cur_ids, len(cur_ids)

    return loss_mask


def _tokenize_messages(
    messages: list[dict],
    tokenizer: PreTrainedTokenizer,
    loss_mask_config: LossMaskConfig,
    seq_len: int,
) -> dict | None:
    """Tokenize a message list and build loss mask. Returns None if too long or no trainable tokens."""
    logger = get_logger()

    input_ids = cast(
        list[int],
        tokenizer.apply_chat_template(messages),
    )
    loss_mask = _build_loss_mask(messages, tokenizer, loss_mask_config)

    # Append EOS if missing
    if tokenizer.eos_token_id not in input_ids:
        input_ids.append(cast(int, tokenizer.eos_token_id))
        loss_mask.append(True)

    # Shift for next-token prediction
    target_ids = input_ids[1:]
    loss_mask = loss_mask[1:]
    input_ids = input_ids[:-1]

    # Check length
    if len(input_ids) > seq_len:
        return None

    # Check trainable tokens exist
    if sum(loss_mask[:seq_len]) == 0:
        return None

    # Pad to seq_len
    pad_len = seq_len - len(input_ids)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    input_ids = input_ids + [pad_id] * pad_len
    target_ids = target_ids + [pad_id] * pad_len
    loss_mask = loss_mask + [False] * pad_len
    position_ids = list(range(seq_len))

    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "loss_mask": loss_mask,
        "position_ids": position_ids,
    }


class DistillDataset(StatefulIterableDataset):
    """Dataset that produces paired (student, teacher) tokenizations for context distillation."""

    def __init__(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        soul_doc: str,
        loss_mask_config: LossMaskConfig = LossMaskConfig(),
        shuffle: bool = True,
        seed: int = 0,
        seq_len: int = 2048,
        max_epochs: int | None = None,
    ):
        super().__init__()
        self.logger = get_logger()
        self.dataset = dataset
        self.num_examples = len(self.dataset)
        self.tokenizer = tokenizer
        self.soul_doc = soul_doc
        self.loss_mask_config = loss_mask_config
        self.shuffle = shuffle
        self.seed = seed
        self.seq_len = seq_len
        self.max_epochs = max_epochs

    def _process(self, example: dict) -> dict | None:
        """Process a single example into paired student/teacher tokenizations."""
        prompt_text = example["prompt"]
        response_text = example["response"]

        # Student messages (no system prompt)
        student_messages = [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": response_text},
        ]

        # Teacher messages (with soul doc as system prompt)
        teacher_messages = [
            {"role": "system", "content": self.soul_doc},
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": response_text},
        ]

        student = _tokenize_messages(student_messages, self.tokenizer, self.loss_mask_config, self.seq_len)
        teacher = _tokenize_messages(teacher_messages, self.tokenizer, self.loss_mask_config, self.seq_len)

        if student is None or teacher is None:
            return None

        # Verify alignment: both should have the same number of trainable tokens
        student_count = sum(student["loss_mask"])
        teacher_count = sum(teacher["loss_mask"])
        if student_count != teacher_count:
            self.logger.warning(
                f"Skipping example: student has {student_count} response tokens but teacher has {teacher_count}"
            )
            return None

        return {
            "student_input_ids": student["input_ids"],
            "student_target_ids": student["target_ids"],
            "student_loss_mask": student["loss_mask"],
            "student_position_ids": student["position_ids"],
            "teacher_input_ids": teacher["input_ids"],
            "teacher_target_ids": teacher["target_ids"],
            "teacher_loss_mask": teacher["loss_mask"],
            "teacher_position_ids": teacher["position_ids"],
        }

    def __iter__(self):
        dataset = self.dataset
        while True:
            if self.max_epochs is not None and self.epoch >= self.max_epochs:
                return

            if self.shuffle:
                dataset = dataset.shuffle(seed=self.seed + self.epoch)

            for i, example in enumerate(dataset):
                self.step += 1

                # Skip samples not assigned to this data rank
                if (self.step - 1) % self.data_world_size != self.data_rank:
                    continue

                result = self._process(example)
                if result is None:
                    continue

                self.num_samples["distill"] += 1
                self.num_tokens["distill"] += sum(result["student_loss_mask"])
                yield result

            self.epoch += 1


def _collate_distill(samples: list[dict]) -> DistillBatch:
    """Collate a list of distill samples into a batch."""
    return {
        "student_input_ids": torch.tensor([s["student_input_ids"] for s in samples]).long(),
        "student_position_ids": torch.tensor([s["student_position_ids"] for s in samples]).long(),
        "student_target_ids": torch.tensor([s["student_target_ids"] for s in samples]).long(),
        "student_loss_mask": torch.tensor([s["student_loss_mask"] for s in samples]).bool(),
        "teacher_input_ids": torch.tensor([s["teacher_input_ids"] for s in samples]).long(),
        "teacher_position_ids": torch.tensor([s["teacher_position_ids"] for s in samples]).long(),
        "teacher_target_ids": torch.tensor([s["teacher_target_ids"] for s in samples]).long(),
        "teacher_loss_mask": torch.tensor([s["teacher_loss_mask"] for s in samples]).bool(),
    }


def setup_dataset(tokenizer: PreTrainedTokenizer, config: DistillDataConfig, non_dp_size: int = 1) -> DistillDataset:
    """Load and prepare the distillation dataset."""
    dataset = load_dataset(config.name, split=config.split)
    assert isinstance(dataset, Dataset)

    soul_doc = config.get_soul_doc()

    return DistillDataset(
        dataset=dataset,
        tokenizer=tokenizer,
        soul_doc=soul_doc,
        loss_mask_config=config.loss_mask,
        shuffle=config.shuffle,
        seed=config.seed,
        seq_len=config.seq_len,
    )


def setup_dataloader(dataset: DistillDataset, config: DistillDataConfig) -> StatefulDataLoader:
    """Create a dataloader for the distillation dataset."""
    return StatefulDataLoader(
        dataset,
        batch_size=1,  # Each sample is already padded to seq_len
        collate_fn=_collate_distill,
        num_workers=0,
    )
