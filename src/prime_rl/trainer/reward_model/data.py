from typing import cast

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.trainer.reward_model.config import RewardModelDataConfig
from prime_rl.trainer.sft.data import StatefulIterableDataset
from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import get_logger


class PreferenceBatch:
    """A batch of tokenized preference pairs."""

    def __init__(
        self,
        chosen_input_ids: torch.Tensor,
        chosen_position_ids: torch.Tensor,
        chosen_lengths: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        rejected_position_ids: torch.Tensor,
        rejected_lengths: torch.Tensor,
    ):
        self.chosen_input_ids = chosen_input_ids
        self.chosen_position_ids = chosen_position_ids
        self.chosen_lengths = chosen_lengths
        self.rejected_input_ids = rejected_input_ids
        self.rejected_position_ids = rejected_position_ids
        self.rejected_lengths = rejected_lengths


def _tokenize_for_reward(
    prompt: str,
    response: str,
    tokenizer: PreTrainedTokenizer,
    seq_len: int,
) -> dict | None:
    """Tokenize a (prompt, response) pair for reward model scoring.

    Returns the token IDs padded to seq_len and the true sequence length
    (position of the last real token, used for reward extraction).
    """
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]

    input_ids = cast(list[int], tokenizer.apply_chat_template(messages))

    # Append EOS if missing
    if tokenizer.eos_token_id not in input_ids:
        input_ids.append(cast(int, tokenizer.eos_token_id))

    if len(input_ids) > seq_len:
        return None

    true_length = len(input_ids)

    # Pad to seq_len
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    pad_len = seq_len - len(input_ids)
    input_ids = input_ids + [pad_id] * pad_len
    position_ids = list(range(seq_len))

    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "true_length": true_length,
    }


class PreferenceDataset(StatefulIterableDataset):
    """Dataset that yields tokenized preference pairs (chosen, rejected)."""

    def __init__(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        config: RewardModelDataConfig,
        max_epochs: int | None = None,
    ):
        super().__init__()
        self.logger = get_logger()
        self.dataset = dataset
        self.num_examples = len(self.dataset)
        self.tokenizer = tokenizer
        self.config = config
        self.max_epochs = max_epochs

    def _process(self, example: dict) -> dict | None:
        prompt = example[self.config.prompt_column]
        chosen = example[self.config.chosen_column]
        rejected = example[self.config.rejected_column]

        chosen_tok = _tokenize_for_reward(prompt, chosen, self.tokenizer, self.config.seq_len)
        rejected_tok = _tokenize_for_reward(prompt, rejected, self.tokenizer, self.config.seq_len)

        if chosen_tok is None or rejected_tok is None:
            return None

        return {
            "chosen_input_ids": chosen_tok["input_ids"],
            "chosen_position_ids": chosen_tok["position_ids"],
            "chosen_length": chosen_tok["true_length"],
            "rejected_input_ids": rejected_tok["input_ids"],
            "rejected_position_ids": rejected_tok["position_ids"],
            "rejected_length": rejected_tok["true_length"],
        }

    def __iter__(self):
        while True:
            if self.max_epochs is not None and self.epoch >= self.max_epochs:
                return

            cur_dataset = self.dataset
            if self.config.shuffle:
                cur_dataset = self.dataset.shuffle(seed=self.config.seed + self.epoch)

            for example in cur_dataset:
                self.step += 1

                if (self.step - 1) % self.data_world_size != self.data_rank:
                    continue

                result = self._process(example)
                if result is None:
                    continue

                self.num_samples["preference"] = self.num_samples.get("preference", 0) + 1
                yield result

            self.epoch += 1


def _collate_preference(samples: list[dict]) -> PreferenceBatch:
    return PreferenceBatch(
        chosen_input_ids=torch.tensor([s["chosen_input_ids"] for s in samples]).long(),
        chosen_position_ids=torch.tensor([s["chosen_position_ids"] for s in samples]).long(),
        chosen_lengths=torch.tensor([s["chosen_length"] for s in samples]).long(),
        rejected_input_ids=torch.tensor([s["rejected_input_ids"] for s in samples]).long(),
        rejected_position_ids=torch.tensor([s["rejected_position_ids"] for s in samples]).long(),
        rejected_lengths=torch.tensor([s["rejected_length"] for s in samples]).long(),
    )


def setup_dataset(
    tokenizer: PreTrainedTokenizer,
    config: RewardModelDataConfig,
) -> PreferenceDataset:
    if config.name.endswith(".jsonl") or config.name.endswith(".json"):
        dataset = load_dataset("json", data_files=config.name, split="train")
    else:
        dataset = load_dataset(config.name, split=config.split)
    assert isinstance(dataset, Dataset)
    return PreferenceDataset(dataset=dataset, tokenizer=tokenizer, config=config)


def setup_dataloader(
    dataset: PreferenceDataset,
    config: RewardModelDataConfig,
) -> StatefulDataLoader:
    return StatefulDataLoader(
        dataset,
        batch_size=config.batch_size,
        collate_fn=_collate_preference,
        num_workers=0,
    )
