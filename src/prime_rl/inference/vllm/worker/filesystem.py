from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.nn import Module
from vllm.distributed.parallel_state import get_tp_group
from vllm.model_executor.model_loader import DefaultModelLoader, get_model_loader
from vllm.model_executor.model_loader.utils import process_weights_after_loading

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

    Worker = Worker
else:
    Worker = object


# Map from HF projection names to merged vLLM names and their order within the merged param
_QKV_PROJS = {"q_proj", "k_proj", "v_proj"}
_GATE_UP_PROJS = {"gate_proj", "up_proj"}
_MERGED_PROJS = _QKV_PROJS | _GATE_UP_PROJS
_ROW_PARALLEL_NAMES = {"o_proj", "down_proj"}


class FileSystemWeightUpdateWorker(Worker):
    """vLLM worker extension for updating weights via filesystem, with LoRA delta support."""

    def init_broadcaster(self) -> None:
        self._init_lora_state()

    def _init_lora_state(self) -> None:
        if not hasattr(self, "_lora_initialized"):
            self._base_weights: dict[str, torch.Tensor] = {}
            self._lora_initialized = False

    def _init_lora_base_weights(self, model: Module, lora_targets: list[str]) -> None:
        """Store a copy of the original base weights for parameters that will be modified."""
        self._base_weights = {}
        for name, param in model.named_parameters():
            # Match any param that contains a LoRA target name or is a merged param (qkv_proj, gate_up_proj)
            should_save = False
            for target in lora_targets:
                if target in name:
                    should_save = True
                    break
            # Also save merged projections that contain LoRA targets
            if not should_save:
                if "qkv_proj" in name and any(t in _QKV_PROJS for t in lora_targets):
                    should_save = True
                if "gate_up_proj" in name and any(t in {"gate_proj", "up_proj"} for t in lora_targets):
                    should_save = True
            if should_save:
                self._base_weights[name] = param.data.clone()
        self._lora_initialized = True

    def _compute_lora_delta(self, lora_a: torch.Tensor, lora_b: torch.Tensor,
                            scaling: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Compute LoRA delta: (B @ A) * scaling."""
        return (lora_b.to(device=device, dtype=dtype) @ lora_a.to(device=device, dtype=dtype)) * scaling

    def _apply_lora_delta(self, model: Module, lora_state: dict[str, torch.Tensor],
                          lora_config: dict, modules_to_save_state: dict[str, torch.Tensor]) -> None:
        """Apply LoRA deltas in-place, handling merged QKV projections."""
        tp_rank = get_tp_group().rank
        tp_size = get_tp_group().world_size
        scaling = lora_config["alpha"] / lora_config["rank"]

        param_dict = dict(model.named_parameters())

        # Group LoRA state by layer prefix
        # e.g. "model.layers.0.self_attn.q_proj.lora_A" -> layer="model.layers.0.self_attn", proj="q_proj"
        lora_by_layer: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
        for key in lora_state:
            if not key.endswith(".lora_A"):
                continue
            prefix = key.rsplit(".lora_A", 1)[0]  # "model.layers.0.self_attn.q_proj"
            parts = prefix.rsplit(".", 1)
            if len(parts) == 2:
                layer_prefix, proj_name = parts
            else:
                continue
            lora_a = lora_state[f"{prefix}.lora_A"]
            lora_b = lora_state[f"{prefix}.lora_B"]
            if layer_prefix not in lora_by_layer:
                lora_by_layer[layer_prefix] = {}
            lora_by_layer[layer_prefix][proj_name] = (lora_a, lora_b)

        applied = 0
        for layer_prefix, proj_dict in lora_by_layer.items():
            # Check if any projections are merged in vLLM
            qkv_projs = {p: ab for p, ab in proj_dict.items() if p in _QKV_PROJS}
            gate_up_projs = {p: ab for p, ab in proj_dict.items() if p in _GATE_UP_PROJS}
            other_projs = {p: ab for p, ab in proj_dict.items() if p not in _MERGED_PROJS}

            # Handle merged QKV
            if qkv_projs:
                qkv_name = f"{layer_prefix}.qkv_proj.weight"
                if qkv_name in self._base_weights and qkv_name in param_dict:
                    base = self._base_weights[qkv_name]
                    device, dtype = base.device, base.dtype
                    param = param_dict[qkv_name]

                    # Compute full (unsharded) output sizes for each projection
                    q_out = qkv_projs["q_proj"][1].shape[0] if "q_proj" in qkv_projs else 0
                    k_out = qkv_projs["k_proj"][1].shape[0] if "k_proj" in qkv_projs else 0
                    v_out = qkv_projs["v_proj"][1].shape[0] if "v_proj" in qkv_projs else 0

                    # Per-rank shard sizes
                    q_shard = q_out // tp_size if q_out > 0 else 0
                    k_shard = k_out // tp_size if k_out > 0 else 0
                    v_shard = v_out // tp_size if v_out > 0 else 0

                    new_weight = base.clone()
                    offset = 0
                    for proj_name, shard_size in [("q_proj", q_shard), ("k_proj", k_shard), ("v_proj", v_shard)]:
                        if proj_name in qkv_projs and shard_size > 0:
                            lora_a, lora_b = qkv_projs[proj_name]
                            lora_b_shard = lora_b[tp_rank * shard_size : (tp_rank + 1) * shard_size, :]
                            delta = self._compute_lora_delta(lora_a, lora_b_shard, scaling, device, dtype)
                            new_weight[offset : offset + shard_size] += delta
                        offset += shard_size

                    param.data.copy_(new_weight)
                    applied += 1

            # Handle merged gate_up_proj
            if gate_up_projs:
                gate_up_name = f"{layer_prefix}.gate_up_proj.weight"
                if gate_up_name in self._base_weights and gate_up_name in param_dict:
                    base = self._base_weights[gate_up_name]
                    device, dtype = base.device, base.dtype
                    param = param_dict[gate_up_name]

                    # gate and up have the same output size (ffn_hidden_size)
                    gate_out = gate_up_projs["gate_proj"][1].shape[0] if "gate_proj" in gate_up_projs else 0
                    up_out = gate_up_projs["up_proj"][1].shape[0] if "up_proj" in gate_up_projs else 0

                    gate_shard = gate_out // tp_size if gate_out > 0 else 0
                    up_shard = up_out // tp_size if up_out > 0 else 0

                    new_weight = base.clone()
                    offset = 0
                    for proj_name, shard_size in [("gate_proj", gate_shard), ("up_proj", up_shard)]:
                        if proj_name in gate_up_projs and shard_size > 0:
                            lora_a, lora_b = gate_up_projs[proj_name]
                            lora_b_shard = lora_b[tp_rank * shard_size : (tp_rank + 1) * shard_size, :]
                            delta = self._compute_lora_delta(lora_a, lora_b_shard, scaling, device, dtype)
                            new_weight[offset : offset + shard_size] += delta
                        offset += shard_size

                    param.data.copy_(new_weight)
                    applied += 1

            # Handle non-merged projections (o_proj, down_proj, etc.)
            for proj_name, (lora_a, lora_b) in other_projs.items():
                base_name = f"{layer_prefix}.{proj_name}.weight"
                if base_name not in self._base_weights or base_name not in param_dict:
                    continue

                base = self._base_weights[base_name]
                device, dtype = base.device, base.dtype

                if tp_size > 1 and proj_name in _ROW_PARALLEL_NAMES:
                    # Row parallel: input dim is sharded
                    in_features = lora_a.shape[1]
                    shard_size = in_features // tp_size
                    lora_a = lora_a[:, tp_rank * shard_size : (tp_rank + 1) * shard_size]

                delta = self._compute_lora_delta(lora_a, lora_b, scaling, device, dtype)
                param_dict[base_name].data.copy_(base + delta)
                applied += 1

        # Apply modules_to_save directly
        for name, new_weight in modules_to_save_state.items():
            if name in param_dict:
                param = param_dict[name]
                new_weight = new_weight.to(device=param.device, dtype=param.dtype)
                if new_weight.shape != param.data.shape:
                    if new_weight.dim() >= 1 and param.data.dim() >= 1:
                        shard_size = param.data.shape[0]
                        start = tp_rank * shard_size
                        if start + shard_size <= new_weight.shape[0]:
                            new_weight = new_weight[start : start + shard_size]
                        else:
                            # Pad: vLLM may pad vocab dim
                            padded = torch.zeros_like(param.data)
                            actual = min(new_weight.shape[0] - start, shard_size)
                            if actual > 0:
                                padded[:actual] = new_weight[start : start + actual]
                            new_weight = padded
                    if new_weight.shape != param.data.shape:
                        continue
                param.data.copy_(new_weight)

    def update_weights(self, weight_path: str) -> None:
        """Update weights from a specified path.

        Supports two modes:
        1. LoRA delta mode: if weight_path contains lora_delta.pt, applies LoRA deltas in-place
        2. Full weight mode: loads full weights via vLLM model loader (fallback)
        """
        self._init_lora_state()
        model_runner = self.model_runner
        model = model_runner.model
        assert isinstance(model, Module)

        weight_dir = Path(weight_path)
        lora_delta_path = weight_dir / "lora_delta.pt"

        if lora_delta_path.exists():
            checkpoint = torch.load(lora_delta_path, map_location="cpu", weights_only=True)
            lora_state = checkpoint["lora_state"]
            lora_config = checkpoint["lora_config"]
            modules_to_save_state = checkpoint.get("modules_to_save_state", {})
            lora_targets = checkpoint.get("lora_targets", ["q_proj", "k_proj", "v_proj", "o_proj"])

            if not self._lora_initialized:
                self._init_lora_base_weights(model, lora_targets)

            self._apply_lora_delta(model, lora_state, lora_config, modules_to_save_state)
        else:
            model_loader = get_model_loader(self.load_config)
            assert isinstance(model_loader, DefaultModelLoader)
            local_source = DefaultModelLoader.Source(
                weight_path,
                revision=None,
                prefix="",
                fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
                allow_patterns_overrides=getattr(model, "allow_patterns_overrides", None),
            )
            weights_iterator = model_loader._get_weights_iterator(local_source)
            model.load_weights(weights_iterator)

            device = next(model.parameters()).device
            process_weights_after_loading(model, self.model_runner.model_config, device)

            # Reset LoRA state — full load changed all weights, so base weights need re-capture
            self._lora_initialized = False
