# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Mapping
from enum import Enum
from typing import Any

import torch
from safetensors.torch import load_file
from vllm.logger import init_logger

logger = init_logger(__name__)

LORA_WEIGHT_NAME = "pytorch_lora_weights.bin"
LORA_WEIGHT_NAME_SAFE = "adapter_model.safetensors"


class LoRAFormat(Enum):
    """Supported LoRA formats."""

    STANDARD = "standard"
    NON_DIFFUSERS_SD = "non-diffusers-sd"
    KOHYA_FLUX = "kohya-flux"
    XLABS_FLUX = "xlabs-flux"
    WAN = "wan"
    QWEN_IMAGE = "qwen-image"


def _has_substring_key(keys: Mapping[str, Any], substr: str) -> bool:
    if isinstance(keys, dict):
        keys = keys.keys()
    return any(substr in k for k in keys)


def _has_prefix_key(keys: Mapping[str, Any], prefix: str) -> bool:
    if isinstance(keys, dict):
        keys = keys.keys()
    return any(k.startswith(prefix) for k in keys)


def _looks_like_xlabs_flux_key(k: str) -> bool:
    if not (k.endswith(".down.weight") or k.endswith(".up.weight")):
        return False
    if not k.startswith(
        ("double_blocks.", "single_blocks.", "diffusion_model.double_blocks", "diffusion_model.single_blocks")
    ):
        return False
    return ".processor." in k or ".proj_lora" in k or ".qkv_lora" in k


def _looks_like_kohya_flux(state_dict: Mapping[str, torch.Tensor]) -> bool:
    if not state_dict:
        return False
    keys = state_dict.keys()
    return any(k.startswith("lora_unet_double_blocks_") or k.startswith("lora_unet_single_blocks_") for k in keys)


def _looks_like_non_diffusers_sd(state_dict: Mapping[str, torch.Tensor]) -> bool:
    if not state_dict:
        return False
    keys = state_dict.keys()
    return all(k.startswith(("lora_unet_", "lora_te_", "lora_te1_", "lora_te2_")) for k in keys)


def _looks_like_wan_lora(state_dict: Mapping[str, torch.Tensor]) -> bool:
    if not state_dict:
        return False
    for k in state_dict.keys():
        if not k.startswith("diffusion_model.blocks."):
            continue
        if ".lora_down" not in k and ".lora_up" not in k:
            continue
        if ".cross_attn." in k or ".self_attn." in k or ".ffn." in k or ".norm3." in k:
            return True
    return False


def _looks_like_qwen_image(state_dict: Mapping[str, torch.Tensor]) -> bool:
    keys = list(state_dict.keys())
    if not keys:
        return False
    return _has_prefix_key(keys, "transformer.transformer_blocks.") and (
        _has_substring_key(keys, ".lora.down.weight") or _has_substring_key(keys, ".lora.up.weight")
    )


def detect_lora_format(state_dict: Mapping[str, torch.Tensor]) -> LoRAFormat:
    """Detect LoRA format from state dict keys."""
    keys = list(state_dict.keys())
    if not keys:
        return LoRAFormat.STANDARD

    if _has_substring_key(keys, ".lora_A") or _has_substring_key(keys, ".lora_B"):
        return LoRAFormat.STANDARD

    if any(_looks_like_xlabs_flux_key(k) for k in keys):
        return LoRAFormat.XLABS_FLUX
    if _looks_like_kohya_flux(state_dict):
        return LoRAFormat.KOHYA_FLUX
    if _looks_like_wan_lora(state_dict):
        return LoRAFormat.WAN
    if _looks_like_qwen_image(state_dict):
        return LoRAFormat.QWEN_IMAGE
    if _looks_like_non_diffusers_sd(state_dict):
        return LoRAFormat.NON_DIFFUSERS_SD
    if _has_substring_key(keys, ".lora.down") or _has_substring_key(keys, ".lora_up"):
        return LoRAFormat.NON_DIFFUSERS_SD

    return LoRAFormat.STANDARD


def _convert_down_up_to_ab(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Generic down/up -> A/B conversion."""
    out = {}
    for name, tensor in state_dict.items():
        new_name = name
        if "lora_down.weight" in new_name:
            new_name = new_name.replace("lora_down.weight", "lora_A.weight")
        elif "lora_up.weight" in new_name:
            new_name = new_name.replace("lora_up.weight", "lora_B.weight")
        elif new_name.endswith(".lora_down"):
            new_name = new_name.replace(".lora_down", ".lora_A")
        elif new_name.endswith(".lora_up"):
            new_name = new_name.replace(".lora_up", ".lora_B")
        out[new_name] = tensor
    return out


def _try_convert_with_diffusers_utils(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor] | None:
    """Try converting using diffusers lora_conversion_utils if available."""
    try:
        from diffusers.loaders import lora_conversion_utils as lcu

        if hasattr(lcu, "maybe_convert_state_dict"):
            converted = lcu.maybe_convert_state_dict(state_dict)
            if not isinstance(converted, dict):
                converted = dict(converted)
            return converted
    except ImportError:
        pass
    except Exception as e:
        logger.warning("diffusers lora_conversion_utils failed: %s", e)
    return None


def convert_lora_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    fmt: LoRAFormat,
) -> dict[str, torch.Tensor]:
    """Convert LoRA state dict to standard format."""
    state_dict = dict(state_dict)

    if fmt == LoRAFormat.QWEN_IMAGE:
        out = {}
        for name, tensor in state_dict.items():
            new_name = name
            if new_name.startswith("transformer."):
                new_name = new_name[len("transformer.") :]
            if new_name.endswith(".lora.down.weight"):
                new_name = new_name.replace(".lora.down.weight", ".lora_A.weight")
            elif new_name.endswith(".lora.up.weight"):
                new_name = new_name.replace(".lora.up.weight", ".lora_B.weight")
            out[new_name] = tensor
        return out

    if fmt in (LoRAFormat.XLABS_FLUX, LoRAFormat.KOHYA_FLUX, LoRAFormat.WAN):
        converted = _try_convert_with_diffusers_utils(state_dict)
        if converted is not None:
            state_dict = converted
        return _convert_down_up_to_ab(state_dict)

    if fmt == LoRAFormat.NON_DIFFUSERS_SD:
        converted = _try_convert_with_diffusers_utils(state_dict)
        if converted is not None:
            state_dict = converted
        return _convert_down_up_to_ab(state_dict)

    if fmt == LoRAFormat.STANDARD:
        converted = _try_convert_with_diffusers_utils(state_dict)
        if converted is not None:
            return converted
        return state_dict

    return dict(state_dict)


def normalize_lora_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Normalize any supported LoRA format to standard format."""
    fmt = detect_lora_format(state_dict)
    return convert_lora_state_dict(state_dict, fmt)


def create_diffusers_weights_mapper():
    """Create a weights mapper function for Diffusers LoRA keys."""

    def mapper(key: str) -> str:
        new_key = key

        if "lora_unet_" in key:
            new_key = key.replace("lora_unet_", "transformer.")
        elif "lora_te_" in key:
            new_key = key.replace("lora_te_", "text_encoder.")
        elif "lora_te2_" in key:
            new_key = key.replace("lora_te2_", "text_encoder_2.")
        elif "diffusion_model." in key:
            new_key = key.replace("diffusion_model.", "transformer.")

        return new_key

    return mapper


def is_diffusers_format(lora_path: str) -> bool:
    adapter_config_path = os.path.join(lora_path, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        return False

    config_path = os.path.join(lora_path, "configuration.json")
    if os.path.exists(config_path):
        return True

    weight_safe = os.path.join(lora_path, LORA_WEIGHT_NAME_SAFE)
    weight_bin = os.path.join(lora_path, LORA_WEIGHT_NAME)
    if os.path.exists(weight_safe) or os.path.exists(weight_bin):
        return True

    for f in os.listdir(lora_path):
        if f.endswith(".safetensors") and "adapter" not in f.lower():
            return True
        if f.endswith(".bin") and "pytorch_lora" in f:
            return True

    return False


def load_diffusers_weights(lora_path: str) -> dict[str, torch.Tensor]:
    weight_safe = os.path.join(lora_path, LORA_WEIGHT_NAME_SAFE)
    weight_bin = os.path.join(lora_path, LORA_WEIGHT_NAME)

    if os.path.exists(weight_safe):
        state_dict = load_file(weight_safe)
    elif os.path.exists(weight_bin):
        state_dict = torch.load(weight_bin, map_location="cpu")
    else:
        for f in os.listdir(lora_path):
            if f.endswith(".safetensors"):
                state_dict = load_file(os.path.join(lora_path, f))
                break
            if f.endswith(".bin") and "lora" in f.lower():
                state_dict = torch.load(os.path.join(lora_path, f), map_location="cpu")
                break
        else:
            raise FileNotFoundError(
                f"Cannot find LoRA weights in {lora_path}. Expected {LORA_WEIGHT_NAME_SAFE} or {LORA_WEIGHT_NAME}"
            )

    if not state_dict:
        raise ValueError(f"Empty LoRA weights loaded from {lora_path}")

    return normalize_lora_state_dict(state_dict)


def save_lora_to_temp(
    state_dict: dict[str, torch.Tensor],
    rank: int,
    lora_alpha: int,
) -> tuple[str, callable]:
    """Save normalized LoRA weights to a temp directory with standard PEFT format.

    Returns:
        tuple: (temp_directory_path, cleanup_function)
        The cleanup_function can be called to remove the temp directory.
        Using the cleanup_function is preferred over relying on context manager
        since we need to pass the path to LoRAModel.from_local_checkpoint.
    """
    import json
    import tempfile

    from safetensors.torch import save_file

    temp_dir_obj = tempfile.TemporaryDirectory(prefix="lora_")
    temp_dir = temp_dir_obj.name

    def cleanup():
        temp_dir_obj.cleanup()

    try:
        weights_path = os.path.join(temp_dir, "adapter_model.safetensors")

        mapper = create_diffusers_weights_mapper()
        filtered_state_dict = {}
        for k, v in state_dict.items():
            if v is not None and v.numel() > 0:
                new_key = mapper(k)
                filtered_state_dict[new_key] = v

        if not filtered_state_dict:
            raise ValueError("No valid tensors found in state_dict")

        save_file(filtered_state_dict, weights_path)

        config = {
            "r": rank,
            "lora_alpha": lora_alpha,
            "target_modules": None,
            "lora_dropout": 0.0,
            "bias": "none",
            "modules_to_save": None,
            "use_rslora": False,
            "use_dora": False,
            "inference_mode": True,
        }
        config_path = os.path.join(temp_dir, "adapter_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        return temp_dir, cleanup

    except Exception:
        cleanup()
        raise


def infer_lora_rank_from_weights(state_dict: dict[str, torch.Tensor]) -> int:
    for key, value in state_dict.items():
        if ".lora_A." in key or ".lora_B." in key:
            if "bias" in key:
                continue
            shape = value.shape
            if len(shape) == 2:
                return min(shape)
    logger.warning("Could not infer LoRA rank from weights, using default rank 16")
    return 16
