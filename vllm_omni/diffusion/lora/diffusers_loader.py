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
    DIFFUSERS_FLUX = "diffusers-flux"
    AI_TOOLKIT_FLUX2 = "ai-toolkit-flux2"


def _has_substring_key(keys: Mapping[str, Any], substr: str) -> bool:
    if isinstance(keys, dict):
        keys = keys.keys()
    return any(substr in k for k in keys)


def _has_prefix_key(keys: Mapping[str, Any], prefix: str) -> bool:
    if isinstance(keys, dict):
        keys = keys.keys()
    return any(k.startswith(prefix) for k in keys)


def _remove_prefix(name: str, prefix: str) -> str:
    """Remove prefix from name if present."""
    if name.startswith(prefix):
        return name[len(prefix) :]
    return name


PEFT_PREFIX = "base_model.model."


def _get_peft_prefix_variants(base_prefix: str) -> list[str]:
    """Return both plain and PEFT-wrapped prefix variants."""
    return [base_prefix, f"{PEFT_PREFIX}{base_prefix}"]


def _looks_like_xlabs_flux_key(k: str) -> bool:
    valid_endings = (
        ".down.weight",
        ".up.weight",
        ".lora_A.weight",
        ".lora_B.weight",
    )
    if not any(k.endswith(ending) for ending in valid_endings):
        return False
    valid_prefixes = (
        "double_blocks.",
        "single_blocks.",
        "diffusion_model.double_blocks",
        "diffusion_model.single_blocks",
        "base_model.model.double_blocks.",
        "base_model.model.single_blocks.",
    )
    if not k.startswith(valid_prefixes):
        return False
    return ".processor." in k or ".proj_lora" in k or ".qkv_lora" in k or ".img_attn." in k or ".txt_attn." in k


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
    # Check for qwen-image LoRA format:
    # - Keys may have "transformer.transformer_blocks." or just "transformer_blocks." prefix
    # - With either .lora.down.weight/.lora.up.weight (kohya-style) or
    #   .lora_A.weight/.lora_B.weight (PEFT-style)
    # - And containing diffusers-style FFN names: ff.linear_in, ff.linear_out, etc.
    has_qwen_prefix = any(
        k.startswith("transformer.transformer_blocks.") or k.startswith("transformer_blocks.") for k in keys
    )
    if not has_qwen_prefix:
        return False
    has_lora_keys = any(
        ".lora.down.weight" in k or ".lora.up.weight" in k or ".lora_A.weight" in k or ".lora_B.weight" in k
        for k in keys
    )
    if not has_lora_keys:
        return False
    # Check for diffusers-style FFN naming (ff.linear_in, ff_context.linear_out, etc.)
    return any(
        ".ff.linear_in" in k or ".ff.linear_out" in k or ".ff_context.linear_in" in k or ".ff_context.linear_out" in k
        for k in keys
    )


def detect_lora_format(state_dict: Mapping[str, torch.Tensor]) -> LoRAFormat:
    """Detect LoRA format from state dict keys."""
    keys = list(state_dict.keys())
    if not keys:
        return LoRAFormat.STANDARD

    # Check AI-Toolkit Flux2 format (diffusion_model.transformer_blocks.X.*)
    # This is the format produced by ai-toolkit training
    if any(k.startswith("diffusion_model.") for k in keys):
        if any("transformer_blocks" in k or "single_transformer_blocks" in k for k in keys):
            return LoRAFormat.AI_TOOLKIT_FLUX2
        if any("double_blocks" in k or "single_blocks" in k for k in keys):
            return LoRAFormat.DIFFUSERS_FLUX

    # Check Diffusers Flux format (base_model.model.double_blocks.X...)
    if _has_substring_key(keys, ".lora_A") or _has_substring_key(keys, ".lora_B"):
        if any(k.startswith("base_model.model.") for k in keys):
            if any("double_blocks" in k or "single_blocks" in k for k in keys):
                return LoRAFormat.DIFFUSERS_FLUX
        # Check for keys starting with double_blocks.X or single_blocks.X
        if any(k.startswith("double_blocks.") or k.startswith("single_blocks.") for k in keys):
            return LoRAFormat.DIFFUSERS_FLUX

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

    if _has_substring_key(keys, ".lora_A") or _has_substring_key(keys, ".lora_B"):
        return LoRAFormat.STANDARD

    return LoRAFormat.STANDARD


def _convert_diffusers_flux_lora(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert Diffusers-trained Flux LoRA to vLLM Flux2-Klein format.

    Reference: diffusers.loaders.lora_conversion_utils._convert_non_diffusers_flux2_lora_to_diffusers

    Handles multiple input formats:
    1. Flux1/PEFT style keys (img_attn.proj, img_attn.qkv, etc.)
    2. FAL style keys (img_attn.to_out, img_attn.to_qkv)
    3. base_model.model.* prefix (PEFT format)

    Input/Output mappings:
      - double_blocks.X.img_attn.proj → transformer_blocks.X.attn.to_out.0
      - double_blocks.X.img_attn.qkv → transformer_blocks.X.attn.to_qkv
      - double_blocks.X.txt_attn.proj → transformer_blocks.X.attn.to_add_out
      - single_blocks.X.linear1 → single_transformer_blocks.X.attn.to_qkv_mlp_proj
      - single_blocks.X.linear2 → single_transformer_blocks.X.attn.to_out
      - img_in → x_embedder
      - txt_in → context_embedder
      - time_in.in_layer → time_guidance_embed.timestep_embedder.linear_1
      - time_in.out_layer → time_guidance_embed.timestep_embedder.linear_2
      - final_layer.linear → proj_out
      - single_stream_modulation.lin → single_stream_modulation.linear
      - double_stream_modulation_img.lin → double_stream_modulation_img.linear
      - double_stream_modulation_txt.lin → double_stream_modulation_txt.linear
    """
    out = {}
    processed_keys: set[str] = set()

    # Count layers for iteration
    num_double_layers = 0
    num_single_layers = 0
    for key in state_dict.keys():
        name = _remove_prefix(key, PEFT_PREFIX)
        if name.startswith("double_blocks."):
            try:
                layer_num = int(name.split(".")[1])
                num_double_layers = max(num_double_layers, layer_num + 1)
            except (ValueError, IndexError):
                pass
        elif name.startswith("single_blocks."):
            try:
                layer_num = int(name.split(".")[1])
                num_single_layers = max(num_single_layers, layer_num + 1)
            except (ValueError, IndexError):
                pass

    lora_keys = ("lora_A", "lora_B")

    # === Process single blocks ===
    for sl in range(num_single_layers):
        single_block_prefix = f"single_blocks.{sl}"
        attn_prefix = f"single_transformer_blocks.{sl}.attn"

        for lora_key in lora_keys:
            # linear1 -> to_qkv_mlp_proj
            for prefix in _get_peft_prefix_variants(single_block_prefix):
                linear1_key = f"{prefix}.linear1.{lora_key}.weight"
                if linear1_key in state_dict:
                    out[f"{attn_prefix}.to_qkv_mlp_proj.{lora_key}.weight"] = state_dict[linear1_key]
                    processed_keys.add(linear1_key)

                # linear2 -> to_out
                linear2_key = f"{prefix}.linear2.{lora_key}.weight"
                if linear2_key in state_dict:
                    out[f"{attn_prefix}.to_out.{lora_key}.weight"] = state_dict[linear2_key]
                    processed_keys.add(linear2_key)

    # === Process double blocks ===
    for dl in range(num_double_layers):
        transformer_block_prefix = f"transformer_blocks.{dl}"

        for lora_key in lora_keys:
            # Handle img_attn and txt_attn
            for prefix in _get_peft_prefix_variants(f"double_blocks.{dl}"):
                # img_attn.to_out -> attn.to_out.0
                to_out_key = f"{prefix}.img_attn.to_out.{lora_key}.weight"
                if to_out_key in state_dict:
                    out[f"{transformer_block_prefix}.attn.to_out.0.{lora_key}.weight"] = state_dict[to_out_key]
                    processed_keys.add(to_out_key)

                # img_attn.proj -> attn.to_out.0 (Flux1 style)
                proj_key = f"{prefix}.img_attn.proj.{lora_key}.weight"
                if proj_key in state_dict:
                    out[f"{transformer_block_prefix}.attn.to_out.0.{lora_key}.weight"] = state_dict[proj_key]
                    processed_keys.add(proj_key)

                # img_attn.to_qkv -> attn.to_qkv
                to_qkv_key = f"{prefix}.img_attn.to_qkv.{lora_key}.weight"
                if to_qkv_key in state_dict:
                    out[f"{transformer_block_prefix}.attn.to_qkv.{lora_key}.weight"] = state_dict[to_qkv_key]
                    processed_keys.add(to_qkv_key)

                # img_attn.qkv -> attn.to_qkv (Flux1 style)
                qkv_key = f"{prefix}.img_attn.qkv.{lora_key}.weight"
                if qkv_key in state_dict:
                    out[f"{transformer_block_prefix}.attn.to_qkv.{lora_key}.weight"] = state_dict[qkv_key]
                    processed_keys.add(qkv_key)

                # txt_attn.to_out -> attn.to_add_out
                txt_to_out_key = f"{prefix}.txt_attn.to_out.{lora_key}.weight"
                if txt_to_out_key in state_dict:
                    out[f"{transformer_block_prefix}.attn.to_add_out.{lora_key}.weight"] = state_dict[txt_to_out_key]
                    processed_keys.add(txt_to_out_key)

                # txt_attn.proj -> attn.to_add_out (Flux1 style)
                txt_proj_key = f"{prefix}.txt_attn.proj.{lora_key}.weight"
                if txt_proj_key in state_dict:
                    out[f"{transformer_block_prefix}.attn.to_add_out.{lora_key}.weight"] = state_dict[txt_proj_key]
                    processed_keys.add(txt_proj_key)

            # Handle MLP layers (for completeness, though Flux2-Klein doesn't have separate MLP LoRA)
            mlp_mappings = [
                ("img_mlp.0", "ff.linear_in"),
                ("img_mlp.2", "ff.linear_out"),
                ("txt_mlp.0", "ff_context.linear_in"),
                ("txt_mlp.2", "ff_context.linear_out"),
            ]
            for org_mlp, diff_mlp in mlp_mappings:
                for prefix in _get_peft_prefix_variants(f"double_blocks.{dl}"):
                    original_key = f"{prefix}.{org_mlp}.{lora_key}.weight"
                    if original_key in state_dict:
                        out[f"{transformer_block_prefix}.{diff_mlp}.{lora_key}.weight"] = state_dict[original_key]
                        processed_keys.add(original_key)

    # === Handle extra mappings (top-level modules) ===
    # Reference: diffusers extra_mappings in _convert_non_diffusers_flux2_lora_to_diffusers
    extra_mappings = {
        "img_in": "x_embedder",
        "txt_in": "context_embedder",
        "time_in.in_layer": "time_guidance_embed.timestep_embedder.linear_1",
        "time_in.out_layer": "time_guidance_embed.timestep_embedder.linear_2",
        "final_layer.linear": "proj_out",
        "single_stream_modulation.lin": "single_stream_modulation.linear",
        "double_stream_modulation_img.lin": "double_stream_modulation_img.linear",
        "double_stream_modulation_txt.lin": "double_stream_modulation_txt.linear",
    }

    for org_key, target_key in extra_mappings.items():
        for lora_key in lora_keys:
            for prefix in ("", PEFT_PREFIX):
                original_key = f"{prefix}{org_key}.{lora_key}.weight"
                if original_key in state_dict:
                    out[f"{target_key}.{lora_key}.weight"] = state_dict[original_key]
                    processed_keys.add(original_key)

    # Warn about unprocessed keys
    unprocessed = set(state_dict.keys()) - processed_keys
    if unprocessed:
        logger.warning(
            "Flux LoRA conversion: %d unprocessed keys (sample): %s",
            len(unprocessed),
            list(unprocessed)[:5],
        )

    return out


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


DIFFUSION_MODEL_PREFIX = "diffusion_model."


def _convert_ai_toolkit_flux2_lora(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Fallback conversion for AI-Toolkit Flux2 LoRA format.

    This handles the case when diffusers' _convert_non_diffusers_flux2_lora_to_diffusers
    is not available or fails. It performs prefix removal and layer name mapping
    for diffusers-trained Flux2 LoRA checkpoints.

    Mapping from diffusers Flux2 to vLLM Flux2-Klein:
    - img_mlp.net.0.proj -> ff.linear_in
    - img_mlp.net.2 -> ff.linear_out
    - txt_mlp.net.0.proj -> ff_context.linear_in
    - txt_mlp.net.2 -> ff_context.linear_out
    """
    out = {}

    for name, tensor in state_dict.items():
        new_name = _remove_prefix(name, DIFFUSION_MODEL_PREFIX)

        # Map MLP layer names
        # img_mlp.net.0.proj -> ff.linear_in (first layer in FeedForward)
        if ".img_mlp.net.0.proj." in new_name:
            new_name = new_name.replace(".img_mlp.net.0.proj.", ".ff.linear_in.")
        # img_mlp.net.2 -> ff.linear_out (last layer in FeedForward)
        elif ".img_mlp.net.2." in new_name:
            new_name = new_name.replace(".img_mlp.net.2.", ".ff.linear_out.")
        # txt_mlp.net.0.proj -> ff_context.linear_in
        elif ".txt_mlp.net.0.proj." in new_name:
            new_name = new_name.replace(".txt_mlp.net.0.proj.", ".ff_context.linear_in.")
        # txt_mlp.net.2 -> ff_context.linear_out
        elif ".txt_mlp.net.2." in new_name:
            new_name = new_name.replace(".txt_mlp.net.2.", ".ff_context.linear_out.")

        out[new_name] = tensor

    # Convert lora_down/lora_up to lora_A/lora_B using shared function
    out = _convert_down_up_to_ab(out)

    return out


def _try_convert_with_diffusers_utils(
    state_dict: dict[str, torch.Tensor], fmt: LoRAFormat | None = None
) -> dict[str, torch.Tensor] | None:
    """Try converting using diffusers lora_conversion_utils if available.

    Args:
        state_dict: The LoRA state dict to convert.
        fmt: The detected LoRA format. If None, will try all conversions.
    """
    try:
        from diffusers.loaders import lora_conversion_utils as lcu

        # Try the public API first (if it exists in future versions)
        if hasattr(lcu, "maybe_convert_state_dict"):
            converted = lcu.maybe_convert_state_dict(state_dict)
            if not isinstance(converted, dict):
                converted = dict(converted)
            return converted

        # Use internal conversion functions based on format
        # These are private functions but necessary for correct conversion
        state_dict_keys = list(state_dict.keys())

        # Kohya Flux format: keys contain ".lora_down.weight"
        if fmt in (LoRAFormat.KOHYA_FLUX, None):
            if any(".lora_down.weight" in k for k in state_dict_keys):
                if hasattr(lcu, "_convert_kohya_flux_lora_to_diffusers"):
                    return lcu._convert_kohya_flux_lora_to_diffusers(state_dict)

        # XLabs Flux format: keys contain "processor"
        if fmt in (LoRAFormat.XLABS_FLUX, None):
            if any("processor" in k for k in state_dict_keys):
                if hasattr(lcu, "_convert_xlabs_flux_lora_to_diffusers"):
                    return lcu._convert_xlabs_flux_lora_to_diffusers(state_dict)

        # WAN format
        if fmt in (LoRAFormat.WAN, None):
            if any(k.startswith("diffusion_model.blocks.") for k in state_dict_keys):
                if hasattr(lcu, "_convert_non_diffusers_wan_lora_to_diffusers"):
                    return lcu._convert_non_diffusers_wan_lora_to_diffusers(state_dict)

        # HunyuanVideo format
        if fmt is None:
            if hasattr(lcu, "_convert_hunyuan_video_lora_to_diffusers"):
                # Check if it looks like HunyuanVideo
                if any("double_blocks" in k or "single_blocks" in k for k in state_dict_keys):
                    pass  # Let Flux converters handle this

        # AI-Toolkit Flux2 format: keys start with "diffusion_model.transformer_blocks"
        if fmt in (LoRAFormat.AI_TOOLKIT_FLUX2, None):
            if any(k.startswith("diffusion_model.") for k in state_dict_keys):
                if hasattr(lcu, "_convert_non_diffusers_flux2_lora_to_diffusers"):
                    return lcu._convert_non_diffusers_flux2_lora_to_diffusers(state_dict)

        # Non-diffusers SD format
        if fmt in (LoRAFormat.NON_DIFFUSERS_SD, None):
            if any(k.startswith(("lora_unet_", "lora_te_", "lora_te1_", "lora_te2_")) for k in state_dict_keys):
                if hasattr(lcu, "_convert_non_diffusers_lora_to_diffusers"):
                    converted, _ = lcu._convert_non_diffusers_lora_to_diffusers(state_dict)
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
            new_name = _remove_prefix(name, "transformer.")
            new_name = new_name.replace(".ff.linear_in.", ".img_mlp.net.0.")
            new_name = new_name.replace(".ff.linear_out.", ".img_mlp.net.2.")
            new_name = new_name.replace(".ff_context.linear_in.", ".txt_mlp.net.0.")
            new_name = new_name.replace(".ff_context.linear_out.", ".txt_mlp.net.2.")
            out[new_name] = tensor
        return _convert_down_up_to_ab(out)

    if fmt in (LoRAFormat.XLABS_FLUX, LoRAFormat.KOHYA_FLUX, LoRAFormat.WAN):
        converted = _try_convert_with_diffusers_utils(state_dict, fmt=fmt)
        if converted is not None:
            # diffusers conversion already handles module name mapping
            # just need to convert down/up to A/B if not already done
            state_dict = converted
            # Check if already converted (has lora_A/lora_B keys)
            if not any(".lora_A." in k or ".lora_B." in k for k in state_dict.keys()):
                state_dict = _convert_down_up_to_ab(state_dict)
            return state_dict
        # Fallback: just do down/up to A/B conversion
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

    if fmt == LoRAFormat.DIFFUSERS_FLUX:
        return _convert_diffusers_flux_lora(state_dict)

    if fmt == LoRAFormat.AI_TOOLKIT_FLUX2:
        # Use diffusers' built-in conversion for ai-toolkit format
        converted = _try_convert_with_diffusers_utils(state_dict, fmt=fmt)
        if converted is not None:
            return converted
        # Fallback: just remove diffusion_model. prefix
        return _convert_ai_toolkit_flux2_lora(state_dict)

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
        state_dict = torch.load(weight_bin, map_location="cpu", weights_only=True)
    else:
        candidates = sorted(f for f in os.listdir(lora_path) if f.endswith(".safetensors"))
        if len(candidates) > 1:
            logger.warning(
                "Multiple .safetensors files found in %s: %s. Loading first: %s",
                lora_path,
                candidates,
                candidates[0],
            )
        if candidates:
            state_dict = load_file(os.path.join(lora_path, candidates[0]))
        else:
            bin_candidates = sorted(f for f in os.listdir(lora_path) if f.endswith(".bin") and "lora" in f.lower())
            if len(bin_candidates) > 1:
                logger.warning(
                    "Multiple .bin files found in %s: %s. Loading first: %s",
                    lora_path,
                    bin_candidates,
                    bin_candidates[0],
                )
            if bin_candidates:
                state_dict = torch.load(
                    os.path.join(lora_path, bin_candidates[0]), map_location="cpu", weights_only=True
                )
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

        filtered_state_dict = {}
        for k, v in state_dict.items():
            if v is not None and v.numel() > 0:
                new_key = k
                for prefix in ("transformer.", "text_encoder.", "text_encoder_2."):
                    new_key = _remove_prefix(new_key, prefix)
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
