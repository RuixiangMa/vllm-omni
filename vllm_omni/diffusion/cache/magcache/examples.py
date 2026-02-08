# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
MagCache Integration Template for New Models.

This module provides a complete template for integrating new diffusion models
into MagCache. Copy this file and modify according to your model's architecture.

Integration Steps:
    1. Analyze model architecture (block structure, I/O format)
    2. Create Strategy class (inherit from MagCacheStrategy)
    3. Implement required methods
    4. Register the strategy
    5. Test and calibrate mag_ratios
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

import torch
from diffusers.hooks._helpers import TransformerBlockRegistry, TransformerBlockMetadata

from vllm_omni.diffusion.cache.magcache.strategy import (
    MagCacheStrategy,
    MagCacheStrategyRegistry,
)


def register_transformer_block(
    model_class,
    return_hidden_states_index: int = 1,
    return_encoder_hidden_states_index: int = 0,
) -> None:
    """Register a transformer block class with the TransformerBlockRegistry."""
    try:
        TransformerBlockRegistry.get(model_class)
    except ValueError:
        TransformerBlockRegistry.register(
            model_class=model_class,
            metadata=TransformerBlockMetadata(
                return_hidden_states_index=return_hidden_states_index,
                return_encoder_hidden_states_index=return_encoder_hidden_states_index,
            ),
        )


# =============================================================================
# EXAMPLE: SD3 (Stable Diffusion 3) Integration
# =============================================================================

class SD3MagCacheStrategy(MagCacheStrategy):
    """
    MagCache strategy for SD3 (Stable Diffusion 3).

    SD3 Architecture Analysis:
    - Single stream: hidden_states only (no encoder_hidden_states separation)
    - Block structure: transformer_blocks (nn.ModuleList)
    - Output: tuple of (hidden_states,)
    - Residual: output - input (simple subtraction)

    Integration Steps:
    1. Identify block name: "transformer_blocks"
    2. Determine I/O indices: return_hidden_states_index=0
    3. Define residual: output - input
    """

    @property
    def transformer_type(self) -> str:
        return "SD3Transformer2DModel"

    @property
    def mag_ratios(self) -> torch.Tensor:
        """Return default mag_ratios for SD3 model."""
        return self.SD3_MAG_RATIOS

    SD3_MAG_RATIOS = torch.tensor(
        [
            1.0,
            1.15,
            1.10,
            1.05,
            1.02,
            1.00,
            0.98,
            0.95,
            0.92,
            0.90,
            0.88,
            0.85,
            0.82,
            0.80,
            0.78,
            0.75,
            0.72,
            0.70,
            0.68,
            0.65,
            0.62,
            0.60,
            0.58,
            0.55,
            0.52,
            0.50,
            0.48,
            0.45,
        ]
    )

    @staticmethod
    def nearest_interp(src_array: torch.Tensor, target_length: int) -> torch.Tensor:
        """Interpolate mag_ratios to target length using nearest neighbor."""
        src_length = len(src_array)
        if target_length == 1:
            return src_array[-1:]

        scale = (src_length - 1) / (target_length - 1)
        grid = torch.arange(target_length, device=src_array.device, dtype=torch.float32)
        mapped_indices = torch.round(grid * scale).long()

        return src_array[mapped_indices]

    @staticmethod
    def register_blocks() -> None:
        """Register SD3 transformer blocks.

        SD3 uses a simple transformer block structure.
        """
        try:
            from diffusers.models.transformers.sd_transformer_2d import (
                SD3TransformerBlock,
            )

            register_transformer_block(
                SD3TransformerBlock,
                return_hidden_states_index=0,
            )
        except ImportError:
            pass

    def create_context(
        self,
        module: torch.nn.Module,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        timestep: torch.Tensor,
        guidance: torch.Tensor | None,
        **kwargs,
    ) -> MagCacheContext:
        """Create context for SD3 model."""
        temb = module.time_embed(timestep)

        def run_transformer_blocks():
            for block in module.transformer_blocks:
                hidden_states = block(hidden_states, temb=temb)
            return hidden_states

        def run_single_transformer_blocks(h):
            return h

        def postprocess(h: torch.Tensor, e: torch.Tensor) -> Any:
            return h

        return MagCacheContext(
            hidden_states=hidden_states,
            encoder_hidden_states=None,
            temb=temb,
            head_block_input=None,
            run_transformer_blocks=run_transformer_blocks,
            run_single_transformer_blocks=run_single_transformer_blocks,
            postprocess=postprocess,
        )

    def get_head_block_input(self, context: MagCacheContext) -> torch.Tensor:
        """Get input to the first transformer block."""
        return context.hidden_states

    def compute_residual(
        self,
        output: torch.Tensor,
        head_input: torch.Tensor,
        context: MagCacheContext | None,
    ) -> torch.Tensor:
        """Compute residual for SD3: output - input."""
        return output - head_input

    def apply_residual(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """Apply residual for SD3: hidden_states + residual."""
        return hidden_states + residual


# Register SD3 strategy
MagCacheStrategyRegistry.register(SD3MagCacheStrategy())


# =============================================================================
# TEMPLATE: Copy and modify for your model
# =============================================================================

# class YourModelMagCacheStrategy(MagCacheStrategy):
#     """
#     MagCache strategy for YourModel.
#
#     Model Architecture Analysis:
#     - [Describe the architecture]
#     - Block name: [e.g., "transformer_blocks", "layers", "blocks"]
#     - I/O format: [Describe input/output format]
#     - Residual: [Describe how to compute residual]
#     """
#
#     @property
#     def transformer_type(self) -> str:
#         return "YourTransformerModel"
#
#     YOUR_MAG_RATIOS = torch.tensor([...])  # Replace with your model's ratios
#
#     @staticmethod
#     def nearest_interp(src_array: torch.Tensor, target_length: int) -> torch.Tensor:
#         """Interpolate mag_ratios to target length."""
#         src_length = len(src_array)
#         if target_length == 1:
#             return src_array[-1:]
#
#         scale = (src_length - 1) / (target_length - 1)
#         grid = torch.arange(target_length, device=src_array.device, dtype=torch.float32)
#         mapped_indices = torch.round(grid * scale).long()
#
#         return src_array[mapped_indices]
#
#     @staticmethod
#     def register_blocks() -> None:
#         """Register your model's transformer blocks."""
#         try:
#             from your_model import YourTransformerBlock
#
#             register_transformer_block(
#                 YourTransformerBlock,
#                 return_hidden_states_index=0,  # Adjust based on your model's output
#             )
#         except ImportError:
#             pass
#
#     def compute_residual(
#         self,
#         output: torch.Tensor,
#         head_input: torch.Tensor,
#         context: MagCacheContext | None,
#     ) -> torch.Tensor:
#         """Compute residual for your model.
#
#         Common patterns:
#         - Simple: output - head_input
#         - Complex: output - head_input (with shape adjustments)
#         """
#         return output - head_input
#
#     def apply_residual(
#         self,
#         hidden_states: torch.Tensor,
#         residual: torch.Tensor,
#     ) -> torch.Tensor:
#         """Apply residual for your model.
#
#         Common patterns:
#         - Simple: hidden_states + residual
#         - Complex: hidden_states + residual (with shape adjustments)
#         """
#         return hidden_states + residual
#
#
# # Register your strategy
# MagCacheStrategyRegistry.register(YourModelMagCacheStrategy())
