# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
MagCache strategy definitions for different model architectures.

This module provides model-specific strategies for MagCache, allowing easy
extension to new models by implementing the MagCacheStrategy interface.

Architecture:
- MagCacheStrategy: Abstract base class defining the strategy interface
- FluxMagCacheStrategy: Strategy for Flux (dual-stream) models
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

import torch
from diffusers.hooks._helpers import TransformerBlockRegistry, TransformerBlockMetadata


def register_transformer_block(
    model_class,
    return_hidden_states_index: int = 1,
    return_encoder_hidden_states_index: int = 0,
) -> None:
    """Register a transformer block class with the TransformerBlockRegistry.

    Args:
        model_class: The transformer block class to register.
        return_hidden_states_index: Index of hidden_states in the forward output tuple.
        return_encoder_hidden_states_index: Index of encoder_hidden_states in the output.
    """
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


@dataclass
class MagCacheContext:
    """
    Context object containing model-specific information for MagCache.

    Attributes:
        hidden_states: Current hidden states before transformer blocks.
        encoder_hidden_states: Optional encoder states (None for single-stream).
        temb: Timestep embedding tensor.
        head_block_input: Input to the first transformer block (for residual calculation).
        run_transformer_blocks: Callable to run transformer blocks.
        run_single_transformer_blocks: Callable to run single transformer blocks.
        postprocess: Callable to produce final output from block outputs.
    """

    hidden_states: torch.Tensor
    encoder_hidden_states: torch.Tensor | None
    temb: torch.Tensor
    head_block_input: torch.Tensor | None
    run_transformer_blocks: Callable[[], tuple[torch.Tensor, torch.Tensor]]
    run_single_transformer_blocks: Callable[[], torch.Tensor]
    postprocess: Callable[[torch.Tensor, torch.Tensor], Any]


class MagCacheStrategy(ABC):
    """
    Abstract base class for MagCache strategies.

    Each model architecture requires a specific strategy to handle:
    - Preprocessing of inputs (embeddings, positional encodings)
    - Running transformer blocks
    - Postprocessing (normalization, projection)
    - Computing residuals for caching

    Implement this class to add support for new model architectures.
    """

    @property
    @abstractmethod
    def transformer_type(self) -> str:
        """Returns the transformer class name this strategy supports."""
        pass

    @property
    @abstractmethod
    def mag_ratios(self) -> torch.Tensor:
        """Return the default mag_ratios tensor for this model.

        This tensor defines caching ratios for each transformer block.
        Values should be calibrated for the specific model architecture.

        Returns:
            1D tensor of mag_ratios (one per transformer block).
        """
        pass

    @abstractmethod
    def create_context(
        self,
        module: torch.nn.Module,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        timestep: torch.Tensor,
        guidance: torch.Tensor | None,
        **kwargs,
    ) -> MagCacheContext:
        """
        Create a MagCacheContext from model inputs.

        Args:
            module: The transformer module.
            hidden_states: Input latents.
            encoder_hidden_states: Text encoder outputs (None for single-stream).
            timestep: Denoising timestep.
            guidance: Guidance scale tensor (optional).
            **kwargs: Additional model-specific arguments.

        Returns:
            MagCacheContext with all information needed for caching.
        """
        pass

    @abstractmethod
    def get_head_block_input(self, context: MagCacheContext) -> torch.Tensor:
        """
        Get the input to the first transformer block.

        Args:
            context: MagCacheContext from create_context.

        Returns:
            Tensor representing the input to the first block.
        """
        pass

    @abstractmethod
    def compute_residual(
        self,
        output: torch.Tensor,
        head_input: torch.Tensor,
        context: MagCacheContext,
    ) -> torch.Tensor:
        """
        Compute residual between output and head input.

        Args:
            output: Output from transformer blocks.
            head_input: Input to the first block.
            context: MagCacheContext.

        Returns:
            Residual tensor for caching.
        """
        pass

    @abstractmethod
    def apply_residual(self, hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """
        Apply cached residual to hidden states.

        Args:
            hidden_states: Current hidden states.
            residual: Cached residual to apply.

        Returns:
            Hidden states with residual added.
        """
        pass

    def apply_residual_tuple(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        residual: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply cached residual tuple to both hidden_states and encoder_hidden_states.

        Default implementation: add residuals separately.
        Override this method for models with specific residual application logic.

        Args:
            hidden_states: Current hidden states.
            encoder_hidden_states: Current encoder hidden states.
            residual: Tuple of (hidden_states_residual, encoder_hidden_states_residual).

        Returns:
            Tuple of (hidden_states, encoder_hidden_states) with residuals applied.
        """
        h_res, e_res = residual
        return hidden_states + h_res, encoder_hidden_states + e_res


class FluxMagCacheStrategy(MagCacheStrategy):
    """
    MagCache strategy for Flux (dual-stream) models.

    Flux architecture:
    - transformer blocks (dual-stream): image tokens and text tokens
      processed independently with separate weights
    - single transformer blocks (single-stream): concatenated sequence
      (image + text tokens) shares the same group of weights
    - Final norm_out and proj_out layers

    This strategy provides:
    - mag_ratios: Pre-computed magnitude ratios for Flux (28 steps)
    - nearest_interp(): Interpolate mag_ratios to match num_inference_steps
    """

    @property
    def transformer_type(self) -> str:
        return "FluxTransformer2DModel"

    @property
    def mag_ratios(self) -> torch.Tensor:
        """Return default mag_ratios for Flux model."""
        return self.FLUX_MAG_RATIOS

    FLUX_MAG_RATIOS = torch.tensor(
        [1.0]
        + [
            1.21094,
            1.11719,
            1.07812,
            1.0625,
            1.03906,
            1.03125,
            1.03906,
            1.02344,
            1.03125,
            1.02344,
            0.98047,
            1.01562,
            1.00781,
            1.0,
            1.00781,
            1.0,
            1.00781,
            1.0,
            1.0,
            0.99609,
            0.99609,
            0.98047,
            0.98828,
            0.96484,
            0.95703,
            0.93359,
            0.89062,
        ]
    )

    def create_context(
        self,
        module: torch.nn.Module,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        timestep: torch.Tensor,
        guidance: torch.Tensor | None,
        **kwargs,
    ) -> MagCacheContext:
        """Create context for Flux model."""
        temb = (
            module.time_text_embed(timestep, guidance)
            if guidance is not None
            else module.time_text_embed(timestep)
        )

        def run_transformer_blocks():
            h = hidden_states
            e = encoder_hidden_states
            for block in module.transformer_blocks:
                e, h = block(
                    hidden_states=h,
                    encoder_hidden_states=e,
                    temb=temb,
                )
            return e, h

        def run_single_transformer_blocks(h):
            for block in module.single_transformer_blocks:
                h = block(
                    hidden_states=h,
                    encoder_hidden_states=torch.zeros(1, 1, h.shape[-1], device=h.device, dtype=h.dtype),
                    temb=temb,
                )
            return h

        def postprocess(e: torch.Tensor, h: torch.Tensor) -> Any:
            h = torch.cat([e, h], dim=1)
            h = module.norm_out(h, temb)
            output = module.proj_out(h)
            from diffusers.models.modeling_outputs import Transformer2DModelOutput

            return Transformer2DModelOutput(sample=output)

        return MagCacheContext(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
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
        context: MagCacheContext,
    ) -> torch.Tensor:
        """Compute residual for Flux single transformer blocks.

        For single transformer blocks, the output is concatenated (encoder + decoder).
        We need to extract encoder residual from the combined output.
        """
        if context is not None:
            encoder_hidden_states = context.encoder_hidden_states
            if encoder_hidden_states is not None:
                encoder_len = encoder_hidden_states.shape[1]
                if isinstance(output, tuple):
                    out_e = output[0]
                    out_h = output[1]
                else:
                    out_e = output[:, :encoder_len, :]
                    out_h = output[:, encoder_len:, :]

                e_res = out_e - encoder_hidden_states
                h_res = out_h - head_input
                return (e_res, h_res)

        return output

    def apply_residual(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """Apply residual by adding to hidden states."""
        return hidden_states + residual

    def apply_residual_tuple(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        residual: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply residual tuple to both hidden_states and encoder_hidden_states.

        Flux architecture:
        - encoder: 512 tokens
        - decoder: 4096 tokens

        The residual tuple (e_res, h_res) comes from compute_residual:
        - e_res: encoder residual (512 tokens)
        - h_res: decoder residual (4096 tokens)

        We apply residuals separately to encoder_hidden_states and hidden_states.
        """
        e_res, h_res = residual

        output = hidden_states + h_res
        enc_output = encoder_hidden_states + e_res

        return output, enc_output

    @staticmethod
    def register_blocks() -> None:
        """Register vLLM-Omni Flux transformer blocks with TransformerBlockRegistry.

        Blocks:
        - FluxTransformerBlock: dual-stream block
        - FluxSingleTransformerBlock: single-stream block
        """
        try:
            from vllm_omni.diffusion.models.flux.flux_transformer import (
                FluxTransformerBlock,
                FluxSingleTransformerBlock,
            )

            register_transformer_block(FluxTransformerBlock)
            register_transformer_block(FluxSingleTransformerBlock)
        except ImportError:
            pass

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


class MagCacheStrategyRegistry:
    """Registry for MagCache strategies by transformer type."""

    _registry: dict[str, MagCacheStrategy] = {}

    @classmethod
    def register(cls, strategy: MagCacheStrategy) -> None:
        """Register a strategy."""
        cls._registry[strategy.transformer_type] = strategy

    @classmethod
    def get(cls, transformer_type: str) -> MagCacheStrategy:
        """Get strategy for given transformer type."""
        if transformer_type not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Unknown model type: '{transformer_type}'. "
                f"Available types: {available}"
            )
        return cls._registry[transformer_type]

    @classmethod
    def get_if_exists(cls, transformer_type: str) -> MagCacheStrategy | None:
        """Get strategy if exists, None otherwise."""
        return cls._registry.get(transformer_type)


# Register default strategies
MagCacheStrategyRegistry.register(FluxMagCacheStrategy())
