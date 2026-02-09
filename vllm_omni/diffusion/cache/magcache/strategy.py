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
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from diffusers.hooks._helpers import TransformerBlockMetadata, TransformerBlockRegistry


def register_transformer_block(
    model_class: type,
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
    - Residual computation (how to calculate the residual for caching)
    - Residual application (how to apply cached residual)
    - Model-specific magnitude ratios

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
    def compute_residual(
        self,
        output: torch.Tensor,
        head_input: torch.Tensor,
    ) -> torch.Tensor:
        """Compute residual between block output and input.

        Args:
            output: Output from transformer blocks.
            head_input: Input to the first block.

        Returns:
            Residual tensor for caching.
        """
        pass

    @abstractmethod
    def apply_residual(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """Apply cached residual to hidden states.

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
        """Apply cached residual tuple to both hidden_states and encoder_hidden_states.

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

    @abstractmethod
    def compute_calibration_metrics(
        self,
        current_residual: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        previous_residual: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[float, float, float]:
        """Compute calibration metrics for mag_ratios generation.

        Args:
            current_residual: Residual from the current step.
            previous_residual: Residual from the previous step (None for first step).

        Returns:
            Tuple of (norm_ratio, norm_std, cos_dis):
            - norm_ratio: Mean ratio of current to previous residual norms
            - norm_std: Standard deviation of the norm ratios
            - cos_dis: Mean cosine dissimilarity (1 - cosine_similarity)
        """
        pass

    def get_calibration_metrics_names(self) -> tuple[str, str, str]:
        """Return the names of calibration metrics for logging.

        Returns:
            Tuple of metric names in order: (norm_ratio_name, norm_std_name, cos_dis_name)
        """
        return ("norm_ratio", "norm_std", "cos_dis")


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
    - compute_calibration_metrics: FLUX-specific metric computation
    """

    FLUX_MAG_RATIOS = torch.tensor(
        [
            1.0,
            1.07313,
            1.21035,
            1.04432,
            1.06818,
            1.05547,
            1.0183,
            1.03405,
            1.02574,
            1.03042,
            1.02739,
            1.01955,
            1.01585,
            1.02439,
            1.01154,
            1.01377,
            1.00994,
            1.01444,
            1.00839,
            1.02269,
            1.0007,
            1.00714,
            1.00484,
            1.01381,
            1.00426,
            0.99764,
            1.00778,
            1.00233,
        ]
    )

    @property
    def transformer_type(self) -> str:
        return "FluxTransformer2DModel"

    @property
    def mag_ratios(self) -> torch.Tensor:
        """Return default mag_ratios for Flux model."""
        return self.FLUX_MAG_RATIOS

    def compute_calibration_metrics(
        self,
        current_residual: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        previous_residual: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[float, float, float]:
        """Compute calibration metrics for FLUX model."""
        import torch.nn.functional as F

        if previous_residual is None:
            return 1.0, 0.0, 0.0

        curr_norm = torch.linalg.norm(current_residual.float(), dim=-1)
        prev_norm = torch.linalg.norm(previous_residual.float(), dim=-1)

        ratio = (curr_norm / (prev_norm + 1e-8)).mean().item()
        std = (curr_norm / (prev_norm + 1e-8)).std().item()
        cos_dis = (1 - F.cosine_similarity(current_residual, previous_residual, dim=-1, eps=1e-8)).mean().item()

        return ratio, std, cos_dis

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

    def compute_residual(
        self,
        output: torch.Tensor,
        head_input: torch.Tensor,
    ) -> torch.Tensor:
        """Compute residual for Flux single transformer blocks."""
        if isinstance(output, tuple):
            decoder_output = output[1] if len(output) > 1 else output[0]
        else:
            decoder_output = output - head_input
        return decoder_output

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
        """Apply residual tuple (for compatibility with hook interface)."""
        if isinstance(residual, tuple):
            decoder_residual = residual[1]
        else:
            decoder_residual = residual

        output = hidden_states + decoder_residual
        enc_output = encoder_hidden_states

        return output, enc_output

    @staticmethod
    def register_blocks() -> None:
        """Register vLLM-Omni Flux transformer blocks with TransformerBlockRegistry."""
        try:
            from vllm_omni.diffusion.models.flux.flux_transformer import (
                FluxSingleTransformerBlock,
                FluxTransformerBlock,
            )

            register_transformer_block(FluxTransformerBlock)
            register_transformer_block(FluxSingleTransformerBlock)
        except ImportError:
            pass


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
            raise ValueError(f"Unknown model type: '{transformer_type}'. Available types: {available}")
        return cls._registry[transformer_type]

    @classmethod
    def get_if_exists(cls, transformer_type: str) -> MagCacheStrategy | None:
        """Get strategy if exists, None otherwise."""
        return cls._registry.get(transformer_type)


# Register default strategies
MagCacheStrategyRegistry.register(FluxMagCacheStrategy())
