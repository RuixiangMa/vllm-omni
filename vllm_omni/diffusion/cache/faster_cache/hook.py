# SPDX-License-Identifier: Apache-2.0

import re
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import logging

from vllm_omni.diffusion.cache.faster_cache.config import FasterCacheConfig
from vllm_omni.diffusion.cache.faster_cache.state import (
    FasterCacheBlockState,
    FasterCacheDenoiserState,
)

logger = logging.get_logger(__name__)


@torch.no_grad()
def _split_low_high_freq(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split tensor into low and high frequency components using FFT.
    
    Args:
        x: Input tensor of shape (..., H, W)
        
    Returns:
        Tuple of (low_freq_fft, high_freq_fft) in frequency domain
    """
    fft = torch.fft.fft2(x)
    fft_shifted = torch.fft.fftshift(fft)
    height, width = x.shape[-2:]
    radius = min(height, width) // 5

    y_grid, x_grid = torch.meshgrid(torch.arange(height), torch.arange(width))
    center_x, center_y = width // 2, height // 2
    mask = (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2 <= radius**2

    low_freq_mask = mask.unsqueeze(0).unsqueeze(0).to(x.device)
    high_freq_mask = ~low_freq_mask

    low_freq_fft = fft_shifted * low_freq_mask
    high_freq_fft = fft_shifted * high_freq_mask

    return low_freq_fft, high_freq_fft


class FasterCacheDenoiserHook:
    """
    Hook for FasterCache denoiser-level operations.
    
    Manages unconditional branch skipping using frequency domain approximation.
    """
    
    def __init__(
        self,
        config: FasterCacheConfig,
        state: FasterCacheDenoiserState,
        current_timestep_callback: Callable[[], int],
    ) -> None:
        self.config = config
        self.state = state
        self.current_timestep_callback = current_timestep_callback
        
    def should_skip_unconditional(self) -> bool:
        """Check if unconditional branch should be skipped."""
        if self.config.is_guidance_distilled:
            return False
            
        is_within_timestep_range = (
            self.config.unconditional_batch_timestep_skip_range[0]
            < self.current_timestep_callback()
            < self.config.unconditional_batch_timestep_skip_range[1]
        )
        
        return (
            self.state.iteration > 0
            and is_within_timestep_range
            and self.state.iteration % self.config.unconditional_batch_skip_range != 0
        )
        
    def _get_conditional_input(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Extract conditional branch from batchwise-concatenated input."""
        _, cond = input_tensor.chunk(2, dim=0)
        return cond
        
    def apply_frequency_weights(self, module: nn.Module) -> Tuple[float, float]:
        """Apply frequency domain weights based on current timestep."""
        low_weight = self.config.low_frequency_weight_callback(module) if self.config.low_frequency_weight_callback else 1.0
        high_weight = self.config.high_frequency_weight_callback(module) if self.config.high_frequency_weight_callback else 1.0
        return low_weight, high_weight
        
    def approximate_unconditional_branch(
        self, 
        conditional_output: torch.Tensor, 
        module: nn.Module,
        tensor_format: str = "BFCHW"
    ) -> torch.Tensor:
        """
        Approximate unconditional branch using frequency domain method.
        
        Args:
            conditional_output: Output from conditional branch
            module: Current module
            tensor_format: Tensor format (BCFHW, BFCHW, BCHW)
            
        Returns:
            Approximated unconditional + conditional output
        """
        batch_size = conditional_output.size(0)
        
        # Apply frequency weights to deltas
        low_weight, high_weight = self.apply_frequency_weights(module)
        if self.state.low_frequency_delta is not None:
            self.state.low_frequency_delta = self.state.low_frequency_delta * low_weight
        if self.state.high_frequency_delta is not None:
            self.state.high_frequency_delta = self.state.high_frequency_delta * high_weight

        # Handle tensor format
        hidden_states = conditional_output
        if tensor_format == "BCFHW":
            hidden_states = hidden_states.permute(0, 2, 1, 3, 4)
        if tensor_format == "BCFHW" or tensor_format == "BFCHW":
            hidden_states = hidden_states.flatten(0, 1)

        # Split into frequency components
        low_freq_cond, high_freq_cond = _split_low_high_freq(hidden_states.float())

        # Approximate unconditional branch
        low_freq_uncond = (self.state.low_frequency_delta + low_freq_cond 
                          if self.state.low_frequency_delta is not None else low_freq_cond)
        high_freq_uncond = (self.state.high_frequency_delta + high_freq_cond 
                           if self.state.high_frequency_delta is not None else high_freq_cond)
        uncond_freq = low_freq_uncond + high_freq_uncond

        # Convert back to spatial domain
        uncond_states = torch.fft.ifftshift(uncond_freq)
        uncond_states = torch.fft.ifft2(uncond_states).real

        # Restore tensor format
        if tensor_format == "BCFHW" or tensor_format == "BFCHW":
            uncond_states = uncond_states.unflatten(0, (batch_size, -1))
            hidden_states = hidden_states.unflatten(0, (batch_size, -1))
        if tensor_format == "BCFHW":
            uncond_states = uncond_states.permute(0, 2, 1, 3, 4)
            hidden_states = hidden_states.permute(0, 2, 1, 3, 4)

        # Concatenate unconditional and conditional branches
        uncond_states = uncond_states.to(hidden_states.dtype)
        result = torch.cat([uncond_states, hidden_states], dim=0)
        
        return result
        
    def update_frequency_deltas(
        self, 
        unconditional_output: torch.Tensor, 
        conditional_output: torch.Tensor,
        tensor_format: str = "BFCHW"
    ) -> None:
        """
        Update frequency domain deltas from computed branches.
        
        Args:
            unconditional_output: Output from unconditional branch
            conditional_output: Output from conditional branch  
            tensor_format: Tensor format (BCFHW, BFCHW, BCHW)
        """
        # Handle tensor format
        uncond_states = unconditional_output
        cond_states = conditional_output
        
        if tensor_format == "BCFHW":
            uncond_states = uncond_states.permute(0, 2, 1, 3, 4)
            cond_states = cond_states.permute(0, 2, 1, 3, 4)
        if tensor_format == "BCFHW" or tensor_format == "BFCHW":
            uncond_states = uncond_states.flatten(0, 1)
            cond_states = cond_states.flatten(0, 1)

        # Split into frequency components and compute deltas
        low_freq_uncond, high_freq_uncond = _split_low_high_freq(uncond_states.float())
        low_freq_cond, high_freq_cond = _split_low_high_freq(cond_states.float())
        
        self.state.low_frequency_delta = low_freq_uncond - low_freq_cond
        self.state.high_frequency_delta = high_freq_uncond - high_freq_cond
        
    def increment_iteration(self) -> None:
        """Increment iteration counter."""
        self.state.iteration += 1


class FasterCacheBlockHook:
    """
    Hook for FasterCache block-level operations.
    
    Manages attention layer caching and approximation.
    """
    
    def __init__(
        self,
        config: FasterCacheConfig,
        state: FasterCacheBlockState,
        block_type: str,
        current_timestep_callback: Callable[[], int],
    ) -> None:
        self.config = config
        self.state = state
        self.block_type = block_type  # "spatial" or "temporal"
        self.current_timestep_callback = current_timestep_callback
        
    def get_skip_range(self) -> int:
        """Get skip range for this block type."""
        if self.block_type == "spatial":
            return self.config.spatial_attention_block_skip_range
        elif self.block_type == "temporal":
            return self.config.temporal_attention_block_skip_range
        else:
            return 1  # No skipping
            
    def get_timestep_range(self) -> Tuple[int, int]:
        """Get timestep range for this block type."""
        if self.block_type == "spatial":
            return self.config.spatial_attention_timestep_skip_range
        elif self.block_type == "temporal":
            return self.config.temporal_attention_timestep_skip_range
        else:
            return (-1, -1)  # No valid range
            
    def should_skip_attention(self, batch_size: int) -> bool:
        """Check if attention computation should be skipped."""
        skip_range = self.get_skip_range()
        if skip_range is None or skip_range <= 1:
            return False
            
        timestep_range = self.get_timestep_range()
        is_within_timestep_range = (
            timestep_range[0] < self.current_timestep_callback() < timestep_range[1]
        )
        
        if not is_within_timestep_range:
            return False
            
        # Check if we should compute attention this iteration
        should_compute_attention = (
            self.state.iteration > 0 and 
            self.state.iteration % skip_range == 0
        )
        should_skip_attention = not should_compute_attention
        
        # Don't skip if we need both unconditional and conditional branches
        if should_skip_attention:
            should_skip_attention = (
                self.config.is_guidance_distilled or 
                self.state.batch_size != batch_size
            )
            
        return should_skip_attention
        
    def _compute_approximated_attention_output(
        self, 
        t_2_output: Union[torch.Tensor, Tuple[torch.Tensor, ...]], 
        t_output: Union[torch.Tensor, Tuple[torch.Tensor, ...]], 
        weight: float, 
        batch_size: int
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Compute approximated attention output using linear extrapolation.
        
        Args:
            t_2_output: Output from 2 iterations ago
            t_output: Output from previous iteration
            weight: Extrapolation weight
            batch_size: Current batch size
            
        Returns:
            Approximated output
        """
        def process_single_tensor(t_2: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            # Handle batchwise-concatenated unconditional-conditional outputs
            if t_2.size(0) != batch_size and t_2.size(0) == 2 * batch_size:
                t_2 = t_2[batch_size:]
            if t.size(0) != batch_size and t.size(0) == 2 * batch_size:
                t = t[batch_size:]
            return t + (t - t_2) * weight
            
        if torch.is_tensor(t_2_output):
            return process_single_tensor(t_2_output, t_output)
        else:
            # Handle multiple return tensors
            result = ()
            for t_2, t in zip(t_2_output, t_output):
                processed = process_single_tensor(t_2, t)
                result += (processed,)
            return result
            
    def update_cache(
        self, 
        output: Union[torch.Tensor, Tuple[torch.Tensor, ...]], 
        batch_size: int
    ) -> None:
        """Update cache with new output."""
        # Extract conditional branch output for caching
        if torch.is_tensor(output):
            cache_output = output
            if (not self.config.is_guidance_distilled and 
                cache_output.size(0) == self.state.batch_size):
                cache_output = cache_output.chunk(2, dim=0)[1]
        else:
            cache_output = ()
            for out in output:
                if (not self.config.is_guidance_distilled and 
                    out.size(0) == self.state.batch_size):
                    out = out.chunk(2, dim=0)[1]
                cache_output += (out,)
                
        # Update cache
        if self.state.cache is None:
            self.state.cache = [cache_output, cache_output]
        else:
            self.state.cache = [self.state.cache[-1], cache_output]
            
    def increment_iteration(self) -> None:
        """Increment iteration counter."""
        self.state.iteration += 1
