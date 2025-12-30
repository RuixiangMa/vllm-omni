# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Tuple, Union

import torch


class FasterCacheDenoiserState:
    """
    State for FasterCache denoiser-level operations.
    
    Manages state for unconditional branch approximation using frequency domain splitting.
    """
    
    def __init__(self) -> None:
        self.iteration: int = 0
        self.low_frequency_delta: Optional[torch.Tensor] = None
        self.high_frequency_delta: Optional[torch.Tensor] = None
        
    def reset(self) -> None:
        """Reset all state."""
        self.iteration = 0
        self.low_frequency_delta = None
        self.high_frequency_delta = None


class FasterCacheBlockState:
    """
    State for FasterCache block-level operations.
    
    Manages state for attention layer caching and approximation.
    """
    
    def __init__(self) -> None:
        self.iteration: int = 0
        self.batch_size: Optional[int] = None
        self.cache: Optional[Tuple[Union[torch.Tensor, Tuple[torch.Tensor, ...]], 
                                  Union[torch.Tensor, Tuple[torch.Tensor, ...]]]] = None
        
    def reset(self) -> None:
        """Reset all state."""
        self.iteration = 0
        self.batch_size = None
        self.cache = None


class FasterCacheState:
    """
    Main state manager for FasterCache operations.
    
    Coordinates state across denoiser and block levels.
    """
    
    def __init__(self) -> None:
        self.denoiser_state: Optional[FasterCacheDenoiserState] = None
        self.block_states: dict[str, FasterCacheBlockState] = {}
        self._global_iteration: int = 0
        
    def initialize_denoiser_state(self) -> FasterCacheDenoiserState:
        """Initialize denoiser state."""
        self.denoiser_state = FasterCacheDenoiserState()
        return self.denoiser_state
        
    def get_or_create_block_state(self, block_name: str) -> FasterCacheBlockState:
        """Get or create block state for a specific block."""
        if block_name not in self.block_states:
            self.block_states[block_name] = FasterCacheBlockState()
        return self.block_states[block_name]
        
    def reset(self) -> None:
        """Reset all states."""
        self._global_iteration = 0
        if self.denoiser_state:
            self.denoiser_state.reset()
        for block_state in self.block_states.values():
            block_state.reset()
            
    def reset_all(self) -> None:
        """Reset all states (alias for reset)."""
        self.reset()
            
    def increment_iteration(self) -> None:
        """Increment global iteration counter."""
        self._global_iteration += 1
        
    @property
    def global_iteration(self) -> int:
        """Get current global iteration."""
        return self._global_iteration