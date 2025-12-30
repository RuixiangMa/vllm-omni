# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional, Any
import logging
import torch
import torch.nn as nn

from vllm_omni.diffusion.cache.base import CacheBackend
from vllm_omni.diffusion.data import DiffusionCacheConfig
from vllm_omni.diffusion.cache.faster_cache.config import FasterCacheConfig, create_faster_cache_config_for_model
from vllm_omni.diffusion.cache.faster_cache.state import FasterCacheState, FasterCacheDenoiserState, FasterCacheBlockState
from vllm_omni.diffusion.cache.faster_cache.hook import FasterCacheDenoiserHook, FasterCacheBlockHook

logger = logging.getLogger(__name__)


class FasterCacheBackend(CacheBackend):
    """
    FasterCache backend implementation for vLLM-Omni.
    
    FasterCache accelerates diffusion inference by:
    1. Skipping unconditional branch computation using frequency domain approximation
    2. Skipping attention layer computations based on iteration patterns  
    3. Using cached attention outputs with linear extrapolation
    
    This implementation follows the decoupled architecture pattern established by TeaCache,
    separating configuration, state management, and hook implementation.
    """
    
    def __init__(self, config: Any = None):
        """
        Initialize FasterCache backend.
        
        Args:
            config: DiffusionCacheConfig instance with FasterCache parameters.
        """
        super().__init__(config)
        
        # Create FasterCache state manager
        self.faster_cache_state = FasterCacheState()
        
        # FasterCache config will be created in enable() when we know the transformer type
        self.faster_cache_config: Optional[FasterCacheConfig] = None
        
        # Current timestep tracking
        self._current_timestep: int = 0
        
        # Hook references
        self._denoiser_hook: Optional[FasterCacheDenoiserHook] = None
        self._block_hooks: dict[str, FasterCacheBlockHook] = {}
        
    def enable(self, model: nn.Module, **kwargs) -> None:
        """
        Enable FasterCache acceleration on the model.
        
        Args:
            model: The transformer model to accelerate
            **kwargs: Additional parameters including:
                - model_type: Type of model (e.g., 'cogvideox', 'ltx_video')
                - custom_config: Custom FasterCacheConfig instance
        """
        # Determine model type and create appropriate config
        model_type = kwargs.get('model_type', 'cogvideox')
        custom_config = kwargs.get('custom_config')
        
        if custom_config is not None:
            self.faster_cache_config = custom_config
        else:
            self.faster_cache_config = create_faster_cache_config_for_model(model_type)
            
        # Set up callbacks
        self.faster_cache_config.current_timestep_callback = self._get_current_timestep
        
        # Apply hooks to the model
        self._apply_hooks(model)
        
        self.enabled = True
        logger.info(f"FasterCache enabled for {model_type} model")
        
    def _get_current_timestep(self) -> int:
        """Get current timestep for callbacks."""
        return self._current_timestep
        
    def _apply_hooks(self, model: nn.Module) -> None:
        """
        Apply FasterCache hooks to the model.
        
        Args:
            model: The transformer model to accelerate
        """
        # Create denoiser hook for CFG acceleration
        self._denoiser_hook = FasterCacheDenoiserHook(self.faster_cache_config, self.faster_cache_state, self._get_current_timestep)
        
        # Register hooks on attention layers
        for name, module in model.named_modules():
            if self._is_attention_layer(name, module):
                # Create block hook for this attention layer
                block_hook = FasterCacheBlockHook(self.faster_cache_config, self.faster_cache_state, name, self._get_current_timestep)
                self._block_hooks[name] = block_hook
                
                # Register hook
                module.register_forward_hook(block_hook.hook_fn)
                
                logger.debug(f"FasterCache hook applied to attention layer: {name}")
                
    def _is_attention_layer(self, name: str, module: nn.Module) -> bool:
        """
        Check if module is an attention layer that should be cached.
        
        Args:
            name: Module name
            module: Module instance
            
        Returns:
            True if this is an attention layer that should be cached
        """
        # Check if it's a spatial or temporal attention layer
        return (self._is_spatial_attention(name, module) or 
                self._is_temporal_attention(name, module))
                
    def _is_spatial_attention(self, name: str, module: nn.Module) -> bool:
        """Check if module is spatial attention layer."""
        if not hasattr(module, 'forward'):
            return False
            
        # Check if name matches spatial attention identifiers
        for identifier in self.faster_cache_config.spatial_attention_block_identifiers:
            if identifier in name:
                return True
                
        return False
        
    def _is_temporal_attention(self, name: str, module: nn.Module) -> bool:
        """Check if module is temporal attention layer."""
        if not hasattr(module, 'forward'):
            return False
            
        # Check if name matches temporal attention identifiers
        for identifier in self.faster_cache_config.temporal_attention_block_identifiers:
            if identifier in name:
                return True
                
        return False
        
    def get_denoiser_hook(self) -> Optional[FasterCacheDenoiserHook]:
        """Get the denoiser hook."""
        return self._denoiser_hook
        
    def get_block_hook(self, name: str) -> Optional[FasterCacheBlockHook]:
        """Get block hook by name."""
        return self._block_hooks.get(name)
        
    def get_config(self) -> Optional[FasterCacheConfig]:
        """Get current FasterCache configuration."""
        return self.faster_cache_config
        
    def get_state(self) -> FasterCacheState:
        """Get FasterCache state manager."""
        return self.faster_cache_state
        
    def refresh(self, pipeline: Any, num_inference_steps: int, verbose: bool = True) -> None:
        """
        Refresh FasterCache state for new generation.
        
        This method clears cached attention outputs, resets counters, and prepares
        for a new generation with the specified number of inference steps.
        
        Args:
            pipeline: Diffusion pipeline instance
            num_inference_steps: Number of inference steps for the current generation
            verbose: Whether to log refresh operations (default: True)
        """
        # Reset all cached states
        self.faster_cache_state.reset()
        
        # Reset current timestep
        self._current_timestep = 0
            
        if verbose:
            logger.info(f"FasterCache refreshed for {num_inference_steps} inference steps")
