# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
FasterCache acceleration implementation for vLLM-Omni.

FasterCache accelerates diffusion inference by:
1. Skipping unconditional branch computation using frequency domain approximation
2. Skipping attention layer computations based on iteration patterns  
3. Using cached attention outputs with linear extrapolation

This implementation follows the decoupled architecture pattern established by TeaCache,
separating configuration, state management, and hook implementation.

Supported Models:
    - FluxPipeline: Flux transformer models
    - CogVideoXPipeline: CogVideoX video generation models  
    - StableDiffusionPipeline: Stable Diffusion models
    - HunyuanVideoPipeline: HunyuanVideo models

Usage:
    from vllm_omni import Omni

    # For Flux models
    omni = Omni(
        model="black-forest-labs/FLUX.1-dev",
        cache_backend="faster_cache",
        cache_config={"spatial_attention_block_skip_range": 2}
    )
    images = omni.generate("a cat")

    # Alternative: Using environment variable
    # export DIFFUSION_CACHE_BACKEND=faster_cache
"""

from vllm_omni.diffusion.cache.faster_cache.config import (
    FasterCacheConfig,
    create_faster_cache_config_for_model,
)
from vllm_omni.diffusion.cache.faster_cache.state import (
    FasterCacheState,
    FasterCacheDenoiserState,
    FasterCacheBlockState,
)
from vllm_omni.diffusion.cache.faster_cache.hook import (
    FasterCacheDenoiserHook,
    FasterCacheBlockHook,
)
from vllm_omni.diffusion.cache.faster_cache.backend import FasterCacheBackend

__all__ = [
    "FasterCacheConfig",
    "FasterCacheState",
    "FasterCacheDenoiserState", 
    "FasterCacheBlockState",
    "FasterCacheDenoiserHook",
    "FasterCacheBlockHook",
    "FasterCacheBackend",
    "create_faster_cache_config_for_model",
]


def create_faster_cache_backend(config: dict | FasterCacheConfig | None = None) -> "FasterCacheBackend":
    """
    Create a FasterCacheBackend instance.
    
    Args:
        config: Configuration for FasterCache. Can be:
            - dict: Configuration parameters
            - FasterCacheConfig: Pre-built config object
            - None: Use default configuration
            
    Returns:
        FasterCacheBackend instance
    """
    if isinstance(config, dict):
        faster_cache_config = FasterCacheConfig(**config)
    elif isinstance(config, FasterCacheConfig):
        faster_cache_config = config
    else:
        faster_cache_config = FasterCacheConfig()
        
    return FasterCacheBackend(faster_cache_config)
