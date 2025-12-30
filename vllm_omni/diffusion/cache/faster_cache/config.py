# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import torch


@dataclass
class FasterCacheConfig:
    """
    Configuration for FasterCache acceleration technique.
    
    FasterCache accelerates diffusion inference by:
    1. Skipping unconditional branch computation using frequency domain approximation
    2. Skipping attention layer computations based on iteration patterns
    3. Using cached attention outputs with linear extrapolation
    
    This implementation is based on the Diffusers FasterCache implementation.
    """
    
    # Spatial attention configuration
    spatial_attention_block_skip_range: int = 2
    spatial_attention_timestep_skip_range: Tuple[int, int] = (-1, 681)
    spatial_attention_block_identifiers: List[str] = field(
        default_factory=lambda: ["transformer_blocks"]
    )
    
    # Temporal attention configuration (optional)
    temporal_attention_block_skip_range: Optional[int] = None
    temporal_attention_timestep_skip_range: Tuple[int, int] = (-1, 681)
    temporal_attention_block_identifiers: List[str] = field(
        default_factory=lambda: ["temporal_transformer_blocks"]
    )
    
    # Frequency domain configuration
    low_frequency_weight_update_timestep_range: Tuple[int, int] = (99, 901)
    high_frequency_weight_update_timestep_range: Tuple[int, int] = (-1, 301)
    alpha_low_frequency: float = 1.1
    alpha_high_frequency: float = 1.1
    
    # Unconditional branch configuration
    unconditional_batch_skip_range: int = 5
    unconditional_batch_timestep_skip_range: Tuple[int, int] = (-1, 641)
    unconditional_conditional_input_kwargs_identifiers: List[str] = field(
        default_factory=lambda: ["hidden_states"]
    )
    
    # General configuration
    tensor_format: str = "BFCHW"  # BCFHW, BFCHW, BCHW
    is_guidance_distilled: bool = False
    
    # Callback functions (will be set by backend)
    current_timestep_callback: Optional[Callable[[], int]] = None
    attention_weight_callback: Optional[Callable[[torch.nn.Module], float]] = None
    low_frequency_weight_callback: Optional[Callable[[torch.nn.Module], float]] = None
    high_frequency_weight_callback: Optional[Callable[[torch.nn.Module], float]] = None
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.tensor_format not in ["BCFHW", "BFCHW", "BCHW"]:
            raise ValueError(
                f"tensor_format must be one of ['BCFHW', 'BFCHW', 'BCHW'], "
                f"got {self.tensor_format}"
            )
        
        if self.spatial_attention_block_skip_range is not None and self.spatial_attention_block_skip_range < 1:
            raise ValueError("spatial_attention_block_skip_range must be >= 1")
        
        if self.temporal_attention_block_skip_range is not None and self.temporal_attention_block_skip_range < 1:
            raise ValueError("temporal_attention_block_skip_range must be >= 1")
        
        if self.unconditional_batch_skip_range < 1:
            raise ValueError("unconditional_batch_skip_range must be >= 1")
        
        # Validate timestep ranges
        for range_name, range_val in [
            ("spatial_attention_timestep_skip_range", self.spatial_attention_timestep_skip_range),
            ("temporal_attention_timestep_skip_range", self.temporal_attention_timestep_skip_range),
            ("low_frequency_weight_update_timestep_range", self.low_frequency_weight_update_timestep_range),
            ("high_frequency_weight_update_timestep_range", self.high_frequency_weight_update_timestep_range),
            ("unconditional_batch_timestep_skip_range", self.unconditional_batch_timestep_skip_range),
        ]:
            if range_val[0] >= range_val[1]:
                raise ValueError(f"{range_name} must have start < end, got {range_val}")


def create_faster_cache_config_for_model(model_type: str) -> FasterCacheConfig:
    """
    Create model-specific FasterCache configuration.
    
    Args:
        model_type: Type of transformer model ('flux', 'cogvideox', 'stablediffusion', 'hunyuanvideo')
        
    Returns:
        Model-specific FasterCacheConfig
    """
    # Default configuration for common models
    configs = {
        "flux": FasterCacheConfig(
            temporal_attention_block_skip_range=2,
            tensor_format="BCHW",
        ),
        "cogvideox": FasterCacheConfig(
            temporal_attention_block_skip_range=3,
            spatial_attention_timestep_skip_range=(0, 50),
            tensor_format="BFCHW",
        ),
        "stablediffusion": FasterCacheConfig(
            spatial_attention_block_skip_range=4,
            spatial_attention_timestep_skip_range=(0, 20),
            tensor_format="BCHW",
        ),
        "hunyuanvideo": FasterCacheConfig(
            temporal_attention_block_skip_range=2,
            spatial_attention_block_skip_range=3,
            tensor_format="BCFHW",
        ),
    }
    
    if model_type not in configs:
        # Return default config for unknown models
        return FasterCacheConfig()
    
    return configs[model_type]
