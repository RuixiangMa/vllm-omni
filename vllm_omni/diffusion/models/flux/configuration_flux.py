# Copyright 2025 The vLLM-Omni Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from dataclasses import dataclass, field
from typing import Optional, Tuple, Union
import os


@dataclass
class FluxTransformerConfig:
    """
    Configuration for FLUX transformer models, optimized for vLLM-Omni architecture.
    
    This config class provides a clean, dataclass-based configuration that supports
    both standard FLUX.1-dev and extended variants like FLUX.1-Kontext-dev.
    """
    
    # Core transformer architecture
    patch_size: int = 1
    in_channels: int = 16
    out_channels: int = 16
    num_layers: int = 19
    num_single_layers: int = 38
    attention_head_dim: int = 128
    num_attention_heads: int = 24
    joint_attention_dim: int = 4096
    pooled_projection_dim: int = 768
    guidance_embeds: bool = True
    axes_dims_rope: Tuple[int, int, int] = (16, 56, 56)
    qkv_mlp_ratio: float = 4.0
    
    # Extended architecture support (for Kontext and future variants)
    text_projection_dim: Optional[int] = None
    image_projection_dim: Optional[int] = None
    dual_input_mode: bool = False
    
    # VAE integration parameters
    vae_scale_factor: int = 8
    latent_channels: int = 16
    scaling_factor: float = 0.3611
    shift_factor: Optional[float] = None
    
    # Model variant identification
    model_type: str = "flux"  # "flux", "flux_kontext", etc.
    
    def __post_init__(self):
        """Post-initialization to set derived parameters."""
        # Calculate correct number of attention heads
        if self.joint_attention_dim % self.attention_head_dim != 0:
            raise ValueError(f"joint_attention_dim ({self.joint_attention_dim}) must be divisible by attention_head_dim ({self.attention_head_dim})")
        
        # Update number of heads to ensure compatibility
        self.num_attention_heads = self.joint_attention_dim // self.attention_head_dim
        
        # Set default projection dimensions if not provided
        if self.text_projection_dim is None:
            self.text_projection_dim = self.joint_attention_dim
        if self.image_projection_dim is None:
            self.image_projection_dim = self.joint_attention_dim
        
        # Validate configuration
        if self.dual_input_mode and (self.text_projection_dim is None or self.image_projection_dim is None):
            raise ValueError("Dual input mode requires both text and image projection dimensions")


@dataclass 
class FluxPipelineConfig:
    """
    Configuration for FLUX pipeline components, following vLLM-Omni patterns.
    
    This separates pipeline-level configuration from transformer architecture,
    enabling better modularity and component reuse.
    """
    
    # Model sources
    model_path: str
    local_files_only: bool = False
    
    # Component subfolders
    transformer_subfolder: str = "transformer"
    vae_subfolder: str = "vae"
    text_encoder_subfolder: str = "text_encoder"
    text_encoder_2_subfolder: str = "text_encoder_2"
    image_encoder_subfolder: Optional[str] = None
    tokenizer_subfolder: str = "tokenizer"
    tokenizer_2_subfolder: str = "tokenizer_2"
    scheduler_subfolder: str = "scheduler"
    feature_extractor_subfolder: Optional[str] = None
    
    # Generation parameters
    default_num_inference_steps: int = 28
    default_guidance_scale: float = 3.5
    default_image_strength: float = 0.9  # For image-to-image
    
    # Processing parameters
    max_sequence_length: int = 512
    patch_size: int = 2  # FLUX uses 2x2 patches
    
    # Variant-specific settings
    supports_dual_input: bool = False  # For Kontext variants
    
    @classmethod
    def from_od_config(cls, od_config) -> "FluxPipelineConfig":
        """
        Create FluxPipelineConfig from OmniDiffusionConfig.
        
        Args:
            od_config: OmniDiffusionConfig instance
            
        Returns:
            FluxPipelineConfig instance
        """
        model_path = od_config.model
        local_files_only = os.path.exists(model_path)
        
        # Determine variant-specific settings
        supports_dual_input = "kontext" in od_config.model_class_name.lower()
        
        return cls(
            model_path=model_path,
            local_files_only=local_files_only,
            supports_dual_input=supports_dual_input,
            default_num_inference_steps=getattr(od_config, "num_inference_steps", 28),
            default_guidance_scale=getattr(od_config, "guidance_scale", 3.5),
        )