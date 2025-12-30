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

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import FP32LayerNorm
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import QKVParallelLinear, ReplicatedLinear
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.models.flux.configuration_flux import FluxTransformerConfig

logger = init_logger(__name__)


class FluxTransformerBlock(nn.Module):
    """
    Single transformer block for FLUX architecture.
    Follows Diffusers' pattern of modular transformer components.
    """
    
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qkv_mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        activation_fn: str = "gelu",
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        
        self.dim = dim
        self.attention_head_dim = attention_head_dim
        # Calculate number of heads to ensure dim = num_heads * head_dim
        self.num_attention_heads = dim // attention_head_dim if dim % attention_head_dim == 0 else num_attention_heads
        
        # Self-attention
        self.norm1 = FP32LayerNorm(dim, eps=norm_eps)
        self.attn = Attention(
            num_heads=num_attention_heads,
            head_size=attention_head_dim,
            causal=False,
            softmax_scale=None,
            num_kv_heads=num_attention_heads,
        )
        
        # Feed-forward
        self.norm2 = FP32LayerNorm(dim, eps=norm_eps)
        self.ff = FeedForward(
            dim,
            dim_out=dim,  # Explicitly set output dimension to match input
            mult=qkv_mlp_ratio,  # Use multiplier instead of absolute hidden dimension
            dropout=dropout,
            activation_fn=activation_fn,
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[dict] = None,
    ) -> torch.Tensor:
        """Forward pass through transformer block."""
        
        # Self-attention (vLLM-Omni style - reshape for multi-head attention)
        batch_size, seq_len, dim = hidden_states.shape
        norm_hidden_states = self.norm1(hidden_states)
        
        # Reshape for multi-head attention: (batch, seq_len, dim) -> (batch, seq_len, num_heads, head_dim)
        norm_hidden_states_reshaped = norm_hidden_states.unflatten(2, (self.num_attention_heads, self.attention_head_dim))
        
        attn_output = self.attn(
            query=norm_hidden_states_reshaped,
            key=norm_hidden_states_reshaped,
            value=norm_hidden_states_reshaped,
            attn_metadata=None,
        )
        
        # Reshape back: (batch, seq_len, num_heads, head_dim) -> (batch, seq_len, dim)
        attn_output = attn_output.flatten(2)
        hidden_states = hidden_states + attn_output
        
        # Feed-forward
        norm_hidden_states = self.norm2(hidden_states)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + ff_output
        
        return hidden_states


class FluxTransformer2DModel(nn.Module):
    """
    FLUX Transformer model following Diffusers' 2D transformer pattern.
    
    This implementation provides a modular, configurable transformer that can
    support both standard FLUX.1-dev and variants like FLUX.1-Kontext-dev.
    """

    def __init__(self, config: FluxTransformerConfig):
        super().__init__()
        
        self.config = config
        
        # Store key dimensions from config
        self.patch_size = config.patch_size
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.num_layers = config.num_layers
        self.num_single_layers = config.num_single_layers
        self.attention_head_dim = config.attention_head_dim
        self.num_attention_heads = config.num_attention_heads
        self.joint_attention_dim = config.joint_attention_dim
        self.pooled_projection_dim = config.pooled_projection_dim
        self.guidance_embeds = config.guidance_embeds
        self.axes_dims_rope = config.axes_dims_rope
        self.qkv_mlp_ratio = config.qkv_mlp_ratio
        
        # Set projection dimensions with fallback
        self.text_projection_dim = config.text_projection_dim or config.joint_attention_dim
        self.image_projection_dim = config.image_projection_dim or config.joint_attention_dim
        
        # Time embedding
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedding = TimestepEmbedding(in_channels=256, time_embed_dim=config.joint_attention_dim)
        
        # Pooled text projection
        self.text_proj = PixArtAlphaTextProjection(
            config.pooled_projection_dim, config.joint_attention_dim, act_fn="silu"
        )
        
        # Image projection for dual input support
        if config.dual_input_mode:
            self.image_proj = PixArtAlphaTextProjection(
                config.pooled_projection_dim, config.joint_attention_dim, act_fn="silu"
            )
        else:
            self.image_proj = None
        
        # Guidance embedding
        if config.guidance_embeds:
            self.guidance_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=config.joint_attention_dim)
        else:
            self.guidance_embedder = None
        
        # Patch embedding
        self.patch_embed = ReplicatedLinear(
            config.in_channels,
            config.joint_attention_dim,
            bias=True,
        )
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            FluxTransformerBlock(
                dim=config.joint_attention_dim,
                num_attention_heads=config.num_attention_heads,
                attention_head_dim=config.attention_head_dim,
                qkv_mlp_ratio=config.qkv_mlp_ratio,
            )
            for _ in range(config.num_layers)
        ])
        
        # Single transformer blocks (for FLUX architecture)
        self.single_transformer_blocks = nn.ModuleList([
            FluxTransformerBlock(
                dim=config.joint_attention_dim,
                num_attention_heads=config.num_attention_heads,
                attention_head_dim=config.attention_head_dim,
                qkv_mlp_ratio=config.qkv_mlp_ratio,
            )
            for _ in range(config.num_single_layers)
        ])
        
        # Output projection
        self.norm_out = FP32LayerNorm(config.joint_attention_dim, eps=1e-6)
        self.proj_out = ReplicatedLinear(
            config.joint_attention_dim,
            config.patch_size * config.patch_size * config.out_channels,
            bias=True,
        )
        
    def _build_positional_encoding(self, height: int, width: int) -> torch.Tensor:
        """Build positional encoding for FLUX architecture based on actual input dimensions."""
        # Calculate actual sequence length after patchification
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        seq_len = num_patches_h * num_patches_w
        
        # Create positional encoding that matches the sequence length
        pos_embed = torch.zeros(1, seq_len, self.joint_attention_dim)
        return pos_embed

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.Tensor = None,
        guidance: torch.Tensor = None,
        image_pooled_projections: torch.Tensor = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        Forward pass through the FLUX transformer.
        
        Args:
            hidden_states: Input hidden states
            encoder_hidden_states: Text encoder hidden states
            pooled_projections: Pooled text projections
            timestep: Timestep tensor
            guidance: Guidance scale tensor
            image_pooled_projections: Pooled image projections (for dual input)
            return_dict: Whether to return dict output
            
        Returns:
            Transformer output
        """
        
        batch_size, channels, height, width = hidden_states.shape
        
        # Patchify input
        hidden_states = self._patchify(hidden_states)
        
        # Apply patch embedding
        hidden_states = self.patch_embed(hidden_states)
        
        # Handle tuple output from ReplicatedLinear
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        
        # Add positional encoding (dynamically generated based on input size)
        pos_embed = self._build_positional_encoding(height, width).to(device=hidden_states.device, dtype=hidden_states.dtype)
        hidden_states = hidden_states + pos_embed
        
        # Time embedding
        if timestep is not None:
            timesteps_proj = self.time_proj(timestep)
            timesteps_emb = self.time_embedding(timesteps_proj.to(dtype=hidden_states.dtype))
            
            # Add timestep embedding to hidden states
            hidden_states = hidden_states + timesteps_emb.unsqueeze(1)
        
        # Pooled projections
        if pooled_projections is not None:
            text_proj = self.text_proj(pooled_projections)
            hidden_states = hidden_states + text_proj.unsqueeze(1)
        
        # Image projections (for dual input mode)
        if self.image_proj is not None and image_pooled_projections is not None:
            image_proj = self.image_proj(image_pooled_projections)
            hidden_states = hidden_states + image_proj.unsqueeze(1)
        
        # Guidance embedding
        if self.guidance_embedder is not None and guidance is not None:
            guidance_proj = self.time_proj(guidance)
            guidance_emb = self.guidance_embedder(guidance_proj.to(dtype=hidden_states.dtype))
            hidden_states = hidden_states + guidance_emb.unsqueeze(1)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
            )
        
        # Apply single transformer blocks
        for block in self.single_transformer_blocks:
            hidden_states = block(hidden_states, timestep=timestep)
        
        # Output projection
        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.proj_out(hidden_states)
        
        # Handle tuple output from ReplicatedLinear
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        
        # Unpatchify
        hidden_states = self._unpatchify(hidden_states, height, width)
        
        if not return_dict:
            return hidden_states
        
        return Transformer2DModelOutput(sample=hidden_states)
    
    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert image to patches."""
        batch_size, channels, height, width = x.shape
        patch_size = self.patch_size
        
        # Reshape to patches
        x = x.view(batch_size, channels, height // patch_size, patch_size, width // patch_size, patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(batch_size, (height // patch_size) * (width // patch_size), channels * patch_size * patch_size)
        
        return x
    
    def _unpatchify(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Convert patches back to image."""
        batch_size, num_patches, patch_dim = x.shape
        channels = self.out_channels
        patch_size = self.patch_size
        
        # Reshape from patches to image
        x = x.view(batch_size, height // patch_size, width // patch_size, channels, patch_size, patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(batch_size, channels, height, width)
        
        return x