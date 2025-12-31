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

from typing import Any, Optional, Union

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

logger = init_logger(__name__)


class FluxKontextTransformer2DModel(nn.Module):
    """
    FLUX.1-Kontext-dev Transformer model for dual-input (text+image) image editing.
    
    This transformer supports both text and image inputs, enabling image-to-image
    generation with text guidance. It's a 12B parameter model based on the FLUX
    transformer architecture with modifications for handling dual inputs.
    
    Reference: https://bfl.ai/announcements/flux-1-kontext-dev
    """

    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 16,
        out_channels: int = 16,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = True,
        axes_dims_rope: list[int] = [16, 56, 56],
        qkv_mlp_ratio: float = 4.0,
        text_projection_dim: Optional[int] = None,
        image_projection_dim: Optional[int] = None,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.num_single_layers = num_single_layers
        self.attention_head_dim = attention_head_dim
        self.num_attention_heads = num_attention_heads
        self.joint_attention_dim = joint_attention_dim
        self.pooled_projection_dim = pooled_projection_dim
        self.guidance_embeds = guidance_embeds
        self.axes_dims_rope = axes_dims_rope
        self.qkv_mlp_ratio = qkv_mlp_ratio
        
        # Set default projection dimensions if not provided
        self.text_projection_dim = text_projection_dim or joint_attention_dim
        self.image_projection_dim = image_projection_dim or joint_attention_dim

        # Time embedding
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedding = TimestepEmbedding(in_channels=256, time_embed_dim=joint_attention_dim)

        # Pooled text projection
        self.text_proj = PixArtAlphaTextProjection(
            pooled_projection_dim, joint_attention_dim, act_fn="silu"
        )

        # Image projection (for dual input support)
        self.image_proj = PixArtAlphaTextProjection(
            pooled_projection_dim, joint_attention_dim, act_fn="silu"
        ) if image_projection_dim else None

        # Guidance embedding (for classifier-free guidance)
        if guidance_embeds:
            self.guidance_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=joint_attention_dim)

        # Patch embedding
        self.patch_embed = ReplicatedLinear(
            in_channels,
            joint_attention_dim,
            bias=True,
        )

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            FluxKontextTransformerBlock(
                dim=joint_attention_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                qkv_mlp_ratio=qkv_mlp_ratio,
                axes_dims_rope=axes_dims_rope,
            )
            for _ in range(num_layers)
        ])

        # Single transformer blocks (for additional processing)
        self.single_transformer_blocks = nn.ModuleList([
            FluxKontextSingleTransformerBlock(
                dim=joint_attention_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                qkv_mlp_ratio=qkv_mlp_ratio,
                axes_dims_rope=axes_dims_rope,
            )
            for _ in range(num_single_layers)
        ])

        # Output projection
        self.norm_out = FP32LayerNorm(joint_attention_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = ReplicatedLinear(
            joint_attention_dim,
            out_channels * patch_size * patch_size,
            bias=True,
            gather_output=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        pooled_projections_image: Optional[torch.Tensor] = None,
        guidance: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        Forward pass of the FLUX.1-Kontext transformer.
        
        Args:
            hidden_states: Input latent tensor
            timestep: Timestep tensor for diffusion
            encoder_hidden_states: Text encoder hidden states
            pooled_projections: Pooled text projections
            encoder_hidden_states_image: Image encoder hidden states (optional)
            pooled_projections_image: Pooled image projections (optional)
            guidance: Guidance scale tensor (optional)
            return_dict: Whether to return a dictionary
            
        Returns:
            Transformed hidden states
        """
        batch_size = hidden_states.shape[0]
        height, width = hidden_states.shape[-2:]

        # Time embedding
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.time_embedding(timesteps_proj.to(dtype=hidden_states.dtype))

        # Text projection
        text_proj = self.text_proj(pooled_projections)
        
        # Image projection (if provided)
        if encoder_hidden_states_image is not None and pooled_projections_image is not None and self.image_proj is not None:
            image_proj = self.image_proj(pooled_projections_image)
            # Combine text and image projections
            pooled_proj = text_proj + image_proj
        else:
            pooled_proj = text_proj

        # Guidance embedding (if provided)
        if guidance is not None and self.guidance_embeds:
            guidance_proj = self.guidance_embedder(self.time_proj(guidance))
            pooled_proj = pooled_proj + guidance_proj.unsqueeze(1)

        # Patch embedding
        hidden_states = self.patch_embed(hidden_states.flatten(2).transpose(1, 2))

        # Apply transformer blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                temb=timesteps_emb,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_image=encoder_hidden_states_image,
            )

        # Apply single transformer blocks
        for block in self.single_transformer_blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                temb=timesteps_emb,
            )

        # Output projection
        hidden_states = self.norm_out(hidden_states, pooled_proj)
        hidden_states = self.proj_out(hidden_states)

        # Reshape back to image format
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, height, width)

        if return_dict:
            return Transformer2DModelOutput(sample=hidden_states)
        else:
            return hidden_states


class FluxKontextTransformerBlock(nn.Module):
    """FLUX transformer block with dual input support."""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qkv_mlp_ratio: float = 4.0,
        axes_dims_rope: list[int] = [16, 56, 56],
    ):
        super().__init__()

        self.norm1 = FP32LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            qkv_mlp_ratio=qkv_mlp_ratio,
            rotary_emb=True,
            rotary_emb_dim=attention_head_dim,
            rotary_emb_axes_dims=axes_dims_rope,
        )

        self.norm2 = FP32LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn2 = Attention(
            dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            qkv_mlp_ratio=qkv_mlp_ratio,
            cross_attention_dim=dim,  # For text conditioning
            rotary_emb=True,
            rotary_emb_dim=attention_head_dim,
            rotary_emb_axes_dims=axes_dims_rope,
        )

        # Optional image cross-attention
        self.norm3 = FP32LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn3 = Attention(
            dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            qkv_mlp_ratio=qkv_mlp_ratio,
            cross_attention_dim=dim,  # For image conditioning
            rotary_emb=True,
            rotary_emb_dim=attention_head_dim,
            rotary_emb_axes_dims=axes_dims_rope,
        ) if True else None  # Enable image cross-attention

        self.norm4 = FP32LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(
            dim,
            dim_out=dim,
            activation_fn="gelu-approximate",
            bias=True,
            dropout=0.0,
            hidden_dropout=0.0,
            final_dropout=0.0,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the transformer block."""
        
        # Self-attention
        norm_hidden_states = self.norm1(hidden_states)
        attn_output = self.attn(norm_hidden_states, temb=temb)
        hidden_states = hidden_states + attn_output

        # Text cross-attention
        norm_hidden_states = self.norm2(hidden_states)
        attn_output = self.attn2(norm_hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb)
        hidden_states = hidden_states + attn_output

        # Image cross-attention (if provided)
        if encoder_hidden_states_image is not None and self.attn3 is not None:
            norm_hidden_states = self.norm3(hidden_states)
            attn_output = self.attn3(norm_hidden_states, encoder_hidden_states=encoder_hidden_states_image, temb=temb)
            hidden_states = hidden_states + attn_output

        # Feed-forward
        norm_hidden_states = self.norm4(hidden_states)
        ff_output = self.ff(norm_hidden_states, temb=temb)
        hidden_states = hidden_states + ff_output

        return hidden_states


class FluxKontextSingleTransformerBlock(nn.Module):
    """Single transformer block for additional processing."""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qkv_mlp_ratio: float = 4.0,
        axes_dims_rope: list[int] = [16, 56, 56],
    ):
        super().__init__()

        self.norm1 = FP32LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            qkv_mlp_ratio=qkv_mlp_ratio,
            rotary_emb=True,
            rotary_emb_dim=attention_head_dim,
            rotary_emb_axes_dims=axes_dims_rope,
        )

        self.norm2 = FP32LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(
            dim,
            dim_out=dim,
            activation_fn="gelu-approximate",
            bias=True,
            dropout=0.0,
            hidden_dropout=0.0,
            final_dropout=0.0,
        )

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        """Forward pass of the single transformer block."""
        
        # Self-attention
        norm_hidden_states = self.norm1(hidden_states)
        attn_output = self.attn(norm_hidden_states, temb=temb)
        hidden_states = hidden_states + attn_output

        # Feed-forward
        norm_hidden_states = self.norm2(hidden_states)
        ff_output = self.ff(norm_hidden_states, temb=temb)
        hidden_states = hidden_states + ff_output

        return hidden_states