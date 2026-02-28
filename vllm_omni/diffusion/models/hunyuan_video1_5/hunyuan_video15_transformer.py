# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable

import torch
import torch.nn as nn
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear, QKVParallelLinear, RowParallelLinear
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.layers.rope import RotaryEmbedding

logger = init_logger(__name__)


class HunyuanVideo15PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int | tuple[int, int, int] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()

        patch_size = (patch_size, patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        return hidden_states


class HunyuanVideo15AdaNorm(nn.Module):
    def __init__(self, in_features: int, out_features: int | None = None) -> None:
        super().__init__()

        out_features = out_features or 2 * in_features
        self.linear = nn.Linear(in_features, out_features)
        self.nonlinearity = nn.SiLU()

    def forward(self, temb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        temb = self.linear(self.nonlinearity(temb))
        gate_msa, gate_mlp = temb.chunk(2, dim=1)
        gate_msa, gate_mlp = gate_msa.unsqueeze(1), gate_mlp.unsqueeze(1)
        return gate_msa, gate_mlp


class HunyuanVideo15TimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, use_meanflow: bool = False):
        super().__init__()

        from diffusers.models.embeddings import TimestepEmbedding, Timesteps

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

        self.use_meanflow = use_meanflow
        self.time_proj_r = None
        self.timestep_embedder_r = None
        if use_meanflow:
            self.time_proj_r = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
            self.timestep_embedder_r = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(
        self,
        timestep: torch.Tensor,
        timestep_r: torch.Tensor | None = None,
    ) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=timestep.dtype))

        if timestep_r is not None:
            timesteps_proj_r = self.time_proj_r(timestep_r)
            timesteps_emb_r = self.timestep_embedder_r(timesteps_proj_r.to(dtype=timestep.dtype))
            timesteps_emb = timesteps_emb + timesteps_emb_r

        return timesteps_emb


class HunyuanVideo15Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        added_kv_proj_dim: int | None = None,
        added_proj_bias: bool | None = True,
        out_bias: bool = True,
        eps: float = 1e-5,
        out_dim: int = None,
        context_pre_only: bool | None = None,
    ):
        super().__init__()

        self.head_dim = dim_head
        self.query_dim = query_dim
        self.use_bias = bias
        self.dropout = dropout
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.context_pre_only = context_pre_only
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.inner_dim = self.out_dim
        self.added_kv_proj_dim = added_kv_proj_dim

        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)

        self.to_qkv = QKVParallelLinear(
            hidden_size=query_dim,
            head_size=self.head_dim,
            total_num_heads=self.heads,
            bias=bias,
            return_bias=False,
        )

        self.heads = self.to_qkv.num_heads

        self.total_num_heads = (added_kv_proj_dim or query_dim) // dim_head

        self.to_out = nn.ModuleList(
            [
                RowParallelLinear(
                    self.inner_dim,
                    self.out_dim,
                    bias=out_bias,
                    input_is_parallel=True,
                    return_bias=False,
                ),
                nn.Dropout(dropout),
            ]
        )

        if added_kv_proj_dim is not None:
            self.norm_added_q = RMSNorm(dim_head, eps=eps)
            self.norm_added_k = RMSNorm(dim_head, eps=eps)

            self.add_kv_proj = nn.Linear(
                added_kv_proj_dim,
                3 * added_kv_proj_dim,
                bias=added_proj_bias,
            )

            self.to_add_out = RowParallelLinear(
                self.inner_dim,
                query_dim,
                bias=out_bias,
                input_is_parallel=True,
                return_bias=False,
            )

        self.rope = RotaryEmbedding(is_neox_style=False)
        self.attn = Attention(
            num_heads=self.to_qkv.num_heads,
            head_size=self.head_dim,
            softmax_scale=1.0 / (self.head_dim**0.5),
            causal=False,
            num_kv_heads=self.to_qkv.num_kv_heads,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        qkv_output = self.to_qkv(hidden_states)
        if isinstance(qkv_output, tuple):
            qkv = qkv_output[0]
        else:
            qkv = qkv_output
        q_size = self.to_qkv.num_heads * self.head_dim
        kv_size = self.to_qkv.num_kv_heads * self.head_dim
        query, key, value = qkv.split([q_size, kv_size, kv_size], dim=-1)

        query = query.unflatten(-1, (self.to_qkv.num_heads, -1))
        key = key.unflatten(-1, (self.to_qkv.num_kv_heads, -1))
        value = value.unflatten(-1, (self.to_qkv.num_kv_heads, -1))

        query = self.norm_q(query)
        key = self.norm_k(key)

        # Apply RoPE to original query and key if present, excluding encoder part
        if image_rotary_emb is not None:
            cos, sin = image_rotary_emb
            cos = cos.to(query.dtype)
            sin = sin.to(query.dtype)
            query = self.rope(query, cos, sin)
            key = self.rope(key, cos, sin)

        if self.added_kv_proj_dim is not None and encoder_hidden_states is not None:
            encoder_qkv = self.add_kv_proj(encoder_hidden_states)
            split_size = self.added_kv_proj_dim
            encoder_query, encoder_key, encoder_value = encoder_qkv.split([split_size, split_size, split_size], dim=-1)

            encoder_query = encoder_query.unflatten(-1, (self.total_num_heads, -1))
            encoder_key = encoder_key.unflatten(-1, (self.total_num_heads, -1))
            encoder_value = encoder_value.unflatten(-1, (self.total_num_heads, -1))

            encoder_query = self.norm_added_q(encoder_query)
            encoder_key = self.norm_added_k(encoder_key)

            # Shard encoder to match TP-sharded video heads
            tp_rank = get_tensor_model_parallel_rank()
            encoder_query = encoder_query[:, :, tp_rank * self.heads : (tp_rank + 1) * self.heads, :]
            encoder_key = encoder_key[:, :, tp_rank * self.heads : (tp_rank + 1) * self.heads, :]
            encoder_value = encoder_value[:, :, tp_rank * self.heads : (tp_rank + 1) * self.heads, :]

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        attn_metadata = None
        if attention_mask is not None:
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            attn_metadata = AttentionMetadata(attn_mask=attention_mask)

        hidden_states = self.attn(
            query,
            key,
            value,
            attn_metadata,
        )

        batch_size, seq_len, num_heads, head_dim = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, seq_len, num_heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            hidden_states = self.to_out[0](hidden_states)
            hidden_states = self.to_out[1](hidden_states)
            encoder_hidden_states = self.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            hidden_states = self.to_out[0](hidden_states)
            hidden_states = self.to_out[1](hidden_states)
            return hidden_states, None


class HunyuanVideo15IndividualTokenRefinerBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_width_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        attention_bias: bool = True,
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)

        self.to_qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=attention_bias)

        self.heads = num_attention_heads
        self.inner_dim = self.heads * attention_head_dim
        self.head_dim = attention_head_dim

        self.norm_q = RMSNorm(self.head_dim, eps=1e-6)
        self.norm_k = RMSNorm(self.head_dim, eps=1e-6)

        self.attn = Attention(
            num_heads=self.heads,
            head_size=attention_head_dim,
            softmax_scale=1.0 / (attention_head_dim**0.5),
            causal=False,
            num_kv_heads=self.heads,
        )

        self.ff = nn.Linear(hidden_size, int(hidden_size * mlp_width_ratio), bias=attention_bias)

        self.ff_2 = nn.Linear(int(hidden_size * mlp_width_ratio), hidden_size, bias=attention_bias)

        self.to_out = nn.Linear(hidden_size, hidden_size, bias=attention_bias)

        self.gate_linear = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, int(2 * hidden_size), bias=attention_bias),
        )
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(mlp_drop_rate)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        norm_hidden_states = self.norm1(hidden_states)

        qkv = self.to_qkv(norm_hidden_states)
        q_size = self.heads * self.head_dim
        kv_size = self.heads * self.head_dim
        query, key, value = qkv.split([q_size, kv_size, kv_size], dim=-1)

        query = query.unflatten(-1, (self.heads, -1))
        key = key.unflatten(-1, (self.heads, -1))
        value = value.unflatten(-1, (self.heads, -1))

        query = self.norm_q(query)
        key = self.norm_k(key)

        attn_output = self.attn(query, key, value)
        batch_size, seq_len, num_heads, head_dim = attn_output.shape
        attn_output = attn_output.view(batch_size, seq_len, num_heads * head_dim)

        attn_output = self.to_out(attn_output)

        gate_msa, gate_mlp = self.get_gates(temb)
        hidden_states = hidden_states + attn_output * gate_msa.unsqueeze(1)

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + gate_mlp.unsqueeze(1))
        ff_intermediate = self.ff(norm_hidden_states)
        ff_output = self.ff_2(self.activation(ff_intermediate))
        hidden_states = hidden_states + ff_output

        return hidden_states

    def get_gates(self, temb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if hasattr(self, "gate_linear") and self.gate_linear is not None:
            gate_output = self.gate_linear(temb)
            gate_msa, gate_mlp = gate_output.chunk(2, dim=-1)
            # Do NOT truncate gate_msa/gate_mlp in TP mode - keep full dimension
            # to match hidden_states and attn_output dimensions
            return gate_msa, gate_mlp
        else:
            result = self.norm_out(temb)
            if len(result) == 5:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = result
            else:
                gate_msa, gate_mlp = result
            # Do NOT truncate gate_msa/gate_mlp in TP mode - keep full dimension
            # to match hidden_states and attn_output dimensions
            return gate_msa, gate_mlp


class HunyuanVideo15IndividualTokenRefiner(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        num_layers: int,
        mlp_width_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        attention_bias: bool = True,
    ) -> None:
        super().__init__()

        self.refiner_blocks = nn.ModuleList(
            [
                HunyuanVideo15IndividualTokenRefinerBlock(
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_drop_rate=mlp_drop_rate,
                    attention_bias=attention_bias,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self_attn_mask = None
        if attention_mask is not None:
            batch_size = attention_mask.shape[0]
            seq_len = attention_mask.shape[1]
            attention_mask = attention_mask.to(hidden_states.device).bool()
            self_attn_mask_1 = attention_mask.view(batch_size, 1, 1, seq_len).repeat(1, 1, seq_len, 1)
            self_attn_mask_2 = self_attn_mask_1.transpose(2, 3)
            self_attn_mask = (self_attn_mask_1 & self_attn_mask_2).bool()

        for block in self.refiner_blocks:
            hidden_states = block(hidden_states, temb, self_attn_mask)

        return hidden_states


class HunyuanVideo15TokenRefiner(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_layers: int,
        mlp_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        attention_bias: bool = True,
    ) -> None:
        super().__init__()

        from diffusers.models.embeddings import CombinedTimestepTextProjEmbeddings

        hidden_size = num_attention_heads * attention_head_dim

        tp_size = get_tensor_model_parallel_world_size()
        tp_hidden_size = hidden_size // tp_size

        self.hidden_size = hidden_size
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=hidden_size, pooled_projection_dim=in_channels
        )

        self.in_channels = in_channels

        self.proj_in = nn.Linear(in_channels, hidden_size, bias=True)
        self.proj_out = nn.Linear(hidden_size, hidden_size, bias=True)

        self.token_refiner = HunyuanVideo15IndividualTokenRefiner(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            num_layers=num_layers,
            mlp_width_ratio=mlp_ratio,
            mlp_drop_rate=mlp_drop_rate,
            attention_bias=attention_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        attention_mask: torch.LongTensor | None = None,
    ) -> torch.Tensor:

        if attention_mask is None:
            pooled_projections = hidden_states.mean(dim=1)
        else:
            original_dtype = hidden_states.dtype
            mask_float = attention_mask.float().unsqueeze(-1)
            pooled_projections = (hidden_states * mask_float).sum(dim=1) / mask_float.sum(dim=1)
            pooled_projections = pooled_projections.to(original_dtype)

        temb = self.time_text_embed(timestep, pooled_projections)

        hidden_states = self.proj_in(hidden_states)

        hidden_states = self.token_refiner(hidden_states, temb, attention_mask)

        hidden_states = self.proj_out(hidden_states)

        return hidden_states


class HunyuanVideo15RotaryPosEmbed(nn.Module):
    def __init__(self, patch_size: int, patch_size_t: int, rope_dim: list[int], theta: float = 256.0) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.rope_dim = rope_dim
        self.theta = theta

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        from diffusers.models.embeddings import get_1d_rotary_pos_embed

        _, _, num_frames, height, width = hidden_states.shape
        rope_sizes = [num_frames // self.patch_size_t, height // self.patch_size, width // self.patch_size]

        axes_grids = []
        for i in range(len(rope_sizes)):
            grid = torch.arange(0, rope_sizes[i], device=hidden_states.device, dtype=torch.float32)
            axes_grids.append(grid)
        grid = torch.meshgrid(*axes_grids, indexing="ij")
        grid = torch.stack(grid, dim=0)

        freqs = []
        for i in range(3):
            freq = get_1d_rotary_pos_embed(self.rope_dim[i], grid[i].reshape(-1), self.theta, use_real=True)
            freqs.append(freq)

        freqs_cos = torch.cat([f[0] for f in freqs], dim=1)
        freqs_sin = torch.cat([f[1] for f in freqs], dim=1)

        # Convert to format compatible with generic RoPE function: from (seqlen, rotary_dim) to (seqlen, rotary_dim/2)
        # Remove repeat_interleave duplication by taking even indices
        freqs_cos = freqs_cos[:, ::2]
        freqs_sin = freqs_sin[:, ::2]

        return freqs_cos, freqs_sin


class HunyuanVideo15ByT5TextProjection(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, out_features: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.linear_1 = nn.Linear(in_features, hidden_size, bias=True)
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class HunyuanVideo15ImageProjection(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int):
        super().__init__()
        self.norm_in = nn.LayerNorm(in_channels)
        self.linear_1 = nn.Linear(in_channels, in_channels, bias=True)
        self.act_fn = nn.GELU()
        self.linear_2 = nn.Linear(in_channels, hidden_size, bias=True)
        self.norm_out = nn.LayerNorm(hidden_size)

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm_in(image_embeds)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        hidden_states = self.norm_out(hidden_states)
        return hidden_states


class HunyuanVideo15TransformerBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()

        from diffusers.models.normalization import AdaLayerNormZero

        hidden_size = num_attention_heads * attention_head_dim

        self.norm1 = AdaLayerNormZero(hidden_size, norm_type="layer_norm")
        self.norm1_context = AdaLayerNormZero(hidden_size, norm_type="layer_norm")

        self.attn = HunyuanVideo15Attention(
            query_dim=hidden_size,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            added_kv_proj_dim=hidden_size,
            out_dim=hidden_size,
            bias=True,
            eps=1e-6,
        )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.ff = ColumnParallelLinear(
            hidden_size,
            int(hidden_size * mlp_ratio),
            bias=True,
            return_bias=False,
        )
        self.ff_2 = RowParallelLinear(
            int(hidden_size * mlp_ratio),
            hidden_size,
            bias=True,
            input_is_parallel=True,
            return_bias=False,
        )
        self.ff_context = nn.Linear(
            hidden_size,
            int(hidden_size * mlp_ratio),
            bias=True,
        )
        self.ff_context_2 = nn.Linear(
            int(hidden_size * mlp_ratio),
            hidden_size,
            bias=True,
        )

        self.norm2_context = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.activation = nn.GELU(approximate="tanh")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        freqs_cis: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )

        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=freqs_cis,
        )

        hidden_states = hidden_states + attn_output * gate_msa.unsqueeze(1)
        encoder_hidden_states = encoder_hidden_states + context_attn_output * c_gate_msa.unsqueeze(1)

        norm_hidden_states = self.norm2(hidden_states)
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)

        norm_hidden_states = norm_hidden_states * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (
            1 + c_scale_mlp.unsqueeze(1)
        ) + c_shift_mlp.unsqueeze(1)

        ff_output = self.ff_2(self.activation(self.ff(norm_hidden_states)))
        ff_context_output = self.ff_context_2(self.activation(self.ff_context(norm_encoder_hidden_states)))

        hidden_states = hidden_states + ff_output * gate_mlp.unsqueeze(1)
        encoder_hidden_states = encoder_hidden_states + ff_context_output * c_gate_mlp.unsqueeze(1)

        return hidden_states, encoder_hidden_states


class HunyuanVideo15Transformer3DModel(nn.Module):
    _layerwise_offload_blocks_attrs = ["transformer_blocks"]
    _repeated_blocks = ["HunyuanVideo15TransformerBlock"]

    def __init__(
        self,
        in_channels: int = 65,
        out_channels: int = 32,
        num_attention_heads: int = 16,
        attention_head_dim: int = 128,
        num_layers: int = 54,
        num_refiner_layers: int = 2,
        mlp_ratio: float = 4.0,
        patch_size: int = 1,
        patch_size_t: int = 1,
        qk_norm: str = "rms_norm",
        text_embed_dim: int = 4096,
        text_embed_2_dim: int = 1472,
        image_embed_dim: int = 1152,
        rope_theta: float = 256.0,
        rope_axes_dim: tuple[int, ...] = (16, 56, 56),
        target_size: int = 640,
        task_type: str = "i2v",
        use_meanflow: bool = False,
        od_config: OmniDiffusionConfig | None = None,
    ):
        super().__init__()

        self.config = type(
            "Config",
            (),
            {
                "in_channels": in_channels,
                "out_channels": out_channels,
                "num_attention_heads": num_attention_heads,
                "attention_head_dim": attention_head_dim,
                "num_layers": num_layers,
                "num_refiner_layers": num_refiner_layers,
                "mlp_ratio": mlp_ratio,
                "patch_size": patch_size,
                "patch_size_t": patch_size_t,
                "qk_norm": qk_norm,
                "text_embed_dim": text_embed_dim,
                "text_embed_2_dim": text_embed_2_dim,
                "image_embed_dim": image_embed_dim,
                "rope_theta": rope_theta,
                "rope_axes_dim": rope_axes_dim,
                "target_size": target_size,
                "task_type": task_type,
                "use_meanflow": use_meanflow,
            },
        )()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        self.x_embedder = HunyuanVideo15PatchEmbed((patch_size_t, patch_size, patch_size), in_channels, inner_dim)
        self.image_embedder = HunyuanVideo15ImageProjection(image_embed_dim, inner_dim)

        self.context_embedder = HunyuanVideo15TokenRefiner(
            text_embed_dim, num_attention_heads, attention_head_dim, num_layers=num_refiner_layers
        )
        self.context_embedder_2 = HunyuanVideo15ByT5TextProjection(text_embed_2_dim, 2048, inner_dim)

        self.time_embed = HunyuanVideo15TimeEmbedding(inner_dim, use_meanflow=use_meanflow)

        self.cond_type_embed = nn.Embedding(3, inner_dim)

        self.rope = HunyuanVideo15RotaryPosEmbed(patch_size, patch_size_t, list(rope_axes_dim), rope_theta)

        self.transformer_blocks = nn.ModuleList(
            [
                HunyuanVideo15TransformerBlock(num_attention_heads, attention_head_dim, mlp_ratio=mlp_ratio)
                for _ in range(num_layers)
            ]
        )

        from diffusers.models.normalization import AdaLayerNormContinuous

        self.norm_out = AdaLayerNormContinuous(inner_dim, inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(inner_dim, patch_size_t * patch_size * patch_size * out_channels)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        timestep_r: torch.LongTensor | None = None,
        encoder_hidden_states_2: torch.Tensor | None = None,
        encoder_attention_mask_2: torch.Tensor | None = None,
        image_embeds: torch.Tensor | None = None,
    ) -> Transformer2DModelOutput:
        batch_size, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size_t, self.config.patch_size, self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        image_rotary_emb = self.rope(hidden_states)

        temb = self.time_embed(timestep, timestep_r=timestep_r)

        hidden_states = self.x_embedder(hidden_states)

        encoder_hidden_states = self.context_embedder(encoder_hidden_states, timestep, encoder_attention_mask)

        encoder_hidden_states_cond_emb = self.cond_type_embed(
            torch.zeros_like(encoder_hidden_states[:, :, 0], dtype=torch.long, device=hidden_states.device)
        )
        encoder_hidden_states = encoder_hidden_states + encoder_hidden_states_cond_emb

        encoder_hidden_states_2 = self.context_embedder_2(encoder_hidden_states_2)

        encoder_hidden_states_2_cond_emb = self.cond_type_embed(
            torch.ones_like(encoder_hidden_states_2[:, :, 0], dtype=torch.long, device=hidden_states.device)
        )
        encoder_hidden_states_2 = encoder_hidden_states_2 + encoder_hidden_states_2_cond_emb

        encoder_hidden_states_3 = self.image_embedder(image_embeds)
        is_t2v = torch.all(image_embeds == 0, dim=tuple(range(1, image_embeds.ndim)))
        if is_t2v:
            encoder_hidden_states_3 = encoder_hidden_states_3 * 0.0
            encoder_attention_mask_3 = torch.zeros(
                (batch_size, encoder_hidden_states_3.shape[1]),
                dtype=encoder_attention_mask.dtype,
                device=encoder_attention_mask.device,
            )
        else:
            encoder_attention_mask_3 = torch.ones(
                (batch_size, encoder_hidden_states_3.shape[1]),
                dtype=encoder_attention_mask.dtype,
                device=encoder_attention_mask.device,
            )
        encoder_hidden_states_3_cond_emb = self.cond_type_embed(
            2
            * torch.ones_like(
                encoder_hidden_states_3[:, :, 0],
                dtype=torch.long,
                device=hidden_states.device,
            )
        )
        encoder_hidden_states_3 = encoder_hidden_states_3 + encoder_hidden_states_3_cond_emb

        encoder_attention_mask = encoder_attention_mask.bool()
        encoder_attention_mask_2 = encoder_attention_mask_2.bool()
        encoder_attention_mask_3 = encoder_attention_mask_3.bool()
        new_encoder_hidden_states = []
        new_encoder_attention_mask = []

        for text, text_mask, text_2, text_mask_2, image, image_mask in zip(
            encoder_hidden_states,
            encoder_attention_mask,
            encoder_hidden_states_2,
            encoder_attention_mask_2,
            encoder_hidden_states_3,
            encoder_attention_mask_3,
        ):
            new_encoder_hidden_states.append(
                torch.cat(
                    [
                        image[image_mask],
                        text_2[text_mask_2],
                        text[text_mask],
                        image[~image_mask],
                        torch.zeros_like(text_2[~text_mask_2]),
                        torch.zeros_like(text[~text_mask]),
                    ],
                    dim=0,
                )
            )

            new_encoder_attention_mask.append(
                torch.cat(
                    [
                        image_mask[image_mask],
                        text_mask_2[text_mask_2],
                        text_mask[text_mask],
                        image_mask[~image_mask],
                        text_mask_2[~text_mask_2],
                        text_mask[~text_mask],
                    ],
                    dim=0,
                )
            )

        encoder_hidden_states = torch.stack(new_encoder_hidden_states)
        encoder_attention_mask = torch.stack(new_encoder_attention_mask)

        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                temb,
                encoder_attention_mask,
                image_rotary_emb,
            )

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1, p_t, p_h, p_w
        )
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        return Transformer2DModelOutput(sample=hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            (".to_qkv", ".to_q", "q"),
            (".to_qkv", ".to_k", "k"),
            (".to_qkv", ".to_v", "v"),
            (".add_kv_proj", ".add_q_proj", "q"),
            (".add_kv_proj", ".add_k_proj", "k"),
            (".add_kv_proj", ".add_v_proj", "v"),
        ]

        params_dict = dict(self.named_parameters())

        for name, buffer in self.named_buffers():
            if name.endswith(".beta") or name.endswith(".eps"):
                params_dict[name] = buffer

        loaded_params: set[str] = set()

        # Temporary storage for refiner QKV fusion
        refiner_qkv: dict[str, dict[str, torch.Tensor]] = {}

        for name, loaded_weight in weights:
            original_name = name

            is_refiner = ".refiner_blocks." in name

            # Transform refiner weight names
            if is_refiner:
                # refiner: .attn.to_q/k/v. -> .to_qkv. (fuse)
                # refiner: .attn.to_out.0. -> .to_out.
                # refiner: .ff.net.0.proj. -> .ff.
                # refiner: .ff.net.2. -> .ff_2.
                if ".attn.to_q." in name:
                    name = name.replace(".attn.to_q.", ".to_qkv.")
                elif ".attn.to_k." in name:
                    name = name.replace(".attn.to_k.", ".to_qkv.")
                elif ".attn.to_v." in name:
                    name = name.replace(".attn.to_v.", ".to_qkv.")
                elif ".attn.to_out.0." in name:
                    name = name.replace(".attn.to_out.0.", ".to_out.")
                elif ".ff.net.0.proj." in name:
                    name = name.replace(".ff.net.0.proj.", ".ff.")
                elif ".ff.net.2." in name:
                    name = name.replace(".ff.net.2.", ".ff_2.")
            else:
                # Transformer blocks: ff and ff_context mapping
                # Diffusers: ff.net.0.proj -> vLLM: ff
                # Diffusers: ff.net.2 -> vLLM: ff_2
                if ".ff.net.0.proj." in name:
                    name = name.replace(".ff.net.0.proj.", ".ff.")
                elif ".ff.net.2." in name:
                    name = name.replace(".ff.net.2.", ".ff_2.")
                elif ".ff_context.net.0.proj." in name:
                    name = name.replace(".ff_context.net.0.proj.", ".ff_context.")
                elif ".ff_context.net.2." in name:
                    name = name.replace(".ff_context.net.2.", ".ff_context_2.")

            # Skip weights not in vLLM model
            if ".norm_out.linear." in name or "context_embedder_2.linear_3" in name:
                loaded_params.add(original_name)
                continue

            # Handle refiner QKV fusion
            if is_refiner and ".to_qkv." in name:
                if name not in refiner_qkv:
                    refiner_qkv[name] = {}
                if ".to_q." in name:
                    refiner_qkv[name]["q"] = loaded_weight
                elif ".to_k." in name:
                    refiner_qkv[name]["k"] = loaded_weight
                elif ".to_v." in name:
                    refiner_qkv[name]["v"] = loaded_weight

                if len(refiner_qkv[name]) == 3:
                    q = refiner_qkv[name]["q"]
                    k = refiner_qkv[name]["k"]
                    v = refiner_qkv[name]["v"]
                    fused = torch.cat([q, k, v], dim=0)
                    if name not in params_dict:
                        logger.warning(f"Skipping weight {name} - not found in model")
                        continue
                    param = params_dict[name]
                    param.data.copy_(fused)
                    loaded_params.add(name)
                    del refiner_qkv[name]
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                new_name = name.replace(weight_name, param_name)
                if new_name not in params_dict:
                    name = original_name
                    break

                param = params_dict[new_name]
                if hasattr(param, "weight_loader"):
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                else:
                    # add_kv_proj is nn.Linear, need manual fusion
                    base_name = new_name
                    if base_name not in refiner_qkv:
                        refiner_qkv[base_name] = {}
                    if shard_id == "q":
                        refiner_qkv[base_name]["q"] = loaded_weight
                    elif shard_id == "k":
                        refiner_qkv[base_name]["k"] = loaded_weight
                    elif shard_id == "v":
                        refiner_qkv[base_name]["v"] = loaded_weight

                    if len(refiner_qkv[base_name]) == 3:
                        q = refiner_qkv[base_name]["q"]
                        k = refiner_qkv[base_name]["k"]
                        v = refiner_qkv[base_name]["v"]
                        fused = torch.cat([q, k, v], dim=0)
                        param.data.copy_(fused)
                        loaded_params.add(base_name)
                        del refiner_qkv[base_name]
                    name = base_name
                break
            else:
                # No mapping needed - use original name directly
                if name not in params_dict:
                    logger.warning(f"Skipping weight {name}")
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

            loaded_params.add(name)

        return loaded_params
