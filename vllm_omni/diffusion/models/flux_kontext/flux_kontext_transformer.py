# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright 2025 Black Forest Labs and The HuggingFace Team. All rights reserved.
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


import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import (
    AdaLayerNormContinuous,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import QKVParallelLinear
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.attention.layer import Attention

logger = init_logger(__name__)


def get_1d_rotary_pos_embed(
    dim: int,
    pos: torch.Tensor,
    theta: float = 10000.0,
    use_real: bool = True,
    repeat_interleave_real: bool = True,
    freqs_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert dim % 2 == 0

    theta = theta
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device) / dim))
    freqs = torch.outer(pos, freqs)

    if use_real and repeat_interleave_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1)
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1)
        return freqs_cos.to(dtype=freqs_dtype), freqs_sin.to(dtype=freqs_dtype)
    else:
        freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
        freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
        return freqs_cos.to(dtype=freqs_dtype), freqs_sin.to(dtype=freqs_dtype)


class FluxPosEmbed(nn.Module):
    def __init__(self, theta: int, axes_dim: list[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n_axes = ids.shape[-1]
        emb_list = []
        for i in range(n_axes):
            cos_out, sin_out = get_1d_rotary_pos_embed(
                self.axes_dim[i],
                ids[..., i],
                self.theta,
                use_real=True,
                repeat_interleave_real=True,
                freqs_dtype=ids.dtype,
            )
            emb_list.append(cos_out)
            emb_list.append(sin_out)

        freqs_cos = torch.cat(emb_list[::2], dim=-1).to(ids.device)
        freqs_sin = torch.cat(emb_list[1::2], dim=-1).to(ids.device)

        return freqs_cos, freqs_sin


class RMSNorm(nn.Module):
    """RMSNorm (Root Mean Square Layer Normalization) following official FLUX implementation."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class AdaLayerNormZero(nn.Module):
    """AdaLayerNormZero matching FLUX.1-Kontext weight structure."""

    def __init__(self, embedding_dim: int, num_channels: int = None, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.act = nn.SiLU()
        self.lin = nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)
        self.num_channels = num_channels

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> tuple:
        emb = self.act(emb)
        emb = self.lin(emb)
        emb = emb.chunk(6, dim=-1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb

        x_dtype = x.dtype
        x = x.float()
        x = x * (1 + scale_msa[:, None]) + shift_msa[:, None]
        x = x.to(dtype=x_dtype)

        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZeroSingle(nn.Module):
    """AdaLayerNormZeroSingle matching FLUX.1-Kontext weight structure."""

    def __init__(self, embedding_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.act = nn.SiLU()
        self.lin = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> tuple:
        emb = self.act(emb)
        emb = self.lin(emb)
        emb = emb.chunk(3, dim=-1)
        shift_msa, scale_msa, gate_msa = emb

        x_dtype = x.dtype
        x = x.float()
        x = x * (1 + scale_msa[:, None]) + shift_msa[:, None]
        x = x.to(dtype=x_dtype)

        return x, gate_msa


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000, time_factor: float = 1000.0) -> torch.Tensor:
    """Create sinusoidal timestep embeddings following official FLUX implementation."""
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(
        -torch.log(torch.tensor(max_period, dtype=t.dtype, device=t.device))
        * torch.arange(start=0, end=half, dtype=t.dtype, device=t.device)
        / half
    )

    args = t[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

    return embedding


class FluxKontextAttention(nn.Module):
    """Attention module following FLUX.1-Kontext weight structure."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        qkv_bias: bool = False,
        added_kv_proj_dim: int | None = None,
        added_proj_bias: bool = True,
        out_bias: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.inner_dim = head_dim * num_heads
        self.query_dim = dim
        self.num_heads = num_heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.out_dim = dim

        self.norm_q = RMSNorm(head_dim, eps=eps)
        self.norm_k = RMSNorm(head_dim, eps=eps)

        self.qkv = QKVParallelLinear(
            hidden_size=dim,
            head_size=head_dim,
            total_num_heads=num_heads,
            bias=qkv_bias,
        )

        self.proj = nn.Linear(self.inner_dim, dim, bias=out_bias)

        if added_kv_proj_dim is not None:
            self.norm_added_q = RMSNorm(head_dim, eps=eps)
            self.norm_added_k = RMSNorm(head_dim, eps=eps)
            self.add_q_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.add_k_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.to_add_out = nn.Linear(self.inner_dim, dim, bias=out_bias)
        else:
            self.norm_added_q = None
            self.norm_added_k = None
            self.add_q_proj = None
            self.add_k_proj = None
            self.add_v_proj = None
            self.to_add_out = None

        self.attn = Attention(
            num_heads=num_heads,
            head_size=head_dim,
            softmax_scale=1.0 / (head_dim**0.5),
            causal=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        qkv, _ = self.qkv(hidden_states)
        query, key, value = qkv.chunk(3, dim=-1)

        query = query.unflatten(-1, (self.num_heads, -1))
        key = key.unflatten(-1, (self.num_heads, -1))
        value = value.unflatten(-1, (self.num_heads, -1))

        query = self.norm_q(query)
        key = self.norm_k(key)

        encoder_query = encoder_key = encoder_value = None
        if encoder_hidden_states is not None and self.added_kv_proj_dim is not None:
            encoder_query = self.add_q_proj(encoder_hidden_states)
            encoder_key = self.add_k_proj(encoder_hidden_states)
            encoder_value = self.add_v_proj(encoder_hidden_states)

            encoder_query = encoder_query.unflatten(-1, (self.num_heads, -1))
            encoder_key = encoder_key.unflatten(-1, (self.num_heads, -1))
            encoder_value = encoder_value.unflatten(-1, (self.num_heads, -1))

            encoder_query = self.norm_added_q(encoder_query)
            encoder_key = self.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=2)
            key = torch.cat([encoder_key, key], dim=2)
            value = torch.cat([encoder_value, value], dim=2)

        hidden_states = self.attn(query, key, value, attention_mask)
        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)

        if encoder_hidden_states is not None:
            context_len = encoder_hidden_states.shape[1]
            encoder_hidden_states_out, hidden_states = hidden_states.split_with_sizes(
                [context_len, hidden_states.shape[1] - context_len],
                dim=1,
            )
            encoder_hidden_states_out = self.to_add_out(encoder_hidden_states_out)
            hidden_states = self.proj(hidden_states)
            return hidden_states, encoder_hidden_states_out
        else:
            hidden_states = self.proj(hidden_states)
            return hidden_states


class DoubleStreamBlock(nn.Module):
    """Double stream block using AdaLayerNormZero (FLUX.1-Kontext style)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
    ):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        self.img_mod = AdaLayerNormZero(hidden_size)
        self.img_attn = FluxKontextAttention(
            dim=hidden_size,
            num_heads=num_heads,
            head_dim=hidden_size // num_heads,
            qkv_bias=qkv_bias,
        )

        self.img_mlp = nn.ModuleList(
            [
                nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
                nn.GELU(approximate="tanh"),
                nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
            ]
        )

        self.txt_mod = AdaLayerNormZero(hidden_size)
        self.txt_attn = self.img_attn

        self.txt_mlp = nn.ModuleList(
            [
                nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
                nn.GELU(approximate="tanh"),
                nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
            ]
        )

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        freqs_cos: torch.Tensor | None = None,
        freqs_sin: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        (img_normed, img_gate_msa, img_shift_mlp, img_scale_mlp, img_gate_mlp) = self.img_mod(img, emb=vec)
        (txt_normed, txt_gate_msa, txt_shift_mlp, txt_scale_mlp, txt_gate_mlp) = self.txt_mod(txt, emb=vec)

        img_qkv, _ = self.img_attn.qkv(img_normed)
        txt_qkv, _ = self.img_attn.qkv(txt_normed)

        img_q, img_k, img_v = img_qkv.chunk(3, dim=-1)
        txt_q, txt_k, txt_v = txt_qkv.chunk(3, dim=-1)

        img_q = img_q.unflatten(-1, (self.num_heads, -1))
        img_k = img_k.unflatten(-1, (self.num_heads, -1))
        img_v = img_v.unflatten(-1, (self.num_heads, -1))
        txt_q = txt_q.unflatten(-1, (self.num_heads, -1))
        txt_k = txt_k.unflatten(-1, (self.num_heads, -1))
        txt_v = txt_v.unflatten(-1, (self.num_heads, -1))

        img_q = self.img_attn.norm_q(img_q)
        img_k = self.img_attn.norm_k(img_k)
        txt_q = self.img_attn.norm_q(txt_q)
        txt_k = self.img_attn.norm_k(txt_k)

        if freqs_cos is not None and freqs_sin is not None:
            txt_seq_len = txt_q.shape[1]
            img_seq_len = img_q.shape[1]

            txt_freqs_cos = freqs_cos[:txt_seq_len]
            txt_freqs_sin = freqs_sin[:txt_seq_len]
            img_freqs_cos = freqs_cos[txt_seq_len : txt_seq_len + img_seq_len]
            img_freqs_sin = freqs_sin[txt_seq_len : txt_seq_len + img_seq_len]

            txt_q = self.apply_rope(txt_q, txt_freqs_cos, txt_freqs_sin)
            txt_k = self.apply_rope(txt_k, txt_freqs_cos, txt_freqs_sin)
            img_q = self.apply_rope(img_q, img_freqs_cos, img_freqs_sin)
            img_k = self.apply_rope(img_k, img_freqs_cos, img_freqs_sin)

        q = torch.cat([txt_q, img_q], dim=1)
        k = torch.cat([txt_k, img_k], dim=1)
        v = torch.cat([txt_v, img_v], dim=1)

        v = v.to(q.dtype)

        attn_output = self.img_attn.attn(q, k, v)
        txt_attn, img_attn = attn_output[:, : txt.shape[1], :, :], attn_output[:, txt.shape[1] :, :, :]

        txt_attn = txt_attn.flatten(2, 3)
        img_attn = img_attn.flatten(2, 3)

        txt_attn = self.img_attn.proj(txt_attn)
        img_attn = self.img_attn.proj(img_attn)

        txt = txt + txt_gate_msa.unsqueeze(1) * txt_attn
        img = img + img_gate_msa.unsqueeze(1) * img_attn

        img_normed = img_normed * (1 + img_scale_mlp[:, None]) + img_shift_mlp[:, None]
        img = img + img_gate_mlp.unsqueeze(1) * self.img_mlp[2](self.img_mlp[1](self.img_mlp[0](img_normed)))

        txt_normed = txt_normed * (1 + txt_scale_mlp[:, None]) + txt_shift_mlp[:, None]
        txt = txt + txt_gate_mlp.unsqueeze(1) * self.txt_mlp[2](self.txt_mlp[1](self.txt_mlp[0](txt_normed)))

        return img, txt

    def apply_rope(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
        batch, seq_len, num_heads, head_dim = x.shape
        half_head_dim = head_dim // 2
        orig_dtype = x.dtype

        x_float = x.float()
        x_rotated = x_float.reshape(batch, seq_len, num_heads, half_head_dim, 2)

        if freqs_cos.dim() == 2:
            freqs_cos = freqs_cos.unsqueeze(0)
            freqs_sin = freqs_sin.unsqueeze(0)

        freqs_cos = freqs_cos[:, :seq_len, :half_head_dim]
        freqs_sin = freqs_sin[:, :seq_len, :half_head_dim]

        freqs_cos = freqs_cos.view(1, seq_len, 1, half_head_dim).expand(batch, -1, -1, -1)
        freqs_sin = freqs_sin.view(1, seq_len, 1, half_head_dim).expand(batch, -1, -1, -1)

        x_rotated_part1 = freqs_cos * x_rotated[..., 0] - freqs_sin * x_rotated[..., 1]
        x_rotated_part2 = freqs_sin * x_rotated[..., 0] + freqs_cos * x_rotated[..., 1]

        x_new = torch.cat([x_rotated_part1, x_rotated_part2], dim=-1)

        return x_new.reshape(batch, seq_len, num_heads, head_dim).to(dtype=orig_dtype)


class SingleStreamBlock(nn.Module):
    """Single stream block following FLUX.1-Kontext original format.

    This implementation matches the Black Forest Labs original weight structure
    where attention q/k/v projections are merged into linear1.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()

        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.mlp_ratio = mlp_ratio
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)

        self.norm = AdaLayerNormZeroSingle(hidden_size)

        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim, bias=True)

        self.norm_q = RMSNorm(self.head_dim, eps=1e-6)
        self.norm_k = RMSNorm(self.head_dim, eps=1e-6)

        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)

        self.act_mlp = nn.GELU(approximate="tanh")

        self.proj_out = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        vec: torch.Tensor,
        freqs_cos: torch.Tensor | None = None,
        freqs_sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = x
        norm_x, gate = self.norm(x, emb=vec)

        qkv, mlp = torch.split(self.linear1(norm_x), [self.hidden_dim * 3, self.mlp_hidden_dim], dim=-1)

        q = qkv[:, :, : self.hidden_dim]
        k = qkv[:, :, self.hidden_dim : self.hidden_dim * 2]
        v = qkv[:, :, self.hidden_dim * 2 : self.hidden_dim * 3]

        q = q.unflatten(-1, (self.num_heads, -1))
        k = k.unflatten(-1, (self.num_heads, -1))
        v = v.unflatten(-1, (self.num_heads, -1))

        q = self.norm_q(q)
        k = self.norm_k(k)

        if freqs_cos is not None and freqs_sin is not None:
            q = self.apply_rope(q, freqs_cos, freqs_sin)
            k = self.apply_rope(k, freqs_cos, freqs_sin)

        attn_output = self.attn(q, k, v)
        attn_output = attn_output.flatten(2, 3)
        attn_output = self.proj(attn_output)

        mlp_hidden = self.act_mlp(mlp)

        output = torch.cat([attn_output, mlp_hidden], dim=-1)
        output = self.proj_out(output)

        output = gate.unsqueeze(1) * output
        output = residual + output

        return output

    def apply_rope(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
        batch, seq_len, num_heads, head_dim = x.shape
        half_head_dim = head_dim // 2
        orig_dtype = x.dtype

        x_float = x.float()
        x_rotated = x_float.reshape(batch, seq_len, num_heads, half_head_dim, 2)

        if freqs_cos.dim() == 2:
            freqs_cos = freqs_cos.unsqueeze(0)
            freqs_sin = freqs_sin.unsqueeze(0)

        freqs_cos = freqs_cos[:, :seq_len, :half_head_dim]
        freqs_sin = freqs_sin[:, :seq_len, :half_head_dim]

        freqs_cos = freqs_cos.view(1, seq_len, 1, half_head_dim).expand(batch, -1, -1, -1)
        freqs_sin = freqs_sin.view(1, seq_len, 1, half_head_dim).expand(batch, -1, -1, -1)

        x_rotated_part1 = freqs_cos * x_rotated[..., 0] - freqs_sin * x_rotated[..., 1]
        x_rotated_part2 = freqs_sin * x_rotated[..., 0] + freqs_cos * x_rotated[..., 1]

        x_new = torch.cat([x_rotated_part1, x_rotated_part2], dim=-1)

        return x_new.reshape(batch, seq_len, num_heads, head_dim).to(dtype=orig_dtype)

    def attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Standard attention computation."""
        head_dim = q.shape[-1]
        attn = torch.einsum("bhid,bhjd->bhij", q, k) * (head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        return out


class FluxKontextTransformer2DModel(nn.Module):
    """
    FLUX.1-Kontext Transformer model following official FLUX implementation patterns.

    This implementation adapts the official FLUX.1-Kontext model to work with
    vLLM-Omni's distributed inference capabilities while maintaining the
    core architecture and modulation mechanisms from the official implementation.
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
        hidden_size: int = 3072,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = True,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.num_single_layers = num_single_layers
        self.attention_head_dim = attention_head_dim
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.joint_attention_dim = joint_attention_dim
        self.mlp_ratio = mlp_ratio
        self.guidance_embeds = guidance_embeds

        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by num_attention_heads {num_attention_heads}"
            )

        self.x_embedder = nn.Linear(in_channels, hidden_size, bias=True)
        self.context_embedder = nn.Linear(joint_attention_dim, hidden_size, bias=True)

        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=[16, 56, 56])

        self.time_text_embed = nn.ModuleDict(
            {
                "timestep_embedder": nn.ModuleDict(
                    {
                        "linear_1": nn.Linear(256, hidden_size, bias=True),
                        "linear_2": nn.Linear(hidden_size, hidden_size, bias=True),
                    }
                ),
                "text_embedder": nn.ModuleDict(
                    {
                        "linear_1": nn.Linear(pooled_projection_dim, hidden_size, bias=True),
                        "linear_2": nn.Linear(hidden_size, hidden_size, bias=True),
                    }
                ),
                "guidance_embedder": nn.ModuleDict(
                    {
                        "linear_1": nn.Linear(256, hidden_size, bias=True),
                        "linear_2": nn.Linear(hidden_size, hidden_size, bias=True),
                    }
                )
                if guidance_embeds
                else None,
            }
        )

        self.transformer_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    hidden_size=hidden_size,
                    num_heads=num_attention_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                )
                for _ in range(num_layers)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    hidden_size=hidden_size,
                    num_heads=num_attention_heads,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(num_single_layers)
            ]
        )

        self.final_layer = None

        self.norm_out = AdaLayerNormContinuous(hidden_size, hidden_size, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(
            hidden_size,
            (patch_size if patch_size is not None else 1)
            * (patch_size if patch_size is not None else 1)
            * (out_channels if out_channels is not None else in_channels),
            bias=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        txt_ids: torch.Tensor | None = None,
        img_ids: torch.Tensor | None = None,
        image_pooled_projections: torch.Tensor | None = None,
        guidance: torch.Tensor | None = None,
        return_dict: bool = True,
    ) -> torch.Tensor:
        timestep_emb = self.time_text_embed["timestep_embedder"]["linear_2"](
            F.silu(self.time_text_embed["timestep_embedder"]["linear_1"](timestep_embedding(timestep, 256)))
        )

        vec = timestep_emb + self.time_text_embed["text_embedder"]["linear_2"](
            F.silu(self.time_text_embed["text_embedder"]["linear_1"](pooled_projections))
        )

        if guidance is not None and self.guidance_embeds:
            if guidance.dim() == 0:
                guidance = guidance.unsqueeze(0)
            guidance_emb = self.time_text_embed["guidance_embedder"]["linear_2"](
                F.silu(self.time_text_embed["guidance_embedder"]["linear_1"](timestep_embedding(guidance, 256)))
            )
            vec = vec + guidance_emb

        if image_pooled_projections is not None:
            image_vec = self.time_text_embed["text_embedder"]["linear_2"](
                F.silu(self.time_text_embed["text_embedder"]["linear_1"](image_pooled_projections))
            )
            vec = vec + image_vec

        batch_size, seq_len, in_channels = hidden_states.shape

        img = self.x_embedder(hidden_states)
        txt = self.context_embedder(encoder_hidden_states)

        ids = torch.cat((txt_ids, img_ids), dim=0)
        freqs_cos, freqs_sin = self.pos_embed(ids)

        for block in self.transformer_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, freqs_cos=freqs_cos, freqs_sin=freqs_sin)

        img = torch.cat((txt, img), dim=1)

        for block in self.single_transformer_blocks:
            img = block(img, vec=vec, freqs_cos=freqs_cos, freqs_sin=freqs_sin)

        img = img[:, txt.shape[1] :, ...]

        img = self.norm_out(img, vec)
        img = self.proj_out(img)

        if return_dict:
            return Transformer2DModelOutput(sample=img)
        else:
            return img

    def load_weights(self, weights):
        weight_name_mapping = {
            "double_blocks": "double_stream_blocks",
            "single_blocks": "single_stream_blocks",
            "txt_mlp": "txt_ff",
            "img_mlp": "img_ff",
            "txt_attn": "txt_attn",
            "img_attn": "img_attn",
            "txt_mod": "txt_mod",
            "img_mod": "img_mod",
            "modulation.lin": "norm.lin",
            "norm.key_norm": "norm_k",
            "norm.query_norm": "norm_q",
        }

        def map_weight_name(name: str) -> str:
            for old_name, new_name in weight_name_mapping.items():
                if old_name in name:
                    name = name.replace(old_name, new_name, 1)
            return name

        params_dict = dict(self.named_parameters())

        for name, buffer in self.named_buffers():
            if name.endswith(".beta") or name.endswith(".eps"):
                params_dict[name] = buffer

        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            original_name = name

            name = map_weight_name(name)

            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(original_name)
        return loaded_params
