# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch
import torch.nn.functional as F


class Flux2KVLayerCache:
    """Per-layer KV cache for reference image tokens."""

    def __init__(self):
        self.k_ref: torch.Tensor | None = None
        self.v_ref: torch.Tensor | None = None

    def store(self, k_ref: torch.Tensor, v_ref: torch.Tensor) -> None:
        self.k_ref = k_ref
        self.v_ref = v_ref

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.k_ref is None or self.v_ref is None:
            raise RuntimeError("KV cache has not been populated yet.")
        return self.k_ref, self.v_ref

    def clear(self) -> None:
        self.k_ref = None
        self.v_ref = None


class Flux2KVCache:
    """Container for all layers' reference-token KV caches."""

    def __init__(self, num_double_layers: int, num_single_layers: int):
        self.double_block_caches = [Flux2KVLayerCache() for _ in range(num_double_layers)]
        self.single_block_caches = [Flux2KVLayerCache() for _ in range(num_single_layers)]
        self.num_ref_tokens: int = 0

    def get_double(self, layer_idx: int) -> Flux2KVLayerCache:
        return self.double_block_caches[layer_idx]

    def get_single(self, layer_idx: int) -> Flux2KVLayerCache:
        return self.single_block_caches[layer_idx]

    def clear(self) -> None:
        for cache in self.double_block_caches:
            cache.clear()
        for cache in self.single_block_caches:
            cache.clear()
        self.num_ref_tokens = 0


def flux2_kv_causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_txt_tokens: int,
    num_ref_tokens: int,
    kv_cache: Flux2KVLayerCache | None = None,
) -> torch.Tensor:
    """Causal attention where reference tokens only self-attend.

    Tensor format: (B, L, H, D).
    """
    if num_ref_tokens == 0 and kv_cache is None:
        B, L, H, D = query.shape
        query_t = query.transpose(1, 2)
        key_t = key.transpose(1, 2)
        value_t = value.transpose(1, 2)
        out = F.scaled_dot_product_attention(query_t, key_t, value_t)
        return out.transpose(1, 2).flatten(2, 3)

    if kv_cache is not None:
        k_ref, v_ref = kv_cache.get()
        k_ref_t = k_ref.transpose(1, 2)
        v_ref_t = v_ref.transpose(1, 2)

        query_t = query.transpose(1, 2)
        key_t = key.transpose(1, 2)
        value_t = value.transpose(1, 2)

        k_all = torch.cat([key_t[:, :, :num_txt_tokens], k_ref_t, key_t[:, :, num_txt_tokens:]], dim=2)
        v_all = torch.cat([value_t[:, :, :num_txt_tokens], v_ref_t, value_t[:, :, num_txt_tokens:]], dim=2)

        out = F.scaled_dot_product_attention(query_t, k_all, v_all)
        return out.transpose(1, 2).flatten(2, 3)

    ref_start = num_txt_tokens
    ref_end = num_txt_tokens + num_ref_tokens

    query_t = query.transpose(1, 2)
    key_t = key.transpose(1, 2)
    value_t = value.transpose(1, 2)

    q_txt = query_t[:, :, :ref_start]
    q_ref = query_t[:, :, ref_start:ref_end]
    q_img = query_t[:, :, ref_end:]

    k_txt = key_t[:, :, :ref_start]
    k_ref = key_t[:, :, ref_start:ref_end]
    k_img = key_t[:, :, ref_end:]

    v_txt = value_t[:, :, :ref_start]
    v_ref = value_t[:, :, ref_start:ref_end]
    v_img = value_t[:, :, ref_end:]

    q_txt_img = torch.cat([q_txt, q_img], dim=2)
    k_all = torch.cat([k_txt, k_ref, k_img], dim=2)
    v_all = torch.cat([v_txt, v_ref, v_img], dim=2)
    attn_txt_img = F.scaled_dot_product_attention(q_txt_img, k_all, v_all)
    attn_txt = attn_txt_img[:, :, :ref_start]
    attn_img = attn_txt_img[:, :, ref_start:]

    attn_ref = F.scaled_dot_product_attention(q_ref, k_ref, v_ref)

    out = torch.cat([attn_txt, attn_ref, attn_img], dim=2)
    return out.transpose(1, 2).flatten(2, 3)


def _blend_mod_params(
    img_params: tuple[torch.Tensor, ...],
    ref_params: tuple[torch.Tensor, ...],
    num_ref: int,
    seq_len: int,
) -> tuple[torch.Tensor, ...]:
    blended = []
    for im, rm in zip(img_params, ref_params):
        if im.ndim == 2:
            im = im.unsqueeze(1)
            rm = rm.unsqueeze(1)
        B = im.shape[0]
        blended.append(torch.cat([rm.expand(B, num_ref, -1), im.expand(B, seq_len, -1)[:, num_ref:, :]], dim=1))
    return tuple(blended)


def blend_double_block_mods(
    img_mod: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...],
    ref_mod: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...],
    num_ref: int,
    seq_len: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]:
    """Blend double-block modulations for [ref, img] layout.

    Args:
        img_mod: Tuple of (shift, scale, gate) modulation tuples for image stream.
            Format: ((shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp))
        ref_mod: Tuple of (shift, scale, gate) modulation tuples for reference tokens.
        num_ref: Number of reference tokens.
        seq_len: Total sequence length (including ref tokens).

    Returns:
        Blended modulation tuple where first num_ref positions use ref_mod.
    """
    blended_sets = []
    for img_set, ref_set in zip(img_mod, ref_mod):
        blended = _blend_mod_params(img_set, ref_set, num_ref, seq_len)
        blended_sets.append(blended)
    return tuple(blended_sets)


def blend_single_block_mods(
    single_mod: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ref_mod: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    num_txt: int,
    num_ref: int,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Blend single-block modulations for [txt, ref, img] layout.

    Args:
        single_mod: Tuple of (shift, scale, gate) for image stream.
        ref_mod: Tuple of (shift, scale, gate) for reference tokens.
        num_txt: Number of text tokens.
        num_ref: Number of reference tokens.
        seq_len: Total sequence length.

    Returns:
        Blended modulation tuple for [txt, ref, img] layout.
    """
    blended = []
    for im, rm in zip(single_mod, ref_mod):
        if im.ndim == 2:
            im = im.unsqueeze(1)
            rm = rm.unsqueeze(1)
        B = im.shape[0]
        im_expanded = im.expand(B, seq_len, -1)
        rm_expanded = rm.expand(B, num_ref, -1)
        blended.append(
            torch.cat([im_expanded[:, :num_txt, :], rm_expanded, im_expanded[:, num_txt + num_ref :, :]], dim=1)
        )
    return tuple(blended)
