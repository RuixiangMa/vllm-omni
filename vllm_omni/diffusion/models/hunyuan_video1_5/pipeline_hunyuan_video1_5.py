# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable
from typing import Any

import numpy as np
import PIL.Image
import torch
from diffusers import AutoencoderKLHunyuanVideo15
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import (
    ByT5Tokenizer,
    Qwen2_5_VLTextModel,
    Qwen2Tokenizer,
    SiglipImageProcessor,
    SiglipVisionModel,
    T5EncoderModel,
)

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.hunyuan_video1_5.hunyuan_video15_transformer import (
    HunyuanVideo15Transformer3DModel,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniTextPrompt

logger = logging.getLogger(__name__)


def retrieve_latents(
    encoder_output: torch.Tensor,
    generator: torch.Generator | None = None,
    sample_mode: str = "sample",
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


def format_text_input(prompt: list[str], system_message: str) -> list[dict[str, Any]]:
    template = [
        [{"role": "system", "content": system_message}, {"role": "user", "content": p if p else " "}] for p in prompt
    ]
    return template


def extract_glyph_texts(prompt: str) -> list[str]:
    import re

    pattern = r'"(.*?)"|"(.*?)"'
    matches = re.findall(pattern, prompt)
    result = [match[0] or match[1] for match in matches]
    result = list(dict.fromkeys(result)) if len(result) > 1 else result

    if result:
        formatted_result = ". ".join([f'Text "{text}"' for text in result]) + ". "
    else:
        formatted_result = None

    return formatted_result


def get_hunyuan_video15_post_process_func(
    od_config: OmniDiffusionConfig,
):
    from diffusers.video_processor import VideoProcessor

    video_processor = VideoProcessor(vae_scale_factor=16)

    def post_process_func(
        video: torch.Tensor,
        output_type: str = "np",
    ):
        if output_type == "latent":
            return video
        return video_processor.postprocess_video(video, output_type=output_type)

    return post_process_func


def get_hunyuan_video15_pre_process_func(
    od_config: OmniDiffusionConfig,
):
    from diffusers.video_processor import VideoProcessor

    video_processor = VideoProcessor(vae_scale_factor=16)

    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        for i, prompt in enumerate(request.prompts):
            multi_modal_data = prompt.get("multi_modal_data", {}) if not isinstance(prompt, str) else None
            raw_image = multi_modal_data.get("image", None) if multi_modal_data is not None else None
            if isinstance(prompt, str):
                prompt = OmniTextPrompt(prompt=prompt)
            if "additional_information" not in prompt:
                prompt["additional_information"] = {}

            if raw_image is None:
                continue

            if not isinstance(raw_image, (str, PIL.Image.Image)):
                raise TypeError(f"Unsupported image format {raw_image.__class__}.")

            image = PIL.Image.open(raw_image).convert("RGB") if isinstance(raw_image, str) else raw_image

            prompt["additional_information"]["preprocessed_image"] = video_processor.preprocess(
                image,
                height=256,
                width=256,
            )
            request.prompts[i] = prompt
        return request

    return pre_process_func


class HunyuanVideo15Pipeline(nn.Module, CFGParallelMixin):
    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config

        self.device = get_local_device()
        getattr(od_config, "dtype", torch.bfloat16)
        model = od_config.model
        local_files_only = os.path.exists(model)

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model,
            subfolder="scheduler",
            local_files_only=local_files_only,
        )
        self.text_encoder = Qwen2_5_VLTextModel.from_pretrained(
            model, subfolder="text_encoder", local_files_only=local_files_only
        ).to(self.device)
        self.tokenizer = Qwen2Tokenizer.from_pretrained(model, subfolder="tokenizer", local_files_only=local_files_only)
        self.text_encoder_2 = T5EncoderModel.from_pretrained(
            model, subfolder="text_encoder_2", local_files_only=local_files_only
        ).to(self.device)
        self.tokenizer_2 = ByT5Tokenizer.from_pretrained(
            model, subfolder="tokenizer_2", local_files_only=local_files_only
        )
        self.vae = AutoencoderKLHunyuanVideo15.from_pretrained(
            model, subfolder="vae", local_files_only=local_files_only
        ).to(self.device)

        model_index_path = os.path.join(model, "model_index.json")
        model_index = {}
        if os.path.exists(model_index_path):
            with open(model_index_path) as f:
                model_index = json.load(f)

        has_image_encoder = "image_encoder" in model_index and model_index["image_encoder"] is not None

        if has_image_encoder:
            self.image_encoder = SiglipVisionModel.from_pretrained(
                model, subfolder="image_encoder", local_files_only=local_files_only
            ).to(self.device)
            self.feature_extractor = SiglipImageProcessor.from_pretrained(
                model, subfolder="feature_extractor", local_files_only=local_files_only
            )
        else:
            self.image_encoder = None
            self.feature_extractor = None

        transformer_kwargs = self._get_transformer_kwargs(od_config)
        self.transformer = HunyuanVideo15Transformer3DModel(
            od_config=od_config,
            **transformer_kwargs,
        )

        self.vae_scale_factor_temporal = (
            self.vae.temporal_compression_ratio if hasattr(self.vae, "temporal_compression_ratio") else 4
        )
        self.vae_scale_factor_spatial = (
            self.vae.spatial_compression_ratio if hasattr(self.vae, "spatial_compression_ratio") else 16
        )
        self.target_size = 640
        self.vision_states_dim = 1152
        self.num_channels_latents = (
            self.vae.config.latent_channels if hasattr(self.vae.config, "latent_channels") else 32
        )
        self.system_message = (
            "You are a helpful assistant. Describe the video by detailing the following aspects: "
            "1. The main content and theme of the video. "
            "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. "
            "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects. "
            "4. background environment, light, style and atmosphere. "
            "5. camera angles, movements, and transitions used in the video."
        )
        self.prompt_template_encode_start_idx = 108
        self.tokenizer_max_length = 1000
        self.tokenizer_2_max_length = 256
        self.vision_num_semantic_tokens = 729

    def _get_transformer_kwargs(self, od_config: OmniDiffusionConfig) -> dict:
        model_path = od_config.model
        subfolder = "transformer"
        os.path.exists(model_path)

        config_path = os.path.join(model_path, subfolder, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Transformer config not found at {config_path}")

        with open(config_path) as f:
            config = json.load(f)

        kwargs = {}
        mapping = {
            "in_channels": "in_channels",
            "out_channels": "out_channels",
            "num_attention_heads": "num_attention_heads",
            "attention_head_dim": "attention_head_dim",
            "num_layers": "num_layers",
            "num_refiner_layers": "num_refiner_layers",
            "mlp_ratio": "mlp_ratio",
            "patch_size": "patch_size",
            "patch_size_t": "patch_size_t",
            "qk_norm": "qk_norm",
            "text_embed_dim": "text_embed_dim",
            "text_embed_2_dim": "text_embed_2_dim",
            "image_embed_dim": "image_embed_dim",
            "rope_theta": "rope_theta",
            "rope_axes_dim": "rope_axes_dim",
            "target_size": "target_size",
            "task_type": "task_type",
            "use_meanflow": "use_meanflow",
        }

        for hf_key, our_key in mapping.items():
            if hf_key in config:
                kwargs[our_key] = config[hf_key]

        return kwargs

    def _get_mllm_prompt_embeds(
        self,
        prompt: list[str],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt = format_text_input(prompt, self.system_message)

        text_inputs = self.tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            padding="max_length",
            max_length=self.tokenizer_max_length + self.prompt_template_encode_start_idx,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(device=device)
        prompt_attention_mask = text_inputs.attention_mask.to(device=device)

        prompt_embeds = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_attention_mask,
            output_hidden_states=True,
        ).hidden_states[-(2 + 1)]

        if self.prompt_template_encode_start_idx > 0:
            prompt_embeds = prompt_embeds[:, self.prompt_template_encode_start_idx :]
            prompt_attention_mask = prompt_attention_mask[:, self.prompt_template_encode_start_idx :]

        return prompt_embeds, prompt_attention_mask

    def _get_byt5_prompt_embeds(
        self,
        prompt: list[str],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        glyph_texts = [extract_glyph_texts(p) for p in prompt]

        prompt_embeds_list = []
        prompt_embeds_mask_list = []

        for glyph_text in glyph_texts:
            if glyph_text is None:
                glyph_text_embeds = torch.zeros(
                    (1, self.tokenizer_2_max_length, self.text_encoder_2.config.d_model),
                    device=device,
                    dtype=self.text_encoder_2.dtype,
                )
                glyph_text_embeds_mask = torch.zeros((1, self.tokenizer_2_max_length), device=device, dtype=torch.int64)
            else:
                txt_tokens = self.tokenizer_2(
                    glyph_text,
                    padding="max_length",
                    max_length=self.tokenizer_2_max_length,
                    truncation=True,
                    add_special_tokens=True,
                    return_tensors="pt",
                ).to(device)

                glyph_text_embeds = self.text_encoder_2(
                    input_ids=txt_tokens.input_ids,
                    attention_mask=txt_tokens.attention_mask.float(),
                )[0]
                glyph_text_embeds = glyph_text_embeds.to(device=device)
                glyph_text_embeds_mask = txt_tokens.attention_mask.to(device=device)

            prompt_embeds_list.append(glyph_text_embeds)
            prompt_embeds_mask_list.append(glyph_text_embeds_mask)

        prompt_embeds = torch.cat(prompt_embeds_list, dim=0)
        prompt_embeds_mask = torch.cat(prompt_embeds_mask_list, dim=0)

        return prompt_embeds, prompt_embeds_mask

    def _get_image_embeds(
        self,
        image: PIL.Image.Image,
        device: torch.device,
    ) -> torch.Tensor:
        if self.image_encoder is None:
            raise ValueError("I2V is not enabled. Set enable_i2v=True in config.")
        image_encoder_dtype = next(self.image_encoder.parameters()).dtype
        image_inputs = self.feature_extractor.preprocess(
            images=image,
            do_resize=True,
            return_tensors="pt",
            do_convert_rgb=True,
        )
        image_inputs = image_inputs.to(device, dtype=image_encoder_dtype)
        image_enc_hidden_states = self.image_encoder(**image_inputs).last_hidden_state
        return image_enc_hidden_states

    def _get_image_latents(
        self,
        image: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        image_latents = retrieve_latents(self.vae.encode(image), sample_mode="argmax")
        image_latents = image_latents * self.vae.config.scaling_factor
        return image_latents

    def encode_prompt(
        self,
        prompt: list[str],
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
        num_videos_per_prompt: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt_embeds, prompt_embeds_mask = self._get_mllm_prompt_embeds(prompt, device)
        prompt_embeds_2, prompt_embeds_mask_2 = self._get_byt5_prompt_embeds(prompt, device)

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_videos_per_prompt, seq_len)

        _, seq_len_2, _ = prompt_embeds_2.shape
        prompt_embeds_2 = prompt_embeds_2.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds_2 = prompt_embeds_2.view(batch_size * num_videos_per_prompt, seq_len_2, -1)
        prompt_embeds_mask_2 = prompt_embeds_mask_2.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds_mask_2 = prompt_embeds_mask_2.view(batch_size * num_videos_per_prompt, seq_len_2)

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds_mask = prompt_embeds_mask.to(dtype=dtype, device=device)
        prompt_embeds_2 = prompt_embeds_2.to(dtype=dtype, device=device)
        prompt_embeds_mask_2 = prompt_embeds_mask_2.to(dtype=dtype, device=device)

        return prompt_embeds, prompt_embeds_mask, prompt_embeds_2, prompt_embeds_mask_2

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        num_frames: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | None = None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shape = (
            batch_size,
            num_channels_latents,
            num_frames,
            height,
            width,
        )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        return latents

    def prepare_cond_latents_and_mask(
        self,
        latents: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cond_latents_concat = torch.zeros_like(latents, dtype=dtype, device=device)
        mask_concat = torch.ones_like(latents[:, :1], dtype=dtype, device=device)
        return cond_latents_concat, mask_concat

    def forward(
        self,
        req: OmniDiffusionRequest,
    ) -> DiffusionOutput:
        device = self.device
        dtype = self.transformer.parameters().__next__().dtype

        if req.prompts is not None:
            prompt_list = []
            image_list = []
            for p in req.prompts:
                if isinstance(p, str):
                    prompt_list.append(p)
                    image_list.append(None)
                elif isinstance(p, dict):
                    prompt_list.append(p.get("prompt", ""))
                    multi_modal_data = p.get("multi_modal_data", {})
                    image_list.append(multi_modal_data.get("image", None))
                else:
                    prompt_list.append("")
                    image_list.append(None)
        else:
            prompt_list = [""]
            image_list = [None]

        sampling_params = req.sampling_params
        num_inference_steps = sampling_params.num_inference_steps or 50
        guidance_scale = sampling_params.guidance_scale or 7.5
        height = sampling_params.height or (self.target_size * self.vae_scale_factor_spatial // 16 * 16)
        width = sampling_params.width or (self.target_size * self.vae_scale_factor_spatial // 16 * 16)
        num_frames = getattr(sampling_params, "num_frames", 49) or 49

        batch_size = len(prompt_list)

        prompt_embeds, prompt_embeds_mask, prompt_embeds_2, prompt_embeds_mask_2 = self.encode_prompt(
            prompt=prompt_list,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
        )

        image_embeds_list = []
        for raw_image in image_list:
            if raw_image is not None and self.image_encoder is not None:
                image_embeds = self._get_image_embeds(raw_image, device)
            else:
                image_embeds = torch.zeros(
                    batch_size,
                    self.vision_num_semantic_tokens,
                    self.vision_states_dim,
                    dtype=dtype,
                    device=device,
                )
            image_embeds_list.append(image_embeds)
        image_embeds = torch.cat(image_embeds_list, dim=0)

        latents = self.prepare_latents(
            batch_size,
            self.num_channels_latents,
            height,
            width,
            num_frames,
            dtype,
            device,
        )
        cond_latents_concat, mask_concat = self.prepare_cond_latents_and_mask(latents, dtype, device)

        sigmas = np.linspace(1.0, 0.0, num_inference_steps + 1)[:-1]
        self.scheduler.set_timesteps(num_inference_steps, device=device, sigmas=sigmas)
        timesteps = self.scheduler.timesteps

        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents, cond_latents_concat, mask_concat], dim=1)
            timestep = t.expand(latent_model_input.shape[0]).to(latent_model_input.dtype)

            guider_inputs = {
                "encoder_hidden_states": (prompt_embeds,),
                "encoder_attention_mask": (prompt_embeds_mask,),
                "encoder_hidden_states_2": (prompt_embeds_2,),
                "encoder_attention_mask_2": (prompt_embeds_mask_2,),
            }

            if hasattr(self, "set_guider_state"):
                self.set_guider_state(
                    step=i,
                    num_inference_steps=num_inference_steps,
                    timestep=t,
                    guidance_scale=guidance_scale,
                    guider_inputs=guider_inputs,
                )

            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_embeds_mask,
                encoder_hidden_states_2=prompt_embeds_2,
                encoder_attention_mask_2=prompt_embeds_mask_2,
                image_embeds=image_embeds,
            )

            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        latents = latents.to(self.vae.dtype) / self.vae.config.scaling_factor
        video = self.vae.decode(latents, return_dict=False)[0]

        return DiffusionOutput(output=video)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        from vllm.model_executor.models.utils import AutoWeightsLoader

        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
