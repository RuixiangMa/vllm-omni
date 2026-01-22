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

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable
from typing import Any

import PIL.Image
import torch
import torch.nn as nn
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import retrieve_latents
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.flux_kontext.flux_kontext_transformer import (
    FluxKontextTransformer2DModel,
)
from vllm_omni.diffusion.models.interface import SupportImageInput
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.utils.tf_utils import get_transformer_config_kwargs
from vllm_omni.logger import init_logger

logger = init_logger(__name__)


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


class FluxKontextImageProcessor(VaeImageProcessor):
    """Image processor to preprocess the reference image for Flux Kontext."""

    def __init__(
        self,
        do_resize: bool = True,
        vae_scale_factor: int = 8,
        vae_latent_channels: int = 16,
        do_normalize: bool = True,
        do_convert_rgb: bool = True,
    ):
        super().__init__(
            do_resize=do_resize,
            vae_scale_factor=vae_scale_factor,
            vae_latent_channels=vae_latent_channels,
            do_normalize=do_normalize,
            do_convert_rgb=do_convert_rgb,
        )

    @staticmethod
    def check_image_input(
        image: PIL.Image.Image,
        max_aspect_ratio: float = 8.0,
        min_side_length: int = 64,
        max_area: int = 1024 * 1024,
    ) -> PIL.Image.Image:
        if not isinstance(image, PIL.Image.Image):
            raise ValueError(f"Image must be a PIL.Image.Image, got {type(image)}")

        width, height = image.size
        if width < min_side_length or height < min_side_length:
            raise ValueError(f"Image too small: {width}x{height}. Both dimensions must be at least {min_side_length}px")

        aspect_ratio = max(width / height, height / width)
        if aspect_ratio > max_aspect_ratio:
            raise ValueError(
                f"Aspect ratio too extreme: {width}x{height} (ratio: {aspect_ratio:.1f}:1). "
                f"Maximum allowed ratio is {max_aspect_ratio}:1"
            )

        if width * height > max_area:
            logger.warning("Image area exceeds recommended maximum; resizing will be applied.")

        return image


def get_flux_kontext_post_process_func(od_config: OmniDiffusionConfig) -> Callable:
    """Get postprocessing function for FLUX.1-Kontext pipeline."""
    model_name = od_config.model
    if os.path.exists(model_name):
        model_path = model_name
    else:
        from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific

        model_path = download_weights_from_hf_specific(model_name, None, ["*"])

    vae_config_path = os.path.join(model_path, "vae", "config.json")
    if os.path.exists(vae_config_path):
        with open(vae_config_path) as f:
            vae_config = json.load(f)
            vae_scale_factor = (
                2 ** (len(vae_config["block_out_channels"]) - 1) if "block_out_channels" in vae_config else 8
            )
    else:
        vae_scale_factor = 8

    image_processor = FluxKontextImageProcessor(vae_scale_factor=vae_scale_factor)

    def post_process_func(images: torch.Tensor) -> list[PIL.Image.Image]:
        return image_processor.postprocess(images)

    return post_process_func


def get_flux_kontext_pre_process_func(od_config: OmniDiffusionConfig) -> Callable:
    """Get preprocessing function for FLUX.1-Kontext pipeline."""

    def pre_process_func(requests: list[OmniDiffusionRequest]) -> list[OmniDiffusionRequest]:
        for request in requests:
            if hasattr(request, "pil_image") and request.pil_image is not None:
                if isinstance(request.pil_image, str):
                    from PIL import Image

                    request.pil_image = Image.open(request.pil_image).convert("RGB")
            return requests

    return pre_process_func


class FluxKontextPipeline(nn.Module, SupportImageInput):
    """FLUX.1-Kontext pipeline for image editing with text guidance."""

    support_image_input = True

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            )
        ]

        self._execution_device = get_local_device()
        model = od_config.model
        local_files_only = os.path.exists(model)

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model,
            subfolder="scheduler",
            local_files_only=local_files_only,
        )

        self.text_encoder = CLIPTextModel.from_pretrained(
            model,
            subfolder="text_encoder",
            local_files_only=local_files_only,
        ).to(self._execution_device)

        self.text_encoder_2 = T5EncoderModel.from_pretrained(
            model,
            subfolder="text_encoder_2",
            local_files_only=local_files_only,
        ).to(self._execution_device)

        self.tokenizer = CLIPTokenizer.from_pretrained(
            model,
            subfolder="tokenizer",
            local_files_only=local_files_only,
        )
        self.tokenizer_2 = T5TokenizerFast.from_pretrained(
            model,
            subfolder="tokenizer_2",
            local_files_only=local_files_only,
        )

        self.vae = AutoencoderKL.from_pretrained(
            model,
            subfolder="vae",
            local_files_only=local_files_only,
        ).to(self._execution_device)

        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            )
        ]

        transformer_kwargs = get_transformer_config_kwargs(od_config.tf_model_config, FluxKontextTransformer2DModel)
        self.transformer = FluxKontextTransformer2DModel(**transformer_kwargs)

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = FluxKontextImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.tokenizer_max_length = 512
        self.default_sample_size = 128

        self._guidance_scale = None
        self._attention_kwargs = None
        self._num_timesteps = None
        self._current_timestep = None
        self._interrupt = False

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    @staticmethod
    def _prepare_text_ids(
        x: torch.Tensor,
    ):
        L = x.shape[1]
        text_ids = torch.zeros(L, 3)
        text_ids[:, 2] = torch.arange(L)
        return text_ids.to(x.device, dtype=x.dtype)

    @staticmethod
    def _prepare_latent_ids(
        latents: torch.Tensor,
    ):
        batch_size, _, height, width = latents.shape

        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

        latent_image_ids = latent_image_ids.reshape(height * width, 3)

        return latent_image_ids.to(latents.device, dtype=latents.dtype)

    @staticmethod
    def _prepare_image_ids(
        image_latents,
    ):
        if isinstance(image_latents, list):
            if len(image_latents) == 1:
                image_latent = image_latents[0]
                if image_latent.ndim == 4:
                    image_latents = image_latent
                else:
                    image_latents = image_latent.unsqueeze(0)
            else:
                image_latents = torch.cat(image_latents, dim=0)

        batch_size, _, height, width = image_latents.shape

        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

        latent_image_ids = latent_image_ids.reshape(height * width, 3)

        return latent_image_ids.to(image_latents.device, dtype=image_latents.dtype)

    @staticmethod
    def _patchify_latents(latents):
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        latents = latents.reshape(batch_size, num_channels_latents * 4, height // 2, width // 2)
        return latents

    @staticmethod
    def _unpatchify_latents(latents):
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), 2, 2, height, width)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), height * 2, width * 2)
        return latents

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
        return latents

    @staticmethod
    def _unpack_latents(latents: torch.Tensor, height: int, width: int, vae_scale_factor: int = 8) -> torch.Tensor:
        if len(latents.shape) == 4:
            return latents
        batch_size, num_patches, channels = latents.shape

        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        # The packed format has channels = original_channels * 4
        # We need to divide by 4 to get the original channel count
        channels_out = channels // 4

        latents = latents.view(batch_size, height // 2, width // 2, channels_out, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels_out, height, width)
        return latents

    @staticmethod
    def _unpack_latents_with_ids(x: torch.Tensor, x_ids: torch.Tensor) -> list[torch.Tensor]:
        x_list = []
        for data, pos in zip(x, x_ids):
            _, ch = data.shape
            h_ids = pos[:, 1].to(torch.int64)
            w_ids = pos[:, 2].to(torch.int64)

            h = torch.max(h_ids) + 1
            w = torch.max(w_ids) + 1

            flat_ids = h_ids * w + w_ids

            out = torch.zeros((h * w, ch), device=data.device, dtype=data.dtype)
            out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)

            out = out.view(h, w, ch).permute(2, 0, 1)
            x_list.append(out)

        return torch.stack(x_list, dim=0)

    def encode_prompt(
        self,
        prompt: str | list[str],
        device: torch.device | None = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        max_sequence_length: int = 512,
        negative_prompt: str | list[str] | None = None,
    ):
        device = device or self._execution_device

        if prompt is None:
            prompt = ""

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            prompt_embeds = self._encode_prompt(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                negative_prompt=negative_prompt,
            )

        return prompt_embeds

    def _encode_prompt(
        self,
        prompt: list[str],
        device: torch.device,
        num_images_per_prompt: int,
        max_sequence_length: int,
        negative_prompt: list[str] | None,
    ) -> torch.Tensor:
        prompt_embeds_list = []

        for prompt_item in prompt:
            text_inputs = self.tokenizer_2(
                prompt_item,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                return_tensors="pt",
            )

            text_input_ids = text_inputs.input_ids.to(device)

            with torch.no_grad():
                prompt_embeds = self.text_encoder_2(text_input_ids)[0]

            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.cat(prompt_embeds_list, dim=0)

        if self.text_encoder is not None:
            clip_input = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.text_encoder.config.max_position_embeddings,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)

            with torch.no_grad():
                pooled_prompt_embeds = self.text_encoder(clip_input, output_hidden_states=False).pooler_output
            pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=prompt_embeds.dtype, device=device)
        else:
            pooled_prompt_embeds = prompt_embeds.mean(dim=-1)

        if num_images_per_prompt > 1:
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
            pooled_prompt_embeds = pooled_prompt_embeds.repeat(num_images_per_prompt, 1)

        text_ids = self._prepare_text_ids(prompt_embeds)
        text_ids = text_ids.to(device)

        return prompt_embeds, pooled_prompt_embeds, text_ids

    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if image.ndim != 4:
            raise ValueError(f"Expected image dims 4, got {image.ndim}.")

        image_latents = retrieve_latents(self.vae.encode(image), generator=generator, sample_mode="argmax")
        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        if image_latents.ndim == 5:
            image_latents = image_latents.squeeze(0)

        return image_latents

    def prepare_latents(
        self,
        batch_size: int,
        num_latents_channels: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator,
        latents: torch.Tensor | None = None,
    ):
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_latents_channels, height, width)

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        latent_ids = self._prepare_latent_ids(latents)
        latent_ids = latent_ids.to(device)

        latents = self._pack_latents(latents, batch_size, num_latents_channels, height, width)

        return latents, latent_ids

    def prepare_image_latents(
        self,
        images: list[torch.Tensor],
        batch_size: int,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ):
        image_latents = []
        for image in images:
            if not isinstance(image, torch.Tensor):
                image = self.image_processor.pil_to_numpy(image)
                image = self.image_processor.numpy_to_pt(image)
            image = image.to(device=device, dtype=dtype)
            image_latent = self._encode_vae_image(image=image, generator=generator)
            image_latents.append(image_latent)

        image_latent_ids = self._prepare_image_ids(image_latents)

        if len(image_latents) == 1:
            image_latent = image_latents[0]
            if image_latent.ndim == 4:
                image_latents = image_latent
            else:
                image_latents = image_latent.unsqueeze(0)
        else:
            image_latents = torch.cat(image_latents, dim=0)

        if batch_size > 1:
            image_latents = image_latents.repeat(batch_size, 1, 1, 1)

        image_latent_ids = image_latent_ids.to(device)

        _, num_channels, height, width = image_latents.shape
        image_latents = self._pack_latents(image_latents, image_latents.shape[0], num_channels, height, width)

        return image_latents, image_latent_ids

    def check_inputs(
        self,
        prompt,
        height,
        width,
        prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        guidance_scale=None,
    ):
        if (
            height is not None
            and height % (self.vae_scale_factor * 2) != 0
            or width is not None
            and width % (self.vae_scale_factor * 2) != 0
        ):
            logger.warning(
                "`height` and `width` have to be divisible by %s but are %s and %s. "
                "Dimensions will be resized accordingly",
                self.vae_scale_factor * 2,
                height,
                width,
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in ["latents", "prompt_embeds"] for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError("`callback_on_step_end_tensor_inputs` must be a subset of ['latents', 'prompt_embeds'].")

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if guidance_scale > 1.0:
            logger.warning(f"Guidance scale {guidance_scale} is not typically used for this model.")

    def check_latents_shape(
        self,
        height: int | None,
        width: int | None,
    ) -> tuple[int, int]:
        """Check and adjust latents shape based on VAE scale factor."""
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        return height, width

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale is not None and self._guidance_scale > 1

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    def forward(
        self,
        req: OmniDiffusionRequest,
        image: PIL.Image.Image | list[PIL.Image.Image] | None = None,
        prompt: str | list[str] | None = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 512,
    ) -> DiffusionOutput:
        prompt = req.prompt if req.prompt is not None else prompt
        image = req.pil_image if req.pil_image is not None else image
        height = req.height or height
        width = req.width or width
        num_inference_steps = req.num_inference_steps or num_inference_steps
        guidance_scale = req.guidance_scale or guidance_scale
        generator = req.generator or generator
        latents = req.latents or latents

        height, width = self.check_latents_shape(height, width)

        self.check_inputs(
            prompt,
            height,
            width,
            prompt_embeds,
            callback_on_step_end_tensor_inputs,
            guidance_scale,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        device = self._execution_device

        prompt_embeds, pooled_prompt_embeds, txt_ids = self.encode_prompt(
            prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            prompt_embeds=prompt_embeds,
            max_sequence_length=max_sequence_length,
            negative_prompt=negative_prompt_embeds,
        )

        num_channels_latents = self.vae.config.latent_channels

        latents, img_ids = self.prepare_latents(
            batch_size=num_images_per_prompt,
            num_latents_channels=num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        if image is not None:
            if not isinstance(image, list):
                image = [image]

            image_latents, image_ids = self.prepare_image_latents(
                image,
                batch_size=num_images_per_prompt,
                generator=generator,
                device=device,
                dtype=prompt_embeds.dtype,
            )

            image_pooled_projections = pooled_prompt_embeds
            latents = image_latents
            img_ids = image_ids
        else:
            image_pooled_projections = None

        if self.scheduler.config.use_dynamic_shifting:
            image_seq_len = latents.shape[1]
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.15),
            )
        else:
            mu = None

        self.scheduler.set_timesteps(num_inference_steps, device=device, mu=mu)
        scheduler_copy = self.scheduler

        self._num_timesteps = len(scheduler_copy.timesteps)
        self._current_timestep = 0

        extra_step_kwargs = {}
        if generator is not None:
            extra_step_kwargs["generator"] = generator

        if attention_kwargs is not None:
            if "cross_attention_kwargs" in attention_kwargs:
                extra_step_kwargs["cross_attention_kwargs"] = attention_kwargs["cross_attention_kwargs"]

        callback_on_step_end_inputs = (self,)

        for i, t in enumerate(scheduler_copy.timesteps):
            if self.interrupt:
                continue

            t = t.to(device=device)
            t = t.to(dtype=self.transformer.parameters().__next__().dtype)
            t = t.expand(latents.shape[0])

            denoised = self.transformer(
                hidden_states=latents,
                timestep=t,
                encoder_hidden_states=prompt_embeds,
                txt_ids=txt_ids,
                img_ids=img_ids,
                pooled_projections=pooled_prompt_embeds,
                image_pooled_projections=image_pooled_projections,
                guidance=t.new_zeros(latents.shape[0]) + guidance_scale,
                return_dict=False,
            )[0]

            latents = scheduler_copy.step(denoised, t, latents, **extra_step_kwargs, return_dict=False)[0]

            if callback_on_step_end_tensor_inputs is not None and callback_on_step_end is not None:
                callback_on_step_end_inputs = callback_on_step_end_tensor_inputs

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for input_name in callback_on_step_end_tensor_inputs:
                    callback_kwargs[input_name] = locals()[input_name]
                callback_on_step_end(i, t, callback_kwargs)

            self._current_timestep = t

        latents = latents.to(dtype=self.vae.dtype)
        if len(latents.shape) != 4:
            num_patches = latents.shape[1]
            vae_scale_factor = self.vae_scale_factor
            patches_h = int(num_patches**0.5)
            height = patches_h * 2 * vae_scale_factor
            width = height
            latents = self._unpack_latents(latents, height, width, vae_scale_factor)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]

        image = (image / 2 + 0.5).clamp(0, 1)

        if output_type == "latent":
            batch_size, num_channels, height, width = image.shape
            latents = self._pack_latents(image, batch_size, num_channels, height, width)
            return DiffusionOutput(output=latents)

        return DiffusionOutput(output=image)
