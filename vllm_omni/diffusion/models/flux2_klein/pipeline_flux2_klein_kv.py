# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.flux2_klein.kv_cache import Flux2KVCache
from vllm_omni.diffusion.models.flux2_klein.pipeline_flux2_klein import Flux2KleinPipeline

if TYPE_CHECKING:
    from vllm_omni.diffusion.request import OmniDiffusionRequest


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b

    return float(mu)


class Flux2KleinKVPipeline(Flux2KleinPipeline):
    """Flux2 klein pipeline with KV cache optimization for fast image editing.

    On the first denoising step, reference image tokens are included in the forward
    pass and their attention K/V projections are cached. On subsequent steps, the
    cached K/V are reused without recomputing, providing faster inference when
    using reference images.
    """

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
        is_distilled: bool = True,
    ):
        super().__init__(od_config=od_config, prefix=prefix, is_distilled=is_distilled)
        self.use_kv_cache = True

    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        """Forward with KV cache optimization."""
        from typing import cast

        import PIL.Image
        from vllm.logger import init_logger

        logger = init_logger(__name__)

        if len(req.prompts) > 1:
            logger.warning("This model only supports a single prompt. Taking only the first.")

        first_prompt = req.prompts[0]
        prompt = first_prompt if isinstance(first_prompt, str) else (first_prompt.get("prompt") or "")

        raw_image = None if isinstance(first_prompt, str) else first_prompt.get("multi_modal_data", {}).get("image")
        image = None
        if raw_image is not None:
            if isinstance(raw_image, list):
                image = [PIL.Image.open(im) if isinstance(im, str) else cast(PIL.Image.Image, im) for im in raw_image]
            else:
                image = PIL.Image.open(raw_image) if isinstance(raw_image, str) else cast(PIL.Image.Image, raw_image)

        height = req.sampling_params.height or 1024
        width = req.sampling_params.width or 1024
        num_inference_steps = req.sampling_params.num_inference_steps or 4
        guidance_scale = req.sampling_params.guidance_scale if req.sampling_params.guidance_scale is not None else 1.0
        generator = req.sampling_params.generator

        self.check_inputs(prompt, height, width, None, None, guidance_scale)

        self._guidance_scale = guidance_scale
        device = self._execution_device

        prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=512,
        )

        condition_images = None
        if image is not None:
            if not isinstance(image, list):
                image = [image]
            condition_images = []
            for img in image:
                self.image_processor.check_image_input(img)
                image_width, image_height = img.size
                if image_width * image_height > 1024 * 1024:
                    img = self.image_processor._resize_to_target_area(img, 1024 * 1024)
                multiple_of = self.vae_scale_factor * 2
                image_width = (image_width // multiple_of) * multiple_of
                image_height = (image_height // multiple_of) * multiple_of
                img = self.image_processor.preprocess(img, height=image_height, width=image_width, resize_mode="crop")
                condition_images.append(img)
                height = height or image_height
                width = width or image_width

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_ids = self.prepare_latents(
            batch_size=1,
            num_latents_channels=num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
        )

        image_latents = None
        image_latent_ids = None
        if condition_images is not None:
            image_latents, image_latent_ids = self.prepare_image_latents(
                images=condition_images,
                batch_size=1,
                generator=generator,
                device=device,
                dtype=self.vae.dtype,
            )

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]
        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )

        kv_cache: Flux2KVCache | None = None

        self.scheduler.set_begin_index(0)
        for i, t in enumerate(timesteps):
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            if i == 0 and image_latents is not None:
                kv_cache_mode = "extract"
                latent_model_input = torch.cat([image_latents, latents], dim=1).to(self.transformer.dtype)
                latent_image_ids = torch.cat([image_latent_ids, latent_ids], dim=1)
                num_ref_tokens = image_latents.shape[1]
            else:
                kv_cache_mode = "cached" if kv_cache is not None else None
                latent_model_input = latents.to(self.transformer.dtype)
                latent_image_ids = latent_ids
                num_ref_tokens = 0

            if kv_cache_mode is not None:
                result = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep / 1000,
                    img_ids=latent_image_ids,
                    txt_ids=text_ids,
                    guidance=None,
                    joint_attention_kwargs=None,
                    return_dict=True,
                    kv_cache=kv_cache,
                    kv_cache_mode=kv_cache_mode,
                    num_ref_tokens=num_ref_tokens,
                    ref_fixed_timestep=0.0,
                )

                if kv_cache_mode == "extract":
                    noise_pred, kv_cache = result
                    noise_pred = noise_pred.sample
                else:
                    noise_pred = result.sample
            else:
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep / 1000,
                    img_ids=latent_image_ids,
                    txt_ids=text_ids,
                    guidance=None,
                    joint_attention_kwargs=None,
                    return_dict=True,
                ).sample

            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        latents = self._unpack_latents_with_ids(latents, latent_ids)

        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(
            latents.device, latents.dtype
        )
        latents = latents * latents_bn_std + latents_bn_mean
        latents = self._unpatchify_latents(latents)

        if latents.dtype != self.vae.dtype:
            latents = latents.to(self.vae.dtype)
        image = self.vae.decode(latents, return_dict=False)[0]

        return DiffusionOutput(output=image)


def get_flux2_klein_post_process_func(
    od_config: OmniDiffusionConfig,
):
    from vllm_omni.diffusion.models.flux2_klein.pipeline_flux2_klein import get_flux2_klein_post_process_func as _func

    return _func(od_config)
