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
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import PIL.Image
import torch
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from torch import nn
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    T5EncoderModel,
    T5TokenizerFast,
)
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.flux.configuration_flux import FluxPipelineConfig, FluxTransformerConfig
from vllm_omni.diffusion.models.flux.flux_kontext_transformer import FluxKontextTransformer2DModel
from vllm_omni.diffusion.models.flux.modeling_flux import FluxTransformer2DModel
from vllm_omni.diffusion.request import OmniDiffusionRequest


def retrieve_latents(
    encoder_output: torch.Tensor,
    generator: torch.Generator | None = None,
    sample_mode: str = "sample",
):
    """Retrieve latents from VAE encoder output."""
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


def load_transformer_config(model_path: str, subfolder: str = "transformer", local_files_only: bool = True) -> dict:
    """Load transformer config from model directory or HF Hub."""
    if local_files_only:
        config_path = os.path.join(model_path, subfolder, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                return json.load(f)
    else:
        # Try to download config from HF Hub
        try:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(
                repo_id=model_path,
                filename=f"{subfolder}/config.json",
            )
            with open(config_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def create_transformer_from_config(config: dict) -> FluxKontextTransformer2DModel:
    """Create FluxKontextTransformer2DModel from config dict."""
    kwargs = {}

    if "patch_size" in config:
        kwargs["patch_size"] = config["patch_size"]
    if "in_channels" in config:
        kwargs["in_channels"] = config["in_channels"]
    if "out_channels" in config:
        kwargs["out_channels"] = config["out_channels"]
    if "num_layers" in config:
        kwargs["num_layers"] = config["num_layers"]
    if "num_single_layers" in config:
        kwargs["num_single_layers"] = config["num_single_layers"]
    if "attention_head_dim" in config:
        kwargs["attention_head_dim"] = config["attention_head_dim"]
    if "num_attention_heads" in config:
        kwargs["num_attention_heads"] = config["num_attention_heads"]
    if "joint_attention_dim" in config:
        kwargs["joint_attention_dim"] = config["joint_attention_dim"]
    if "pooled_projection_dim" in config:
        kwargs["pooled_projection_dim"] = config["pooled_projection_dim"]
    if "guidance_embeds" in config:
        kwargs["guidance_embeds"] = config["guidance_embeds"]
    if "axes_dims_rope" in config:
        kwargs["axes_dims_rope"] = config["axes_dims_rope"]
    if "qkv_mlp_ratio" in config:
        kwargs["qkv_mlp_ratio"] = config["qkv_mlp_ratio"]
    if "text_projection_dim" in config:
        kwargs["text_projection_dim"] = config["text_projection_dim"]
    if "image_projection_dim" in config:
        kwargs["image_projection_dim"] = config["image_projection_dim"]

    return FluxKontextTransformer2DModel(**kwargs)


def get_flux_kontext_pre_process_func(od_config=None) -> Callable:
    """Get preprocessing function for FLUX.1-Kontext pipeline."""
    
    def pre_process_func(
        requests: list[OmniDiffusionRequest],
    ):
        """Preprocess inputs for FLUX.1-Kontext pipeline."""
        
        for request in requests:
            # Handle image input (required for Kontext)
            if hasattr(request, "pil_image") and request.pil_image is not None:
                if isinstance(request.pil_image, str):
                    # Load image from path
                    image = Image.open(request.pil_image).convert("RGB")
                elif isinstance(request.pil_image, Image.Image):
                    image = request.pil_image
                else:
                    raise ValueError(f"Unsupported image type: {type(request.pil_image)}")
            else:
                raise ValueError("FLUX.1-Kontext requires an input image for editing")
            
            # Handle text prompt
            prompt = request.prompt if hasattr(request, "prompt") and request.prompt else ""
            negative_prompt = getattr(request, "negative_prompt", "")
            
            # Handle generation parameters
            num_inference_steps = getattr(request, "num_inference_steps", 28)
            guidance_scale = getattr(request, "guidance_scale", 3.5)
            image_strength = getattr(request, "image_strength", 0.9)
            generator = getattr(request, "generator", None)
            
            # Store preprocessed data in request
            request.preprocessed_image = image
            request.preprocessed_prompt = prompt
            request.preprocessed_negative_prompt = negative_prompt
            request.preprocessed_num_inference_steps = num_inference_steps
            request.preprocessed_guidance_scale = guidance_scale
            request.preprocessed_image_strength = image_strength
            request.preprocessed_generator = generator
        
        return requests
    
    return pre_process_func


def get_flux_kontext_post_process_func(od_config=None) -> Callable:
    """Get postprocessing function for FLUX.1-Kontext pipeline."""
    
    def post_process_func(
        images: torch.Tensor,
    ):
        """Postprocess pipeline output."""
        
        if isinstance(images, torch.Tensor):
            # Convert tensor to PIL Image
            if images.dim() == 4:
                # Batch of images
                result = []
                for i in range(images.shape[0]):
                    img_tensor = images[i]
                    # Convert from tensor to PIL
                    img_tensor = (img_tensor.clamp(-1, 1) + 1) / 2 * 255
                    img_tensor = img_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    result.append(Image.fromarray(img_tensor))
                return result
            else:
                # Single image
                img_tensor = (images.clamp(-1, 1) + 1) / 2 * 255
                img_tensor = img_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                return [Image.fromarray(img_tensor)]
        elif isinstance(images, PIL.Image.Image):
            return [images]
        elif isinstance(images, list):
            return images
        else:
            raise ValueError(f"Unsupported output type: {type(images)}")
    
    return post_process_func


class FluxKontextPipeline(nn.Module):
    """
    FLUX.1-Kontext pipeline for image-to-image generation with text guidance.
    
    This pipeline supports dual input (text + image) for editing images based on text prompts.
    It's a 12B parameter model that can perform various image editing tasks while maintaining
    the original image structure and style.
    
    This implementation follows vLLM-Omni patterns with configuration-driven component loading
    and modular architecture for better extensibility.
    
    Reference: https://bfl.ai/announcements/flux-1-kontext-dev
    """

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config
        
        # Create pipeline configuration from OmniDiffusionConfig
        self.pipeline_config = FluxPipelineConfig.from_od_config(od_config)
        
        # Initialize weights loader
        self.weights_loader = AutoWeightsLoader(od_config.model)
        
        # Device configuration
        self.device = get_local_device()
        
        # Load components using configuration
        self._load_components()
        
        # Set up processing parameters from config
        self.vae_scale_factor = self.transformer_config.vae_scale_factor
        self.patch_size = self.pipeline_config.patch_size
        self.latent_scale_factor = self.vae_scale_factor * self.patch_size
        self.max_sequence_length = self.pipeline_config.max_sequence_length
        
        # Default generation parameters from config
        self.default_num_inference_steps = self.pipeline_config.default_num_inference_steps
        self.default_guidance_scale = self.pipeline_config.default_guidance_scale
        self.default_image_strength = self.pipeline_config.default_image_strength

    def _load_transformer_config(self) -> FluxTransformerConfig:
        """Load transformer configuration from model."""
        # Try to load from local config first
        config_dict = load_transformer_config(
            self.pipeline_config.model_path,
            self.pipeline_config.transformer_subfolder,
            self.pipeline_config.local_files_only
        )
        
        # Create FluxTransformerConfig from loaded config or use defaults
        if config_dict:
            # Handle cases where out_channels might be None in the config
            out_channels = config_dict.get("out_channels")
            if out_channels is None:
                # For FLUX models, out_channels typically matches in_channels
                out_channels = config_dict.get("in_channels", 16)
                
            return FluxTransformerConfig(
                patch_size=config_dict.get("patch_size", 1),
                in_channels=config_dict.get("in_channels", 16),
                out_channels=out_channels,
                num_layers=config_dict.get("num_layers", 19),
                num_single_layers=config_dict.get("num_single_layers", 38),
                attention_head_dim=config_dict.get("attention_head_dim", 128),
                num_attention_heads=config_dict.get("num_attention_heads", 24),
                joint_attention_dim=config_dict.get("joint_attention_dim", 4096),
                pooled_projection_dim=config_dict.get("pooled_projection_dim", 768),
                guidance_embeds=config_dict.get("guidance_embeds", True),
                axes_dims_rope=tuple(config_dict.get("axes_dims_rope", [16, 56, 56])),
                qkv_mlp_ratio=config_dict.get("qkv_mlp_ratio", 4.0),
                text_projection_dim=config_dict.get("text_projection_dim"),
                image_projection_dim=config_dict.get("image_projection_dim"),
                dual_input_mode=self.pipeline_config.supports_dual_input,
                vae_scale_factor=config_dict.get("vae_scale_factor", 8),
                latent_channels=config_dict.get("latent_channels", 16),
                scaling_factor=config_dict.get("scaling_factor", 0.3611),
                model_type="flux_kontext",
            )
        else:
            # Use default configuration for Kontext
            return FluxTransformerConfig(
                dual_input_mode=self.pipeline_config.supports_dual_input,
                model_type="flux_kontext",
            )

    def _load_components(self):
        """Load pipeline components using configuration."""
        # Load scheduler
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.pipeline_config.model_path,
            subfolder=self.pipeline_config.scheduler_subfolder,
            local_files_only=self.pipeline_config.local_files_only,
        )
        
        # Load tokenizers
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.pipeline_config.model_path,
            subfolder=self.pipeline_config.tokenizer_subfolder,
            local_files_only=self.pipeline_config.local_files_only,
        )
        self.tokenizer_2 = T5TokenizerFast.from_pretrained(
            self.pipeline_config.model_path,
            subfolder=self.pipeline_config.tokenizer_2_subfolder,
            local_files_only=self.pipeline_config.local_files_only,
        )
        
        # Load text encoders
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.pipeline_config.model_path,
            subfolder=self.pipeline_config.text_encoder_subfolder,
            local_files_only=self.pipeline_config.local_files_only,
        ).to(self.device)
        self.text_encoder_2 = T5EncoderModel.from_pretrained(
            self.pipeline_config.model_path,
            subfolder=self.pipeline_config.text_encoder_2_subfolder,
            local_files_only=self.pipeline_config.local_files_only,
        ).to(self.device)
        
        # Load image encoder if supported
        if self.pipeline_config.supports_dual_input and self.pipeline_config.image_encoder_subfolder:
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                self.pipeline_config.model_path,
                subfolder=self.pipeline_config.image_encoder_subfolder,
                local_files_only=self.pipeline_config.local_files_only,
            ).to(self.device)
            self.feature_extractor = CLIPImageProcessor.from_pretrained(
                self.pipeline_config.model_path,
                subfolder=self.pipeline_config.feature_extractor_subfolder,
                local_files_only=self.pipeline_config.local_files_only,
            ) if self.pipeline_config.feature_extractor_subfolder else None
        else:
            self.image_encoder = None
            self.feature_extractor = None
        
        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(
            self.pipeline_config.model_path,
            subfolder=self.pipeline_config.vae_subfolder,
            local_files_only=self.pipeline_config.local_files_only,
        ).to(self.device)
        
        # Load and configure transformer
        self.transformer_config = self._load_transformer_config()
        self.transformer = FluxTransformer2DModel(self.transformer_config)

    def encode_image(self, image: PIL.Image.Image, device: Union[str, torch.device], dtype: torch.dtype) -> torch.Tensor:
        """Encode input image to latent space."""
        if self.vae is None:
            raise ValueError("VAE is required for image encoding")
        
        # Preprocess image
        if isinstance(image, PIL.Image.Image):
            # Convert to tensor and normalize
            image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            image_tensor = (image_tensor * 2.0 - 1.0).unsqueeze(0)  # Normalize to [-1, 1]
        else:
            image_tensor = image
        
        # Encode to latent space
        image_tensor = image_tensor.to(device=device, dtype=dtype)
        with torch.no_grad():
            latents = self.vae.encode(image_tensor).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        
        return latents

    def encode_text(
        self,
        prompt: Union[str, List[str]],
        device: Union[str, torch.device],
        dtype: torch.dtype,
        max_sequence_length: int = 512,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode text prompt to embeddings."""
        if self.text_encoder_2 is None or self.tokenizer_2 is None:
            raise ValueError("T5 text encoder and tokenizer are required for text encoding")
        
        if isinstance(prompt, str):
            prompt = [prompt]
        
        # Tokenize with T5
        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        
        # Encode with T5
        with torch.no_grad():
            prompt_embeds = self.text_encoder_2(text_input_ids)[0]
        
        # Get pooled projections (simplified version)
        pooled_prompt_embeds = prompt_embeds.mean(dim=1)
        
        return prompt_embeds, pooled_prompt_embeds

    def encode_image_features(
        self,
        image: PIL.Image.Image,
        device: Union[str, torch.device],
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input image to CLIP features for dual input."""
        if self.image_encoder is None or self.feature_extractor is None:
            raise ValueError("Image encoder and feature extractor are required for image feature encoding")
        
        # Preprocess image for CLIP
        if isinstance(image, PIL.Image.Image):
            image_inputs = self.feature_extractor(images=image, return_tensors="pt")
            image_tensor = image_inputs.pixel_values.to(device=device, dtype=dtype)
        else:
            image_tensor = image
        
        # Encode with CLIP
        with torch.no_grad():
            image_embeds = self.image_encoder(image_tensor)
            if hasattr(image_embeds, 'image_embeds'):
                image_features = image_embeds.image_embeds
            else:
                image_features = image_embeds.last_hidden_state
        
        # Create pooled projections (simplified)
        pooled_image_features = image_features.mean(dim=1) if image_features.dim() > 2 else image_features
        
        return image_features, pooled_image_features

    def denoise(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        image_features: Optional[torch.Tensor] = None,
        pooled_image_features: Optional[torch.Tensor] = None,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Perform denoising with the transformer."""
        
        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=latents.device)
        timesteps = self.scheduler.timesteps
        
        # Prepare guidance tensor if needed
        guidance = torch.tensor([guidance_scale], device=latents.device, dtype=latents.dtype)
        
        # Denoising loop
        for i, t in enumerate(timesteps):
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
            
            # Prepare timestep
            timestep = t.expand(latent_model_input.shape[0])
            
            # Predict noise
            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                image_pooled_projections=pooled_image_features,
                guidance=guidance,
                return_dict=False,
            )
            
            # Apply classifier-free guidance
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Compute previous noisy sample
            latents = self.scheduler.step(noise_pred, t, latents, generator=generator).prev_sample
        
        return latents

    def decode_latents(self, latents: torch.Tensor) -> PIL.Image.Image:
        """Decode latents back to image space."""
        if self.vae is None:
            raise ValueError("VAE is required for latent decoding")
        
        # Scale latents
        latents = latents / self.vae.config.scaling_factor
        
        # Decode
        with torch.no_grad():
            image = self.vae.decode(latents).sample
        
        # Convert to PIL Image
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).round().astype("uint8")
        
        if image.shape[0] == 1:
            return PIL.Image.fromarray(image[0])
        else:
            return [PIL.Image.fromarray(img) for img in image]

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """
        Load weights into the pipeline components.
        
        Args:
            weights: Iterable of (name, tensor) tuples
            
        Returns:
            Set of loaded weight names
        """
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    def forward(
        self,
        req: OmniDiffusionRequest,
    ) -> DiffusionOutput:
        """
        Forward pass for image editing with text guidance.
        
        Args:
            req: OmniDiffusionRequest containing all parameters
            
        Returns:
            DiffusionOutput with generated image
        """
        from vllm_omni.diffusion.data import DiffusionOutput
        
        # Extract parameters from request
        image = getattr(req, 'image', None)
        prompt = getattr(req, 'prompt', '')
        negative_prompt = getattr(req, 'negative_prompt', '')
        num_inference_steps = getattr(req, 'num_inference_steps', self.default_num_inference_steps)
        guidance_scale = getattr(req, 'guidance_scale', self.default_guidance_scale)
        image_strength = getattr(req, 'image_strength', self.default_image_strength)
        generator = getattr(req, 'generator', None)
        
        # Use preprocessed data if available
        if hasattr(req, 'preprocessed_image') and req.preprocessed_image is not None:
            image = req.preprocessed_image
        if hasattr(req, 'preprocessed_prompt') and req.preprocessed_prompt is not None:
            prompt = req.preprocessed_prompt
        if hasattr(req, 'preprocessed_num_inference_steps') and req.preprocessed_num_inference_steps is not None:
            num_inference_steps = req.preprocessed_num_inference_steps
        if hasattr(req, 'preprocessed_guidance_scale') and req.preprocessed_guidance_scale is not None:
            guidance_scale = req.preprocessed_guidance_scale
        if hasattr(req, 'preprocessed_image_strength') and req.preprocessed_image_strength is not None:
            image_strength = req.preprocessed_image_strength
        if hasattr(req, 'preprocessed_generator') and req.preprocessed_generator is not None:
            generator = req.preprocessed_generator
        
        # Validate input
        if image is None:
            raise ValueError("FLUX.1-Kontext requires an input image for editing")
        
        # Convert image to PIL if needed
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, torch.Tensor):
            # Convert tensor to PIL
            img_tensor = (image.clamp(-1, 1) + 1) / 2 * 255
            img_tensor = img_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            image = Image.fromarray(img_tensor)
        
        # Get the dtype from transformer to ensure compatibility
        transformer_dtype = next(self.transformer.parameters()).dtype
        
        # Encode input image to latents with transformer dtype
        latents = self.encode_image(image, self.device, transformer_dtype)
        
        # Encode text prompts with transformer dtype
        prompt_embeds, pooled_prompt_embeds = self.encode_text(prompt, self.device, transformer_dtype)
        
        # Encode image features for dual input with transformer dtype
        image_features, pooled_image_features = self.encode_image_features(image, self.device, transformer_dtype)
        
        # Apply image strength (how much to preserve original)
        if image_strength < 1.0:
            # Add noise to latents based on image strength
            noise = randn_tensor(latents.shape, generator=generator, device=latents.device, dtype=latents.dtype)
            latents = latents * image_strength + noise * (1.0 - image_strength)
        
        # Perform denoising
        denoised_latents = self.denoise(
            latents=latents,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            image_features=image_features,
            pooled_image_features=pooled_image_features,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        
        # Decode latents back to image
        output_image = self.decode_latents(denoised_latents)
        
        # Convert to tensor for DiffusionOutput
        if isinstance(output_image, list):
            # Handle list of images
            images = []
            for img in output_image:
                if isinstance(img, Image.Image):
                    img_array = np.array(img)
                    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
                    images.append(img_tensor)
            output_tensor = torch.stack(images)
        elif isinstance(output_image, Image.Image):
            img_array = np.array(output_image)
            output_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
            output_tensor = output_tensor.unsqueeze(0)
        else:
            output_tensor = output_image
        
        return DiffusionOutput(output=output_tensor)