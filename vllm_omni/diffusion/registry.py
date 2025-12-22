# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib

from vllm.logger import init_logger
from vllm.model_executor.models.registry import _LazyRegisteredModel, _ModelRegistry

from vllm_omni.diffusion.data import OmniDiffusionConfig

logger = init_logger(__name__)

_DIFFUSION_MODELS = {
    # arch:(mod_folder, mod_relname, cls_name)
    "QwenImagePipeline": (
        "qwen_image",
        "pipeline_qwen_image",
        "QwenImagePipeline",
    ),
    "QwenImageEditPipeline": (
        "qwen_image",
        "pipeline_qwen_image_edit",
        "QwenImageEditPipeline",
    ),
    "QwenImageEditPlusPipeline": (
        "qwen_image",
        "pipeline_qwen_image_edit_plus",
        "QwenImageEditPlusPipeline",
    ),
    "QwenImageLayeredPipeline": (
        "qwen_image",
        "pipeline_qwen_image_layered",
        "QwenImageLayeredPipeline",
    ),
    "ZImagePipeline": (
        "z_image",
        "pipeline_z_image",
        "ZImagePipeline",
    ),
    "OvisImagePipeline": (
        "ovis_image",
        "pipeline_ovis_image",
        "OvisImagePipeline",
    ),
    "WanPipeline": (
        "wan2_2",
        "pipeline_wan2_2",
        "Wan22Pipeline",
    ),
    "LongCatImagePipeline": (
        "longcat_image",
        "pipeline_longcat_image",
        "LongCatImagePipeline",
    ),
}


DiffusionModelRegistry = _ModelRegistry(
    {
        model_arch: _LazyRegisteredModel(
            module_name=f"vllm_omni.diffusion.models.{mod_folder}.{mod_relname}",
            class_name=cls_name,
        )
        for model_arch, (mod_folder, mod_relname, cls_name) in _DIFFUSION_MODELS.items()
    }
)


def initialize_model(
    od_config: OmniDiffusionConfig,
):
    model_class = DiffusionModelRegistry._try_load_model_cls(od_config.model_class_name)
    if model_class is not None:
        model = model_class(od_config=od_config)
        # Configure VAE memory optimization settings from config
        if hasattr(model.vae, "use_slicing"):
            model.vae.use_slicing = od_config.vae_use_slicing
        if hasattr(model.vae, "use_tiling"):
            model.vae.use_tiling = od_config.vae_use_tiling
        
        # Apply CPU offload based on configuration
        _apply_cpu_offload(model, od_config)
        
        return model
    else:
        raise ValueError(f"Model class {od_config.model_class_name} not found in diffusion model registry.")


def _apply_cpu_offload(model, od_config: OmniDiffusionConfig):
    """Apply CPU offload to model components based on configuration."""
    import torch
    
    # Set pin memory for CPU tensors if enabled
    pin_memory_device = None
    if od_config.pin_cpu_memory and torch.cuda.is_available():
        pin_memory_device = "cuda:0"  # Pin to first CUDA device for faster transfers
        logger.info("Pin memory enabled for CPU offloaded tensors")
    
    # Offload VAE to CPU if enabled
    if hasattr(model, 'vae') and od_config.vae_cpu_offload:
        if hasattr(model.vae, 'to'):
            model.vae.to("cpu", memory_format=torch.preserve_format)
            if pin_memory_device and hasattr(model.vae, 'pin_memory'):
                model.vae.pin_memory(device=pin_memory_device)
            logger.info("VAE offloaded to CPU")
    
    # Offload text encoder to CPU if enabled
    if hasattr(model, 'text_encoder') and od_config.text_encoder_cpu_offload:
        if hasattr(model.text_encoder, 'to'):
            model.text_encoder.to("cpu", memory_format=torch.preserve_format)
            if pin_memory_device and hasattr(model.text_encoder, 'pin_memory'):
                model.text_encoder.pin_memory(device=pin_memory_device)
            logger.info("Text encoder offloaded to CPU")
    
    # Offload image encoder to CPU if enabled
    if hasattr(model, 'image_encoder') and od_config.image_encoder_cpu_offload:
        if hasattr(model.image_encoder, 'to'):
            model.image_encoder.to("cpu", memory_format=torch.preserve_format)
            if pin_memory_device and hasattr(model.image_encoder, 'pin_memory'):
                model.image_encoder.pin_memory(device=pin_memory_device)
            logger.info("Image encoder offloaded to CPU")
    
    # Offload DiT/Transformer to CPU if enabled
    if hasattr(model, 'transformer') and od_config.dit_cpu_offload:
        if hasattr(model.transformer, 'to'):
            model.transformer.to("cpu", memory_format=torch.preserve_format)
            if pin_memory_device and hasattr(model.transformer, 'pin_memory'):
                model.transformer.pin_memory(device=pin_memory_device)
            logger.info("DiT/Transformer offloaded to CPU")
    elif hasattr(model, 'transformer_2') and od_config.dit_cpu_offload:
        if hasattr(model.transformer_2, 'to'):
            model.transformer_2.to("cpu", memory_format=torch.preserve_format)
            if pin_memory_device and hasattr(model.transformer_2, 'pin_memory'):
                model.transformer_2.pin_memory(device=pin_memory_device)
            logger.info("DiT/Transformer_2 offloaded to CPU")


_DIFFUSION_POST_PROCESS_FUNCS = {
    # arch: post_process_func
    # `post_process_func` function must be placed in {mod_folder}/{mod_relname}.py,
    # where mod_folder and mod_relname are  defined and mapped using `_DIFFUSION_MODELS` via the `arch` key
    "QwenImagePipeline": "get_qwen_image_post_process_func",
    "QwenImageEditPipeline": "get_qwen_image_edit_post_process_func",
    "QwenImageEditPlusPipeline": "get_qwen_image_edit_plus_post_process_func",
    "ZImagePipeline": "get_post_process_func",
    "OvisImagePipeline": "get_ovis_image_post_process_func",
    "WanPipeline": "get_wan22_post_process_func",
    "LongCatImagePipeline": "get_longcat_image_post_process_func",
}

_DIFFUSION_PRE_PROCESS_FUNCS = {
    # arch: pre_process_func
    # `pre_process_func` function must be placed in {mod_folder}/{mod_relname}.py,
    # where mod_folder and mod_relname are  defined and mapped using `_DIFFUSION_MODELS` via the `arch` key
    "QwenImageEditPipeline": "get_qwen_image_edit_pre_process_func",
    "QwenImageEditPlusPipeline": "get_qwen_image_edit_plus_pre_process_func",
    "QwenImageLayeredPipeline": "get_qwen_image_layered_pre_process_func",
}


def _load_process_func(od_config: OmniDiffusionConfig, func_name: str):
    """Load and return a process function from the appropriate module."""
    mod_folder, mod_relname, _ = _DIFFUSION_MODELS[od_config.model_class_name]
    module_name = f"vllm_omni.diffusion.models.{mod_folder}.{mod_relname}"
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)
    return func(od_config)


def get_diffusion_post_process_func(od_config: OmniDiffusionConfig):
    if od_config.model_class_name not in _DIFFUSION_POST_PROCESS_FUNCS:
        return None
    func_name = _DIFFUSION_POST_PROCESS_FUNCS[od_config.model_class_name]
    return _load_process_func(od_config, func_name)


def get_diffusion_pre_process_func(od_config: OmniDiffusionConfig):
    if od_config.model_class_name not in _DIFFUSION_PRE_PROCESS_FUNCS:
        return None  # Return None if no pre-processing function is registered (for backward compatibility)
    func_name = _DIFFUSION_PRE_PROCESS_FUNCS[od_config.model_class_name]
    return _load_process_func(od_config, func_name)
