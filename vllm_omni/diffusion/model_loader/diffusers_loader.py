# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import dataclasses
import glob
import os
import time
from collections.abc import Generator, Iterable
from pathlib import Path
from typing import cast

import torch
from torch import nn
from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import (
    download_safetensors_index_file_from_hf,
    download_weights_from_hf,
    filter_duplicate_safetensors_files,
    filter_files_not_needed_for_inference,
    maybe_download_from_modelscope,
    safetensors_weights_iterator,
)
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.registry import initialize_model

logger = init_logger(__name__)


MODEL_INDEX = "model_index.json"
DIFFUSION_MODEL_WEIGHTS_INDEX = "diffusion_pytorch_model.safetensors.index.json"


class DiffusersPipelineLoader:
    """Model loader that can load diffusers pipeline components from disk."""

    # default number of thread when enable multithread weight loading
    DEFAULT_NUM_THREADS = 8

    @dataclasses.dataclass
    class ComponentSource:
        """A source for weights."""

        model_or_path: str
        """The model ID or path."""

        subfolder: str | None
        """The subfolder inside the model repo."""

        revision: str | None
        """The optional model revision."""

        prefix: str = ""
        """A prefix to prepend to all weights."""

        fall_back_to_pt: bool = True
        """Whether .pt weights can be used."""

        allow_patterns_overrides: list[str] | None = None
        """If defined, weights will load exclusively using these patterns."""

    counter_before_loading_weights: float = 0.0
    counter_after_loading_weights: float = 0.0

    def __init__(self, load_config: LoadConfig):
        self.load_config = load_config

        # TODO(Isotr0py): Enable multithreaded weight loading
        # extra_config = load_config.model_loader_extra_config
        # allowed_keys = {"enable_multithread_load", "num_threads"}
        # unexpected_keys = set(extra_config.keys()) - allowed_keys

        # if unexpected_keys:
        #     raise ValueError(
        #         f"Unexpected extra config keys for load format {load_config.load_format}: {unexpected_keys}"
        #     )

    def _prepare_weights(
        self,
        model_name_or_path: Path,
        subfolder: str | None,
        revision: str | None,
        fall_back_to_pt: bool,
        allow_patterns_overrides: list[str] | None,
    ) -> tuple[str, list[str], bool]:
        """Prepare weights for the model.

        If the model is not local, it will be downloaded."""
        model_name_or_path = maybe_download_from_modelscope(model_name_or_path, revision) or model_name_or_path

        is_local = os.path.isdir(model_name_or_path)
        load_format = self.load_config.load_format
        use_safetensors = False
        index_file = DIFFUSION_MODEL_WEIGHTS_INDEX

        # only hf is supported currently
        if load_format == "auto":
            load_format = "hf"

        # Some quantized models use .pt files for storing the weights.
        if load_format == "hf":
            allow_patterns = ["*.safetensors", "*.bin"]
        else:
            raise ValueError(f"Unknown load_format: {load_format}")

        if fall_back_to_pt:
            allow_patterns += ["*.pt"]

        if allow_patterns_overrides is not None:
            allow_patterns = allow_patterns_overrides

        if subfolder is not None:
            allow_patterns = [f"{subfolder}/{pattern}" for pattern in allow_patterns]

        if not is_local:
            hf_folder = download_weights_from_hf(
                model_name_or_path,
                self.load_config.download_dir,
                allow_patterns,
                revision,
                ignore_patterns=self.load_config.ignore_patterns,
            )
        else:
            hf_folder = model_name_or_path

        hf_weights_files: list[str] = []
        for pattern in allow_patterns:
            hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))
            if len(hf_weights_files) > 0:
                if pattern == "*.safetensors":
                    use_safetensors = True
                break

        if use_safetensors:
            # For models like Mistral-7B-Instruct-v0.3
            # there are both sharded safetensors files and a consolidated
            # safetensors file. Using both breaks.
            # Here, we download the `model.safetensors.index.json` and filter
            # any files not found in the index.
            if not is_local:
                download_safetensors_index_file_from_hf(
                    model_name_or_path,
                    index_file,
                    self.load_config.download_dir,
                    revision,
                )
            hf_weights_files = filter_duplicate_safetensors_files(hf_weights_files, hf_folder, index_file)
        else:
            hf_weights_files = filter_files_not_needed_for_inference(hf_weights_files)

        if len(hf_weights_files) == 0:
            raise RuntimeError(f"Cannot find any model weights with `{model_name_or_path}`")

        return hf_folder, hf_weights_files, use_safetensors

    def _get_weights_iterator(self, source: "ComponentSource") -> Generator[tuple[str, torch.Tensor], None, None]:
        """Get an iterator for the model weights based on the load format."""
        hf_folder, hf_weights_files, use_safetensors = self._prepare_weights(
            source.model_or_path,
            source.subfolder,
            source.revision,
            source.fall_back_to_pt,
            source.allow_patterns_overrides,
        )
        weights_iterator = safetensors_weights_iterator(
            hf_weights_files,
            self.load_config.use_tqdm_on_load,
            self.load_config.safetensors_load_strategy,
        )

        if self.counter_before_loading_weights == 0.0:
            self.counter_before_loading_weights = time.perf_counter()
        # Apply the prefix.
        return ((source.prefix + name, tensor) for (name, tensor) in weights_iterator)

    def get_all_weights(
        self,
        model: nn.Module,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        sources = cast(
            Iterable[DiffusersPipelineLoader.ComponentSource],
            getattr(model, "weights_sources", ()),
        )
        for source in sources:
            yield from self._get_weights_iterator(source)

    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_weights(
            model_config.model,
            model_config.revision,
            fall_back_to_pt=True,
            allow_patterns_overrides=None,
        )

    def _log_memory_usage(self, stage: str, device: str = None):
        """Log current GPU memory usage."""
        if torch.cuda.is_available():
            if device is None:
                device = torch.cuda.current_device()
            else:
                device = torch.device(device).index or 0
            
            allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(device) / 1024**3   # GB
            total = torch.cuda.get_device_properties(device).total_memory / 1024**3  # GB
            
            logger.info(f"[{stage}] GPU {device} memory: allocated={allocated:.2f}GB, "
                       f"reserved={reserved:.2f}GB, total={total:.2f}GB, "
                       f"free={total-reserved:.2f}GB")
    
    def _get_optimal_pin_device(self, target_device: str) -> str | None:
        """Get the optimal device for pin_memory based on availability.
        
        Returns:
            str: Optimal CUDA device identifier (e.g., 'cuda:0', 'cuda:1')
            None: If no suitable device is found or CUDA is not available
        """
        if not torch.cuda.is_available():
            logger.debug("CUDA not available, pin_memory disabled")
            return None
            
        try:
            target_device_idx = torch.device(target_device).index or 0
            
            # First try the target device
            try:
                total_memory = torch.cuda.get_device_properties(target_device_idx).total_memory
                allocated_memory = torch.cuda.memory_allocated(target_device_idx)
                memory_usage_ratio = allocated_memory / total_memory
                
                logger.debug(f"Target device cuda:{target_device_idx} memory usage: {memory_usage_ratio:.1%}")
                if memory_usage_ratio < 0.9:  # Less than 90% used
                    logger.debug(f"Selected target device cuda:{target_device_idx} for pin_memory")
                    return f"cuda:{target_device_idx}"
            except Exception as e:
                logger.warning(f"Failed to check target device cuda:{target_device_idx} memory: {e}")
            
            # Try other available devices, sorted by memory usage
            device_candidates = []
            for i in range(torch.cuda.device_count()):
                if i == target_device_idx:
                    continue
                try:
                    total_memory = torch.cuda.get_device_properties(i).total_memory
                    allocated_memory = torch.cuda.memory_allocated(i)
                    memory_usage_ratio = allocated_memory / total_memory
                    device_candidates.append((memory_usage_ratio, i))
                    logger.debug(f"Device cuda:{i} memory usage: {memory_usage_ratio:.1%}")
                except Exception as e:
                    logger.warning(f"Failed to check device cuda:{i} memory: {e}")
                    continue
            
            # Sort by memory usage and select the least used device
            device_candidates.sort(key=lambda x: x[0])
            for memory_ratio, device_idx in device_candidates:
                if memory_ratio < 0.5:  # Less than 50% used
                    logger.debug(f"Selected alternative device cuda:{device_idx} for pin_memory")
                    return f"cuda:{device_idx}"
            
            # If no device meets our criteria, check if we can use any device at all
            if device_candidates:
                # Use the least loaded device, even if it's >50% used
                best_device_idx = device_candidates[0][1]
                logger.warning(f"All devices are heavily loaded, using least loaded device cuda:{best_device_idx}")
                return f"cuda:{best_device_idx}"
            
            # Last resort: if target device exists, use it regardless of memory usage
            if target_device_idx < torch.cuda.device_count():
                logger.warning(f"Using target device cuda:{target_device_idx} despite high memory usage")
                return f"cuda:{target_device_idx}"
                
        except Exception as e:
            logger.error(f"Error in optimal device selection: {e}")
        
        logger.error("No suitable CUDA device found for pin_memory")
        return None

    def load_model(self, od_config: OmniDiffusionConfig, load_device: str) -> nn.Module:
        """Load a model with the given configurations."""
        target_device = torch.device(load_device)
        
        # Log initial memory usage
        self._log_memory_usage("Initial")
        
        # Set pin memory for CPU tensors if enabled
        pin_memory_device = None
        if od_config.pin_cpu_memory and torch.cuda.is_available():
            pin_memory_device = self._get_optimal_pin_device(load_device)
            logger.info("Pin memory enabled for CPU offloaded tensors (device: %s)", pin_memory_device)
        
        # For components that will be offloaded to CPU, load them to CPU first
        # then move to target device if needed during initialization
        
        with set_default_torch_dtype(od_config.dtype):
            # Initialize model on CPU to avoid GPU memory issues
            logger.info("Initializing model on CPU to avoid GPU memory issues...")
            with torch.device("cpu"):
                model = initialize_model(od_config)
            
            logger.debug("Loading weights on %s ...", load_device)
            # Quantization does not happen in `load_weights` but after it
            self.load_weights(model)
            
            # Log memory after weight loading
            self._log_memory_usage("After weight loading")
        
        # Apply CPU offload and device placement based on configuration
        logger.info("Applying CPU offload and device placement to model components...")
        
        # Define component configurations
        component_configs = {
            'vae': od_config.vae_cpu_offload,
            'text_encoder': od_config.text_encoder_cpu_offload,
            'image_encoder': od_config.image_encoder_cpu_offload,
            'transformer': od_config.dit_cpu_offload,
            'transformer_2': od_config.dit_cpu_offload,
        }
        
        # Track processed components for memory monitoring
        processed_components = []
        
        # Process each component with error handling
        for attr_name, should_offload in component_configs.items():
            if hasattr(model, attr_name):
                attr = getattr(model, attr_name)
                if attr is not None and hasattr(attr, 'to') and hasattr(attr, 'parameters'):
                    try:
                        if should_offload:
                            # Move to CPU and optionally pin memory
                            logger.debug("Offloading %s to CPU...", attr_name)
                            attr.to("cpu", memory_format=torch.preserve_format)
                            
                            if pin_memory_device and hasattr(attr, 'pin_memory'):
                                try:
                                    if pin_memory_device:  # Only pin if we have a valid device
                                        attr.pin_memory(pin_memory_device)
                                        logger.debug("%s pinned to %s", attr_name, pin_memory_device)
                                    else:
                                        logger.debug("No suitable device for pin_memory, skipping")
                                except Exception as pin_error:
                                    logger.warning("Failed to pin %s memory: %s. Continuing without pinning.", 
                                                 attr_name, str(pin_error))
                            
                            logger.info("%s offloaded to CPU", attr_name.title().replace('_', ' '))
                        else:
                            # Move to target device
                            logger.debug("Moving %s to target device %s...", attr_name, target_device)
                            attr.to(target_device)
                            logger.info("%s moved to target device", attr_name.title().replace('_', ' '))
                        
                        processed_components.append(attr_name)
                        
                    except torch.cuda.OutOfMemoryError as e:
                        logger.error("CUDA out of memory when processing %s: %s", attr_name, str(e))
                        if should_offload:
                            logger.info("Attempting to offload %s to CPU as fallback", attr_name)
                            try:
                                attr.to("cpu", memory_format=torch.preserve_format)
                                logger.info("%s fallback offloaded to CPU", attr_name)
                            except Exception as fallback_error:
                                logger.error("Failed to offload %s even to CPU: %s", 
                                           attr_name, str(fallback_error))
                                raise
                        else:
                            raise
                    except Exception as e:
                        logger.error("Error processing %s: %s", attr_name, str(e))
                        raise
        
        # Log memory usage after component processing
        if processed_components:
            self._log_memory_usage(f"After processing {', '.join(processed_components)}")
        
        # Log final memory summary
        self._log_memory_usage("Final")
        
        # Evaluate model and return
        model.eval()
        logger.info("Model loading and offloading completed successfully")
        
        return model

    def load_weights(self, model: nn.Module) -> None:
        weights_to_load = {name for name, _ in model.named_parameters()}
        loaded_weights = model.load_weights(self.get_all_weights(model))

        self.counter_after_loading_weights = time.perf_counter()
        logger.info_once(
            "Loading weights took %.2f seconds",
            self.counter_after_loading_weights - self.counter_before_loading_weights,
        )
        # TODO(Isotr0py): Enable weights loading check after decoupling
        # all components' weights loading (AutoModel.from_pretrained etc).
        # We only enable strict check for non-quantized models
        # that have loaded weights tracking currently.
        if loaded_weights is not None:
            _ = weights_to_load - loaded_weights
        #     if weights_not_loaded:
        #         raise ValueError(
        #             "Following weights were not initialized from "
        #             f"checkpoint: {weights_not_loaded}"
        #         )
