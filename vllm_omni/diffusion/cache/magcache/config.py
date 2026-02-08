# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Optional, Union

import torch


@dataclass
class MagCacheConfig:
    """
    Configuration for MagCache applied to transformer models.

    MagCache (Magnitude-based Cache) is an adaptive caching technique that speeds up
    diffusion model inference by reusing transformer block computations based on
    magnitude ratios between consecutive timesteps.

    Reference: https://github.com/Zehong-Ma/MagCache

    Args:
        threshold: The threshold for the accumulated error. If the accumulated error
            is below this threshold, the block computation is skipped. A higher threshold
            allows for more aggressive skipping (faster) but may degrade quality.
            Default: 0.06
        max_skip_steps: The maximum number of consecutive steps that can be skipped (K).
            Default: 3
        retention_ratio: The fraction of initial steps during which skipping is disabled
            to ensure stability. For example, if num_inference_steps is 28 and
            retention_ratio is 0.2, the first 6 steps will never be skipped.
            Default: 0.2
        num_inference_steps: The number of inference steps used in the pipeline.
            This is required to interpolate mag_ratios correctly.
            Default: 28
        mag_ratios: The pre-computed magnitude ratios for the model. These are
            checkpoint-dependent. If not provided, you must set calibrate=True to
            calculate them for your specific model. For Flux models, you can use
            FLUX_MAG_RATIOS.
            Default: None
        calibrate: If True, enables calibration mode. In this mode, no blocks are skipped.
            Instead, the hook calculates the magnitude ratios for the current run and logs
            them at the end. Use this to obtain mag_ratios for new models or schedulers.
            Default: False
        transformer_type: Transformer class name for logging and identification.
            Auto-detected from pipeline.transformer.__class__.__name__ in backend.
            Default: "FluxTransformer2DModel"
    """

    threshold: float = 0.06
    max_skip_steps: int = 3
    retention_ratio: float = 0.2
    num_inference_steps: int = 28
    mag_ratios: Optional[Union[torch.Tensor, list[float]]] = None
    calibrate: bool = False
    transformer_type: str = "FluxTransformer2DModel"

    def __post_init__(self) -> None:
        """Validate and set default coefficients."""
        if self.threshold <= 0:
            raise ValueError(f"threshold must be positive, got {self.threshold}")

        if self.max_skip_steps <= 0:
            raise ValueError(f"max_skip_steps must be positive, got {self.max_skip_steps}")

        if not 0 < self.retention_ratio < 1:
            raise ValueError(f"retention_ratio must be in (0, 1), got {self.retention_ratio}")

        if self.num_inference_steps is None:
            raise ValueError(
                "num_inference_steps must be provided for MagCache. "
                "This is required to determine retention steps and interpolate mag_ratios. "
                "For Flux models, use num_inference_steps=28."
            )

        if self.num_inference_steps <= 0:
            raise ValueError(f"num_inference_steps must be positive, got {self.num_inference_steps}")

        if not self.calibrate and self.mag_ratios is None:
            raise ValueError(
                "mag_ratios must be provided for MagCache inference because these ratios "
                "are model-dependent. To get them for your model:\n"
                "1. Initialize MagCacheConfig(calibrate=True, ...)\n"
                "2. Run inference on your model once.\n"
                "3. Copy the printed ratios array and pass it to mag_ratios in the config.\n"
                "For Flux models, you can import FLUX_MAG_RATIOS from vllm_omni.diffusion.cache.magcache.strategy."
            )

        if not self.calibrate and self.mag_ratios is not None:
            if not torch.is_tensor(self.mag_ratios):
                self.mag_ratios = torch.tensor(self.mag_ratios)


FLUX_MAG_RATIOS = None


def get_flux_mag_ratios() -> torch.Tensor:
    """Get FLUX_MAG_RATIOS from FluxMagCacheStrategy, importing only when needed."""
    global FLUX_MAG_RATIOS
    if FLUX_MAG_RATIOS is None:
        from vllm_omni.diffusion.cache.magcache.strategy import FluxMagCacheStrategy

        FLUX_MAG_RATIOS = FluxMagCacheStrategy.FLUX_MAG_RATIOS
    return FLUX_MAG_RATIOS
