from types import SimpleNamespace

import pytest
import torch
from PIL import Image

import vllm_omni.diffusion.models.flux2_klein.pipeline_flux2_klein_kv as kv_pipeline_module
from vllm_omni.diffusion.models.flux2_klein.flux2_klein_transformer import Flux2Attention
from vllm_omni.diffusion.models.flux2_klein.pipeline_flux2_klein_kv import Flux2KleinKVPipeline

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _DummyKVCache:
    def store(self, *args, **kwargs):
        pass


class _DummyAttention:
    def __call__(self, query, key, value, attn_metadata=None):
        del key, value, attn_metadata
        return query


class _DummyLinear:
    def __init__(self, output):
        self.output = output

    def __call__(self, *args, **kwargs):
        del args, kwargs
        return self.output


class _DummyRope:
    def __call__(self, x, *args, **kwargs):
        del args, kwargs
        return x


class _DummyKVConfig:
    sequence_parallel_size = 1


def test_flux2_attention_kv_extract_handles_encoder_tokens_without_shape_mismatch():
    attn = object.__new__(Flux2Attention)
    attn.added_kv_proj_dim = 8
    attn.query_num_heads = 2
    attn.kv_num_heads = 2
    attn.head_dim = 4
    attn.add_query_num_heads = 2
    attn.add_kv_num_heads = 2
    attn.to_qkv = _DummyLinear((torch.arange(72, dtype=torch.float32).reshape(1, 3, 24), None))
    attn.add_kv_proj = _DummyLinear((torch.arange(48, dtype=torch.float32).reshape(1, 2, 24), None))
    attn.norm_q = lambda x: x
    attn.norm_k = lambda x: x
    attn.norm_added_q = lambda x: x
    attn.norm_added_k = lambda x: x
    attn.rope = _DummyRope()
    attn.attn = _DummyAttention()
    attn.to_add_out = lambda x: x
    attn.to_out = [lambda x: x, lambda x: x]

    query = torch.zeros(1, 3, 8)
    encoder_hidden_states = torch.zeros(1, 2, 8)
    output = attn.forward(
        hidden_states=query,
        encoder_hidden_states=encoder_hidden_states,
        kv_cache=_DummyKVCache(),
        kv_cache_mode="extract",
        num_ref_tokens=1,
    )

    assert isinstance(output, tuple)
    assert output[0].shape[0] == 1


class _DummyImageProcessor:
    def __init__(self, resize_size: tuple[int, int] | None = None):
        self.resize_size = resize_size
        self.check_calls: list[tuple[int, int]] = []
        self.resize_calls: list[tuple[tuple[int, int], int]] = []
        self.preprocess_calls: list[tuple[tuple[int, int], int, int, str]] = []

    def check_image_input(self, image: Image.Image):
        self.check_calls.append(image.size)
        return image

    def _resize_to_target_area(self, image: Image.Image, target_area: int = 1024 * 1024):
        self.resize_calls.append((image.size, target_area))
        if self.resize_size is None:
            return image
        return image.resize(self.resize_size)

    def preprocess(self, image: Image.Image, height: int, width: int, resize_mode: str = "crop"):
        self.preprocess_calls.append((image.size, height, width, resize_mode))
        return torch.zeros(1, 3, height, width)


class _DummyTransformer:
    def __init__(self):
        self.config = SimpleNamespace(in_channels=16)
        self.dtype = torch.float32
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        hidden_states = kwargs["hidden_states"]
        output = SimpleNamespace(sample=torch.zeros_like(hidden_states))
        if kwargs.get("kv_cache_mode") == "extract":
            return output, object()
        return output


class _DummyScheduler:
    def __init__(self):
        self.begin_indexes = []

    def set_begin_index(self, begin_index: int):
        self.begin_indexes.append(begin_index)

    def step(self, noise_pred, t, latents, return_dict: bool = False):
        del noise_pred, t, return_dict
        return (latents,)


class _DummyVAE:
    def __init__(self):
        self.dtype = torch.float32
        self.bn = SimpleNamespace(running_mean=torch.zeros(1), running_var=torch.ones(1))
        self.config = SimpleNamespace(batch_norm_eps=1e-6)

    def decode(self, latents, return_dict: bool = False):
        del latents, return_dict
        return (torch.zeros(1, 3, 8, 8),)


def _make_request(
    *,
    image: Image.Image | None,
    height: int | None = None,
    width: int | None = None,
):
    sampling_params = SimpleNamespace(
        height=height,
        width=width,
        num_inference_steps=4,
        guidance_scale=None,
        generator=None,
    )
    prompts = [{"prompt": "a prompt", "multi_modal_data": {"image": image}}]
    return SimpleNamespace(prompts=prompts, sampling_params=sampling_params)


def _make_pipeline(*, resize_size: tuple[int, int] | None = None):
    pipeline = object.__new__(Flux2KleinKVPipeline)
    pipeline.default_sample_size = 128
    pipeline.vae_scale_factor = 8
    pipeline._execution_device = torch.device("cpu")
    pipeline.image_processor = _DummyImageProcessor(resize_size=resize_size)
    pipeline.transformer = _DummyTransformer()
    pipeline.scheduler = _DummyScheduler()
    pipeline.vae = _DummyVAE()
    pipeline.check_inputs = lambda *args, **kwargs: None
    pipeline.encode_prompt = lambda prompt, device, num_images_per_prompt, max_sequence_length: (
        torch.zeros(1, 2, 4),
        torch.zeros(1, 2, 4, dtype=torch.long),
    )

    pipeline.prepare_latents_calls = []

    def _prepare_latents(*, batch_size, num_latents_channels, height, width, dtype, device, generator):
        pipeline.prepare_latents_calls.append(
            {
                "batch_size": batch_size,
                "num_latents_channels": num_latents_channels,
                "height": height,
                "width": width,
                "dtype": dtype,
                "device": device,
                "generator": generator,
            }
        )
        latents = torch.zeros(1, 4, 1, dtype=dtype, device=device)
        latent_ids = torch.zeros(1, 4, 4, dtype=torch.long, device=device)
        return latents, latent_ids

    pipeline.prepare_latents = _prepare_latents

    pipeline.prepare_image_latents_calls = []

    def _prepare_image_latents(*, images, batch_size, generator, device, dtype):
        pipeline.prepare_image_latents_calls.append(
            {
                "batch_size": batch_size,
                "num_images": len(images),
                "shapes": [tuple(img.shape) for img in images],
                "generator": generator,
                "device": device,
                "dtype": dtype,
            }
        )
        image_latents = torch.zeros(1, 2, 1, dtype=dtype, device=device)
        image_latent_ids = torch.zeros(1, 2, 4, dtype=torch.long, device=device)
        return image_latents, image_latent_ids

    pipeline.prepare_image_latents = _prepare_image_latents
    pipeline._unpack_latents_with_ids = lambda latents, latent_ids: torch.zeros(1, 4, 2, 2, dtype=latents.dtype)
    pipeline._unpatchify_latents = lambda latents: latents
    return pipeline


def test_forward_derives_size_from_reference_image_when_dimensions_are_omitted(monkeypatch):
    pipeline = _make_pipeline()
    req = _make_request(image=Image.new("RGB", (640, 480)))

    monkeypatch.setattr(kv_pipeline_module, "retrieve_timesteps", lambda *args, **kwargs: ([torch.tensor(1.0)], 1))

    output = pipeline.forward(req)

    assert output.output.shape == (1, 3, 8, 8)
    assert pipeline.image_processor.check_calls == [(640, 480)]
    assert pipeline.image_processor.resize_calls == []
    assert pipeline.image_processor.preprocess_calls == [((640, 480), 480, 640, "crop")]
    assert pipeline.prepare_latents_calls[0]["height"] == 480
    assert pipeline.prepare_latents_calls[0]["width"] == 640


def test_forward_refreshes_reference_image_size_after_area_resize(monkeypatch):
    pipeline = _make_pipeline(resize_size=(1000, 500))
    req = _make_request(image=Image.new("RGB", (2001, 1001)))

    monkeypatch.setattr(kv_pipeline_module, "retrieve_timesteps", lambda *args, **kwargs: ([torch.tensor(1.0)], 1))

    output = pipeline.forward(req)

    assert output.output.shape == (1, 3, 8, 8)
    assert pipeline.image_processor.resize_calls == [((2001, 1001), 1024 * 1024)]
    assert pipeline.image_processor.preprocess_calls == [((1000, 500), 496, 992, "crop")]
    assert pipeline.prepare_latents_calls[0]["height"] == 496
    assert pipeline.prepare_latents_calls[0]["width"] == 992


def test_forward_keeps_explicit_dimensions_over_reference_image(monkeypatch):
    pipeline = _make_pipeline(resize_size=(1000, 500))
    req = _make_request(image=Image.new("RGB", (2001, 1001)), height=768, width=640)

    monkeypatch.setattr(kv_pipeline_module, "retrieve_timesteps", lambda *args, **kwargs: ([torch.tensor(1.0)], 1))

    output = pipeline.forward(req)

    assert output.output.shape == (1, 3, 8, 8)
    assert pipeline.image_processor.preprocess_calls == [((1000, 500), 496, 992, "crop")]
    assert pipeline.prepare_latents_calls[0]["height"] == 768
    assert pipeline.prepare_latents_calls[0]["width"] == 640
