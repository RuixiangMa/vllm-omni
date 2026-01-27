# Adapted from https://github.com/microsoft/VibeVoice

from typing import ClassVar, Literal

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP
from vllm.model_executor.models.utils import init_vllm_registered_model, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.custom_process_mixin import CustomProcessMixin
from vllm_omni.model_executor.models.vibevoice_asr.audio_encoder import (
    VibeVoiceAudioEncoder,
)
from vllm_omni.model_executor.models.vibevoice_asr.processing import (
    VibeVoiceASRDummyInputsBuilder,
    VibeVoiceASRMultiModalProcessor,
    VibeVoiceASRProcessingInfo,
)

logger = init_logger(__name__)

AUDIO_SAMPLE_RATE = 24000


@MULTIMODAL_REGISTRY.register_processor(
    VibeVoiceASRMultiModalProcessor,
    info=VibeVoiceASRProcessingInfo,
    dummy_inputs=VibeVoiceASRDummyInputsBuilder,
)
class VibeVoiceASRForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP, CustomProcessMixin):
    """
    This model combines VibeVoice acoustic/semantic tokenizers for audio encoding
    with a causal language model for text generation.
    """

    supports_transcription: ClassVar[Literal[True]] = True
    supports_transcription_only: ClassVar[bool] = False
    supports_segment_timestamp: ClassVar[bool] = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.has_preprocess = False
        self.have_multimodal_outputs = False

        config = vllm_config.model_config.hf_config
        self.config = config
        self.vllm_config = vllm_config
        self._model_path = vllm_config.model_config.model

        self.audio_encoder = VibeVoiceAudioEncoder(config)

        decoder_config = getattr(config, "decoder_config", config)
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=decoder_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Qwen2ForCausalLM"],
        )

        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

        lm_dtype = vllm_config.model_config.dtype
        if lm_dtype is not None:
            self.audio_encoder._lm_dtype = lm_dtype

        try:
            self.audio_encoder._ensure_audio_encoder_dtype()
        except Exception:
            pass

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        """Forward pass for VibeVoice ASR model."""
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if intermediate_tensors is not None:
            inputs_embeds = None

        language_model = self.language_model
        if hasattr(language_model, "language_model"):
            language_model = language_model.language_model

        hidden_states = language_model.model(
            input_ids=None,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        if hidden_states is None:
            return None

        return self.language_model.compute_logits(hidden_states)

    def get_input_embeddings(self) -> torch.nn.Module:
        """Return the text embedding layer."""
        if hasattr(self.language_model, "model") and hasattr(self.language_model.model, "embed_tokens"):
            return self.language_model.model.embed_tokens
        elif hasattr(self.language_model, "embed_tokens"):
            return self.language_model.embed_tokens

        inner = self.language_model
        if hasattr(inner, "language_model"):
            inner = inner.language_model
        if hasattr(inner, "model") and hasattr(inner.model, "embed_tokens"):
            return inner.model.embed_tokens

        raise AttributeError("Cannot find embed_tokens layer")

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: torch.Tensor | list[torch.Tensor] | None = None,
        is_multimodal: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Apply token embeddings to input_ids and merge with multimodal embeddings."""
        from vllm.model_executor.models.utils import _merge_multimodal_embeddings

        embed_tokens = self.get_input_embeddings()
        inputs_embeds = embed_tokens(input_ids)

        if multimodal_embeddings is not None and is_multimodal is not None:
            inputs_embeds = _merge_multimodal_embeddings(
                inputs_embeds,
                multimodal_embeddings,
                is_multimodal,
            )

        return inputs_embeds

    def embed_multimodal(self, **kwargs: object) -> tuple[torch.Tensor, ...]:
        """Extract audio embeddings using VibeVoice's acoustic/semantic tokenizers."""
        raw_audio = kwargs.get("raw_audio")
        if raw_audio is None:
            raw_audio = kwargs.get("audio")
        raw_audio_lengths = kwargs.get("raw_audio_lengths")
        if raw_audio_lengths is None:
            raw_audio_lengths = kwargs.get("audio_length")

        if raw_audio is None:
            return ()

        if isinstance(raw_audio, (list, tuple)) and len(raw_audio) == 0:
            return ()

        def flatten_lengths(lengths):
            """Flatten nested lists/tensors of lengths to a single list."""
            if lengths is None:
                return []

            result = []
            if isinstance(lengths, torch.Tensor):
                lengths = lengths.tolist()

            if isinstance(lengths, (list, tuple)):
                for item in lengths:
                    if isinstance(item, (list, tuple)):
                        result.extend(flatten_lengths(item))
                    elif isinstance(item, torch.Tensor):
                        if item.dim() == 0:
                            result.append(item.item())
                        else:
                            result.extend(item.tolist())
                    else:
                        result.append(item)
            else:
                result.append(lengths)
            return result

        raw_audio_lengths = flatten_lengths(raw_audio_lengths)

        embeddings = []

        if isinstance(raw_audio, torch.Tensor):
            if raw_audio.dim() == 3:
                num_audios = raw_audio.shape[0]
                audio_list = [raw_audio[i].squeeze(0) for i in range(num_audios)]
            elif raw_audio.dim() == 2:
                num_audios = raw_audio.shape[0]
                audio_list = [raw_audio[i] for i in range(num_audios)]
            else:
                audio_list = [raw_audio]
        else:
            audio_list = list(raw_audio)

        for i, audio_tensor in enumerate(audio_list):
            try:
                if isinstance(audio_tensor, list):
                    audio_tensor = torch.stack(audio_tensor)

                if not isinstance(audio_tensor, torch.Tensor):
                    audio_tensor = torch.tensor(audio_tensor)

                if audio_tensor.dim() == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)

                device = next(self.audio_encoder.parameters()).device
                audio_tensor = audio_tensor.to(device=device, dtype=torch.float32)

                if raw_audio_lengths and i < len(raw_audio_lengths):
                    actual_len = int(raw_audio_lengths[i])
                    if actual_len > 0 and actual_len <= audio_tensor.shape[-1]:
                        audio_tensor = audio_tensor[..., :actual_len]

                if audio_tensor.numel() < 160:
                    continue

                audio_embeds = self.audio_encoder(audio_tensor)
                final_embed = audio_embeds.squeeze(0)
                embeddings.append(final_embed)

            except Exception as e:
                logger.warning(f"[VibeVoice] Failed to encode audio at index {i}: {e}")

        return tuple(embeddings)

    def get_language_model(self) -> torch.nn.Module:
        """Return the language model backbone."""
        return self.language_model

    def load_weights(self, weights: list[tuple[str, torch.Tensor]]) -> set[str]:
        """Load model weights from checkpoint."""
        from vllm.model_executor.models.utils import (
            AutoWeightsLoader,
            WeightsMapper,
        )

        mapper = WeightsMapper(
            orig_to_new_prefix={
                "model.acoustic_tokenizer.": "audio_encoder.acoustic_tokenizer.",
                "model.semantic_tokenizer.": "audio_encoder.semantic_tokenizer.",
                "model.acoustic_connector.": "audio_encoder.acoustic_connector.",
                "model.semantic_connector.": "audio_encoder.semantic_connector.",
                "model.language_model.": "language_model.model.",
                "lm_head.": "language_model.lm_head.",
            }
        )

        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=mapper)

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        """Return the placeholder string format for a given modality."""
        if modality.startswith("audio"):
            return "<|AUDIO|>"
        raise ValueError("Only audio modality is supported")
