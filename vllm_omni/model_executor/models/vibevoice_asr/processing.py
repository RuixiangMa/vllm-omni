# Adapted from https://github.com/microsoft/VibeVoice

import json
import logging
import os
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from transformers import BatchFeature
from transformers.models.whisper import WhisperFeatureExtractor

try:
    from vibevoice.processor.audio_utils import (
        AudioNormalizer,
        load_audio_bytes_use_ffmpeg,
        load_audio_use_ffmpeg,
    )
except ImportError as e:
    raise ImportError(
        "The 'vibevoice' package is required to use VibeVoice ASR processing but "
        "is not installed. Please install it (e.g., `pip install vibevoice`) "
        "and try again."
    ) from e

from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalInputs,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import MultiModalDataParser
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs

# Configure logger
logger = logging.getLogger(__name__)

AUDIO_SAMPLE_RATE = 24000


class VibeVoiceASRProcessingInfo(BaseProcessingInfo):
    """Processing info for VibeVoice ASR multimodal model."""

    @staticmethod
    def load_audio(audio_path: str, target_sr: int = AUDIO_SAMPLE_RATE) -> np.ndarray:
        """Load and normalize audio from file path."""
        audio, _ = load_audio_use_ffmpeg(audio_path, resample=True, target_sr=target_sr)
        audio = AudioNormalizer()(audio)
        return audio

    @staticmethod
    def audio_input_mapper(ctx, data: str | bytes | np.ndarray | list[str]) -> MultiModalInputs:
        """Map audio input data to vLLM MultiModalInputs format."""
        if isinstance(data, list):
            data = data[0]

        if isinstance(data, str):
            audio_waveform = VibeVoiceASRProcessingInfo.load_audio(data)
        elif isinstance(data, bytes):
            audio_waveform, _ = load_audio_bytes_use_ffmpeg(data, resample=True, target_sr=AUDIO_SAMPLE_RATE)
            audio_waveform = AudioNormalizer()(audio_waveform)
        elif isinstance(data, np.ndarray):
            audio_waveform = data
        else:
            raise ValueError(f"Unsupported audio data type: {type(data)}")

        audio_tensor = torch.from_numpy(audio_waveform).float()

        return MultiModalInputs(
            {
                "audio": audio_tensor,
                "audio_length": audio_tensor.shape[0],
            }
        )

    @staticmethod
    def get_field_config(hf_inputs: Mapping[str, torch.Tensor]):
        """Map HF processor output keys to audio modality."""
        config = {
            "raw_audio": MultiModalFieldConfig.batched("audio"),
            "raw_audio_lengths": MultiModalFieldConfig.batched("audio"),
            "salt": MultiModalFieldConfig.batched("audio"),
        }

        if "input_features" in hf_inputs:
            config["input_features"] = MultiModalFieldConfig.batched("audio")
        if "feature_attention_mask" in hf_inputs:
            config["feature_attention_mask"] = MultiModalFieldConfig.batched("audio")

        return config

    def get_hf_config(self):
        return self.ctx.get_hf_config()

    def get_feature_extractor(self, **kwargs) -> WhisperFeatureExtractor:
        model_path = self.ctx.model_config.model
        preprocessor_path = os.path.join(model_path, "preprocessor_config.json")

        config = {
            "sampling_rate": AUDIO_SAMPLE_RATE,
            "feature_size": 128,
            "hop_length": 240,
            "chunk_length": 30,
            "n_fft": 400,
            "padding_value": 0.0,
        }

        if os.path.exists(preprocessor_path):
            try:
                with open(preprocessor_path) as f:
                    file_config = json.load(f)
                    config.update({k: file_config[k] for k in config if k in file_config})
            except Exception as e:
                logger.warning(
                    f"Failed to load preprocessor config from {preprocessor_path}: {str(e)}. "
                    "Using default feature extractor configuration."
                )

        return WhisperFeatureExtractor(**config)

    def get_audio_token_info(self) -> dict[str, Any]:
        """Get audio special tokens and their IDs."""
        tokenizer = self.get_tokenizer()
        vocab = tokenizer.get_vocab()

        tokens = {
            "audio_token": "<|AUDIO|>",
            "audio_bos_token": "<|audio_bos|>",
            "audio_eos_token": "<|audio_eos|>",
        }

        tokens["audio_token_id"] = vocab.get(tokens["audio_token"])
        tokens["audio_bos_id"] = vocab.get(tokens["audio_bos_token"])
        tokens["audio_eos_id"] = vocab.get(tokens["audio_eos_token"])

        return tokens

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None}


class VibeVoiceASRDummyInputsBuilder(BaseDummyInputsBuilder[VibeVoiceASRProcessingInfo]):
    """Dummy inputs builder for VibeVoice ASR model profiling."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        """Generate dummy text with audio placeholders."""
        num_audios = mm_counts.get("audio", 0)
        if num_audios <= 0:
            return ""

        token_info = self.info.get_audio_token_info()
        return token_info["audio_token"] * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate dummy audio data for profiling."""
        feature_extractor = self.info.get_feature_extractor()
        audio_len = feature_extractor.chunk_length * feature_extractor.sampling_rate
        num_audios = mm_counts.get("audio", 0)

        return {"audio": [np.zeros(audio_len, dtype=np.float32) for _ in range(num_audios)]}

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, Any] | None = None,
    ) -> ProcessorInputs:
        """Build ProcessorInputs for dummy profiling."""
        return ProcessorInputs(
            prompt=self.get_dummy_text(mm_counts),
            mm_data=self.get_dummy_mm_data(seq_len, mm_counts, mm_options),
        )


class VibeVoiceASRMultiModalProcessor(BaseMultiModalProcessor[VibeVoiceASRProcessingInfo]):
    """Multi-modal processor for VibeVoice ASR model."""

    def _get_data_parser(self) -> MultiModalDataParser:
        """Create a data parser with the correct target sample rate (24kHz)."""
        return MultiModalDataParser(target_sr=AUDIO_SAMPLE_RATE)

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        mm_data = dict(mm_data)
        audios = mm_data.pop("audios", None)
        if audios is not None and "audio" not in mm_data:
            mm_data["audio"] = audios

        if "audio" not in mm_data or mm_data["audio"] is None:
            prompt_ids = self.info.get_tokenizer().encode(prompt)
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        raw_audio_list = mm_data.get("audio")
        if isinstance(raw_audio_list, np.ndarray):
            raw_audio_list = [raw_audio_list]
        elif not isinstance(raw_audio_list, list):
            raw_audio_list = list(raw_audio_list)

        tokenizer = self.info.get_tokenizer()
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)

        result = BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        max_len = max(len(a) for a in raw_audio_list)
        raw_audio_tensors = []
        audio_lengths = []
        for audio in raw_audio_list:
            audio_len = len(audio)
            audio_lengths.append(audio_len)
            if audio_len < max_len:
                audio = np.pad(audio, (0, max_len - audio_len), mode="constant")
            raw_audio_tensors.append(torch.from_numpy(audio).float())

        result["raw_audio"] = torch.stack(raw_audio_tensors, dim=0)
        result["raw_audio_lengths"] = torch.tensor(audio_lengths, dtype=torch.long)

        salt_val = hash(str(uuid.uuid4())) % 100000
        result["salt"] = torch.tensor([salt_val], dtype=torch.long).expand(len(raw_audio_list))

        return result

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        """Return whether the HF processor applies prompt updates."""
        return False

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        """Configure which HF output fields map to which modality."""
        return VibeVoiceASRProcessingInfo.get_field_config(hf_inputs)

    def _get_prompt_updates(
        self,
        mm_items,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        token_info = self.info.get_audio_token_info()
        audio_token = token_info["audio_token"]
        audio_token_id = token_info["audio_token_id"]
        audio_bos_id = token_info.get("audio_bos_id")
        audio_eos_id = token_info.get("audio_eos_id")

        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        def _tok_id(name: str) -> int | None:
            return vocab.get(name)

        speech_start_id = (
            _tok_id("<|object_ref_start|>")
            or getattr(tokenizer, "speech_start_id", None)
            or _tok_id("<|speech_start|>")
        )
        speech_end_id = (
            _tok_id("<|object_ref_end|>") or getattr(tokenizer, "speech_end_id", None) or _tok_id("<|speech_end|>")
        )
        speech_pad_id = (
            _tok_id("<|box_start|>") or getattr(tokenizer, "speech_pad_id", None) or _tok_id("<|speech_pad|>")
        )

        if audio_token_id is None:
            return []

        out_mm_data = out_mm_kwargs.get_data()
        raw_audio_lengths = out_mm_data.get("raw_audio_lengths", [])

        hf_config = self.info.get_hf_config()
        if isinstance(hf_config, dict):
            compress_ratio = int(hf_config.get("speech_tok_compress_ratio", 3200))
        else:
            compress_ratio = int(getattr(hf_config, "speech_tok_compress_ratio", 3200))

        def _to_int_len(x) -> int:
            if x is None:
                return 0
            if isinstance(x, torch.Tensor):
                if x.numel() == 1:
                    return int(x.item())
                return int(x.shape[0])
            return int(x)

        def get_replacement(item_idx: int) -> PromptUpdateDetails:
            if raw_audio_lengths and item_idx < len(raw_audio_lengths):
                audio_len = _to_int_len(raw_audio_lengths[item_idx])
                num_features = max(1, int(np.ceil(audio_len / compress_ratio)))
            else:
                num_features = int(np.ceil(30 * AUDIO_SAMPLE_RATE / compress_ratio))

            if num_features == 0:
                raise ValueError(f"Audio at index {item_idx} is too short")

            # Derive the newline token ID from the tokenizer where possible, with a fallback.
            newline_id = _tok_id("\n") or getattr(tokenizer, "newline_id", None)
            if newline_id is None:
                try:
                    encoded_newline = tokenizer.encode("\n", add_special_tokens=False)
                    if isinstance(encoded_newline, list) and encoded_newline:
                        newline_id = int(encoded_newline[0])
                except Exception:
                    newline_id = None
            if newline_id is None:
                # Fallback to the legacy hardcoded value to preserve existing behavior.
                newline_id = 198
                logger.debug(f"Using fallback newline_id={newline_id} for tokenizer {type(tokenizer).__name__}")

            if speech_start_id is not None and speech_pad_id is not None and speech_end_id is not None:
                embed_id = int(speech_pad_id)
                replacement_ids = [int(speech_start_id)] + [embed_id] * num_features + [int(speech_end_id), newline_id]
            elif audio_bos_id is not None and audio_eos_id is not None:
                embed_id = int(audio_token_id)
                replacement_ids = [int(audio_bos_id)] + [embed_id] * num_features + [int(audio_eos_id)]
            else:
                embed_id = int(audio_token_id)
                replacement_ids = [embed_id] * num_features

            return PromptUpdateDetails.select_token_id(replacement_ids, embed_token_id=int(embed_id))

        return [
            PromptReplacement(
                modality="audio",
                target=audio_token,
                replacement=get_replacement,
            )
        ]
