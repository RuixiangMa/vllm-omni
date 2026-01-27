# Adapted from https://github.com/microsoft/VibeVoice

# Configure logger
import logging
import os
import sys

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

try:
    from vibevoice.modular.configuration_vibevoice import (
        VibeVoiceAcousticTokenizerConfig,
        VibeVoiceSemanticTokenizerConfig,
    )
    from vibevoice.modular.modular_vibevoice_tokenizer import (
        VibeVoiceAcousticTokenizerModel,
        VibeVoiceSemanticTokenizerModel,
        VibeVoiceTokenizerEncoderOutput,
        VibeVoiceTokenizerStreamingCache,
    )
except ImportError as e:
    raise ImportError(
        "VibeVoice ASR support requires the optional 'vibevoice' package. Install it with: pip install vibevoice"
    ) from e


class VibeVoiceASRSpeechConnector(nn.Module):
    """Projects speech features to language model hidden dimension for VibeVoice ASR.

    Architecture: fc1 -> RMSNorm -> fc2 (no activation function)
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.norm = VibeVoiceASRRMSNorm(output_dim, eps=1e-6)
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.norm(x)
        x = self.fc2(x)
        return x


class VibeVoiceASRRMSNorm(nn.Module):
    """RMSNorm layer used in VibeVoiceASRSpeechConnector for VibeVoice audio processing."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class VibeVoiceAudioEncoder(nn.Module):
    """
    VibeVoice Audio Encoder module.

    Encapsulates Acoustic/Semantic VAE Tokenizers and projection Connectors.
    Converts raw audio waveforms into embeddings compatible with the language model.

    Features:
        - Streaming support for long audio (>60s by default)
        - Configurable dtype for numerical precision
        - Supports both sampling and deterministic (mean) modes
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        def get_cfg(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        self.acoustic_vae_dim = get_cfg(config, "acoustic_vae_dim", 64)
        self.semantic_vae_dim = get_cfg(config, "semantic_vae_dim", 128)

        decoder_config = get_cfg(config, "decoder_config")
        text_config = get_cfg(config, "text_config")

        target_hidden_size = None

        if decoder_config is not None:
            target_hidden_size = get_cfg(decoder_config, "hidden_size")

        if target_hidden_size is None and text_config is not None:
            target_hidden_size = get_cfg(text_config, "hidden_size")

        if target_hidden_size is None:
            target_hidden_size = get_cfg(config, "hidden_size")

        if target_hidden_size is None:
            print("[VibeVoice] WARN: Could not find hidden_size in config! Defaulting to 3584 (7B).", file=sys.stderr)
            self.hidden_size = 3584
        else:
            self.hidden_size = target_hidden_size

        ac_cfg = get_cfg(config, "acoustic_tokenizer_config")
        sc_cfg = get_cfg(config, "semantic_tokenizer_config")

        if ac_cfg is None or sc_cfg is None:
            raise ValueError("Missing acoustic/semantic tokenizer config in model config")

        # Handle both dict and already-constructed config objects
        if isinstance(ac_cfg, VibeVoiceAcousticTokenizerConfig):
            acoustic_config = ac_cfg
        elif isinstance(ac_cfg, dict):
            acoustic_config = VibeVoiceAcousticTokenizerConfig(**ac_cfg)
        else:
            raise TypeError(f"acoustic_tokenizer_config has unexpected type: {type(ac_cfg)}")

        if isinstance(sc_cfg, VibeVoiceSemanticTokenizerConfig):
            semantic_config = sc_cfg
        elif isinstance(sc_cfg, dict):
            semantic_config = VibeVoiceSemanticTokenizerConfig(**sc_cfg)
        else:
            raise TypeError(f"semantic_tokenizer_config has unexpected type: {type(sc_cfg)}")

        # Tokenizers use float32 for numerical precision
        self.acoustic_tokenizer = VibeVoiceAcousticTokenizerModel(acoustic_config)
        self.semantic_tokenizer = VibeVoiceSemanticTokenizerModel(semantic_config)

        # Get audio encoder dtype from config (defaults to float32 for precision)
        root_torch_dtype = get_cfg(config, "torch_dtype", None)
        if root_torch_dtype is not None:
            if isinstance(root_torch_dtype, str):
                self._audio_encoder_dtype = getattr(torch, root_torch_dtype)
            else:
                self._audio_encoder_dtype = root_torch_dtype
        else:
            self._audio_encoder_dtype = torch.float32

        self.acoustic_connector = VibeVoiceASRSpeechConnector(self.acoustic_vae_dim, self.hidden_size)
        self.semantic_connector = VibeVoiceASRSpeechConnector(self.semantic_vae_dim, self.hidden_size)

        self.compress_ratio = get_cfg(config, "speech_tok_compress_ratio", 3200)

        # Streaming controls
        self.sample_rate = get_cfg(config, "target_sample_rate", 24000)

        # Default to True (per requirement): segment + cache inside one forward call.
        self.enable_streaming = get_cfg(config, "enable_streaming", True)
        self.streaming_segment_duration = get_cfg(config, "streaming_segment_duration", 60.0)

        use_mean_env = os.getenv("VIBEVOICE_USE_MEAN", "").strip().lower()
        self.use_sample = use_mean_env not in ("1", "true", "yes")

        self._lm_dtype: torch.dtype = torch.bfloat16

    def _ensure_audio_encoder_dtype(self):
        target_dtype = self._audio_encoder_dtype

        try:
            acoustic_dtype = next(self.acoustic_tokenizer.parameters()).dtype
            if acoustic_dtype != target_dtype:
                self.acoustic_tokenizer = self.acoustic_tokenizer.to(dtype=target_dtype)
                print(
                    f"[VibeVoice] Converted acoustic_tokenizer to {target_dtype} (was {acoustic_dtype})",
                    file=sys.stderr,
                )
        except (StopIteration, AttributeError, TypeError) as exc:
            logger.warning(
                "Failed to ensure acoustic tokenizer dtype during initialization: %s",
                exc,
            )

        try:
            semantic_dtype = next(self.semantic_tokenizer.parameters()).dtype
            if semantic_dtype != target_dtype:
                self.semantic_tokenizer = self.semantic_tokenizer.to(dtype=target_dtype)
                print(
                    f"[VibeVoice] Converted semantic_tokenizer to {target_dtype} (was {semantic_dtype})",
                    file=sys.stderr,
                )
        except (StopIteration, AttributeError, TypeError) as exc:
            logger.warning(
                "Failed to ensure semantic tokenizer dtype during initialization: %s",
                exc,
            )

        try:
            ac_conn_dtype = next(self.acoustic_connector.parameters()).dtype
            if ac_conn_dtype != target_dtype:
                self.acoustic_connector = self.acoustic_connector.to(dtype=target_dtype)
                print(
                    f"[VibeVoice] Converted acoustic_connector to {target_dtype} (was {ac_conn_dtype})", file=sys.stderr
                )
        except (StopIteration, AttributeError, TypeError) as exc:
            logger.warning(
                "Failed to ensure acoustic connector dtype during initialization: %s",
                exc,
            )

        try:
            sc_conn_dtype = next(self.semantic_connector.parameters()).dtype
            if sc_conn_dtype != target_dtype:
                self.semantic_connector = self.semantic_connector.to(dtype=target_dtype)
                print(
                    f"[VibeVoice] Converted semantic_connector to {target_dtype} (was {sc_conn_dtype})", file=sys.stderr
                )
        except (StopIteration, AttributeError, TypeError) as exc:
            logger.warning(
                "Failed to ensure semantic connector dtype during initialization: %s",
                exc,
            )

    def forward(
        self,
        audio: torch.Tensor,
        *,
        use_streaming: bool = True,
        segment_duration_s: float | None = None,
        use_sample: bool | None = None,
    ) -> torch.Tensor:
        """Encode audio with optional streaming for long clips.

        Args:
            audio: Input audio tensor [B, T] or [T]
            use_streaming: Whether to enable segmented encoding for long audio
            segment_duration_s: Segment length in seconds (defaults to 60s)
            use_sample: If True, use sampling for acoustic tokens; if False, use mean
                       Defaults to self.use_sample (controlled by VIBEVOICE_USE_MEAN env var)

        Returns:
            Audio embeddings tensor compatible with the language model
        """
        # Ensure audio encoder components use correct dtype
        self._ensure_audio_encoder_dtype()

        # Audio input should match the audio encoder dtype
        audio = audio.to(dtype=self._audio_encoder_dtype)

        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        # Resolve streaming options
        segment_duration = segment_duration_s or self.streaming_segment_duration
        sample_rate = self.sample_rate
        total_samples = audio.shape[-1]
        segment_samples = int(segment_duration * sample_rate)

        use_streaming = use_streaming and self.enable_streaming and total_samples > segment_samples

        # Resolve use_sample flag
        if use_sample is None:
            use_sample = self.use_sample

        with torch.no_grad():
            if not use_streaming:
                acoustic_input = audio.unsqueeze(1)
                acoustic_out = self.acoustic_tokenizer.encode(acoustic_input)
                if use_sample:
                    acoustic_tokens = acoustic_out.sample(dist_type=self.acoustic_tokenizer.std_dist_type)[0]
                else:
                    acoustic_tokens = acoustic_out.mean

                acoustic_embeds = self.acoustic_connector(acoustic_tokens)

                semantic_out = self.semantic_tokenizer.encode(acoustic_input)
                semantic_tokens = semantic_out.mean
                semantic_embeds = self.semantic_connector(semantic_tokens)
            else:
                # ==========================================
                # Streaming path (Retained for future use)
                # ==========================================
                acoustic_cache = VibeVoiceTokenizerStreamingCache()
                semantic_cache = VibeVoiceTokenizerStreamingCache()
                acoustic_mean_segments = []
                semantic_mean_segments = []
                batch_size = audio.shape[0]
                sample_indices = torch.arange(batch_size, device=audio.device)

                def _iter_segments(total_length: int, segment_length: int):
                    for start in range(0, total_length, segment_length):
                        end = min(start + segment_length, total_length)
                        if end > start:
                            yield start, end

                segments = list(_iter_segments(total_samples, segment_samples))
                num_segments = len(segments)
                for seg_idx, (start, end) in enumerate(segments):
                    chunk = audio[:, start:end].contiguous()
                    if chunk.numel() == 0:
                        continue

                    # Check if this is the final segment
                    is_final = seg_idx == num_segments - 1

                    # --- Acoustic Encode ---
                    acoustic_enc_out = self.acoustic_tokenizer.encode(
                        chunk.unsqueeze(1),
                        cache=acoustic_cache,
                        sample_indices=sample_indices,
                        use_cache=True,
                        is_final_chunk=is_final,
                    )
                    acoustic_mean_segments.append(acoustic_enc_out.mean)

                    # --- Semantic Encode ---
                    semantic_enc_out = self.semantic_tokenizer.encode(
                        chunk.unsqueeze(1),
                        cache=semantic_cache,
                        sample_indices=sample_indices,
                        use_cache=True,
                        is_final_chunk=is_final,
                    )
                    semantic_mean_segments.append(semantic_enc_out.mean)

                if len(acoustic_mean_segments) == 0:
                    acoustic_mean_full = torch.zeros(
                        (batch_size, 0, self.acoustic_vae_dim),
                        device=audio.device,
                        dtype=self._audio_encoder_dtype,
                    )
                else:
                    acoustic_mean_full = torch.cat(acoustic_mean_segments, dim=1).contiguous()

                acoustic_enc_full = VibeVoiceTokenizerEncoderOutput(
                    mean=acoustic_mean_full,
                    std=self.acoustic_tokenizer.fix_std,
                )
                if use_sample:
                    acoustic_tokens = acoustic_enc_full.sample(dist_type=self.acoustic_tokenizer.std_dist_type)[0]
                else:
                    acoustic_tokens = acoustic_enc_full.mean
                acoustic_embeds = self.acoustic_connector(acoustic_tokens)

                # Concatenate sequence outputs (Semantic)
                if len(semantic_mean_segments) == 0:
                    semantic_tokens = torch.zeros(
                        (batch_size, 0, self.semantic_vae_dim),
                        device=audio.device,
                        dtype=self._audio_encoder_dtype,  # Use config dtype
                    )
                else:
                    semantic_tokens = torch.cat(semantic_mean_segments, dim=1).contiguous()
                # Connector uses same dtype as tokenizer
                semantic_embeds = self.semantic_connector(semantic_tokens)

        # Combine acoustic and semantic embeddings
        combined_embeds = acoustic_embeds + semantic_embeds

        # Convert to language model dtype for compatibility
        # Audio encoder uses config.torch_dtype (typically float32) for numerical precision,
        # but LM expects the dtype specified by vLLM's --dtype flag (e.g., bfloat16, float16)
        combined_embeds = combined_embeds.to(dtype=self._lm_dtype)

        return combined_embeds
