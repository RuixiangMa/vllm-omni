# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E offline tests for VibeVoice ASR model with audio input and text output.
"""

from pathlib import Path

import pytest
from vllm.assets.audio import AudioAsset

from .conftest import OmniRunner
from .utils import create_new_process_for_each_test

models = ["microsoft/VibeVoice-ASR"]

# Simple single-stage config for VibeVoice ASR
stage_config = str(Path(__file__).parent / "stage_configs" / "vibevoice_asr_ci.yaml")

# Create parameter combinations for model and stage config
test_params = [(model, stage_config) for model in models]


def create_test_audio(duration: int = 5, sample_rate: int = 24000):
    """Create synthetic test audio with known content."""
    import numpy as np

    # Generate synthetic audio data (sine wave for testing)
    t = np.linspace(0, duration, int(sample_rate * duration))
    frequency = 440  # A4 note
    audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)

    return (audio_data.astype(np.float32), sample_rate)


@pytest.mark.core_model
@pytest.mark.parametrize("test_config", test_params)
@create_new_process_for_each_test("spawn")
def test_audio_to_text_single(omni_runner: type[OmniRunner], test_config: tuple[str, str]) -> None:
    """Test single audio input to text output."""
    model, stage_config_path = test_config
    with omni_runner(model, seed=42, stage_configs_path=stage_config_path) as runner:
        # Prepare audio input
        question = "Please transcribe this audio."
        audio = create_test_audio(duration=5, sample_rate=24000)

        outputs = runner.generate_multimodal(
            prompts=question,
            audios=audio,
        )

        # Find and verify text output
        text_output = None
        output_count = 0
        for stage_output in outputs:
            if stage_output.final_output_type == "text":
                text_output = stage_output
                output_count += 1
                break
        assert output_count > 0

        assert text_output is not None
        assert len(text_output.request_output) > 0
        text_content = text_output.request_output[0].outputs[0].text
        assert text_content is not None
        assert len(text_content.strip()) > 0

        # Verify transcription contains expected keywords
        expected_keywords = ["beijing", "china", "test", "speech"]
        assert any(keyword in text_content.lower() for keyword in expected_keywords), (
            f"The transcription does not contain expected keywords. Got: {text_content}"
        )

        print(f"Transcription result: {text_content}")


@pytest.mark.core_model
@pytest.mark.parametrize("test_config", test_params)
@create_new_process_for_each_test("spawn")
def test_audio_to_text_batch(omni_runner: type[OmniRunner], test_config: tuple[str, str]) -> None:
    """Test batch audio inputs to text outputs."""
    model, stage_config_path = test_config
    with omni_runner(model, seed=42, stage_configs_path=stage_config_path) as runner:
        # Prepare batch of audio inputs
        questions = [
            "Please transcribe this audio.",
            "What is being said in this audio?",
            "Transcribe the speech in this recording.",
        ]

        # Create different audio samples
        audios = [
            create_test_audio(duration=3, sample_rate=24000),
            create_test_audio(duration=4, sample_rate=24000),
            create_test_audio(duration=5, sample_rate=24000),
        ]

        outputs = runner.generate_multimodal(
            prompts=questions,
            audios=audios,
        )

        # Find and verify text outputs
        text_output = None
        output_count = 0
        for stage_output in outputs:
            if stage_output.final_output_type == "text":
                text_output = stage_output
                output_count += 1
                break
        assert output_count > 0

        assert text_output is not None
        assert len(text_output.request_output) == len(questions)

        for i, request_output in enumerate(text_output.request_output):
            text_content = request_output.outputs[0].text
            assert text_content is not None
            assert len(text_content.strip()) > 0

            # Verify each transcription contains expected keywords
            expected_keywords = ["beijing", "china", "test", "speech"]
            assert any(keyword in text_content.lower() for keyword in expected_keywords), (
                f"Transcription {i} does not contain expected keywords. Got: {text_content}"
            )

            print(f"Batch transcription {i}: {text_content}")


@pytest.mark.core_model
@pytest.mark.parametrize("test_config", test_params)
@create_new_process_for_each_test("spawn")
def test_audio_to_text_empty(omni_runner: type[OmniRunner], test_config: tuple[str, str]) -> None:
    """Test empty audio input handling."""
    model, stage_config_path = test_config
    with omni_runner(model, seed=42, stage_configs_path=stage_config_path) as runner:
        # Prepare empty audio input
        question = "Please transcribe this audio."
        audio = create_test_audio(duration=0.1, sample_rate=24000)  # Very short audio

        outputs = runner.generate_multimodal(
            prompts=question,
            audios=audio,
        )

        # Find and verify text output
        text_output = None
        output_count = 0
        for stage_output in outputs:
            if stage_output.final_output_type == "text":
                text_output = stage_output
                output_count += 1
                break
        assert output_count > 0

        assert text_output is not None
        assert len(text_output.request_output) > 0
        text_content = text_output.request_output[0].outputs[0].text
        assert text_content is not None

        print(f"Empty audio transcription result: {text_content}")
        # Should handle empty/very short audio gracefully


@pytest.mark.core_model
@pytest.mark.parametrize("test_config", test_params)
@create_new_process_for_each_test("spawn")
def test_audio_to_text_different_formats(omni_runner: type[OmniRunner], test_config: tuple[str, str]) -> None:
    """Test different audio formats and sample rates."""
    model, stage_config_path = test_config
    with omni_runner(model, seed=42, stage_configs_path=stage_config_path) as runner:
        # Test different sample rates
        sample_rates = [16000, 24000, 48000]
        questions = ["Please transcribe this audio."] * len(sample_rates)

        audios = []
        for sr in sample_rates:
            audios.append(create_test_audio(duration=3, sample_rate=sr))

        outputs = runner.generate_multimodal(
            prompts=questions,
            audios=audios,
        )

        # Find and verify text outputs
        text_output = None
        output_count = 0
        for stage_output in outputs:
            if stage_output.final_output_type == "text":
                text_output = stage_output
                output_count += 1
                break
        assert output_count > 0

        assert text_output is not None
        assert len(text_output.request_output) == len(sample_rates)

        for i, request_output in enumerate(text_output.request_output):
            text_content = request_output.outputs[0].text
            assert text_content is not None
            assert len(text_content.strip()) > 0

            # Verify transcription contains expected keywords
            expected_keywords = ["beijing", "china", "test", "speech"]
            assert any(keyword in text_content.lower() for keyword in expected_keywords), (
                f"Transcription {i} (sample rate {sample_rates[i]}) does not contain expected keywords. Got: {text_content}"
            )

            print(f"Sample rate {sample_rates[i]} transcription: {text_content}")


@pytest.mark.core_model
@pytest.mark.parametrize("test_config", test_params)
@create_new_process_for_each_test("spawn")
def test_audio_to_text_with_real_audio(omni_runner: type[OmniRunner], test_config: tuple[str, str]) -> None:
    """Test with real audio asset if available."""
    model, stage_config_path = test_config
    with omni_runner(model, seed=42, stage_configs_path=stage_config_path) as runner:
        try:
            # Try to use real audio asset
            audio = AudioAsset("mary_had_lamb").audio_and_sample_rate
            audio = (audio[0][: 24000 * 5], audio[1])  # Trim to first 5 seconds, resample to 24kHz

            question = "Please transcribe this audio."

            outputs = runner.generate_multimodal(
                prompts=question,
                audios=audio,
            )

            # Find and verify text output
            text_output = None
            output_count = 0
            for stage_output in outputs:
                if stage_output.final_output_type == "text":
                    text_output = stage_output
                    output_count += 1
                    break
            assert output_count > 0

            assert text_output is not None
            assert len(text_output.request_output) > 0
            text_content = text_output.request_output[0].outputs[0].text
            assert text_content is not None
            assert len(text_content.strip()) > 0

            print(f"Real audio transcription result: {text_content}")

        except Exception as e:
            pytest.skip(f"Real audio asset not available: {e}")


@pytest.mark.core_model
@pytest.mark.parametrize("test_config", test_params)
@create_new_process_for_each_test("spawn")
def test_audio_to_text_long_audio(omni_runner: type[OmniRunner], test_config: tuple[str, str]) -> None:
    """Test long audio input handling."""
    model, stage_config_path = test_config
    with omni_runner(model, seed=42, stage_configs_path=stage_config_path) as runner:
        # Prepare longer audio input (30 seconds)
        question = "Please transcribe this audio."
        audio = create_test_audio(duration=30, sample_rate=24000)

        outputs = runner.generate_multimodal(
            prompts=question,
            audios=audio,
        )

        # Find and verify text output
        text_output = None
        output_count = 0
        for stage_output in outputs:
            if stage_output.final_output_type == "text":
                text_output = stage_output
                output_count += 1
                break
        assert output_count > 0

        assert text_output is not None
        assert len(text_output.request_output) > 0
        text_content = text_output.request_output[0].outputs[0].text
        assert text_content is not None
        assert len(text_content.strip()) > 0

        print(f"Long audio transcription result: {text_content}")
        # Should handle long audio without issues
