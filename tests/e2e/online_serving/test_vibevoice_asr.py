# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E Online tests for VibeVoice ASR model with audio input and text output.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import concurrent.futures
import time
from pathlib import Path

import openai
import pytest

from tests.conftest import (
    OmniServer,
    generate_synthetic_audio,
)

models = ["microsoft/VibeVoice-ASR"]

# Simple single-stage config for VibeVoice ASR
stage_configs = [str(Path(__file__).parent / "stage_configs" / "vibevoice_asr_ci.yaml")]

# Create parameter combinations for model and stage config
test_params = [(model, stage_config) for model in models for stage_config in stage_configs]


@pytest.fixture(scope="module")
def omni_server(request):
    """Start vLLM-Omni server as a subprocess with actual model weights.

    Uses session scope so the server starts only once for the entire test session.
    Multi-stage initialization can take 10-20+ minutes.
    """
    model, stage_config_path = request.param

    print(f"Starting OmniServer with model: {model}")
    print("This may take 10-20+ minutes for initialization...")

    with OmniServer(
        model,
        [
            "--stage-configs-path",
            stage_config_path,
            "--stage-init-timeout",
            "120",
        ],
    ) as server:
        print("OmniServer started successfully")
        yield server
        print("OmniServer stopped")


@pytest.fixture
def client(omni_server):
    """OpenAI client for the running vLLM-Omni server."""
    return openai.OpenAI(
        base_url=f"http://{omni_server.host}:{omni_server.port}/v1",
        api_key="EMPTY",
    )


def get_system_prompt():
    return {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": (
                    "You are VibeVoice ASR, an automatic speech recognition system. "
                    "Transcribe the audio input accurately and return only the text transcription."
                ),
            }
        ],
    }


def dummy_messages_from_audio_data(
    audio_data_url: str,
    content_text: str = "Transcribe this audio.",
):
    """Create messages with audio data URL for OpenAI API."""
    return [
        get_system_prompt(),
        {
            "role": "user",
            "content": [
                {"type": "audio_url", "audio_url": {"url": audio_data_url}},
                {"type": "text", "text": content_text},
            ],
        },
    ]


@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_audio_to_text_001(client: openai.OpenAI, omni_server) -> None:
    """Test audio input processing and text output generation via OpenAI API.

    Deploy Setting: default yaml
    Input Modal: audio
    Output Modal: text
    Datasets: single request
    """
    audio_data_url = f"data:audio/wav;base64,{generate_synthetic_audio(5, 1)['base64']}"
    messages = dummy_messages_from_audio_data(
        audio_data_url=audio_data_url,
        content_text="Please transcribe this audio.",
    )

    start_time = time.perf_counter()
    chat_completion = client.chat.completions.create(
        model=omni_server.model,
        messages=messages,
        stream=False,
    )
    current_e2e = time.perf_counter() - start_time
    print(f"The request e2e is: {current_e2e}")

    text_content = chat_completion.choices[0].message.content
    assert text_content is not None, "No text output is generated"
    assert isinstance(text_content, str), "Text output should be a string"
    print(f"Transcription result: {text_content}")


@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_audio_to_text_streaming_002(client: openai.OpenAI, omni_server) -> None:
    """Test audio input processing and text output generation via OpenAI API with streaming.

    Deploy Setting: default yaml
    Input Modal: audio
    Output Modal: text
    Datasets: single request with streaming
    """
    audio_data_url = f"data:audio/wav;base64,{generate_synthetic_audio(5, 1)['base64']}"
    messages = dummy_messages_from_audio_data(
        audio_data_url=audio_data_url,
        content_text="Please transcribe this audio.",
    )

    start_time = time.perf_counter()
    chat_completion = client.chat.completions.create(
        model=omni_server.model,
        messages=messages,
        stream=True,
    )

    text_content = ""
    for chunk in chat_completion:
        for choice in chunk.choices:
            if hasattr(choice, "delta"):
                content = getattr(choice.delta, "content", None)
                if content:
                    text_content += content

    current_e2e = time.perf_counter() - start_time
    print(f"The streaming request e2e is: {current_e2e}")

    assert text_content is not None, "No text output is generated"
    assert isinstance(text_content, str), "Text output should be a string"
    print(f"Streaming transcription result: {text_content}")


@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_audio_to_text_concurrent_003(client: openai.OpenAI, omni_server) -> None:
    """Test concurrent audio input processing and text output generation via OpenAI API.

    Deploy Setting: default yaml
    Input Modal: audio
    Output Modal: text
    Datasets: multiple concurrent requests
    """
    num_concurrent_requests = 3
    audio_data_url = f"data:audio/wav;base64,{generate_synthetic_audio(5, 1)['base64']}"
    messages = dummy_messages_from_audio_data(
        audio_data_url=audio_data_url,
        content_text="Please transcribe this audio.",
    )

    e2e_list = list()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent_requests) as executor:
        futures = [
            executor.submit(
                client.chat.completions.create,
                model=omni_server.model,
                messages=messages,
                stream=False,
            )
            for _ in range(num_concurrent_requests)
        ]

        start_time = time.perf_counter()
        chat_completions = list()
        for future in concurrent.futures.as_completed(futures):
            chat_completions.append(future.result())
            current_e2e = time.perf_counter() - start_time
            print(f"The concurrent request e2e is: {current_e2e}")
            e2e_list.append(current_e2e)

    print(f"The avg e2e is: {sum(e2e_list) / len(e2e_list)}")

    assert len(chat_completions) == num_concurrent_requests, "Not all requests succeeded."

    for i, chat_completion in enumerate(chat_completions):
        text_content = chat_completion.choices[0].message.content
        assert text_content is not None, f"Request {i}: No text output is generated"
        assert isinstance(text_content, str), f"Request {i}: Text output should be a string"
        print(f"Concurrent transcription {i}: {text_content}")


@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_audio_to_text_empty_004(client: openai.OpenAI, omni_server) -> None:
    """Test audio input processing with empty audio via OpenAI API.

    Deploy Setting: default yaml
    Input Modal: empty audio
    Output Modal: text
    Datasets: single request with empty audio
    """
    empty_audio_data_url = f"data:audio/wav;base64,{generate_synthetic_audio(0.1, 1)['base64']}"
    messages = dummy_messages_from_audio_data(
        audio_data_url=empty_audio_data_url,
        content_text="Please transcribe this audio.",
    )

    chat_completion = client.chat.completions.create(
        model=omni_server.model,
        messages=messages,
        stream=False,
    )

    text_content = chat_completion.choices[0].message.content
    print(f"Empty audio transcription result: {text_content}")
    assert text_content is not None, "No text output is generated"
    assert isinstance(text_content, str), "Text output should be a string"
