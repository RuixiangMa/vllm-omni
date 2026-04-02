# Text-To-Audio

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/text_to_audio>.


The `stabilityai/stable-audio-open-1.0` pipeline generates audio from text prompts.

## Prerequisites

If you use a gated model (e.g., `stabilityai/stable-audio-open-1.0`), ensure you have access:

1. **Accept Model License**: Visit the model page on Hugging Face (e.g., [stabilityai/stable-audio-open-1.0]) and accept the user agreement.
2. **Authenticate**: Log in to Hugging Face locally to access the gated model.
   ```bash
   huggingface-cli login
   ```

## Start Server

### Basic Start

```bash
vllm serve stabilityai/stable-audio-open-1.0 --omni --port 8091
```

### Start with Parameters

You can pass additional parameters directly to the `vllm serve` command:

```bash
# With specific port and log settings
vllm serve stabilityai/stable-audio-open-1.0 --omni --port 8091 \
  --disable-log-stats
```

## API Calls

### Method 1: Using curl

```bash
curl -s http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "The sound of a dog barking"}
    ],
    "extra_body": {
      "num_inference_steps": 100,
      "guidance_scale": 7.0,
      "audio_start_in_s": 0.0,
      "audio_end_in_s": 10.0,
      "seed": 42
    }
  }' | jq -r '.choices[0].message.content[0].audio_url.url' | cut -d',' -f2- | base64 -d > output.wav
```

### Method 2: Using OpenAI Python SDK

```python
from openai import OpenAI
import base64

client = OpenAI(base_url="http://localhost:8091/v1", api_key="none")

response = client.chat.completions.create(
    model="stabilityai/stable-audio-open-1.0",
    messages=[{"role": "user", "content": "The sound of a dog barking"}],
    extra_body={
        "num_inference_steps": 100,
        "guidance_scale": 7.0,
        "audio_start_in_s": 0.0,
        "audio_end_in_s": 10.0,
        "seed": 42,
    },
)

audio_url = response.choices[0].message.content[0].audio_url.url
_, b64_data = audio_url.split(",", 1)
with open("output.wav", "wb") as f:
    f.write(base64.b64decode(b64_data))
```

## Request Format

### Simple Text Generation

```json
{
  "messages": [
    {"role": "user", "content": "The sound of a dog barking"}
  ]
}
```

### Generation with Parameters

Wrap generation parameters inside `extra_body` in the request JSON:

```json
{
  "messages": [
    {"role": "user", "content": "The sound of a dog barking"}
  ],
  "extra_body": {
    "num_inference_steps": 100,
    "guidance_scale": 7.0,
    "audio_start_in_s": 0.0,
    "audio_end_in_s": 10.0,
    "seed": 42
  }
}
```

!!! tip "Using the OpenAI SDK"
    When using the OpenAI Python SDK, pass these parameters via the `extra_body`
    keyword argument. The SDK merges them into the top-level request body automatically:

    ```python
    client.chat.completions.create(
        model="stabilityai/stable-audio-open-1.0",
        messages=[...],
        extra_body={"num_inference_steps": 100, "guidance_scale": 7.0},
    )
    ```

    For details on how generation parameters are handled across different clients, see the
    [Diffusion Chat API guide](../../../../serving/diffusion_chat_api.md).

## Generation Parameters

When using `/v1/chat/completions`, pass these inside `extra_body` in the curl
JSON, or via the `extra_body` keyword argument in the OpenAI Python SDK (see the
[Diffusion Chat API guide](../../../../serving/diffusion_chat_api.md)).

| Parameter                | Type  | Default | Description                               |
| ------------------------ | ----- | ------- | ----------------------------------------- |
| `num_inference_steps`    | int   | 100     | Number of denoising steps                 |
| `guidance_scale`         | float | 7.0     | Classifier-free guidance scale            |
| `audio_start_in_s`       | float | 0.0     | Start time of audio segment in seconds    |
| `audio_end_in_s`         | float | 10.0    | End time of audio segment in seconds      |
| `seed`                   | int   | None    | Random seed (reproducible)                |
| `negative_prompt`        | str   | None    | Negative prompt                           |
| `num_outputs_per_prompt` | int   | 1       | Number of audio clips to generate         |

## Response Format

```json
{
  "id": "chatcmpl-xxx",
  "created": 1234567890,
  "model": "stabilityai/stable-audio-open-1.0",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": [{
        "type": "audio_url",
        "audio_url": {
          "url": "data:audio/wav;base64,..."
        }
      }]
    },
    "finish_reason": "stop"
  }],
  "usage": {...}
}
```

## Extract Audio

```bash
# Extract base64 from response and decode to WAV
cat response.json | jq -r '.choices[0].message.content[0].audio_url.url' | cut -d',' -f2- | base64 -d > output.wav
```

## File Description

| File                        | Description                  |
| --------------------------- | ---------------------------- |
| `run_curl_text_to_audio.sh` | curl example                 |
| `openai_chat_client.py`     | Python client                |

## Example materials

??? abstract "openai_chat_client.py"
    ``````py
    --8<-- "examples/online_serving/text_to_audio/openai_chat_client.py"
    ``````
??? abstract "run_curl_text_to_audio.sh"
    ``````sh
    --8<-- "examples/online_serving/text_to_audio/run_curl_text_to_audio.sh"
    ``````
