# VibeVoice ASR with vLLM-Omni

This example demonstrates how to run VibeVoice ASR (Automatic Speech Recognition) model using vLLM-Omni for efficient inference.

## About VibeVoice

VibeVoice-ASR is a unified speech-to-text model designed to handle 60-minute long-form audio in a single pass, generating structured transcriptions containing:
- **Who** (Speaker)
- **When** (Timestamps)
- **What** (Content)

## Setup

### 1. Install System Dependencies

FFmpeg and audio libraries are required for audio processing:

```bash
apt-get update
apt-get install -y ffmpeg libsndfile1
```

### 2. Install VibeVoice Python Package

Install VibeVoice

```bash
git clone https://github.com/microsoft/VibeVoice.git
cd VibeVoice
pip install -e .
```

### 3. Generate Tokenizer Files

Generate the tokenizer files required for the model:

```bash
python examples/online_serving/vibevoice_asr/generate_tokenizer_files.py --output /path/to/model
```

### 4. Start vLLM Server

Start the vLLM server with VibeVoice-ASR:

```bash

vllm serve microsoft/VibeVoice-ASR \
    --omni \
    --port 8000 \
    --trust-remote-code

## Testing

Run the ASR client to test transcription:

```bash

python examples/online_serving/vibevoice_asr/openai_asr_client.py audio.wav


python examples/online_serving/vibevoice_asr/openai_asr_client.py audio.wav --url http://localhost:8000
``

### curl Command

```bash

curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"microsoft/VibeVoice-ASR","messages":[{"role":"system","content":"You are a helpful assistant that transcribes audio input into text output in JSON format."},{"role":"user","content":[{"type":"audio_url","audio_url":{"url":"data:audio/wav;base64,..."}},{"type":"text","text":"Please transcribe this audio"}]}],"max_tokens":4096,"temperature":0.0,"stream":true}'

```
## References

- [VibeVoice GitHub](https://github.com/microsoft/VibeVoice)
- [VibeVoice-ASR HuggingFace](https://huggingface.co/microsoft/VibeVoice-ASR)
