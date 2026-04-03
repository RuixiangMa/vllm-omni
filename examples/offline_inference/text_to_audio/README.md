# Text-To-Audio

The offline `text_to_audio.py` example supports diffusion-based text-to-audio models, including `stabilityai/stable-audio-open-1.0` and `meituan-longcat/LongCat-AudioDiT-1B`.

## Prerequisites

If you use a gated model (for example `stabilityai/stable-audio-open-1.0`), ensure you have access:

1. **Accept Model License**: Visit the model page on Hugging Face and accept the user agreement.
2. **Authenticate**: Log in to Hugging Face locally to access the gated model.
   ```bash
   huggingface-cli login
   ```

## Local CLI Usage

Stable Audio example:

```bash
python text_to_audio.py \
  --model stabilityai/stable-audio-open-1.0 \
  --prompt "The sound of a hammer hitting a wooden surface" \
  --negative-prompt "Low quality" \
  --seed 42 \
  --guidance-scale 7.0 \
  --audio-length 10.0 \
  --num-inference-steps 100 \
  --sample-rate 44100 \
  --output stable_audio_output.wav
```

LongCat-AudioDiT example:

```bash
python text_to_audio.py \
  --model meituan-longcat/LongCat-AudioDiT-1B \
  --prompt "A calm ocean wave ambience with soft wind in the background" \
  --negative-prompt "distorted, clipping, noisy" \
  --seed 42 \
  --guidance-scale 4.0 \
  --audio-length 5.0 \
  --num-inference-steps 16 \
  --sample-rate 24000 \
  --output longcat_audio_dit_output.wav
```

Key arguments:

- `--model`: model name or local path.
- `--prompt`: text description.
- `--negative-prompt`: negative prompt for classifier-free guidance.
- `--seed`: integer seed for deterministic generation.
- `--guidance-scale`: classifier-free guidance scale.
- `--audio-length`: audio duration in seconds.
- `--num-inference-steps`: diffusion sampling steps (more steps usually improve quality but take longer).
- `--sample-rate`: output WAV sample rate. Set this to match the selected model.
- `--output`: path to save the generated WAV file.
