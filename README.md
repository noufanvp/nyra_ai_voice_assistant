# Hybrid AI Voice Assistant

A **low-latency, fully local** voice assistant combining:
- 🎙️ **faster-whisper** (local STT, `base.en` INT8 on CPU)
- 🔊 **kokoro-onnx** (local TTS, zero-API-latency)
- 🤖 **Groq API** (`llama-3.1-8b-instant`, streaming)
- 🎤 **Silero VAD** (auto speech-end detection)

---

## Requirements

- Python 3.10 – 3.12
- Working microphone (ALSA / PulseAudio / PipeWire on Linux)
- Internet connection (first run downloads Whisper + kokoro models)
- A Groq API key (free tier available at [console.groq.com](https://console.groq.com))

---

## Installation

### 1. Clone / navigate to project directory
```bash
cd voice_assistant_hybrid
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install system audio library (Linux only)
```bash
sudo apt-get install portaudio19-dev espeak-ng   # Ubuntu/Debian
# or:
sudo dnf install portaudio-devel espeak-ng       # Fedora
```

### 4. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure your Groq API key
```bash
cp .env.example .env
# Edit .env and replace 'your_groq_api_key_here' with your real key
```

### 6. Download Kokoro TTS model files
The app will **auto-download** kokoro model files (~160MB total) on first run.
Alternatively, download manually:
```bash
mkdir -p models
# kokoro-v1.0.onnx
wget -O models/kokoro-v1.0.onnx \
  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx

# voices-v1.0.bin
wget -O models/voices-v1.0.bin \
  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

---

## Running the Assistant

```bash
python main.py
```

On first run, the `base.en` Whisper model (~150MB) is downloaded automatically via HuggingFace.

### Expected output
```
============================================================
  Hybrid AI Voice Assistant
  STT: faster-whisper | LLM: Groq | TTS: kokoro-onnx
  Press Ctrl+C to exit
============================================================
[00:01:05] INFO     main — All components ready. TTS backend: kokoro-onnx
[00:01:05] INFO     main — Voice assistant is live. Speak into your microphone.
[00:01:08] INFO     vad_worker — 🎙  Listening… (speak now)
[00:01:11] INFO     main — [STT latency: 234 ms] Transcript: 'What is the capital of France?'
[00:01:12] INFO     main — Groq first-token latency: 412 ms
[00:01:12] INFO     main — [TTS: 89 ms] Speaking: 'The capital of France is Paris.'
```

Press **Ctrl+C** to exit cleanly.

---

## Running Unit Tests

```bash
# All tests (some require hardware / model download)
pytest tests/ -v

# Skip live Groq API test (no key needed)
pytest tests/ -v -k "not live"

# Only fast, no-hardware tests
pytest tests/test_llm.py -v -k "not live"
```

---

## Configuration

All settings are in `config.py`. Key tunable parameters:

| Parameter | Default | Description |
|---|---|---|
| `stt.model_size` | `base.en` | Whisper model size |
| `stt.compute_type` | `int8` | Quantization (int8/float16/float32) |
| `audio.silence_duration_s` | `0.6` | Silence before recording stops |
| `llm.model` | `llama-3.1-8b-instant` | Groq model |
| `llm.max_tokens` | `200` | Max response tokens |
| `tts.voice` | `af_heart` | Kokoro voice ID |
| `tts.speed` | `1.0` | Speaking speed multiplier |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     main.py                             │
│                                                         │
│  Thread 1 (vad_worker)                                  │
│  ┌─────────────────────┐                                │
│  │ Mic → Silero VAD    │──► audio_queue                 │
│  └─────────────────────┘                                │
│                                                         │
│  Thread 2 (stt_worker)                                  │
│  ┌─────────────────────────────┐                        │
│  │ audio_queue → faster-whisper│──► transcript_queue    │
│  └─────────────────────────────┘                        │
│                                                         │
│  Thread 3 (llm_tts_worker)                              │
│  ┌────────────────────────────────────────────────────┐ │
│  │ transcript_queue → Groq (streaming)                │ │
│  │   ├─ sentence 1 ──► kokoro-onnx ──► play_audio()  │ │
│  │   ├─ sentence 2 ──► kokoro-onnx ──► play_audio()  │ │
│  │   └─ sentence N ──► kokoro-onnx ──► play_audio()  │ │
│  └────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

---

## Edge Case Handling

| Case | Behavior |
|---|---|
| Silent mic / no speech | VAD discards; no LLM call made |
| Groq API unavailable | Speaks: *"AI service unavailable"* |
| Groq rate limit (429) | Exponential backoff (1s, 2s, 4s), then error message |
| kokoro-onnx missing | Falls back to pyttsx3 automatically |
| Ctrl+C | Stops all threads + audio playback cleanly |

---

## Troubleshooting

**No audio devices found:**
```bash
python -c "import sounddevice; print(sounddevice.query_devices())"
```

**PortAudio error on Linux:**
```bash
sudo apt-get install portaudio19-dev
pip install --force-reinstall sounddevice
```

**Whisper model download fails:**
```bash
pip install huggingface_hub
huggingface-cli download Systran/faster-whisper-base.en
```

**kokoro-onnx download fails:**
Manually download from: https://github.com/thewh1teagle/kokoro-onnx/releases
and place in the `models/` directory.
