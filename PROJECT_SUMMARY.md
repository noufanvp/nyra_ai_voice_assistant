# 🤖 Hybrid AI Voice Assistant — Project Summary

> A **fully local, low-latency voice assistant** that listens via microphone, transcribes speech locally, reasons via Groq's cloud LLM, and speaks back using a local TTS engine — all in real time.

---

## 📁 Project Structure

```
voice_assistant_hybrid/
│
├── main.py              ← Entry point + thread orchestration
├── config.py            ← All settings in one place (dataclasses)
├── requirements.txt     ← Python dependencies
├── .env                 ← Your Groq API key
├── .env.example         ← Template for the .env
├── models/              ← Kokoro TTS model files (auto-downloaded)
│
├── core/                ← The 4 core engine modules
│   ├── audio_io.py      ← Mic capture + WebRTC VAD + audio playback
│   ├── stt.py           ← Speech-to-text (faster-whisper)
│   ├── llm.py           ← Groq API client (streaming)
│   └── tts.py           ← Text-to-speech (kokoro-onnx / pyttsx3)
│
└── tests/               ← Unit tests for all 4 modules
    ├── test_audio.py
    ├── test_stt.py
    ├── test_llm.py
    └── test_tts.py
```

---

## ⚙️ How It Works — Pipeline

The app runs **3 background threads** connected by queues:

```
Microphone
    │
    ▼
[Thread 1 — vad_worker]          core/audio_io.py
  WebRTC VAD listens for speech
  Stops recording after 0.6s of silence
    │
    │  audio_queue  (numpy float32 array)
    ▼
[Thread 2 — stt_worker]          core/stt.py
  faster-whisper transcribes audio → text
  Discards empty / silent transcripts
    │
    │  transcript_queue  (string)
    ▼
[Thread 3 — llm_tts_worker]      core/llm.py + core/tts.py
  Sends text to Groq API (streaming)
  As each sentence arrives → synthesized to audio immediately
  Plays audio while next sentence is still being generated
    │
    ▼
  Speaker 🔊
```

---

## 🧩 Module Breakdown

| Module | Role | Key Class / Function |
|---|---|---|
| `audio_io.py` | Mic input + Voice Activity Detection | `VADRecorder`, `play_audio()` |
| `stt.py` | Local speech transcription | `WhisperTranscriber` |
| `llm.py` | Groq streaming + sentence splitting | `GroqClientWrapper.stream_sentences()` |
| `tts.py` | Speech synthesis (local) | `LocalTTSEngine` (kokoro-onnx → pyttsx3 fallback) |
| `config.py` | All tunable settings | `CONFIG` singleton |
| `main.py` | Thread orchestration + lifecycle | `vad_worker`, `stt_worker`, `llm_tts_worker` |

---

## 🔑 Key Design Decisions

| Feature | Detail |
|---|---|
| **VAD** | WebRTC VAD (lightweight, no PyTorch). Stops recording after **0.6s** of silence |
| **STT** | `faster-whisper base.en`, INT8 quantized on CPU, ~150 MB model auto-downloaded |
| **LLM** | `llama-3.1-8b-instant` via Groq streaming. Sentences yielded one-by-one for minimal latency |
| **TTS** | `kokoro-onnx` as primary (high quality, fully local). Falls back to `pyttsx3` if unavailable |
| **Conversation Memory** | Last **5 exchanges** (10 messages) kept in context for multi-turn conversations |
| **Error Handling** | Groq rate limits → exponential backoff (1s, 2s, 4s). API down → friendly spoken error message |
| **Concurrency** | A sub-thread inside Thread 3 interleaves TTS playback with ongoing LLM token generation |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Speech-to-Text (STT)** | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — `base.en`, INT8, CPU |
| **Language Model (LLM)** | [Groq API](https://console.groq.com) — `llama-3.1-8b-instant`, streaming |
| **Text-to-Speech (TTS)** | [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) — voice: `af_heart` |
| **TTS Fallback** | [pyttsx3](https://pyttsx3.readthedocs.io/) — offline, no models needed |
| **Voice Activity Detection** | [webrtcvad](https://github.com/wiseman/py-webrtcvad) — 30ms frame analysis |
| **Audio I/O** | [sounddevice](https://python-sounddevice.readthedocs.io/) + PortAudio |
| **Config** | Python `dataclasses` + `python-dotenv` |

---

## ⚙️ Configuration (config.py)

All settings are tunable in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `stt.model_size` | `base.en` | Whisper model size |
| `stt.compute_type` | `int8` | Quantization type |
| `audio.silence_duration_s` | `0.6` | Silence before recording stops |
| `audio.min_speech_duration_s` | `0.2` | Minimum valid speech duration |
| `audio.max_recording_duration_s` | `30.0` | Safety cap on recording length |
| `llm.model` | `llama-3.1-8b-instant` | Groq model ID |
| `llm.max_tokens` | `200` | Max response tokens |
| `llm.temperature` | `0.7` | LLM creativity level |
| `tts.voice` | `af_heart` | Kokoro voice ID |
| `tts.speed` | `1.0` | Speaking speed multiplier |

---

## 🚨 Edge Case Handling

| Case | Behavior |
|---|---|
| Silent mic / no speech | VAD discards audio; no LLM call made |
| Speech < 0.2s | Treated as noise and ignored |
| Groq API unavailable | Speaks: *"AI service currently unavailable"* |
| Groq rate limit (HTTP 429) | Exponential backoff (1s → 2s → 4s), then error message |
| kokoro-onnx missing | Automatically falls back to pyttsx3 |
| Ctrl+C | Stops all threads + audio playback cleanly |

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Skip tests that require the Groq API key
pytest tests/ -v -k "not live"

# Only fast, no-hardware tests
pytest tests/test_llm.py -v -k "not live"
```

---

## 🚀 Quick Start

```bash
# 1. Navigate to project
cd voice_assistant_hybrid

# 2. Activate virtual environment
source .venv/bin/activate

# 3. Install dependencies (first time only)
pip install -r requirements.txt

# 4. Run the assistant
python main.py
```

> **First run** auto-downloads:
> - Whisper `base.en` model (~150 MB) via HuggingFace
> - Kokoro TTS model + voices (~160 MB) from GitHub

Press **Ctrl+C** to exit cleanly.

---

## ✅ Current Project Status

- ✅ API key configured in `.env`
- ✅ All 4 core modules implemented (`audio_io`, `stt`, `llm`, `tts`)
- ✅ 3-thread pipeline with inter-thread queues
- ✅ Sentence-level streaming for minimal TTS latency
- ✅ Fallback TTS engine (pyttsx3) if kokoro unavailable
- ✅ Exponential backoff on Groq rate limits
- ✅ Multi-turn conversation memory (last 5 exchanges)
- ✅ Unit tests for all modules
- ⬜ Models auto-download on first run (~310 MB total)
