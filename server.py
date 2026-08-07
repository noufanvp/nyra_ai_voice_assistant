"""
server.py — FastAPI & WebSocket Server for Nyra Mobile App Backend.

Provides:
  - GET / : Health check and service status
  - GET /app : Full PySide6-matched Mobile Web App (Android & iOS)
  - GET /assets/avatar.png : Avatar image asset
  - GET /serviceworker.js & /favicon.ico : PWA support endpoints
  - POST /api/chat : REST endpoint for text query & audio response
  - WS /ws/voice : Bidirectional WebSocket endpoint for real-time mobile interaction
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import wave
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

from config import CONFIG
from core.stt import WhisperTranscriber
from core.tts import LocalTTSEngine
from core.llm import GroqClientWrapper

# ---------------------------------------------------------------------------
# Setup Logging & App
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, CONFIG.logging.level.upper(), logging.INFO),
    format=CONFIG.logging.fmt,
    datefmt=CONFIG.logging.datefmt,
)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
logger = logging.getLogger("nyra.server")

app = FastAPI(
    title="Nyra AI Voice Assistant Server",
    description="Backend API & WebSockets for Android & iOS mobile applications",
    version="1.0.0",
)

# Enable CORS for mobile development & Web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Lazy-Loaded Global AI Engines (Prevents startup memory spikes & instant port binding)
# ---------------------------------------------------------------------------
_stt_engine = None
_tts_engine = None
_llm_engine = None

def get_stt_engine():
    global _stt_engine
    if _stt_engine is None:
        try:
            logger.info("Lazy-loading STT Engine (Whisper)...")
            import gc
            _stt_engine = WhisperTranscriber(CONFIG.stt)
            gc.collect()
        except Exception as exc:
            logger.error("Failed to load STT engine: %s", exc)
    return _stt_engine

def get_tts_engine():
    global _tts_engine
    if _tts_engine is None:
        try:
            logger.info("Lazy-loading TTS Engine (Kokoro)...")
            import gc
            _tts_engine = LocalTTSEngine(CONFIG.tts)
            gc.collect()
        except Exception as exc:
            logger.error("Failed to load TTS engine: %s", exc)
    return _tts_engine

def get_llm_engine():
    global _llm_engine
    if _llm_engine is None:
        try:
            _llm_engine = GroqClientWrapper(CONFIG.llm)
        except Exception as exc:
            logger.error("Failed to load LLM engine: %s", exc)
    return _llm_engine


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def audio_to_base64_wav(pcm_float32: np.ndarray, sample_rate: int) -> str:
    """Convert float32 PCM array to Base64 encoded WAV string."""
    if len(pcm_float32) == 0:
        return ""
    audio_clamped = np.clip(pcm_float32, -1.0, 1.0)
    audio_int16 = (audio_clamped * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def base64_wav_to_numpy(b64_str: str) -> tuple[np.ndarray, int]:
    """
    Convert Base64 encoded audio (WAV, WebM, Opus, MP4, AAC, OGG) to float32 numpy array & sample rate.
    Uses wave module, soundfile, and temporary file PyAV as robust fallbacks for mobile browsers.
    """
    if not b64_str:
        return np.array([], dtype=np.float32), 16000

    try:
        raw_bytes = base64.b64decode(b64_str)
    except Exception as err:
        logger.warning("Base64 decode error: %s", err)
        return np.array([], dtype=np.float32), 16000

    if not raw_bytes or len(raw_bytes) < 300:
        logger.debug("Audio payload too short (%d bytes), skipping.", len(raw_bytes) if raw_bytes else 0)
        return np.array([], dtype=np.float32), 16000

    # 1. Try standard WAV parser first
    buf = io.BytesIO(raw_bytes)
    try:
        with wave.open(buf, "rb") as wf:
            sr = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
            pcm_int16 = np.frombuffer(frames, dtype=np.int16)
            pcm_float32 = pcm_int16.astype(np.float32) / 32768.0
            return pcm_float32, sr
    except Exception:
        pass

    # 2. Try soundfile parser
    try:
        import soundfile as sf
        buf.seek(0)
        data, sr = sf.read(buf, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data, sr
    except Exception:
        pass

    # 3. Temporary file PyAV fallback for WebM/Opus, MP4/AAC, OGG from mobile browsers
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name

    try:
        import av
        container = av.open(tmp_path)
        resampler = av.AudioResampler(format="flt", layout="mono", rate=16000)
        samples = []
        for frame in container.decode(audio=0):
            resample_out = resampler.resample(frame)
            if resample_out:
                for rf in resample_out:
                    samples.append(rf.to_ndarray())
        container.close()
        if samples:
            audio_data = np.concatenate(samples, axis=1).squeeze(0)
            return audio_data.astype(np.float32), 16000
    except Exception as exc:
        logger.warning("Audio decoding notice (%d bytes): %s", len(raw_bytes), exc)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return np.array([], dtype=np.float32), 16000



# ---------------------------------------------------------------------------
# Static Assets & PWA Endpoints
# ---------------------------------------------------------------------------
@app.get("/assets/avatar.png")
def get_avatar():
    """Serve the circular avatar character image."""
    avatar_path = Path(__file__).parent / "assets" / "avatar.png"
    if avatar_path.exists():
        return FileResponse(str(avatar_path), media_type="image/png")
    return Response(status_code=404)


@app.get("/manifest.json")
def get_manifest():
    """Serve Web App Manifest for smartphone PWA installation."""
    manifest_data = {
        "name": "Nyra AI Voice Assistant",
        "short_name": "Nyra AI",
        "description": "Cyber-Glassmorphic AI Voice Assistant powered by Whisper, Kokoro TTS, and Llama 3.",
        "start_url": "/app",
        "display": "standalone",
        "orientation": "portrait",
        "background_color": "#090A0F",
        "theme_color": "#090A0F",
        "icons": [
            {
                "src": "/assets/avatar.png",
                "sizes": "192x192 512x512",
                "type": "image/png",
                "purpose": "any maskable"
            }
        ]
    }
    return Response(content=json.dumps(manifest_data), media_type="application/json")


@app.get("/serviceworker.js")
def service_worker():
    """Serve active PWA Service Worker script for offline caching and installation."""
    sw_script = """
const CACHE_NAME = 'nyra-pwa-v1';
const ASSETS = ['/app', '/assets/avatar.png', '/manifest.json'];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE_NAME).then((cache) => cache.addAll(ASSETS)));
  self.skipWaiting();
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', (e) => {
  if (e.request.mode === 'navigate') {
    e.respondWith(fetch(e.request).catch(() => caches.match('/app')));
  } else {
    e.respondWith(caches.match(e.request).then((res) => res || fetch(e.request)));
  }
});
"""
    return Response(content=sw_script, media_type="application/javascript")


@app.get("/favicon.ico")
def favicon():
    """Return 204 No Content for favicon requests to prevent 404 noise."""
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Mobile Web App Endpoint (Matches PySide6 Desktop UI/UX)
# ---------------------------------------------------------------------------
@app.api_route("/", methods=["GET", "HEAD"])
@app.api_route("/app", methods=["GET", "HEAD"])
def mobile_web_app():
    """Serves a modern mobile web app with exact UI/UX matching PySide6 desktop app."""
    html_content = r"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
  <title>Nyra AI — Voice Assistant</title>
  <link rel="manifest" href="/manifest.json">
  <meta name="theme-color" content="#090A0F">
  <meta name="mobile-web-app-capable" content="yes">
  <meta name="apple-mobile-web-app-capable" content="yes">
  <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
  <meta name="apple-mobile-web-app-title" content="Nyra AI">
  <link rel="apple-touch-icon" href="/assets/avatar.png">
  <style>
    :root[data-theme="dark"] {
      --bg: #090A0F;
      --surface: #141722;
      --surface-alt: #1E2235;
      --accent: #6366F1;
      --accent-hover: #818CF8;
      --cyan: #06B6D4;
      --purple: #A855F7;
      --emerald: #10B981;
      --amber: #F59E0B;
      --blue: #3B82F6;
      --red: #EF4444;
      --text: #F8FAFC;
      --text-muted: #94A3B8;
      --user-bubble: #312E81;
      --user-bubble-text: #FFFFFF;
      --assistant-bubble: #131722;
      --dock-bg: rgba(14, 17, 30, 0.96);
      --dock-border: rgba(255, 255, 255, 0.10);
      --dock-sep: rgba(255, 255, 255, 0.08);
      --btn-bg: rgba(22, 27, 44, 0.80);
      --titlebar-bg: #090A0F;
      --titlebar-border: rgba(255, 255, 255, 0.06);
    }

    :root[data-theme="light"] {
      --bg: #F0F4F8;
      --surface: #FFFFFF;
      --surface-alt: #E2E8F0;
      --accent: #4F46E5;
      --accent-hover: #6366F1;
      --cyan: #0891B2;
      --purple: #9333EA;
      --emerald: #059669;
      --amber: #D97706;
      --blue: #2563EB;
      --red: #DC2626;
      --text: #0F172A;
      --text-muted: #64748B;
      --user-bubble: #EEF2FF;
      --user-bubble-text: #312E81;
      --assistant-bubble: #F8FAFC;
      --dock-bg: rgba(255, 255, 255, 0.96);
      --dock-border: rgba(148, 163, 184, 0.30);
      --dock-sep: rgba(148, 163, 184, 0.25);
      --btn-bg: rgba(241, 245, 249, 0.90);
      --titlebar-bg: #F8FAFC;
      --titlebar-border: rgba(148, 163, 184, 0.25);
    }

    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; -webkit-tap-highlight-color: transparent; }
    
    html, body {
      height: 100%;
      height: 100dvh;
      width: 100vw;
      margin: 0;
      padding: 0;
      overflow: hidden;
      position: fixed;
      top: 0; left: 0; right: 0; bottom: 0;
      background-color: var(--bg);
      color: var(--text);
      display: flex;
      flex-direction: column;
      transition: background-color 0.3s ease, color 0.3s ease;
    }

    /* --- Custom Title Bar --- */
    header {
      height: 48px;
      flex: 0 0 48px;
      padding: 0 14px;
      background: var(--titlebar-bg);
      border-bottom: 1px solid var(--titlebar-border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      user-select: none;
    }
    .brand-col { display: flex; align-items: center; gap: 8px; }
    .brand-dot { color: var(--cyan); font-size: 16px; }
    .title-col { display: flex; flex-direction: column; }
    .app-title { font-size: 13px; font-weight: 700; color: var(--text); line-height: 1.2; }
    .app-subtitle { font-size: 9px; color: var(--text-muted); }

    /* --- Visualizer Stage Canvas --- */
    #stage-container {
      position: relative;
      height: 140px;
      flex: 0 0 140px;
      background: var(--surface);
      border-bottom: 1px solid var(--dock-sep);
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
    }
    #vis-canvas { width: 100%; height: 100%; position: absolute; top: 0; left: 0; }
    
    .avatar-wrapper {
      position: relative;
      width: 80px;
      height: 80px;
      border-radius: 50%;
      z-index: 2;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .avatar-img {
      width: 72px;
      height: 72px;
      border-radius: 50%;
      object-fit: cover;
      border: 2px solid var(--accent);
      box-shadow: 0 0 20px rgba(99, 102, 241, 0.3);
      transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }

    /* --- Status Badge & Telemetry Bar --- */
    .status-bar {
      padding: 6px 14px 2px 14px;
      flex: 0 0 auto;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .status-badge {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      padding: 3px 10px;
      border-radius: 12px;
      font-size: 9px;
      font-weight: 700;
      letter-spacing: 0.6px;
      text-transform: uppercase;
      background: rgba(16, 185, 129, 0.15);
      color: var(--emerald);
      border: 1px solid var(--emerald);
      transition: all 0.3s ease;
    }
    .badge-dot { width: 5px; height: 5px; border-radius: 50%; background: currentColor; }
    .telemetry-tag { font-size: 9px; font-weight: 700; color: var(--cyan); letter-spacing: 0.6px; }

    /* --- Streaming Chat Display --- */
    #chat-display {
      flex: 1 1 auto;
      min-height: 0;
      overflow-y: auto;
      padding: 10px 14px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .msg-row { display: flex; gap: 8px; align-items: flex-start; max-width: 88%; }
    .msg-row.user { align-self: flex-end; flex-direction: row-reverse; }
    .msg-row.assistant { align-self: flex-start; }
    
    .chat-avatar {
      width: 24px;
      height: 24px;
      border-radius: 50%;
      object-fit: cover;
      border: 1px solid var(--accent);
      flex-shrink: 0;
    }
    .chat-bubble {
      padding: 8px 12px;
      border-radius: 14px;
      font-size: 0.85rem;
      line-height: 1.4;
      word-break: break-word;
    }
    .msg-row.user .chat-bubble {
      background: var(--user-bubble);
      color: var(--user-bubble-text);
      border-bottom-right-radius: 4px;
    }
    .msg-row.assistant .chat-bubble {
      background: var(--assistant-bubble);
      color: var(--text);
      border: 1px solid var(--dock-sep);
      border-bottom-left-radius: 4px;
    }

    /* --- Student Preset Chips Bar --- */
    .preset-chips-bar {
      display: flex;
      gap: 6px;
      padding: 4px 14px;
      overflow-x: auto;
      white-space: nowrap;
      scrollbar-width: none;
      flex-shrink: 0;
    }
    .preset-chips-bar::-webkit-scrollbar { display: none; }
    .chip-btn {
      background: var(--surface-alt);
      color: var(--text);
      border: 1px solid var(--dock-border);
      border-radius: 12px;
      padding: 5px 10px;
      font-size: 10px;
      font-weight: 600;
      cursor: pointer;
      flex-shrink: 0;
      transition: all 0.2s ease;
    }
    .chip-btn:hover, .chip-btn:active {
      background: var(--accent);
      color: #FFFFFF;
      border-color: var(--accent-hover);
    }

    /* --- Floating Control Dock Nav Bar --- */
    .dock-area {
      flex: 0 0 auto;
      padding: 6px 10px calc(8px + env(safe-area-inset-bottom, 0px)) 10px;
      width: 100%;
      max-width: 480px;
      margin: 0 auto;
      display: flex;
      justify-content: center;
      background: transparent;
    }
    .control-dock {
      width: 100%;
      height: 62px;
      background: var(--dock-bg);
      border: 1px solid var(--dock-border);
      border-radius: 31px;
      padding: 4px 6px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 4px;
      backdrop-filter: blur(16px);
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    }
    .dock-btn {
      flex: 1;
      max-width: 64px;
      min-width: 44px;
      height: 52px;
      border-radius: 14px;
      background: var(--btn-bg);
      border: 1px solid var(--dock-sep);
      color: var(--text-muted);
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 2px;
      cursor: pointer;
      font-size: 7px;
      font-weight: 700;
      letter-spacing: 0.6px;
      text-transform: uppercase;
      transition: all 0.2s ease;
      outline: none;
    }
    .dock-btn svg { width: 18px; height: 18px; fill: none; stroke: currentColor; stroke-width: 2.2; stroke-linecap: round; stroke-linejoin: round; }
    .dock-btn:active { transform: scale(0.94); }
    
    .dock-btn.hero {
      flex: 1.4;
      max-width: 90px;
      height: 56px;
      border-radius: 16px;
      background: linear-gradient(135deg, #4F46E5, #7C3AED);
      color: #FFFFFF;
      border: 1.5px solid #A78BFA;
      box-shadow: 0 0 16px rgba(124, 58, 237, 0.4);
      user-select: none;
      -webkit-user-select: none;
      touch-action: manipulation;
    }
    .dock-btn.hero.recording {
      background: linear-gradient(135deg, #B91C1C, #EF4444) !important;
      color: #FFFFFF !important;
      border: 1.5px solid #FCA5A5 !important;
      box-shadow: 0 0 20px rgba(239, 68, 68, 0.6) !important;
      animation: pulse-ring 1.2s infinite;
    }
    @keyframes pulse-ring {
      0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
      70% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
      100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }
    .dock-btn.hero.sleeping {
      background: linear-gradient(135deg, #312E81, #4338CA);
      color: #FFFFFF;
      border: 1.5px solid #818CF8;
    }
    .dock-btn.muted {
      background: linear-gradient(135deg, #7F1D1D, #DC2626);
      color: #FFFFFF;
      border: 1.5px solid #FF6B6B;
    }

    .dock-sep { width: 1px; height: 24px; background: var(--dock-sep); }

    /* --- Settings Drawer Modal --- */
    .modal-overlay {
      position: fixed;
      top: 0; left: 0; right: 0; bottom: 0;
      background: rgba(0, 0, 0, 0.6);
      backdrop-filter: blur(6px);
      display: flex;
      align-items: flex-end;
      z-index: 100;
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.3s ease;
    }
    .modal-overlay.open { opacity: 1; pointer-events: auto; }
    .modal-drawer {
      width: 100%;
      background: var(--surface);
      border-top-left-radius: 24px;
      border-top-right-radius: 24px;
      padding: 24px;
      border-top: 1px solid var(--dock-border);
      display: flex;
      flex-direction: column;
      gap: 16px;
      transform: translateY(100%);
      transition: transform 0.3s cubic-bezier(0.1, 0.9, 0.2, 1);
    }
    .modal-overlay.open .modal-drawer { transform: translateY(0); }
    .drawer-header { display: flex; justify-content: space-between; align-items: center; }
    .drawer-title { font-size: 16px; font-weight: 700; }
    .close-btn { background: transparent; border: none; color: var(--text-muted); font-size: 20px; cursor: pointer; }
    .info-card { background: var(--surface-alt); padding: 12px; border-radius: 12px; font-size: 12px; line-height: 1.6; }
  </style>
</head>
<body>

  <!-- Header / Title Bar -->
  <header>
    <div class="brand-col">
      <span class="brand-dot">◈</span>
      <div class="title-col">
        <span class="app-title">Nyra AI</span>
        <span class="app-subtitle">Al Irshad Public School · Mentored by Aitute</span>
      </div>
    </div>
  </header>

  <!-- AI Character Visualizer Stage -->
  <div id="stage-container">
    <canvas id="vis-canvas"></canvas>
    <div class="avatar-wrapper">
      <img src="/assets/avatar.png" alt="Nyra Avatar" class="avatar-img" id="avatar-img" onerror="this.src='data:image/svg+xml;utf8,<svg xmlns=\'http://www.w3.org/2000/svg\' width=\'96\' height=\'96\' viewBox=\'0 0 96 96\'><circle cx=\'48\' cy=\'48\' r=\'46\' fill=\'%236366F1\'/></svg>'">
    </div>
  </div>

  <!-- Live Status Badge Row -->
  <div class="status-bar">
    <div class="status-badge" id="status-badge">
      <span class="badge-dot"></span>
      <span id="badge-text">IDLE</span>
    </div>
    <span class="telemetry-tag">● CYBER ONLINE [EN]</span>
  </div>

  <!-- Streaming Chat Display -->
  <div id="chat-display"></div>

  <!-- Student Quick Preset Chips -->
  <div class="preset-chips-bar">
    <button class="chip-btn" onclick="sendPresetText('What is the previous month performance of my son Zain?')">📊 Zain's Performance Report</button>
    <button class="chip-btn" onclick="sendPresetText('What is photosynthesis?')">🌱 Photosynthesis</button>
    <button class="chip-btn" onclick="sendPresetText('What is Pythagorean theorem?')">📐 Pythagorean Theorem</button>
    <button class="chip-btn" onclick="sendPresetText('What is Newton\'s first law of motion?')">⚛️ Newton's 1st Law</button>
    <button class="chip-btn" onclick="sendPresetText('What is Python programming language?')">💻 Python</button>
    <button class="chip-btn" onclick="sendPresetText('How to study effectively?')">📚 Study Tips</button>
  </div>

  <!-- Floating Cyber Control Dock Nav Bar -->
  <div class="dock-area">
    <div class="control-dock">
      
      <!-- 1. Mute Button -->
      <button class="dock-btn" id="btn-mute">
        <svg viewBox="0 0 24 24"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="22"/></svg>
        <span id="label-mute">MIC ON</span>
      </button>

      <div class="dock-sep"></div>

      <!-- 2. Settings Button -->
      <button class="dock-btn" id="btn-settings">
        <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
        <span>SETTINGS</span>
      </button>

      <div class="dock-sep"></div>

      <!-- 3. CENTER HERO MIC BUTTON (Tap & Hold to Speak) -->
      <button class="dock-btn hero sleeping" id="btn-wake">
        <svg viewBox="0 0 24 24" id="wake-svg"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="22"/></svg>
        <span id="label-wake">HOLD TO SPEAK</span>
      </button>

      <div class="dock-sep"></div>

      <!-- 4. Clear Chat Button -->
      <button class="dock-btn" id="btn-clear">
        <svg viewBox="0 0 24 24"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
        <span>CLEAR</span>
      </button>

      <div class="dock-sep"></div>

      <!-- 5. Theme Toggle Button -->
      <button class="dock-btn" id="btn-theme">
        <svg viewBox="0 0 24 24" id="theme-svg"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>
        <span id="label-theme">LIGHT</span>
      </button>

    </div>
  </div>

  <!-- Settings Drawer Modal -->
  <div class="modal-overlay" id="modal-settings">
    <div class="modal-drawer">
      <div class="drawer-header">
        <span class="drawer-title">Nyra Settings & Telemetry</span>
        <button class="close-btn" id="close-drawer">✕</button>
      </div>
      <div class="info-card">
        <strong>Backend Architecture:</strong><br>
        • STT: faster-whisper (tiny.en, int8)<br>
        • LLM: Groq (llama-3.1-8b-instant)<br>
        • TTS: Kokoro-ONNX (af_heart)<br>
        • Mode: Tap & Hold Push-to-Talk
      </div>
    </div>
  </div>

  <script>
    // --- WebSocket Connection ---
    const wsUrl = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws/voice';
    let ws = new WebSocket(wsUrl);

    // --- State Variables ---
    let state = 'IDLE';
    let isMuted = false;
    let isAwake = false;
    let isHolding = false;
    let theme = 'dark';
    let mediaRecorder = null;
    let audioChunks = [];

    // --- DOM Elements ---
    const badgeText = document.getElementById('badge-text');
    const statusBadge = document.getElementById('status-badge');
    const chatDisplay = document.getElementById('chat-display');
    const btnMute = document.getElementById('btn-mute');
    const labelMute = document.getElementById('label-mute');
    const btnWake = document.getElementById('btn-wake');
    const labelWake = document.getElementById('label-wake');
    const btnTheme = document.getElementById('btn-theme');
    const labelTheme = document.getElementById('label-theme');
    const btnClear = document.getElementById('btn-clear');
    const btnSettings = document.getElementById('btn-settings');
    const modalSettings = document.getElementById('modal-settings');
    const closeDrawer = document.getElementById('close-drawer');

    // --- Canvas Visualizer setup ---
    const canvas = document.getElementById('vis-canvas');
    const ctx = canvas.getContext('2d');
    let phase = 0;

    function resizeCanvas() {
      canvas.width = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
    }
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();

    function drawVisualizer() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const w = canvas.width;
      const h = canvas.height;
      const cx = w / 2;
      const cy = h / 2;

      phase += 0.05;

      // Draw Sine Waves
      ctx.beginPath();
      ctx.lineWidth = 2;
      let waveColor = '#6366F1';
      if (state === 'LISTENING') waveColor = '#10B981';
      if (state === 'PROCESSING') waveColor = '#F59E0B';
      if (state === 'SPEAKING') waveColor = '#3B82F6';

      ctx.strokeStyle = waveColor;
      for (let x = 0; x < w; x += 4) {
        let amp = (state === 'IDLE') ? 6 : 22;
        let y = cy + Math.sin((x * 0.02) + phase) * amp;
        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      requestAnimationFrame(drawVisualizer);
    }
    drawVisualizer();

    // --- State Handler ---
    function setState(newState) {
      state = newState.toUpperCase();
      badgeText.innerText = state;
      
      statusBadge.style.borderColor = 'currentColor';
      if (state === 'LISTENING') {
        statusBadge.style.color = '#10B981';
        statusBadge.style.background = 'rgba(16, 185, 129, 0.15)';
      } else if (state === 'PROCESSING') {
        statusBadge.style.color = '#F59E0B';
        statusBadge.style.background = 'rgba(245, 158, 11, 0.15)';
      } else if (state === 'SPEAKING') {
        statusBadge.style.color = '#3B82F6';
        statusBadge.style.background = 'rgba(59, 130, 246, 0.15)';
      } else {
        statusBadge.style.color = '#10B981';
        statusBadge.style.background = 'rgba(16, 185, 129, 0.15)';
      }
    }

    // --- Audio Queue Management for Sequential Playback ---
    let audioQueue = [];
    let isPlayingAudio = false;
    let currentAudioElement = null;

    function stopAllAudio() {
      audioQueue = [];
      if (currentAudioElement) {
        currentAudioElement.pause();
        currentAudioElement.currentTime = 0;
        currentAudioElement = null;
      }
      isPlayingAudio = false;
    }

    function playNextAudioInQueue() {
      if (isPlayingAudio || audioQueue.length === 0) {
        return;
      }

      isPlayingAudio = true;
      const b64Audio = audioQueue.shift();
      const audio = new Audio("data:audio/wav;base64," + b64Audio);
      currentAudioElement = audio;

      audio.onended = () => {
        currentAudioElement = null;
        isPlayingAudio = false;
        playNextAudioInQueue();
      };

      audio.onerror = () => {
        currentAudioElement = null;
        isPlayingAudio = false;
        playNextAudioInQueue();
      };

      audio.play().catch(err => {
        console.warn("Autoplay notice: click/tap interaction required for audio playback.", err);
        currentAudioElement = null;
        isPlayingAudio = false;
        playNextAudioInQueue();
      });
    }

    function enqueueAudio(b64Audio) {
      audioQueue.push(b64Audio);
      playNextAudioInQueue();
    }

    function sendPresetText(query) {
      if (!query || !ws || ws.readyState !== WebSocket.OPEN) return;
      stopAllAudio();
      appendMessage(query, 'user');
      setState('PROCESSING');
      ws.send(JSON.stringify({ type: 'text', text: query }));
    }

    // --- WebSocket Messaging ---
    let currentAssistantBubble = null;

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'transcript') {
        currentAssistantBubble = null;
        stopAllAudio();
        if (data.text) {
          appendMessage(data.text, 'user');
          setState('PROCESSING');
        }
      } else if (data.type === 'response_chunk') {
        setState('SPEAKING');
        if (data.text) {
          appendAssistantChunk(data.text);
        }
        if (data.audio_b64) {
          enqueueAudio(data.audio_b64);
        }
      } else if (data.type === 'audio_chunk') {
        setState('SPEAKING');
        if (data.audio_b64) {
          enqueueAudio(data.audio_b64);
        }
      } else if (data.type === 'done') {
        currentAssistantBubble = null;
        if (!isPlayingAudio && audioQueue.length === 0) {
          setState('IDLE');
        }
      } else if (data.type === 'error') {
        currentAssistantBubble = null;
        stopAllAudio();
        setState('IDLE');
        if (data.message) {
          appendMessage("⚠️ " + data.message, 'assistant');
        }
      }
    };

    function appendAssistantChunk(text) {
      if (currentAssistantBubble) {
        currentAssistantBubble.textContent += " " + text;
      } else {
        const row = document.createElement('div');
        row.className = 'msg-row assistant';
        const img = document.createElement('img');
        img.src = '/assets/avatar.png';
        img.className = 'chat-avatar';
        img.alt = 'Nyra';
        row.appendChild(img);

        const bubble = document.createElement('div');
        bubble.className = 'chat-bubble';
        bubble.textContent = text;
        row.appendChild(bubble);

        currentAssistantBubble = bubble;
        chatDisplay.appendChild(row);
      }
      chatDisplay.scrollTop = chatDisplay.scrollHeight;
    }

    function appendMessage(text, sender) {
      currentAssistantBubble = null;
      const row = document.createElement('div');
      row.className = 'msg-row ' + sender;

      const bubble = document.createElement('div');
      bubble.className = 'chat-bubble';
      bubble.textContent = text;

      if (sender === 'assistant') {
        const img = document.createElement('img');
        img.src = '/assets/avatar.png';
        img.className = 'chat-avatar';
        img.alt = 'Nyra';
        row.appendChild(img);
      }

      row.appendChild(bubble);
      chatDisplay.appendChild(row);
      chatDisplay.scrollTop = chatDisplay.scrollHeight;
    }

    // --- Controls ---
    btnMute.onclick = () => {
      isMuted = !isMuted;
      btnMute.classList.toggle('muted', isMuted);
      labelMute.innerText = isMuted ? 'MUTED' : 'MIC ON';
    };

    // --- Web Audio Analyser & Tap-and-Hold Recording ---
    let audioCtx = null;
    let analyser = null;
    let micStream = null;
    let maxRecTimeout = null;

    async function startRecording() {
      stopAllAudio();
      if (isMuted) return;
      try {
        micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        try {
          audioCtx = new (window.AudioContext || window.webkitAudioContext)();
          const source = audioCtx.createMediaStreamSource(micStream);
          analyser = audioCtx.createAnalyser();
          analyser.fftSize = 32;
          source.connect(analyser);
        } catch (e) {
          console.warn("AudioContext visualizer notice:", e);
        }

        const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 'audio/webm;codecs=opus' :
                         MediaRecorder.isTypeSupported('audio/mp4') ? 'audio/mp4' :
                         MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : 'audio/wav';
        
        mediaRecorder = new MediaRecorder(micStream, { mimeType: mimeType });
        audioChunks = [];
        
        let recStartTime = Date.now();
        mediaRecorder.ondataavailable = e => { if (e.data.size > 0) audioChunks.push(e.data); };
        mediaRecorder.onstop = () => {
          const duration = Date.now() - recStartTime;
          if (duration < 300) {
            console.log("Tap too short (" + duration + "ms), skipping audio send.");
            if (micStream) micStream.getTracks().forEach(t => t.stop());
            if (audioCtx && audioCtx.state !== 'closed') audioCtx.close();
            return;
          }

          const blob = new Blob(audioChunks, { type: mimeType });
          const reader = new FileReader();
          reader.readAsDataURL(blob);
          reader.onloadend = () => {
            const b64 = reader.result.split(',')[1];
            if (b64 && b64.length > 200 && ws && ws.readyState === WebSocket.OPEN) {
              ws.send(JSON.stringify({ type: 'audio', audio_b64: b64 }));
            }
          };
          if (micStream) micStream.getTracks().forEach(t => t.stop());
          if (audioCtx && audioCtx.state !== 'closed') audioCtx.close();
        };

        mediaRecorder.start(100);
        isAwake = true;
        btnWake.classList.remove('sleeping');
        btnWake.classList.add('recording');
        labelWake.innerText = 'LISTENING...';
        setState('LISTENING');

        // Automatically cap max recording at 15 seconds to prevent accidental runaway recording
        if (maxRecTimeout) clearTimeout(maxRecTimeout);
        maxRecTimeout = setTimeout(() => {
          if (isHolding) {
            handlePressEnd();
          }
        }, 15000);
      } catch (err) {
        isHolding = false;
        alert('Microphone permission required! Please allow mic access.');
      }
    }

    function stopRecording() {
      if (maxRecTimeout) {
        clearTimeout(maxRecTimeout);
        maxRecTimeout = null;
      }
      if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
      }
      isAwake = false;
      btnWake.classList.remove('recording');
      btnWake.classList.add('sleeping');
      labelWake.innerText = 'HOLD TO SPEAK';
      setState('PROCESSING');
    }

    // Push-to-Talk (Hold and Release) Event Handlers
    function handlePressStart(e) {
      if (e) e.preventDefault();
      if (isMuted || isHolding) return;
      isHolding = true;
      startRecording();
    }

    function handlePressEnd(e) {
      if (e) e.preventDefault();
      if (!isHolding) return;
      isHolding = false;
      stopRecording();
    }

    btnWake.addEventListener('pointerdown', handlePressStart);
    btnWake.addEventListener('pointerup', handlePressEnd);
    btnWake.addEventListener('pointercancel', handlePressEnd);
    btnWake.addEventListener('pointerleave', (e) => { if (isHolding) handlePressEnd(e); });

    btnWake.addEventListener('touchstart', handlePressStart, { passive: false });
    btnWake.addEventListener('touchend', handlePressEnd, { passive: false });
    btnWake.addEventListener('touchcancel', handlePressEnd, { passive: false });

    btnClear.onclick = () => {
      chatDisplay.innerHTML = '';
      appendMessage("Chat cleared! Tap and hold the mic button to speak with Nyra.", "assistant");
    };

    btnTheme.onclick = () => {
      theme = (theme === 'dark') ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', theme);
      labelTheme.innerText = (theme === 'dark') ? 'LIGHT' : 'DARK';
    };

    btnSettings.onclick = () => modalSettings.classList.add('open');
    closeDrawer.onclick = () => modalSettings.classList.remove('open');
    modalSettings.onclick = (e) => { if (e.target === modalSettings) modalSettings.classList.remove('open'); };

    // --- Register PWA Service Worker ---
    if ('serviceWorker' in navigator) {
      window.addEventListener('load', () => {
        navigator.serviceWorker.register('/serviceworker.js').catch(err => console.warn('PWA SW notice:', err));
      });
    }
  </script>
</body>
</html>"""
    return HTMLResponse(content=html_content)


# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = "default_user"


class ChatResponse(BaseModel):
    user_message: str
    assistant_text: str
    audio_b64: str
    sample_rate: int


@app.api_route("/health", methods=["GET", "HEAD"])
def health_check():
    """Health check and engine status endpoint for Render load balancers."""
    return {
        "status": "online",
        "assistant": CONFIG.wake_word.assistant_name,
        "engines": {
            "stt": _stt_engine.cfg.model_size if _stt_engine else "ready",
            "tts": _tts_engine.backend_name if _tts_engine else "ready",
            "llm": _llm_engine.cfg.model if _llm_engine else "ready",
        },
        "mobile_web_app": "/app",
    }


@app.post("/api/chat", response_model=ChatResponse)
def text_chat(req: ChatRequest):
    """Processes a text query and returns assistant text and synthesized speech audio."""
    llm = get_llm_engine()
    tts = get_tts_engine()
    if not llm:
        raise HTTPException(status_code=503, detail="LLM Engine not available")

    full_text = ""
    for sentence in llm.stream_sentences(req.message):
        full_text += sentence + " "
    full_text = full_text.strip()

    audio_samples, sample_rate = np.array([], dtype=np.float32), 22050
    if tts and full_text:
        audio_samples, sample_rate = tts.synthesize(full_text)

    audio_b64 = audio_to_base64_wav(audio_samples, sample_rate)

    return ChatResponse(
        user_message=req.message,
        assistant_text=full_text,
        audio_b64=audio_b64,
        sample_rate=sample_rate,
    )


# ---------------------------------------------------------------------------
# WebSocket Real-time Voice Endpoint
# ---------------------------------------------------------------------------
@app.websocket("/ws/voice")
async def voice_websocket(websocket: WebSocket):
    """
    Bidirectional WebSocket connection for mobile clients.
    
    Handles:
      - JSON text query: {"type": "text", "text": "..."}
      - Base64 WAV/PCM audio query: {"type": "audio", "audio_b64": "..."}
    """
    await websocket.accept()
    logger.info("Mobile client connected via WebSocket.")

    # Automatically send introduction greeting speech on link open
    try:
        intro_text = "Hello! I am Nyra, an AI assistant made by the students of Al Irshad Public School with the help of Aitute. Tap and hold the mic button to speak with me!"
        tts = get_tts_engine()
        audio_b64 = ""
        sr = 22050
        if tts:
            samples, sr = tts.synthesize(intro_text)
            audio_b64 = audio_to_base64_wav(samples, sr)

        await websocket.send_json({
            "type": "response_chunk",
            "text": intro_text,
            "audio_b64": audio_b64,
            "sample_rate": sr,
        })
        await websocket.send_json({
            "type": "done",
            "full_text": intro_text,
        })
    except Exception as exc:
        logger.error("Error sending introduction greeting: %s", exc)

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "")

            user_query = ""

            if msg_type == "text":
                user_query = data.get("text", "")
            elif msg_type == "audio":
                b64_audio = data.get("audio_b64", "")
                stt = get_stt_engine()
                if b64_audio and stt:
                    try:
                        pcm, sr = base64_wav_to_numpy(b64_audio)
                        user_query = stt.transcribe(pcm, sample_rate=sr)
                        await websocket.send_json({
                            "type": "transcript",
                            "text": user_query,
                        })
                    except Exception as err:
                        logger.error("STT Error in WebSocket: %s", err)
                        await websocket.send_json({
                            "type": "error",
                            "message": "Failed to process audio transcription",
                        })
                        continue

            if not user_query:
                await websocket.send_json({
                    "type": "error",
                    "message": "No input detected",
                })
                continue

            # Stream LLM responses and TTS chunks
            llm = get_llm_engine()
            tts = get_tts_engine()
            if llm:
                full_reply = ""
                for sentence in llm.stream_sentences(user_query):
                    full_reply += sentence + " "

                    # 1. Send text response chunk instantly so client UI updates in <150ms
                    await websocket.send_json({
                        "type": "response_chunk",
                        "text": sentence,
                        "audio_b64": "",
                    })

                    # 2. Synthesize TTS audio asynchronously without blocking text delivery
                    if tts:
                        try:
                            samples, sr = await asyncio.to_thread(tts.synthesize, sentence)
                            if samples is not None and len(samples) > 0:
                                audio_b64 = audio_to_base64_wav(samples, sr)
                                await websocket.send_json({
                                    "type": "audio_chunk",
                                    "audio_b64": audio_b64,
                                    "sample_rate": sr,
                                })
                        except Exception as tts_err:
                            logger.warning("TTS streaming error: %s", tts_err)

                await websocket.send_json({
                    "type": "done",
                    "full_text": full_reply.strip(),
                })

    except WebSocketDisconnect:
        logger.info("Mobile client disconnected.")
    except Exception as exc:
        logger.error("WebSocket error: %s", exc)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    cert_file = Path(__file__).parent / "cert.pem"
    key_file = Path(__file__).parent / "key.pem"

    if cert_file.exists() and key_file.exists():
        logger.info("Starting Nyra Mobile Server on https://0.0.0.0:%d (Local SSL Enabled)", port)
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            ssl_keyfile=str(key_file),
            ssl_certfile=str(cert_file),
        )
    else:
        logger.info("Starting Nyra Mobile Server on http://0.0.0.0:%d", port)
        uvicorn.run(app, host="0.0.0.0", port=port)

