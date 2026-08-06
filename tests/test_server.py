"""
tests/test_server.py — Automated tests for Nyra FastAPI / WebSocket Mobile Server.
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
# pyrefly: ignore [missing-import]
from fastapi.testclient import TestClient

from server import app, audio_to_base64_wav, base64_wav_to_numpy


class TestServerHelpers(unittest.TestCase):
    def test_audio_base64_roundtrip(self):
        """Verify float32 PCM numpy array can be encoded to Base64 WAV and decoded back."""
        sample_rate = 16000
        duration_s = 0.5
        t = np.linspace(0, duration_s, int(sample_rate * duration_s), dtype=np.float32)
        pcm_orig = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone

        b64_wav = audio_to_base64_wav(pcm_orig, sample_rate)
        self.assertIsInstance(b64_wav, str)
        self.assertTrue(len(b64_wav) > 0)

        pcm_decoded, sr_decoded = base64_wav_to_numpy(b64_wav)
        self.assertEqual(sr_decoded, sample_rate)
        self.assertEqual(len(pcm_decoded), len(pcm_orig))
        np.testing.assert_allclose(pcm_decoded, pcm_orig, atol=1e-3)


class TestServerEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_check(self):
        """GET /health should return 200 and online status."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "online")
        self.assertIn("engines", data)

    def test_pwa_endpoints(self):
        """GET /manifest.json and /serviceworker.js should return 200 OK."""
        manifest_res = self.client.get("/manifest.json")
        self.assertEqual(manifest_res.status_code, 200)
        manifest_json = manifest_res.json()
        self.assertEqual(manifest_json.get("short_name"), "Nyra AI")
        self.assertEqual(manifest_json.get("display"), "standalone")

        sw_res = self.client.get("/serviceworker.js")
        self.assertEqual(sw_res.status_code, 200)
        self.assertIn("CACHE_NAME", sw_res.text)

    @patch("server.get_llm_engine")
    @patch("server.get_tts_engine")
    def test_text_chat_rest(self, mock_get_tts, mock_get_llm):
        """POST /api/chat should return LLM text and synthesized audio payload."""
        mock_llm = MagicMock()
        mock_tts = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_get_tts.return_value = mock_tts

        mock_llm.stream_sentences.return_value = ["Hello!", "How can I help you today?"]
        mock_tts.synthesize.return_value = (np.zeros(16000, dtype=np.float32), 16000)

        response = self.client.post("/api/chat", json={"message": "Hi Nyra"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("user_message"), "Hi Nyra")
        self.assertIn("Hello! How can I help you today?", data.get("assistant_text", ""))
        self.assertIn("audio_b64", data)

    @patch("server.get_llm_engine")
    @patch("server.get_tts_engine")
    def test_websocket_text_flow(self, mock_get_tts, mock_get_llm):
        """WS /ws/voice should stream response chunks over WebSocket."""
        mock_llm = MagicMock()
        mock_tts = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_get_tts.return_value = mock_tts

        mock_llm.stream_sentences.return_value = ["Hello from WebSocket."]
        mock_tts.synthesize.return_value = (np.zeros(8000, dtype=np.float32), 16000)

        with self.client.websocket_connect("/ws/voice") as websocket:
            websocket.send_json({"type": "text", "text": "Hello"})

            chunk = websocket.receive_json()
            self.assertEqual(chunk.get("type"), "response_chunk")
            self.assertEqual(chunk.get("text"), "Hello from WebSocket.")

            done = websocket.receive_json()
            self.assertEqual(done.get("type"), "done")


if __name__ == "__main__":
    unittest.main()
