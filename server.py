"""Mutara voice server: speaks announcement text with Kokoro-82M.

A Mutara node with voice_server set sends each announcement here as text and
plays the WAV that comes back; it keeps its own copy, and uses the built-in
Allison clips when the server does not answer in time. See README.md.

    GET  /v1/voices   {"voices": ["af_sarah", ...], "default": "af_sarah"}
    POST /v1/speech   {"text": "...", "voice": "af_sarah", "speed": 0.9}
                      -> audio/wav, 24 kHz mono 16-bit, speech at -20 dBFS
    GET  /healthz     ok

There is no authentication. Listen on localhost or a Tailscale address
only, never on a public one.
"""

import argparse
import hashlib
import io
import json
import os
import sys
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

RATE = 24000
TARGET_DB = -20.0  # speech RMS, dBFS: Mutara's voice.ClipDB
PEAK_DB = -1.0
PAD = 0.05  # seconds of silence kept at each end
MAX_TEXT = 400
DEFAULT_VOICE = "af_sarah"
# MODEL names the model in cache keys, so new model files do not reuse old
# audio.
MODEL = "kokoro-v1.0"


def espeak():
    """Point kokoro-onnx at a system espeak-ng if there is one; its bundled
    one fails on macOS. On Linux the bundled one works."""
    from kokoro_onnx import EspeakConfig

    for prefix in ("/opt/homebrew", "/usr/local"):
        lib = os.path.join(prefix, "lib/libespeak-ng.dylib")
        data = os.path.join(prefix, "share/espeak-ng-data")
        if os.path.exists(lib) and os.path.isdir(data):
            return EspeakConfig(lib_path=lib, data_path=data)
    return None


def finish(x):
    """Trims silence to PAD, sets the speech level and keeps peaks under
    PEAK_DB. x is float samples at RATE."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return x
    loud = np.flatnonzero(np.abs(x) > np.max(np.abs(x)) * 0.02)
    if len(loud):
        pad = int(PAD * RATE)
        x = x[max(loud[0] - pad, 0) : loud[-1] + pad]
    n = RATE // 50
    blocks = [x[i : i + n] for i in range(0, len(x) - n + 1, n)] or [x]
    rms = np.array([np.sqrt(np.mean(b**2)) for b in blocks])
    speech = rms[rms >= np.median(rms)]
    level = 20 * np.log10(max(np.sqrt(np.mean(speech**2)), 1e-9))
    gain = 10 ** ((TARGET_DB - level) / 20)
    peak = np.max(np.abs(x)) * gain
    if peak > 10 ** (PEAK_DB / 20):
        gain *= 10 ** (PEAK_DB / 20) / peak
    x = x * gain
    fade = min(int(0.005 * RATE), len(x) // 2)
    if fade:
        ramp = np.linspace(0, 1, fade)
        x[:fade] *= ramp
        x[-fade:] *= ramp[::-1]
    return x


def to_wav(x):
    pcm = np.clip(np.round(x * 32767), -32768, 32767).astype("<i2").tobytes()
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm)
    return b.getvalue()


class Cache:
    """WAV files by key in a folder, oldest removed past max_bytes."""

    def __init__(self, path, max_bytes):
        self.path, self.max = path, max_bytes
        self.lock = threading.Lock()
        os.makedirs(path, exist_ok=True)

    def get(self, key):
        p = os.path.join(self.path, key + ".wav")
        try:
            with open(p, "rb") as f:
                b = f.read()
            os.utime(p)
            return b
        except OSError:
            return None

    def put(self, key, b):
        with self.lock:
            tmp = os.path.join(self.path, key + ".tmp")
            with open(tmp, "wb") as f:
                f.write(b)
            os.replace(tmp, os.path.join(self.path, key + ".wav"))
            files = []
            for e in os.scandir(self.path):
                if e.name.endswith(".wav"):
                    st = e.stat()
                    files.append((st.st_mtime, st.st_size, e.path))
            total = sum(f[1] for f in files)
            for _, size, p in sorted(files):
                if total <= self.max:
                    break
                os.remove(p)
                total -= size


class Speaker:
    def __init__(self, models, cache):
        from kokoro_onnx import Kokoro

        self.k = Kokoro(
            os.path.join(models, "kokoro-v1.0.onnx"),
            os.path.join(models, "voices-v1.0.bin"),
            espeak_config=espeak(),
        )
        # English only: announcements are English. a* voices are American,
        # b* British.
        self.voices = sorted(v for v in self.k.get_voices() if v[:1] in "ab")
        self.cache = cache
        self.lock = threading.Lock()  # one synthesis at a time

    def speech(self, text, voice, speed):
        key = hashlib.sha256(f"{MODEL}\n{voice}\n{speed:.2f}\n{text}".encode()).hexdigest()
        b = self.cache.get(key)
        if b is not None:
            return b, True
        with self.lock:
            lang = "en-gb" if voice.startswith("b") else "en-us"
            samples, rate = self.k.create(text, voice=voice, speed=speed, lang=lang)
        assert rate == RATE, rate
        b = to_wav(finish(samples))
        self.cache.put(key, b)
        return b, False


def handler(sp, log):
    class H(BaseHTTPRequestHandler):
        def send(self, code, body, ctype="application/json"):
            if isinstance(body, (dict, list)):
                body = json.dumps(body).encode()
            elif isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/healthz":
                return self.send(200, "ok\n", "text/plain")
            if self.path == "/v1/voices":
                return self.send(200, {"voices": sp.voices, "default": DEFAULT_VOICE})
            self.send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/speech":
                return self.send(404, {"error": "not found"})
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if n > 4096:
                    raise ValueError("request too large")
                req = json.loads(self.rfile.read(n))
                text = str(req.get("text", "")).strip()
                voice = str(req.get("voice") or DEFAULT_VOICE)
                speed = round(float(req.get("speed") or 0.9), 2)
                if not text or len(text) > MAX_TEXT:
                    raise ValueError(f"text must be 1 to {MAX_TEXT} characters")
                if voice not in sp.voices:
                    raise ValueError(f"unknown voice {voice!r}")
                if not 0.5 <= speed <= 2.0:
                    raise ValueError("speed must be 0.5 to 2.0")
            except (ValueError, TypeError, AttributeError, json.JSONDecodeError) as e:
                return self.send(400, {"error": str(e)})
            try:
                b, hit = sp.speech(text, voice, speed)
            except Exception as e:  # noqa: BLE001 - report any synthesis failure
                log(f"speech failed: {e}")
                return self.send(500, {"error": "speech failed"})
            log(f"{'cached' if hit else 'made'} {voice} {speed}: {text}")
            self.send(200, b, "audio/wav")

        def log_message(self, fmt, *args):
            pass

    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--listen", default="127.0.0.1:7374", help="address:port; localhost or Tailscale only")
    ap.add_argument("--models", default="models", help="folder with kokoro-v1.0.onnx and voices-v1.0.bin")
    ap.add_argument("--cache", default="cache", help="folder for made speech")
    ap.add_argument("--cache-mb", type=int, default=500, help="cache size limit")
    args = ap.parse_args()

    def log(msg):
        print(msg, file=sys.stderr, flush=True)

    sp = Speaker(args.models, Cache(args.cache, args.cache_mb << 20))
    host, _, port = args.listen.rpartition(":")
    srv = ThreadingHTTPServer((host.strip("[]") or "0.0.0.0", int(port)), handler(sp, log))
    log(f"voice server on {args.listen}, {len(sp.voices)} voices, default {DEFAULT_VOICE}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
