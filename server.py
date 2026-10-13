"""Mutara voice server: speaks announcement text with Kokoro-82M.

A Mutara node with voice_server set sends each announcement here as text and
plays the WAV that comes back; it keeps its own copy, and uses the built-in
Allison clips when the server does not answer in time. See README.md.

    GET  /v1/voices   {"voices": ["af_sarah", ...], "default": "af_sarah"}
    POST /v1/speech   {"text": "...", "voice": "af_sarah", "speed": 0.8}
                      -> audio/wav, 24 kHz mono 16-bit, speech at -20 dBFS
    GET  /healthz     ok
    GET  /health      {"status": "ok", "version": ..., "uptime_s": ...}
    GET  /stats       counters for the node's Health page; see Stats.json

There is no authentication. Listen on localhost or a Tailscale address
only, never on a public one.
"""

import argparse
import collections
import hashlib
import io
import json
import os
import sys
import tempfile
import threading
import time
import urllib.request
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
VERSION = "1.1.0"
# LISTEN_FILE keeps the address the server listens on, for --check (the
# Docker health check), which can't see the command line.
LISTEN_FILE = os.path.join(tempfile.gettempdir(), "mutara-voiceserver.listen")


def utc(t):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t)) if t else None


class Stats:
    """Counters for GET /stats. They are cheap: a lock, some integers and
    the times of the last WINDOW syntheses."""

    WINDOW = 200

    def __init__(self):
        self.lock = threading.Lock()
        self.start = time.time()
        self.requests = self.ok = self.bad = self.errors = 0
        self.hits = self.misses = 0
        self.in_flight = 0  # speech requests being answered
        self.waiting = 0  # requests waiting for the synthesizer
        self.times = collections.deque(maxlen=self.WINDOW)  # ms
        self.last_error, self.last_error_at = "", 0.0

    def add(self, **n):
        with self.lock:
            for k, v in n.items():
                setattr(self, k, getattr(self, k) + v)

    def failed(self, msg):
        with self.lock:
            self.errors += 1
            self.last_error, self.last_error_at = msg, time.time()

    def json(self, sp):
        with self.lock:
            t = sorted(self.times)
            last = self.times[-1] if self.times else None
            out = {
                "version": VERSION,
                "model": MODEL,
                "model_loaded": True,
                "voices": len(sp.voices),
                "default_voice": DEFAULT_VOICE,
                "uptime_s": round(time.time() - self.start, 1),
                "requests": self.requests,
                "ok": self.ok,
                "bad_requests": self.bad,
                "errors": self.errors,
                "in_flight": self.in_flight,
                "waiting": self.waiting,
                "last_error": self.last_error,
                "last_error_at": utc(self.last_error_at),
                "synth": {
                    "window": len(t),
                    "last_ms": last,
                    "avg_ms": round(sum(t) / len(t), 1) if t else None,
                    "p50_ms": t[len(t) // 2] if t else None,
                    "p95_ms": t[min(len(t) - 1, len(t) * 95 // 100)] if t else None,
                    "max_ms": t[-1] if t else None,
                },
            }
        out["cache"] = {"hits": self.hits, "misses": self.misses, **sp.cache.size()}
        return out


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
        # entries and bytes are counted at start and after each put, so
        # /stats does not read the folder.
        self.entries, self.bytes = 0, 0
        for e in os.scandir(path):
            if e.name.endswith(".wav"):
                self.entries += 1
                self.bytes += e.stat().st_size

    def size(self):
        with self.lock:
            return {"entries": self.entries, "bytes": self.bytes, "max_bytes": self.max}

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
            total, count = sum(f[1] for f in files), len(files)
            for _, size, p in sorted(files):
                if total <= self.max:
                    break
                os.remove(p)
                total -= size
                count -= 1
            self.entries, self.bytes = count, total


class Speaker:
    def __init__(self, models, cache, stats):
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
        self.stats = stats
        self.lock = threading.Lock()  # one synthesis at a time

    def speech(self, text, voice, speed):
        key = hashlib.sha256(f"{MODEL}\n{voice}\n{speed:.2f}\n{text}".encode()).hexdigest()
        b = self.cache.get(key)
        if b is not None:
            self.stats.add(hits=1)
            return b, True
        self.stats.add(misses=1, waiting=1)
        with self.lock:
            self.stats.add(waiting=-1)
            t0 = time.monotonic()
            lang = "en-gb" if voice.startswith("b") else "en-us"
            samples, rate = self.k.create(text, voice=voice, speed=speed, lang=lang)
            ms = round((time.monotonic() - t0) * 1000)
        with self.stats.lock:
            self.stats.times.append(ms)
        assert rate == RATE, rate
        b = to_wav(finish(samples))
        self.cache.put(key, b)
        return b, False


def handler(sp, log):
    st = sp.stats

    class H(BaseHTTPRequestHandler):
        def send(self, code, body, ctype="application/json", headers=()):
            if isinstance(body, (dict, list)):
                body = json.dumps(body).encode()
            elif isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in headers:
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/healthz":
                return self.send(200, "ok\n", "text/plain")
            if self.path == "/health":
                up = round(time.time() - st.start, 1)
                return self.send(200, {"status": "ok", "version": VERSION, "model": MODEL, "voices": len(sp.voices), "uptime_s": up})
            if self.path == "/stats":
                return self.send(200, st.json(sp))
            if self.path == "/v1/voices":
                return self.send(200, {"voices": sp.voices, "default": DEFAULT_VOICE})
            self.send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/speech":
                return self.send(404, {"error": "not found"})
            st.add(requests=1, in_flight=1)
            try:
                self.speech()
            finally:
                st.add(in_flight=-1)

        def speech(self):
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if n > 4096:
                    raise ValueError("request too large")
                req = json.loads(self.rfile.read(n))
                text = str(req.get("text", "")).strip()
                voice = str(req.get("voice") or DEFAULT_VOICE)
                speed = round(float(req.get("speed") or 0.8), 2)
                if not text or len(text) > MAX_TEXT:
                    raise ValueError(f"text must be 1 to {MAX_TEXT} characters")
                if voice not in sp.voices:
                    raise ValueError(f"unknown voice {voice!r}")
                if not 0.5 <= speed <= 2.0:
                    raise ValueError("speed must be 0.5 to 2.0")
            except (ValueError, TypeError, AttributeError, json.JSONDecodeError) as e:
                st.add(bad=1)
                return self.send(400, {"error": str(e)})
            try:
                b, hit = sp.speech(text, voice, speed)
            except Exception as e:  # noqa: BLE001 - report any synthesis failure
                log(f"speech failed: {e}")
                st.failed(f"speech failed: {e}"[:200])
                return self.send(500, {"error": "speech failed"})
            st.add(ok=1)
            log(f"{'cached' if hit else 'made'} {voice} {speed}: {text}")
            self.send(200, b, "audio/wav", [("X-Cache", "hit" if hit else "miss")])

        def log_message(self, fmt, *args):
            pass

    return H


def check():
    """The Docker health check: GET /health on the address in LISTEN_FILE.
    Exits 0 when the server answers."""
    try:
        with open(LISTEN_FILE) as f:
            host, _, port = f.read().strip().rpartition(":")
        host = host.strip("[]")
        if host in ("", "0.0.0.0", "::"):
            host = "127.0.0.1"
        if ":" in host:
            host = f"[{host}]"
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=5) as r:
            sys.exit(0 if r.status == 200 else 1)
    except Exception as e:  # noqa: BLE001 - any failure is unhealthy
        print(f"unhealthy: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--listen", default="127.0.0.1:7374", help="address:port; localhost or Tailscale only")
    ap.add_argument("--models", default="models", help="folder with kokoro-v1.0.onnx and voices-v1.0.bin")
    ap.add_argument("--cache", default="cache", help="folder for made speech")
    ap.add_argument("--cache-mb", type=int, default=500, help="cache size limit")
    ap.add_argument("--check", action="store_true", help="check that the running server answers, and exit")
    args = ap.parse_args()
    if args.check:
        check()

    def log(msg):
        print(msg, file=sys.stderr, flush=True)

    sp = Speaker(args.models, Cache(args.cache, args.cache_mb << 20), Stats())
    host, _, port = args.listen.rpartition(":")
    srv = ThreadingHTTPServer((host.strip("[]") or "0.0.0.0", int(port)), handler(sp, log))
    with open(LISTEN_FILE, "w") as f:
        f.write(args.listen)
    log(f"voice server on {args.listen}, {len(sp.voices)} voices, default {DEFAULT_VOICE}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
