"""Mutara voice server: speaks announcement text with Kokoro-82M.

A Mutara node with voice_server set sends each announcement here as text and
plays the WAV that comes back; it keeps its own copy, and uses the built-in
Allison clips when the server does not answer in time. See README.md.

    GET  /v1/voices   {"voices": ["af_sarah", ...], "default": "af_sarah"}
    POST /v1/speech   {"text": "...", "voice": "af_sarah", "speed": 0.8}
                      header X-Voice-Priority: now (the default) or later
                      -> audio/wav, 24 kHz mono 16-bit, speech at -20 dBFS
                      -> 503 with Retry-After when the later queue is full
    GET  /healthz     ok
    GET  /health      {"status": "ok", "version": ..., "uptime_s": ...}
    GET  /stats       counters for the node's Health page; see Stats.json

Requests wait in a queue (ADR 0009): now before later, in order of arrival
within each, one sentence at a time, so a now request waits for at most
one sentence of a later text.

There is no authentication. Listen on localhost or a Tailscale address
only, never on a public one.
"""

import argparse
import collections
import hashlib
import io
import json
import math
import os
import re
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
VERSION = "1.2.0"
# LISTEN_FILE keeps the address the server listens on, for --check (the
# Docker health check), which can't see the command line.
LISTEN_FILE = os.path.join(tempfile.gettempdir(), "mutara-voiceserver.listen")
NOW, LATER = "now", "later"
GAP = 0.25  # seconds of silence between sentences: kokoro-onnx's sentence_pause
LATER_LIMIT = 32


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
        q = sp.queue_info()
        out["waiting"] = q["waiting"]
        out["queue"], out["wait_ms"], out["busy_503"] = q["queue"], q["wait_ms"], q["busy_503"]
        out["workers"] = q["workers"]
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


# ABBREV are words that end with a full stop inside a sentence.
ABBREV = {"mr", "mrs", "ms", "dr", "st", "mt", "ft", "jr", "sr", "vs", "no", "approx", "est"}


def sentences(text):
    """Splits text after . ! ? or ; and a space. Each part keeps its mark.
    It does not split after an abbreviation such as "St." or "a.m.", or
    after one letter, because the parts are joined with a GAP of silence."""
    out, start = [], 0
    for m in re.finditer(r"[.!?;]+\s+", text):
        words = text[start : m.start()].split()
        w = words[-1].lower() if words else ""
        if m.group().rstrip() == "." and (w in ABBREV or "." in w or len(w) == 1):
            continue
        out.append(text[start : m.end()].strip())
        start = m.end()
    if text[start:].strip():
        out.append(text[start:].strip())
    return out or [text]


class Busy(Exception):
    """The later queue is full. retry_after is in seconds."""

    def __init__(self, retry_after):
        super().__init__(f"busy, retry after {retry_after} s")
        self.retry_after = retry_after


class Job:
    """One text to make. All requests for its key wait on done."""

    def __init__(self, key, text, voice, speed, prio):
        self.key, self.voice, self.speed, self.prio = key, voice, speed, prio
        self.parts = sentences(text)
        self.audio = []  # float samples, one per finished sentence
        self.busy = False  # a worker makes one of its sentences now
        self.started = None
        self.synth_ms = 0
        self.done = threading.Event()
        self.result = self.error = None


def load_kokoro(models):
    """Loads Kokoro and returns (synth, voices). synth(text, voice, speed,
    lang) -> (samples, rate).

    All workers share one model. kokoro-onnx 0.6.1 keeps no state between
    calls except its tokenizer, which phonemizes under its own lock
    (espeak-ng is not thread-safe), and onnxruntime allows Run() from more
    than one thread. If an other kokoro-onnx has no such lock, the calls
    are made one at a time."""
    import kokoro_onnx.tokenizer
    from kokoro_onnx import Kokoro

    k = Kokoro(
        os.path.join(models, "kokoro-v1.0.onnx"),
        os.path.join(models, "voices-v1.0.bin"),
        espeak_config=espeak(),
    )
    # English only: announcements are English. a* voices are American,
    # b* British.
    voices = sorted(v for v in k.get_voices() if v[:1] in "ab")
    if hasattr(kokoro_onnx.tokenizer, "_espeak_lock"):
        def synth(text, voice, speed, lang):
            return k.create(text, voice=voice, speed=speed, lang=lang)
    else:
        one = threading.Lock()

        def synth(text, voice, speed, lang):
            with one:
                return k.create(text, voice=voice, speed=speed, lang=lang)
    return synth, voices


class Speaker:
    """Makes speech in a priority queue (ADR 0009).

    A now request always goes before a later one; within a priority the
    order of arrival is kept. A worker makes one sentence of the first job
    in each turn, so a now request waits for one sentence at most. A job
    stays first in its queue until all its sentences are made. The full
    text is cached as one answer under the same key as before.

    synth is synth(text, voice, speed, lang) -> (samples, rate); tests pass
    a fake one."""

    def __init__(self, synth, voices, cache, stats, workers=1, later_limit=LATER_LIMIT):
        self.synth, self.voices = synth, voices
        self.cache, self.stats = cache, stats
        self.workers, self.later_limit = max(1, workers), later_limit
        self.cv = threading.Condition()
        self.queues = {NOW: collections.deque(), LATER: collections.deque()}
        self.jobs = {}  # key -> Job, from the first request until it is cached
        self.wait_ms = {NOW: None, LATER: None}
        self.busy_503 = 0
        self.sentence_s = 1.0  # average time for one sentence, for Retry-After
        for i in range(self.workers):
            # With 2 or more workers, worker 0 takes only now requests.
            now_only = self.workers >= 2 and i == 0
            threading.Thread(target=self.work, args=(now_only,), daemon=True, name=f"synth-{i}").start()

    @staticmethod
    def key(text, voice, speed):
        return hashlib.sha256(f"{MODEL}\n{voice}\n{speed:.2f}\n{text}".encode()).hexdigest()

    def speech(self, text, voice, speed, prio=NOW):
        """Returns (wav, hit). Raises Busy for a later request when the later
        queue is full."""
        key = self.key(text, voice, speed)
        b = self.cache.get(key)
        if b is not None:
            self.stats.add(hits=1)
            return b, True
        t0 = time.monotonic()
        with self.cv:
            job = self.jobs.get(key)
            if job is None:
                if prio == LATER and len(self.queues[LATER]) >= self.later_limit:
                    self.busy_503 += 1
                    raise Busy(self.retry_after())
                job = Job(key, text, voice, speed, prio)
                self.jobs[key] = job
                self.queues[prio].append(job)
                self.cv.notify_all()
            elif prio == NOW and job.prio == LATER:
                # Promotion: the queued later job becomes now, at the end of
                # the now queue, and both requests get its one answer.
                job.prio = NOW
                if job in self.queues[LATER]:
                    self.queues[LATER].remove(job)
                    self.queues[NOW].append(job)
                self.cv.notify_all()
        self.stats.add(misses=1)
        job.done.wait()
        with self.cv:
            started = job.started if job.started is not None else time.monotonic()
            self.wait_ms[prio] = round(max(0.0, started - t0) * 1000)
        if job.error is not None:
            raise job.error
        return job.result, False

    def retry_after(self):
        """Seconds until the first later job is made and the queue has room,
        roughly. Called with cv held."""
        q = self.queues[LATER]
        left = len(q[0].parts) - len(q[0].audio) if q else 1
        return max(1, min(300, math.ceil(left * self.sentence_s)))

    def pick(self, now_only):
        for prio in (NOW,) if now_only else (NOW, LATER):
            for job in self.queues[prio]:
                if not job.busy:
                    return job
        return None

    def work(self, now_only):
        while True:
            with self.cv:
                job = self.pick(now_only)
                while job is None:
                    self.cv.wait()
                    job = self.pick(now_only)
                job.busy = True
                if job.started is None:
                    job.started = time.monotonic()
                part = job.parts[len(job.audio)]
            lang = "en-gb" if job.voice.startswith("b") else "en-us"
            t0 = time.monotonic()
            err = samples = None
            try:
                samples, rate = self.synth(part, job.voice, job.speed, lang)
                assert rate == RATE, rate
            except Exception as e:  # noqa: BLE001 - given to the waiting requests
                err = e
            s = time.monotonic() - t0
            with self.cv:
                job.busy = False
                job.synth_ms += round(s * 1000)
                self.sentence_s = 0.8 * self.sentence_s + 0.2 * s
                if err is None:
                    job.audio.append(np.asarray(samples, dtype=np.float64))
                finished = err is not None or len(job.audio) == len(job.parts)
                if finished:
                    self.queues[job.prio].remove(job)
                self.cv.notify_all()
            if finished:
                self.finish(job, err)

    def finish(self, job, err):
        if err is None:
            try:
                gap = np.zeros(int(GAP * RATE))
                x = job.audio[0]
                for a in job.audio[1:]:
                    x = np.concatenate([x, gap, a])
                job.result = to_wav(finish(x))
                self.cache.put(job.key, job.result)
                with self.stats.lock:
                    self.stats.times.append(job.synth_ms)
            except Exception as e:  # noqa: BLE001
                err = e
        job.error = err
        with self.cv:
            self.jobs.pop(job.key, None)
        job.done.set()

    def queue_info(self):
        with self.cv:
            waiting = sum(1 for q in self.queues.values() for j in q if j.started is None)
            return {
                "queue": {NOW: len(self.queues[NOW]), LATER: len(self.queues[LATER])},
                "wait_ms": dict(self.wait_ms),
                "busy_503": self.busy_503,
                "waiting": waiting,
                "workers": self.workers,
            }


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
                q = sp.queue_info()
                return self.send(
                    200,
                    {
                        "status": "ok",
                        "version": VERSION,
                        "model": MODEL,
                        "voices": len(sp.voices),
                        "uptime_s": up,
                        "workers": q["workers"],
                        "queue": q["queue"],
                        "wait_ms": q["wait_ms"],
                        "busy_503": q["busy_503"],
                    },
                )
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
            # No header, or an unknown one, is now: an old client works as
            # before.
            prio = LATER if self.headers.get("X-Voice-Priority", "").strip().lower() == LATER else NOW
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
                b, hit = sp.speech(text, voice, speed, prio)
            except Busy as e:
                return self.send(503, {"error": "busy, try again later"}, headers=[("Retry-After", str(e.retry_after))])
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
    ap.add_argument("--workers", type=int, default=1, help="syntheses at the same time; with 2 or more, one takes only now requests")
    ap.add_argument("--later-limit", type=int, default=LATER_LIMIT, help="later requests in the queue before a new one gets 503")
    ap.add_argument("--check", action="store_true", help="check that the running server answers, and exit")
    args = ap.parse_args()
    if args.check:
        check()
    if args.workers < 1 or args.later_limit < 1:
        ap.error("--workers and --later-limit must be 1 or more")

    def log(msg):
        print(msg, file=sys.stderr, flush=True)

    synth, voices = load_kokoro(args.models)
    cache = Cache(args.cache, args.cache_mb << 20)
    sp = Speaker(synth, voices, cache, Stats(), workers=args.workers, later_limit=args.later_limit)
    host, _, port = args.listen.rpartition(":")
    srv = ThreadingHTTPServer((host.strip("[]") or "0.0.0.0", int(port)), handler(sp, log))
    with open(LISTEN_FILE, "w") as f:
        f.write(args.listen)
    log(f"voice server on {args.listen}, {len(sp.voices)} voices, default {DEFAULT_VOICE}, {sp.workers} workers, later limit {sp.later_limit}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
