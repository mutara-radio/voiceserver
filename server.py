"""Mutara voice server: speaks announcement text with Kokoro-82M, and
writes the text of recorded clips with faster-whisper.

A Mutara node with voice_server set sends each announcement here as text and
plays the WAV that comes back; it keeps its own copy, and uses the built-in
Allison clips when the server does not answer in time. See README.md.

    GET  /v1/voices     {"voices": ["af_sarah", ...], "default": "af_sarah"}
    POST /v1/speech     {"text": "...", "voice": "af_sarah", "speed": 0.8}
                        header X-Voice-Priority: now (the default) or later
                        -> audio/wav, 24 kHz mono 16-bit, speech at -20 dBFS
                        -> 503 with Retry-After when the later queue is full
    POST /v1/transcribe multipart/form-data: file (Ogg Opus), language,
                        prompt, id; the same priority header
                        -> {"text": ..., "segments": [...], ...}
    POST /v1/transcriptions   the same form -> 202 {"id": ..., "state": "queued"}
    GET  /v1/transcriptions/{id}?wait=s  -> {"state": ..., "result": ...}
    DELETE /v1/transcriptions/{id}       -> 204
    GET  /healthz       ok
    GET  /health        {"status": "ok", "version": ..., "uptime_s": ...,
                         "transcribe": {"model": ..., "ready": ...}}
    GET  /stats         counters for the node's Health page; see Stats.json

Speech requests wait in a queue (ADR 0009): now before later, in order of
arrival within each, one sentence at a time, so a now request waits for at
most one sentence of a later text. Transcriptions have a queue and a worker
of their own (ADR 0001), so speech never waits for them.

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
import urllib.parse
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
VERSION = "1.4.0"
STT_MODEL = "small.en"  # the default faster-whisper model
STT_RATE = 16000  # Whisper's sample rate
MAX_AUDIO = 8 << 20  # bytes in the file part of /v1/transcribe
MAX_FORM = MAX_AUDIO + (64 << 10)  # the whole multipart body
MAX_PROMPT = 1000
OPUS_BPS = 2500  # bytes per second of a clip, roughly, for Retry-After
# LISTEN_FILE keeps the address the server listens on, for --check (the
# Docker health check), which can't see the command line.
LISTEN_FILE = os.path.join(tempfile.gettempdir(), "mutara-voiceserver.listen")
NOW, LATER = "now", "later"
GAP = 0.25  # seconds of silence between sentences: kokoro-onnx's sentence_pause
LATER_LIMIT = 32
# The transcription queue (ADR 0001): jobs and bytes of audio in it before a
# new one gets 503, and how long a finished job's result is kept.
STT_QUEUE_LIMIT = 64
STT_QUEUE_MB = 256
STT_KEEP_S = 3600
STT_NICE = 10
JOBS = "/v1/transcriptions"
MAX_WAIT = 60  # seconds, for GET /v1/transcriptions/{id}?wait=


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
        # The same for POST /v1/transcribe. stt_audio_ms and stt_work_ms
        # add up the clips and the time to make their text.
        self.stt_requests = self.stt_ok = self.stt_bad = self.stt_errors = 0
        self.stt_in_flight = self.stt_audio_ms = self.stt_work_ms = 0
        self.stt_last_ms = None
        self.last_error, self.last_error_at = "", 0.0

    def add(self, **n):
        with self.lock:
            for k, v in n.items():
                setattr(self, k, getattr(self, k) + v)

    def failed(self, msg, counter="errors"):
        with self.lock:
            setattr(self, counter, getattr(self, counter) + 1)
            self.last_error, self.last_error_at = msg, time.time()

    def json(self, sp, stt, tr):
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
                "transcribe": {
                    **stt.info(),
                    "requests": self.stt_requests,
                    "ok": self.stt_ok,
                    "bad_requests": self.stt_bad,
                    "errors": self.stt_errors,
                    "in_flight": self.stt_in_flight,
                    "audio_s": round(self.stt_audio_ms / 1000, 1),
                    "work_s": round(self.stt_work_ms / 1000, 1),
                    "last_ms": self.stt_last_ms,
                },
            }
        out["transcribe"].update(tr.queue_info(), jobs=True)
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


class BadAudio(ValueError):
    """The file of a transcription is not Ogg Opus that can be decoded."""


class TooLarge(ValueError):
    """A request body or file is over its limit."""


class Job:
    """One text to make. All requests for its key wait on done."""

    def __init__(self, key, text, voice, speed, prio):
        self.key, self.voice, self.speed, self.prio = key, voice, speed, prio
        self.parts = sentences(text)
        self.audio = []  # float samples, one per finished sentence
        self.busy = False  # a worker makes one of its sentences now
        self.started = None
        self.work_ms = 0
        self.done = threading.Event()
        self.result = self.error = None

    def step(self, sp):
        """Makes the next sentence. Returns True when all are made."""
        lang = "en-gb" if self.voice.startswith("b") else "en-us"
        t0 = time.monotonic()
        try:
            samples, rate = sp.synth(self.parts[len(self.audio)], self.voice, self.speed, lang)
            assert rate == RATE, rate
        finally:
            sp.sentence_s = 0.8 * sp.sentence_s + 0.2 * (time.monotonic() - t0)
        self.audio.append(np.asarray(samples, dtype=np.float64))
        return len(self.audio) == len(self.parts)

    def left_s(self, sp):
        return (len(self.parts) - len(self.audio)) * sp.sentence_s


def is_ogg_opus(b):
    """True when b starts with an Ogg page that holds an Opus header. The
    decoder checks the rest."""
    if len(b) < 28 or b[:4] != b"OggS":
        return False
    body = 27 + b[26]  # the page header, then one byte for each segment
    return b[body : body + 8] == b"OpusHead"


def form(ctype, body):
    """Parses a multipart/form-data body. Returns {name: bytes}, the first
    part of each name."""
    m = re.search(r'boundary="?([^";]+)"?', ctype)
    if ctype.split(";")[0].strip().lower() != "multipart/form-data" or not m:
        raise ValueError("the body must be multipart/form-data")
    out = {}
    for part in body.split(b"--" + m.group(1).encode())[1:]:
        if part.startswith(b"--"):
            break
        head, sep, data = part.partition(b"\r\n\r\n")
        name = re.search(rb'(?i)content-disposition:[^\r\n]*?\bname="([^"]*)"', head)
        if sep and name:
            out.setdefault(name.group(1).decode(errors="replace"), data[:-2] if data.endswith(b"\r\n") else data)
    return out


def decode_ogg(data):
    """Decodes Ogg Opus bytes to float32 samples at STT_RATE, mono, with
    PyAV (faster-whisper's decode_audio)."""
    from faster_whisper import decode_audio

    try:
        return decode_audio(io.BytesIO(data), sampling_rate=STT_RATE)
    except Exception as e:  # noqa: BLE001 - any decoder failure is bad audio
        raise BadAudio(f"the file can't be decoded: {e}") from e


def confidence(avg_logprob):
    """Maps a segment's average log probability (0 or less) to 0..1: e to
    its power, the geometric mean of the token probabilities."""
    return round(min(1.0, max(0.0, math.exp(avg_logprob))), 2)


class NoModel(RuntimeError):
    """Speech to text is off, or its model did not load."""


QUEUED, PROCESSING, DONE, FAILED = "queued", "processing", "done", "failed"


class Transcription:
    """One clip to transcribe: a job of the Transcriber. All requests for
    its key wait on done.

    The worker makes it in steps: the first step decodes the audio, runs
    the VAD and starts the model; each next step decodes one segment. So
    the job shows its progress, and a cancel stops it between two
    segments."""

    def __init__(self, key, data, language, prompt, prio):
        self.key, self.prio = key, prio
        self.data, self.size = data, len(data)
        self.language, self.prompt = language, prompt
        self.segments = self.info = None
        self.out = []  # the finished segments
        self.state = QUEUED
        self.cancelled = False
        self.queued_at, self.started, self.finished_at = time.monotonic(), None, None
        self.work_ms = 0
        self.done = threading.Event()
        self.result = None
        # exc is the exception that the waiting requests raise; reason and
        # error say why a failed job failed.
        self.exc, self.reason, self.error = None, None, ""

    @staticmethod
    def key_of(model, data, language, prompt):
        h = hashlib.sha256(f"stt\n{model}\n{language}\n{prompt}\n".encode())
        h.update(data)
        return h.hexdigest()

    def step(self, stt):
        """Starts the model, or makes one segment. Returns True when all
        are made."""
        if self.segments is None:
            audio = stt.decode(self.data)
            self.data = None
            # The prompt goes as hotwords: faster-whisper puts them before
            # every 30 s window, where initial_prompt is used for the first
            # window only (condition_on_previous_text is off). In the first
            # window both do the same.
            self.segments, self.info = stt.model.transcribe(
                audio, language=self.language or None, hotwords=self.prompt or None, **stt.options
            )
            return False
        seg = next(self.segments, None)
        if seg is None:
            return True
        self.out.append(seg)
        return False

    def progress_ms(self):
        return round(self.out[-1].end * 1000) if self.out else 0

    def left_s(self, rtf):
        if self.info is None:
            return self.size / OPUS_BPS * rtf
        return max(0.0, self.info.duration - self.progress_ms() / 1000) * rtf

    def answer(self, label):
        segs = []
        for s in self.out:
            text = s.text.strip()
            if text:
                segs.append(
                    {
                        "start_ms": round(s.start * 1000),
                        "end_ms": round(s.end * 1000),
                        "text": text,
                        "confidence": confidence(s.avg_logprob),
                    }
                )
        return {
            "text": " ".join(s["text"] for s in segs),
            "language": self.info.language,
            "duration_ms": round(self.info.duration * 1000),
            "model": label,
            "segments": segs,
        }


def lower_priority(n):
    """Lowers the OS priority of the calling thread by n (nice), on Linux
    only: there a thread has its own nice value, and the threads it starts
    get it too. Elsewhere it would change the whole process."""
    if n and sys.platform.startswith("linux"):
        try:
            os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), n)
        except OSError:
            pass


def half_the_cores():
    """Half the cores that the server can use: the CPU affinity, or a
    cgroup v2 limit (Docker's cpus) when it is lower."""
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:  # macOS
        n = os.cpu_count() or 2
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()[:2]
        if quota != "max":
            n = min(n, max(1, int(quota) // int(period)))
    except (OSError, ValueError):
        pass
    return max(1, n // 2)


class Transcriber:
    """Transcriptions in a queue of their own, with one worker (ADR 0001),
    so speech never waits for them. A now job (one clip that a user asked
    for) goes before a later one (a bundle); within a priority the order
    of arrival is kept. A now request for a queued later job promotes it.

    Jobs are kept by key, so the same clip, language and prompt make one
    job. A finished job stays for keep_s seconds, so its result can be
    picked up; a failed job is replaced by the next request for it.

    The queue holds at most queue_limit jobs and queue_bytes of audio; a
    new job over a limit raises Busy. stt is an STT; stats a Stats."""

    def __init__(self, stt, stats, queue_limit=STT_QUEUE_LIMIT, queue_bytes=STT_QUEUE_MB << 20, keep_s=STT_KEEP_S, nice=0):
        self.stt, self.stats = stt, stats
        self.queue_limit, self.queue_bytes, self.keep_s = queue_limit, queue_bytes, keep_s
        self.cv = threading.Condition()
        self.queues = {NOW: collections.deque(), LATER: collections.deque()}
        self.jobs = {}  # key -> Transcription, queued, processing or finished
        self.current = None  # the job in processing
        self.wait_ms = {NOW: None, LATER: None}
        self.busy_503 = 0
        self.rtf = 0.5  # transcription time / clip time, for Retry-After
        threading.Thread(target=self.work, args=(nice,), daemon=True, name="stt").start()

    def submit(self, data, language, prompt, prio=NOW, refuse_now=True, new=True):
        """Returns the job for the clip: a new one, or the one that is
        queued, processing or done already. Raises Busy when a new job
        does not fit; with refuse_now False, only a later one. With new
        False, returns None in place of a new job."""
        key = Transcription.key_of(self.stt.label, data, language, prompt)
        with self.cv:
            self.prune()
            job = self.jobs.get(key)
            if job is not None and job.state == FAILED:
                job = None
            if job is None and not new:
                return None
            if job is None:
                queued = [j for q in self.queues.values() for j in q]
                full = len(queued) >= self.queue_limit or sum(j.size for j in queued) + len(data) > self.queue_bytes
                if full and (prio == LATER or refuse_now):
                    self.busy_503 += 1
                    raise Busy(self.retry_after())
                job = Transcription(key, data, language, prompt, prio)
                self.jobs[key] = job
                self.queues[prio].append(job)
                self.cv.notify_all()
            elif prio == NOW and job.prio == LATER and job.state == QUEUED:
                job.prio = NOW
                self.queues[LATER].remove(job)
                self.queues[NOW].append(job)
            return job

    def transcribe(self, data, language, prompt, prio=NOW):
        """POST /v1/transcribe: makes the job and waits for it. Returns its
        answer. Raises Busy for a later request when the queue is full,
        and the job's exception when it fails."""
        job = self.submit(data, language, prompt, prio, refuse_now=False)
        job.done.wait()
        if job.exc is not None:
            raise job.exc
        return job.result

    def prune(self):
        """Forgets finished jobs older than keep_s. Called with cv held."""
        now = time.monotonic()
        for key in [k for k, j in self.jobs.items() if j.finished_at is not None and now - j.finished_at > self.keep_s]:
            del self.jobs[key]

    def retry_after(self):
        """Seconds until the queue has room, roughly: the time of the job
        in processing and the first queued one. Called with cv held."""
        s = sum(j.left_s(self.rtf) for j in [self.current, *(q[0] for q in self.queues.values() if q)] if j is not None)
        return max(1, min(300, math.ceil(s)))

    def work(self, nice):
        lower_priority(nice)
        # Jobs sent while the model loads stay queued until it is ready.
        self.stt.loaded.wait()
        while True:
            with self.cv:
                while not (self.queues[NOW] or self.queues[LATER]):
                    self.cv.wait()
                job = (self.queues[NOW] or self.queues[LATER]).popleft()
                job.state, job.started = PROCESSING, time.monotonic()
                self.wait_ms[job.prio] = round((job.started - job.queued_at) * 1000)
                self.current = job
            err = None
            try:
                if self.stt.state != "ready":
                    raise NoModel("speech to text is not available")
                while not job.cancelled:
                    t0 = time.monotonic()
                    done = job.step(self.stt)
                    job.work_ms += round((time.monotonic() - t0) * 1000)
                    if done:
                        break
            except Exception as e:  # noqa: BLE001 - given to the waiting requests
                err = e
            self.finish(job, err)

    def finish(self, job, err):
        """Ends the job, and counts it: ok, bad_requests (a clip that can't
        be decoded) or errors, and its audio and work time. Two requests
        that share a job count as one."""
        st = self.stats
        if err is None and not job.cancelled:
            try:
                job.result = job.answer(self.stt.label)
            except Exception as e:  # noqa: BLE001
                err = e
        if job.result is not None:
            audio_ms = job.result["duration_ms"]
            if audio_ms > 0:
                self.rtf = 0.8 * self.rtf + 0.2 * job.work_ms / audio_ms
            with st.lock:
                st.stt_ok += 1
                st.stt_audio_ms += audio_ms
                st.stt_work_ms += job.work_ms
                st.stt_last_ms = job.work_ms
            state = DONE
        else:
            state = FAILED
            if job.cancelled:
                job.reason, job.exc = "cancelled", RuntimeError("cancelled")
            elif isinstance(err, BadAudio):
                job.reason, job.exc = "bad_audio", err
                st.add(stt_bad=1)
            elif isinstance(err, NoModel):
                job.reason, job.exc = "model", err
            else:
                job.reason, job.exc = "error", err
                st.failed(f"transcribe failed: {err}"[:200], "stt_errors")
            job.error = str(job.exc)[:200]
        job.data, job.segments = None, None
        # The state changes last, under cv, so status() never shows a
        # finished job without its result or reason.
        with self.cv:
            job.state = state
            job.finished_at = time.monotonic()
            if self.current is job:
                self.current = None
            self.cv.notify_all()
        job.done.set()

    def get(self, key):
        """Returns the job with this ID, or None."""
        with self.cv:
            self.prune()
            return self.jobs.get(key)

    def status(self, job):
        """The answer of GET /v1/transcriptions/{id}."""
        with self.cv:
            state = job.state
            out = {"id": job.key, "state": state}
            if state == QUEUED:
                out["priority"] = job.prio
                out["position"] = self.position(job)
            elif state == PROCESSING:
                out["priority"] = job.prio
                out["progress_ms"] = job.progress_ms()
                if job.info is not None:
                    out["duration_ms"] = round(job.info.duration * 1000)
            elif state == DONE:
                out["result"] = job.result
            else:
                out["reason"], out["error"] = job.reason, job.error
        return out

    def position(self, job):
        """The jobs before a queued job: the one in processing, the now
        queue, and for a later job the later jobs before it. Called with cv
        held."""
        n = 1 if self.current is not None else 0
        if job.prio == LATER:
            n += len(self.queues[NOW])
        q = self.queues[job.prio]
        return n + (q.index(job) if job in q else 0)

    def cancel(self, key):
        """DELETE /v1/transcriptions/{id}: a queued job leaves the queue, a
        job in processing stops after its segment, and a finished one is
        forgotten. Returns False when there is no such job. Requests that
        wait for the job get a failure with reason cancelled."""
        with self.cv:
            job = self.jobs.pop(key, None)
            if job is None:
                return False
            job.cancelled = True
            queued = job.state == QUEUED
            if queued:
                self.queues[job.prio].remove(job)
        if queued:
            self.finish(job, None)
        return True

    def queue_info(self):
        with self.cv:
            return {
                "queue": {NOW: len(self.queues[NOW]), LATER: len(self.queues[LATER])},
                "processing": 1 if self.current is not None else 0,
                "wait_ms": dict(self.wait_ms),
                "busy_503": self.busy_503,
            }


class STT:
    """The speech to text model. main loads it in a thread, so the server
    speaks while it loads; when it fails, speech works on.

    state is off (no model set), loading, ready or failed. loader(name)
    returns a model with faster-whisper's transcribe(); tests pass a fake
    one, and a fake decode."""

    # vad_filter cuts silence and the noise of a squelch tail before the
    # model, so Whisper does not make words from noise.
    # condition_on_previous_text=False stops one wrong segment from
    # repeating in the next ones.
    OPTIONS = {
        "beam_size": 5,
        "vad_filter": True,
        "vad_parameters": {"min_silence_duration_ms": 500, "speech_pad_ms": 200},
        "condition_on_previous_text": False,
    }

    def __init__(self, name, loader=None, decode=decode_ogg):
        self.name = "" if name.strip().lower() in ("", "off") else name.strip()
        # label is the model's name in answers: a folder shows its last part.
        self.label = os.path.basename(self.name.rstrip("/")) if self.name else None
        self.loader, self.decode = loader, decode
        self.options = dict(self.OPTIONS)
        self.model, self.error = None, ""
        self.state = "loading" if self.name else "off"
        self.loaded = threading.Event()
        if not self.name:
            self.loaded.set()

    def load(self, log):
        t0 = time.monotonic()
        try:
            self.model = self.loader(self.name)
            self.state = "ready"
            log(f"speech to text: {self.label} ready in {time.monotonic() - t0:.1f} s")
        except Exception as e:  # noqa: BLE001 - speech works on without it
            self.error = f"{type(e).__name__}: {e}"[:200]
            self.state = "failed"
            log(f"speech to text: {self.label} failed to load: {self.error}")
        self.loaded.set()

    def info(self):
        out = {"model": self.label, "ready": self.state == "ready", "state": self.state}
        if self.error:
            out["error"] = self.error
        return out


def load_whisper(compute, root, threads=0):
    """Returns loader(name) for STT: a faster-whisper model on the CPU.
    name is a model name, such as small.en, or a folder with a converted
    model. A name is downloaded from Hugging Face into root once. threads
is CTranslate2's cpu_threads: 0 is its own default."""

    def loader(name):
        from faster_whisper import WhisperModel

        return WhisperModel(name, device="cpu", compute_type=compute, download_root=root, cpu_threads=threads)

    return loader


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
        job, t0 = self.enqueue(key, lambda: Job(key, text, voice, speed, prio), prio)
        self.stats.add(misses=1)
        return self.wait(job, prio, t0), False

    def enqueue(self, key, make, prio):
        """Returns (job, t0): the job for key, new from make() or one that
        is in the queue already."""
        t0 = time.monotonic()
        with self.cv:
            job = self.jobs.get(key)
            if job is None:
                if prio == LATER and len(self.queues[LATER]) >= self.later_limit:
                    self.busy_503 += 1
                    raise Busy(self.retry_after())
                job = make()
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
        return job, t0

    def wait(self, job, prio, t0):
        job.done.wait()
        with self.cv:
            started = job.started if job.started is not None else time.monotonic()
            self.wait_ms[prio] = round(max(0.0, started - t0) * 1000)
        if job.error is not None:
            raise job.error
        return job.result

    def retry_after(self):
        """Seconds until the first later job is made and the queue has room,
        roughly. Called with cv held."""
        q = self.queues[LATER]
        s = q[0].left_s(self) if q else self.sentence_s
        return max(1, min(300, math.ceil(s)))

    def pick(self, now_only):
        for prio in (NOW,) if now_only else (NOW, LATER):
            job = next((job for job in self.queues[prio] if not job.busy), None)
            if job is not None:
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
            t0 = time.monotonic()
            err = None
            try:
                finished = job.step(self)
            except Exception as e:  # noqa: BLE001 - given to the waiting requests
                err, finished = e, True
            with self.cv:
                job.busy = False
                job.work_ms += round((time.monotonic() - t0) * 1000)
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
                    self.stats.times.append(job.work_ms)
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


def handler(sp, log, stt=None, tr=None):
    """Returns the request handler class. sp is the Speaker; stt the STT
    and tr its Transcriber, made here when not given."""
    st = sp.stats
    stt = stt or STT("")
    tr = tr or Transcriber(stt, st)

    def stt_info():
        return {**stt.info(), **tr.queue_info(), "jobs": True}

    class H(BaseHTTPRequestHandler):
        def send(self, code, body, ctype="application/json", headers=()):
            if isinstance(body, (dict, list)):
                body = json.dumps(body).encode()
            elif isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            if code != 204:  # a 204 has no body and no Content-Length
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
            for k, v in headers:
                self.send_header(k, v)
            self.end_headers()
            if code != 204:
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
                        "transcribe": stt_info(),
                    },
                )
            if self.path == "/stats":
                return self.send(200, st.json(sp, stt, tr))
            if self.path == "/v1/voices":
                return self.send(200, {"voices": sp.voices, "default": DEFAULT_VOICE})
            if self.path.startswith(JOBS + "/"):
                return self.job()
            self.send(404, {"error": "not found"})

        def do_DELETE(self):
            path = self.path.split("?")[0]
            if not path.startswith(JOBS + "/"):
                return self.send(404, {"error": "not found"})
            if not tr.cancel(path[len(JOBS) + 1 :]):
                return self.send(404, {"state": "unknown", "error": "no such transcription"})
            self.send(204, b"")

        def job(self):
            path, _, query = self.path.partition("?")
            job = tr.get(path[len(JOBS) + 1 :])
            if job is None:
                return self.send(404, {"state": "unknown", "error": "no such transcription"})
            try:
                wait = float(urllib.parse.parse_qs(query).get("wait", ["0"])[0])
            except ValueError:
                return self.send(400, {"error": "wait must be a number of seconds"})
            if wait > 0:
                job.done.wait(min(wait, MAX_WAIT))
            self.send(200, tr.status(job))

        def do_POST(self):
            if self.path in ("/v1/transcribe", "/v1/transcriptions"):
                st.add(stt_requests=1, stt_in_flight=1)
                try:
                    if self.path == "/v1/transcribe":
                        self.transcribe()
                    else:
                        self.submit()
                finally:
                    st.add(stt_in_flight=-1)
                return
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

        def body(self, limit):
            """Reads the request body, sent with Content-Length or chunked.
            Raises TooLarge past limit bytes."""
            if self.headers.get("Transfer-Encoding", "").strip().lower() == "chunked":
                out = bytearray()
                while True:
                    n = int(self.rfile.readline(100).split(b";")[0].strip(), 16)
                    if n == 0:
                        while self.rfile.readline(1000) not in (b"\r\n", b"\n", b""):
                            pass
                        return bytes(out)
                    if len(out) + n > limit:
                        raise TooLarge()
                    out += self.rfile.read(n)
                    self.rfile.readline(10)
            n = int(self.headers.get("Content-Length") or 0)
            if n > limit:
                # Read it all if it is not huge, so the client gets the
                # answer and not a reset connection.
                while 0 < n <= 8 * limit and self.rfile.read(min(n, 1 << 16)):
                    n -= 1 << 16
                raise TooLarge()
            if n < 0:
                raise ValueError("bad Content-Length")
            return self.rfile.read(n)

        def clip(self):
            """Reads and checks a transcription request. Returns (data,
            language, prompt, clip ID, priority), or None when it answered
            the request with an error."""
            prio = LATER if self.headers.get("X-Voice-Priority", "").strip().lower() == LATER else NOW
            try:
                f = form(self.headers.get("Content-Type", ""), self.body(MAX_FORM))
                data = f.get("file") or b""
                if not data:
                    raise ValueError("no file")
                if len(data) > MAX_AUDIO:
                    raise TooLarge()
                if not is_ogg_opus(data):
                    raise ValueError("the file is not Ogg Opus")
                language = f.get("language", b"").decode().strip().lower()
                if language and not re.fullmatch(r"[a-z]{2}", language):
                    raise ValueError("language must be an ISO 639-1 code, such as en")
                prompt = f.get("prompt", b"").decode().strip()
                if len(prompt) > MAX_PROMPT:
                    raise ValueError(f"prompt must be at most {MAX_PROMPT} characters")
                clip = re.sub(r"[^\w.:-]", "", f.get("id", b"").decode(errors="replace"))[:64]
            except TooLarge:
                st.add(stt_bad=1)
                return self.send(413, {"error": f"the file must be at most {MAX_AUDIO >> 20} MB"})
            except ValueError as e:
                st.add(stt_bad=1)
                return self.send(400, {"error": str(e)})
            return data, language, prompt, clip, prio

        def submit(self):
            """POST /v1/transcriptions: queues the clip and answers at once
            with its job ID (ADR 0001)."""
            req = self.clip()
            if req is None:
                return
            data, language, prompt, clip, prio = req
            if stt.state in ("off", "failed"):
                return self.send(501, {"error": "speech to text is not available"})
            try:
                # While the model loads, a later request joins a job for the
                # same clip, but makes no new one.
                job = tr.submit(data, language, prompt, prio, new=not (stt.state == "loading" and prio == LATER))
                if job is None:
                    return self.send(503, {"error": "the speech to text model is loading"}, headers=[("Retry-After", "30")])
            except Busy as e:
                return self.send(503, {"error": "busy, try again later"}, headers=[("Retry-After", str(e.retry_after))])
            log(f"queued {clip or '-'} ({prio}) as {job.key[:12]}")
            self.send(202, tr.status(job), headers=[("Location", f"{JOBS}/{job.key}")])

        def transcribe(self):
            req = self.clip()
            if req is None:
                return
            data, language, prompt, clip, prio = req
            if not stt.loaded.is_set():
                # The model loads after start. A now request waits for it;
                # a later one comes back.
                if prio == LATER or not stt.loaded.wait(120):
                    return self.send(503, {"error": "the speech to text model is loading"}, headers=[("Retry-After", "30")])
            if stt.state != "ready":
                return self.send(501, {"error": "speech to text is not available"})
            try:
                out = tr.transcribe(data, language, prompt, prio)
            except Busy as e:
                return self.send(503, {"error": "busy, try again later"}, headers=[("Retry-After", str(e.retry_after))])
            except BadAudio as e:
                return self.send(400, {"error": str(e)[:200]})
            except NoModel:
                return self.send(501, {"error": "speech to text is not available"})
            except Exception as e:  # noqa: BLE001 - report any model failure
                log(f"transcribe failed: {e}")
                return self.send(500, {"error": "transcription failed"})
            # The log has no text: it can be private.
            log(f"transcribed {clip or '-'} ({prio}): {out['duration_ms'] / 1000:.1f} s, {len(out['segments'])} segments")
            self.send(200, out)

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
    env = os.environ.get
    ap.add_argument(
        "--stt-model",
        default=env("VOICESERVER_STT_MODEL", STT_MODEL),
        help="faster-whisper model for /v1/transcribe: a name such as small.en, a folder, or off (env VOICESERVER_STT_MODEL)",
    )
    ap.add_argument("--stt-compute", default=env("VOICESERVER_STT_COMPUTE", "int8"), help="CTranslate2 compute type on the CPU (env VOICESERVER_STT_COMPUTE)")
    ap.add_argument(
        "--stt-threads",
        type=int,
        default=int(env("VOICESERVER_STT_THREADS") or "0"),
        help="CPU threads for speech to text; 0 is half the cores (env VOICESERVER_STT_THREADS)",
    )
    ap.add_argument("--stt-queue-limit", type=int, default=STT_QUEUE_LIMIT, help="transcriptions in the queue before a new one gets 503")
    ap.add_argument("--stt-queue-mb", type=int, default=STT_QUEUE_MB, help="MB of audio in the transcription queue before a new one gets 503")
    ap.add_argument("--stt-dir", default=env("VOICESERVER_STT_DIR", "stt-models"), help="folder for downloaded speech to text models (env VOICESERVER_STT_DIR)")
    ap.add_argument("--check", action="store_true", help="check that the running server answers, and exit")
    args = ap.parse_args()
    if args.check:
        check()
    if args.workers < 1 or args.later_limit < 1:
        ap.error("--workers and --later-limit must be 1 or more")
    if args.stt_threads < 0 or args.stt_queue_limit < 1 or args.stt_queue_mb < 1:
        ap.error("--stt-threads must be 0 or more, --stt-queue-limit and --stt-queue-mb 1 or more")

    def log(msg):
        print(msg, file=sys.stderr, flush=True)

    synth, voices = load_kokoro(args.models)
    cache = Cache(args.cache, args.cache_mb << 20)
    sp = Speaker(synth, voices, cache, Stats(), workers=args.workers, later_limit=args.later_limit)
    host, _, port = args.listen.rpartition(":")
    threads = args.stt_threads or half_the_cores()
    stt = STT(args.stt_model, load_whisper(args.stt_compute, args.stt_dir, threads))
    tr = Transcriber(stt, sp.stats, args.stt_queue_limit, args.stt_queue_mb << 20, nice=STT_NICE)
    srv = ThreadingHTTPServer((host.strip("[]") or "0.0.0.0", int(port)), handler(sp, log, stt, tr))
    with open(LISTEN_FILE, "w") as f:
        f.write(args.listen)
    log(f"voice server on {args.listen}, {len(sp.voices)} voices, default {DEFAULT_VOICE}, {sp.workers} workers, later limit {sp.later_limit}")
    if stt.name:
        # The server speaks while the model loads (and downloads, the first
        # time); /health says when it is ready.
        # The model's threads are made here, so they get the lower priority
        # of the thread that loads it.
        def load():
            lower_priority(STT_NICE)
            stt.load(log)

        log(f"speech to text: loading {stt.label}, {threads} threads")
        threading.Thread(target=load, daemon=True, name="stt-load").start()
    else:
        log("speech to text is off")
    srv.serve_forever()


if __name__ == "__main__":
    main()
