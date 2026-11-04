"""Tests for the voice server queue (ADR 0009) and POST /v1/transcribe.
They use a fake synth and a fake Whisper model, so they need numpy only,
not Kokoro or the model files:

    python3 -I test_server.py

DecodeTest decodes real Ogg Opus when PyAV and faster-whisper are
installed (requirements.txt), and is skipped when they are not.
"""

import hashlib
import json
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import server  # noqa: E402

VOICE = "af_sarah"


class FakeSynth:
    """Sleeps delay seconds for each call and returns a short tone. A text in
    gates waits for its event first, so a test can keep a worker busy."""

    def __init__(self, delay=0.0):
        self.delay = delay
        self.calls = []
        self.lock = threading.Lock()
        self.gates = {}
        self.started = threading.Event()

    def gate(self, text):
        self.gates[text] = threading.Event()
        return self.gates[text]

    def __call__(self, text, voice, speed, lang):
        with self.lock:
            self.calls.append(text)
        self.started.set()
        if text in self.gates:
            self.gates[text].wait(10)
        time.sleep(self.delay)
        t = np.arange(server.RATE // 10) / server.RATE
        return 0.3 * np.sin(2 * np.pi * 440 * t), server.RATE

    def count(self, text):
        with self.lock:
            return self.calls.count(text)


def wait_until(cond, timeout=5):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.005)
    raise AssertionError("timed out")


class Base(unittest.TestCase):
    workers = 1
    later_limit = server.LATER_LIMIT
    delay = 0.0

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.synth = FakeSynth(self.delay)
        cache = server.Cache(self.tmp.name, 50 << 20)
        self.sp = server.Speaker(self.synth, [VOICE, "bf_emma"], cache, server.Stats(), self.workers, self.later_limit)
        self.results = {}
        self.threads = []

    def tearDown(self):
        for g in self.synth.gates.values():
            g.set()
        for t in self.threads:
            t.join(10)
        self.tmp.cleanup()

    def ask(self, text, prio=server.NOW):
        """Starts a request in a thread. Its (wav, hit) or exception goes to
        self.results[name]."""
        name = f"{prio}:{text}:{len(self.threads)}"

        def run():
            try:
                self.results[name] = self.sp.speech(text, VOICE, 0.8, prio)
            except Exception as e:  # noqa: BLE001
                self.results[name] = e

        t = threading.Thread(target=run)
        t.start()
        self.threads.append(t)
        return name

    def queued(self, now, later):
        wait_until(lambda: self.sp.queue_info()["queue"] == {"now": now, "later": later})

    def join(self):
        for t in self.threads:
            t.join(10)


class QueueTest(Base):
    def test_now_goes_before_queued_later(self):
        self.synth.gate("A.").clear()
        self.ask("A.", server.LATER)
        wait_until(self.synth.started.is_set)
        self.ask("B.", server.LATER)
        self.queued(0, 2)
        self.ask("C.", server.NOW)
        self.queued(1, 2)
        self.synth.gates["A."].set()
        self.join()
        self.assertEqual(self.synth.calls, ["A.", "C.", "B."])

    def test_fifo_within_a_priority(self):
        self.synth.gate("A.")
        self.ask("A.", server.LATER)
        wait_until(self.synth.started.is_set)
        for i, (text, prio) in enumerate([("L1.", "later"), ("N1.", "now"), ("L2.", "later"), ("N2.", "now"), ("L3.", "later")]):
            self.ask(text, prio)
            wait_until(lambda i=i: sum(self.sp.queue_info()["queue"].values()) == i + 2)
        self.synth.gates["A."].set()
        self.join()
        self.assertEqual(self.synth.calls, ["A.", "N1.", "N2.", "L1.", "L2.", "L3."])

    def test_promotion_makes_the_text_once(self):
        self.synth.gate("A.")
        self.ask("A.", server.LATER)
        wait_until(self.synth.started.is_set)
        self.ask("C.", server.LATER)
        self.queued(0, 2)
        later = self.ask("B.", server.LATER)
        self.queued(0, 3)
        now = self.ask("B.", server.NOW)
        self.queued(1, 2)
        self.synth.gates["A."].set()
        self.join()
        self.assertEqual(self.synth.count("B."), 1)
        self.assertEqual(self.synth.calls, ["A.", "B.", "C."])
        self.assertEqual(self.results[later][0], self.results[now][0])
        self.assertTrue(self.results[now][0].startswith(b"RIFF"))

    def test_now_waits_for_one_sentence_of_a_long_later_text(self):
        long = "One is here. Two is here. Three is here. Four is here. Five is here."
        self.synth.gate("One is here.")
        later = self.ask(long, server.LATER)
        wait_until(self.synth.started.is_set)
        now = self.ask("Now.", server.NOW)
        self.queued(1, 1)
        self.synth.gates["One is here."].set()
        self.join()
        self.assertEqual(self.synth.calls, ["One is here.", "Now.", "Two is here.", "Three is here.", "Four is here.", "Five is here."])
        self.assertIsInstance(self.results[later][0], bytes)
        self.assertIsInstance(self.results[now][0], bytes)

    def test_whole_text_is_cached_under_the_old_key(self):
        text = "First part. Second part."
        name = self.ask(text, server.LATER)
        self.join()
        wav, hit = self.results[name]
        self.assertFalse(hit)
        old = hashlib.sha256(f"{server.MODEL}\n{VOICE}\n0.80\n{text}".encode()).hexdigest()
        self.assertEqual(self.sp.cache.get(old), wav)
        self.assertEqual(self.sp.speech(text, VOICE, 0.8), (wav, True))
        self.assertEqual(self.synth.calls, ["First part.", "Second part."])

    def test_synth_error_reaches_every_waiter(self):
        def bad(text, voice, speed, lang):
            raise RuntimeError("boom")

        self.sp.synth = bad
        with self.assertRaises(RuntimeError):
            self.sp.speech("Fails.", VOICE, 0.8)
        self.assertEqual(self.sp.queue_info()["queue"], {"now": 0, "later": 0})


class TimingTest(Base):
    delay = 0.2

    def test_now_wait_is_about_one_sentence(self):
        long = " ".join(f"Sentence {i} is here." for i in range(8))
        later = self.ask(long, server.LATER)
        wait_until(self.synth.started.is_set)
        time.sleep(0.05)
        t0 = time.monotonic()
        self.assertEqual(self.sp.speech("Now.", VOICE, 0.8)[1], False)
        took = time.monotonic() - t0
        # One later sentence (at most 0.2 s left) and its own (0.2 s).
        self.assertLess(took, 0.6)
        self.assertNotIn(later, self.results)  # the later text is not done yet
        self.join()
        self.assertLess(self.sp.queue_info()["wait_ms"]["now"], 400)


class WorkersTest(Base):
    workers = 2

    def test_one_worker_takes_only_now(self):
        self.synth.gate("A.")
        self.ask("A.", server.LATER)
        wait_until(self.synth.started.is_set)
        self.ask("B.", server.LATER)
        self.queued(0, 2)
        time.sleep(0.05)
        # The now-only worker is free but must not take B.
        self.assertEqual(self.synth.calls, ["A."])
        self.assertEqual(self.sp.speech("N.", VOICE, 0.8)[1], False)
        self.assertEqual(self.synth.calls, ["A.", "N."])
        self.synth.gates["A."].set()
        self.join()
        self.assertEqual(self.synth.calls, ["A.", "N.", "B."])


class HTTPTest(Base):
    later_limit = 2

    def setUp(self):
        super().setUp()
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), server.handler(self.sp, lambda m: None))
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.srv.server_address[1]}"

    def tearDown(self):
        super().tearDown()
        self.srv.shutdown()
        self.srv.server_close()

    def post(self, text, prio=None):
        h = {"Content-Type": "application/json"}
        if prio is not None:
            h["X-Voice-Priority"] = prio
        req = urllib.request.Request(self.base + "/v1/speech", json.dumps({"text": text}).encode(), h)
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, dict(r.headers), r.read()
        except urllib.error.HTTPError as e:
            with e:
                return e.code, dict(e.headers), e.read()

    def post_bg(self, text, prio=None):
        name = f"http:{text}:{len(self.threads)}"
        t = threading.Thread(target=lambda: self.results.__setitem__(name, self.post(text, prio)))
        t.start()
        self.threads.append(t)
        return name

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=10) as r:
            return json.loads(r.read())

    def test_no_header_and_unknown_header_are_now(self):
        self.synth.gate("A.")
        self.post_bg("A.", "later")
        wait_until(self.synth.started.is_set)
        self.post_bg("B.", "later")
        self.queued(0, 2)
        self.post_bg("C.")
        self.queued(1, 2)
        self.post_bg("D.", "urgent")
        self.queued(2, 2)
        self.synth.gates["A."].set()
        self.join()
        self.assertEqual(self.synth.calls, ["A.", "C.", "D.", "B."])
        for status, headers, body in self.results.values():
            self.assertEqual(status, 200)
            self.assertEqual(headers["Content-Type"], "audio/wav")
            self.assertEqual(headers["X-Cache"], "miss")
            self.assertTrue(body.startswith(b"RIFF"))
        status, headers, _ = self.post("C.")
        self.assertEqual((status, headers["X-Cache"]), (200, "hit"))

    def test_busy_503_for_later_only(self):
        self.synth.gate("A.")
        self.post_bg("A.", "later")
        wait_until(self.synth.started.is_set)
        self.post_bg("B.", "later")
        self.queued(0, 2)
        status, headers, body = self.post("C.", "later")
        self.assertEqual(status, 503)
        self.assertGreaterEqual(int(headers["Retry-After"]), 1)
        self.assertIn("error", json.loads(body))
        # A later request for a text in the queue joins it, no 503.
        self.post_bg("B.", "later")
        # A now request is never refused.
        self.post_bg("N.")
        self.queued(1, 2)
        h = self.get("/health")
        self.assertEqual(h["busy_503"], 1)
        self.assertEqual(h["queue"], {"now": 1, "later": 2})
        self.synth.gates["A."].set()
        self.join()
        self.assertEqual([r[0] for r in self.results.values()], [200, 200, 200, 200])
        self.assertEqual(self.synth.count("B."), 1)

    def test_health_fields(self):
        self.assertEqual(self.post("Hello.", "later")[0], 200)
        self.assertEqual(self.post("Hi.")[0], 200)
        h = self.get("/health")
        for k in ("status", "version", "model", "voices", "uptime_s"):
            self.assertIn(k, h)
        self.assertEqual(h["status"], "ok")
        self.assertEqual(h["queue"], {"now": 0, "later": 0})
        self.assertEqual(set(h["wait_ms"]), {"now", "later"})
        self.assertIsInstance(h["wait_ms"]["now"], int)
        self.assertIsInstance(h["wait_ms"]["later"], int)
        self.assertEqual(h["busy_503"], 0)
        self.assertEqual(h["workers"], 1)
        s = self.get("/stats")
        for k in ("requests", "ok", "in_flight", "waiting", "synth", "cache", "queue", "wait_ms", "busy_503"):
            self.assertIn(k, s)
        self.assertEqual((s["requests"], s["ok"], s["cache"]["misses"]), (2, 2, 2))

    def test_bad_request_unchanged(self):
        status, _, body = self.post("")
        self.assertEqual(status, 400)
        self.assertIn("error", json.loads(body))


class Seg:
    """A faster-whisper Segment, with the fields that the server uses."""

    def __init__(self, start, end, text, avg_logprob=-0.2):
        self.start, self.end, self.text, self.avg_logprob = start, end, text, avg_logprob


class Info:
    def __init__(self, language, duration):
        self.language, self.duration = language, duration


SEGS = [Seg(0.0, 3.1, " W1AW, this is K1ABC,", -0.2), Seg(3.1, 6.24, " checking in, no traffic.", -0.1)]


class FakeWhisper:
    """A faster-whisper model. Its segments are yielded one at a time, and
    each one is noted in events (the FakeSynth's calls), so a test can
    see the order of speech and segments. A segment text in gates waits
    for its event first."""

    def __init__(self, synth, segments=SEGS):
        self.synth, self.segments = synth, segments
        self.calls = []  # (language, hotwords, options) of each transcribe
        self.gates = {}

    def transcribe(self, audio, language=None, hotwords=None, **kw):
        self.calls.append((language, hotwords, kw))

        def gen():
            for s in self.segments:
                with self.synth.lock:
                    self.synth.calls.append("seg:" + s.text.strip())
                self.synth.started.set()
                if s.text in self.gates:
                    self.gates[s.text].wait(10)
                yield s

        return gen(), Info(language or "en", len(audio) / server.STT_RATE)


def fake_decode(data):
    if b"BAD" in data:
        raise server.BadAudio("the file can't be decoded: bad")
    return np.zeros(int(6.24 * server.STT_RATE), dtype=np.float32)


def ogg(extra=b""):
    """Bytes that start as an Ogg Opus file: one page with OpusHead."""
    return b"OggS" + bytes(22) + bytes([1, 19]) + b"OpusHead" + bytes(11) + extra


def multipart(fields):
    """Returns (body, content type) for fields {name: bytes or str}."""
    bd = "x7Kf9TestBoundary"
    out = b""
    for k, v in fields.items():
        v = v.encode() if isinstance(v, str) else v
        extra = '; filename="clip.opus"\r\nContent-Type: audio/ogg' if k == "file" else ""
        out += f'--{bd}\r\nContent-Disposition: form-data; name="{k}"{extra}\r\n\r\n'.encode() + v + b"\r\n"
    return out + f"--{bd}--\r\n".encode(), f"multipart/form-data; boundary={bd}"


class TranscribeTest(Base):
    later_limit = 2

    def setUp(self):
        super().setUp()
        self.whisper = FakeWhisper(self.synth)
        self.stt = self.make_stt()
        self.tr = server.Transcriber(self.stt, self.sp.stats, queue_limit=2)
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), server.handler(self.sp, lambda m: None, self.stt, self.tr))
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.srv.server_address[1]}"

    def make_stt(self):
        stt = server.STT("small.en", lambda name: self.whisper, fake_decode)
        stt.load(lambda m: None)
        return stt

    def tearDown(self):
        for g in self.whisper.gates.values():
            g.set()
        super().tearDown()
        self.srv.shutdown()
        self.srv.server_close()

    def request(self, path, body, ctype, prio=None):
        h = {"Content-Type": ctype}
        if prio is not None:
            h["X-Voice-Priority"] = prio
        req = urllib.request.Request(self.base + path, body, h)
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, dict(r.headers), r.read()
        except urllib.error.HTTPError as e:
            with e:
                return e.code, dict(e.headers), e.read()

    def post(self, fields, prio=None):
        status, headers, body = self.request("/v1/transcribe", *multipart(fields), prio)
        return status, headers, json.loads(body) if body else None

    def post_bg(self, fields, prio=None):
        name = f"stt:{len(self.threads)}"
        t = threading.Thread(target=lambda: self.results.__setitem__(name, self.post(fields, prio)))
        t.start()
        self.threads.append(t)
        return name

    def speech_bg(self, text, prio=None):
        name = f"tts:{text}:{len(self.threads)}"

        def run():
            self.results[name] = self.request("/v1/speech", json.dumps({"text": text}).encode(), "application/json", prio)[0]

        t = threading.Thread(target=run)
        t.start()
        self.threads.append(t)
        return name

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=10) as r:
            return json.loads(r.read())

    def stt_queued(self, now, later):
        wait_until(lambda: self.tr.queue_info()["queue"] == {"now": now, "later": later})

    def gate(self):
        """Holds every transcription at its first segment until set."""
        g = self.whisper.gates[SEGS[0].text] = threading.Event()
        return g

    def test_answer(self):
        status, headers, out = self.post({"file": ogg(), "language": "EN", "prompt": "Net. Callsigns: W1AW, K1ABC.", "id": "clip-1"})
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(
            out,
            {
                "text": "W1AW, this is K1ABC, checking in, no traffic.",
                "language": "en",
                "duration_ms": 6240,
                "model": "small.en",
                "segments": [
                    {"start_ms": 0, "end_ms": 3100, "text": "W1AW, this is K1ABC,", "confidence": 0.82},
                    {"start_ms": 3100, "end_ms": 6240, "text": "checking in, no traffic.", "confidence": 0.9},
                ],
            },
        )
        language, prompt, kw = self.whisper.calls[0]
        self.assertEqual((language, prompt), ("en", "Net. Callsigns: W1AW, K1ABC."))
        self.assertTrue(kw["vad_filter"])

    def test_no_language_and_no_prompt_are_none(self):
        self.assertEqual(self.post({"file": ogg()})[0], 200)
        self.assertEqual(self.whisper.calls[0][:2], (None, None))

    def test_empty_text(self):
        self.whisper.segments = [Seg(0.0, 1.0, " ", -0.5)]
        status, _, out = self.post({"file": ogg()})
        self.assertEqual(status, 200)
        self.assertEqual((out["text"], out["segments"]), ("", []))
        self.whisper.segments = []
        status, _, out = self.post({"file": ogg(b"other clip")})
        self.assertEqual((status, out["text"], out["segments"]), (200, "", []))

    def test_confidence(self):
        self.assertEqual(server.confidence(0.0), 1.0)
        self.assertEqual(server.confidence(-0.693), 0.5)
        self.assertEqual(server.confidence(-50), 0.0)
        self.assertEqual(server.confidence(0.3), 1.0)

    def test_bad_requests(self):
        cases = [
            ({"language": "en"}, "no file"),
            ({"file": b"RIFF....WAVEfmt "}, "not Ogg Opus"),
            ({"file": b"OggS" + bytes(22) + bytes([1, 19]) + b"OpusTags"}, "not Ogg Opus"),
            ({"file": ogg(), "language": "english"}, "ISO 639-1"),
            ({"file": ogg(), "prompt": "x" * 1001}, "1000"),
            ({"file": ogg(b"BAD")}, "decoded"),
        ]
        for fields, msg in cases:
            status, _, out = self.post(fields)
            self.assertEqual(status, 400, fields.keys())
            self.assertIn(msg, out["error"])
        status, _, body = self.request("/v1/transcribe", b'{"file": "x"}', "application/json")
        self.assertEqual(status, 400)
        self.assertIn("multipart", json.loads(body)["error"])
        self.assertEqual(self.get("/stats")["transcribe"]["bad_requests"], 7)

    def test_too_large(self):
        old = server.MAX_AUDIO, server.MAX_FORM
        server.MAX_AUDIO, server.MAX_FORM = 1000, 2000
        try:
            status, _, out = self.post({"file": ogg(bytes(1200))})
            self.assertEqual(status, 413)
            self.assertIn("error", out)
            status, _, out = self.post({"file": ogg(bytes(3000))})
            self.assertEqual(status, 413)
            self.assertEqual(self.post({"file": ogg(bytes(500))})[0], 200)
        finally:
            server.MAX_AUDIO, server.MAX_FORM = old

    def test_chunked_body(self):
        import http.client

        body, ctype = multipart({"file": ogg()})
        c = http.client.HTTPConnection("127.0.0.1", self.srv.server_address[1], timeout=10)
        c.request("POST", "/v1/transcribe", iter([body[:50], body[50:]]), {"Content-Type": ctype}, encode_chunked=True)
        r = c.getresponse()
        self.assertEqual(r.status, 200)
        self.assertEqual(json.loads(r.read())["duration_ms"], 6240)
        c.close()

    def test_busy_503_for_later_only(self):
        g = self.gate()
        first = self.post_bg({"file": ogg(b"0")}, "later")
        wait_until(lambda: self.tr.queue_info()["processing"] == 1)
        self.post_bg({"file": ogg(b"1")}, "later")
        self.post_bg({"file": ogg(b"2")}, "later")
        self.stt_queued(0, 2)
        status, headers, out = self.post({"file": ogg(b"3")}, "later")
        self.assertEqual(status, 503)
        self.assertGreaterEqual(int(headers["Retry-After"]), 1)
        self.assertIn("error", out)
        # A now request to /v1/transcribe is never refused.
        now = self.post_bg({"file": ogg(b"4")})
        self.stt_queued(1, 2)
        g.set()
        self.join()
        self.assertEqual((self.results[first][0], self.results[now][0]), (200, 200))
        self.assertEqual(self.get("/health")["transcribe"]["busy_503"], 1)
        # Transcriptions do not fill the speech queue.
        self.assertEqual(self.get("/health")["busy_503"], 0)

    def test_speech_does_not_wait_for_transcription(self):
        g = self.gate()
        stt = self.post_bg({"file": ogg()}, "later")
        wait_until(lambda: "seg:" + SEGS[0].text.strip() in self.synth.calls)
        now, later = self.speech_bg("N."), self.speech_bg("L.", "later")
        for t in self.threads[1:]:
            t.join(10)
        self.assertEqual((self.results[now], self.results[later]), (200, 200))
        self.assertNotIn(stt, self.results)
        g.set()
        self.join()
        self.assertEqual(self.results[stt][0], 200)

    def test_now_goes_before_queued_later(self):
        g = self.gate()
        self.post_bg({"file": ogg(b"0")}, "later")
        wait_until(lambda: self.tr.queue_info()["processing"] == 1)
        self.post_bg({"file": ogg(b"LL")}, "later")
        self.stt_queued(0, 1)
        self.post_bg({"file": ogg(b"NNN")})
        self.stt_queued(1, 1)
        g.set()
        self.join()
        order = [j.size for j in sorted(self.tr.jobs.values(), key=lambda j: j.started)]
        self.assertEqual(order, [len(ogg(b"0")), len(ogg(b"NNN")), len(ogg(b"LL"))])

    def test_queue_bytes_limit(self):
        self.tr.queue_bytes = len(ogg(b"1" * 10)) + len(ogg(b"2" * 10)) - 1
        g = self.gate()
        self.post_bg({"file": ogg(b"0")}, "later")
        wait_until(lambda: self.tr.queue_info()["processing"] == 1)
        self.post_bg({"file": ogg(b"1" * 10)}, "later")
        self.stt_queued(0, 1)
        self.assertEqual(self.post({"file": ogg(b"2" * 10)}, "later")[0], 503)
        g.set()
        self.join()

    def test_hotwords_go_to_every_window(self):
        self.assertEqual(self.post({"file": ogg(), "prompt": "KC Wide. Callsigns: KZ4JON."})[0], 200)
        _, hotwords, kw = self.whisper.calls[0]
        self.assertEqual(hotwords, "KC Wide. Callsigns: KZ4JON.")
        self.assertNotIn("initial_prompt", kw)

    def test_same_clip_is_made_once(self):
        g = self.gate()
        self.post_bg({"file": ogg(b"0")}, "later")
        wait_until(lambda: self.tr.queue_info()["processing"] == 1)
        a = self.post_bg({"file": ogg(), "id": "a"}, "later")
        self.stt_queued(0, 1)
        b = self.post_bg({"file": ogg(), "id": "b"})
        self.stt_queued(1, 0)  # b promoted a
        g.set()
        self.join()
        self.assertEqual(len(self.whisper.calls), 2)
        self.assertEqual(self.results[a][2], self.results[b][2])
        # The finished job answers the next request without the model.
        self.assertEqual(self.post({"file": ogg()})[2], self.results[a][2])
        self.assertEqual(len(self.whisper.calls), 2)

    def test_finished_jobs_expire(self):
        self.assertEqual(self.post({"file": ogg()})[0], 200)
        self.tr.keep_s = 0
        time.sleep(0.01)
        self.assertEqual(self.post({"file": ogg(b"other")})[0], 200)
        self.assertEqual(len(self.tr.jobs), 1)

    def call(self, method, path):
        req = urllib.request.Request(self.base + path, method=method)
        try:
            with urllib.request.urlopen(req, timeout=70) as r:
                body = r.read()
                return r.status, json.loads(body) if body else None
        except urllib.error.HTTPError as e:
            with e:
                body = e.read()
                return e.code, json.loads(body) if body else None

    def submit(self, fields, prio=None):
        status, headers, body = self.request(server.JOBS, *multipart(fields), prio)
        return status, headers, json.loads(body) if body else None

    def test_job_flow(self):
        status, headers, out = self.submit({"file": ogg(), "language": "en", "id": "clip-1"})
        self.assertEqual(status, 202)
        self.assertEqual(headers["Location"], f"{server.JOBS}/{out['id']}")
        self.assertIn(out["state"], ("queued", "processing", "done"))
        status, done = self.call("GET", f"{server.JOBS}/{out['id']}?wait=5")
        self.assertEqual((status, done["state"]), (200, "done"))
        # The result is the answer of POST /v1/transcribe for the same clip.
        self.assertEqual(done["result"], self.post({"file": ogg(), "language": "en"})[2])
        self.assertEqual(len(self.whisper.calls), 1)
        t = self.get("/stats")["transcribe"]
        self.assertEqual((t["requests"], t["ok"], t["jobs"]), (2, 1, True))

    def test_job_states(self):
        g = self.gate()
        a = self.submit({"file": ogg(b"A")}, "later")[2]["id"]
        wait_until(lambda: self.tr.queue_info()["processing"] == 1)
        status, _, b = self.submit({"file": ogg(b"B")}, "later")
        self.assertEqual((status, b["state"], b["position"], b["priority"]), (202, "queued", 1, "later"))
        n = self.submit({"file": ogg(b"N")})[2]
        self.assertEqual((n["position"], self.call("GET", f"{server.JOBS}/{b['id']}")[1]["position"]), (1, 2))
        status, s = self.call("GET", f"{server.JOBS}/{a}")
        self.assertEqual((status, s["state"], s["progress_ms"], s["duration_ms"]), (200, "processing", 0, 6240))
        g.set()
        self.assertEqual(self.call("GET", f"{server.JOBS}/{b['id']}?wait=5")[1]["state"], "done")

    def test_job_cancel(self):
        g = self.gate()
        a = self.submit({"file": ogg(b"A")}, "later")[2]["id"]
        wait_until(lambda: self.tr.queue_info()["processing"] == 1)
        b = self.submit({"file": ogg(b"B")}, "later")[2]["id"]
        self.assertEqual(self.call("DELETE", f"{server.JOBS}/{b}"), (204, None))
        self.assertEqual(self.tr.queue_info()["queue"], {"now": 0, "later": 0})
        self.assertEqual(self.call("GET", f"{server.JOBS}/{b}")[0], 404)
        job = self.tr.get(a)
        self.assertEqual(self.call("DELETE", f"{server.JOBS}/{a}")[0], 204)
        g.set()
        self.assertTrue(job.done.wait(5))
        self.assertEqual((job.state, job.reason), ("failed", "cancelled"))
        self.assertEqual(len(self.whisper.calls), 1)
        wait_until(lambda: self.tr.queue_info()["processing"] == 0)
        # A new request for a cancelled clip makes it again.
        c = self.submit({"file": ogg(b"A")})[2]["id"]
        self.assertEqual(self.call("GET", f"{server.JOBS}/{c}?wait=5")[1]["state"], "done")

    def test_job_failed(self):
        out = self.submit({"file": ogg(b"BAD")})[2]
        status, s = self.call("GET", f"{server.JOBS}/{out['id']}?wait=5")
        self.assertEqual((status, s["state"], s["reason"]), (200, "failed", "bad_audio"))
        self.assertIn("decoded", s["error"])
        self.assertEqual(self.get("/stats")["transcribe"]["bad_requests"], 1)
        self.assertEqual(self.submit({"language": "en"})[0], 400)

    def test_job_busy(self):
        g = self.gate()
        self.submit({"file": ogg(b"0")}, "later")
        wait_until(lambda: self.tr.queue_info()["processing"] == 1)
        self.submit({"file": ogg(b"1")}, "later")
        self.submit({"file": ogg(b"2")}, "later")
        status, headers, _ = self.submit({"file": ogg(b"3")}, "later")
        self.assertEqual(status, 503)
        self.assertGreaterEqual(int(headers["Retry-After"]), 1)
        # A job is never waited for, so a full queue refuses now too.
        self.assertEqual(self.submit({"file": ogg(b"4")})[0], 503)
        # The same clip joins its job.
        self.assertEqual(self.submit({"file": ogg(b"1")})[0], 202)
        g.set()

    def test_unknown_job(self):
        for method in ("GET", "DELETE"):
            status, out = self.call(method, f"{server.JOBS}/nothing")
            self.assertEqual((status, out["state"]), (404, "unknown"))
        self.assertEqual(self.call("GET", f"{server.JOBS}/x?wait=soon")[0], 404)
        self.assertTrue(self.get("/health")["transcribe"]["jobs"])

    def test_health_and_stats(self):
        self.assertEqual(self.post({"file": ogg()})[0], 200)
        h = self.get("/health")
        self.assertEqual(
            h["transcribe"],
            {
                "model": "small.en",
                "ready": True,
                "state": "ready",
                "queue": {"now": 0, "later": 0},
                "processing": 0,
                "wait_ms": {"now": h["transcribe"]["wait_ms"]["now"], "later": None},
                "busy_503": 0,
                "jobs": True,
            },
        )
        self.assertEqual(h["version"], server.VERSION)
        t = self.get("/stats")["transcribe"]
        self.assertEqual((t["requests"], t["ok"], t["errors"], t["in_flight"], t["ready"]), (1, 1, 0, 0, True))
        self.assertEqual(t["audio_s"], 6.2)
        self.assertEqual((t["queue"], t["processing"], t["busy_503"]), ({"now": 0, "later": 0}, 0, 0))
        # Speech counters stay for speech only.
        self.assertEqual(self.get("/stats")["requests"], 0)

    def test_model_error_is_500(self):
        def bad(*a, **kw):
            raise RuntimeError("boom")

        self.whisper.transcribe = bad
        status, _, out = self.post({"file": ogg()})
        self.assertEqual((status, out), (500, {"error": "transcription failed"}))
        s = self.get("/stats")
        self.assertEqual((s["transcribe"]["errors"], s["errors"]), (1, 0))
        self.assertEqual(self.tr.queue_info()["queue"], {"now": 0, "later": 0})
        # A failed job is made again by the next request.
        del self.whisper.transcribe
        self.assertEqual(self.post({"file": ogg()})[0], 200)

    def test_speech_still_works(self):
        status, headers, body = self.request("/v1/speech", json.dumps({"text": "Hello."}).encode(), "application/json")
        self.assertEqual((status, headers["Content-Type"]), (200, "audio/wav"))
        self.assertTrue(body.startswith(b"RIFF"))
        self.assertEqual(self.request("/v1/nothing", b"", "text/plain")[0], 404)


class TranscribeOffTest(TranscribeTest):
    """The model is off, or did not load: 501, and speech works."""

    def make_stt(self):
        return server.STT("off")

    def test_answer(self):
        status, _, out = self.post({"file": ogg()})
        self.assertEqual(status, 501)
        self.assertIn("error", out)
        t = self.get("/health")["transcribe"]
        self.assertEqual((t["model"], t["ready"], t["state"]), (None, False, "off"))
        self.assertEqual(self.submit({"file": ogg()})[0], 501)

    def test_failed_load(self):
        def fails(name):
            raise OSError("no such model")

        stt = server.STT("small.en", fails)
        stt.load(lambda m: None)
        self.assertEqual(stt.info(), {"model": "small.en", "ready": False, "state": "failed", "error": "OSError: no such model"})
        self.srv.RequestHandlerClass = server.handler(self.sp, lambda m: None, stt)
        self.assertEqual(self.post({"file": ogg()})[0], 501)
        self.assertEqual(self.get("/health")["transcribe"]["state"], "failed")

    # The tests that need a model do not apply.
    test_no_language_and_no_prompt_are_none = test_empty_text = test_bad_requests = None
    test_too_large = test_chunked_body = test_busy_503_for_later_only = None
    test_speech_does_not_wait_for_transcription = test_same_clip_is_made_once = None
    test_health_and_stats = test_model_error_is_500 = test_now_goes_before_queued_later = None
    test_queue_bytes_limit = test_hotwords_go_to_every_window = test_finished_jobs_expire = None
    test_job_flow = test_job_states = test_job_cancel = test_job_failed = test_job_busy = test_unknown_job = None


class TranscribeLoadingTest(TranscribeTest):
    """The model loads after start: later gets 503, now waits for it."""

    def make_stt(self):
        return server.STT("small.en", lambda name: self.whisper, fake_decode)

    def test_answer(self):
        self.assertEqual(self.get("/health")["transcribe"]["state"], "loading")
        status, headers, _ = self.post({"file": ogg()}, "later")
        self.assertEqual((status, headers["Retry-After"]), (503, "30"))
        now = self.post_bg({"file": ogg()})
        time.sleep(0.1)
        self.assertNotIn(now, self.results)
        # A job: later comes back, now is queued until the model is ready.
        status, headers, _ = self.submit({"file": ogg(b"job")}, "later")
        self.assertEqual((status, headers["Retry-After"]), (503, "30"))
        status, _, job = self.submit({"file": ogg(b"job")})
        self.assertEqual((status, job["state"]), (202, "queued"))
        self.stt.load(lambda m: None)
        self.join()
        self.assertEqual(self.results[now][0], 200)
        self.assertEqual(self.call("GET", f"{server.JOBS}/{job['id']}?wait=5")[1]["state"], "done")

    test_no_language_and_no_prompt_are_none = test_empty_text = test_bad_requests = None
    test_too_large = test_chunked_body = test_busy_503_for_later_only = None
    test_speech_does_not_wait_for_transcription = test_same_clip_is_made_once = None
    test_health_and_stats = test_model_error_is_500 = test_now_goes_before_queued_later = None
    test_queue_bytes_limit = test_hotwords_go_to_every_window = test_finished_jobs_expire = None
    test_job_flow = test_job_states = test_job_cancel = test_job_failed = test_job_busy = test_unknown_job = None


class FormTest(unittest.TestCase):
    def test_binary_file_survives(self):
        data = ogg(b"\r\n--x7Kf9\r\n\r\n\x00\xff" * 50 + b"\r\n")
        body, ctype = multipart({"id": "c1", "file": data, "prompt": "W1AW"})
        f = server.form(ctype, body)
        self.assertEqual(f, {"id": b"c1", "file": data, "prompt": b"W1AW"})
        self.assertEqual(server.form(ctype.replace("boundary=", 'boundary="') + '"', body)["file"], data)

    def test_ogg_opus_check(self):
        self.assertTrue(server.is_ogg_opus(ogg()))
        self.assertFalse(server.is_ogg_opus(b"OggS"))
        self.assertFalse(server.is_ogg_opus(b"ID3" + bytes(40)))


try:
    import av  # noqa: F401
    import faster_whisper  # noqa: F401

    HAVE_AV = True
except ImportError:
    HAVE_AV = False


@unittest.skipUnless(HAVE_AV, "needs PyAV and faster-whisper (requirements.txt)")
class DecodeTest(unittest.TestCase):
    def test_real_ogg_opus(self):
        """One second of a tone at 48 kHz mono, as the node records it,
        decodes to one second at 16 kHz."""
        import io

        import av

        b = io.BytesIO()
        with av.open(b, "w", format="ogg") as out:
            st = out.add_stream("libopus", rate=48000, layout="mono")
            t = np.arange(48000) / 48000
            pcm = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16).reshape(1, -1)
            for i in range(0, 48000, 960):
                fr = av.AudioFrame.from_ndarray(pcm[:, i : i + 960], format="s16", layout="mono")
                fr.sample_rate, fr.pts = 48000, i
                for p in st.encode(fr):
                    out.mux(p)
            for p in st.encode(None):
                out.mux(p)
        data = b.getvalue()
        self.assertTrue(server.is_ogg_opus(data))
        x = server.decode_ogg(data)
        self.assertEqual(x.dtype, np.float32)
        self.assertAlmostEqual(len(x) / server.STT_RATE, 1.0, delta=0.05)
        with self.assertRaises(server.BadAudio):
            server.decode_ogg(data[:40] + bytes(200))


class SentencesTest(unittest.TestCase):
    def test_split(self):
        s = server.sentences
        self.assertEqual(s("Node 2, 1, 0, 0, connected."), ["Node 2, 1, 0, 0, connected."])
        self.assertEqual(s("One. Two! Three? Four; five"), ["One.", "Two!", "Three?", "Four;", "five"])
        self.assertEqual(s("Until 10 a.m. Eastern time. Then rain."), ["Until 10 a.m. Eastern time.", "Then rain."])
        self.assertEqual(s("Near St. Louis. Wind W. at 10."), ["Near St. Louis.", "Wind W. at 10."])
        self.assertEqual(s("No mark"), ["No mark"])


if __name__ == "__main__":
    unittest.main()
