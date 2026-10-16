"""Tests for the voice server queue (ADR 0009). They use a fake synth, so
they need numpy only, not Kokoro or the model files:

    python3 -I test_server.py
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
