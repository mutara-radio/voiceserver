# Voice server API

This is the contract between the voice server and a Mutara node. The
node's side is `internal/voice/remote.go` in the Mutara repo. A change
here that an old node or an old server cannot use is a new major
version of this contract.

- **Contract version:** 1
- **Server version that this text agrees with:** 1.2.0

## POST /v1/speech

Makes speech from text.

Request body, JSON:

| Field   | Type   | Note                                              |
|---------|--------|---------------------------------------------------|
| `text`  | string | 1 to 400 characters after trimming spaces.        |
| `voice` | string | A Kokoro voice, such as `af_sarah`. Empty: the default. |
| `speed` | number | 0.5 to 2.0, rounded to 2 decimals. 0 or missing: 0.8. |

Request header, optional:

    X-Voice-Priority: now | later

- `now`: a listener waits for the answer.
- `later`: nobody waits now. The node gets the phrase before it plays.
- A missing header, or any other value, is `now`.

Answers:

- `200`, `audio/wav`: 24 kHz, mono, 16-bit, speech at -20 dBFS RMS.
  Header `X-Cache: hit` or `miss`.
- `400`: the request is not correct (no text, text too long, unknown
  voice, speed out of range, a body over 4096 bytes).
- `503` with `Retry-After` (seconds): the `later` queue is full. Only a
  `later` request gets this answer. The node waits for that time, then
  tries again.
- `500`: the synthesis failed.

The same text, voice and speed always give the same answer. Both sides
keep a cache. The server's cache key is the SHA-256 of
`"{model}\n{voice}\n{speed:.2f}\n{text}"`. This key does not change between
versions, so a cache stays valid after an upgrade.

## GET /v1/voices

    {"voices": ["af_alloy", ...], "default": "af_sarah"}

## GET /health

    {"status": "ok", "version": "1.2.0", "model": "...", "voices": 28,
     "uptime_s": 1234, "workers": 1, "queue": {"now": 0, "later": 3},
     "wait_ms": {"now": 120, "later": 4100}, "busy_503": 0}

The fields `workers`, `queue`, `wait_ms` and `busy_503` are new in 1.2.0.
A node must accept a `/health` answer without them.

## GET /stats

Counters for the node's Health page. The node shows the fields that it
knows and ignores the others.

## GET /healthz

    ok

## Compatibility rules

- A new server must answer an old node, which sends no priority header,
  as before.
- A new node must work with an old server, which ignores the header and
  never sends `503`.
- New fields in a JSON answer are allowed. A field is never removed or
  changed in type within one contract version.
