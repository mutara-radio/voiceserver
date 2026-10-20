# Voice server API

This is the contract between the voice server and a Mutara node. The
node's side is `internal/voice/remote.go` in the Mutara repo. A change
here that an old node or an old server cannot use is a new major
version of this contract.

- **Contract version:** 1
- **Server version that this text agrees with:** 1.3.0

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

## POST /v1/transcribe

Makes the text of a recorded clip (speech to text). New in 1.3.0. The
server uses faster-whisper; the model is `small.en` by default (see
README.md, "Speech to text").

Request body, `multipart/form-data`:

| Part       | Required | Value |
|------------|----------|-------|
| `file`     | yes | The clip as recorded: Ogg Opus (`audio/ogg; codecs=opus`), 48 kHz mono. At most 8 MB (8388608 bytes). The part's own content type is not checked; the bytes are. |
| `language` | no  | ISO 639-1, two letters, such as `en`. Empty: the model finds the language. An English-only model (`*.en`) always answers `en`. |
| `prompt`   | no  | A hint for the model, at most 1000 characters, such as `Amateur radio net. Callsigns: W1AW, K1ABC.` The server gives it to Whisper as `initial_prompt`. |
| `id`       | no  | The clip ID. The server writes it in its log only. |

The body can have a `Content-Length` or be chunked. The whole body is
at most 8 MB + 64 KB.

Request header, optional: `X-Voice-Priority: now | later`, with the same
meaning as for `/v1/speech`. No authentication, as `/v1/speech`.

Answer `200`, `application/json`:

    {"text": "W1AW, this is K1ABC, checking in with no traffic.",
     "language": "en",
     "duration_ms": 5021,
     "model": "small.en",
     "segments": [
       {"start_ms": 0, "end_ms": 4840,
        "text": "W1AW, this is K1ABC, checking in with no traffic.",
        "confidence": 0.83}]}

- `text`: the text of all segments, joined with one space. An empty
  `text` with no segments is a valid answer: the clip has no speech,
  for example a carrier, a tone or noise.
- `duration_ms`: the length of the clip.
- `start_ms`, `end_ms`: the time of a segment in the clip.
- `confidence`: 0 to 1, with 2 decimals. It is `e ^ avg_logprob`, where
  `avg_logprob` is Whisper's average log probability of the segment's
  tokens (0 or less). So it is the geometric mean of the token
  probabilities: 1.0 is sure, 0.5 is an average token probability of
  one half. Below about 0.4, look at the line again.
- `model`: the model's name (the last part of a folder name).

The server does not keep the audio or the text. Its log has the `id`,
the length and the number of segments, not the text.

Answers that are not `200` have the body `{"error": "..."}`:

- `400`: the request is not correct: not `multipart/form-data`, no
  `file`, a file that is not Ogg Opus or can't be decoded, a bad
  `language`, a `prompt` that is too long.
- `413`: the file or the body is too large.
- `501`: speech to text is off on this server (`--stt-model off`), or
  its model failed to load. `/health` says which.
- `503` with `Retry-After` (seconds): the `later` queue is full, as for
  `/v1/speech`. Only a `later` request gets this answer for this
  reason. A `later` request also gets `503` with `Retry-After: 30`
  while the model loads after a start; a `now` request waits for the
  model (at most 120 s).
- `500`: the transcription failed.
- `404`: an old server (before 1.3.0) has no such endpoint.

Two requests for the same file, language and prompt at the same time
become one job, as for speech. A `now` request promotes a queued `later`
one. The server keeps no cache of transcripts.

### The queue

Speech and transcriptions share one queue and one CPU (ADR 0009):

- A `now` request goes before a `later` request, for both kinds.
- The server makes a transcription one step at a time: the first step
  decodes the audio and finds the speech, and each next step makes one
  segment (up to 30 s of audio). Between two steps, waiting speech of
  the same or a higher priority goes first. So a speech request waits
  for one step at most. For `small.en` on a 4-core CPU, a step can take
  some seconds.
- Speech goes before a transcription of the same priority; between
  transcriptions, the order of arrival is kept.
- `--later-limit` counts speech and transcriptions together.
- `queue`, `wait_ms` and `busy_503` in `/health` count both kinds.
- With `--workers 2` or more, a transcription and speech can run at the
  same time, as two texts can.

## GET /v1/voices

    {"voices": ["af_alloy", ...], "default": "af_sarah"}

## GET /health

    {"status": "ok", "version": "1.3.0", "model": "kokoro-v1.0", "voices": 28,
     "uptime_s": 1234, "workers": 1, "queue": {"now": 0, "later": 3},
     "wait_ms": {"now": 120, "later": 4100}, "busy_503": 0,
     "transcribe": {"model": "small.en", "ready": true, "state": "ready"}}

The fields `workers`, `queue`, `wait_ms` and `busy_503` are new in 1.2.0.
`transcribe` is new in 1.3.0. A node must accept a `/health` answer
without them.

`transcribe` tells the node if it can send `POST /v1/transcribe`:

- `ready`: `true` when the model is loaded. Send transcriptions only
  then.
- `state`: `ready`, `loading` (after a start; the first start downloads
  the model), `failed` (the model did not load; speech works on) or
  `off` (`--stt-model off`).
- `model`: the model's name, or `null` when the state is `off`.
- `error`: only when the state is `failed`; why it failed.

So a node finds speech to text this way:

1. No `transcribe` field: an old server, with no speech to text.
2. `transcribe.ready` is `false`: not now. Ask again later.
3. `transcribe.ready` is `true`: send transcriptions.

A `404` or `501` on `POST /v1/transcribe` also means "no speech to text"
until the next check. `GET /stats` has the same `transcribe` fields.

## GET /stats

Counters for the node's Health page. The node shows the fields that it
knows and ignores the others. The speech counters (`requests`, `ok`,
`errors` and the others) count `/v1/speech` only. New in 1.3.0:

    "transcribe": {"model": "small.en", "ready": true, "state": "ready",
                   "requests": 8, "ok": 8, "bad_requests": 0, "errors": 0,
                   "in_flight": 0, "audio_s": 149.8, "work_s": 54.8,
                   "last_ms": 23651}

`audio_s` is the clip time and `work_s` the time to make the text, both
since start. `last_ms` is the time of the last transcription.

## GET /healthz

    ok

## Compatibility rules

- A new server must answer an old node, which sends no priority header,
  as before.
- A new node must work with an old server, which ignores the header and
  never sends `503`.
- A new node must work with a server that has no speech to text: an old
  server (no `transcribe` in `/health`, `404`), or one with the model
  off or not loaded (`ready` is `false`, `501`).
- New fields in a JSON answer are allowed. A field is never removed or
  changed in type within one contract version.
