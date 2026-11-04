# ADR 0001: A queue of its own for transcriptions, and transcription jobs

- **Status:** Proposed
- **Date:** 2026-11-04
- **Deciders:** project owner
- **Related:** Mutara ADR 0009 (voice priority, the shared queue),
  Mutara ADR 0014 (clip bundles and transcripts), [API.md](../../API.md),
  `server.py` (`Speaker`, `Transcription`, `STT`), Mutara
  `internal/voice/transcribe.go` and `internal/bundle/stt.go`, Mutara
  Regula card u116

This is the first ADR of this repo. Mutara's ADRs 0009 and 0014 made the
voice server; this one changes how it makes transcriptions. It is a
design only.

## Context

### What the server does today (1.3.0)

- **One queue for both.** `Speaker` keeps a `now` and a `later` queue.
  Speech (`Job`) and transcriptions (`Transcription`) are in the same
  queues and use the same workers (`--workers`, 1 by default). A worker
  makes one step of a job in each turn: one sentence of a text, or one
  segment (up to 30 s of audio) of a transcription. So an announcement
  waits for one transcription step at most.
- **One request, one answer.** `POST /v1/transcribe` holds the HTTP
  request open until the text is made. The node waits up to 5 minutes
  for a `now` request (`nowTranscribeWait`); a `later` request has no
  limit of its own and can get `503` with `Retry-After`.
- **The model.** `small.en`, `int8`, beam 5, the VAD filter on,
  `condition_on_previous_text` off. `--stt-model` and `--stt-compute`
  choose another. The node's prompt (callsigns and the module's hint)
  goes to Whisper as `initial_prompt`.

### What the node uses it for

The node sends a transcription only when a user asks: **Transcribe** on
one clip (a `now` request), or the transcript of a bundle (`later`
requests, one clip at a time, `internal/bundle/stt.go`). No clip is
transcribed by itself. Nobody listens on the air for a transcript.

### The problems

1. **Accuracy.** `small.en` makes mistakes on radio speech. In a test on
   2026-11-04 (a 63.8 s kcwide clip, card u116) it wrote "anything goes
   in that" for "anything goes net" and "KCY" for "KC Wide". A larger
   model (`large-v3-turbo`, `distil-large-v3`, `large-v3`) is more
   accurate, but 3 to 10 times slower on a CPU. A 60 s clip with
   `large-v3` can take minutes.
2. **A slow step holds up speech.** With a larger model, one
   transcription step can take longer than the node's 5 s speech
   timeout. With one worker, an announcement that comes during that
   step misses the timeout and the node speaks with Allison. With a
   weather alert (30 s) it can be lost.
3. **A long request is fragile.** A request that is open for minutes
   breaks with a restart of the server, a network blip on the tailnet,
   or the node's own 5 minute limit. Then all the work is lost and
   starts again.
4. **The prompt helps only the first 30 s.** `initial_prompt` is used
   for the first window only when `condition_on_previous_text` is off.
   In the test the text after 30 s lost its capitals and stops, and
   "KZ4JON" became "kz for jon".

The owner (2026-11-04): transcriptions are used for bundles, so their
time does not matter. Announcements must not wait for them. The model
must stay a setting, so a small server can use a small model.

## Decision

### 1. Transcriptions get their own queue and their own worker

- The server keeps two parts:
  - **Speech:** the `now` and `later` queues of ADR 0009, as today,
    with `--workers` workers. Only speech jobs.
  - **Transcriptions:** a queue of its own and **one** worker of its
    own. It makes one clip at a time. It still reads the segments one
    by one, for the progress (decision 3) and so a cancel stops it, but
    speech no longer waits between them.
- In the transcription queue, a `now` job (one clip that a user asked
  for) goes before a `later` job (a bundle). Within a priority the
  order of arrival is kept. Promotion stays: a `now` request for a
  queued `later` job makes it `now`.
- `--later-limit` counts speech only. The transcription queue has its
  own limits (decision 3).

### 2. The transcription worker shares the CPU gently

Two queues do not make two CPUs. While Whisper runs on all cores,
Kokoro gets slower. So:

- **`--stt-threads N`** (`VOICESERVER_STT_THREADS`): the CPU threads of
  the Whisper model (CTranslate2 `cpu_threads`). Default: half of the
  cores, at least 1. With 8 cores Whisper uses 4, and Kokoro has the
  others when an announcement comes.
- The transcription thread runs at a lower OS priority (`nice` +10 on
  Linux, set on the thread with `os.setpriority` and its native ID;
  CTranslate2's own threads are started from it), so the kernel gives
  the speech threads the CPU first. To check in the test: do
  CTranslate2's threads keep the lower priority?
- **The test:** the time of a speech request while a long
  transcription runs. With `small.en` and with the chosen large model,
  on blue (8 cores), it must stay well under 5 s. This test decides the
  defaults; when it fails, the defaults change, not the design.

### 3. Transcription jobs: drop off the clip, pick up the text later

New endpoints, in contract version 1 (only new routes and fields):

```
POST   /v1/transcriptions        multipart/form-data, the same parts as
                                 POST /v1/transcribe; header
                                 X-Voice-Priority: now | later
  -> 202 {"id": "<job id>", "state": "queued", "position": 3}
  -> 400, 413, 501 as POST /v1/transcribe
  -> 503 + Retry-After when the transcription queue is full

GET    /v1/transcriptions/{id}   optional ?wait=N (seconds, at most 60)
  -> 200 {"id": ..., "state": "queued" | "working" | "done" | "failed",
          "position": 2,                       (queued)
          "progress_ms": 31000,                (working: audio done)
          "duration_ms": 63800,
          "result": { the answer of POST /v1/transcribe },   (done)
          "error": "..."}                      (failed)
  -> 404 no such job (unknown, expired, or the server restarted)

DELETE /v1/transcriptions/{id}   cancel a queued job, or drop a result
  -> 204
```

- **The job ID** is the job's key: the SHA-256 of the model, the
  language, the prompt and the audio (`Transcription.key_of`). So the
  same clip sent again gets the same job, queued, working or done, and
  a retry is always safe. It does not show the text or the audio.
- **`?wait=N`** holds the answer up to N seconds, until the state
  changes to `done` or `failed`. So the node asks seldom and still gets
  the text at once.
- **Polling, not a callback.** The server never calls the node: no new
  network path, no secret, no address to set. The node knows when it
  sent a clip.
- **Memory only.** The queue and the results are in memory. A result
  is kept for 1 hour after `done` or `failed`, or until a `DELETE`.
  After a restart every job is gone, and the node gets `404` and sends
  the clip again. So the server still keeps no audio and no text on
  disk, as README.md promises.
- **Limits:** at most 64 jobs in the queue, and at most 256 MB of
  queued audio (`--stt-queue-limit`, `--stt-queue-mb`). A new job over
  a limit gets `503` with `Retry-After`. A job sent again joins its
  job and never gets `503`.
- **`POST /v1/transcribe` stays**, with the same answer, for nodes that
  do not know jobs. It makes a job in the new queue and waits for it.
  Its `503` and `Retry-After` now come from the transcription queue.
- **Discovery.** `/health` gets `"transcribe": {"jobs": true, ...}`. A
  node that sees it uses the jobs; else it uses `POST /v1/transcribe`.

### 4. The hint for the whole clip, and the model as a setting

- **Hotwords.** The prompt also goes to Whisper as `hotwords`, which it
  uses in every 30 s window, not only the first. `initial_prompt`
  stays for the first window. The prompt rules (1000 characters) do not
  change. To check with card u116's clips: does the text after 30 s
  keep its callsigns, capitals and stops?
- **The model stays a setting** (`--stt-model`, `--stt-compute`), with
  `small.en` the default. README.md gets a table of the models: memory,
  disk, speed on a 4-core and an 8-core x86-64 computer, and accuracy
  on our sample clips. Rough numbers before the test (int8, Whisper
  only, one job at a time; Kokoro adds about 1 GB):

  | Model | Memory | Disk | Speed vs `small.en` |
  |---|---|---|---|
  | `tiny.en`, `base.en` | under 0.5 GB | 75-150 MB | 2-5x faster |
  | `small.en` (default) | about 1 GB | about 470 MB | 1x |
  | `medium.en` | about 1.5 GB | about 800 MB | 2-3x slower |
  | `large-v3-turbo`, `distil-large-v3` | 1.5-2 GB | about 800 MB | 3-5x slower |
  | `large-v3` | 2.5-3.5 GB | about 1.6 GB | 8-10x slower |

  The installer's memory check stays at 4 GB; the README says how much
  more a large model needs.

### 5. The node

The node's side is Mutara work (`internal/voice/transcribe.go`,
`internal/bundle/stt.go`, `internal/web/bundles.go`):

- When `/health` has `transcribe.jobs`, `Transcribe` posts the clip to
  `/v1/transcriptions` and then asks `GET ...?wait=30` until `done` or
  `failed`. On `404` it posts the clip again (the server restarted).
- No 5 minute limit for a job; the node gives up only when the job is
  `failed`, or when the server is down for longer than a limit (say
  30 minutes for a bundle).
- The single-clip **Transcribe** shows "Transcribing" as today, and
  "Waiting: 3 before" from `position` when it has to wait.

## Options not chosen

- **Keep one queue, and only make the steps smaller.** A step of a
  large model is still seconds long, and the speech worker still waits
  for it. It does not fix the long request either.
- **A second container (or server) for transcriptions.** Clean, but a
  second install, a second address on the node, and Kokoro and Whisper
  on one host compete for the CPU all the same. The threads limit and
  `nice` do most of the good in one process. A large install can still
  run two servers, one with `--stt-model off`, later.
- **A callback (webhook) to the node.** The server would need the
  node's address and a secret, and a way through the node's firewall.
  Polling with `?wait` is almost as fast and needs none of that.
- **Jobs on disk, so they live through a restart.** It breaks the "no
  audio and no text on disk" rule for little gain: the node sends the
  clip again.
- **Change `POST /v1/transcribe` itself to answer 202.** Old nodes
  would break. A new route keeps contract version 1.

## Consequences

- An announcement no longer waits for a transcription step. It still
  shares the CPU, so the test of decision 2 is part of the work.
- A large model becomes usable on blue. The owner chooses it after the
  benchmark; blue's container limit (3 GB today) may need 4 to 6 GB.
- `/health` and `/stats`: `queue`, `wait_ms`, `busy_503` and
  `--later-limit` count speech only. The `transcribe` object gets its
  own `queue`, `wait_ms`, `busy_503` and `jobs` fields. This is a
  change of meaning of fields that a 1.3.0 node shows, so API.md says
  it in the 1.4.0 notes.
- The work, in this order:
  1. voiceserver: separate queue and worker, `--stt-threads`, `nice`,
     hotwords, the model table (1.4.0).
  2. voiceserver: the job endpoints (1.4.0 or 1.5.0).
  3. A benchmark of the models on our sample clips, to choose blue's
     model and memory.
  4. Mutara: the node uses the jobs.
  5. Deploy to blue, with the owner's OK.
