# voiceserver

An optional voice server for Mutara. It speaks announcements with
[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), a natural voice
that is too large for a Raspberry Pi (about 350 MB of model files and up to
1 GB of memory), so it runs on another machine. The default voice is Sarah
(`af_sarah`); any American (`af_*`, `am_*`) or British (`bf_*`, `bm_*`)
Kokoro voice works.

The node sends each announcement as text, such as "Node 2, 1, 0, 0,
connected to node 2, 2, 0, 0.", and plays the speech that comes back.
The server and the node both keep a copy, so a phrase is made only once;
the node's copies are in `/var/lib/mutara/voice-cache`. When the server
does not answer within `voice_server.timeout` (5 s), the node uses its
built-in Allison voice. The voice ID never waits: the node makes it ready
at start and uses Allison until it is.

Weather alerts need this server: Allison has a fixed list of words and
can't read an alert. An alert waits up to 30 s for the server; if the
server does not answer, the alert is not spoken and the dashboard's event
log says so.

## What it needs

Not a Raspberry Pi. The installer refuses a Raspberry Pi, ARM computers
and a host below these minimums, and it has no option to skip them:

- Linux on x86-64. Other CPU types, ARM servers too, are refused until
  we test them.
- 4 or more cores and 4 GB or more memory. AVX2 is not needed: on an
  8-core virtual machine without AVX, 3 s of speech takes about 2 s.
- Docker Engine, and about 1.5 GB of disk: the image (about 420 MB), the
  model (about 350 MB) and the cache (up to `--cache-mb`, 500 MB).

## Security

The server has no password. Run it where only your nodes can reach it: on
the node's own machine (`127.0.0.1`) or on your Tailscale or home network.
The installer refuses `0.0.0.0` and public addresses. Do not forward a
router port to it. Securing the host is the admin's job.

The container uses the host's network, so a host firewall such as ufw sees
it as an ordinary port (Docker's published ports go around ufw's input
rules). With ufw, let only Tailscale reach it:

    sudo ufw allow in on tailscale0 to any port 7374 proto tcp

## Install with Docker

The model is a separate download; this folder and the image do not contain
it. Get `kokoro-v1.0.onnx` and `voices-v1.0.bin` from the release
`model-files-v1.0` of [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx)
and put them in one folder.

Check the host, Docker, the model files (SHA-256) and the address. This
changes nothing:

    ./install.sh --check --models ~/mutara-voice/models --listen 100.64.0.5:7374

Then install: the same command without `--check`. The script builds the
image here, starts the container `mutara-voiceserver` with
`--restart unless-stopped`, host networking, the models read-only and the
cache in the volume `mutara-voice-cache`. It waits for the server, makes a
test sentence, and prints the address to set on the node. `--listen`
defaults to `127.0.0.1:7374`, for a node on the same machine.

To install a new version, run it again with `--replace`; the cache stays.
`./install.sh --help` lists the other options (container, image and volume
names, cache size).

## Run without Docker

Needs Python 3.12 (espeak-ng from Homebrew on macOS):

    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
    .venv/bin/python server.py --models models --listen 127.0.0.1:7374

## Set up the node

In Settings → Announcements, set **Voice server** to the server's address, such
as `http://100.64.0.5:7374`, and pick the voice. Or in the settings file:

```yaml
voice_server:
  url: http://100.64.0.5:7374
  voice: af_sarah   # the default
  speed: 0.8        # the default
  timeout: 5s       # the default
  # The voice ID as a sentence (node.id_mode voice or both). It must
  # contain the callsign. Write node numbers as single digits.
  id_text: "This is N0CALL, Mutara node 2 1 0 0"
```

The node makes each phrase once and keeps a copy, so the first ID after a
change is spelled by Allison while the server makes the new one.

## API

    GET  /v1/voices   {"voices": ["af_alloy", ...], "default": "af_sarah"}
    POST /v1/speech   {"text": "...", "voice": "af_sarah", "speed": 0.8}
                      -> audio/wav, 24 kHz mono 16-bit, speech at -20 dBFS RMS
    GET  /health      {"status": "ok", "version": "1.1.0", "model": ..., "voices": 28, "uptime_s": ...}
    GET  /stats       counters: requests, errors, in flight, waiting, last error,
                      speech time (last, avg, p50, p95, max of the last 200),
                      cache hits, misses, entries and bytes
    GET  /healthz     ok

A speech answer has `X-Cache: hit` or `miss`. The Docker health check runs
`python server.py --check`, which asks `/health`.

Text is up to 400 characters. One phrase is made at a time; made speech is
kept in the cache folder, up to `--cache-mb` (500 MB), oldest out first.

## On the dashboard

Settings → Health → Voice server (admins only) shows the node's side: the
requests, the round trip (last, average, 95%), timeouts and errors, the
node's copies, how often Allison spoke instead, the alerts not spoken and
the last error. Below it is the server's `/stats`, which the node asks at
most every 5 s. **Test voice** makes a test sentence and plays it in the
browser, not on the air.
