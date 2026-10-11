# voiceserver

An optional voice server for Mutara. It speaks announcements with
[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), a natural voice
that is too large for a Raspberry Pi (about 350 MB of model files and up to
1 GB of memory), so it runs on another machine. The default voice is Sarah
(`af_sarah`); any American (`af_*`, `am_*`) or British (`bf_*`, `bm_*`)
Kokoro voice works.

The node sends each announcement as text, such as "Node 5, 5, 0, 4, 6, 0,
connected to node 5, 9, 4, 9, 5, 0.", and plays the speech that comes back.
The server and the node both keep a copy, so a phrase is made only once;
the node's copies are in `/var/lib/mutara/voice-cache`. When the server
does not answer within `voice_server.timeout` (5 s), the node uses its
built-in Allison voice. The voice ID never waits: the node makes it ready
at start and uses Allison until it is.

Weather alerts need this server: Allison has a fixed list of words and
can't read an alert. An alert waits up to 30 s for the server; if the
server does not answer, the alert is not spoken and the dashboard's event
log says so.

## Security

The server has no password. Run it where only your nodes can reach it: on
the node's own machine (`127.0.0.1`) or on your Tailscale network. Do not
publish it on a public address or forward a port to it.

## Run with Docker

Get the model files (about 350 MB) into a folder:

    mkdir models && cd models
    curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
    curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin

Build and start it, listening on this machine's Tailscale address only.
It uses the host's network, so a host firewall such as ufw sees it as an
ordinary port (Docker's published ports go around ufw's input rules):

    docker build -t mutara-voiceserver tools/voiceserver
    docker run -d --name mutara-voiceserver --restart unless-stopped \
      --network host \
      -v "$PWD/models:/models:ro" -v mutara-voice-cache:/cache \
      mutara-voiceserver python server.py --listen "$(tailscale ip -4):7374" \
        --models /models --cache /cache

With ufw, let Tailscale reach it:

    sudo ufw allow in on tailscale0 to any port 7374 proto tcp

For a node on the same machine, use `--listen 127.0.0.1:7374` instead.

## Run without Docker

Needs Python 3.12 (espeak-ng from Homebrew on macOS):

    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
    .venv/bin/python server.py --models models --listen 127.0.0.1:7374

## Set up the node

In Settings → Station, set **Voice server** to the server's address, such
as `http://100.64.0.5:7374`, and pick the voice. Or in the settings file:

```yaml
voice_server:
  url: http://100.64.0.5:7374
  voice: af_sarah   # the default
  speed: 0.8        # the default
  timeout: 5s       # the default
  # The voice ID as a sentence (node.id_mode voice or both). It must
  # contain the callsign. Write node numbers as single digits.
  id_text: "This is N0CALL, Mutara node 1 9 9 9"
```

The node makes each phrase once and keeps a copy, so the first ID after a
change is spelled by Allison while the server makes the new one.

## API

    GET  /v1/voices   {"voices": ["af_alloy", ...], "default": "af_sarah"}
    POST /v1/speech   {"text": "...", "voice": "af_sarah", "speed": 0.8}
                      -> audio/wav, 24 kHz mono 16-bit, speech at -20 dBFS RMS
    GET  /healthz     ok

Text is up to 400 characters. One phrase is made at a time; made speech is
kept in the cache folder, up to `--cache-mb` (500 MB), oldest out first.
