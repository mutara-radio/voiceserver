#!/usr/bin/env bash
# Installs the Mutara voice server in Docker on this computer.
#
#   ./install.sh --models DIR [--listen ADDR:PORT] [--check] [--replace]
#
# It checks that this computer is strong enough, that Docker works, that
# the model files are in DIR and that the address is a safe one. Then it
# builds the image from this folder and starts the container, which starts
# again after a reboot. With --check it only does the checks and changes
# nothing.
#
# The script does not download the model. Get the two files from the
# kokoro-onnx project (see README.md) and put them in DIR.
#
# The server has no password. It listens only on localhost or on a private
# address, such as your Tailscale address. The script refuses any other
# address.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)

models=""
listen="127.0.0.1:7374"
name="mutara-voiceserver"
image="mutara-voiceserver"
volume="mutara-voice-cache"
cache_mb=500
check_only=0
replace=0

# The host minimums. Kokoro needs about 1 GB of memory while it speaks,
# and one sentence keeps all the cores busy for some seconds. With fewer
# cores or less memory, an alert can take longer than the 30 s that the
# node waits for it.
min_cores=4
min_mem_kb=3500000 # a "4 GB" computer shows about 3.7 to 3.9 GB

# model_sha gives the SHA-256 of a kokoro-onnx v1.0 model file
# (model-files-v1.0).
model_sha() {
  case "$1" in
    kokoro-v1.0.onnx) echo 7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5 ;;
    voices-v1.0.bin) echo bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d ;;
  esac
}
model_page="https://github.com/thewh1teagle/kokoro-onnx"

usage() {
  cat <<EOF
Usage: $0 --models DIR [options]

  --models DIR        the folder with kokoro-v1.0.onnx and voices-v1.0.bin
  --listen ADDR:PORT  where the server listens (default $listen); use
                      127.0.0.1 for a node on this computer, or this
                      computer's Tailscale or other private address
  --name NAME         the container name (default $name)
  --image NAME        the image name (default $image)
  --cache-volume VOL  the Docker volume for the server's cache (default $volume)
  --cache-mb N        the most MB of speech that the server keeps (default $cache_mb)
  --replace           replace a container with the same name (an upgrade)
  --check             do the checks only, and change nothing
EOF
}

die() { printf 'Error: %s\n' "$*" >&2; exit 1; }
ok() { printf '  ok: %s\n' "$*"; }
note() { printf '  note: %s\n' "$*"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --models) models=${2:?}; shift 2 ;;
    --listen) listen=${2:?}; shift 2 ;;
    --name) name=${2:?}; shift 2 ;;
    --image) image=${2:?}; shift 2 ;;
    --cache-volume) volume=${2:?}; shift 2 ;;
    --cache-mb) cache_mb=${2:?}; shift 2 ;;
    --replace) replace=1; shift ;;
    --check) check_only=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown option $1" ;;
  esac
done

# check_host refuses a Raspberry Pi, other small boards and weak
# computers. There is no option to skip it: on such a computer the voice
# server is too slow for the node to use.
check_host() {
  echo "Checking this computer"
  [ "$(uname -s)" = Linux ] || die "the installer is for Linux. On macOS, run server.py by hand (see README.md)."
  local model=""
  for f in /proc/device-tree/model /sys/firmware/devicetree/base/model; do
    if [ -r "$f" ]; then model=$(tr -d '\0' <"$f"); break; fi
  done
  case "$model" in
    *"Raspberry Pi"*) die "this is a $model. A Raspberry Pi is too slow for the voice server. Install it on a PC or server, and set its address in the node's settings." ;;
  esac
  local arch
  arch=$(uname -m)
  case "$arch" in
    x86_64) ok "CPU type $arch" ;;
    aarch64|arm64)
      # A 64-bit ARM board (it has a device tree) is too slow; an ARM
      # server is not.
      [ -z "$model" ] || die "this is a $model ($arch). Small ARM boards are too slow for the voice server. Install it on a PC or server."
      ok "CPU type $arch"
      note "the voice server is tested on x86-64 Linux; this CPU type is not tested"
      ;;
    *) die "the CPU type $arch is not supported. The voice server needs a 64-bit PC or server (x86-64)." ;;
  esac
  local cores mem_kb
  cores=$(nproc)
  [ "$cores" -ge "$min_cores" ] || die "this computer has $cores CPU cores. The voice server needs at least $min_cores."
  ok "$cores CPU cores"
  mem_kb=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
  [ "$mem_kb" -ge "$min_mem_kb" ] || die "this computer has $((mem_kb / 1024)) MB of memory. The voice server needs at least 4 GB."
  ok "$((mem_kb / 1024)) MB of memory"
}

check_docker() {
  echo "Checking Docker"
  command -v docker >/dev/null || die "Docker is not installed. Install Docker Engine (https://docs.docker.com/engine/install/), then run this again."
  docker info >/dev/null 2>&1 || die "Docker does not answer. Start it, or add your user to the docker group, then run this again."
  ok "Docker $(docker version --format '{{.Server.Version}}' 2>/dev/null || echo answers)"
}

check_models() {
  echo "Checking the model files"
  [ -n "$models" ] || die "give the model folder with --models DIR. The model is a separate download: get kokoro-v1.0.onnx and voices-v1.0.bin from $model_page (the model-files-v1.0 release) and put them in one folder."
  [ -d "$models" ] || die "the folder $models does not exist."
  models=$(cd "$models" && pwd)
  local missing=() f sum
  for f in kokoro-v1.0.onnx voices-v1.0.bin; do
    [ -f "$models/$f" ] || missing+=("$f")
  done
  [ ${#missing[@]} -eq 0 ] || die "$models does not have ${missing[*]}. Get the model files from $model_page (the model-files-v1.0 release) and put them in $models."
  for f in kokoro-v1.0.onnx voices-v1.0.bin; do
    sum=$(sha256sum "$models/$f" | cut -d' ' -f1)
    [ "$sum" = "$(model_sha "$f")" ] || die "$models/$f is not the v1.0 file (its SHA-256 is $sum, not $(model_sha "$f")). Download it again from $model_page."
    ok "$f"
  done
}

# check_listen accepts localhost and the private IPv4 ranges only: 10/8,
# 172.16/12, 192.168/16 and 100.64/10 (Tailscale). The server has no
# password, so it must never listen on a public address or on all
# addresses (0.0.0.0).
check_listen() {
  echo "Checking the address"
  local host=${listen%:*} port=${listen##*:}
  [ "$host" != "$listen" ] || die "--listen must be ADDR:PORT, such as 127.0.0.1:7374."
  [[ "$port" =~ ^[0-9]+$ ]] && [ "$port" -ge 1 ] && [ "$port" -le 65535 ] || die "the port $port is not a port number."
  [[ "$host" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)\.([0-9]+)$ ]] || die "the address $host must be an IPv4 address, such as 127.0.0.1 or your Tailscale address."
  local a=${BASH_REMATCH[1]} b=${BASH_REMATCH[2]} x
  for x in "${BASH_REMATCH[@]:1}"; do [ "$x" -le 255 ] || die "$host is not an IPv4 address."; done
  local kind=""
  if [ "$a" -eq 127 ]; then kind=localhost
  elif [ "$a" -eq 10 ] || { [ "$a" -eq 172 ] && [ "$b" -ge 16 ] && [ "$b" -le 31 ]; } || { [ "$a" -eq 192 ] && [ "$b" -eq 168 ]; }; then kind="a private address"
  elif [ "$a" -eq 100 ] && [ "$b" -ge 64 ] && [ "$b" -le 127 ]; then kind="a Tailscale address"
  fi
  [ -n "$kind" ] || die "$host is not localhost or a private address. The voice server has no password, so it must not listen on a public address or on all addresses (0.0.0.0). Use 127.0.0.1, or this computer's Tailscale or other private address."
  if [ "$kind" != localhost ]; then
    ip -o -4 addr show 2>/dev/null | grep -qw "inet $host" || die "this computer does not have the address $host."
  fi
  if [ "$replace" -eq 0 ] && ss -ltnH 2>/dev/null | awk '{print $4}' | grep -qE "^($host|0\.0\.0\.0|\*):$port$"; then
    die "something on this computer already listens on port $port. Choose a different port."
  fi
  ok "$listen is $kind"
}

check_host
check_docker
check_models
check_listen

if docker container inspect "$name" >/dev/null 2>&1 && [ "$replace" -eq 0 ]; then
  die "a container named $name already exists. To upgrade it, run this again with --replace."
fi

if [ "$check_only" -eq 1 ]; then
  echo "All checks passed. Nothing was changed (--check)."
  exit 0
fi

echo "Building the image $image"
docker build -t "$image" "$here"

if docker container inspect "$name" >/dev/null 2>&1; then
  echo "Removing the old container $name"
  docker rm -f "$name" >/dev/null
fi

echo "Starting the container $name"
# The host's network: the server listens only on the address given, and
# a host firewall such as ufw sees the port as an ordinary one (Docker's
# published ports go around ufw's input rules).
docker run -d --name "$name" --restart unless-stopped --network host \
  -v "$models:/models:ro" -v "$volume:/cache" \
  "$image" python server.py --listen "$listen" --models /models --cache /cache --cache-mb "$cache_mb" >/dev/null

echo "Waiting for the server to load the model"
for _ in $(seq 60); do
  if docker exec "$name" python server.py --check 2>/dev/null; then break; fi
  [ "$(docker inspect -f '{{.State.Running}}' "$name")" = true ] || { docker logs --tail 20 "$name" >&2; die "the container stopped. Its log is above."; }
  sleep 2
done
docker exec "$name" python server.py --check || { docker logs --tail 20 "$name" >&2; die "the server does not answer. Its log is above."; }
ok "the server answers"

echo "Making a test sentence"
docker exec -i "$name" python - "$listen" <<'EOF'
import json, sys, time, urllib.request
host, _, port = sys.argv[1].rpartition(":")
req = urllib.request.Request(f"http://{host}:{port}/v1/speech", json.dumps({"text": "This is a test of the Mutara voice server."}).encode(), {"Content-Type": "application/json"})
t = time.monotonic()
with urllib.request.urlopen(req, timeout=120) as r:
    n = len(r.read())
    cache = r.headers.get("X-Cache")
s = time.monotonic() - t
print(f"  ok: {n / 48000:.1f} s of speech in {s:.1f} s{' (from the cache)' if cache == 'hit' else ''}")
if cache != "hit" and s > 10:
    print("  note: this computer is slow for the voice server. The node waits 5 s for most announcements and 30 s for weather alerts.")
EOF

host=${listen%:*}
echo
echo "The voice server is ready. In Mutara, open Settings, then Announcements,"
echo "and set Voice server to:"
echo
echo "    http://$listen"
echo
if [ "${host%%.*}" = 127 ]; then
  echo "This address works only for a node on this computer."
else
  echo "Only computers on your private network or Tailscale can use this address."
  echo "If this computer has a firewall, let your node reach TCP port ${listen##*:},"
  echo "for example with ufw and Tailscale:"
  echo
  echo "    sudo ufw allow in on tailscale0 to any port ${listen##*:} proto tcp"
  echo
  echo "Do not forward a router port to the voice server."
fi
