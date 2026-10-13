# Mutara voice server. The model files are not in the image; mount a folder
# with kokoro-v1.0.onnx and voices-v1.0.bin at /models. See README.md, or
# use install.sh, which checks the host and the files first.
#
# The base image is pinned by version and digest (python:3.12.14-slim-trixie,
# 2026-09-25), and requirements.txt pins every package, so a build next
# month makes the same image.
FROM python:3.12.14-slim-trixie@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && pip check
COPY server.py .
RUN useradd --system --home /app mutara && mkdir /cache && chown mutara /cache
USER mutara
VOLUME ["/models", "/cache"]
EXPOSE 7374
# The check asks GET /health on the address that the server listens on.
# The first start loads the model, which takes some seconds.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD ["python", "server.py", "--check"]
# 0.0.0.0 inside the container; install.sh runs it with the host's network
# and a --listen on localhost or a private address (see README.md).
CMD ["python", "server.py", "--listen", "0.0.0.0:7374", "--models", "/models", "--cache", "/cache"]
