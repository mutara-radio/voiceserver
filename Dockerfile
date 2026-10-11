# Mutara voice server. The model files are not in the image; mount a folder
# with kokoro-v1.0.onnx and voices-v1.0.bin at /models. See README.md.
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY server.py .
RUN useradd --system --home /app mutara && mkdir /cache && chown mutara /cache
USER mutara
VOLUME ["/models", "/cache"]
EXPOSE 7374
# 0.0.0.0 inside the container; publish the port on localhost or a
# Tailscale address only (see README.md).
CMD ["python", "server.py", "--listen", "0.0.0.0:7374", "--models", "/models", "--cache", "/cache"]
