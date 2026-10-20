# Third-party boundaries

## Project source

voiceserver was the `tools/voiceserver` folder of Mutara until version 1.2.0,
as documented in the retained README and history. The project source is
licensed under GPL-3.0-or-later. This license choice does not relicense
third-party software, models, voices or generated content. Existing commit
authorship and history are preserved.

## Runtime and build dependencies

The dependencies are referenced in requirements.txt; their source and binary
packages are not vendored here. License information below identifies the
upstream projects and pinned-package metadata reviewed for this source
publication. It is not a complete notice bundle for a built image or every
platform-specific binary. Anyone redistributing dependencies or built images
must retain applicable copyright/license notices, review bundled components
and satisfy corresponding-source and other obligations where required.

| Component | Upstream license / boundary |
| --- | --- |
| [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) | MIT; separate from model weights |
| [phonemizer](https://github.com/bootphon/phonemizer) | GPL-3.0-or-later |
| [espeakng-loader](https://github.com/thewh1teagle/espeakng-loader) | MIT wrapper; bundled eSpeak NG is separately licensed |
| [eSpeak NG](https://github.com/espeak-ng/espeak-ng) | GPL-3.0-or-later; see upstream COPYING and component notices, including bundled data |
| [NumPy](https://github.com/numpy/numpy) | Pinned metadata: BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0; binary distributions may include additional notices |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | MIT; see upstream third-party notices for bundled components |
| [attrs](https://github.com/python-attrs/attrs) | MIT |
| [dlinfo](https://github.com/fphammerle/python-dlinfo) | MIT |
| [joblib](https://github.com/joblib/joblib) | BSD-3-Clause |
| [cloudpickle](https://github.com/cloudpipe/cloudpickle) | BSD-3-Clause |
| [typing_extensions](https://github.com/python/typing_extensions) | PSF-2.0 |
| [FlatBuffers](https://github.com/google/flatbuffers) | Apache-2.0 |
| [packaging](https://github.com/pypa/packaging) | Apache-2.0 OR BSD-2-Clause |
| [Protocol Buffers](https://github.com/protocolbuffers/protobuf) | BSD-3-Clause |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | MIT; its package includes the Silero VAD model (MIT) |
| [CTranslate2](https://github.com/OpenNMT/CTranslate2) | MIT; binary wheels include other libraries with their own notices |
| [PyAV](https://github.com/PyAV-Org/PyAV) | BSD-3-Clause; binary wheels include FFmpeg and codec libraries (such as libopus) under their own licenses |
| [tokenizers](https://github.com/huggingface/tokenizers) | Apache-2.0 |
| [huggingface_hub](https://github.com/huggingface/huggingface_hub) | Apache-2.0 |
| [hf-xet](https://github.com/huggingface/xet-core) | Apache-2.0 |
| [filelock](https://github.com/tox-dev/filelock) | MIT |
| [fsspec](https://github.com/fsspec/filesystem_spec) | BSD-3-Clause |
| [PyYAML](https://github.com/yaml/pyyaml) | MIT |
| [tqdm](https://github.com/tqdm/tqdm) | MPL-2.0 AND MIT |
| [click](https://github.com/pallets/click) | BSD-3-Clause |
| [HTTPX](https://github.com/encode/httpx) | BSD-3-Clause |
| [HTTP Core](https://github.com/encode/httpcore) | BSD-3-Clause |
| [h11](https://github.com/python-hyper/h11) | MIT |
| [AnyIO](https://github.com/agronholm/anyio) | MIT |
| [idna](https://github.com/kjd/idna) | BSD-3-Clause |
| [certifi](https://github.com/certifi/python-certifi) | MPL-2.0 |

The Dockerfile references a Python/Debian base image. Python, Debian and its
included packages retain their own licenses. This repository does not publish
that image or a built derivative. Local builds download and combine these
components; source-only publication does not remove downstream obligations.

## Models and voices

`kokoro-v1.0.onnx` and `voices-v1.0.bin` are not included in this repository.
The installer verifies user-supplied files; it does not download them.
The [kokoro-onnx model-files-v1.0 release](https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0)
is the documented download location. The upstream
[Kokoro-82M model card](https://huggingface.co/hexgrad/Kokoro-82M)
identifies Apache-2.0 for the model weights. Consult the exact downloaded
artifacts and upstream notices for applicable model/voice terms and provenance;
the project's GPL license does not cover them. No Allison audio is included.

The speech to text model is not included either. On its first start the
server downloads the faster-whisper model that `--stt-model` names (default
`small.en`, from [Systran/faster-whisper-small.en](https://huggingface.co/Systran/faster-whisper-small.en))
from Hugging Face into `--stt-dir`. It is a CTranslate2 conversion of
OpenAI's [Whisper](https://github.com/openai/whisper) model, whose code and
weights are MIT-licensed. Consult the model card of the exact model that you
use.

No models, voices, generated audio, images, compiled binaries or packages are
published by this repository. Hobby use is not an exception to license terms.
