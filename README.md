[CI](https://github.com/SYSTRAN/faster-whisper/actions?query=workflow%3ACI) [PyPI version](https://badge.fury.io/py/faster-whisper)

# 使用 CTranslate2 的 Faster Whisper 转录

**faster-whisper** 是对 OpenAI Whisper 模型的一个基于 [CTranslate2](https://github.com/OpenNMT/CTranslate2/) 的重新实现。CTranslate2 是一个面向 Transformer 模型的高性能推理引擎。

在保持相同精度的前提下，这个实现相比 [openai/whisper](https://github.com/openai/whisper) 最多可快 4 倍，同时占用更少内存。在 CPU 和 GPU 上启用 8 位量化后，效率还可以进一步提升。

## 基准测试

### Whisper

作为参考，下面展示了使用不同实现转录 **[13 分钟](https://www.youtube.com/watch?v=0u7tTptBo9I)** 音频时所需的时间和内存占用：

- [openai/whisper](https://github.com/openai/whisper)@[v20240930](https://github.com/openai/whisper/tree/v20240930)
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp)@[v1.7.2](https://github.com/ggerganov/whisper.cpp/tree/v1.7.2)
- [transformers](https://github.com/huggingface/transformers)@[v4.46.3](https://github.com/huggingface/transformers/tree/v4.46.3)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)@[v1.1.0](https://github.com/SYSTRAN/faster-whisper/tree/v1.1.0)

### Large-v2 model on GPU


| Implementation                  | Precision | Beam size | Time  | VRAM Usage |
| ------------------------------- | --------- | --------- | ----- | ---------- |
| openai/whisper                  | fp16      | 5         | 2m23s | 4708MB     |
| whisper.cpp (Flash Attention)   | fp16      | 5         | 1m05s | 4127MB     |
| transformers (SDPA)[^1]         | fp16      | 5         | 1m52s | 4960MB     |
| faster-whisper                  | fp16      | 5         | 1m03s | 4525MB     |
| faster-whisper (`batch_size=8`) | fp16      | 5         | 17s   | 6090MB     |
| faster-whisper                  | int8      | 5         | 59s   | 2926MB     |
| faster-whisper (`batch_size=8`) | int8      | 5         | 16s   | 4500MB     |


### distil-whisper-large-v3 model on GPU


| Implementation                        | Precision | Beam size | Time   | YT Commons WER |
| ------------------------------------- | --------- | --------- | ------ | -------------- |
| transformers (SDPA) (`batch_size=16`) | fp16      | 5         | 46m12s | 14.801         |
| faster-whisper (`batch_size=16`)      | fp16      | 5         | 25m50s | 13.527         |


*GPU 基准测试在 NVIDIA RTX 3070 Ti 8GB 上使用 CUDA 12.4 执行。*
[^1]: transformers OOM for any batch size > 1

### Small model on CPU


| Implementation                  | Precision | Beam size | Time  | RAM Usage |
| ------------------------------- | --------- | --------- | ----- | --------- |
| openai/whisper                  | fp32      | 5         | 6m58s | 2335MB    |
| whisper.cpp                     | fp32      | 5         | 2m05s | 1049MB    |
| whisper.cpp (OpenVINO)          | fp32      | 5         | 1m45s | 1642MB    |
| faster-whisper                  | fp32      | 5         | 2m37s | 2257MB    |
| faster-whisper (`batch_size=8`) | fp32      | 5         | 1m06s | 4230MB    |
| faster-whisper                  | int8      | 5         | 1m42s | 1477MB    |
| faster-whisper (`batch_size=8`) | int8      | 5         | 51s   | 3608MB    |


*在 Intel Core i7-12700K 上使用 8 个线程执行。*

## 环境要求

- Python 3.9 或更高版本

与 openai-whisper 不同，系统中**不需要**安装 FFmpeg。音频解码由 Python 库 [PyAV](https://github.com/PyAV-Org/PyAV) 完成，该库已在其发行包中内置 FFmpeg 相关库。

### GPU

使用 GPU 运行时，需要安装以下 NVIDIA 库：

- [cuBLAS for CUDA 12](https://developer.nvidia.com/cublas)
- [cuDNN 9 for CUDA 12](https://developer.nvidia.com/cudnn)

**注意**：最新版 `ctranslate2` 仅支持 CUDA 12 和 cuDNN 9。对于 CUDA 11 和 cuDNN 8，目前的解决方法是将 `ctranslate2` 降级到 `3.24.0`；对于 CUDA 12 和 cuDNN 8，则降级到 `4.4.0`。（可通过 `pip install --force-reinstall ctranslate2==4.4.0` 或在 `requirements.txt` 中指定版本来完成。）

安装上述 NVIDIA 库有多种方式。推荐方式见 NVIDIA 官方文档，下面也给出了一些可选安装方法。

其他安装方式（点击展开）

**注意：** 对于下面所有方法，请留意上文关于 CUDA 版本的说明。根据你的环境，你可能需要安装与下述 CUDA 12 库对应的 *CUDA 11* 版本库。

#### Use Docker

这些库（cuBLAS、cuDNN）已经包含在官方 NVIDIA CUDA Docker 镜像中：`nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04`。

#### Install with `pip` (Linux only)

在 Linux 上可以通过 `pip` 安装这些库。注意，启动 Python 前必须先设置 `LD_LIBRARY_PATH`。

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*

export LD_LIBRARY_PATH=`python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))'`
```

#### Download the libraries from Purfview's repository (Windows & Linux)

Purfview 的 [whisper-standalone-win](https://github.com/Purfview/whisper-standalone-win) 提供了 Windows 和 Linux 所需的 NVIDIA 库，并打包在一个[单独归档文件](https://github.com/Purfview/whisper-standalone-win/releases/tag/libs)中。解压后，将这些库放到 `PATH` 包含的目录中即可。



## 安装

可以通过 [PyPI](https://pypi.org/project/faster-whisper/) 安装该模块：

```bash
pip install faster-whisper
```

其他安装方式（点击展开）

### Install the master branch

```bash
pip install --force-reinstall "faster-whisper @ https://github.com/SYSTRAN/faster-whisper/archive/refs/heads/master.tar.gz"
```

### Install a specific commit

```bash
pip install --force-reinstall "faster-whisper @ https://github.com/SYSTRAN/faster-whisper/archive/a4f1cc8f11433e454c3934442b5e1a4ed5e865c3.tar.gz"
```



## 使用方法

### Faster-whisper

```python
from faster_whisper import WhisperModel

model_size = "large-v3"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

segments, info = model.transcribe("audio.mp3", beam_size=5)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
```

**警告：** `segments` 是一个 *generator*，因此只有在你遍历它时才会真正开始转录。你可以通过将其收集到列表中，或使用 `for` 循环来完整执行转录：

```python
segments, _ = model.transcribe("audio.mp3")
segments = list(segments)  # The transcription will actually run here.
```

### Batched Transcription

下面的代码片段演示了如何对示例音频文件执行批量转录。`BatchedInferencePipeline.transcribe` 可以直接替代 `WhisperModel.transcribe`。

```python
from faster_whisper import WhisperModel, BatchedInferencePipeline

model = WhisperModel("turbo", device="cuda", compute_type="float16")
batched_model = BatchedInferencePipeline(model=model)
segments, info = batched_model.transcribe("audio.mp3", batch_size=16)

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
```

### Faster Distil-Whisper

Distil-Whisper 的检查点与 Faster-Whisper 包兼容。特别是最新的 [distil-large-v3](https://huggingface.co/distil-whisper/distil-large-v3)
检查点本身就是为 Faster-Whisper 的转录算法设计的。下面的代码片段演示了如何在指定音频文件上使用 distil-large-v3 进行推理：

```python
from faster_whisper import WhisperModel

model_size = "distil-large-v3"

model = WhisperModel(model_size, device="cuda", compute_type="float16")
segments, info = model.transcribe("audio.mp3", beam_size=5, language="en", condition_on_previous_text=False)

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
```

关于 distil-large-v3 模型的更多信息，请参阅其原始 [model card](https://huggingface.co/distil-whisper/distil-large-v3)。

### Word-level timestamps

```python
segments, _ = model.transcribe("audio.mp3", word_timestamps=True)

for segment in segments:
    for word in segment.words:
        print("[%.2fs -> %.2fs] %s" % (word.start, word.end, word.word))
```

### VAD filter

该库集成了 [Silero VAD](https://github.com/snakers4/silero-vad) 模型，用于过滤音频中不含语音的片段：

```python
segments, _ = model.transcribe("audio.mp3", vad_filter=True)
```

默认行为较为保守，只会移除超过 2 秒的静音片段。可用的 VAD 参数及默认值见[源码](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/vad.py)。你也可以通过字典参数 `vad_parameters` 自定义它们：

```python
segments, _ = model.transcribe(
    "audio.mp3",
    vad_filter=True,
    vad_parameters=dict(min_silence_duration_ms=500),
)
```

批量转录默认启用 VAD 过滤。

### Logging

可以这样配置该库的日志级别：

```python
import logging

logging.basicConfig()
logging.getLogger("faster_whisper").setLevel(logging.DEBUG)
```

### Going further

更多模型和转录选项可查看 `[WhisperModel](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py)` 类的实现。

## 社区集成

下面列出了一些使用 faster-whisper 的开源项目，列表并不完整。欢迎把你的项目也加进来！

- [speaches](https://github.com/speaches-ai/speaches) is an OpenAI compatible server using `faster-whisper`. It's easily deployable with Docker, works with OpenAI SDKs/CLI, supports streaming, and live transcription.
- [WhisperX](https://github.com/m-bain/whisperX) is an award-winning Python library that offers speaker diarization and accurate word-level timestamps using wav2vec2 alignment
- [whisper-ctranslate2](https://github.com/Softcatala/whisper-ctranslate2) is a command line client based on faster-whisper and compatible with the original client from openai/whisper.
- [whisper-diarize](https://github.com/MahmoudAshraf97/whisper-diarization) is a speaker diarization tool that is based on faster-whisper and NVIDIA NeMo.
- [whisper-standalone-win](https://github.com/Purfview/whisper-standalone-win) Standalone CLI executables of faster-whisper for Windows, Linux & macOS. 
- [asr-sd-pipeline](https://github.com/hedrergudene/asr-sd-pipeline) provides a scalable, modular, end to end multi-speaker speech to text solution implemented using AzureML pipelines.
- [Open-Lyrics](https://github.com/zh-plus/Open-Lyrics) is a Python library that transcribes voice files using faster-whisper, and translates/polishes the resulting text into `.lrc` files in the desired language using OpenAI-GPT.
- [wscribe](https://github.com/geekodour/wscribe) is a flexible transcript generation tool supporting faster-whisper, it can export word level transcript and the exported transcript then can be edited with [wscribe-editor](https://github.com/geekodour/wscribe-editor)
- [aTrain](https://github.com/BANDAS-Center/aTrain) is a graphical user interface implementation of faster-whisper developed at the BANDAS-Center at the University of Graz for transcription and diarization in Windows ([Windows Store App](https://apps.microsoft.com/detail/atrain/9N15Q44SZNS2)) and Linux.
- [Whisper-Streaming](https://github.com/ufal/whisper_streaming) implements real-time mode for offline Whisper-like speech-to-text models with faster-whisper as the most recommended back-end. It implements a streaming policy with self-adaptive latency based on the actual source complexity, and demonstrates the state of the art.
- [WhisperLive](https://github.com/collabora/WhisperLive) is a nearly-live implementation of OpenAI's Whisper which uses faster-whisper as the backend to transcribe audio in real-time.
- [Faster-Whisper-Transcriber](https://github.com/BBC-Esq/ctranslate2-faster-whisper-transcriber) is a simple but reliable voice transcriber that provides a user-friendly interface.
- [Open-dubbing](https://github.com/softcatala/open-dubbing) is open dubbing is an AI dubbing system which uses machine learning models to automatically translate and synchronize audio dialogue into different languages.
- [Whisper-FastAPI](https://github.com/heimoshuiyu/whisper-fastapi) whisper-fastapi is a very simple script that provides an API backend compatible with OpenAI, HomeAssistant, and Konele (Android voice typing) formats.

## 模型转换

当你通过模型大小加载模型，例如 `WhisperModel("large-v3")` 时，对应的 CTranslate2 模型会自动从 [Hugging Face Hub](https://huggingface.co/Systran) 下载。

我们也提供了脚本，用于转换任何与 Transformers 库兼容的 Whisper 模型。这些模型既可以是 OpenAI 官方原始模型，也可以是用户自行微调后的模型。

例如，下面的命令会将[原始 "large-v3" Whisper 模型](https://huggingface.co/openai/whisper-large-v3) 转换并以 FP16 格式保存权重：

```bash
pip install transformers[torch]>=4.23

ct2-transformers-converter --model openai/whisper-large-v3 --output_dir whisper-large-v3-ct2
--copy_files tokenizer.json preprocessor_config.json --quantization float16
```

- `--model` 选项接受 Hub 上的模型名称，或本地模型目录路径。
- 如果不使用 `--copy_files tokenizer.json` 选项，则在后续加载模型时会自动下载 tokenizer 配置。

模型也可以通过代码进行转换。参见 [conversion API](https://opennmt.net/CTranslate2/python/ctranslate2.converters.TransformersConverter.html)。

### Load a converted model

1. 直接从本地目录加载模型：

```python
model = faster_whisper.WhisperModel("whisper-large-v3-ct2")
```

1. [将你的模型上传到 Hugging Face Hub](https://huggingface.co/docs/transformers/model_sharing#upload-with-the-web-interface)，然后通过名称加载：

```python
model = faster_whisper.WhisperModel("username/whisper-large-v3-ct2")
```

## 与其他实现进行性能对比

如果你要将它与其他 Whisper 实现做性能对比，请确保在相近设置下进行。尤其要注意：

- 确认使用了相同的转录选项，尤其是相同的 beam size。例如在 openai/whisper 中，`model.transcribe` 的默认 beam size 是 1，而这里默认使用 5。
- 转录速度与输出文本的词数密切相关，因此请确保其他实现与本实现的 WER（词错误率）大致相当。
- 在 CPU 上运行时，确保设置相同的线程数。许多框架会读取环境变量 `OMP_NUM_THREADS`，你可以在运行脚本时这样设置：

```bash
OMP_NUM_THREADS=4 python3 my_script.py
```

