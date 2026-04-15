#!/usr/bin/env python3
"""transcribe_audio.py — 音频转文字（faster-whisper），可选说话人标签（pyannote）。

【命令格式】
  python transcribe_audio.py <输入音频> <输出.txt> [选项]
  python transcribe_audio.py -h    # 各参数一行说明；完整说明即本段文档
  拆多行时：bash 续行是「行尾单独一个反斜杠」再回车；若打成两个反斜杠，
  会把反斜杠当作普通参数传给 Python，出现 unrecognized arguments: \\

【环境与依赖】
  · 本仓库 ASR：在项目目录执行
      pip install -r requirements.txt && pip install -e .
  · GPU：需 NVIDIA 驱动；并安装 CUDA12 的 pip 动态库（否则常见 libcublas.so.12 报错）：
      pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
    脚本在导入 faster-whisper 之前会把上述包内的 lib 目录追加到 LD_LIBRARY_PATH，
    一般不必在 shell 里再 export。
  · Hugging Face：拉 Whisper / pyannote 权重时，建议设置 HF_TOKEN 或 huggingface-cli login，
    以减少匿名限流；使用 --diarize 时 token 为必需（见下）。

【主要选项（与 faster-whisper 对应）】
  --model-size      模型名或尺寸，默认 small
  --device          cpu | cuda | auto
  --compute-type    如 int8、float16、int8_float16（与 device 搭配）
  --beam-size       解码 beam，默认 5
  --language        如 zh、en；省略则自动检测
  --task            transcribe | translate
  --vad-filter      开启语音活动检测，滤非语音段
  --min-silence-ms  与 VAD 配合，默认 500
  --with-timestamps 每行带 [起止秒] 时间戳

【说话人 --diarize（非 faster-whisper 内置）】
  · 额外安装：pip install pyannote.audio torch（版本需互相兼容）
  · 在 Hugging Face 网页上对下列模型点击同意用户条款（gated）：
      pyannote/segmentation-3.0
      pyannote/speaker-diarization-3.1
    并设置环境变量 HF_TOKEN，或命令行传入 --hf-token <token>。
  · 若输入是 m4a/mp4/webm 等容器格式，脚本会在 diarization 前自动用 ffmpeg
    转成临时 wav，避免 pyannote 通过 soundfile 读取时报 “Format not recognised”。
  · 流程：先跑 pyannote 分离，释放 pipeline 后再加载 Whisper；转写后按「时间重叠」
    把每段文字标上 [SPEAKER_..]；对不齐的段标 UNKNOWN。
  · 可选：--diarize-model（默认 pyannote/speaker-diarization-3.1）、
    --min-speakers / --max-speakers 作为人数提示传给 pyannote。

【示例】（第一个参数必须是磁盘上存在的音频路径；勿照抄不存在的文件名）
  python transcribe_audio.py ./20260414_141848.m4a out.txt --device cuda --compute-type float16 --model-size small --vad-filter

  python transcribe_audio.py ./20260414_141848.m4a out.txt --device cpu --compute-type int8 --language zh --with-timestamps

  export HF_TOKEN=你的token
  python transcribe_audio.py ./20260414_141848.m4a out.txt --diarize --device cuda --compute-type float16 --model-size small

  python transcribe_audio.py ./20260414_141848.m4a out.txt --diarize --min-speakers 2 --max-speakers 6 --device cuda --compute-type float16
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

import yaml


def _prepend_pip_nvidia_cuda_libs() -> None:
    """在导入 faster_whisper 之前追加 pip 内 nvidia cuBLAS/cuDNN 的 lib（见脚本顶部文档）。"""
    if os.name != "posix":
        return
    try:
        import nvidia.cublas  # type: ignore[import-untyped]
        import nvidia.cudnn  # type: ignore[import-untyped]
    except ImportError:
        return

    cublas_lib = os.path.join(list(nvidia.cublas.__path__)[0], "lib")
    cudnn_lib = os.path.join(list(nvidia.cudnn.__path__)[0], "lib")
    extra = f"{cublas_lib}:{cudnn_lib}"
    old = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = extra + (":" + old if old else "")


_prepend_pip_nvidia_cuda_libs()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="输入音频路径")
    parser.add_argument("output", help="输出文本路径（.txt）")
    parser.add_argument(
        "--model-size",
        default="small",
        help="Whisper 模型名或尺寸（默认 small）",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="推理设备（默认 cpu）",
    )
    parser.add_argument(
        "--compute-type",
        default="int8",
        help="计算精度，如 int8 / float16（默认 int8）",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="beam size（默认 5）",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="语言码如 zh、en；省略则自动检测",
    )
    parser.add_argument(
        "--task",
        default="transcribe",
        choices=["transcribe", "translate"],
        help="转写或翻译为英语（默认 transcribe）",
    )
    parser.add_argument(
        "--vad-filter",
        action="store_true",
        help="开启 VAD，弱化静音/非语音段",
    )
    parser.add_argument(
        "--min-silence-ms",
        type=int,
        default=500,
        help="VAD 最小静音毫秒（默认 500）",
    )
    parser.add_argument(
        "--with-timestamps",
        action="store_true",
        help="每行附带 [起止秒] 时间戳",
    )
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="pyannote 说话人分离，行前加 [SPEAKER_..]（依赖与 token 见顶部文档）",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HF token；不设则用环境变量 HF_TOKEN / HUGGING_FACE_HUB_TOKEN",
    )
    parser.add_argument(
        "--diarize-model",
        default="pyannote/speaker-diarization-3.1",
        help="pyannote Pipeline 模型 ID",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=None,
        help="说话人人数下限提示（仅 --diarize）",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="说话人人数上限提示（仅 --diarize）",
    )
    return parser


def _resolve_hf_token(explicit: Optional[str]) -> Optional[str]:
    return (
        explicit
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )


def _effective_torch_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


@contextmanager
def _temporary_env(updates: dict[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _hf_download_token_kwargs(hf_hub_download: Any, hf_token: str) -> dict[str, str]:
    import inspect

    sig = inspect.signature(hf_hub_download)
    if "token" in sig.parameters:
        return {"token": hf_token}
    if "use_auth_token" in sig.parameters:
        return {"use_auth_token": hf_token}
    return {}


def _pyannote_env_updates(hf_token: str) -> dict[str, str]:
    updates = {
        "HF_TOKEN": hf_token,
        "HUGGING_FACE_HUB_TOKEN": hf_token,
    }
    # torch 2.6+ defaults torch.load(weights_only=True), but pyannote/lightning
    # checkpoints still need full checkpoint loading in common diarization flows.
    if "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD" not in os.environ:
        updates["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    return updates


def _pipeline_dependency_model_ids(config_yml: str) -> dict[str, str]:
    with open(config_yml, "r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp) or {}

    params = config.get("pipeline", {}).get("params", {})
    model_ids: dict[str, str] = {}
    for name in ("segmentation", "embedding"):
        value = params.get(name)
        if isinstance(value, str) and not os.path.isfile(value):
            model_ids[name] = value
    return model_ids


def _download_pyannote_model_checkpoint(
    hf_hub_download: Any,
    *,
    hf_token: str,
    model_id: str,
    model_role: str,
    checkpoint_name: str,
) -> None:
    from huggingface_hub.errors import (
        GatedRepoError,
        HfHubHTTPError,
        RepositoryNotFoundError,
    )

    download_kwargs: dict[str, Any] = {
        "repo_id": model_id,
        "filename": checkpoint_name,
        "repo_type": "model",
        "library_name": "pyannote",
    }
    download_kwargs.update(_hf_download_token_kwargs(hf_hub_download, hf_token))

    try:
        hf_hub_download(**download_kwargs)
    except GatedRepoError as e:
        raise SystemExit(
            "无法下载 pyannote 说话人分离依赖模型。\n"
            f"依赖角色: {model_role}\n"
            f"模型: {model_id}\n"
            "这通常表示当前账号尚未接受该 gated 模型的使用条款，或启动 Web 服务时没有带上 HF_TOKEN。\n"
            f"请访问 https://hf.co/{model_id} 接受条款，并在启动服务前导出 HF_TOKEN 后重试。"
        ) from e
    except RepositoryNotFoundError as e:
        raise SystemExit(
            "无法下载 pyannote 说话人分离依赖模型。\n"
            f"依赖角色: {model_role}\n"
            f"模型: {model_id}\n"
            "请确认该模型 ID 存在，并且当前 HF_TOKEN 对它有访问权限。"
        ) from e
    except HfHubHTTPError as e:
        raise SystemExit(
            "下载 pyannote 说话人分离依赖模型失败。\n"
            f"依赖角色: {model_role}\n"
            f"模型: {model_id}\n"
            f"HTTP 错误: {e}"
        ) from e


def _speaker_hint_kwargs(
    min_speakers: Optional[int], max_speakers: Optional[int]
) -> dict[str, int]:
    kwargs: dict[str, int] = {}
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers
    return kwargs


def _format_output_line(
    start: float,
    end: float,
    text: str,
    *,
    speaker: Optional[str],
    with_timestamps: bool,
) -> str:
    prefix = f"[{start:.2f}s -> {end:.2f}s] " if with_timestamps else ""
    speaker_prefix = f"[{speaker}] " if speaker is not None else ""
    return f"{prefix}{speaker_prefix}{text}"


@contextmanager
def _pyannote_audio_input(audio_path: Path) -> Iterator[Path]:
    """Yield an audio path pyannote can read reliably via soundfile."""
    if audio_path.suffix.lower() == ".wav":
        yield audio_path
        return

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit(
            "当前输入音频需要先转成 wav 才能做 pyannote 说话人分离，但系统里找不到 ffmpeg。\n"
            "请先安装 ffmpeg，或改用 wav/flac 等更通用的输入格式。"
        )

    with tempfile.TemporaryDirectory(prefix="fw_pyannote_audio_") as tmp_dir:
        wav_path = Path(tmp_dir) / f"{audio_path.stem}.wav"
        cmd = [
            ffmpeg,
            "-nostdin",
            "-y",
            "-i",
            str(audio_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(wav_path),
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0 or not wav_path.is_file():
            detail = (proc.stderr or proc.stdout or "").strip()
            raise SystemExit(
                "ffmpeg 预处理音频失败，无法为 pyannote 生成可读取的 wav。\n"
                f"输入文件: {audio_path}\n"
                + (f"ffmpeg 输出:\n{detail[:4000]}" if detail else "请检查输入音频是否损坏。")
            )
        yield wav_path


def _run_pyannote_diarization(
    audio_path: Path,
    *,
    hf_token: str,
    model_id: str,
    device: str,
    min_speakers: Optional[int],
    max_speakers: Optional[int],
) -> Any:
    try:
        import torch
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import (
            GatedRepoError,
            HfHubHTTPError,
            RepositoryNotFoundError,
        )
        from pyannote.audio import Pipeline  # type: ignore[import-untyped]
        from pyannote.audio.core.model import HF_PYTORCH_WEIGHTS_NAME
        from pyannote.audio.core.pipeline import PIPELINE_PARAMS_NAME
    except ImportError as e:
        raise SystemExit(
            "未安装 pyannote / torch，无法使用 --diarize。请执行:\n"
            "  pip install pyannote.audio torch\n"
            f"原始错误: {e}"
        ) from e

    # pyannote.audio 3.4.x 在部分下载失败场景会直接返回 None；先自行拉取 config.yaml，
    # 这样能把 gated/token/模型 ID/HTTP 错误转换成明确提示，而不是后面才炸 NoneType。
    with _temporary_env(_pyannote_env_updates(hf_token)):
        download_kwargs: dict[str, Any] = {
            "repo_id": model_id,
            "filename": PIPELINE_PARAMS_NAME,
            "repo_type": "model",
            "library_name": "pyannote",
        }
        download_kwargs.update(_hf_download_token_kwargs(hf_hub_download, hf_token))

        try:
            config_yml = hf_hub_download(**download_kwargs)
        except GatedRepoError as e:
            raise SystemExit(
                "无法下载 pyannote 说话人分离模型：当前账号还没有该 gated 模型的访问权限。\n"
                f"请访问 https://hf.co/{model_id} 并接受使用条款，然后重试。"
            ) from e
        except RepositoryNotFoundError as e:
            raise SystemExit(
                "无法下载 pyannote 说话人分离模型：模型 ID 不存在，或当前 token 无权访问。\n"
                f"请检查 --diarize-model 是否正确，并确认 HF_TOKEN 对 {model_id} 有权限。"
            ) from e
        except HfHubHTTPError as e:
            raise SystemExit(
                "下载 pyannote 说话人分离模型失败。\n"
                f"模型: {model_id}\n"
                f"HTTP 错误: {e}"
            ) from e

        dependency_model_ids = _pipeline_dependency_model_ids(config_yml)
        for model_role, dependency_model_id in dependency_model_ids.items():
            _download_pyannote_model_checkpoint(
                hf_hub_download,
                hf_token=hf_token,
                model_id=dependency_model_id,
                model_role=model_role,
                checkpoint_name=HF_PYTORCH_WEIGHTS_NAME,
            )

        try:
            pipeline = Pipeline.from_pretrained(config_yml, use_auth_token=hf_token)
        except AttributeError as e:
            if "'NoneType' object has no attribute 'eval'" in str(e):
                raise SystemExit(
                    "pyannote Pipeline 初始化失败：某个依赖模型没有正确加载。\n"
                    "最常见原因是 segmentation/embedding 子模型未授权、HF_TOKEN 未传入到当前进程，"
                    "或者缓存里只有主 pipeline 配置而没有依赖模型权重。\n"
                    "请确认你已经接受以下模型条款，并在启动 Web 服务前导出 HF_TOKEN：\n"
                    "  https://hf.co/pyannote/speaker-diarization-3.1\n"
                    "  https://hf.co/pyannote/segmentation-3.0"
                ) from e
            raise
        except Exception as e:
            if "Weights only load failed" in str(e):
                raise SystemExit(
                    "pyannote 模型已下载，但在当前 torch/pyannote 版本组合下加载 checkpoint 失败。\n"
                    "脚本已自动启用 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 来兼容 torch 2.6+；"
                    "如果你仍然看到这个错误，请确认当前进程确实运行的是本仓库这份脚本，"
                    "并且使用的是 faster-whisper 环境。"
                ) from e
            raise

    if pipeline is None:
        raise SystemExit(
            "pyannote Pipeline 加载失败，返回了空对象。\n"
            f"请确认 HF_TOKEN 有效，并且你已在 https://hf.co/{model_id} 接受模型条款。"
        )

    if device == "cuda":
        pipeline.to(torch.device("cuda"))

    with _pyannote_audio_input(audio_path) as diarize_audio_path:
        diarization = pipeline(
            str(diarize_audio_path),
            **_speaker_hint_kwargs(min_speakers, max_speakers),
        )
    del pipeline
    if device == "cuda":
        torch.cuda.empty_cache()
    return diarization


def _best_speaker_for_interval(
    start: float, end: float, diarization: Any
) -> str:
    best_label: Optional[str] = None
    best_overlap = 0.0
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        overlap = max(0.0, min(end, turn.end) - max(start, turn.start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = speaker
    return best_label if best_label is not None else "UNKNOWN"


def main() -> None:
    args = build_parser().parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"找不到输入音频: {input_path}\n"
            "请把第一个参数换成你机器上存在的路径（文档示例里的文件名需按实际修改）。"
        )

    diarization = None
    if args.diarize:
        token = _resolve_hf_token(args.hf_token)
        if not token:
            raise SystemExit(
                "--diarize 需要 Hugging Face Token（已同意 pyannote 相关模型条款）。"
                "请设置环境变量 HF_TOKEN 或传入 --hf-token。"
            )
        diarization = _run_pyannote_diarization(
            input_path,
            hf_token=token,
            model_id=args.diarize_model,
            device=_effective_torch_device(args.device),
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
        )

    from faster_whisper import WhisperModel

    model = WhisperModel(
        args.model_size,
        device=args.device,
        compute_type=args.compute_type,
    )

    # VAD parameters only take effect when vad_filter is enabled.
    vad_parameters = {"min_silence_duration_ms": args.min_silence_ms}
    segments, info = model.transcribe(
        str(input_path),
        beam_size=args.beam_size,
        language=args.language,
        task=args.task,
        vad_filter=args.vad_filter,
        vad_parameters=vad_parameters,
    )

    lines: list[str] = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        speaker = (
            _best_speaker_for_interval(seg.start, seg.end, diarization)
            if diarization is not None
            else None
        )
        lines.append(
            _format_output_line(
                seg.start,
                seg.end,
                text,
                speaker=speaker,
                with_timestamps=args.with_timestamps,
            )
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Detected language: {info.language} (p={info.language_probability:.4f})")
    print(f"Wrote {len(lines)} lines to: {output_path}")


if __name__ == "__main__":
    main()
