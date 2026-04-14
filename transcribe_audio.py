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

from pathlib import Path
from typing import Any, Optional

from faster_whisper import WhisperModel


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
        from pyannote.audio import Pipeline  # type: ignore[import-untyped]
    except ImportError as e:
        raise SystemExit(
            "未安装 pyannote / torch，无法使用 --diarize。请执行:\n"
            "  pip install pyannote.audio torch\n"
            f"原始错误: {e}"
        ) from e

    # 旧版 pyannote 只认 use_auth_token，新版 huggingface_hub 已去掉该参数；统一用环境变量最稳。
    _tok_old = os.environ.get("HF_TOKEN")
    _hub_old = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    try:
        pipeline = Pipeline.from_pretrained(model_id)
    finally:
        if _tok_old is None:
            os.environ.pop("HF_TOKEN", None)
        else:
            os.environ["HF_TOKEN"] = _tok_old
        if _hub_old is None:
            os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
        else:
            os.environ["HUGGING_FACE_HUB_TOKEN"] = _hub_old

    if device == "cuda":
        pipeline.to(torch.device("cuda"))

    infer_kw: dict[str, int] = {}
    if min_speakers is not None:
        infer_kw["min_speakers"] = min_speakers
    if max_speakers is not None:
        infer_kw["max_speakers"] = max_speakers

    diarization = pipeline(str(audio_path), **infer_kw)
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

    if not input_path.exists():
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
        spk = (
            _best_speaker_for_interval(seg.start, seg.end, diarization)
            if diarization is not None
            else None
        )
        if args.with_timestamps:
            if spk is not None:
                lines.append(
                    f"[{seg.start:.2f}s -> {seg.end:.2f}s] [{spk}] {text}"
                )
            else:
                lines.append(f"[{seg.start:.2f}s -> {seg.end:.2f}s] {text}")
        else:
            if spk is not None:
                lines.append(f"[{spk}] {text}")
            else:
                lines.append(text)

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Detected language: {info.language} (p={info.language_probability:.4f})")
    print(f"Wrote {len(lines)} lines to: {output_path}")


if __name__ == "__main__":
    main()
