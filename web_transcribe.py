#!/usr/bin/env python3
"""本目录 Web UI：监听 11221，上传音频后子进程调用 transcribe_audio.py，浏览器下载 txt。

【虚拟环境】与命令行转写相同，使用已有环境 faster-whisper（勿另建新 venv）::

    conda activate faster-whisper

【依赖】在仓库根目录（本脚本所在目录）::

    pip install -r requirements.txt
    pip install -e .

【启动】先 cd 到本脚本所在目录（faster-whisper 仓库根）::

    cd <path/to/faster-whisper>
    python web_transcribe.py

服务绑定 0.0.0.0:11221。浏览器访问 http://127.0.0.1:11221/ （局域网用本机 IP）。

【说明】
  · GPU 与 cuBLAS/cuDNN 等与 transcribe_audio.py 一致，请已在该环境中按脚本顶部文档配置。
  · 勾选 diarize 时，请在启动本服务前在 shell 中 export HF_TOKEN（网页不传 token）。
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from flask import Flask, Response, after_this_request, render_template, request, send_file
from werkzeug.utils import secure_filename


def _fail(msg: str, code: int = 400) -> Response:
    return Response(
        msg + "\n",
        status=code,
        mimetype="text/plain; charset=utf-8",
    )

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "transcribe_audio.py"
ALLOWED_SUFFIX = frozenset(
    {".m4a", ".mp3", ".wav", ".webm", ".ogg", ".flac", ".mp4", ".mpeg", ".opus"}
)
MODEL_SIZE_RE = re.compile(r"^[\w.\-]+$")

app = Flask(__name__, template_folder=str(ROOT / "templates"))
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024


@app.get("/")
def index():
    return render_template("transcribe.html")


@app.post("/transcribe")
def transcribe():
    if "audio" not in request.files:
        return _fail("缺少表单字段 audio")
    up = request.files["audio"]
    if not up.filename:
        return _fail("未选择文件")

    raw_name = secure_filename(up.filename) or "audio"
    suf = Path(raw_name).suffix.lower()
    if suf not in ALLOWED_SUFFIX:
        return _fail(
            f"不支持的扩展名: {suf}，允许: {', '.join(sorted(ALLOWED_SUFFIX))}"
        )

    model_size = request.form.get("model_size", "small").strip()
    if not MODEL_SIZE_RE.match(model_size):
        return _fail("非法 model_size")

    device = request.form.get("device", "cpu")
    if device not in ("cpu", "cuda", "auto"):
        return _fail("非法 device")

    compute_type = request.form.get("compute_type", "int8").strip()
    if not re.match(r"^[\w.\-]+$", compute_type):
        return _fail("非法 compute_type")

    try:
        beam_size = int(request.form.get("beam_size", "5"))
    except ValueError:
        return _fail("beam_size 须为整数")
    if not 1 <= beam_size <= 20:
        return _fail("beam_size 须在 1–20")

    language = request.form.get("language", "").strip() or None
    if language is not None and not re.match(r"^[a-zA-Z\-]{2,15}$", language):
        return _fail("非法 language")

    task = request.form.get("task", "transcribe")
    if task not in ("transcribe", "translate"):
        return _fail("非法 task")

    try:
        min_silence_ms = int(request.form.get("min_silence_ms", "500"))
    except ValueError:
        return _fail("min_silence_ms 须为整数")
    if not 0 <= min_silence_ms <= 10000:
        return _fail("min_silence_ms 须在 0–10000")

    vad_filter = request.form.get("vad_filter") == "1"
    with_timestamps = request.form.get("with_timestamps") == "1"
    diarize = request.form.get("diarize") == "1"

    min_spk = request.form.get("min_speakers", "").strip()
    max_spk = request.form.get("max_speakers", "").strip()
    if min_spk:
        try:
            min_spk_i = int(min_spk)
        except ValueError:
            return _fail("min_speakers 须为整数")
        if not 1 <= min_spk_i <= 32:
            return _fail("min_speakers 须在 1–32")
    else:
        min_spk_i = None
    if max_spk:
        try:
            max_spk_i = int(max_spk)
        except ValueError:
            return _fail("max_speakers 须为整数")
        if not 1 <= max_spk_i <= 32:
            return _fail("max_speakers 须在 1–32")
    else:
        max_spk_i = None

    if not SCRIPT.is_file():
        return _fail("找不到 transcribe_audio.py", 500)

    tmp_dir = Path(tempfile.mkdtemp(prefix="fw_web_"))
    in_path = tmp_dir / f"{uuid.uuid4().hex}{suf}"
    out_path = tmp_dir / "out.txt"
    up.save(in_path)

    cmd: list[str] = [
        sys.executable,
        str(SCRIPT),
        str(in_path),
        str(out_path),
        "--model-size",
        model_size,
        "--device",
        device,
        "--compute-type",
        compute_type,
        "--beam-size",
        str(beam_size),
        "--task",
        task,
        "--min-silence-ms",
        str(min_silence_ms),
    ]
    if language:
        cmd.extend(["--language", language])
    if vad_filter:
        cmd.append("--vad-filter")
    if with_timestamps:
        cmd.append("--with-timestamps")
    if diarize:
        cmd.append("--diarize")
        if min_spk_i is not None:
            cmd.extend(["--min-speakers", str(min_spk_i)])
        if max_spk_i is not None:
            cmd.extend(["--max-speakers", str(max_spk_i)])

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=7200,
            check=False,
        )
    except subprocess.TimeoutExpired:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return _fail("转写超时（>2h）", 504)

    if proc.returncode != 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        err = (proc.stderr or proc.stdout or "").strip() or f"exit {proc.returncode}"
        return _fail(err[:8000], 500)

    if not out_path.is_file():
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return _fail("未生成输出文件", 500)

    stem = Path(raw_name).stem or "transcript"
    safe_stem = re.sub(r"[^\w\-]", "_", stem)[:120] or "transcript"
    download_name = f"{safe_stem}_transcript.txt"

    @after_this_request
    def _cleanup(_response):
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return _response

    return send_file(
        out_path,
        as_attachment=True,
        download_name=download_name,
        mimetype="text/plain; charset=utf-8",
    )


def main() -> None:
    app.run(host="0.0.0.0", port=11221, threaded=True)


if __name__ == "__main__":
    main()
