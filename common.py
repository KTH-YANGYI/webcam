from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2


# 把项目里最重要的路径集中放在这里，避免每个脚本各自处理路径时出现不一致。
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
VIDEOS_DIR = OUTPUTS_DIR / "videos"
LABELS_DIR = OUTPUTS_DIR / "labels"
LOGS_DIR = OUTPUTS_DIR / "logs"

# 这些名字就是命令行里允许用户输入的摄像头后端选项。
BACKEND_CHOICES = ("auto", "dshow", "msmf", "v4l2")
_BACKEND_API_IDS = {
    "auto": None,
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "v4l2": cv2.CAP_V4L2,
}


def resolve_project_path(value: str | Path) -> Path:
    # 所有相对路径都固定按 webcam_app/ 目录来解释，
    # 不依赖当前终端所在目录，这样配置文件才不会“时好时坏”。
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def ensure_output_dirs() -> None:
    # 在写日志、视频、标签之前，先确保输出目录已经存在。
    for directory in (VIDEOS_DIR, LABELS_DIR, LOGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sanitize_name(value: str) -> str:
    # Windows 文件名不适合带太多特殊字符，这里先做一次清洗。
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return sanitized.strip("._-") or f"run_{timestamp_slug()}"


def default_backend_order() -> list[str]:
    # 不同操作系统适合的摄像头后端不同，所以这里给出默认优先级。
    if sys.platform.startswith("win"):
        return ["dshow", "msmf", "auto"]
    if sys.platform.startswith("linux"):
        return ["auto", "v4l2"]
    return ["auto"]


def open_camera(index: int, backend: str) -> cv2.VideoCapture:
    # auto 表示让 OpenCV 自己选后端；
    # 其他值则表示强制指定某个后端，例如 dshow 或 msmf。
    api_id = _BACKEND_API_IDS[backend]
    if api_id is None:
        return cv2.VideoCapture(index)
    return cv2.VideoCapture(index, api_id)


def pretty_json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True)
