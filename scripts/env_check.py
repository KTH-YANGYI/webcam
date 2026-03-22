from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import cv2
import torch
import ultralytics
import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 把 webcam_app/ 加到导入路径里，这样脚本即使从别的目录启动，
# 也能正常导入 common.py。
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import DEFAULT_CONFIG_PATH, pretty_json, resolve_project_path


def parse_args() -> argparse.Namespace:
    # 这个脚本的职责很单一：
    # 只检查环境是否正常、模型能不能加载，不做正式推理。
    parser = argparse.ArgumentParser(
        description="Validate the Python environment and default YOLO model."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the YAML config file. Relative paths resolve from webcam_app/.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional model override. Relative paths resolve from webcam_app/.",
    )
    return parser.parse_args()


def load_model_path(config_path: Path, model_override: str | None) -> Path:
    # 模型路径优先使用命令行传入的值；
    # 如果命令行没传，就回到 default.yaml 里读取。
    if model_override:
        return resolve_project_path(model_override)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if "model" not in config:
        raise KeyError(f"Missing 'model' in config file: {config_path}")
    return resolve_project_path(config["model"])


def main() -> int:
    # 第一步：先确定本次到底要用哪个配置文件、哪个模型文件。
    args = parse_args()
    config_path = resolve_project_path(args.config)
    model_path = load_model_path(config_path, args.model)

    # 第二步：把最关键的环境信息打印出来。
    # 这样可以快速确认 Python、OpenCV、Torch、CUDA、Ultralytics
    # 是不是都来自你想要的那个环境。
    print(f"project_root: {PROJECT_ROOT}")
    print(f"config_path: {config_path}")
    print(f"model_path: {model_path}")
    print(f"python_executable: {sys.executable}")
    print(f"python_version: {sys.version}")
    print(f"opencv_version: {cv2.__version__}")
    print(f"torch_version: {torch.__version__}")
    print(f"torch_cuda_version: {torch.version.cuda}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_device_0: {torch.cuda.get_device_name(0)}")
    print(f"ultralytics_version: {ultralytics.__version__}")
    ultralytics.checks()

    # 第三步：在真正加载模型前，先确认权重文件确实存在。
    if not model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    # 第四步：用 Ultralytics 加载模型。
    # 如果这里失败，通常说明环境不对、路径不对，或者权重和当前运行方式不兼容。
    model = YOLO(str(model_path))
    model_names = getattr(model, "names", None)
    print("model_load: OK")
    print("model_names:")
    print(pretty_json(model_names))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        # 出错时把完整报错栈打印出来，方便后面排查环境问题。
        print(f"env_check_failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
