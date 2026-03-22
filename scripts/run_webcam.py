from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path
from time import perf_counter

import cv2
import torch
import ultralytics
import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 把 webcam_app/ 加到导入路径里，
# 这样无论命令从哪个目录执行，都能正确导入 common.py。
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import (
    BACKEND_CHOICES,
    DEFAULT_CONFIG_PATH,
    LABELS_DIR,
    LOGS_DIR,
    VIDEOS_DIR,
    ensure_output_dirs,
    open_camera,
    pretty_json,
    resolve_project_path,
    sanitize_name,
    timestamp_slug,
)


def announce(message: str) -> None:
    # 这个函数专门用来把关键状态立即打印到终端。
    # 使用 flush=True 的目的，是避免用户觉得“命令执行后完全没反应”。
    print(message, flush=True)


DEFAULTS = {
    # 这是代码里的内置默认值。
    # 最终运行配置的优先级是：内置默认值 -> YAML 配置 -> 命令行参数。
    "model": "../yolo11n_train_v3_datachanged_2/weights/best.pt",
    "device": "0",
    "imgsz": 640,
    "conf": 0.25,
    "backend": "dshow",
    "source": 0,
    # 摄像头直接采 1080p 会明显拖慢整条链路；
    # 这里改成 1280x720，通常能比 1080p 更流畅，同时比 640x480 保留更多细节。
    "width": 1280,
    "height": 720,
    "fps": 30,
    # 这台机器是 RTX 5060，默认打开 FP16 一般能提升一点推理速度。
    "half": True,
    # 很多 USB UVC 摄像头在 Windows + DirectShow 下使用 MJPG 会更流畅。
    "fourcc": "MJPG",
    "save": False,
    "save_txt": False,
    "classes": None,
    "name": None,
    "buffer_size": 1,
}


def build_parser() -> argparse.ArgumentParser:
    
    parser = argparse.ArgumentParser(description="Run YOLO webcam inference.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to YAML config. Relative paths resolve from webcam_app/.",
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--source", type=int, default=None)
    parser.add_argument("--backend", choices=BACKEND_CHOICES, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--half", action="store_true", default=None)
    parser.add_argument("--fourcc", type=str, default=None)
    parser.add_argument("--save", action="store_true", default=None)
    parser.add_argument("--save-txt", dest="save_txt", action="store_true", default=None)
    parser.add_argument("--classes", type=int, nargs="*", default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--buffer-size", dest="buffer_size", type=int, default=None)
    return parser


def load_config(config_path: Path) -> dict[str, object]:
    # 先读 YAML，再覆盖到代码默认值上。
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise TypeError(f"Config file must contain a mapping: {config_path}")
    config = dict(DEFAULTS)
    config.update(loaded)
    return config


def merge_cli_overrides(config: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    # 只有命令行里真的传了值的字段，才会覆盖 YAML 里的配置。
    merged = dict(config)
    for key, value in vars(args).items():
        if key == "config" or value is None:
            continue
        merged[key] = value
    return merged


def normalize_runtime_config(config: dict[str, object]) -> dict[str, object]:
    # 把 YAML 和命令行混合出来的配置统一整理成“最终运行配置”。
    # 这里会顺手把字符串、整数、浮点数、布尔值都转换成合适类型，
    # 避免后面的实时循环里出现类型错误。
    runtime = dict(config)
    runtime["model"] = str(resolve_project_path(runtime["model"]))
    runtime["source"] = int(runtime["source"])
    runtime["imgsz"] = int(runtime["imgsz"])
    runtime["conf"] = float(runtime["conf"])
    runtime["device"] = str(runtime["device"])
    runtime["backend"] = str(runtime["backend"])
    runtime["width"] = int(runtime["width"])
    runtime["height"] = int(runtime["height"])
    runtime["fps"] = float(runtime["fps"])
    runtime["half"] = bool(runtime["half"])
    runtime["fourcc"] = None if config.get("fourcc") in (None, "", "none", "NONE") else str(runtime["fourcc"]).upper()
    runtime["save"] = bool(runtime["save"])
    runtime["save_txt"] = bool(runtime["save_txt"])
    runtime["buffer_size"] = int(runtime["buffer_size"])
    runtime["classes"] = None if runtime["classes"] in (None, []) else list(runtime["classes"])
    runtime["name"] = sanitize_name(runtime["name"] or f"webcam_{timestamp_slug()}")
    return runtime


def setup_logger(run_id: str) -> tuple[logging.Logger, Path]:
    # 日志同时写到终端和文件里。
    # 这样即使这次运行已经结束，后面仍然可以回看当时到底用了什么配置。
    ensure_output_dirs()
    log_path = LOGS_DIR / f"run_{run_id}.log"
    logger = logging.getLogger(f"webcam_app.{run_id}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger, log_path


def maybe_fallback_device(device: str, use_half: bool, logger: logging.Logger) -> tuple[str, bool]:
    # 如果用户要求用 GPU，但当前 CUDA 不可用，
    # 脚本不会立刻崩掉，而是打印警告并自动退回 CPU。
    # 这样至少还能先把整条流程跑通。
    normalized = device.strip().lower()
    wants_cuda = normalized in {"0", "cuda", "cuda:0"}
    if wants_cuda and not torch.cuda.is_available():
        logger.warning("CUDA not available. Falling back to CPU.")
        return "cpu", False
    if normalized == "cpu" and use_half:
        logger.warning("FP16 is disabled on CPU. Falling back to half=False.")
        return "cpu", False
    return device, use_half


def configure_capture(
    source: int, backend: str, width: int, height: int, fps: float, buffer_size: int, fourcc: str | None
) -> cv2.VideoCapture:
    # 先打开摄像头，再尝试设置分辨率和 FPS。
    # 这里属于“尽量设置”，因为有些驱动会忽略这些请求。
    capture = open_camera(source, backend)
    if capture.isOpened():
        # 把缓冲区压小，减少“看到的是几帧前画面”的延迟。
        if buffer_size > 0:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
        # 对很多 UVC 摄像头来说，MJPG 比默认原始格式更省带宽，常常能换来更高帧率。
        if fourcc and len(fourcc) == 4:
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        if width > 0:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height > 0:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps > 0:
            capture.set(cv2.CAP_PROP_FPS, fps)
    return capture


def create_writer(output_name: str, frame_shape: tuple[int, int, int], fps: float) -> cv2.VideoWriter:
    # 输出视频的尺寸直接使用当前带框图像的真实尺寸，
    # 不靠手工猜测，避免保存出来的视频尺寸不对。
    height, width = frame_shape[:2]
    output_path = VIDEOS_DIR / f"{output_name}.mp4"
    writer_fps = fps if fps > 0 else 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(output_path), fourcc, writer_fps, (width, height))


def draw_status_overlay(
    image,
    frame_index: int,
    fps_value: float,
    detection_count: int,
    model_label: str,
) -> None:
    # OpenCV 自带字体不适合直接写中文，所以状态栏使用简单英文，
    # 这样在任何 Windows 机器上都能稳定显示。
    status_text = f"Frame: {frame_index}  FPS: {fps_value:.1f}  Detections: {detection_count}"
    model_text = f"Model: {model_label}"
    cv2.rectangle(image, (12, 12), (900, 92), (0, 0, 0), -1)
    cv2.putText(
        image,
        status_text,
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        model_text,
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def main() -> int:
    # 第一步：把默认值、YAML、命令行参数合并成最终运行配置。
    parser = build_parser()
    args = parser.parse_args()

    config_path = resolve_project_path(args.config)
    config = load_config(config_path)
    runtime = normalize_runtime_config(merge_cli_overrides(config, args))

    run_id = timestamp_slug()
    logger, log_path = setup_logger(run_id)
    logger.info("Resolved runtime config:\n%s", pretty_json(runtime))
    logger.info("Config path: %s", config_path)
    logger.info("Log path: %s", log_path)
    announce("脚本已启动，正在准备运行配置...")

    model_path = Path(runtime["model"])
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # 第二步：根据 CUDA 实际状态，确定这次到底用 GPU 还是 CPU。
    device, use_half = maybe_fallback_device(str(runtime["device"]), bool(runtime["half"]), logger)
    runtime["device"] = device
    runtime["half"] = use_half

    # 第三步：把环境信息写进日志，方便后面排查问题或复现实验。
    logger.info("Python executable: %s", sys.executable)
    logger.info("Python version: %s", sys.version)
    logger.info("OpenCV version: %s", cv2.__version__)
    logger.info("Torch version: %s", torch.__version__)
    logger.info("Torch CUDA version: %s", torch.version.cuda)
    logger.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("CUDA device: %s", torch.cuda.get_device_name(0))
    logger.info("Ultralytics version: %s", ultralytics.__version__)
    announce("环境检查完成，准备加载模型...")

    # 第四步：加载模型，并打开用户选中的摄像头。
    logger.info("Starting model load: %s", model_path)
    model = YOLO(str(model_path))
    logger.info("Model load finished.")
    announce("模型加载完成，准备打开摄像头...")
    model_label = model_path.name

    logger.info(
        "Opening camera source=%s backend=%s width=%s height=%s fps=%s fourcc=%s",
        runtime["source"],
        runtime["backend"],
        runtime["width"],
        runtime["height"],
        runtime["fps"],
        runtime["fourcc"],
    )
    capture = configure_capture(
        source=int(runtime["source"]),
        backend=str(runtime["backend"]),
        width=int(runtime["width"]),
        height=int(runtime["height"]),
        fps=float(runtime["fps"]),
        buffer_size=int(runtime["buffer_size"]),
        fourcc=runtime["fourcc"],
    )
    if not capture.isOpened():
        raise RuntimeError(
            f"Cannot open camera source={runtime['source']} backend={runtime['backend']}"
        )
    logger.info("Camera opened successfully.")
    announce("摄像头已打开，检测窗口应该会弹出...")

    writer = None
    frame_index = 0
    first_frame_logged = False
    last_frame_time = None
    fps_value = 0.0
    window_name = f"YOLO Webcam Detection - {runtime['name']}"

    try:
        # 第五步：进入实时循环。
        # 每次循环都做同一件事：读一帧 -> 推理 -> 画框 -> 显示/保存。
        logger.info("Entering realtime inference loop.")
        if str(runtime["device"]).lower() in {"0", "cuda", "cuda:0"} and torch.cuda.is_available():
            # 输入尺寸固定时，打开 cudnn benchmark 往往能让 GPU 推理更快一点。
            torch.backends.cudnn.benchmark = True
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                logger.error("Frame read failed. Stopping inference loop.")
                break
            current_time = perf_counter()
            if last_frame_time is not None:
                delta = current_time - last_frame_time
                if delta > 0:
                    instant_fps = 1.0 / delta
                    # 用一个轻量平滑值，避免 FPS 数字每帧跳动太厉害。
                    fps_value = instant_fps if fps_value == 0.0 else (fps_value * 0.85 + instant_fps * 0.15)
            last_frame_time = current_time
            if not first_frame_logged:
                logger.info("First frame read successfully: shape=%s", frame.shape)
                announce("已经成功读取第一帧，正在进行实时检测...")
                first_frame_logged = True

            # Ultralytics 可以直接接收 OpenCV 读出来的 NumPy 图像，
            # 不需要先把每一帧保存成图片文件再读回去。
            results = model(
                frame,
                imgsz=int(runtime["imgsz"]),
                conf=float(runtime["conf"]),
                device=str(runtime["device"]),
                half=bool(runtime["half"]),
                classes=runtime["classes"],
                verbose=False,
            )
            result = results[0]
            # plot() 会返回一张“已经画好框和标签”的图像。
            annotated = result.plot()
            detection_count = 0 if result.boxes is None else len(result.boxes)
            draw_status_overlay(
                annotated,
                frame_index=frame_index,
                fps_value=fps_value,
                detection_count=detection_count,
                model_label=model_label,
            )

            if bool(runtime["save"]):
                if writer is None:
                    # 只有拿到真实画面尺寸后，才能安全创建视频写入器。
                    writer = create_writer(str(runtime["name"]), annotated.shape, float(runtime["fps"]))
                    logger.info("Video writer initialized: %s", VIDEOS_DIR / f"{runtime['name']}.mp4")
                writer.write(annotated)

            if bool(runtime["save_txt"]):
                # 每一帧单独保存一个标签文件，后面可以逐帧检查检测结果。
                label_path = LABELS_DIR / f"{runtime['name']}_{frame_index:06d}.txt"
                result.save_txt(str(label_path), save_conf=True)

            cv2.imshow(window_name, annotated)
            frame_index += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Received quit signal from keyboard.")
                break
    finally:
        # 无论循环是正常退出还是中途报错，都必须释放摄像头、视频写入器和窗口资源。
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        logger.info("Resources released. Processed %s frames.", frame_index)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        # 130 是常见的“用户手动中断”退出码。
        print("Interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        # 发生异常时打印完整报错栈，方便定位运行时问题。
        print(f"run_webcam_failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
