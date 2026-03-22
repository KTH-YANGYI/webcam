from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 把 webcam_app/ 加到导入路径里，这样脚本可以导入共享工具模块。
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import (
    BACKEND_CHOICES,
    default_backend_order,
    open_camera,
    pretty_json,
    resolve_project_path,
    sanitize_name,
    timestamp_slug,
)


def parse_args() -> argparse.Namespace:
    # 这个脚本只解决一个问题：
    # 哪个“摄像头索引 + 后端”的组合，真的能在这台机器上读到画面。
    parser = argparse.ArgumentParser(
        description="Probe camera indices and OpenCV backends."
    )
    parser.add_argument("--max-index", type=int, default=5)
    parser.add_argument(
        "--backend-order",
        nargs="+",
        choices=BACKEND_CHOICES,
        default=None,
        help="Backend probe order. Defaults to dshow->msmf->auto on Windows.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Open a preview window for the first successful probe result.",
    )
    parser.add_argument(
        "--write-json",
        nargs="?",
        const="camera_probe_results.json",
        default=None,
        help="Write probe results to JSON. Optional custom path.",
    )
    return parser.parse_args()


def probe_one(index: int, backend: str) -> dict[str, object]:
    # 这里只测试一个具体组合：一个索引 + 一个后端。
    capture = open_camera(index, backend)
    record: dict[str, object] = {
        "index": index,
        "backend": backend,
        "opened": bool(capture.isOpened()),
        "successful_reads": 0,
        "width": None,
        "height": None,
        "reported_fps": None,
        "measured_read_fps": None,
    }

    if not capture.isOpened():
        capture.release()
        return record

    # 不能只看 isOpened()。
    # 有些摄像头“看起来打开了”，但真正读帧时还是会失败，所以这里额外读 3 帧。
    start = time.perf_counter()
    last_frame = None
    successful_reads = 0
    for _ in range(3):
        ok, frame = capture.read()
        if ok and frame is not None:
            last_frame = frame
            successful_reads += 1
    elapsed = time.perf_counter() - start

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 有些驱动不会正确返回宽高，所以必要时改用实际图像的 shape 兜底。
    if last_frame is not None and (width <= 0 or height <= 0):
        height, width = last_frame.shape[:2]

    record.update(
        {
            "successful_reads": successful_reads,
            "width": width if width > 0 else None,
            "height": height if height > 0 else None,
            "reported_fps": round(float(capture.get(cv2.CAP_PROP_FPS)), 2),
            "measured_read_fps": round(successful_reads / elapsed, 2)
            if elapsed > 0 and successful_reads
            else None,
        }
    )
    capture.release()
    return record


def preview(record: dict[str, object]) -> None:
    # 重新打开选中的摄像头，并显示实时预览，
    # 这样用户可以肉眼确认这是不是自己想要的那个设备。
    index = int(record["index"])
    backend = str(record["backend"])
    capture = open_camera(index, backend)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot reopen camera index={index} backend={backend}")

    window_name = f"camera_preview_{backend}_{index}"
    print(f"Previewing index={index} backend={backend}. Press q to quit.")
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                print("Preview frame read failed, exiting preview.")
                break
            # 把当前索引和后端写在画面上，避免预览时不知道自己看的到底是哪一路。
            cv2.putText(
                frame,
                f"index={index} backend={backend}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def maybe_write_json(path_value: str, data: dict[str, object]) -> Path:
    # 把探测结果保存成 JSON，方便你对比“插摄像头前”和“插摄像头后”的状态。
    target = resolve_project_path(path_value)
    if target.suffix.lower() != ".json":
        stem = sanitize_name(target.name or f"camera_probe_results_{timestamp_slug()}")
        target = target.parent / f"{stem}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def main() -> int:
    args = parse_args()
    backends = args.backend_order or default_backend_order()
    records: list[dict[str, object]] = []

    # 依次扫描所有候选索引和后端，把每个结果都打印出来。
    for index in range(args.max_index + 1):
        for backend in backends:
            record = probe_one(index, backend)
            records.append(record)
            width = record["width"]
            height = record["height"]
            resolution = f"{width}x{height}" if width and height else "n/a"
            print(
                f"index={index} backend={backend} opened={record['opened']} "
                f"reads={record['successful_reads']}/3 resolution={resolution} "
                f"reported_fps={record['reported_fps']} "
                f"measured_read_fps={record['measured_read_fps']}"
            )

    summary = {
        "project_root": str(PROJECT_ROOT),
        "backend_order": backends,
        "max_index": args.max_index,
        "results": records,
    }
    print("probe_summary:")
    print(pretty_json(summary))

    if args.write_json:
        json_path = maybe_write_json(args.write_json, summary)
        print(f"json_written: {json_path}")

    # “成功”不是指打开了设备，而是真正读到了至少一帧有效图像。
    successful = [record for record in records if int(record["successful_reads"]) > 0]
    if args.preview:
        if not successful:
            raise RuntimeError("No successful camera probe results available for preview.")
        preview(successful[0])

    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
