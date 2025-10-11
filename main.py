from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Tuple

DEFAULT_HEIGHT = 480
DEFAULT_WIDTH = 640
DEFAULT_FPS = 30.0

def build_arg_parser():
    p = argparse.ArgumentParser(description="Run FoundationStereo locally without streaming to a server.")
    p.add_argument("--gui", action="store_true", help="Launch the GUI.")
    p.add_argument("--engine", help="Path to the TensorRT engine used for local inference.")
    p.add_argument("--left-cam", type=int)
    p.add_argument("--right-cam", type=int)
    p.add_argument("--left-file")
    p.add_argument("--right-file")
    p.add_argument("--use-realsense", action="store_true")
    p.add_argument("--frame-width", type=int, default=DEFAULT_WIDTH)
    p.add_argument("--frame-height", type=int, default=DEFAULT_HEIGHT)
    p.add_argument("--fps", type=float, default=DEFAULT_FPS)
    p.add_argument("--session-id", default=None)
    p.add_argument("--save-dir", default=None)
    p.add_argument("--preview", action="store_true", help="Show an OpenCV preview window.")
    p.add_argument("--max-disp", type=float, default=256.0, help="Maximum disparity value used for encoding results.")
    p.add_argument("--disp-scale", type=float, default=256.0, help="Scale factor applied before encoding disparity to PNG16.")
    return p

def _resolve_sources(args) -> Tuple[str, str, str]:
    if args.use_realsense and (args.left_file or args.right_file):
        raise ValueError("--use-realsense cannot be combined with video file inputs.")

    if args.use_realsense:
        return "realsense", "realsense", "stream"

    if args.left_cam is not None and args.right_cam is not None:
        return str(args.left_cam), str(args.right_cam), "stream"

    if args.left_file and args.right_file:
        left_path = Path(args.left_file)
        right_path = Path(args.right_file)
        if not (left_path.exists() and right_path.exists()):
            raise ValueError("File paths are invalid.")
        return str(left_path), str(right_path), "file"

    raise ValueError(
        "Specify either --left-cam/--right-cam, --left-file/--right-file, or --use-realsense."
    )

def main_cli(args):
    from local_inference import (
        LocalEngineRunner,
        TensorRTUnavailableError,
        ensure_tensorrt_runtime,
    )

    try:
        left_src, right_src, mode = _resolve_sources(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)

    if not args.engine:
        print("--engine is required when running without the GUI.", file=sys.stderr)
        sys.exit(2)

    engine_path = Path(args.engine)
    if not engine_path.exists():
        print(f"TensorRT engine not found: {engine_path}", file=sys.stderr)
        sys.exit(2)

    try:
        ensure_tensorrt_runtime()
    except TensorRTUnavailableError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    session_id = args.session_id or f"local-{int(time.time())}"
    save_dir = Path(args.save_dir).expanduser() if args.save_dir else None

    runner = LocalEngineRunner(
        engine_path=str(engine_path),
        left_src=left_src,
        right_src=right_src,
        mode=mode,
        frame_width=args.frame_width or DEFAULT_WIDTH,
        frame_height=args.frame_height or DEFAULT_HEIGHT,
        fps=args.fps or DEFAULT_FPS,
        session_id=session_id,
        save_dir=save_dir,
        preview=bool(args.preview),
        max_disp=args.max_disp,
        disp_scale=args.disp_scale,
        on_log=lambda s: print(s),
        on_result=lambda *a: print(
            f"[result] seq={a[0]} kind={a[1]} enc={a[2]} {a[3]}x{a[4]} bytes={len(a[5])}"
        ),
        use_realsense=bool(args.use_realsense)
    )

    try:
        runner.run()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    ap = build_arg_parser()
    args = ap.parse_args()
    if args.gui:
        try:
            from GUI import launch_gui
        except ImportError:
            print("PySide6 GUI not available. Install with pip install PySide6", file=sys.stderr)
            sys.exit(1)
        sys.exit(launch_gui())
    else:
        main_cli(args)