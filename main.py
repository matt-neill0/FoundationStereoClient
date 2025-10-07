import argparse, pathlib, time, os, sys, asyncio
from sender_core import StereoSenderClient

ALLOWED_RESOLUTIONS = ["320 x 240", "426 x 240", "480 x 360", "640 x 360", "640 x 480", "848 x 480", "960 x 720", "1280 x 720"]

def parse_res_label(label: str) -> tuple[int, int]:
    w_str, h_str = label.split("x")
    return int(w_str.strip()), int(h_str.strip())

def build_arg_parser():
    p = argparse.ArgumentParser(description="Stereo sender. Use --gui to launch desktop app.")
    p.add_argument("--gui", action="store_true", help="Launch the GUI.")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--path", default="/foundation-stereo")
    p.add_argument("--left-cam", type=int)
    p.add_argument("--right-cam", type=int)
    p.add_argument("--left-file")
    p.add_argument("--right-file")
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--jpeg-quality", type=int, default=90)
    p.add_argument("--resolution", choices=ALLOWED_RESOLUTIONS, default="640 x 480", help="Live stream resolution (WxH)")
    p.add_argument("--session-id", default=None)
    p.add_argument("--save-dir", default=None)
    p.add_argument("--preview", action="store_true")
    return p

def main_cli(args):
    if args.left_cam is not None and args.right_cam is not None:
        left_src, right_src, mode = str(args.left_cam), str(args.right_cam), "stream"
    elif args.left_file and args.right_file:
        if not (os.path.exists(args.left_file) and os.path.exists(args.right_file)):
            print("File paths are invalid.", file=sys.stderr); sys.exit(2)
        left_src, right_src, mode = args.left_file, args.right_file, "file"
    else:
        print("Specify either --left-cam/--right-cam or --left-file/--right-file", file=sys.stderr); sys.exit(2)

    session_id = args.session_id or f"sess-{int(time.time())}"
    save_dir = pathlib.Path(args.save_dir) if args.save_dir else None
    frame_w, frame_h = parse_res_label(args.resolution)
    client = StereoSenderClient(
        args.host, args.port, args.path,
        args.fps, args.jpeg_quality, session_id,
        frame_width=frame_w, frame_height=frame_h,
        save_dir=save_dir, preview=args.preview,
        on_log=lambda s: print(s),
        on_result=lambda *a: print(f"[result] seq={a[0]} kind={a[1]} enc={a[2]} {a[3]}x{a[4]} bytes={len(a[5])}"),
        on_start=lambda: print("[client] started"),
        on_finish=lambda: print("[client] finished")
    )

    try:
        asyncio.run(client.start_stream(left_src, right_src, mode))
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