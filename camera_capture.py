import cv2, threading, time
from typing import List, Optional
import numpy as np

try:
    import pyrealsense2 as rs  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    rs = None

# This module is imported by depth testing .py files.  It provides two “grabber” threads:
#   • capture_camera   – for generic UVC/-v4l2 webcams
#   • capture_realsense – for Intel RealSense D435/D455 (left/right IR)
# plus thread-safe helpers to fetch frames and calibration.

# ───────────────────────── shared state ─────────────────────────
_lock = threading.Lock()  # guards every read/write to _frame1/_frame2
_frame1: Optional[np.ndarray] = None  # latest LEFT  frame (NumPy uint8, Gray)
_frame2: Optional[np.ndarray] = None  # latest RIGHT frame (NumPy uint8, Gray)
_stop_flag = False  # set → all threads exit cleanly
_ready = threading.Event()  # set once fx + baseline are known

# stereo intrinsics (populated exactly once by RealSense thread)
_fx: Optional[float] = None  # focal length in **pixels**
_baseline_m: Optional[float] = None  # baseline in **metres**

def capture_camera(cam_id: int, is_left: bool) -> None:
    """Grab grayscale frames from a USB/RGB camera and store them in _frame1/2."""
    global _frame1, _frame2, _stop_flag
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"[CameraCapture] Cannot open camera {cam_id}")
        _stop_flag = True
        return

    while not _stop_flag:
        ok, frame = cap.read()
        if not ok:
            print(f"[CameraCapture] Lost camera {cam_id}")
            _stop_flag = True
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        with _lock:
            if is_left:
                _frame1 = gray
            else:
                _frame2 = gray
    cap.release()


def _rs_device_available(timeout_s: int = 2) -> bool:
    """Return True if any RealSense enumerates on USB-3 within timeout."""
    if rs is None:
        return False
    ctx, t0 = rs.context(), time.time()
    while time.time() - t0 < timeout_s:
        if any(dev.supports(rs.camera_info.product_line) for dev in ctx.query_devices()):
            return True
        time.sleep(0.1)
    return False


def capture_realsense() -> None:
    """Continuously grab Infra 1 and Infra 2 grayscale frames from RealSense."""
    global _frame1, _frame2, _fx, _baseline_m, _stop_flag

    if rs is None:
        print("[CameraCapture] pyrealsense2 is not installed → cannot use RealSense")
        _stop_flag = True
        return

    if not _rs_device_available():
        print("[CameraCapture] No RealSense found on USB-3 → giving up")
        _stop_flag = True
        return

    while not _stop_flag:
        pipeline, cfg = rs.pipeline(), rs.config()
        cfg.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
        cfg.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)

        try:
            prof = pipeline.start(cfg)

            if _fx is None:
                intr = (
                    prof.get_stream(rs.stream.infrared, 1)
                    .as_video_stream_profile()
                    .get_intrinsics()
                )
                extr = prof.get_stream(rs.stream.infrared, 1).get_extrinsics_to(
                    prof.get_stream(rs.stream.infrared, 2)
                )
                _fx = intr.fx
                _baseline_m = abs(extr.translation[0])
                _ready.set()

            while not _stop_flag:
                if not pipeline.poll_for_frames():
                    time.sleep(0.002)
                    continue
                frames = pipeline.wait_for_frames(timeout_ms=100)
                ir_l = frames.get_infrared_frame(1)
                ir_r = frames.get_infrared_frame(2)
                if not ir_l or not ir_r:
                    continue
                with _lock:
                    _frame1 = np.asanyarray(ir_l.get_data())
                    _frame2 = np.asanyarray(ir_r.get_data())

        except Exception as e:  # pragma: no cover - hardware error path
            print(f"[CameraCapture] RealSense error → {e}")
            pipeline.stop()
            time.sleep(0.5)
        else:
            pipeline.stop()


def get_frames() -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Thread-safe access to the latest stereo pair."""
    with _lock:
        left = _frame1.copy() if _frame1 is not None else None
        right = _frame2.copy() if _frame2 is not None else None
    return left, right


def get_calibration() -> tuple[float, float]:
    """Block until calibration is known, then return (fx_px, baseline_m)."""
    _ready.wait()
    if _fx is None or _baseline_m is None:
        raise RuntimeError("RealSense calibration unavailable")
    return _fx, _baseline_m


def signal_stop() -> None:
    """Ask all capture threads to terminate gracefully."""
    global _stop_flag
    _stop_flag = True


def reset_state() -> None:
    """Reset cached frames/calibration. Useful between successive runs."""
    global _frame1, _frame2, _stop_flag, _fx, _baseline_m
    with _lock:
        _frame1 = None
        _frame2 = None
    _stop_flag = False
    _fx = None
    _baseline_m = None
    _ready.clear()


def query_usb_camera_ids(max_id: int = 10, timeout_s: float = 1.0) -> List[int]:
    """Scan /dev/video devices and return usable camera IDs."""
    good: List[int] = []
    for cam_id in range(max_id):
        cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
        if not cap.isOpened():
            continue

        t0 = time.time()
        while time.time() - t0 < timeout_s:
            ok, _ = cap.read()
            if ok:
                good.append(cam_id)
                break
        cap.release()
    return good


if __name__ == "__main__":  # pragma: no cover - manual utility
    ids = query_usb_camera_ids(max_id=8)
    print("Detected USB cameras:", ids)
