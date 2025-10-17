from __future__ import annotations

import contextlib
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, List

import cv2
import numpy as np

from main import DEFAULT_FPS, DEFAULT_HEIGHT, DEFAULT_WIDTH
from pose_augmentation import (
    get_pose_model_info, lift_keypoints_to_3d,
    depth_guided_nms, get_skeleton_edges,
    repair_occluded_joints, OcclusionRepairConfig,
    draw_skeletons
)
from pose_backends import PoseBackend, build_pose_backend
try:
    import camera_capture as cam
except Exception as exc:  # pragma: no cover - during unit tests CameraCapture may not import
    cam = None
    _CAM_IMPORT_ERROR = exc
else:  # pragma: no cover - exercised when camera_capture imports successfully
    _CAM_IMPORT_ERROR = None

class TensorRTUnavailableError(RuntimeError):
    """Raised when TensorRT or PyCUDA are missing from the environment."""

_TRT_MODULE = None
_CUDA_MODULE = None
_CUDA_AUTOINIT = None

def ensure_tensorrt_runtime():
    """Import TensorRT + PyCUDA exactly once and cache the modules."""
    global _TRT_MODULE, _CUDA_MODULE, _CUDA_AUTOINIT

    if _TRT_MODULE is not None and _CUDA_MODULE is not None:
        return _TRT_MODULE, _CUDA_MODULE

    try:
        import tensorrt as trt  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on environment
        raise TensorRTUnavailableError(
            "TensorRT Python bindings are not available. Install TensorRT before "
            "running local inference."
        ) from exc

    try:
        import pycuda.autoinit  # type: ignore  # noqa: F401  - side-effect import
        import pycuda.driver as cuda  # type: ignore
    except Exception as exc:
        raise TensorRTUnavailableError(
            "PyCUDA is required for TensorRT execution. Install pycuda before "
            "running local inference."
        ) from exc

    _TRT_MODULE, _CUDA_MODULE, _CUDA_AUTOINIT = trt, cuda, pycuda.autoinit
    return trt, cuda

@contextlib.contextmanager
def push_cuda_context():
    """Context manager that pushes the PyCUDA context if available."""
    ensure_tensorrt_runtime()
    ctx = getattr(_CUDA_AUTOINIT, "context", None)
    if ctx is None:
        yield None
        return

    push = getattr(ctx, "push", None)
    pop = getattr(ctx, "pop", None)
    if push is None or pop is None:
        yield ctx
        return

    push()
    try:
        yield ctx
    finally:
        pop()

def disparity_to_png16(disp: np.ndarray, max_disp: float, scale: float) -> bytes:
    d = np.nan_to_num(disp, nan=0.0, posinf=max_disp, neginf=0.0)
    d = np.clip(d, 0.0, max_disp)
    scaled = np.round(d * scale)
    d16 = np.ascontiguousarray(np.clip(scaled, 0.0, 65535).astype(np.uint16))
    ok, buf = cv2.imencode(".png", d16)
    if not ok:
        raise RuntimeError("cv2.imencode PNG failed")
    return buf.tobytes()

@dataclass
class _Binding:
    name: str
    dtype: np.dtype
    shape: Tuple[int, ...]
    host_mem: Optional[np.ndarray]
    device_mem: Optional[Any]

class TensorRTPipeline:
    """Small helper around TensorRT execution based on the live TRT demo script."""

    def __init__(self, engine_path: str):
        trt, cuda = ensure_tensorrt_runtime()
        self.trt = trt
        self.cuda = cuda
        self.logger = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, "rb") as f:
            engine_bytes = f.read()

        runtime = trt.Runtime(self.logger)
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        if engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")

        self.engine = engine
        self.context = engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        self.stream = cuda.Stream()
        self.input_left, self.input_right, self.output = self._collect_tensor_names()
        self.bindings = self._allocate_buffers()

    def _collect_tensor_names(self) -> Tuple[str, str, str]:
        engine = self.engine
        trt = self.trt
        inputs, outputs = [], []
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                inputs.append(name)
            else:
                outputs.append(name)
        if len(inputs) < 2 or not outputs:
            raise RuntimeError("Engine must expose at least two inputs and one output")
        return inputs[0], inputs[1], outputs[0]

    def _allocate_buffers(self) -> Dict[str, _Binding]:
        engine = self.engine
        cuda = self.cuda
        context = self.context
        bindings: Dict[str, _Binding] = {}
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            dtype = np.dtype(self.trt.nptype(engine.get_tensor_dtype(name)))
            shape = [1 if d == -1 else d for d in engine.get_tensor_shape(name)]
            if engine.get_tensor_mode(name) == self.trt.TensorIOMode.INPUT:
                context.set_input_shape(name, shape)
            size = int(np.prod(shape)) * dtype.itemsize
            device_mem = cuda.mem_alloc(size)
            context.set_tensor_address(name, int(device_mem))
            host_mem = (
                cuda.pagelocked_empty(shape, dtype)
                if engine.get_tensor_mode(name) == self.trt.TensorIOMode.OUTPUT
                else None
            )
            bindings[name] = _Binding(name, dtype, tuple(shape), host_mem, device_mem)
        return bindings

    def infer(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
        left = np.ascontiguousarray(left_bgr.transpose(2, 0, 1)[None]).astype(np.float32)
        right = np.ascontiguousarray(right_bgr.transpose(2, 0, 1)[None]).astype(np.float32)
        cuda = self.cuda
        stream = self.stream

        cuda.memcpy_htod_async(self.bindings[self.input_left].device_mem, left, stream)
        cuda.memcpy_htod_async(self.bindings[self.input_right].device_mem, right, stream)

        self.context.execute_async_v3(stream.handle)

        out_binding = self.bindings[self.output]
        assert out_binding.host_mem is not None
        cuda.memcpy_dtoh_async(out_binding.host_mem, out_binding.device_mem, stream)
        stream.synchronize()

        return np.array(out_binding.host_mem, copy=True).squeeze()

def _normalize_for_display(disp: np.ndarray) -> np.ndarray:
    disp_min, disp_max = float(np.min(disp)), float(np.max(disp))
    if disp_max <= disp_min:
        return np.zeros_like(disp, dtype=np.uint8)
    disp_u8 = ((disp - disp_min) / (disp_max - disp_min) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(disp_u8, cv2.COLORMAP_TURBO)

class LocalEngineRunner:
    """Run inference locally using a TensorRT engine and camera capture threads."""

    def __init__(
            self,
            engine_path: Optional[str],
            left_src: str,
            right_src: str,
            mode: str,
            frame_width: int,
            frame_height: int,
            session_id: str,
            fps: float = DEFAULT_FPS,
            save_dir: Optional[Path] = None,
            preview: bool = False,
            max_disp: float = 256.0,
            disp_scale: float = 256.0,
            on_log: Optional[Callable[[str], None]] = None,
            on_result: Optional[
                Callable[[int, str, str, int, int, bytes, Dict[str, Any]], None]
            ] = None,
            on_start: Optional[Callable[[], None]] = None,
            on_finish: Optional[Callable[[], None]] = None,
            use_realsense: Optional[bool] = None,
            fx_px: Optional[float] = None,
            baseline_m: Optional[float] = None,
            pose_enabled: bool = False,
            pose_model: Optional[str] = None,
            depth_enabled: bool = True
    ) -> None:
        if mode not in {"stream", "file"}:
            raise ValueError("mode must be either 'stream' or 'file'")

        self.engine_path = engine_path
        self.left_src = left_src
        self.right_src = right_src
        self.mode = mode
        self.frame_width = int(frame_width) if frame_width else DEFAULT_WIDTH
        self.frame_height = int(frame_height) if frame_height else DEFAULT_HEIGHT
        self.fps = float(fps) if fps else DEFAULT_FPS
        self.session_id = session_id
        self.save_dir = save_dir
        self.preview = preview
        self.max_disp = max_disp
        self.disp_scale = disp_scale
        self._manual_fx_px = fx_px
        self._manual_baseline_m = baseline_m
        self.pose_enabled = bool(pose_enabled)
        self.pose_model = pose_model
        self.depth_enabled = bool(depth_enabled)
        if self.depth_enabled and not self.engine_path:
            raise ValueError("engine_path is required when depth inference is enabled")
        self.on_log = on_log or (lambda s: None)
        self.on_result = on_result or (lambda *args: None)
        self.on_start = on_start or (lambda: None)
        self.on_finish = on_finish or (lambda: None)

        self._stop_event = threading.Event()
        self._capture_started = False
        self._use_realsense = self._determine_realsense_flag(use_realsense)
        self._save_dir_prepared = False
        self._rs_fx_px: Optional[float] = None
        self._rs_baseline_m: Optional[float] = None
        self._log_pose_configuration()
        self._pose_runner: Optional[PoseBackend] = None
        self._resolved_pose_key: Optional[str] = (
            str(self.pose_model).lower() if self.pose_model else None
        )
        self._last_pose_meta: List[Dict[str, Any]] = []
        self._latest_preview_bgr: Optional[np.ndarray] = None

    def _determine_realsense_flag(self, override: Optional[bool]) -> bool:
        if override is not None:
            return bool(override)
        if self.mode != "stream":
            return False
        tokens = {str(self.left_src).strip().lower(), str(self.right_src).strip().lower()}
        realsense_tokens = {"rs", "realsense", "d435", "d455"}
        return bool(tokens & realsense_tokens)

    def stop(self) -> None:
        self._stop_event.set()
        self._stop_capture_threads()

    def _log(self, msg: str) -> None:
        try:
            self.on_log(msg)
        except Exception:
            print(msg)

    def _pose_metadata(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"enabled": bool(self.pose_enabled)}
        if not self.pose_enabled:
            return data

        model_key = self._resolved_pose_key or (
            str(self.pose_model).lower() if self.pose_model else None
        )
        if model_key is not None:
            data["model"] = model_key
        model_info = get_pose_model_info(model_key)
        if model_info is not None:
            data["model_display"] = model_info.display_name
            data["model_description"] = model_info.description

        return data

    def _log_pose_configuration(self) -> None:
        if not self.pose_enabled:
            self._log("[pose] Depth-guided pose augmentation disabled.")
            return

        model_info = get_pose_model_info(self.pose_model)
        if model_info is not None:
            self._log(
                f"[pose] Using {model_info.display_name} pose model ({model_info.key})."
            )
        elif self.pose_model:
            self._log(f"[pose] Pose model '{self.pose_model}' is not recognised.")
        else:
            self._log("[pose] Pose estimation enabled without an explicit model key.")

    def _prepare_engine(self) -> TensorRTPipeline:
        if not self.engine_path:
            raise RuntimeError("Depth inference disabled; no TensorRT engine configured")
        self._log(f"[local] Loading TensorRT engine: {self.engine_path}")
        pipeline = TensorRTPipeline(self.engine_path)
        self._log("[local] TensorRT engine ready.")
        return pipeline

    def _start_camera_threads(self) -> None:
        if cam is None:
            raise RuntimeError(
                "CameraCapture module failed to import" if _CAM_IMPORT_ERROR is None else str(_CAM_IMPORT_ERROR)
            )
        cam.reset_state()
        if self._use_realsense:
            threading.Thread(target=cam.capture_realsense, daemon=True).start()
            self._log("[local] Started RealSense capture thread")
        else:
            try:
                left_idx = int(self.left_src)
                right_idx = int(self.right_src)
            except ValueError as exc:
                raise RuntimeError("Camera indices must be integers when not using RealSense") from exc
            threading.Thread(target=cam.capture_camera, args=(left_idx, True), daemon=True).start()
            threading.Thread(target=cam.capture_camera, args=(right_idx, False), daemon=True).start()
            self._log(
                f"[local] Started USB capture threads (left={left_idx}, right={right_idx})"
            )
        self._capture_started = True

    def _stream_from_cameras(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        assert cam is not None
        self._start_camera_threads()
        if self._use_realsense:
            try:
                fx_px, baseline_m = cam.get_calibration()
                self._rs_fx_px, self._rs_baseline_m = fx_px, baseline_m
                self._log(f"[local] RealSense calibration: fx={fx_px:.1f} baseline={baseline_m:.4f}m")
            except Exception as exc:
                self._log(f"[local] Failed to fetch RealSense calibration: {exc}")

        while not self._stop_event.is_set():
            left, right = cam.get_frames()
            if left is None or right is None:
                if self._stop_event.wait(timeout=0.005):
                    break
                continue
            yield left, right

    def _stream_from_files(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        cap_left = cv2.VideoCapture(self.left_src)
        cap_right = cv2.VideoCapture(self.right_src)
        if not (cap_left.isOpened() and cap_right.isOpened()):
            with contextlib.suppress(Exception):
                cap_left.release()
                cap_right.release()
            raise RuntimeError("Failed to open one or both video files")
        try:
            while not self._stop_event.is_set():
                ok_l, frame_l = cap_left.read()
                ok_r, frame_r = cap_right.read()
                if not ok_l or not ok_r or frame_l is None or frame_r is None:
                    self._log("[local] End of video files.")
                    break
                yield frame_l, frame_r
        finally:
            with contextlib.suppress(Exception):
                cap_left.release()
            with contextlib.suppress(Exception):
                cap_right.release()

    def _frame_stream(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        if self.mode == "stream":
            yield from self._stream_from_cameras()
        else:
            yield from self._stream_from_files()

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame.shape[1] == self.frame_width and frame.shape[0] == self.frame_height:
            return frame
        return cv2.resize(frame, (self.frame_width, self.frame_height), interpolation=cv2.INTER_AREA)

    def _prepare_pair(self, left: np.ndarray, right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        left_r = self._resize_frame(left)
        right_r = self._resize_frame(right)

        if left_r.ndim == 2:
            left_bgr = cv2.cvtColor(left_r, cv2.COLOR_GRAY2BGR)
        else:
            left_bgr = left_r

        if right_r.ndim == 2:
            right_bgr = cv2.cvtColor(right_r, cv2.COLOR_GRAY2BGR)
        else:
            right_bgr = right_r

        return left_bgr, right_bgr

    def _realsense_preview_frame(self) -> Optional[np.ndarray]:
        if not self._use_realsense or cam is None:
            return None

        preview = cam.get_preview_frame()
        if preview is None:
            return None

        preview_r = self._resize_frame(preview)
        if preview_r.ndim == 2:
            return cv2.cvtColor(preview_r, cv2.COLOR_GRAY2BGR)
        if preview_r.ndim == 3:
            if preview_r.shape[2] == 3:
                return preview_r
            if preview_r.shape[2] > 3:
                return preview_r[:, :, :3]
        return None

    def _select_preview_frame(self, left_bgr: np.ndarray) -> np.ndarray:
        """Choose the RGB frame that should be used for pose estimation.

        If a RealSense preview frame is available we favour that, otherwise we
        fall back to the rectified left image. The selected frame is cached so
        that subsequent consumers (e.g. pose backends) always receive a valid
        array even when the RealSense preview momentarily drops a frame.
        """

        preview = self._realsense_preview_frame()

        if preview is None:
            if self._use_realsense and self._latest_preview_bgr is not None:
                preview = self._latest_preview_bgr
            else:
                preview = left_bgr

        self._latest_preview_bgr = preview
        return preview

    def _encode_disparity(self, disp: np.ndarray) -> bytes:
        try:
            return disparity_to_png16(disp, self.max_disp, self.disp_scale)
        except Exception as exc:
            self._log(f"[local] Failed to encode disparity: {exc}")
            raise

    def _common_meta(self) -> Dict[str, Any]:
        meta: Dict[str, Any] = {
            "session_id": self.session_id,
            "source_mode": self.mode,
            "sender_wh": [int(self.frame_width), int(self.frame_height)],
            "realsense": bool(self._use_realsense),
            "pose": self._pose_metadata(),
            "poses": getattr(self, "_last_pose_meta", []),
        }
        if (
            self._use_realsense
            and self._rs_fx_px is not None
            and self._rs_baseline_m is not None
        ):
            meta["fx_px"] = float(self._rs_fx_px)
            meta["baseline_m"] = float(self._rs_baseline_m)
        return meta

    def _emit_result(self, seq: int, png16: bytes) -> None:
        meta = self._common_meta()
        meta["disp_scale"] = float(self.disp_scale)
        meta["max_disp"] = float(self.max_disp)
        self.on_result(
            seq,
            "disparity",
            "png16",
            int(self.frame_width),
            int(self.frame_height),
            png16,
            meta,
        )

    def _emit_rgb_preview(self, seq: int, image_bgr: np.ndarray) -> None:
        meta = self._common_meta()
        preview = np.ascontiguousarray(image_bgr)
        ok, buf = cv2.imencode(".jpg", preview)
        if not ok:
            self._log("[local] Failed to encode RGB preview frame.")
            return
        self.on_result(
            seq,
            "rgb_preview",
            "jpeg",
            int(self.frame_width),
            int(self.frame_height),
            buf.tobytes(),
            meta,
        )

    def _ensure_save_dir(self) -> None:
        if self.save_dir is None or self._save_dir_prepared:
            return
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._save_dir_prepared = True

    def _save_result(self, seq: int, png16: bytes) -> None:
        if self.save_dir is None:
            return
        self._ensure_save_dir()
        fname = self.save_dir / f"disparity_seq{seq:06d}.png"
        with open(fname, "wb") as f:
            f.write(png16)

    def _apply_pose_overlay(
        self, image_bgr: np.ndarray, poses_uvc: Optional[List[np.ndarray]]
    ) -> np.ndarray:
        if not poses_uvc:
            return image_bgr
        try:
            return draw_skeletons(
                image_bgr,
                [np.asarray(kp, dtype=np.float32) for kp in poses_uvc],
                self._resolved_pose_key or self.pose_model,
            )
        except Exception as exc:
            self._log(f"[pose] Failed to draw preview skeletons: {exc}")
            return image_bgr

    def _render_preview(
        self,
        base_bgr: np.ndarray,
        disp: Optional[np.ndarray],
        fps: float,
    ) -> bool:
        if not self.preview:
            return False
        base = base_bgr

        if disp is not None:
            disp_color = _normalize_for_display(disp)
            combo = np.hstack((base, disp_color))
        else:
            combo = base
        cv2.putText(
            combo,
            f"{fps:.1f} FPS",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )
        cv2.imshow("FoundationStereo TensorRT", combo)
        if cv2.waitKey(1) & 0xFF == 27:
            self._log("[local] ESC pressed – stopping preview")
            return True
        return False

    def _respect_frame_rate(self, next_deadline: float, interval: float) -> float:
        next_deadline += interval
        sleep_s = next_deadline - time.perf_counter()
        if sleep_s > 0 and self._stop_event.wait(timeout=sleep_s):
            return next_deadline
        return next_deadline

    def _current_pose_intrinsics(self) -> Optional[Tuple[float, float, float, float]]:
        fx = self._rs_fx_px if self._rs_fx_px is not None else self._manual_fx_px
        bl = self._rs_baseline_m if self._rs_baseline_m is not None else self._manual_baseline_m
        if fx is None or bl is None:
            return None
        fy = fx
        cx = self.frame_width / 2.0
        cy = self.frame_height / 2.0
        return float(fx), float(fy), float(cx), float(cy)

    def _stop_capture_threads(self) -> None:
        if self._capture_started and cam is not None:
            cam.signal_stop()

    def _cleanup_after_run(self) -> None:
        if self.preview:
            cv2.destroyAllWindows()
        self._stop_capture_threads()

    def run(self) -> None:
        self._stop_event.clear()
        ctx_mgr = push_cuda_context() if self.depth_enabled else contextlib.nullcontext()
        with ctx_mgr:
            pipeline = self._prepare_engine() if self.depth_enabled else None
            self.on_start()
            seq = 0
            frame_interval = (1.0 / self.fps) if (self.fps and self.fps > 0) else 0.0
            next_deadline = time.perf_counter()
            try:
                for left_raw, right_raw in self._frame_stream():
                    if self._stop_event.is_set():
                        break
                    left_bgr, right_bgr = self._prepare_pair(left_raw, right_raw)
        preview_source = self._select_preview_frame(left_bgr)

                    start = time.perf_counter()
                    disp: Optional[np.ndarray]
                    depth_m: Optional[np.ndarray]
                    if pipeline is not None:
                        disp = pipeline.infer(left_bgr, right_bgr)
                        depth_m = self._depth_from_disparity(disp)
                    else:
                        disp = None
                        depth_m = None

                    # Pose estimation & depth-aware late fusion
                    poses_uvc: List[np.ndarray] = []
                    pose_scores: List[float] = []
                    if self.pose_enabled:
                        if self._pose_runner is None:
                            self._pose_runner = self._init_pose_backend()
                        if self._pose_runner is not None:
                            try:
                                poses_uvc, pose_scores = self._pose_runner(preview_source, depth_m)
                            except Exception as exc:
                                self._log(f"[pose] backend failed: {exc}")
                    if poses_uvc and depth_m is not None:
                        edges = get_skeleton_edges(self.pose_model or "coco17")
                        keep_idx = depth_guided_nms(
                            poses_uvc, pose_scores, depth_m, edges, iou_threshold=0.5
                        )
                        poses_uvc = [poses_uvc[i] for i in keep_idx]
                        pose_scores = [pose_scores[i] for i in keep_idx]

                        repaired_poses = []
                        for kp in poses_uvc:
                            arr = np.asarray(kp, dtype=np.float32)
                            uv = arr[:, :2]
                            conf = (
                                arr[:, 2]
                                if arr.shape[1] > 2
                                else np.ones((arr.shape[0],), dtype=np.float32)
                            )

                            uv_fixed, conf_fixed = repair_occluded_joints(
                                uv,
                                conf,
                                depth_m,
                                edges,
                                # OcclusionRepairConfig(confidence_threshold=0.2, search_radius=20.0, steps=10)
                            )

                            fixed = np.column_stack([uv_fixed, conf_fixed]).astype(np.float32)
                            repaired_poses.append(fixed)

                        poses_uvc = repaired_poses

                    pose_meta: List[Dict[str, Any]] = []
                    intrinsics = (
                        self._current_pose_intrinsics() if depth_m is not None else None
                    )
                    for kps, sc in zip(poses_uvc, pose_scores):
                        entry: Dict[str, Any] = {
                            "score": float(sc),
                            "keypoints_uvc": np.asarray(kps, dtype=np.float32).tolist(),
                        }
                        if intrinsics is not None and depth_m is not None:
                            xyz = lift_keypoints_to_3d(
                                np.asarray(kps, dtype=np.float32)[:, :2],
                                depth_m,
                                intrinsics,
                            )
                            entry["keypoints_xyz"] = np.asarray(xyz, dtype=np.float32).tolist()
                        pose_meta.append(entry)
                    self._last_pose_meta = pose_meta

                    if self.pose_enabled:
                        if pose_scores:
                            avg_score = float(np.mean(pose_scores))
                            best_score = float(np.max(pose_scores))
                            self._log(
                                "[pose] seq=%d detections=%d avg_score=%.3f best_score=%.3f"
                                % (seq, len(pose_scores), avg_score, best_score)
                            )
                        else:
                            self._log(f"[pose] seq={seq} detections=0")

                    if self.pose_enabled:
                        if pose_scores:
                            avg_score = float(np.mean(pose_scores))
                            best_score = float(np.max(pose_scores))
                            self._log(
                                "[pose] seq=%d detections=%d avg_score=%.3f best_score=%.3f"
                                % (seq, len(pose_scores), avg_score, best_score)
                            )
                        else:
                            self._log(f"[pose] seq={seq} detections=0")

                    fps = 1.0 / max(time.perf_counter() - start, 1e-6)

                    preview_base = self._realsense_preview_frame() or left_bgr

                    if disp is not None:
                        png16 = self._encode_disparity(disp)
                        self._emit_result(seq, png16)
                        self._save_result(seq, png16)
                    else:
                        self._emit_rgb_preview(seq, preview_base)

                    preview_poses = poses_uvc if self.pose_enabled else None
                    preview_frame = self._apply_pose_overlay(preview_base, preview_poses)

                    if disp is None:
                        self._emit_rgb_preview(seq, preview_frame)

                    if self._render_preview(preview_frame, disp, fps):
                        break

                    seq += 1
                    if frame_interval > 0.0:
                        next_deadline = self._respect_frame_rate(next_deadline, frame_interval)
                        if self._stop_event.is_set():
                            break

            finally:
                self._cleanup_after_run()
                self.on_finish()

    def _init_pose_backend(self) -> Optional[PoseBackend]:
        """Initialise the configured RGB-D aware pose backend."""
        try:
            backend, resolved_key = build_pose_backend(self.pose_model, self._log)
        except Exception as exc:  # pragma: no cover - defensive logging
            self._log(f"[pose] Failed to initialise pose backend: {exc}")
            return None
        if backend is None:
            self._log("[pose] No pose backend could be initialised.")
            return None

        self._pose_runner = backend
        self._resolved_pose_key = resolved_key

        if resolved_key is not None:
            info = get_pose_model_info(resolved_key)
            if info is not None:
                self._log(f"[pose] {info.display_name} backend ready ({info.key}).")
            else:
                self._log(f"[pose] Pose backend ready for key '{resolved_key}'.")

        return backend

    def _depth_from_disparity(self, disparity: np.ndarray) -> Optional[np.ndarray]:
        fx = self._rs_fx_px if self._rs_fx_px is not None else self._manual_fx_px
        bl = self._rs_baseline_m if self._rs_baseline_m is not None else self._manual_baseline_m

        if fx is None or bl is None:
            return None
        d = np.asarray(disparity, dtype=np.float32)
        eps = 1e-6
        denom = np.maximum(d, eps)
        depth = (float(fx) * float(bl)) / denom
        invalid = ~np.isfinite(d) | (d <= eps)
        depth[invalid] = np.nan
        return depth