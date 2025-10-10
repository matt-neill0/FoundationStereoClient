from __future__ import annotations
import contextlib, pathlib, threading, time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import cv2
import numpy as np

from main import DEFAULT_FPS

from sender_core import (
    open_cam,
    read_next_pair_from_caps,
    read_next_pair_from_files
)

class TensorRTUnavailableError(RuntimeError):
    pass

_TRT_MODULE = None
_CUDA_MODULE = None
_CUDA_AUTOINIT = None

def ensure_tensorrt_runtime():
    global _TRT_MODULE, _CUDA_MODULE, _CUDA_AUTOINIT

    if _TRT_MODULE is not None and _CUDA_MODULE is not None:
        return _TRT_MODULE, _CUDA_MODULE

    try:
        import tensorrt as trt
    except Exception:
        raise TensorRTUnavailableError(
            "TensorRT Python bindings are not available. Install TensorRT before "
            "running local inference."
        ) from Exception

    try:
        import pycuda.autoinit
        import pycuda.driver as cuda
    except Exception:
        raise TensorRTUnavailableError(
            "PyCUDA is required for TensorRT execution. Install pycuda before "
            "running local inference."
        ) from Exception

    _TRT_MODULE, _CUDA_MODULE, _CUDA_AUTOINIT = trt, cuda, pycuda.autoinit
    return trt, cuda

@contextlib.contextmanager
def push_cuda_context():
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


def align32(x:int) -> int:
    return (x + 31) // 32 * 32

def letterbox_resize(img: np.ndarray, dst_hw: Tuple[int, int]) -> np.ndarray:
    H, W = dst_hw
    h0, w0 = img.shape[:2]
    r = min(W / w0, H / h0)
    nw, nh = int(round(w0 * r)), int(round(h0 * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((H, W, 3), dtype=resized.dtype)
    y0 = (H - nh) // 2
    x0 = (W - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas

def preprocess_rgb(image_bgr: np.ndarray) -> np.ndarray:
    x = image_bgr[:, :, ::-1].astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None, ...]
    return x

def disparity_to_png16(disp: np.ndarray, max_disp: float, scale: float) -> bytes:
    d = np.nan_to_num(disp, nan=0.0, posinf=max_disp, neginf=0.0)
    d = np.clip(d, 0.0, max_disp)
    scaled = np.round(d * scale)
    if scaled.dtype != np.float32 and scaled.dtype != np.float64:
        scaled = scaled.astype(np.float32, copy=False)
    scaled = np.clip(scaled, 0.0, np.float32(np.iinfo(np.uint16).max))
    d16 = np.ascontiguousarray(scaled.astype(np.uint16))
    ok, buf = cv2.imencode(".png", d16)
    if not ok:
        raise RuntimeError("cv2.imencode PNG failed")
    return buf.tobytes()

@dataclass
class _Binding:
    name: str
    index: int
    is_input: bool
    dtype: np.dtype
    shape: Tuple[int, ...]
    host_mem: Optional[np.ndarray] = None
    device_mem: Optional[Any] = None

class TensorRTEngine:
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
        if engine.num_optimization_profiles > 0:
            if hasattr(self.context, "set_optimization_profile_async"):
                ok = self.context.set_optimization_profile_async(0, self.stream.handle)
                if not ok:
                    raise RuntimeError("Failed to set TensorRT optimization profile 0")
            else:
                prop = getattr(type(self.context), "active_optimization_profile", None)
                if isinstance(prop, property) and prop.fset is not None:
                    self.context.active_optimization_profile = 0
                else:
                    raise RuntimeError(
                        "TensorRT context does not support selecting optimization profile"
                    )

        self._legacy_bindings = hasattr(engine, "num_bindings")
        self._bindings: List[_Binding] = []
        if self._legacy_bindings:
            for idx in range(engine.num_bindings):
                name = engine.get_binding_name(idx)
                dtype = np.dtype(trt.nptype(engine.get_binding_dtype(idx)))
                shape = tuple(engine.get_binding_shape(idx))
                binding = _Binding(
                    name=name,
                    index=idx,
                    is_input=engine.binding_is_input(idx),
                    dtype=dtype,
                    shape=shape,
                )
                self._bindings.append(binding)
        else:
            if not hasattr(engine, "num_io_tensors"):
                raise RuntimeError(
                    "TensorRT engine does not expose binding enumeration APIs"
                )

            tensor_mode = getattr(trt, "TensorIOMode", None)
            if tensor_mode is None:
                raise RuntimeError(
                    "TensorRT runtime is missing TensorIOMode; cannot classify IO tensors"
                )

            for idx in range(engine.num_io_tensors):
                name = engine.get_tensor_name(idx)
                mode = engine.get_tensor_mode(name)
                dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))
                shape = tuple(engine.get_tensor_shape(name))
                binding = _Binding(
                    name=name,
                    index=idx,
                    is_input=(mode == tensor_mode.INPUT),
                    dtype=dtype,
                    shape=shape,
                )
                self._bindings.append(binding)

    @property
    def inputs(self) -> Iterable[_Binding]:
        return (b for b in self._bindings if b.is_input)

    @property
    def outputs(self) -> Iterable[_Binding]:
        return (b for b in self._bindings if not b.is_input)

    def _set_input_shape(self, binding: _Binding, shape: Tuple[int, ...]):
        if not binding.is_input:
            return

        if self._legacy_bindings:
            self.context.set_binding_shape(binding.index, shape)
            binding.shape = tuple(self.context.get_binding_shape(binding.index))
        else:
            set_shape = getattr(self.context, "set_input_shape", None)
            if set_shape is None:
                raise RuntimeError("TensorRT context lacks set_input_shape for tensor API")
            set_shape(binding.name, shape)
            binding.shape = tuple(self.context.get_tensor_shape(binding.name))

    def _ensure_allocation(self, binding: _Binding):
        if self._legacy_bindings:
            shape = tuple(self.context.get_binding_shape(binding.index))
        else:
            shape = tuple(self.context.get_tensor_shape(binding.name))
        size = int(np.prod(shape))
        if size <= 0:
            raise RuntimeError(f"Invalid binding shape for {binding.name!r}: {shape}")

        needs_alloc = (
            binding.host_mem is None
            or binding.host_mem.size != size
            or binding.host_mem.dtype != binding.dtype
        )

        if needs_alloc:
            host_mem = self.cuda.pagelocked_empty(size, binding.dtype)
            device_mem = self.cuda.mem_alloc(host_mem.nbytes)
            binding.host_mem = host_mem
            binding.device_mem = device_mem

    def infer(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        inputs = list(self.inputs)
        outputs = list(self.outputs)
        if len(inputs) < 2 or len(outputs) == 0:
            raise RuntimeError("Engine must have at least two inputs and one output")

        left_binding, right_binding = inputs[:2]
        out_binding = outputs[0]

        self._set_input_shape(left_binding, tuple(int(x) for x in left.shape))
        self._set_input_shape(right_binding, tuple(int(x) for x in right.shape))

        if self._legacy_bindings:
            out_shape = tuple(self.context.get_binding_shape(out_binding.index))
        else:
            out_shape = tuple(self.context.get_tensor_shape(out_binding.name))

        self._ensure_allocation(left_binding)
        self._ensure_allocation(right_binding)
        self._ensure_allocation(out_binding)

        np.copyto(left_binding.host_mem, np.asarray(left, dtype=left_binding.dtype).ravel())
        np.copyto(right_binding.host_mem, np.asarray(right, dtype=right_binding.dtype).ravel())

        if self._legacy_bindings:
            bindings: List[int] = [0] * self.engine.num_bindings
            bindings[left_binding.index] = int(left_binding.device_mem)
            bindings[right_binding.index] = int(right_binding.device_mem)
            bindings[out_binding.index] = int(out_binding.device_mem)

            self.cuda.memcpy_htod_async(left_binding.device_mem, left_binding.host_mem, self.stream)
            self.cuda.memcpy_htod_async(right_binding.device_mem, right_binding.host_mem, self.stream)

            self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.handle)

        else:
            set_addr = getattr(self.context, "set_tensor_address", None)
            enqueue = None
            for name in (
                    "enqueue_async_v3",
                    "enqueue_v3",
                    "enqueueV3",
                    "execute_async_v3",
                    "execute_v3",
                    "executeV3",
            ):
                enqueue = getattr(self.context, name, None)
                if enqueue is not None:
                    break

            if set_addr is None or enqueue is None:
                raise RuntimeError(
                    "TensorRT context lacks tensor API execution methods"
                )

            self.cuda.memcpy_htod_async(left_binding.device_mem, left_binding.host_mem, self.stream)
            self.cuda.memcpy_htod_async(right_binding.device_mem, right_binding.host_mem, self.stream)

            set_addr(left_binding.name, int(left_binding.device_mem))
            set_addr(right_binding.name, int(right_binding.device_mem))
            set_addr(out_binding.name, int(out_binding.device_mem))

            try:
                ok = enqueue(self.stream.handle)
            except TypeError:
                ok = enqueue()

            if ok is False:
                raise RuntimeError("TensorRT tensor API execution failed")

        self.cuda.memcpy_dtoh_async(out_binding.host_mem, out_binding.device_mem, self.stream)
        self.stream.synchronize()

        out_np = np.array(out_binding.host_mem).reshape(out_shape)
        if out_np.ndim == 3 and out_np.shape[0] == 1:
            out_np = out_np[0]
        if out_np.ndim == 3 and out_np.shape[-1] == 1:
            out_np = out_np[..., 0]
        return out_np

class LocalEngineRunner:
    def __init__(
            self,
            engine_path: str,
            left_src: str,
            right_src: str,
            mode: str,
            frame_width: int,
            frame_height: int,
            session_id: str,
            fps: float = DEFAULT_FPS,
            save_dir: Optional[pathlib.Path] = None,
            preview: bool = True,
            align_to_32: bool = True,
            letterbox: bool = False,
            max_disp: float = 256.0,
            disp_scale: float = 256.0,
            on_log: Optional[Callable[[str], None]] = None,
            on_result: Optional[
                Callable[[int, str, str, int, int, bytes, Dict[str, Any]], None]
            ] = None,
            on_start: Optional[Callable[[], None]] = None,
            on_finish: Optional[Callable[[], None]] = None,
    ):
        self.engine_path = engine_path
        self.left_src = left_src
        self.right_src = right_src
        self.mode = mode
        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.fps = float(fps) if fps else DEFAULT_FPS
        self.session_id = session_id
        self.save_dir = save_dir
        self.preview = preview
        self.align_to_32 = align_to_32
        self.letterbox = letterbox
        self.max_disp = max_disp
        self.disp_scale = disp_scale
        self.on_log = on_log or (lambda s: None)
        self.on_result = on_result or (lambda *args: None)
        self.on_start = on_start or (lambda: None)
        self.on_finish = on_finish or (lambda: None)

        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def _prepare_frames(
            self, left: np.ndarray, right: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int], Tuple[int, int]]:
        Hs, Ws = left.shape[:2]
        Ht, Wt = Hs, Ws
        if self.align_to_32:
            Ht, Wt = align32(Ht), align32(Wt)

        if (Ht, Wt) != (Hs, Ws):
            if self.letterbox:
                left = letterbox_resize(left, (Ht, Wt))
                right = letterbox_resize(right, (Ht, Wt))
            else:
                left = cv2.resize(left, (Wt, Ht), interpolation=cv2.INTER_AREA)
                right = cv2.resize(right, (Wt, Ht), interpolation=cv2.INTER_AREA)

        return preprocess_rgb(left), preprocess_rgb(right), (Ht, Wt), (Hs, Ws)

    def _open_sources(self):
        if self.mode == "stream":
            capL = open_cam(
                int(self.left_src),
                self.frame_height,
                self.frame_width,
                self.fps,
                self.on_log,
            )
            capR = open_cam(
                int(self.right_src),
                self.frame_height,
                self.frame_width,
                self.fps,
                self.on_log,
            )
            if capL is None or capR is None:
                with contextlib.suppress(Exception):
                    if capL is not None:
                        capL.release()
                    if capR is not None:
                        capR.release()
                raise RuntimeError("Failed to open one or both camera streams")
        else:
            capL = cv2.VideoCapture(self.left_src)
            capR = cv2.VideoCapture(self.right_src)
            if not (capL.isOpened() and capR.isOpened()):
                with contextlib.suppress(Exception):
                    capL.release()
                    capR.release()
                raise RuntimeError("Failed to open one or both video files")
        return capL, capR

    def _read_pair(self, capL, capR):
        if self.mode == "stream":
            return read_next_pair_from_caps(capL, capR)
        return read_next_pair_from_files(capL, capR)

    def run(self):
        self._stop_event.clear()
        with push_cuda_context():
            self.on_log(f"[local] Loading TensorRT engine: {self.engine_path}")
            engine = TensorRTEngine(self.engine_path)
            self.on_log("[local] TensorRT engine ready.")

            capL = capR = None
            seq = 0
            frame_interval = (1.0 / self.fps) if (self.fps and self.fps > 0) else 0.0
            next_t = time.perf_counter()

            try:
                capL, capR = self._open_sources()
                self.on_start()

                if self.save_dir:
                    self.save_dir.mkdir(parents=True, exist_ok=True)

                while not self._stop_event.is_set():
                    pair = self._read_pair(capL, capR)
                    if pair is None:
                        self.on_log("[local] End of source(s).")
                        break

                    frameL, frameR = pair

                    if (
                            self.frame_width > 0
                            and self.frame_height > 0
                            and (frameL.shape[1] != self.frame_width or frameL.shape[0] != self.frame_height)
                    ):
                        frameL = cv2.resize(
                            frameL, (self.frame_width, self.frame_height), interpolation=cv2.INTER_AREA
                        )
                    if (
                            self.frame_width > 0
                            and self.frame_height > 0
                            and (frameR.shape[1] != self.frame_width or frameR.shape[0] != self.frame_height)
                    ):
                        frameR = cv2.resize(
                            frameR, (self.frame_width, self.frame_height), interpolation=cv2.INTER_AREA
                        )

                    try:
                        left_pre, right_pre, (mh, mw), (sh, sw) = self._prepare_frames(frameL, frameR)
                    except Exception as exc:
                        self.on_log(f"[local] Failed to prepare frames: {exc}")
                        raise

                    try:
                        disp = engine.infer(left_pre, right_pre)
                    except Exception as exc:
                        self.on_log(f"[local] TensorRT inference failed: {exc}")
                        raise

                    try:
                        png16 = disparity_to_png16(disp, self.max_disp, self.disp_scale)
                    except Exception as exc:
                        self.on_log(f"[local] Failed to encode disparity: {exc}")
                        raise

                    meta = {
                        "session_id": self.session_id,
                        "source_mode": self.mode,
                        "sender_wh": [int(sw), int(sh)],
                        "aligned32": bool(self.align_to_32),
                        "letterbox": bool(self.letterbox),
                    }

                    self.on_result(seq, "disparity", "png16", int(mw), int(mh), png16, meta)

                    if self.save_dir:
                        fname = self.save_dir / f"disparity_seq{seq:06d}.png"
                        with open(fname, "wb") as f:
                            f.write(png16)

                    seq += 1

                    if frame_interval > 0.0:
                        next_t += frame_interval
                        sleep_s = next_t - time.perf_counter()
                        if sleep_s > 0:
                            if self._stop_event.wait(timeout=sleep_s):
                                break
            finally:
                with contextlib.suppress(Exception):
                    if capL is not None:
                        capL.release()
                with contextlib.suppress(Exception):
                    if capR is not None:
                        capR.release()
                self.on_finish()