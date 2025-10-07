import asyncio, json, time, contextlib, pathlib, os
from typing import Optional, Tuple, Callable, Dict, Any
import cv2
import numpy as np
import websockets

DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480

def logprint(cb, msg: str):
    try:
        if cb: cb(msg)
        else: print(msg)
    except Exception:
        print(msg)

def encode_jpeg(img: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()

def read_next_pair_from_caps(capL: cv2.VideoCapture, capR: cv2.VideoCapture, retries: int = 2, backoff_s: float = 0.005) -> tuple | None:
    def _read_with_retry(cap: cv2.VideoCapture) -> tuple[bool, any]:
        ok, frame = cap.read()
        tries = 0
        while (not ok or frame is None) and tries < retries:
            time.sleep(backoff_s)
            ok, frame = cap.read()
            tries += 1
        return ok, frame

    okL, frameL = _read_with_retry(capL)
    if not okL or frameL is None:
        return None

    okR, frameR = _read_with_retry(capR)
    if not okR or frameR is None:
        return None

    return frameL, frameR

def read_next_pair_from_files(capL: cv2.VideoCapture, capR: cv2.VideoCapture, skip_once: bool = True):
    okL, frameL = capL.read()
    okR, frameR = capR.read()

    if okL and okR and frameL is not None and frameR is not None:
        return frameL, frameR

    if not skip_once:
        return None

    if not okL or frameL is None:
        okL, frameL = capL.read()
    if not okR or frameR is None:
        okR, frameR = capR.read()

    if okL and okR and frameL is not None and frameR is not None:
        return frameL, frameR

    return None

_BACKEND_LABELS: Dict[int, str] = {}
if hasattr(cv2, "CAP_DSHOW"):
    _BACKEND_LABELS[cv2.CAP_DSHOW] = "CAP_DSHOW"
if hasattr(cv2, "CAP_MSMF"):
    _BACKEND_LABELS[cv2.CAP_MSMF] = "CAP_MSMF"
if hasattr(cv2, "CAP_ANY"):
    _BACKEND_LABELS[cv2.CAP_ANY] = "CAP_ANY"

def _backend_name(backend: int | None) -> str:
    if backend is None:
        return "default"
    return _BACKEND_LABELS.get(backend, str(backend))

def open_cam(idx: int, height: int, width: int, target_fps: float | None, on_log=None) -> cv2.VideoCapture | None:
    backend_candidates: list[tuple[int | None, Callable[[int], cv2.VideoCapture]]] = []
    if os.name == "nt":
        dshow = getattr(cv2, "CAP_DSHOW", None)
        if isinstance(dshow, int):
            backend_candidates.append((dshow, lambda i, b=dshow:cv2.VideoCapture(i, b)))
        msmf = getattr(cv2, "CAP_MSMF", None)
        if isinstance(dshow, int):
            backend_candidates.append((msmf, lambda i, b=msmf: cv2.VideoCapture(i, b)))
    backend_candidates.append((None, lambda i: cv2.VideoCapture(i)))

    cap = None
    used_backend: int | None = None
    for backend, factory in backend_candidates:
        try:
            cap = factory(idx)
        except Exception:
            cap = None
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            logprint(on_log, f"[camera] backend {_backend_name(backend)} failed to open index {idx}")
            cap = None
            continue
        used_backend = backend
        break
    if cap is None:
        logprint(on_log, f"[camera] failed to open index {idx}")
        return None
    logprint(on_log, f"[camera {idx}] opened using {_backend_name(used_backend)}")
    with contextlib.suppress(Exception):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
    if target_fps:
        cap.set(cv2.CAP_PROP_FPS, float(target_fps))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    got_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    got_fps = cap.get(cv2.CAP_PROP_FPS)
    logprint(on_log, f"[camera {idx}] requested {width}x{height} @ {target_fps or 'auto'}fps -> got {got_w}x{got_h} @ {got_fps:.2f}fps")
    return cap

class StereoSenderClient:
    def __init__(
            self,
            host: str,
            port: int,
            path: str,
            fps: float,
            jpeg_quality: int,
            session_id: str,
            frame_width: int = DEFAULT_WIDTH,
            frame_height: int = DEFAULT_HEIGHT,
            save_dir: Optional[pathlib.Path] = None,
            preview: bool = False,
            on_log: Optional[Callable[[str], None]] = None,
            on_result: Optional[Callable[[int, str, str, int, int, bytes, Dict[str, Any]], None]] = None,
            on_start: Optional[Callable[[], None]] = None,
            on_finish: Optional[Callable[[], None]] = None
    ):
        self.host = host
        self.port = port
        self.path = path
        self.fps = fps
        self.jpeg_quality = jpeg_quality
        self.session_id = session_id
        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.save_dir = save_dir
        self.preview = preview

        self._on_log = on_log or (lambda S: None)
        self._on_result = on_result or (lambda *args: None)
        self._on_start = on_start or (lambda: None)
        self._on_finish = on_finish or (lambda: None)

        self._stop_flag = asyncio.Event()
        self._ws = None
        self._recv_task=None

    def log(self, msg: str):
        self._on_log(msg)

    async def _receiver_loop(self):
        ws = self._ws
        assert ws is not None
        while not self._stop_flag.is_set():
            msg = await ws.recv()
            if isinstance(msg, (bytes, bytearray)):
                continue
            try:
                head = json.loads(msg)
            except Exception:
                continue
            if head.get("type") != "result":
                continue

            payload = await ws.recv()
            if not isinstance(payload, (bytes, bytearray)):
                continue

            seq = int(head.get("seq", -1))
            kind = head.get("kind", "depth")
            enc = head.get("encoding", "png16")
            w, h = int(head.get("w", 0)), int(head.get("h", 0))

            if self.save_dir:
                self.save_dir.mkdir(parents = True, exist_ok = True)
                ext = ".png" if "png" in enc else ".jpg"
                fname = self.save_dir / f"{kind}_seq{seq:06d}{ext}"
                with open(fname, "wb") as f:
                    f.write(payload)
            self._on_result(seq, kind, enc, w, h, payload, head)

    async def _handshake(self):
        await self._ws.send(json.dumps({"type": "hello", "role": "sender", "version": 1}))
        ack = await self._ws.recv()
        j = json.loads(ack)
        if not (j.get("type") == "hello_ack" and j.get("ok") is True):
            raise RuntimeError("fServer did not ack hello: {ack}")

    async def start_stream(self, left_src: str, right_src: str, mode: str):
        if mode == "stream":
            capL = open_cam(int(left_src), self.frame_height, self.frame_width, self.fps, self._on_log)
            capR = open_cam(int(right_src), self.frame_height, self.frame_width, self.fps, self._on_log)
            if capL is None or capR is None:
                self._on_log("[stream] could not open one or both cameras; stopping.")
                self._on_finish()
                return
        else:
            capL = cv2.VideoCapture(left_src)
            capR = cv2.VideoCapture(right_src)

        uri = f"ws://{self.host}:{self.port}{self.path}"
        self.log(f"Connecting to {uri} ...")

        async with websockets.connect(uri, max_size=None, ping_interval=20, ping_timeout=20) as ws:
            self._ws = ws
            await self._handshake()
            self.log("Handshake OK.")

            start_msg = {
                "type": "start", "mode": mode, "fmt": "jpeg",
                "fps": float(self.fps) if self.fps else None,
                "width": self.frame_width, "height": self.frame_height,
                "meta": {"session_id": self.session_id}
            }
            await ws.send(json.dumps(start_msg))

            self._recv_task = asyncio.create_task(self._receiver_loop())
            self._on_start()

            seq = 0
            frame_interval = (1.0 / self.fps) if (self.fps and self.fps > 0) else 0.0
            next_t = time.perf_counter()

            try:
                while not self._stop_flag.is_set():
                    pair = read_next_pair_from_caps(capL, capR) if mode=="stream" else read_next_pair_from_files(capL, capR)
                    if pair is None:
                        self.log("End of source(s).")
                        if mode == "stream":
                            okL, _ = capL.read()
                            okR, _ = capR.read()
                            self._on_log(f"End of Source(s). camL ok? {okL} camR ok? {okR}")
                        else:
                            self._on_log(f"End of Source(s).")
                        break
                    frameL, frameR = pair
                    frameL = cv2.resize(frameL, (self.frame_width, self.frame_height), interpolation=cv2.INTER_AREA)
                    frameR = cv2.resize(frameR, (self.frame_width, self.frame_height), interpolation=cv2.INTER_AREA)
                    jpegL = encode_jpeg(frameL, self.jpeg_quality)
                    jpegR = encode_jpeg(frameR, self.jpeg_quality)

                    payload = {
                        "type": "frame",
                        "seq": seq, "ts": time.time(),
                        "left": {"w": self.frame_width, "h": self.frame_height, "jpeg": len(jpegL)},
                        "right": {"w": self.frame_width, "h": self.frame_height, "jpeg": len(jpegR)},
                    }
                    await ws.send(json.dumps(payload))
                    await ws.send(jpegL)
                    await ws.send(jpegR)
                    seq += 1

                    if frame_interval > 0:
                        next_t += frame_interval
                        sleep_s = next_t - time.perf_counter()
                        if sleep_s > 0:
                            await asyncio.sleep(sleep_s)
            finally:
                with contextlib.suppress(Exception):
                    await ws.send(json.dumps({"type": "end"}))
                if self._recv_task:
                    self._recv_task.cancel()
                    with contextlib.suppress(Exception):
                        await self._recv_task
                with contextlib.suppress(Exception):
                    if mode == "stream":
                        capL.release(); capR.release()
                self._on_finish()

    def stop(self):
        self._stop_flag.set()