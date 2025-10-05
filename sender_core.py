import asyncio, json, time, contextlib, pathlib
from typing import Optional, Tuple, Callable, Dict, Any
import cv2
import numpy as np
import websockets

FRAME_HEIGHT = 640
FRAME_WIDTH = 480

def encode_jpeg(img: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()

def read_next_pair_from_caps(capL: cv2.VideoCapture, capR: cv2.VideoCapture) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    okL, frameL = capL.read()
    okR, frameR = capR.read()
    if not (okL and okR):
        return None
    return frameL, frameR

class StereoSenderClient:
    def __init__(
            self,
            host: str,
            port: int,
            path: str,
            fps: float,
            jpeg_quality: int,
            session_id: str,
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
            capL = cv2.VideoCapture(int(left_src))
            capR = cv2.VideoCapture(int(right_src))
            if self.fps:
                for c in (capL, capR):
                    c.set(cv2.CAP_PROP_FPS, self.fps)

        else:
            capL = cv2.VideoCapture(left_src)
            capR = cv2.VideoCapture(right_src)

        uri = f"ws://{self.host}:{self.port}{self.path}"
        self.log(f"Connecting to {uri} ...")

        async with websockets.connect(uri, max_size = None, ping_interval = 20, ping_timeout = 20) as ws:
            self._ws = ws
            await self._handshake()
            self.log("Handshake OK.")

            start_msg = {
                "type": "start", "mode": mode, "fmt": "jpeg",
                "fps": float(self.fps) if self.fps else None,
                "width": FRAME_WIDTH, "height": FRAME_HEIGHT,
                "meta": {"session_id": self.session_id}
            }
        await ws.send(json.dumps(start_msg))

        self._recv_task = asyncio.create_task(self._receiver_loop())
        self._on_start()

        seq = 0
        frame_interval = (1.0/self.fps) if (self.fps and self.fps > 0) else 0.0
        next_t = time.perf_counter()

        while not self._stop_flag.is_set():
            pair = read_next_pair_from_caps(capL, capR)
            if pair is None:
                self.log("End of source(s).")
                break
            frameL, frameR = pair
            frameL = cv2.resize(frameL, (FRAME_WIDTH, FRAME_HEIGHT), interpolation = cv2.INTER_AREA)
            frameR = cv2.resize(frameR, (FRAME_WIDTH, FRAME_HEIGHT), interpolation = cv2.INTER_AREA)
            jpegL = encode_jpeg(frameL, self.jpeg_quality)
            jpegR = encode_jpeg(frameR, self.jpeg_quality)

            ts_ns = time.time_ns()
            meta = {"type":"frame", "ts":ts_ns, "seq":seq,
                    "left_bytes":len(jpegL), "right_bytes": len(jpegR)}
            await ws.send(json.dumps(meta))
            await ws.send(jpegL)
            await ws.send(jpegR)
            seq += 1

            if frame_interval > 0:
                next_t += frame_interval
                sleep_s = next_t - time.perf_counter()
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)

        with contextlib.suppress(Exception):
            await ws.send(json.dumps({"type": "end"}))
        if self._recv_task:
            self._recv_task.cancel()
            with contextlib.suppress(Exception):
                await self._recv_task

        self._on_finish()

    def stop(self):
        self._stop_flag.set()