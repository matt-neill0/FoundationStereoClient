import asyncio, json, logging
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import cv2
import onnx
import onnxruntime as ort
import websockets

LISTEN_HOST, LISTEN_PORT, WS_PATH = "0.0.0.0", 8765, "/foundation-stereo"
ALIGN_TO_32 = True
LETTERBOX = False
MAX_DISP = 256.0
SCALE_16U = 256.0
MODEL_PATH = "pretrained_models/deployable_foundation_stereo_l_dynamic.onnx"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("fs_server")

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
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas

def preprocess_rgb(image_bgr: np.ndarray) -> np.ndarray:
    x = image_bgr[:, :, ::-1].astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None, ...]
    return x

def disparity_to_png16(disp: np.ndarray, max_disp:float, scale: float) -> bytes:
    d = np.nan_to_num(disp, nan=0.0, posinf=max_disp, neginf=0.0)
    d = np.clip(d, 0.0, max_disp)
    scaled = np.round(d * scale)
    if d.ndim > 2:
        d = np.squeeze(d)
    if d.ndim != 2:
        raise ValueError(f"Expected 2D disparity map, got shape {disp.shape}")
    if scaled.dtype != np.float32 and scaled.dtype != np.float64:
        scaled = scaled.astype(np.float32, copy=False)
    scaled = np.clip(scaled, 0.0, np.float32(np.iinfo(np.uint16).max))
    d16 = np.ascontiguousarray(scaled.astype(np.uint16))
    if 0 in d16.shape:
        raise ValueError(f"Invalid disparity image shape after squeeze: {d16.shape}")
    ok, buf = cv2.imencode(".png", d16)
    if not ok:
        raise RuntimeError("cv2.imencode PNG failed")
    return buf.tobytes()

class ORTDynamicBackend:
    def __init__(self, onnx_path: str):
        prov = ort.get_available_providers()
        if "CUDAExecutionProvider" in prov:
            providers = [("CUDAExecutionProvider", {
                "cudnn_conv_use_max_workspace": "1",
                "cudnn_conv_algo_search": "HEURISTIC"
            })]
            log.info("Using CUDAExecutionProvider")
        else:
            providers = ["CPUExecutionProvider"]
            log.warning("CUDAExecutionProvider not available; using CPUExecutionProvider")

        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.model = onnx.load(onnx_path)
        self.in_names = [i.name for i in self.model.graph.input]
        if len(self.in_names) < 2:
            raise RuntimeError(f"Expected two inputs; got {self.in_names}")
        self.out_checked = False

    def prepare_pair(self, jpegL: bytes, jpegR: bytes) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int], Tuple[int, int]]:
        L = cv2.imdecode(np.frombuffer(jpegL, np.uint8), cv2.IMREAD_COLOR)
        R = cv2.imdecode(np.frombuffer(jpegR, np.uint8), cv2.IMREAD_COLOR)
        if L is None or R is None:
            raise RuntimeError("cv2.imdecode failed")

        Hs, Ws = int(L.shape[0]), int(L.shape[1])
        Ht, Wt = Hs, Ws
        if ALIGN_TO_32:
            Ht, Wt = align32(Ht), align32(Wt)

        if (Hs, Ws) != (Ht, Wt):
            if LETTERBOX:
                L = letterbox_resize(L, (Ht, Wt))
                R = letterbox_resize(R, (Ht, Wt))
            else:
                L = cv2.resize(L, (Wt, Ht), interpolation = cv2.INTER_AREA)
                R = cv2.resize(R, (Wt, Ht), interpolation = cv2.INTER_AREA)
        return preprocess_rgb(L), preprocess_rgb(R), (Ht, Wt), (Hs, Ws)

    def infer(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        feeds = {self.in_names[0]: left, self.in_names[1]: right}
        outs = self.sess.run(None, feeds)
        out = outs[0]
        if not self.out_checked:
            self.out_checked = True
            try:
                shapes = [Tuple(x.shape) for x in outs]
            except Exception:
                shapes = ["?"]
            log.info(f"Outputs: {shapes} (using index 0)")
        disp = np.squeeze(out)
        if disp.ndim == 3:
            disp = disp[0]
        if disp.ndim != 2:
            raise RuntimeError(f"Unexpected output shape: {out.shape}")
        return disp

backend = ORTDynamicBackend(MODEL_PATH)

async def handle_ws(ws):
    req_path = getattr(ws, "path", None)
    if req_path is None:
        req = getattr(ws, "request", None)
        req_path = getattr(req, "path", None)
        
    if req_path != WS_PATH:
    	log.warning(f"Unexpected path: {req_path!r} (expected {WS_PATH!r})")
    	await ws.close()
    	return
        
    hello = json.loads(await ws.recv())
    if not (hello.get("type") == "hello" and hello.get("role") == "sender"):
        await ws.close(); return
    await ws.send(json.dumps({"type": "hello_ack", "ok": True}))

    start = json.loads(await ws.recv())
    if start.get("type") != "start":
        await ws.close(); return
    log.info(f"Start: mode={start.get('mode')} fps={start.get('fps')} sender_wh={start.get('width')}x{start.get('height')}")
    start_meta = start.get("meta") if isinstance(start.get("meta"), dict) else {}
    session_id = start_meta.get("session_id")

    while True:
        head_msg = await ws.recv()
        if isinstance(head_msg, (bytes, bytearray)):
            continue
        h = json.loads(head_msg)
        if h.get("type") == "end":
            break
        if h.get("type") != "frame":
            continue

        seq = int(h.get("seq", -1))
        jpegL = await ws.recv()
        jpegR = await ws.recv()
        if not isinstance(jpegL, (bytes, bytearray)) or not isinstance(jpegR, (bytes, bytearray)):
            continue

        loop = asyncio.get_event_loop()
        try:
            left, right, (mh, mw), (sh, sw) = await loop.run_in_executor(None, backend.prepare_pair, jpegL, jpegR)
        except Exception:
            log.exception("Failed to prepare frames for seq=%s", seq)
            continue
        try:
            disp = await loop.run_in_executor(None, backend.infer, left, right)
            png16 = await loop.run_in_executor(None, disparity_to_png16, disp, MAX_DISP, SCALE_16U)
        except Exception:
            log.exception("Inference failed for seq=%s", seq)
            continue           

        head_out = {
            "type": "result",
            "seq": seq,
            "kind": "disparity",
            "encoding": "png16",
            "w": int(mw), "h": int(mh),
            "meta": {
                "sender_wh": [int(sw), int(sh)],
                "aligned32":bool(ALIGN_TO_32),
                "letterbox": bool(LETTERBOX),
                "disp_scale": float(SCALE_16U),
                "max_disp": float(MAX_DISP)
            }
        }
        if session_id is not None:
            head_out["meta"]["session_id"] = session_id
        await ws.send(json.dumps(head_out))
        await ws.send(png16)

async def main():
    async with websockets.serve(handle_ws, LISTEN_HOST, LISTEN_PORT, max_size=None, ping_interval=20, ping_timeout=20):
        log.info(f"Listening on ws://{LISTEN_HOST}:{LISTEN_PORT}{WS_PATH}")
        while True:
            await asyncio.sleep(3600)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
