"""RGB-D aware pose backend loaders.

This module centralises the creation of callable pose backends so that
``LocalEngineRunner`` can focus on orchestration.  Each backend receives an RGB
frame plus an optional depth map and returns a list of 2D keypoints along with a
confidence score per pose.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Type alias describing the callable interface implemented by pose backends.
PoseBackend = Callable[[np.ndarray, Optional[np.ndarray]], Tuple[List[np.ndarray], List[float]]]


def compose_rgbd_input(image_bgr: np.ndarray, depth_m: Optional[np.ndarray]) -> np.ndarray:
    """Blend a depth map into the RGB frame to create an RGB-D style input."""

    if depth_m is None:
        return image_bgr

    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.shape[:2] != image_bgr.shape[:2]:
        depth = cv2.resize(depth, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return image_bgr

    depth_valid = depth[valid]
    d_min = float(np.percentile(depth_valid, 5.0))
    d_max = float(np.percentile(depth_valid, 95.0))
    if not np.isfinite(d_min) or not np.isfinite(d_max) or d_max <= d_min:
        return image_bgr

    depth_norm = np.clip((depth - d_min) / (d_max - d_min), 0.0, 1.0)
    depth_u8 = (depth_norm * 255.0).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_VIRIDIS)
    fused = cv2.addWeighted(image_bgr, 0.65, depth_color, 0.35, 0.0)
    return fused


def depth_weighted_scores(
    poses: Sequence[np.ndarray], scores: Sequence[float], depth_m: Optional[np.ndarray]
) -> List[float]:
    """Adjust pose confidences based on valid depth coverage."""

    if depth_m is None:
        return [float(s) for s in scores]

    h, w = depth_m.shape[:2]
    adjusted: List[float] = []
    for pose, score in zip(poses, scores):
        pts = np.asarray(pose)
        if pts.size == 0:
            adjusted.append(float(score))
            continue

        valid = 0
        considered = 0
        for (u, v, *_rest) in pts:
            ui = int(round(float(u)))
            vi = int(round(float(v)))
            if 0 <= ui < w and 0 <= vi < h:
                considered += 1
                depth_val = float(depth_m[vi, ui])
                if np.isfinite(depth_val) and depth_val > 0:
                    valid += 1

        if considered == 0:
            adjusted.append(float(score) * 0.5)
            continue

        coverage = valid / considered
        adjusted.append(float(score) * (0.5 + 0.5 * coverage))

    return adjusted


_MEDIAPIPE_COCO17_SELECTION = [
    0,  # nose
    2,  # left_eye
    5,  # right_eye
    7,  # left_ear
    8,  # right_ear
    11,  # left_shoulder
    12,  # right_shoulder
    13,  # left_elbow
    14,  # right_elbow
    15,  # left_wrist
    16,  # right_wrist
    23,  # left_hip
    24,  # right_hip
    25,  # left_knee
    26,  # right_knee
    27,  # left_ankle
    28,  # right_ankle
]


def _build_blazepose_backend(log: Callable[[str], None], key: str) -> Optional[PoseBackend]:
    try:
        import mediapipe as mp  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        log(f"[pose] MediaPipe not available; skipping BlazePose backend ({exc}).")
        return None

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    def runner(img_bgr: np.ndarray, depth: Optional[np.ndarray]):
        fused = compose_rgbd_input(img_bgr, depth)
        res = pose.process(cv2.cvtColor(fused, cv2.COLOR_BGR2RGB))
        poses: List[np.ndarray] = []
        scores: List[float] = []
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            h, w = fused.shape[:2]
            kps = []
            for idx in _MEDIAPIPE_COCO17_SELECTION:
                p = lm[idx]
                u, v = p.x * w, p.y * h
                c = float(p.visibility) if p.visibility is not None else 0.5
                kps.append([u, v, c])
            poses.append(np.array(kps, dtype=np.float32))
            scores.append(1.0)
        scores = depth_weighted_scores(poses, scores, depth)
        return poses, scores

    log("[pose] MediaPipe BlazePose backend initialised.")
    return runner


def _build_holistic_backend(log: Callable[[str], None], key: str) -> Optional[PoseBackend]:
    try:
        import mediapipe as mp  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        log(f"[pose] MediaPipe not available; skipping Holistic backend ({exc}).")
        return None

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    def runner(img_bgr: np.ndarray, depth: Optional[np.ndarray]):
        fused = compose_rgbd_input(img_bgr, depth)
        res = holistic.process(cv2.cvtColor(fused, cv2.COLOR_BGR2RGB))
        poses: List[np.ndarray] = []
        scores: List[float] = []
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            h, w = fused.shape[:2]
            kps = []
            for idx in _MEDIAPIPE_COCO17_SELECTION:
                p = lm[idx]
                u, v = p.x * w, p.y * h
                c = float(p.visibility) if p.visibility is not None else 0.5
                kps.append([u, v, c])
            poses.append(np.array(kps, dtype=np.float32))
            scores.append(1.0)
        scores = depth_weighted_scores(poses, scores, depth)
        return poses, scores

    log("[pose] MediaPipe Holistic backend initialised.")
    return

def _resolve_pose_model_path(path: str) -> str:
    """Return an absolute path for ``path`` searching common asset directories."""

    candidate = Path(path)
    if candidate.is_file():
        return str(candidate)

    search_roots: List[Path] = []

    env_dirs = os.getenv("POSE_MODEL_DIR", "").strip()
    if env_dirs:
        for entry in env_dirs.split(os.pathsep):
            if entry:
                search_roots.append(Path(entry))

    module_dir = Path(__file__).resolve().parent
    search_roots.extend(
        [
            module_dir,
            module_dir / "pose_models",
            Path.cwd(),
            Path.cwd() / "pose_models",
        ]
    )

    seen: set[Path] = set()
    for root in search_roots:
        try:
            root_resolved = root.resolve()
        except FileNotFoundError:
            continue
        if root_resolved in seen:
            continue
        seen.add(root_resolved)
        test_path = root_resolved / path
        if test_path.is_file():
            return str(test_path)

    return str(candidate)

def _resolve_movenet_model(model_key: str) -> Tuple[str, int]:
    lowered = model_key.lower()
    if "movenet_lightning" in lowered:
        path = os.getenv("MOVENET_ONNX_PATH_LIGHTNING", "movenet_lightning.onnx")
        return _resolve_pose_model_path(path), 192
    if "movenet_thunder" in lowered:
        path = os.getenv("MOVENET_ONNX_PATH_THUNDER", "movenet_thunder.onnx")
        return _resolve_pose_model_path(path), 256
    # Default to thunder unless explicitly overridden via MOVENET_ONNX_PATH
    env_path = os.getenv("MOVENET_ONNX_PATH")
    if env_path:
        return _resolve_pose_model_path(env_path), 256
    return _resolve_pose_model_path("movenet_thunder.onnx"), 256


def _build_movenet_backend(log: Callable[[str], None], key: str) -> Optional[PoseBackend]:
    onnx_path, input_size = _resolve_movenet_model(key)

    # Prefer ONNX Runtime when available because it supports a wider range of
    # ONNX operators (e.g. GatherND which is used by the official MoveNet
    # export) than OpenCV's DNN module.  Falling back to OpenCV keeps the
    # dependency optional for environments where installing onnxruntime is not
    # possible.
    ort_session = None
    ort_input_name: Optional[str] = None
    ort_output_name: Optional[str] = None
    try:  # pragma: no cover - optional dependency
        import onnxruntime as ort  # type: ignore

        providers = ort.get_available_providers()
        ort_session = ort.InferenceSession(onnx_path, providers=providers)
        ort_input_name = ort_session.get_inputs()[0].name
        ort_output_name = ort_session.get_outputs()[0].name
        log(f"[pose] MoveNet backend initialised via ONNX Runtime from '{onnx_path}'.")
    except ImportError:
        log("[pose] onnxruntime not available; falling back to OpenCV DNN for MoveNet.")
    except Exception as exc:  # pragma: no cover - optional dependency
        log(
            "[pose] Failed to initialise MoveNet with onnxruntime; "
            f"falling back to OpenCV DNN ({exc})."
        )
        ort_session = None
        ort_input_name = None
        ort_output_name = None

    if ort_session is not None and ort_input_name and ort_output_name:
        input_meta = ort_session.get_inputs()[0]
        input_type = input_meta.type

        _ORT_TYPE_TO_DTYPE = {
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(double)": np.float64,
            "tensor(uint8)": np.uint8,
            "tensor(int8)": np.int8,
            "tensor(int16)": np.int16,
            "tensor(int32)": np.int32,
            "tensor(int64)": np.int64,
        }
        ort_input_dtype = _ORT_TYPE_TO_DTYPE.get(input_type)
        if ort_input_dtype is None:
            log(
                "[pose] MoveNet ONNX input type '%s' not explicitly handled; "
                "defaulting to float32 normalised input." % input_type
            )
            ort_input_dtype = np.float32

        def _prepare_ort_input(rgb: np.ndarray) -> np.ndarray:
            if ort_input_dtype == np.float16:
                arr = (rgb.astype(np.float32) / 255.0).astype(np.float16)
            elif np.issubdtype(ort_input_dtype, np.floating):
                arr = rgb.astype(ort_input_dtype) / 255.0
            else:
                arr = rgb.astype(ort_input_dtype)
            return np.expand_dims(arr, axis=0)

        def runner(img_bgr: np.ndarray, depth: Optional[np.ndarray]):
            fused = compose_rgbd_input(img_bgr, depth)
            resized = cv2.resize(fused, (input_size, input_size))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            arr = _prepare_ort_input(rgb)
            out = ort_session.run([ort_output_name], {ort_input_name: arr})[0]
            out = np.asarray(out).reshape(-1, 3)
            h, w = fused.shape[:2]
            kps = []
            for (y, x, c) in out:
                u = float(x) * w
                v = float(y) * h
                kps.append([u, v, float(c)])
            poses = [np.array(kps, dtype=np.float32)]
            scores = depth_weighted_scores(poses, [1.0], depth)
            return poses, scores

        return runner

    try:
        net = cv2.dnn.readNetFromONNX(onnx_path)
    except Exception as exc:  # pragma: no cover - depends on optional assets
        log(f"[pose] MoveNet ONNX failed to load from '{onnx_path}': {exc}")
        return None

    def runner(img_bgr: np.ndarray, depth: Optional[np.ndarray]):
        fused = compose_rgbd_input(img_bgr, depth)
        inp = cv2.resize(fused, (input_size, input_size))
        blob = cv2.dnn.blobFromImage(
            inp,
            scalefactor=1 / 255.0,
            size=(input_size, input_size),
            swapRB=True,
            crop=False,
        )
        net.setInput(blob)
        out = net.forward().reshape(-1, 3)
        h, w = fused.shape[:2]
        kps = []
        for (y, x, c) in out:
            u = float(x) * w
            v = float(y) * h
            kps.append([u, v, float(c)])
        poses = [np.array(kps, dtype=np.float32)]
        scores = depth_weighted_scores(poses, [1.0], depth)
        return poses, scores

    log(f"[pose] MoveNet backend initialised from '{onnx_path}'.")
    return runner

def _resolve_yolov8_pose_model(key: str) -> str:
    env_override = os.getenv("YOLOV8_POSE_MODEL")
    if env_override:
        return _resolve_pose_model_path(env_override)
    lowered = key.lower()
    if lowered.endswith(".pt") or lowered.endswith(".onnx"):
        return _resolve_pose_model_path(key)

    for variant in ("n", "s", "m", "l", "x"):
        token = f"yolov8{variant}"
        if token in lowered:
            return _resolve_pose_model_path(f"{token}-pose.pt")
    return _resolve_pose_model_path("yolov8n-pose.pt")


def _build_yolov8_pose_backend(log: Callable[[str], None], key: str) -> Optional[PoseBackend]:
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        log(f"[pose] Ultralytics YOLO not available; skipping YOLOv8 backend ({exc}).")
        return None

    model_path = _resolve_yolov8_pose_model(key)
    try:  # pragma: no cover - depends on optional assets
        model = YOLO(model_path)
    except Exception as exc:
        log(f"[pose] YOLOv8 pose model failed to load from '{model_path}': {exc}")
        return None

    try:
        import torch
    except Exception as exc:  # pragma: no cover - optional dependency
        log(f"[pose] PyTorch not available; skipping YOLOv8 backend ({exc}).")
        return None

    inference_context = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad


    def runner(img_bgr: np.ndarray, depth: Optional[np.ndarray]):
        fused = compose_rgbd_input(img_bgr, depth)
        rgb = cv2.cvtColor(fused, cv2.COLOR_BGR2RGB)

        with inference_context():
            results = model.predict(rgb, verbose=False)

        if not results:
            return [], []

        result = results[0]
        keypoints = getattr(result, "keypoints", None)
        if keypoints is None:
            return [], []

        xy = getattr(keypoints, "xy", None)
        if xy is None:
            return [], []

        conf = getattr(keypoints, "conf", None)

        if hasattr(xy, "cpu"):
            xy_np = xy.cpu().numpy()
        else:
            xy_np = np.asarray(xy)

        conf_np: Optional[np.ndarray]
        if conf is None:
            conf_np = None
        elif hasattr(conf, "cpu"):
            conf_np = conf.cpu().numpy()
        else:
            conf_np = np.asarray(conf)

        poses: List[np.ndarray] = []
        scores: List[float] = []
        for idx in range(xy_np.shape[0]):
            coords = xy_np[idx]
            if conf_np is not None:
                coord_conf = conf_np[idx]
            else:
                coord_conf = np.ones(coords.shape[0], dtype=np.float32)

            keypoints_list: List[List[float]] = []
            for (x, y), c in zip(coords, coord_conf):
                conf_val = float(c) if np.isfinite(c) else 0.0
                keypoints_list.append([float(x), float(y), conf_val])

            poses.append(np.array(keypoints_list[:17], dtype=np.float32))
            mean_conf = float(np.nanmean(coord_conf)) if coord_conf.size else 0.0
            scores.append(mean_conf if np.isfinite(mean_conf) else 0.0)

        scores = depth_weighted_scores(poses, scores, depth)
        return poses, scores

    log(f"[pose] YOLOv8 backend initialised from '{model_path}'.")
    return runner


Builder = Callable[[Callable[[str], None], str], Optional[PoseBackend]]

_POSE_BACKEND_BUILDERS: Dict[str, Builder] = {
    "blazepose": _build_blazepose_backend,
    "holistic": _build_holistic_backend,
    "movenet": _build_movenet_backend,
    "yolov8": _build_yolov8_pose_backend,
}

_PREFERRED_BACKEND_ORDER: Tuple[str, ...] = (
    "blazepose",
    "holistic",
    "movenet",
    "yolov8",
)


def _normalise_key(model_key: str) -> str:
    lowered = model_key.lower()
    if lowered.startswith("movenet"):
        return "movenet"
    return lowered


def build_pose_backend(
    model_key: Optional[str], log: Callable[[str], None]
) -> Tuple[Optional[PoseBackend], Optional[str]]:
    """Return a callable pose backend and the resolved model key."""

    keys_to_try: Iterable[str]
    if model_key:
        keys_to_try = [model_key]
    else:
        keys_to_try = _PREFERRED_BACKEND_ORDER

    for key in keys_to_try:
        builder_key = _normalise_key(key)
        builder = _POSE_BACKEND_BUILDERS.get(builder_key)
        if builder is None:
            log(f"[pose] No backend registered for model key '{key}'.")
            continue
        backend = builder(log, key)
        if backend is not None:
            return backend, key.lower()
    return None, None


__all__ = [
    "PoseBackend",
    "build_pose_backend",
    "compose_rgbd_input",
    "depth_weighted_scores",
]
