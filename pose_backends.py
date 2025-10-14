"""RGB-D aware pose backend loaders.

This module centralises the creation of callable pose backends so that
``LocalEngineRunner`` can focus on orchestration.  Each backend receives an RGB
frame plus an optional depth map and returns a list of 2D keypoints along with a
confidence score per pose.
"""

from __future__ import annotations

import os
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
    return runner


def _resolve_movenet_model(model_key: str) -> Tuple[str, int]:
    lowered = model_key.lower()
    if "movenet_lightning" in lowered:
        return os.getenv("MOVENET_ONNX_PATH_LIGHTNING", "movenet_lightning.onnx"), 192
    if "movenet_thunder" in lowered:
        return os.getenv("MOVENET_ONNX_PATH_THUNDER", "movenet_thunder.onnx"), 256
    # Default to thunder unless explicitly overridden via MOVENET_ONNX_PATH
    env_path = os.getenv("MOVENET_ONNX_PATH")
    if env_path:
        return env_path, 256
    return "movenet_thunder.onnx", 256


def _build_movenet_backend(log: Callable[[str], None], key: str) -> Optional[PoseBackend]:
    onnx_path, input_size = _resolve_movenet_model(key)
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


def _resolve_openpose_paths() -> Tuple[str, str]:
    proto = os.getenv(
        "OPENPOSE_PROTO",
        "pose_deploy_linevec_faster_4_stages.prototxt",
    )
    weights = os.getenv(
        "OPENPOSE_WEIGHTS",
        "pose_iter_440000.caffemodel",
    )
    return proto, weights


def _build_openpose_backend(log: Callable[[str], None], key: str) -> Optional[PoseBackend]:
    proto, weights = _resolve_openpose_paths()
    try:
        net = cv2.dnn.readNetFromCaffe(proto, weights)
    except Exception as exc:  # pragma: no cover - optional assets
        log(f"[pose] OpenPose model failed to load (proto='{proto}', weights='{weights}'): {exc}")
        return None

    def runner(img_bgr: np.ndarray, depth: Optional[np.ndarray]):
        fused = compose_rgbd_input(img_bgr, depth)
        blob = cv2.dnn.blobFromImage(
            fused,
            scalefactor=1.0 / 255,
            size=(368, 368),
            mean=(0, 0, 0),
            swapRB=False,
            crop=False,
        )
        net.setInput(blob)
        out = net.forward()
        num_points = out.shape[1]
        heat_h, heat_w = out.shape[2:]
        points: List[List[float]] = []
        for idx in range(num_points - 1):  # last channel is background
            heat_map = out[0, idx, :, :]
            _, conf, _, point = cv2.minMaxLoc(heat_map)
            x = (img_bgr.shape[1] * point[0]) / heat_w
            y = (img_bgr.shape[0] * point[1]) / heat_h
            points.append([float(x), float(y), float(conf)])
        poses = [np.array(points[:17], dtype=np.float32)]
        scores = depth_weighted_scores(poses, [1.0], depth)
        return poses, scores

    log("[pose] OpenPose backend initialised.")
    return runner


Builder = Callable[[Callable[[str], None], str], Optional[PoseBackend]]

_POSE_BACKEND_BUILDERS: Dict[str, Builder] = {
    "blazepose": _build_blazepose_backend,
    "holistic": _build_holistic_backend,
    "movenet": _build_movenet_backend,
    "openpose": _build_openpose_backend,
}

_PREFERRED_BACKEND_ORDER: Tuple[str, ...] = (
    "blazepose",
    "holistic",
    "movenet",
    "openpose",
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
