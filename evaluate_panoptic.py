"""Evaluate FoundationStereoClient pose backends on CMU Panoptic sequences.

This script projects the CMU Panoptic Studio 3D annotations into a chosen HD
camera, runs one of the RGB/RGB-D pose backends on the corresponding image, and
reports standard accuracy metrics (PCK and mean pixel error).

Example usage::

    python panoptic_evaluation.py \
        --dataset-root /data/panoptic/171204_pose1 \
        --left-camera 00_16 \
        --pose-model movenet_thunder \
        --max-frames 500 \
        --stride 5

By default the evaluator samples every frame from the requested camera.  Use the
``--right-camera``/``--disparity-engine`` flags if you want to provide a stereo
pair and a TensorRT disparity network so that RGB-D aware backends can consume a
fused depth map.  When no disparity engine is supplied the pose backend receives
only the RGB frame (matching ``--pose-only`` mode in ``LocalEngineRunner``).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from pose_backends import PoseBackend, build_pose_backend
from pose_augmentation import draw_skeletons, get_pose_model_info

try:  # SciPy is optional; fall back to a greedy matcher if unavailable.
    from scipy.optimize import linear_sum_assignment  # type: ignore
except Exception:  # pragma: no cover - SciPy is an optional dependency
    linear_sum_assignment = None

# TensorRT/PyCUDA are optional; only import when the user requests disparity
# inference.  The local_inference module already wraps these dependencies.
try:  # pragma: no cover - heavy optional dependency
    from local_inference import TensorRTPipeline, TensorRTUnavailableError
except Exception:  # pragma: no cover - TensorRT not installed
    TensorRTPipeline = None  # type: ignore
    TensorRTUnavailableError = RuntimeError  # type: ignore

# Mapping from CMU Panoptic "COCO19" joint indices to COCO-17 order expected by
# the Foundation pose backends.  The Panoptic labels are::
#   0:Neck 1:Nose 2:RShoulder 3:RElbow 4:RWrist 5:LShoulder 6:LElbow
#   7:LWrist 8:RHip 9:RKnee 10:RAnkle 11:LHip 12:LKnee 13:LAnkle
#   14:REye 15:LEye 16:REar 17:LEar 18:Chest
PANOPTIC_COCO19_TO_COCO17 = [
    1,   # nose
    15,  # left_eye
    14,  # right_eye
    17,  # left_ear
    16,  # right_ear
    5,   # left_shoulder
    2,   # right_shoulder
    6,   # left_elbow
    3,   # right_elbow
    7,   # left_wrist
    4,   # right_wrist
    11,  # left_hip
    8,   # right_hip
    12,  # left_knee
    9,   # right_knee
    13,  # left_ankle
    10,  # right_ankle
]


PREDICTION_JOINT_COLOR = (0, 255, 0)
PREDICTION_LIMB_COLOR = (255, 200, 0)
GROUND_TRUTH_JOINT_COLOR = (0, 0, 255)
GROUND_TRUTH_LIMB_COLOR = (0, 0, 255)


@dataclass
class PanopticCamera:
    name: str
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    dist: np.ndarray
    resolution: Tuple[int, int]

    def __post_init__(self) -> None:
        # Normalise calibration arrays so downstream code can assume contiguous
        # float data.  ``cv2.projectPoints`` is strict about dtypes and rejects
        # anything other than 32/64-bit floats, so pre-convert here.
        self.K = np.ascontiguousarray(np.asarray(self.K, dtype=np.float32).reshape(3, 3))
        self.R = np.ascontiguousarray(np.asarray(self.R, dtype=np.float32).reshape(3, 3))
        self.t = np.ascontiguousarray(np.asarray(self.t, dtype=np.float32).reshape(3, 1))
        self.dist = np.ascontiguousarray(np.asarray(self.dist, dtype=np.float32).reshape(-1))

        # Cache float64 views for the OpenCV projection helper.
        R64 = self.R.astype(np.float64)
        self._R = R64
        self._t = self.t.astype(np.float64)
        self._K = self.K.astype(np.float64)
        self._dist = None if self.dist.size == 0 else self.dist.astype(np.float64).reshape(1, -1)
        rvec, _ = cv2.Rodrigues(R64)
        self._rvec = np.ascontiguousarray(rvec)
        self._tvec = np.ascontiguousarray(self._t.reshape(3, 1))

    @property
    def width(self) -> int:
        return int(self.resolution[0])

    @property
    def height(self) -> int:
        return int(self.resolution[1])

    def project(self, xyz_world: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project XYZ world coordinates (in millimetres) into image pixels."""

        xyz = np.asarray(xyz_world, dtype=np.float64)

        if xyz.ndim != 2 or xyz.shape[1] < 3:
            raise ValueError("xyz_world must be shaped (N,3[+])")

        if xyz.shape[0] == 0:
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )

        pts = np.ascontiguousarray(xyz[:, :3], dtype=np.float64)
        cam = self._R @ pts.T + self._t  # camera coordinates
        z = cam[2, :]
        eps = 1e-6
        valid = z > eps

        projected, _ = cv2.projectPoints(
            pts.reshape(-1, 1, 3),
            self._rvec,
            self._tvec,
            self._K,
            self._dist,
        )

        uv = projected.reshape(-1, 2)
        uv[~valid, :] = np.nan

        depth = z.copy()
        depth[~valid] = np.nan

        return uv.astype(np.float32), depth.astype(np.float32)


_CAMERA_FAMILY_PRIORITY = {
    "hd": 3,
    "vga": 2,
    "kinect": 1,
}


def _camera_family_priority(name: str) -> int:
    key = name.lower().strip()
    if "_" in key:
        key = key.split("_", 1)[0]
    if key.startswith("cam"):
        key = "cam"
    return _CAMERA_FAMILY_PRIORITY.get(key, 0)


def _camera_aliases(name: str) -> List[str]:
    base = name.lower().strip()
    aliases = {base}

    # Allow selecting cameras via their numeric suffix (e.g. ``00_16``)
    for prefix in ("hd_", "vga_", "kinect_", "mono_", "optical_"):
        if base.startswith(prefix):
            aliases.add(base[len(prefix) :])

    # Support short forms such as ``cam00`` → ``00``.
    if base.startswith("cam"):
        aliases.add(base[3:])

    clean_aliases = {
        alias.strip("_")
        for alias in aliases
        if alias.strip("_")
    }
    return sorted(clean_aliases)


def load_panoptic_calibration(calib_path: Path) -> Dict[str, PanopticCamera]:
    with open(calib_path, "r", encoding="utf-8") as f:
        calib = json.load(f)

    cameras: Dict[str, PanopticCamera] = {}
    alias_priority: Dict[str, int] = {}
    entries: List[dict] = []

    if "cameras" in calib:
        cam_block = calib["cameras"]
        if isinstance(cam_block, dict):
            for group in cam_block.values():
                if isinstance(group, Sequence):
                    for item in group:
                        if isinstance(item, dict):
                            entries.append(item)
        elif isinstance(cam_block, Sequence):
            for item in cam_block:
                if isinstance(item, dict):
                    entries.append(item)
    elif isinstance(calib, Sequence):
        for item in calib:
            if isinstance(item, dict):
                entries.append(item)
    else:
        raise ValueError("Unsupported calibration JSON structure")

    for entry in entries:
        name = str(entry.get("name", "")).strip()
        if not name:
            continue
        resolution = tuple(entry.get("resolution", [1920, 1080]))
        cam = PanopticCamera(
            name=name,
            K=np.asarray(entry["K"], dtype=np.float32).reshape(3, 3),
            R=np.asarray(entry["R"], dtype=np.float32).reshape(3, 3),
            t=np.asarray(entry["t"], dtype=np.float32).reshape(3, 1),
            dist=np.asarray(entry.get("dist", [0, 0, 0, 0, 0]), dtype=np.float32),
            resolution=(int(resolution[0]), int(resolution[1])),
        )
        priority = _camera_family_priority(name)
        for alias in _camera_aliases(name):
            current_priority = alias_priority.get(alias)
            if current_priority is not None and current_priority > priority:
                continue
            alias_priority[alias] = priority
            cameras[alias] = cam
    if not cameras:
        raise ValueError("No camera calibration entries found")
    return cameras


@dataclass
class PoseSample:
    keypoints_uv: np.ndarray  # shape (17, 2)
    visibility: np.ndarray  # shape (17,) with {0,1}


def _pose_sample_to_uvc(sample: PoseSample) -> np.ndarray:
    keypoints = np.asarray(sample.keypoints_uv, dtype=np.float32)
    conf = sample.visibility.astype(np.float32).reshape(-1, 1)
    if keypoints.ndim != 2 or keypoints.shape[1] < 2:
        raise ValueError("PoseSample keypoints must be shaped (K, 2)")
    return np.concatenate([keypoints[:, :2], conf], axis=1)


def _normalise_pose_uvc(pose: np.ndarray) -> Optional[np.ndarray]:
    arr = np.asarray(pose, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return None
    if arr.shape[1] == 2:
        conf = np.ones((arr.shape[0], 1), dtype=np.float32)
        return np.concatenate([arr[:, :2], conf], axis=1)
    return arr[:, :3].astype(np.float32, copy=True)


def _body_to_pose_sample(body: dict, camera: PanopticCamera) -> PoseSample:
    joints = np.asarray(body.get("joints19", []), dtype=np.float32).reshape(-1, 4)
    if joints.size == 0:
        return PoseSample(np.full((17, 2), np.nan, dtype=np.float32), np.zeros((17,), dtype=np.uint8))

    uv_world, depth_world = camera.project(joints[:, :3])
    uv_camera, depth_camera = _project_camera_coordinates(camera, joints[:, :3])

    score_world = _score_projection(uv_world, depth_world, camera.width, camera.height)
    score_camera = _score_projection(uv_camera, depth_camera, camera.width, camera.height)

    if score_camera > score_world:
        uv = uv_camera
    else:
        uv = uv_world
    conf = joints[:, 3]

    keypoints = np.full((17, 2), np.nan, dtype=np.float32)
    visible = np.zeros((17,), dtype=np.uint8)

    for coco_idx, panoptic_idx in enumerate(PANOPTIC_COCO19_TO_COCO17):
        if panoptic_idx >= uv.shape[0]:
            continue
        if not np.isfinite(uv[panoptic_idx, 0]) or not np.isfinite(uv[panoptic_idx, 1]):
            continue
        if panoptic_idx >= conf.shape[0] or conf[panoptic_idx] <= 0.0:
            continue
        keypoints[coco_idx] = uv[panoptic_idx]
        visible[coco_idx] = 1
    return PoseSample(keypoints, visible)


def _body2d_to_pose_sample(body: dict) -> PoseSample:
    joints = np.asarray(body.get("joints19", []), dtype=np.float32)
    if joints.size == 0:
        return PoseSample(
            np.full((17, 2), np.nan, dtype=np.float32),
            np.zeros((17,), dtype=np.uint8),
        )

    joints = joints.reshape(-1, 3)

    keypoints = np.full((17, 2), np.nan, dtype=np.float32)
    visible = np.zeros((17,), dtype=np.uint8)

    for coco_idx, panoptic_idx in enumerate(PANOPTIC_COCO19_TO_COCO17):
        if panoptic_idx >= joints.shape[0]:
            continue
        u, v, conf = joints[panoptic_idx]
        if conf <= 0.0 or not (np.isfinite(u) and np.isfinite(v)):
            continue
        keypoints[coco_idx] = (u, v)
        visible[coco_idx] = 1

    return PoseSample(keypoints, visible)


def load_frame_annotations(
    json_path: Path,
    camera: PanopticCamera,
    json_2d_path: Optional[Path] = None,
) -> List[PoseSample]:
    if json_2d_path is not None and json_2d_path.exists():
        with open(json_2d_path, "r", encoding="utf-8") as f:
            data_2d = json.load(f)
        bodies_2d = data_2d.get("bodies", [])
        poses_2d = [_body2d_to_pose_sample(body) for body in bodies_2d]
        poses_2d = [pose for pose in poses_2d if pose.visibility.any()]
        if poses_2d:
            return poses_2d

    if not json_path.exists():
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    bodies = data.get("bodies", [])
    return [_body_to_pose_sample(body, camera) for body in bodies]


def _project_camera_coordinates(
    camera: PanopticCamera, xyz_world: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Project world coordinates after transforming them into the camera frame.

    The Panoptic annotations provide 3D joint locations in the global studio
    frame.  Some sequences also ship with per-camera coordinates, but these are
    not consistently available.  This helper mirrors :meth:`PanopticCamera.project`
    while explicitly applying the rigid transform so we can prefer whichever
    projection produces in-frame results for the current sample.
    """

    pts = np.asarray(xyz_world, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError("xyz_world must be shaped (N, 3[+])")

    if pts.shape[0] == 0:
        empty_uv = np.empty((0, 2), dtype=np.float32)
        empty_depth = np.empty((0,), dtype=np.float32)
        return empty_uv, empty_depth

    cam = camera.R @ pts[:, :3].T + camera.t  # camera coordinates (mm)
    cam = cam.T.astype(np.float64, copy=False)
    z = cam[:, 2].astype(np.float32, copy=False)

    eps = np.float32(1e-6)
    valid = z > eps

    uv = np.full((pts.shape[0], 2), np.nan, dtype=np.float32)

    if np.any(valid):
        cam_valid = cam[valid]
        projected, _ = cv2.projectPoints(
            cam_valid.reshape(-1, 1, 3),
            np.zeros((3, 1), dtype=np.float64),
            np.zeros((3, 1), dtype=np.float64),
            camera._K,
            camera._dist,
        )
        projected = projected.reshape(-1, 2).astype(np.float32)
        uv[valid] = projected

    depth = np.full((pts.shape[0],), np.nan, dtype=np.float32)
    depth[valid] = z[valid]

    return uv, depth


def _score_projection(uv: np.ndarray, depth: np.ndarray, width: int, height: int) -> float:
    """Score a projection based on how many joints land within the image."""

    if uv.ndim != 2 or uv.shape[1] < 2 or depth.ndim != 1:
        return 0.0

    uv = np.asarray(uv, dtype=np.float32)
    depth = np.asarray(depth, dtype=np.float32)

    valid = (
        np.isfinite(depth)
        & (depth > 0.0)
        & np.isfinite(uv[:, 0])
        & np.isfinite(uv[:, 1])
    )

    if not np.any(valid):
        return 0.0

    inside = (
        valid
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < float(width))
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < float(height))
    )

    # Prefer projections that yield more in-frame keypoints; break ties by the
    # total number of finite keypoints so that partially visible skeletons still
    # return a small positive score instead of zero.
    return float(inside.sum()) + 0.1 * float(valid.sum())


def _frame_id_from_path(image_path: Path) -> str:
    stem = image_path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    if digits:
        # Some Panoptic image dumps insert separators (e.g. frame_000000000999)
        # leading to more than eight digits in the stem.  The annotation JSONs
        # are consistently zero-padded to eight characters, so trim to the last
        # eight digits before zero-filling to avoid requesting 12-digit files.
        return digits[-8:].zfill(8)
    return stem


def _list_image_paths(camera_dir: Path) -> List[Path]:
    # CMU Panoptic stores HD frames as frame000000.jpg, frame000001.jpg, ...
    # Fall back to lexicographic sorting for robustness.
    if not camera_dir.exists():
        raise FileNotFoundError(f"Camera directory not found: {camera_dir}")
    candidates = sorted(camera_dir.glob("*.jpg"))
    if not candidates:
        candidates = sorted(camera_dir.glob("*.png"))
    if not candidates:
        raise FileNotFoundError(f"No image files found under {camera_dir}")
    return candidates


def _resize_if_needed(image: np.ndarray, width: Optional[int], height: Optional[int]) -> np.ndarray:
    if width is None and height is None:
        return image
    target_w = width if width is not None else image.shape[1]
    target_h = height if height is not None else image.shape[0]
    if image.shape[1] == target_w and image.shape[0] == target_h:
        return image
    return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)


@dataclass
class Metrics:
    total_gt_people: int = 0
    matched_people: int = 0
    total_gt_joints: int = 0
    correct_joints: int = 0
    total_joint_error: float = 0.0
    evaluated_joints: int = 0
    false_positives: int = 0

    def add_ground_truth(self, gt: PoseSample) -> None:
        self.total_gt_joints += int(gt.visibility.sum())

    def update_pair(self, pred: np.ndarray, gt: PoseSample, threshold_px: float) -> None:
        gt_uv = gt.keypoints_uv
        pred_uv = pred[:, :2]
        vis = gt.visibility.astype(bool)
        valid = vis & np.isfinite(gt_uv[:, 0]) & np.isfinite(pred_uv[:, 0])
        if not np.any(valid):
            return
        dists = np.linalg.norm(pred_uv[valid] - gt_uv[valid], axis=1)
        self.correct_joints += int(np.sum(dists <= threshold_px))
        self.total_joint_error += float(np.sum(dists))
        self.evaluated_joints += int(valid.sum())

    def summary(self) -> Dict[str, float]:
        pck = (self.correct_joints / self.total_gt_joints) if self.total_gt_joints else 0.0
        mean_err = (self.total_joint_error / self.evaluated_joints) if self.evaluated_joints else 0.0
        recall = (self.matched_people / self.total_gt_people) if self.total_gt_people else 0.0
        precision = (
            self.matched_people / (self.matched_people + self.false_positives)
            if (self.matched_people + self.false_positives) > 0
            else 0.0
        )
        return {
            "pck": float(pck),
            "mean_pixel_error": float(mean_err),
            "person_recall": float(recall),
            "person_precision": float(precision),
        }


def _match_poses(preds: Sequence[np.ndarray], gts: Sequence[PoseSample]) -> List[Tuple[int, int, float]]:
    if not preds or not gts:
        return []

    cost = np.zeros((len(preds), len(gts)), dtype=np.float32)
    for i, pred in enumerate(preds):
        pred_uv = pred[:, :2]
        for j, gt in enumerate(gts):
            vis = gt.visibility.astype(bool)
            valid = vis & np.isfinite(gt.keypoints_uv[:, 0]) & np.isfinite(pred_uv[:, 0])
            if not np.any(valid):
                cost[i, j] = 1e6
            else:
                dists = np.linalg.norm(pred_uv[valid] - gt.keypoints_uv[valid], axis=1)
                cost[i, j] = float(np.mean(dists))

    if linear_sum_assignment is not None:
        row_ind, col_ind = linear_sum_assignment(cost)
        return [(int(i), int(j), float(cost[int(i), int(j)])) for i, j in zip(row_ind, col_ind)]

    # Greedy fallback when SciPy is unavailable.
    matches: List[Tuple[int, int, float]] = []
    used_cols: set[int] = set()
    for i in np.argsort(cost.min(axis=1)):
        j = int(np.argmin(cost[i]))
        if j in used_cols:
            continue
        used_cols.add(j)
        matches.append((int(i), j, float(cost[int(i), j])))
    return matches


def _camera_center(camera: PanopticCamera) -> np.ndarray:
    R = camera.R.reshape(3, 3)
    t = camera.t.reshape(3, 1)
    center = -R.T @ t
    return center.ravel()


def _depth_from_disparity(
    disparity: np.ndarray, fx_px: float, baseline_m: float
) -> np.ndarray:
    disp = np.asarray(disparity, dtype=np.float32)
    eps = 1e-6
    depth = (fx_px * baseline_m) / np.maximum(disp, eps)
    depth[disp <= eps] = np.nan
    return depth


def _load_disparity_engine(engine_path: Optional[Path]) -> Optional[TensorRTPipeline]:
    if engine_path is None:
        return None
    if TensorRTPipeline is None:
        raise RuntimeError("TensorRT runtime is unavailable; install TensorRT/PyCUDA or omit --disparity-engine")
    try:
        return TensorRTPipeline(str(engine_path))
    except TensorRTUnavailableError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(f"Failed to initialise TensorRT engine: {exc}") from exc


def evaluate_sequence(
    dataset_root: Path,
    left_camera: str,
    right_camera: Optional[str],
    pose_backend: PoseBackend,
    frame_width: Optional[int],
    frame_height: Optional[int],
    max_frames: Optional[int],
    stride: int,
    pck_threshold: float,
    disparity_engine: Optional[TensorRTPipeline],
    on_log,
    pose_model_key: Optional[str] = None,
    visualise_dir: Optional[Path] = None,
) -> Metrics:
    calib_path = dataset_root / f"calibration_{dataset_root.name}.json"
    cameras = load_panoptic_calibration(calib_path)

    vis_dir = visualise_dir
    if vis_dir is not None:
        vis_dir.mkdir(parents=True, exist_ok=True)
        on_log(f"[info] Saving pose visualisations to {vis_dir}")

    def resolve_camera(name: str) -> PanopticCamera:
        key = str(name).lower().strip()
        if key in cameras:
            return cameras[key]

        if key.startswith("hd_") and key[3:] in cameras:
            return cameras[key[3:]]
        if not key.startswith("hd_"):
            hd_key = f"hd_{key}".strip("_")
            if hd_key in cameras:
                return cameras[hd_key]

        raise KeyError(f"Camera '{name}' not found in calibration file {calib_path}")

    left_cam = resolve_camera(left_camera)
    right_cam = resolve_camera(right_camera) if right_camera else None

    baseline_m: Optional[float] = None
    fx_px: Optional[float] = None
    if right_cam is not None:
        center_left = _camera_center(left_cam)
        center_right = _camera_center(right_cam)
        baseline_m = float(np.linalg.norm(center_left - center_right) / 1000.0)
        fx_px = float(left_cam.K.reshape(3, 3)[0, 0])
        if baseline_m <= 1e-6:
            on_log(
                f"[warn] Computed near-zero baseline for cameras {left_cam.name}/{right_cam.name};"
                " disabling depth fusion"
            )
            baseline_m = None
            fx_px = None
        else:
            on_log(
                f"[info] Estimated stereo baseline {baseline_m:.3f} m using cameras "
                f"{left_cam.name} / {right_cam.name}"
            )
    elif disparity_engine is not None:
        on_log("[warn] Disparity engine supplied without a stereo camera pair; depth disabled")

    left_dir = dataset_root / "hdImgs" / f"{left_cam.name.replace('hd_', '')}"
    if not left_dir.exists():
        # Support both hd_00_16 and 00_16 directory layouts.
        left_dir = dataset_root / "hdImgs" / left_cam.name
    right_dir: Optional[Path] = None
    if right_cam is not None:
        right_dir = dataset_root / "hdImgs" / f"{right_cam.name.replace('hd_', '')}"
        if not right_dir.exists():
            right_dir = dataset_root / "hdImgs" / right_cam.name

    left_frames = _list_image_paths(left_dir)
    if right_cam is not None:
        right_frames = _list_image_paths(right_dir) if right_dir is not None else []
        if len(right_frames) != len(left_frames):
            on_log(
                f"[warn] Left/right frame counts differ ({len(left_frames)} vs {len(right_frames)});"
                " truncating to the shorter sequence"
            )
            limit = min(len(left_frames), len(right_frames))
            left_frames = left_frames[:limit]
            right_frames = right_frames[:limit]

    annot_dir = dataset_root / "hdPose3d_stage1_coco19"
    annot2d_dir: Optional[Path] = None
    annot2d_root = dataset_root / "hdPose2d_stage1_coco19"
    if annot2d_root.exists():
        candidate_names = [left_cam.name.replace("hd_", ""), left_cam.name]
        for candidate in candidate_names:
            candidate = candidate.strip("_")
            if not candidate:
                continue
            candidate_dir = annot2d_root / candidate
            if candidate_dir.exists():
                annot2d_dir = candidate_dir
                on_log(
                    f"[info] Using 2D annotations from {candidate_dir.relative_to(dataset_root)}"
                )
                break

    metrics = Metrics()

    total_frames = len(left_frames)
    frame_indices = range(0, total_frames, stride)
    if max_frames is not None:
        frame_indices = list(frame_indices)[:max_frames]
    else:
        frame_indices = list(frame_indices)

    native_width = left_cam.width
    native_height = left_cam.height
    threshold_px = pck_threshold * max(native_width, native_height)
    on_log(
        f"[info] Evaluating {len(frame_indices)} frames from camera {left_cam.name}"
        f" with PCK threshold {threshold_px:.1f} px"
    )

    skipped_missing_annotations = 0
    frames_with_annotations = 0
    frames_with_ground_truth = 0
    frames_with_predictions = 0
    frames_with_matches = 0

    for idx_pos, frame_idx in enumerate(frame_indices):
        left_path = left_frames[frame_idx]
        right_path = right_frames[frame_idx] if right_cam is not None else None

        left_img_native = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
        if left_img_native is None:
            on_log(f"[warn] Failed to read {left_path}; skipping frame")
            continue
        native_h, native_w = left_img_native.shape[:2]
        left_img = _resize_if_needed(left_img_native, frame_width, frame_height)
        resized_h, resized_w = left_img.shape[:2]
        scale_x = native_w / float(resized_w) if resized_w else 1.0
        scale_y = native_h / float(resized_h) if resized_h else 1.0

        depth_map: Optional[np.ndarray] = None
        if (
            disparity_engine is not None
            and right_path is not None
            and baseline_m is not None
            and fx_px is not None
        ):
            right_img_native = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
            if right_img_native is None:
                on_log(f"[warn] Failed to read {right_path}; skipping disparity")
            else:
                right_img = _resize_if_needed(right_img_native, frame_width, frame_height)
                disp = disparity_engine.infer(left_img, right_img)
                if disp is not None:
                    depth_map = _depth_from_disparity(disp, fx_px, baseline_m)
        poses, _scores = pose_backend(left_img, depth_map)

        if scale_x != 1.0 or scale_y != 1.0:
            scaled: List[np.ndarray] = []
            for pose in poses:
                arr = np.asarray(pose, dtype=np.float32)
                if arr.ndim != 2 or arr.shape[1] < 2:
                    scaled.append(arr)
                    continue
                arr = arr.copy()
                arr[:, 0] *= scale_x
                arr[:, 1] *= scale_y
                scaled.append(arr)
            poses = scaled

        frame_id = _frame_id_from_path(left_path)
        frame_json = annot_dir / f"body3DScene_{frame_id}.json"
        frame_json_2d = (
            annot2d_dir / f"body2DScene_{frame_id}.json" if annot2d_dir is not None else None
        )
        if (
            (frame_json_2d is None or not frame_json_2d.exists())
            and not frame_json.exists()
        ):
            if skipped_missing_annotations < 5:
                on_log(
                    f"[warn] Missing annotation file {frame_json.name}; skipping frame"
                )
            skipped_missing_annotations += 1
            continue

        gt_poses = load_frame_annotations(frame_json, left_cam, frame_json_2d)
        frames_with_annotations += 1
        if gt_poses:
            frames_with_ground_truth += 1

        metrics.total_gt_people += len(gt_poses)
        for gt_pose in gt_poses:
            metrics.add_ground_truth(gt_pose)

        matches = _match_poses(poses, gt_poses)
        matched_preds: set[int] = set()
        accepted_match = False
        if poses:
            frames_with_predictions += 1
        distance_gate = threshold_px * 2.0
        for pred_idx, gt_idx, avg_dist in matches:
            if avg_dist > distance_gate:
                continue
            pred = poses[pred_idx]
            gt = gt_poses[gt_idx]
            metrics.matched_people += 1
            metrics.update_pair(pred, gt, threshold_px)
            matched_preds.add(pred_idx)
            accepted_match = True

        if accepted_match:
            frames_with_matches += 1

        metrics.false_positives += max(len(poses) - len(matched_preds), 0)

        if vis_dir is not None:
            pred_uvc: List[np.ndarray] = []
            for pose in poses:
                arr = _normalise_pose_uvc(pose)
                if arr is not None:
                    pred_uvc.append(arr)
            gt_uvc = [_pose_sample_to_uvc(gt_pose) for gt_pose in gt_poses]
            if pred_uvc or gt_uvc:
                vis_image = left_img_native.copy()
                if pred_uvc:
                    vis_image = draw_skeletons(
                        vis_image,
                        pred_uvc,
                        pose_model_key,
                        joint_color=PREDICTION_JOINT_COLOR,
                        limb_color=PREDICTION_LIMB_COLOR,
                    )
                if gt_uvc:
                    vis_image = draw_skeletons(
                        vis_image,
                        gt_uvc,
                        pose_model_key,
                        joint_color=GROUND_TRUTH_JOINT_COLOR,
                        limb_color=GROUND_TRUTH_LIMB_COLOR,
                    )

                legend_entries: List[Tuple[str, Tuple[int, int, int]]] = []
                if pred_uvc:
                    legend_entries.append(("Prediction", PREDICTION_JOINT_COLOR))
                if gt_uvc:
                    legend_entries.append(("Ground truth", GROUND_TRUTH_JOINT_COLOR))
                for line_idx, (label, colour) in enumerate(legend_entries):
                    origin = (10, 30 + line_idx * 25)
                    cv2.putText(
                        vis_image,
                        label,
                        origin,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        colour,
                        2,
                        cv2.LINE_AA,
                    )

                vis_path = vis_dir / f"{frame_id}.jpg"
                if not cv2.imwrite(str(vis_path), vis_image):
                    on_log(f"[warn] Failed to write visualisation to {vis_path}")

        if idx_pos and idx_pos % 50 == 0:
            summary = metrics.summary()
            on_log(
                f"[info] Processed {idx_pos+1}/{len(frame_indices)} frames → "
                f"PCK={summary['pck']*100:.1f}% MPE={summary['mean_pixel_error']:.1f}px"
            )

    if skipped_missing_annotations:
        on_log(
            f"[warn] Skipped {skipped_missing_annotations} frame(s) with missing annotations"
        )

    on_log(
        "[info] Frame coverage: "
        f"{frames_with_annotations} annotation files / {frames_with_ground_truth} with people / "
        f"{frames_with_predictions} with detections / {frames_with_matches} matched"
    )

    if frames_with_annotations == 0:
        on_log(
            "[error] No ground-truth annotations were loaded; check dataset paths and frame numbering"
        )
    elif frames_with_ground_truth == 0:
        on_log(
            "[warn] Annotation files were present but contained no people; accuracy metrics will remain zero"
        )
    elif frames_with_predictions == 0:
        on_log(
            "[warn] Pose backend did not return any detections on annotated frames; accuracy will be zero"
        )
    elif frames_with_matches == 0:
        on_log(
            "[warn] Detections never satisfied the distance gate; try increasing --pck-threshold or review pose quality"
        )

    return metrics


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pose accuracy on CMU Panoptic sequences.")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--left-camera", required=True, help="Camera ID (e.g. 00_16) used for evaluation.")
    parser.add_argument("--right-camera", help="Optional camera ID to provide a stereo pair.")
    parser.add_argument("--pose-model", default="movenet_thunder", help="Pose backend key (movenet_thunder, blazepose, …).")
    parser.add_argument("--frame-width", type=int, help="Resize frames to this width before inference.")
    parser.add_argument("--frame-height", type=int, help="Resize frames to this height before inference.")
    parser.add_argument("--max-frames", type=int, help="Maximum number of frames to evaluate.")
    parser.add_argument("--stride", type=int, default=1, help="Sample every Nth frame from the sequence.")
    parser.add_argument(
        "--pck-threshold",
        type=float,
        default=0.05,
        help="PCK threshold as a fraction of the larger evaluation image dimension.",
    )
    parser.add_argument("--disparity-engine", type=Path, help="Optional TensorRT engine for disparity estimation.")
    parser.add_argument(
        "--visualise-dir",
        type=Path,
        help="Directory where per-frame pose overlays (prediction vs. ground truth) will be saved.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    def log(msg: str) -> None:
        print(msg)

    pose_backend, resolved_key = build_pose_backend(args.pose_model, log)
    if pose_backend is None:
        log("[error] Failed to initialise the requested pose backend.")
        return 1

    model_key = resolved_key or args.pose_model
    info = get_pose_model_info(model_key)
    if info is not None:
        log(f"[info] Using pose backend: {info.display_name} ({info.key})")

    disparity_engine = None
    if args.disparity_engine:
        disparity_engine = _load_disparity_engine(args.disparity_engine)
        log(f"[info] Disparity engine initialised from {args.disparity_engine}")

    metrics = evaluate_sequence(
        dataset_root=args.dataset_root,
        left_camera=args.left_camera,
        right_camera=args.right_camera,
        pose_backend=pose_backend,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        max_frames=args.max_frames,
        stride=max(1, args.stride),
        pck_threshold=float(args.pck_threshold),
        disparity_engine=disparity_engine,
        on_log=log,
        pose_model_key=model_key,
        visualise_dir=args.visualise_dir,
    )

    summary = metrics.summary()
    log("[summary] Evaluation complete")
    log(f"          People recall:    {summary['person_recall']*100:.2f}%")
    log(f"          People precision: {summary['person_precision']*100:.2f}%")
    log(f"          PCK:              {summary['pck']*100:.2f}%")
    log(f"          Mean pixel error:  {summary['mean_pixel_error']:.2f} px")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry-point
    sys.exit(main())
