"""Utilities for augmenting RGB pose estimation with depth information.

This module implements several complementary strategies that exploit aligned
RGB-D frames.  They are inspired by the description provided in the task:

1.  **Depth-aware post-processing (late fusion)** operating on the output of an
    arbitrary 2D pose detector.  Joints are lifted to 3D using the sampled depth
    value, overlapping skeletons are filtered using depth-guided
    non-maximum-suppression, and occluded joints can be hallucinated by
    searching for depth-consistent candidates along the incident limb.
2.  **Early fusion** that adapts a backbone that was originally trained on RGB
    images to also receive depth as an additional channel.  The helper mirrors
    the manual procedure of cloning the pretrained RGB filters and initialising
    a new depth kernel with their mean.
3.  **Mid/late fusion** helpers that combine the feature maps produced by two
    backbones – one running on RGB and one on depth (or HHA encoded depth) –
    before feeding them to the pose head.

The code intentionally favours clarity and drop-in usability over raw speed so
that it can be reused from notebooks or small inference scripts without further
infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn

# ---------------------------------------------------------------------------
# Registries used by the GUI / orchestration layer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PoseModelInfo:
    """Metadata describing an RGB(-D) pose estimator."""

    key: str
    display_name: str
    description: str


@dataclass(frozen=True)
class PoseAugmentationInfo:
    """Metadata describing a depth-aware augmentation strategy."""

    key: str
    display_name: str
    description: str
    components: Tuple[str, ...] = ()


_POSE_MODELS: Tuple[PoseModelInfo, ...] = (
    PoseModelInfo(
        key="openpose",
        display_name="OpenPose",
        description=(
            "Canonical multi-person pose estimator with strong multi-view support. "
            "Depth cues can be used for late fusion to improve skeleton selection."
        ),
    ),
    PoseModelInfo(
        key="movenet",
        display_name="MoveNet",
        description=(
            "Lightweight single-person model from Google. Ideal for edge devices and "
            "real-time inference when combined with depth-aware post-processing."
        ),
    ),
    PoseModelInfo(
        key="blazepose",
        display_name="BlazePose",
        description=(
            "MediaPipe full-body pose network well suited for fitness and AR "
            "applications. Responds well to early fusion with RGB-D inputs."
        ),
    ),
)


_POSE_AUGMENTATIONS: Tuple[PoseAugmentationInfo, ...] = (
    PoseAugmentationInfo(
        key="depth_postprocess",
        display_name="Depth-aware post-processing",
        description=(
            "Lift 2D keypoints into 3D, apply depth-guided pose NMS and repair "
            "occluded joints by searching along limbs at consistent depth."
        ),
        components=(
            "lift_keypoints_to_3d",
            "depth_guided_nms",
            "repair_occluded_joints",
        ),
    ),
    PoseAugmentationInfo(
        key="early_fusion",
        display_name="Early RGB-D fusion",
        description=(
            "Adapt RGB backbones to accept depth (or HHA) as an extra channel and "
            "fine-tune for scale-aware feature learning."
        ),
        components=(
            "adapt_first_conv_for_depth",
            "convert_depth_to_hha",
        ),
    ),
    PoseAugmentationInfo(
        key="two_stream",
        display_name="Two-stream fusion",
        description=(
            "Run dedicated RGB and depth encoders whose intermediate features are "
            "fused (concat or attention) before the pose head."
        ),
        components=(
            "FeatureFusion",
            "TwoStreamPoseBackbone",
        ),
    ),
)


_POSE_MODELS_BY_KEY = {info.key: info for info in _POSE_MODELS}
_POSE_AUGMENTATIONS_BY_KEY = {info.key: info for info in _POSE_AUGMENTATIONS}


def list_pose_models() -> List[PoseModelInfo]:
    """Return the supported pose estimators."""

    return list(_POSE_MODELS)


def list_pose_augmentations() -> List[PoseAugmentationInfo]:
    """Return the supported depth-aware augmentation strategies."""

    return list(_POSE_AUGMENTATIONS)


def get_pose_model_info(key: Optional[str]) -> Optional[PoseModelInfo]:
    """Retrieve :class:`PoseModelInfo` for ``key`` (case insensitive)."""

    if key is None:
        return None
    return _POSE_MODELS_BY_KEY.get(str(key).lower())


def get_pose_augmentation_info(key: Optional[str]) -> Optional[PoseAugmentationInfo]:
    """Retrieve :class:`PoseAugmentationInfo` for ``key`` (case insensitive)."""

    if key is None:
        return None
    return _POSE_AUGMENTATIONS_BY_KEY.get(str(key).lower())


# ---------------------------------------------------------------------------
# Depth aware post processing helpers
# ---------------------------------------------------------------------------


def lift_keypoints_to_3d(
    keypoints: np.ndarray,
    depth_map: np.ndarray,
    intrinsics: Sequence[float],
    invalid_value: float = np.nan,
) -> np.ndarray:
    """Lift 2D keypoints to camera-centric 3D coordinates.

    Parameters
    ----------
    keypoints:
        Array with shape ``(N, 2)`` or ``(N, 3)`` containing pixel coordinates
        ``(u, v)`` and optionally confidence scores.  Coordinates are assumed to
        be in pixels and aligned with ``depth_map``.
    depth_map:
        Single-channel depth image where ``depth_map[v, u]`` returns the depth
        (in metres) for the pixel at ``(u, v)``.
    intrinsics:
        Camera intrinsics given as ``(fx, fy, cx, cy)``.
    invalid_value:
        Value used to fill missing 3D coordinates when the depth lookup fails.

    Returns
    -------
    np.ndarray
        Array of shape ``(N, 3)`` containing camera-centric XYZ coordinates.
        Points for which the depth was invalid are filled with ``invalid_value``.
    """

    fx, fy, cx, cy = intrinsics
    keypoints = np.asarray(keypoints)
    if keypoints.ndim != 2 or keypoints.shape[1] < 2:
        raise ValueError("keypoints must be of shape (N, 2) or (N, 3)")

    depth_map = np.asarray(depth_map)
    if depth_map.ndim != 2:
        raise ValueError("depth_map must be a single channel image")

    xyz = np.full((keypoints.shape[0], 3), invalid_value, dtype=np.float32)

    for idx, (u, v, *_rest) in enumerate(keypoints):
        u_i = int(round(float(u)))
        v_i = int(round(float(v)))
        if not (0 <= v_i < depth_map.shape[0] and 0 <= u_i < depth_map.shape[1]):
            continue

        z = float(depth_map[v_i, u_i])
        if not np.isfinite(z) or z <= 0:
            continue

        x = (float(u) - cx) * z / fx
        y = (float(v) - cy) * z / fy
        xyz[idx] = (x, y, z)

    return xyz


def _compute_bbox(points: np.ndarray) -> Tuple[float, float, float, float]:
    """Compute the axis aligned bounding box of 2D points."""

    valid = np.isfinite(points).all(axis=1)
    if not np.any(valid):
        return (np.nan, np.nan, np.nan, np.nan)

    pts = points[valid]
    x_min = float(np.min(pts[:, 0]))
    y_min = float(np.min(pts[:, 1]))
    x_max = float(np.max(pts[:, 0]))
    y_max = float(np.max(pts[:, 1]))
    return x_min, y_min, x_max, y_max


def _bbox_iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    """Intersection over Union of two axis aligned bounding boxes."""

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    if np.isnan(ax1) or np.isnan(bx1):
        return 0.0

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter_area
    if denom <= 0:
        return 0.0
    return inter_area / denom


def _sample_depth(depth_map: np.ndarray, point: Sequence[float]) -> float:
    """Sample depth at a floating point location using nearest neighbour lookup."""

    u, v = point
    u_i = int(round(float(u)))
    v_i = int(round(float(v)))
    if not (0 <= v_i < depth_map.shape[0] and 0 <= u_i < depth_map.shape[1]):
        return float("nan")
    return float(depth_map[v_i, u_i])


def _depth_consistency(
    pose: np.ndarray,
    depth_map: np.ndarray,
    skeleton_edges: Sequence[Tuple[int, int]],
) -> float:
    """Compute a depth consistency score for a skeleton.

    The score is the mean absolute depth difference along valid limbs.  Lower
    values indicate better consistency.  When no limb can be evaluated the
    function returns ``np.inf`` so that such poses are disfavoured in NMS.
    """

    diffs: List[float] = []
    for a, b in skeleton_edges:
        joint_a = pose[a]
        joint_b = pose[b]
        if not (np.isfinite(joint_a).all() and np.isfinite(joint_b).all()):
            continue

        depth_a = _sample_depth(depth_map, joint_a[:2])
        depth_b = _sample_depth(depth_map, joint_b[:2])
        if not (np.isfinite(depth_a) and np.isfinite(depth_b)):
            continue
        diffs.append(abs(depth_a - depth_b))

    if not diffs:
        return float("inf")
    return float(np.mean(diffs))


def depth_guided_nms(
    poses: Sequence[np.ndarray],
    scores: Sequence[float],
    depth_map: np.ndarray,
    skeleton_edges: Sequence[Tuple[int, int]],
    iou_threshold: float = 0.5,
) -> List[int]:
    """Perform depth-aware non-maximum suppression on pose detections.

    Parameters
    ----------
    poses:
        Iterable of ``(K, >=2)`` arrays representing the per-joint coordinates.
    scores:
        Detection confidence for each pose (e.g. average joint score).
    depth_map:
        Depth image aligned with the pose coordinates.
    skeleton_edges:
        Pairs of joint indices describing the articulated skeleton structure.
    iou_threshold:
        Suppress poses with IoU larger than this value.

    Returns
    -------
    list[int]
        Indices of the poses that are kept.  The list is sorted by descending
        score, mirroring the behaviour of regular NMS routines.

    Notes
    -----
    When two poses overlap strongly, the pose with the lower depth consistency
    score (i.e. the one whose limbs lie on a more coherent depth surface) is
    preferred.  If both poses have identical scores the one with the higher
    detection score is retained.
    """

    if len(poses) != len(scores):
        raise ValueError("poses and scores must have the same length")

    boxes = [_compute_bbox(np.asarray(pose)[..., :2]) for pose in poses]
    depth_scores = [
        _depth_consistency(np.asarray(pose), depth_map, skeleton_edges) for pose in poses
    ]

    order = sorted(range(len(poses)), key=lambda i: scores[i], reverse=True)
    keep: List[int] = []

    while order:
        current = order.pop(0)
        keep.append(current)

        remaining: List[int] = []
        for idx in order:
            iou = _bbox_iou(boxes[current], boxes[idx])
            if iou < iou_threshold:
                remaining.append(idx)
                continue

            # Overlapping poses -> compare depth coherence.
            consistency_current = depth_scores[current]
            consistency_other = depth_scores[idx]

            if consistency_other + 1e-6 < consistency_current:
                # The other pose is more depth-consistent; keep it instead of the
                # current one.
                keep.pop()  # remove the current pose
                keep.append(idx)
                current = idx
                consistency_current = consistency_other
            # Suppress the other pose regardless of which one won.
        order = remaining

    # Sort retained indices by their original score for deterministic behaviour.
    keep.sort(key=lambda i: scores[i], reverse=True)
    return keep


@dataclass
class OcclusionRepairConfig:
    """Configuration used by :func:`repair_occluded_joints`."""

    confidence_threshold: float = 0.2
    search_radius: float = 20.0
    steps: int = 10
    depth_tolerance: float = 0.1  # relative tolerance, 10% by default
    min_step: float = 1.5


def _build_adjacency(edges: Sequence[Tuple[int, int]], num_joints: int) -> List[List[int]]:
    adjacency: List[List[int]] = [[] for _ in range(num_joints)]
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    return adjacency


def _estimate_limb_direction(
    pose: np.ndarray,
    joint_idx: int,
    neighbor_idx: int,
    adjacency: Sequence[Sequence[int]],
) -> np.ndarray:
    """Estimate a search direction for a missing joint.

    Preference is given to the vector between the neighbour and the (possibly
    low-confidence) joint.  If this information is unavailable the direction is
    derived from other limbs connected to the neighbour.  As a final fallback a
    horizontal direction is returned.
    """

    neighbour_point = pose[neighbor_idx]
    joint_point = pose[joint_idx]
    if np.isfinite(joint_point).all():
        vec = joint_point - neighbour_point
        norm = np.linalg.norm(vec)
        if norm > 1e-3:
            return vec / norm

    for other in adjacency[neighbor_idx]:
        if other == joint_idx or not np.isfinite(pose[other]).all():
            continue
        vec = neighbour_point - pose[other]
        norm = np.linalg.norm(vec)
        if norm > 1e-3:
            return vec / norm

    return np.array([1.0, 0.0], dtype=np.float32)


def repair_occluded_joints(
    pose: np.ndarray,
    confidences: np.ndarray,
    depth_map: np.ndarray,
    skeleton_edges: Sequence[Tuple[int, int]],
    config: OcclusionRepairConfig | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Attempt to inpaint low-confidence or missing joints.

    Parameters
    ----------
    pose:
        ``(K, 2)`` array containing joint coordinates.
    confidences:
        ``(K,)`` array containing per-joint confidences in ``[0, 1]``.  Joints
        with a confidence smaller than ``config.confidence_threshold`` are
        treated as occluded.
    depth_map:
        Depth image aligned with ``pose``.
    skeleton_edges:
        Pairs of joints that form limbs.
    config:
        Optional :class:`OcclusionRepairConfig` controlling the search.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The repaired pose and the updated confidences.  The returned arrays are
        copies and therefore safe to mutate by the caller.
    """

    pose = np.asarray(pose, dtype=np.float32).copy()
    confidences = np.asarray(confidences, dtype=np.float32).copy()
    if pose.ndim != 2 or pose.shape[1] != 2:
        raise ValueError("pose must have shape (K, 2)")
    if confidences.shape[0] != pose.shape[0]:
        raise ValueError("confidences must have shape (K,)")

    cfg = config or OcclusionRepairConfig()
    adjacency = _build_adjacency(skeleton_edges, pose.shape[0])

    for joint_idx, confidence in enumerate(confidences):
        if confidence >= cfg.confidence_threshold and np.isfinite(pose[joint_idx]).all():
            continue

        neighbours = [n for n in adjacency[joint_idx] if np.isfinite(pose[n]).all()]
        if not neighbours:
            continue

        best_candidate = None
        best_confidence = confidence
        for neighbour_idx in neighbours:
            base = pose[neighbour_idx]
            neighbour_depth = _sample_depth(depth_map, base)
            if not np.isfinite(neighbour_depth) or neighbour_depth <= 0:
                continue

            direction = _estimate_limb_direction(pose, joint_idx, neighbour_idx, adjacency)
            if np.linalg.norm(direction) < 1e-3:
                continue

            direction = direction / np.linalg.norm(direction)
            step = max(cfg.min_step, cfg.search_radius / max(cfg.steps, 1))

            for t in range(1, cfg.steps + 1):
                candidate = base + direction * (t * step)
                u, v = candidate
                if not (0 <= v < depth_map.shape[0] and 0 <= u < depth_map.shape[1]):
                    break
                candidate_depth = _sample_depth(depth_map, candidate)
                if not np.isfinite(candidate_depth) or candidate_depth <= 0:
                    continue
                if abs(candidate_depth - neighbour_depth) > cfg.depth_tolerance * neighbour_depth:
                    continue

                # The candidate is depth-consistent.  Confidence decays with the
                # travelled distance to favour closer repairs.
                new_conf = max(confidence, float(np.exp(-0.1 * t)))
                if best_candidate is None or new_conf > best_confidence:
                    best_candidate = candidate
                    best_confidence = new_conf
                break  # Stop searching further for this neighbour.

        if best_candidate is not None:
            pose[joint_idx] = best_candidate
            confidences[joint_idx] = best_confidence

    return pose, confidences


# ---------------------------------------------------------------------------
# Early fusion helpers
# ---------------------------------------------------------------------------


def adapt_first_conv_for_depth(
    backbone: nn.Module,
    attr_name: str = "conv1",
    depth_channels: int = 1,
    init_strategy: str = "mean",
) -> nn.Module:
    """Adapt the first convolution of a backbone to ingest depth data.

    Parameters
    ----------
    backbone:
        Model whose attribute ``attr_name`` points to the first convolutional
        layer.  The function modifies the model in-place and also returns it for
        convenience.
    attr_name:
        Name of the attribute that stores the initial convolution.
    depth_channels:
        Number of additional depth channels.  ``1`` corresponds to concatenating
        raw depth, ``3`` can be used for HHA encoded depth.
    init_strategy:
        Either ``"mean"`` (default) or ``"zeros"``.  ``"mean"`` clones the RGB
        weights and initialises the additional depth filters with their mean.
        ``"zeros"`` keeps the new kernel at zero while copying the RGB weights.

    Returns
    -------
    nn.Module
        The modified backbone.
    """

    if depth_channels < 1:
        raise ValueError("depth_channels must be >= 1")

    old_conv = getattr(backbone, attr_name, None)
    if not isinstance(old_conv, nn.Conv2d):
        raise AttributeError(f"{attr_name} must be an instance of nn.Conv2d")

    new_in_channels = old_conv.in_channels + depth_channels
    new_conv = nn.Conv2d(
        in_channels=new_in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=old_conv.bias is not None,
        padding_mode=old_conv.padding_mode,
    )

    with torch.no_grad():
        new_conv.weight[:, : old_conv.in_channels] = old_conv.weight
        if init_strategy == "mean":
            depth_weight = old_conv.weight.mean(dim=1, keepdim=True)
        elif init_strategy == "zeros":
            depth_weight = torch.zeros_like(old_conv.weight[:, :1])
        else:
            raise ValueError("init_strategy must be either 'mean' or 'zeros'")

        depth_weight = depth_weight.repeat(1, depth_channels, 1, 1)
        new_conv.weight[:, old_conv.in_channels :] = depth_weight
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    setattr(backbone, attr_name, new_conv)
    return backbone


def convert_depth_to_hha(
    depth_map: np.ndarray,
    fx: float,
    fy: float,
    baseline: float = 1.0,
    max_depth: float = 10.0,
    invalid_value: float = 0.0,
) -> np.ndarray:
    """Convert a raw depth map to HHA (Horizontal disparity, Height, Angle).

    The implementation follows Gupta et al., ECCV 2014.  The HHA representation
    often yields better results than raw depth when used as input to CNNs.
    ``fx`` and ``fy`` are the focal lengths of the camera (in pixels) and the
    ``baseline`` is the stereo baseline used to compute disparity.
    """

    depth = np.asarray(depth_map, dtype=np.float32)
    h, w = depth.shape

    # Horizontal disparity (scaled inverse depth)
    disparity = np.where(depth > 0, baseline * fx / np.clip(depth, 1e-6, max_depth), 0)

    # Height above ground plane approximation: assume gravity is aligned with
    # the negative Y-axis of the camera coordinate system.
    grid_y, grid_x = np.mgrid[0:h, 0:w]
    z = depth
    x = (grid_x - w / 2.0) * z / fx
    y = (grid_y - h / 2.0) * z / fy
    height = -y  # camera Y-axis typically points downwards

    # Angle with inferred gravity vector.  Approximate surface normals via
    # central differences in 3D space.
    dzdx = np.gradient(z, axis=1)
    dzdy = np.gradient(z, axis=0)
    nx = -dzdx
    ny = -dzdy
    nz = np.ones_like(z)
    norm = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2) + 1e-8
    nx /= norm
    ny /= norm
    nz /= norm
    angle = np.degrees(np.arccos(np.clip(ny, -1.0, 1.0)))

    hha = np.stack([disparity, height, angle], axis=-1)
    hha[~np.isfinite(hha)] = invalid_value
    return hha


# ---------------------------------------------------------------------------
# Mid / late fusion helpers
# ---------------------------------------------------------------------------


class FeatureFusion(nn.Module):
    """Fuse RGB and depth feature maps using concatenation or attention."""

    def __init__(self, in_channels: int, fusion_channels: int, mode: str = "concat") -> None:
        super().__init__()
        self.mode = mode
        if mode == "concat":
            self.fuse = nn.Sequential(
                nn.Conv2d(in_channels * 2, fusion_channels, kernel_size=1),
                nn.BatchNorm2d(fusion_channels),
                nn.ReLU(inplace=True),
            )
        elif mode == "attention":
            self.key = nn.Conv2d(in_channels, fusion_channels, kernel_size=1)
            self.query = nn.Conv2d(in_channels, fusion_channels, kernel_size=1)
            self.value = nn.Conv2d(in_channels, fusion_channels, kernel_size=1)
            self.out_proj = nn.Conv2d(fusion_channels, fusion_channels, kernel_size=1)
        else:
            raise ValueError("mode must be either 'concat' or 'attention'")

    def forward(self, rgb_feat: Tensor, depth_feat: Tensor) -> Tensor:
        if self.mode == "concat":
            fused = torch.cat([rgb_feat, depth_feat], dim=1)
            return self.fuse(fused)

        # Attention mode: compute simple multiplicative attention map.
        q = self.query(rgb_feat)
        k = self.key(depth_feat)
        v = self.value(depth_feat)
        attn = torch.sigmoid(q * k)
        fused = self.out_proj(attn * v + rgb_feat)
        return fused


class TwoStreamPoseBackbone(nn.Module):
    """Combine two backbones (RGB and depth) with feature fusion layers.

    The module expects the caller to pass feature extractor callables that return
    intermediate feature maps (e.g. the C3/C4 features of a ResNet).  Only the
    fusion logic is implemented here to keep the component generic.
    """

    def __init__(
        self,
        rgb_backbone: Callable[[Tensor], Sequence[Tensor]],
        depth_backbone: Callable[[Tensor], Sequence[Tensor]],
        fusion_modules: Sequence[nn.Module],
    ) -> None:
        super().__init__()
        self.rgb_backbone = rgb_backbone
        self.depth_backbone = depth_backbone
        self.fusion_modules = nn.ModuleList(fusion_modules)

        if len(self.fusion_modules) == 0:
            raise ValueError("At least one fusion module must be provided")

    def forward(self, rgb: Tensor, depth: Tensor) -> List[Tensor]:
        rgb_feats = self.rgb_backbone(rgb)
        depth_feats = self.depth_backbone(depth)
        if len(rgb_feats) != len(depth_feats) or len(rgb_feats) != len(self.fusion_modules):
            raise ValueError("Number of feature levels must match the fusion modules")

        fused: List[Tensor] = []
        for rgb_feat, depth_feat, fuse in zip(rgb_feats, depth_feats, self.fusion_modules):
            fused.append(fuse(rgb_feat, depth_feat))
        return fused


__all__ = [
    "PoseModelInfo",
    "PoseAugmentationInfo",
    "list_pose_models",
    "list_pose_augmentations",
    "get_pose_model_info",
    "get_pose_augmentation_info",
    "lift_keypoints_to_3d",
    "depth_guided_nms",
    "repair_occluded_joints",
    "OcclusionRepairConfig",
    "adapt_first_conv_for_depth",
    "convert_depth_to_hha",
    "FeatureFusion",
    "TwoStreamPoseBackbone",
]