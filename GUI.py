from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import os
import platform
import re
import subprocess
import time
import shutil
from pathlib import Path

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from local_inference import (
    LocalEngineRunner,
    TensorRTUnavailableError,
    ensure_tensorrt_runtime,
)

from pose_augmentation import (
    draw_skeletons,
    get_skeleton_edges,
    list_pose_models,
)
from main import DEFAULT_FPS, DEFAULT_HEIGHT, DEFAULT_WIDTH

VIDEO_FILE_FILTER = "Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*)"


@dataclass(frozen=True)
class SessionConfig:
    engine_path: Optional[str]
    left_src: str
    right_src: str
    mode: str
    frame_width: int
    frame_height: int
    fps: float
    session_id: str
    save_dir: Optional[Path]
    preview: bool
    use_realsense: bool
    pose_enabled: bool = False
    pose_model: Optional[str] = None
    depth_enabled: bool = True

def _linux_list_cameras_v4l2() -> List[Tuple[int, str]]:
    try:
        proc = subprocess.run(
            ['v4l2-ctl', "--list-devices"],
            check=True, text=True, capture_output=True
        )
        text = proc.stdout
    except Exception:
        devs = sorted([d for d in os.listdir("/dev") if d.startswith("video")])
        out = []
        for d in devs:
            m = re.match(r"video(\d+)$", d)
            if m:
                out.append((int(m.group(1)), f"/dev/{d}"))
        return out

    blocks = re.split(r"\n(?=\S)", text.strip())
    mapping: List[Tuple[int, str]] = []
    for blk in blocks:
        lines = [ln for ln in blk.splitlines() if ln.strip()]
        if not lines:
            continue
        header = lines[0].strip()
        for ln in lines[1:]:
            dev = ln.strip()
            if dev.startswith("/dev/video"):
                m = re.match(r"video(\d+)$", dev)
                if m:
                    idx = int(m.group(1))
                    mapping.append((idx, f"{header} ({dev})"))
    mapping = sorted(set(mapping), key=lambda x: x[0])
    return mapping


def _probe_indices_fallback(max_idx: int = 10) -> List[Tuple[int, str]]:
    out = []
    is_linux = (platform.system().lower() == "linux")
    for idx in range(max_idx):
        cap = cv2.VideoCapture(idx)
        ok = cap.isOpened()
        if ok:
            name = f"Camera {idx}"
            if is_linux and os.path.exists(f"/dev/video{idx}"):
                name = f"/dev/video{idx}"
            out.append((idx, name))
        cap.release()
    return out


def list_cameras() -> List[Tuple[int, str]]:
    is_linux = (platform.system().lower() == "linux")
    if is_linux and shutil.which("v4l2-ctl"):
        cams = _linux_list_cameras_v4l2()
        if cams:
            return cams
    return _probe_indices_fallback(10)


class LocalEngineWorker(QtCore.QThread):
    log_signal = QtCore.Signal(str)
    result_signal = QtCore.Signal(int, str, str, int, int, bytes, dict)
    start_signal = QtCore.Signal()
    finish_signal = QtCore.Signal()

    def __init__(
            self,
            engine_path: Optional[str],
            left_src: str,
            right_src: str,
            mode: str,
            frame_width: int,
            frame_height: int,
            fps: float,
            session_id: str,
            save_dir: Optional[Path],
            preview: bool,
            use_realsense: bool,
            pose_enabled: bool = False,
            pose_model: Optional[str] = None,
            depth_enabled: bool = True
    ):
        super().__init__()
        self._start_emitted = False
        self._finish_emitted = False

        def _start_wrapper():
            if not self._start_emitted:
                self._start_emitted = True
                self.start_signal.emit()

        def _finish_wrapper():
            if not self._finish_emitted:
                self._finish_emitted = True
                self.finish_signal.emit()

        self.runner = LocalEngineRunner(
            engine_path=engine_path,
            left_src=left_src,
            right_src=right_src,
            mode=mode,
            frame_width=frame_width,
            frame_height=frame_height,
            fps=fps,
            session_id=session_id,
            save_dir=save_dir,
            preview=preview,
            on_log=lambda s: self.log_signal.emit(s),
            on_result=lambda *a: self.result_signal.emit(*a),
            on_start=_start_wrapper,
            on_finish=_finish_wrapper,
            use_realsense=use_realsense,
            pose_enabled=pose_enabled,
            pose_model=pose_model,
            depth_enabled=depth_enabled
        )

    def run(self):
        try:
            self.runner.run()
        except Exception as exc:
            self.log_signal.emit(f"[local] Error: {exc}")
            if not self._finish_emitted:
                self.finish_signal.emit()

    def stop(self):
        self.runner.stop()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FoundationStereo Local Runner")
        self.setMinimumSize(1024, 740)

        w = QtWidgets.QWidget(self)
        layout = QtWidgets.QGridLayout(w)
        self.setCentralWidget(w)

        self._disp_preview_default_scale = 256.0
        self._disp_vis_alpha_rise = 0.35
        self._disp_vis_alpha_decay = 0.08
        self._reset_preview_state()

        self.engine_file = QtWidgets.QLineEdit()
        self.engine_file.setPlaceholderText("TensorRT engine path")
        self.engine_browse = QtWidgets.QPushButton("Browse TensorRT Engine...")

        self.src_mode = QtWidgets.QComboBox(); self.src_mode.addItems(["Live Cameras", "Video Files"])
        self.left_cam = QtWidgets.QSpinBox(); self.left_cam.setRange(0, 9); self.left_cam.setValue(0)
        self.right_cam = QtWidgets.QSpinBox(); self.right_cam.setRange(0, 9); self.right_cam.setValue(1)
        self.show_cams_btn = QtWidgets.QPushButton("Show Cameras")

        self.resolution_display = QtWidgets.QLabel(f"Resolution: {DEFAULT_WIDTH} x {DEFAULT_HEIGHT}")
        self.fps_display = QtWidgets.QLabel(f"FPS: {int(DEFAULT_FPS)}")

        self.left_file = QtWidgets.QLineEdit(); self.left_browse = QtWidgets.QPushButton("Browse...")
        self.right_file = QtWidgets.QLineEdit(); self.right_browse = QtWidgets.QPushButton("Browse...")

        self.output_mode = QtWidgets.QComboBox(); self.output_mode.addItems(["Disparity", "Depth"])
        self.depth_group = QtWidgets.QGroupBox("Depth Options")
        depth_layout = QtWidgets.QGridLayout(self.depth_group)
        self.focal_px_edit = QtWidgets.QLineEdit(); self.focal_px_edit.setPlaceholderText("Focal length (pixels)")
        self.baseline_m_edit = QtWidgets.QLineEdit(); self.baseline_m_edit.setPlaceholderText("Baseline (meters)")
        dblv = QtGui.QDoubleValidator(0.0, 1e9, 6, self)
        self.focal_px_edit.setValidator(dblv)
        self.baseline_m_edit.setValidator(dblv)
        depth_layout.addWidget(QtWidgets.QLabel("Focal length (px)"), 0, 0)
        depth_layout.addWidget(self.focal_px_edit, 0, 1)
        depth_layout.addWidget(QtWidgets.QLabel("Baseline(m)"), 1, 0)
        depth_layout.addWidget(self.baseline_m_edit, 1, 1)
        depth_layout.setColumnStretch(1, 1)

        self.pose_group = QtWidgets.QGroupBox("Pose Options")
        pose_layout = QtWidgets.QGridLayout(self.pose_group)
        self.pose_checkbox = QtWidgets.QCheckBox("Enable pose estimation")
        self.pose_model_label = QtWidgets.QLabel("Pose model")
        self.pose_model_combo = QtWidgets.QComboBox()
        for info in list_pose_models():
            self.pose_model_combo.addItem(info.display_name, info.key)
            idx = self.pose_model_combo.count() - 1
            self.pose_model_combo.setItemData(idx, info.description, QtCore.Qt.ItemDataRole.ToolTipRole)
        pose_layout.addWidget(self.pose_checkbox, 0, 0, 1, 2)
        pose_layout.addWidget(self.pose_model_label, 1, 0)
        pose_layout.addWidget(self.pose_model_combo, 1, 1)
        pose_layout.setColumnStretch(1, 1)

        self.depth_enabled_checkbox = QtWidgets.QCheckBox("Run depth pipeline")
        self.depth_enabled_checkbox.setChecked(True)

        self.use_realsense = QtWidgets.QCheckBox("Use RealSense")

        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn = QtWidgets.QPushButton("Stop"); self.stop_btn.setEnabled(False)

        self.save_dir = QtWidgets.QLineEdit()
        self.save_browse = QtWidgets.QPushButton("Save to...")

        self.log_box = QtWidgets.QPlainTextEdit(); self.log_box.setReadOnly(True)
        self.preview_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(280)
        self.preview_label.setStyleSheet("background: #111; color: #aaa; border: 1px solid #333")

        r = 0
        layout.addWidget(QtWidgets.QLabel("TensorRT Engine"), r, 0)
        layout.addWidget(self.engine_file, r, 3, 1, 2)
        layout.addWidget(self.engine_browse, r, 5); r += 1

        layout.addWidget(QtWidgets.QLabel("Source"), r, 0); layout.addWidget(self.src_mode, r, 1, 1, 2)
        layout.addWidget(self.use_realsense, r, 3, 1, 2); r += 1

        cam_box = QtWidgets.QHBoxLayout()
        cam_box.addWidget(QtWidgets.QLabel("Left Cam #")); cam_box.addWidget(self.left_cam)
        cam_box.addSpacing(10)
        cam_box.addWidget(QtWidgets.QLabel("Right Cam #")); cam_box.addWidget(self.right_cam)
        cam_box.addStretch(1)
        cam_box.addWidget(self.show_cams_btn)
        layout.addLayout(cam_box, r, 0, 1, 6); r += 1

        res_box = QtWidgets.QHBoxLayout()
        res_box.addWidget(self.resolution_display)
        res_box.addSpacing(12)
        res_box.addWidget(self.fps_display)
        res_box.addStretch(1)
        layout.addLayout(res_box, r, 0, 1, 6); r += 1

        file_box = QtWidgets.QGridLayout()
        file_box.addWidget(QtWidgets.QLabel("Left file"), 0, 0); file_box.addWidget(self.left_file, 0, 1); file_box.addWidget(self.left_browse, 0, 2)
        file_box.addWidget(QtWidgets.QLabel("Right file"), 1, 0); file_box.addWidget(self.right_file, 1, 1); file_box.addWidget(self.right_browse, 1, 2)
        layout.addLayout(file_box, r, 0, 1, 6); r += 1

        layout.addWidget(QtWidgets.QLabel("Output"), r, 0); layout.addWidget(self.output_mode, r, 1)
        layout.addWidget(self.depth_enabled_checkbox, r, 2, 1, 2); r += 1
        layout.addWidget(self.depth_group, r, 0, 1, 6); r += 1
        layout.addWidget(self.pose_group, r, 0, 1, 6); r += 1

        save_box = QtWidgets.QHBoxLayout()
        save_box.addWidget(QtWidgets.QLabel("Save dir"))
        save_box.addWidget(self.save_dir); save_box.addWidget(self.save_browse)
        layout.addLayout(save_box, r, 0, 1, 6); r += 1

        layout.addWidget(self.start_btn, r, 4); layout.addWidget(self.stop_btn, r, 5); r += 1

        layout.addWidget(self.preview_label, r, 0, 1, 6); r += 1
        layout.addWidget(self.log_box, r, 0, 1, 6); r += 1

        self._connect_signals()

        self.worker: Optional[LocalEngineWorker] = None
        self._update_source_rows()
        self._sync_depth_controls()
        self._toggle_realsense_capture()

    def _connect_signals(self) -> None:
        self.engine_browse.clicked.connect(self._browse_engine)
        self.src_mode.currentIndexChanged.connect(self._update_source_rows)
        self.show_cams_btn.clicked.connect(self._show_cameras)
        self.left_browse.clicked.connect(self._browse_left)
        self.right_browse.clicked.connect(self._browse_right)
        self.save_browse.clicked.connect(self._browse_save)
        self.output_mode.currentIndexChanged.connect(self._sync_depth_controls)
        self.use_realsense.toggled.connect(self._toggle_realsense_capture)
        self.use_realsense.toggled.connect(self._sync_depth_controls)
        self.pose_checkbox.toggled.connect(self._sync_depth_controls)
        self.depth_enabled_checkbox.toggled.connect(self._sync_depth_controls)
        self.start_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self._stop)

    def _reset_preview_state(self):
        self._disp_preview_scale = self._disp_preview_default_scale
        self._disp_preview_max: Optional[float] = None
        self._disp_preview_session: Optional[str] = None
        self._disp_vis_norm: Optional[float] = None
        self._preview_buffer: Optional[np.ndarray] = None

    def _show_message(self, level: str, title: str, text: str) -> None:
        handlers = {
            "information": QtWidgets.QMessageBox.information,
            "warning": QtWidgets.QMessageBox.warning,
            "critical": QtWidgets.QMessageBox.critical,
        }
        handler = handlers.get(level)
        if handler:
            handler(self, title, text)

    @staticmethod
    def _coerce_float(value: object) -> Optional[float]:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                return float(text)
            except ValueError:
                return None
        return None

    def _selected_engine_path(self) -> Optional[Path]:
        text = self.engine_file.text().strip()
        if not text:
            self._show_message(
                "warning",
                "TensorRT Engine",
                "Please select a TensorRT engine before running locally.",
            )
            return None

        path = Path(text)
        if not path.exists():
            self._show_message(
                "warning",
                "TensorRT Engine",
                "Selected TensorRT engine file does not exist.",
            )
            return None

        return path

    def _selected_save_dir(self) -> Optional[Path]:
        text = self.save_dir.text().strip()
        if not text:
            return None
        return Path(text).expanduser()

    def _configure_depth_parameters(self) -> None:
        if not self.depth_enabled_checkbox.isChecked():
            return

        if self.output_mode.currentText() != "Depth":
            return

        if self.depth_fx is None or self.depth_baseline_m is None:
            fx = self._coerce_float(self.focal_px_edit.text())
            bl = self._coerce_float(self.baseline_m_edit.text())
            if (fx is None or bl is None) and not self.use_realsense.isChecked():
                self._show_message(
                    "warning",
                    "Depth parameters",
                    "Manual depth requires BOTH focal length (pixels) and baseline (meters).",
                )

    def _determine_sources(
            self, use_realsense: bool
    ) -> Optional[Tuple[str, str, str]]:
        if self.src_mode.currentText() == "Video Files" and not use_realsense:
            left = self.left_file.text().strip()
            right = self.right_file.text().strip()
            if not (left and right):
                self._log("Please select valid left/right files.")
                return None

            left_path, right_path = Path(left), Path(right)
            if not (left_path.exists() and right_path.exists()):
                self._log("Please select valid left/right files.")
                return None

            return left, right, "file"

        if use_realsense:
            return "realsense", "realsense", "stream"

        return str(self.left_cam.value()), str(self.right_cam.value()), "stream"

    def _build_session_config(self, engine_path: Optional[Path]) -> Optional[SessionConfig]:
        use_realsense = self.use_realsense.isChecked()
        sources = self._determine_sources(use_realsense)
        if sources is None:
            return None

        left_src, right_src, mode = sources
        session_id = f"gui-{int(time.time())}"

        pose_enabled = self.pose_checkbox.isChecked()
        pose_model = (
            self.pose_model_combo.currentData(QtCore.Qt.ItemDataRole.UserRole)
            if pose_enabled
            else None
        )
        depth_enabled = self.depth_enabled_checkbox.isChecked()

        return SessionConfig(
            engine_path=str(engine_path) if engine_path else None,
            left_src=left_src,
            right_src=right_src,
            mode=mode,
            frame_width=DEFAULT_WIDTH,
            frame_height=DEFAULT_HEIGHT,
            fps=DEFAULT_FPS,
            session_id=session_id,
            save_dir=self._selected_save_dir(),
            preview=False,
            use_realsense=use_realsense,
            pose_enabled=pose_enabled,
            pose_model=pose_model,
            depth_enabled=depth_enabled
        )

    def _attach_worker_signals(self, worker: LocalEngineWorker) -> None:
        worker.log_signal.connect(self._log)
        worker.result_signal.connect(self._handle_worker_result)
        worker.start_signal.connect(self._handle_worker_started)
        worker.finish_signal.connect(self._handle_worker_finished)

    def _handle_worker_result(self, seq: int, kind: str, fmt: str, width: int, height: int, payload: bytes, meta: dict) -> None:
        kind = (kind or "").lower()
        if kind == "pose_preview":
            self._on_pose_preview(payload, meta)
            return
        if kind == "disparity":
            self._on_result_image(payload, meta)
            return
        # Fallback: try to interpret as a generic colour frame
        self._on_pose_preview(payload, meta)

    def _browse_engine(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select TensorRT Engine",
            "",
            "TensorRT Engines (*.engine);;All Files (*)"
        )
        if fn:
            self.engine_file.setText(fn)

    def _update_source_rows(self):
        use_rs = self.use_realsense.isChecked()
        is_files = (self.src_mode.currentText() == "Video Files") and not use_rs
        for w in [self.left_cam, self.right_cam, self.show_cams_btn]:
            w.setEnabled(not is_files and not use_rs)
        for w in [self.left_file, self.left_browse, self.right_file, self.right_browse]:
            w.setEnabled(is_files)

    def _toggle_realsense_capture(self):
        use_rs = self.use_realsense.isChecked()
        if use_rs:
            if self.src_mode.currentText() != "Live Cameras":
                self.src_mode.setCurrentIndex(0)
        self.src_mode.setEnabled(not use_rs)
        self._update_source_rows()

    def _sync_depth_controls(self):
        depth_enabled = self.depth_enabled_checkbox.isChecked()
        is_depth_mode = depth_enabled and (self.output_mode.currentText() == "Depth")
        self.output_mode.setEnabled(depth_enabled)
        self.depth_group.setEnabled(depth_enabled)
        rs = self.use_realsense.isChecked()
        self.focal_px_edit.setEnabled(is_depth_mode and not rs)
        self.baseline_m_edit.setEnabled(is_depth_mode and not rs)
        pose_controls_visible = self.pose_checkbox.isChecked()
        for w in [self.pose_model_label, self.pose_model_combo]:
            w.setVisible(pose_controls_visible)
            w.setEnabled(pose_controls_visible)
        self.pose_group.setEnabled(True)

    def _show_cameras(self):
        cams = list_cameras()
        if not cams:
            QtWidgets.QMessageBox.warning(self, "Cameras", "No camera found")
            return
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Cameras")
        dlg.setMinimumWidth(520)
        v = QtWidgets.QVBoxLayout(dlg)

        table = QtWidgets.QTableWidget(len(cams), 2, dlg)
        table.setHorizontalHeaderLabels(["ID", "Name / Path"])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        table.horizontalHeader().setStretchLastSection(True)

        for r, (idx, name) in enumerate(sorted(cams, key=lambda x: x[0])):
            table.setItem(r, 0, QtWidgets.QTableWidgetItem(str(idx)))
            table.setItem(r, 1, QtWidgets.QTableWidgetItem(str(name)))

        v.addWidget(table)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(dlg.reject)
        btns.accepted.connect(dlg.accept)
        v.addWidget(btns)

        dlg.exec()

    def _browse_left(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Left Video", "", VIDEO_FILE_FILTER
        )
        if fn:
            self.left_file.setText(fn)

    def _browse_right(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Right Video", "", VIDEO_FILE_FILTER
        )
        if fn:
            self.right_file.setText(fn)

    def _browse_save(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if d: self.save_dir.setText(d)

    def _start(self):
        if self.worker is not None:
            self._log("A processing session is already running.")
            return
        depth_enabled = self.depth_enabled_checkbox.isChecked()
        engine_path: Optional[Path] = None
        if depth_enabled:
            engine_path = self._selected_engine_path()
            if engine_path is None:
                return
            try:
                ensure_tensorrt_runtime()
            except TensorRTUnavailableError as exc:
                self._show_message("critical", "TensorRT", str(exc))
                return
            self._log(f"[local] Selected TensorRT engine: {engine_path}")
        elif not self.pose_checkbox.isChecked():
            self._show_message(
                "warning",
                "Configuration",
                "Enable the depth pipeline or pose estimation before starting.",
            )
            return
        else:
            self._log("[local] Depth pipeline disabled; running pose estimation without TensorRT engine.")
        self.depth_fx, self.depth_baseline_m = None, None
        self._configure_depth_parameters()

        config = self._build_session_config(engine_path)
        if config is None:
            return

        self._reset_preview_state()
        self.worker = LocalEngineWorker(**asdict(config))
        self._attach_worker_signals(self.worker)
        self.worker.start()

    def _stop(self):
        if self.worker:
            self.worker.stop()
        else:
            self._log("No active worker.")

    def _handle_worker_started(self):
        self._set_running(True)

    def _handle_worker_finished(self):
        self._set_running(False)
        sender = self.sender()
        if sender is self.worker or self.worker is None:
            self.worker = None

    def _set_running(self, running: bool):
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)

    def _log(self, msg: str):
        self.log_box.appendPlainText(msg)

    def _decode_disparity(self, disp_img: np.ndarray) -> np.ndarray:
        disp = disp_img.astype(np.float32)
        if disp_img.dtype == np.uint16:
            scale = self._disp_preview_scale if self._disp_preview_scale and self._disp_preview_scale > 0 else self._disp_preview_default_scale
            if scale <= 0:
                scale = 1.0
            disp = disp / float(scale)
        disp = np.nan_to_num(disp, nan=0.0, posinf=0.0, neginf=0.0)
        if self._disp_preview_max is not None:
            disp = np.clip(disp, 0.0, float(self._disp_preview_max))
        else:
            disp = np.maximum(disp, 0.0)
        return disp

    def _overlay_poses(self, vis_bgr: np.ndarray, meta_payload: dict) -> np.ndarray:
        payload = meta_payload or {}
        poses = payload.get("poses") or []
        if not poses:
            return vis_bgr
        # Edges for COCO-17-ish layouts
        pose_meta = payload.get("pose") or {}
        model_key = pose_meta.get("model")
        if not model_key:
            display_name = pose_meta.get("model_display")
            if display_name:
                for info in list_pose_models():
                    if info.display_name == display_name:
                        model_key = info.key
                        break
        if model_key:
            model_key = str(model_key).lower()

        # Convert pose dictionaries into numpy arrays compatible with draw_skeletons.
        pose_arrays: List[np.ndarray] = []
        for entry in poses:
            kps = entry.get("keypoints_uvc") if isinstance(entry, dict) else None
            if not kps:
                continue
            arr = np.asarray(kps, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[0] == 0:
                continue
            if arr.shape[1] == 2:
                ones = np.ones((arr.shape[0], 1), dtype=np.float32)
                arr = np.concatenate([arr, ones], axis=1)
            elif arr.shape[1] > 3:
                arr = arr[:, :3]
            pose_arrays.append(arr)

        if not pose_arrays:
            return vis_bgr

        try:
            return draw_skeletons(vis_bgr, pose_arrays, model_key)
        except Exception:
            # Fall back to a simple COCO-17 style skeleton if specialised drawing fails.
            edges = get_skeleton_edges(model_key)
            out = vis_bgr.copy()
            for arr in pose_arrays:
                for (u, v, c) in arr:
                    if c >= 0.2 and np.isfinite(u) and np.isfinite(v):
                        cv2.circle(out, (int(round(u)), int(round(v))), 2, (0, 255, 0), -1)
                for a, b in edges:
                    if a < arr.shape[0] and b < arr.shape[0]:
                        ua, va, ca = arr[a]
                        ub, vb, cb = arr[b]
                        if min(ca, cb) >= 0.2 and all(np.isfinite(x) for x in (ua, va, ub, vb)):
                            cv2.line(
                                out,
                                (int(round(ua)), int(round(va))),
                                (int(round(ub)), int(round(vb))),
                                (255, 200, 0),
                                2,
                            )
            return out


    def _update_preview_pixmap(self, vis_rgb: np.ndarray) -> None:
        if vis_rgb.ndim == 2:
            vis_rgb = np.stack([vis_rgb] * 3, axis=-1)
        vis_rgb = np.ascontiguousarray(vis_rgb)
        self._preview_buffer = vis_rgb
        hq, wq = vis_rgb.shape[:2]
        qimg = QtGui.QImage(
            vis_rgb.data,
            wq,
            hq,
            vis_rgb.strides[0],
            QtGui.QImage.Format.Format_RGB888,
        )
        pix = QtGui.QPixmap.fromImage(qimg).scaled(
            self.preview_label.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setPixmap(pix)


    def _on_pose_preview(self, payload: bytes, meta: dict) -> None:
        arr = np.frombuffer(payload, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return

        meta_payload: dict = meta if isinstance(meta, dict) else {}
        session_id = meta_payload.get("session_id")
        if session_id is not None:
            session_str = str(session_id)
            if session_str != self._disp_preview_session:
                self._disp_preview_session = session_str
                self._disp_vis_norm = None
                self._disp_preview_max = None
                self._disp_preview_scale = self._disp_preview_default_scale

        vis = self._overlay_poses(img, meta_payload)
        fps_est = self._coerce_float(meta_payload.get("fps_est"))
        if fps_est is not None and fps_est > 0:
            cv2.putText(
                vis,
                f"{fps_est:.1f} FPS",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        self._update_preview_pixmap(vis_rgb)


    def _visualize_disparity(self, disparity: np.ndarray) -> np.ndarray:
        disp = np.nan_to_num(disparity.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if self._disp_preview_max is not None:
            disp = np.clip(disp, 0.0, float(self._disp_preview_max))
        positives = disp[disp > 0]
        if positives.size:
            hi_candidate = float(np.percentile(positives, 99.5))
            if self._disp_preview_max is not None:
                hi_candidate = min(hi_candidate, float(self._disp_preview_max))
        else:
            hi_candidate = self._disp_preview_max or (self._disp_vis_norm or 1.0)
        hi_candidate = max(hi_candidate, 1e-3)
        if self._disp_vis_norm is None:
            self._disp_vis_norm = hi_candidate
        else:
            alpha = self._disp_vis_alpha_rise if hi_candidate > self._disp_vis_norm else self._disp_vis_alpha_decay
            self._disp_vis_norm = (1 - alpha) * self._disp_vis_norm + alpha * hi_candidate
        norm = max(self._disp_vis_norm if self._disp_vis_norm else hi_candidate, 1e-3)
        vis_gray = np.clip((disp / norm) * 255.0, 0.0, 255.0).astype(np.uint8)
        vis_gray[disp <= 0] = 0
        vis = cv2.applyColorMap(vis_gray, cv2.COLORMAP_INFERNO)
        return vis

    def _depth_from_disparity(self, disparity: np.ndarray, fx: float, baseline_m: float) -> np.ndarray:
        d = np.nan_to_num(disparity.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        eps = 1e-6
        denom = np.maximum(d, eps)
        depth = (fx * baseline_m) / denom
        depth[d <= eps] = 0.0
        depth[~np.isfinite(depth)] = 0.0
        return depth

    def _visualize_depth(self, depth_m: np.ndarray) -> np.ndarray:
        dm = depth_m.copy()
        dm = np.clip(dm, 0.0, 5.0)
        inv = (1.0 - (dm / 5.0)) * 255.0
        vis = inv.astype(np.uint8)
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
        return vis

    def _on_result_image(self, payload: bytes, meta: dict):
        arr = np.frombuffer(payload, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            return

        meta_payload: dict = meta if isinstance(meta, dict) else {}
        session_id = meta_payload.get("session_id")
        if session_id is not None:
            session_str = str(session_id)
            if session_str != self._disp_preview_session:
                self._disp_preview_session = session_str
                self._disp_vis_norm = None
                self._disp_preview_max = None
                self._disp_preview_scale = self._disp_preview_default_scale

        scale_val = self._coerce_float(meta_payload.get("disp_scale"))
        if scale_val is not None and scale_val > 0:
            self._disp_preview_scale = scale_val
        elif self._disp_preview_scale is None or self._disp_preview_scale <= 0:
            self._disp_preview_scale = self._disp_preview_default_scale

        max_val = self._coerce_float(meta_payload.get("max_disp"))
        if max_val is not None and max_val > 0:
            self._disp_preview_max = max_val

        if self.use_realsense.isChecked():
            fx_px = self._coerce_float(meta_payload.get("fx_px"))
            bl_m = self._coerce_float(meta_payload.get("baseline_m"))
            if fx_px and bl_m and (self.depth_fx is None or self.depth_baseline_m is None):
                self.depth_fx, self.depth_baseline_m = fx_px, bl_m
                self._log(f"[depth] RealSense calibration received: fx={fx_px:.2f}, baseline={bl_m:.6f} m")

        want_depth = (self.output_mode.currentText() == "Depth")
        disparity = self._decode_disparity(img)
        if want_depth and self.depth_fx and self.depth_baseline_m:
            depth_m = self._depth_from_disparity(disparity, self.depth_fx, self.depth_baseline_m)
            vis = self._visualize_depth(depth_m)
        else:
            vis = self._visualize_disparity(disparity)

        # Overlay poses if provided in meta
        vis = self._overlay_poses(vis, meta_payload)

        if vis.ndim == 2:
            vis_rgb = cv2.cvtColor(vis, cv2.COLOR_GRAY2RGB)
        else:
            vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        self._update_preview_pixmap(vis_rgb)

def launch_gui():
    app = QtWidgets.QApplication([])
    mw = MainWindow()
    mw.show()
    return app.exec()