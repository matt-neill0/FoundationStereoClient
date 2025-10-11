from typing import Optional, List, Tuple
import os, time, pathlib, subprocess, platform, re, contextlib
import numpy as np, cv2
from PySide6 import QtCore, QtGui, QtWidgets
from sender_core import StereoSenderClient
from local_inference import LocalEngineRunner, TensorRTUnavailableError, ensure_tensorrt_runtime
from main import DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_FPS

def shutil_which(cmd: str) -> bool:
    from shutil import which
    return which(cmd) is not None

def _linux_list_cameras_v4l2() -> List[Tuple[int, str]]:
    try:
        proc = subprocess.run(
            ['v4l2-ctl', "--list-devices"],
            check = True, text = True, capture_output = True
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
    mapping = sorted(set(mapping), key = lambda x: x[0])
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
    if is_linux and shutil_which("v4l2-ctl"):
        cams = _linux_list_cameras_v4l2()
        if cams:
            return cams
    return _probe_indices_fallback(10)

def try_get_realsense_fx_baseline() -> Optional[Tuple[float, float]]:
    try:
        import pyrealsense2 as rs
    except Exception:
        return None
    try:
        ctx = rs.context()
        if len(ctx.devices) == 0:
            return None
        for dev in ctx.devices:
            name = dev.get_info(rs.camera_info.name)
            pipe = rs.pipeline()
            cfg = rs.config()

            cfg.enable_stream(rs.stream.left, 640, 480, rs.format.y8, 30)
            cfg.enable_stream(rs.stream.right, 640, 480, rs.format.y8, 30)
            profile = pipe.start(cfg)
            try:
                sp_left = profile.get_stream(rs.stream.left).as_video_stream_profile()
                sp_right = profile.get_stream(rs.stream.right).as_video_stream_profile()
                intr = sp_left.get_intrinsics()
                fx = float(intr.fx)

                extr = sp_left.get_extrinsics_to(sp_right)
                baseline_m = float(abs(extr.translation[0]))
                if fx > 0 and baseline_m > 0:
                    return fx, baseline_m
            finally:
                with contextlib.suppress(Exception):
                    pipe.stop()
        return None
    except Exception:
        return None

class ClientWorker(QtCore.QThread):
    log_signal = QtCore.Signal(str)
    result_signal = QtCore.Signal(int, str, str, int, int, bytes, dict)
    start_signal = QtCore.Signal()
    finish_signal = QtCore.Signal()

    def __init__(self, client: StereoSenderClient, left_src: str, right_src, mode: str):
        super().__init__()
        self.client = client
        self.left_src = left_src
        self.right_src = right_src
        self.mode = mode
        self.client._on_log = lambda s: self.log_signal.emit(s)
        self.client._on_result = lambda *args: self.result_signal.emit(*args)
        self.client._on_start = lambda: self.start_signal.emit()
        self.client._on_finish = lambda: self.finish_signal.emit()

    def run(self):
        import asyncio
        with contextlib.suppress(Exception):
            asyncio.run(self.client.start_stream(self.left_src, self.right_src, self.mode))

    def stop(self):
        self.client.stop()

class LocalEngineWorker(QtCore.QThread):
    log_signal = QtCore.Signal(str)
    result_signal = QtCore.Signal(int, str, str, int, int, bytes, dict)
    start_signal = QtCore.Signal()
    finish_signal = QtCore.Signal()

    def __init__(
        self,
        engine_path: str,
        left_src: str,
        right_src: str,
        mode: str,
        frame_width: int,
        frame_height: int,
        fps: float,
        session_id: str,
        save_dir: Optional[pathlib.Path],
        preview: bool
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
        self.setWindowTitle("FoundationStereo Sender")
        self.setMinimumSize(1024, 740)

        w = QtWidgets.QWidget(self)
        layout = QtWidgets.QGridLayout(w)
        self.setCentralWidget(w)

        self._disp_preview_default_scale = 256.0
        self._disp_vis_alpha_rise = 0.35
        self._disp_vis_alpha_decay = 0.08
        self._reset_preview_state()

        self.host = QtWidgets.QLineEdit("jetson.local")
        self.port = QtWidgets.QSpinBox(); self.port.setMaximum(83353); self.port.setValue(8765)
        self.path = QtWidgets.QLineEdit("/foundation-stereo")
        self.send_to_server = QtWidgets.QCheckBox("Send to server")
        self.send_to_server.setChecked(False)
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
        self.use_realsense = QtWidgets.QCheckBox("Use Realsense")
        self.focal_px_edit = QtWidgets.QLineEdit(); self.focal_px_edit.setPlaceholderText("Focal length (pixels)")
        self.baseline_m_edit = QtWidgets.QLineEdit(); self.baseline_m_edit.setPlaceholderText("Baseline (meters)")
        dblv = QtGui.QDoubleValidator(0.0, 1e9, 6, self)
        self.focal_px_edit.setValidator(dblv)
        self.baseline_m_edit.setValidator(dblv)
        depth_layout.addWidget(self.use_realsense, 0, 0, 1, 2)
        depth_layout.addWidget(QtWidgets.QLabel("Focal length (px)"), 1, 0)
        depth_layout.addWidget(self.focal_px_edit, 1, 1)
        depth_layout.addWidget(QtWidgets.QLabel("Baseline(m)"), 2, 0)
        depth_layout.addWidget(self.baseline_m_edit, 2, 1)

        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn = QtWidgets.QPushButton("Stop"); self.stop_btn.setEnabled(False)

        self.save_dir = QtWidgets.QLineEdit()
        self.save_browse = QtWidgets.QPushButton("Save to...")
        self.preview = QtWidgets.QCheckBox("Preview results"); self.stop_btn.setChecked(True)

        self.log_box = QtWidgets.QPlainTextEdit(); self.log_box.setReadOnly(True)
        self.preview_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(280)
        self.preview_label.setStyleSheet("background: #111; color: #aaa; border: 1px solid #333")

        r = 0
        layout.addWidget(QtWidgets.QLabel("Host"), r, 0); layout.addWidget(self.host, r, 1)
        layout.addWidget(QtWidgets.QLabel("Port"), r, 2); layout.addWidget(self.port, r, 3)
        layout.addWidget(QtWidgets.QLabel("Path"), r, 4); layout.addWidget(self.path, r, 5); r += 1

        layout.addWidget(self.send_to_server, r, 0, 1, 2)
        layout.addWidget(QtWidgets.QLabel("TensorRT Engine"), r, 2)
        layout.addWidget(self.engine_file, r, 3, 1, 2)
        layout.addWidget(self.engine_browse, r, 5); r += 1

        layout.addWidget(QtWidgets.QLabel("Source"), r, 0); layout.addWidget(self.src_mode, r, 1, 1, 2); r += 1

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

        layout.addWidget(QtWidgets.QLabel("Output"), r, 0); layout.addWidget(self.output_mode, r, 1); r += 1
        layout.addWidget(self.depth_group, r, 0, 1, 6); r += 1

        save_box = QtWidgets.QHBoxLayout()
        save_box.addWidget(QtWidgets.QLabel("Save dir"))
        save_box.addWidget(self.save_dir); save_box.addWidget(self.save_browse)
        layout.addLayout(save_box, r, 0, 1, 6); r += 1

        layout.addWidget(self.preview, r, 0, 1, 2)
        layout.addWidget(self.start_btn, r, 4); layout.addWidget(self.stop_btn, r, 5); r += 1

        layout.addWidget(self.preview_label, r, 0, 1, 6); r += 1
        layout.addWidget(self.log_box, r, 0, 1, 6); r += 1

        self.send_to_server.toggled.connect(self._toggle_server_destination)
        self.engine_browse.clicked.connect(self._browse_engine)
        self.src_mode.currentIndexChanged.connect(self._update_source_rows)
        self.show_cams_btn.clicked.connect(self._show_cameras)
        self.left_browse.clicked.connect(self._browse_left)
        self.right_browse.clicked.connect(self._browse_right)
        self.save_browse.clicked.connect(self._browse_save)
        self.output_mode.currentIndexChanged.connect(self._toggle_depth_controls)
        self.use_realsense.toggled.connect(self._toggle_depth_controls)
        self.start_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self._stop)

        self.worker: Optional[ClientWorker] = None
        self._update_source_rows()
        self._toggle_server_destination()
        self._toggle_depth_controls()

    def _reset_preview_state(self):
        self._disp_preview_scale = self._disp_preview_default_scale
        self._disp_preview_max: Optional[float] = None
        self._disp_preview_session: Optional[str] = None
        self._disp_vis_norm: Optional[float] = None

    def _toggle_server_destination(self):
        send = self.send_to_server.isChecked()
        for w in [self.host, self.port, self.path]:
            w.setEnabled(send)
        self.engine_file.setEnabled(not send)
        self.engine_browse.setEnabled(not send)

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
        is_files = (self.src_mode.currentText() == "Video Files")
        for w in [self.left_cam, self.right_cam]:
            w.setEnabled(not is_files)
        for w in [self.left_file, self.left_browse, self.right_file, self.right_browse]:
            w.setEnabled(is_files)

    def _toggle_depth_controls(self):
        is_depth = (self.output_mode.currentText() == "Depth")
        self.depth_group.setEnabled(is_depth)
        rs = self.use_realsense.isChecked()
        self.focal_px_edit.setEnabled(is_depth and not rs)
        self.baseline_m_edit.setEnabled(is_depth and not rs)

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

        for r, (idx, name) in enumerate(sorted(cams, key = lambda x: x[0])):
            table.setItem(r, 0, QtWidgets.QTableWidgetItem(str(idx)))
            table.setItem(r, 1, QtWidgets.QTableWidgetItem(str(name)))

        v.addWidget(table)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(dlg.reject)
        btns.accepted.connect(dlg.accept)
        v.addWidget(btns)

        dlg.exec()

    def _browse_left(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Left Video", "", "Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*)")
        if fn: self.left_file.setText(fn)

    def _browse_right(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Right Video", "", "Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*)")
        if fn: self.right_file.setText(fn)

    def _browse_save(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if d: self.save_dir.setText(d)

    def _start(self):
        send_to_server = self.send_to_server.isChecked()
        engine_path = self.engine_file.text().strip()
        if not send_to_server:
            if not engine_path:
                QtWidgets.QMessageBox.warning(
                    self,
                    "TensorRT Engine",
                    "Please select a TensorRT engine before running locally.",
                )
                return
            if not os.path.exists(engine_path):
                QtWidgets.QMessageBox.warning(
                    self,
                    "TensorRT Engine",
                    "Selected TensorRT engine file does not exist.",
                )
                return
            try:
                ensure_tensorrt_runtime()
            except TensorRTUnavailableError as exc:
                QtWidgets.QMessageBox.critical(self, "TensorRT", str(exc))
                return
            self._log(f"[local] Selected TensorRT engine: {engine_path}")
        self.depth_fx, self.depth_baseline_m = None, None
        if self.output_mode.currentText() == "Depth":
            if self.use_realsense.isChecked():
                params = try_get_realsense_fx_baseline()
                if params is None:
                    QtWidgets.QMessageBox.warning(self, "RealSense",
                        "Could not read RealSense intrinsics on this host. "
                        "Install `pyrealsense2` and connect a supported device, "
                        "or enter focal length (px) and baseline (m) manually.")
                else:
                    self.depth_fx, self.depth_baseline_m = params
                    self._log(f"[depth] Using RealSense params: fx={self.depth_fx:.2f}, baseline={self.depth_baseline_m:.6f} m")
            if self.depth_fx is None or self.depth_baseline_m is None:
                try:
                    fx = float(self.focal_px_edit.text()) if self.focal_px_edit.text().strip() else None
                    bl = float(self.baseline_m_edit.text()) if self.baseline_m_edit.text().strip() else None
                except ValueError:
                    fx, bl = None, None
                if fx is None or bl is None:
                    QtWidgets.QMessageBox.warning(self, "Depth parameters",
                                                  "Manual depth requires BOTH focal length (pixels) and baseline (meters).")
                else:
                    self.depth_fx, self.depth_baseline_m = fx, bl
                    self._log(f"[depth] Using manual params: fx={fx:.2f}, baseline={bl:.6f} m")

        fps = DEFAULT_FPS
        frame_w, frame_h = DEFAULT_WIDTH, DEFAULT_HEIGHT

        save_dir = pathlib.Path(self.save_dir.text()) if self.save_dir.text().strip() else None
        session_id = f"gui-{int(time.time())}"
        preview = self.preview.isChecked()

        if self.src_mode.currentText() == "Video Files":
            left_src = self.left_file.text().strip()
            right_src = self.right_file.text().strip()
            if not (left_src and right_src and os.path.exists(left_src) and os.path.exists(right_src)):
                self._log("Please select valid left/right files.")
                return
            mode = "file"
        else:
            left_src = str(self.left_cam.value())
            right_src = str(self.right_cam.value())
            mode = "stream"

        self._reset_preview_state()

        if send_to_server:
            host = self.host.text().strip()
            port = self.port.value()
            path = self.path.text().strip() or "/foundation-stereo"
            client = StereoSenderClient(
                host,
                port,
                path,
                fps = fps,
                session_id=session_id,
                frame_width=frame_w,
                frame_height=frame_h,
                save_dir=save_dir,
                preview=preview,
                on_log=self._log,
                on_result=self._on_result_image,
                on_start=lambda: self._set_running(True),
                on_finish=lambda: self._set_running(False),
            )
            self.worker = ClientWorker(client, left_src, right_src, mode)
        else:
            self.worker = LocalEngineWorker(
                engine_path=engine_path,
                left_src=left_src,
                right_src=right_src,
                mode=mode,
                frame_width=frame_w,
                frame_height=frame_h,
                fps=fps,
                session_id=session_id,
                save_dir=save_dir,
                preview=preview,
            )
        self.worker.log_signal.connect(self._log)
        self.worker.result_signal.connect(self._on_result_image)
        self.worker.start_signal.connect(lambda: self._set_running(True))
        self.worker.finish_signal.connect(lambda: self._set_running(False))
        self.worker.start()

    def _stop(self):
        if self.worker:
            self.worker.stop()
        else:
            self._log("No active worker.")

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

    def _on_result_image(self, seq: int, kind: str, enc: str, w: int, h: int, payload: bytes, meta: dict):
        arr = np.frombuffer(payload, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            return

        meta_payload: dict = {}
        if isinstance(meta, dict):
            meta_payload = meta.get("meta") if isinstance(meta.get("meta"), dict) else meta
        session_id = meta_payload.get("session_id")
        if session_id is not None:
            session_str = str(session_id)
            if session_str != self._disp_preview_session:
                self._disp_preview_session = session_str
                self._disp_vis_norm = None
                self._disp_preview_max = None
                self._disp_preview_scale = self._disp_preview_default_scale

        def _try_float(value) -> Optional[float]:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        scale_val = _try_float(meta_payload.get("disp_scale"))
        if scale_val is not None and scale_val > 0:
            self._disp_preview_scale = scale_val
        elif self._disp_preview_scale is None or self._disp_preview_scale <= 0:
            self._disp_preview_scale = self._disp_preview_default_scale

        max_val = _try_float(meta_payload.get("max_disp"))
        if max_val is not None and max_val > 0:
            self._disp_preview_max = max_val

        want_depth = (self.output_mode.currentText() == "Depth")
        vis: np.ndarray
        if kind == "disparity":
            disparity = self._decode_disparity(img)
            if want_depth and self.depth_fx and self.depth_baseline_m:
                depth_m = self._depth_from_disparity(disparity, self.depth_fx, self.depth_baseline_m)
                vis = self._visualize_depth(depth_m)
            else:
                vis = self._visualize_disparity(disparity)
        else:
            if img.dtype == np.uint16:
                nz = img[img > 0]
                m = nz.mean() if nz.size else 1.0
                disp = np.clip((img.astype(np.float32) / (3.0 * m)) * 255.0, 0, 255).astype(np.uint8)
                vis = cv2.applyColorMap(disp, cv2.COLORMAP_INFERNO)
            else:
                vis = img
                if vis.ndim == 2:
                    vis = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)

        if vis.ndim == 2:
            vis_rgb = cv2.cvtColor(vis, cv2.COLOR_GRAY2RGB)
        else:
            vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        hq, wq = vis_rgb.shape[:2]
        qimg = QtGui.QImage(vis_rgb.data, wq, hq, vis_rgb.strides[0], QtGui.QImage.Format.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(self.preview_label.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
        self.preview_label.setPixmap(pix)

def launch_gui():
    app = QtWidgets.QApplication([])
    mw = MainWindow()
    mw.show()
    return app.exec()