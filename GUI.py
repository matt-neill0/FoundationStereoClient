from typing import Optional, List, Tuple
import os, time, pathlib, subprocess, platform, re, math, contextlib
import numpy as np, cv2
from PySide6 import QtCore, QtGui, QtWidgets
from sender_core import StereoSenderClient

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

def _probs_indices_fallback(max_idx: int = 10) -> List[Tuple[int,str]]:
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
    return _probs_indices_fallback(10)

def try_get_realsense_fx_baseline() -> Optional [Tuple[float, float]]:
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
                baseline_m = float(abs(extr.transnlation[0]))
                pipe.stop()
                if fx > 0 and baseline_m > 0:
                    return fx, baseline_m
            finally:
                with contextlib.suppress(Exception):
                    pipe.stop()
        return None
    except Exception:
        return None

class ClientWorker(QtCore.QThread):
    logSignal = QtCore.Signal(str)
    resultSignal = QtCore.Signal(int, str, str, int, int, bytes, dict)
    startSignal = QtCore.Signal()
    finishSignal = QtCore.Signal()

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

    def stop (self):
        self.client.stop()

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FoundationStereo Sender")
        self.setMinimumSize(1024, 740)

        w = QtWidgets.QWidget(self)
        layout = QtWidgets.QGridLayout(w)
        self.setCentralWidget(w)

        self.host = QtWidgets.QLineEdit("jetson.local")
        self.port = QtWidgets.QSpinBox(); self.port.setMaximum(83353); self.port.setValue(8765)
        self.path = QtWidgets.QLineEdit("/foundation-stereo")

        self.src_mode = QtWidgets.QComboBox(); self.src_mode.addItems(["Live Cameras", "Video Files"])
        self.left_cam = QtWidgets.QSpinBox(); self.left_cam.setRange(0, 9); self.left_cam.setValue(0)
        self.right_cam = QtWidgets.QSpinBox(); self.right_cam.setRange(0, 9); self.right_cam.setValue(1)
        self.show_cams_btn = QtWidgets.QPushButton("Show Cameras")

        self.left_file = QtWidgets.QLineEdit(); self.left_browse = QtWidgets.QPushButton("Browse...")
        self.right_file = QtWidgets.QLineEdit(); self.right_browse = QtWidgets.QPushButton("Browse...")

        self.fps = QtWidgets.QDoubleSpinBox(); self.fps.setRange(0, 240); self.fps.setValue(10); self.fps.setSingleStep(1)
        self.jpeg_quality = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal); self.jpeg_quality.setRange(50, 100); self.jpeg_quality.setValue(90)
        self.jpeg_label = QtWidgets.QLabel("JPEG Quality: 90    (size fixed: 480x640)")

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
        layout.addWidget(QtWidgets.QLabel("Path"), r, 4); layout.addWidget(self.port, r, 5); r += 1

        layout.addWidget(QtWidgets.QLabel("Source"), r, 0); layout.addWidget(self.src_mode, r, 1, 1, 2); r += 1

        cam_box = QtWidgets.QHBoxLayout()
        cam_box.addWidget(QtWidgets.QLabel("Left Cam #")); cam_box.addWidget(self.left_cam)
        cam_box.addSpacing(10)
        cam_box.addWidget(QtWidgets.QLabel("Right Cam #")); cam_box.addWidget(self.right_cam)
        cam_box.addStretch(1)
        cam_box.addWidget(self.show_cams_btn)
        layout.addLayout(cam_box, r, 0, 1, 6); r += 1

        file_box = QtWidgets.QGridLayout()
        file_box.addWidget(QtWidgets.QLabel("Left file"), 0, 0); file_box.addWidget(self.left_file, 0, 1); file_box.addWidget(self.left_browse, 0, 2)
        file_box.addWidget(QtWidgets.QLabel("Right file"), 1, 0); file_box.addWidget(self.right_file, 1, 1); file_box.addWidget(self.right_browse, 1, 2)
        layout.addLayout(file_box, r, 0, 1, 6); r += 1

        layout.addWidget(QtWidgets.QLabel("FPS"), r, 0); layout.addWidget(self.fps, r, 1)
        layout.addWidget(self.jpeg_label, r, 2); layout.addWidget(self.jpeg_quality, r, 3, 1, 3); r += 1

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

        self.src_mode.currentIndexChanged.connect(self._update_source_rows)
        self.show_cams_btn.clicked.connect(self._show_cameras)
        self.left_browse.clicked.connect(self._browse_left)
        self.right_browse.clicked.connect(self._browse_right)
        self.save_browse.clicked.connect(self._browse_save)
        self.jpeg_quality.valueChanged.connect(lambda v: self.jpeg_label.setText(f"JPEG Quality: {v}    (size fixed: 480x640)"))
        self.output_mode.currentIndexChanged.connect(self._toggle_depth_controls)
        self.use_realsense.toggled.connect(self._toggle_depth_controls)
        self.start_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self._stop)

        self.worker: Optional[ClientWorker] = None
        self._update_source_rows()

    def _update_source_rows(self):
        is_files = (self.src_mode.currentText() == "Video Files")
        for w in [self.left_cam, self.right_cam]:
            w.setEnabled(not is_files if hasattr(bool, "__invert__") else (not is_files))
        for w in [self.left_file, self.left_browse, self.right_file, self.right_browse]:
            w.setEnabled(is_files)

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
        v.addWidgets(btns)

        dlg.exec()

    def _browse_left(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Left Video", "", "Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*)")
        if fn: self.left_file.setText(fn)

    def _browse_right(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Right Video", "", "Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*)")
        if fn: self.right_file.setText(fn)

    def _browse_save(self):
        d = QtWidgets.QFileDialog.getOpenFileName(self, "Select Output Directory")
        if d: self.save_dir.setText(d)

    def _toggle_depth_controls(self):
        is_depth = (self.output_mode.currentText() == "Depth")
        self.depth_group.setEnabled(is_depth)
        rs = self.use_realsense.isChecked()
        self.focal_px_edit.setEnabled(is_depth and not rs)
        self.baseline_m_edit.setEnabled(is_depth and not rs)

    def _start(self):
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

        host = self.host.text().strip()
        port = self.port.value()
        path = self.path.text().strip() or "/foundation-stereo"
        fps = float(self.fps.value())
        jpeg_quality = int(self.jpeg_quality.value())
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

        client = StereoSenderClient(
            host, port, path, fps, jpeg_quality, session_id,
            save_dir = save_dir, preview = preview,
            on_log = self._log,
            on_result = self._on_result_image,
            on_start = lambda: self._set_running(True),
            on_finish = lambda: self._set_running(False)
        )
        self.worker = ClientWorker(client, left_src, right_src, mode)
        self.worker.log_signal.conect(self._log)
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



    def _on_result_image(self, seq: int, kind: str, enc: str, w: int, h: int, payload: bytes):
        arr = np.frombuffer(payload, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            return
        if img.dtype == np.uint16:
            nz = img[img > 0]
            m = nz.mean() if nz.size else 1.0
            vis = np.clip((img.astype(np.float32) / (3.0*m)) * 255.0, 0, 255).astype(np.uint8)
            vis = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
        else:
            vis = img
            if vis.ndim == 2:
                vis = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
        vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        hq, wq = vis.shape[:2]
        qimg = QtGui.QImage(vis.data, wq, hq, vis.strides[0], QtGui.QImage.Format.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(self.preview_label.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
        self.preview_label.setPixmap(pix)

def launch_gui():
    app = QtWidgets.QApplication([])
    mw = MainWindow()
    mw.show()
    return app.exec()