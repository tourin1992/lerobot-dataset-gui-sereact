import sys
import warnings
from pathlib import Path
from typing import Optional, Dict, List

# 过滤 torchvision 和 lerobot 的视频解码相关警告
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision.io._video_deprecation_warning')
warnings.filterwarnings('ignore', message='.*torchcodec.*not available.*')

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QPushButton, QVBoxLayout, QWidget, QMessageBox, QListWidget,
    QSlider, QSplitter, QScrollArea, QCheckBox, QGroupBox, QTextEdit,
    QTreeWidget, QTreeWidgetItem
)

from processor import DatasetProcessor


class LoaderThread(QThread):
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, processor: DatasetProcessor, repo_id: str):
        super().__init__()
        self.processor = processor
        self.repo_id = repo_id

    def run(self):
        try:
            dataset = self.processor.load_dataset(self.repo_id)
            self.finished.emit(dataset)
        except Exception as e:
            self.error.emit(str(e))


class EpisodeLoaderThread(QThread):
    """Thread for loading episode data asynchronously."""
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, processor: DatasetProcessor, episode_idx: int, keys: list):
        super().__init__()
        self.processor = processor
        self.episode_idx = episode_idx
        self.keys = keys

    def run(self):
        try:
            data = self.processor.get_episode_data(self.episode_idx, self.keys)
            self.finished.emit(data)
        except Exception as e:
            self.error.emit(str(e))


class DatasetGui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.processor = DatasetProcessor()
        self.plots: Dict[str, pg.PlotWidget] = {}
        self.plot_curves: Dict[str, List[pg.PlotDataItem]] = {}
        self.v_lines: Dict[str, pg.InfiniteLine] = {}
        self.current_ep_data: Dict[str, np.ndarray] = {}
        self.vector_keys = []
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("LeRobot Dataset Visualizer")
        self.resize(1400, 950)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # Top Bar (保持不变)
        top_layout = QHBoxLayout()
        self.repo_input = QLineEdit("lerobot/pusht")
        self.load_btn = QPushButton("Load Dataset")
        self.load_btn.clicked.connect(self.on_load_clicked)
        self.save_img_btn = QPushButton("Save Test Image")
        self.save_img_btn.setEnabled(False)
        self.save_img_btn.clicked.connect(self.save_test_image)
        self.show_info_btn = QPushButton("Show Info")
        self.show_info_btn.clicked.connect(lambda: self.info_panel.show())
        
        top_layout.addWidget(QLabel("Repo ID:"))
        top_layout.addWidget(self.repo_input)
        top_layout.addWidget(self.load_btn)
        top_layout.addWidget(self.save_img_btn)
        top_layout.addWidget(self.show_info_btn)
        main_layout.addLayout(top_layout)

        # Main Vertical Splitter
        self.main_splitter = QSplitter(Qt.Vertical)
        self.horizontal_splitter = QSplitter(Qt.Horizontal)
        
        # 1. Left: Episode List
        self.ep_list = QListWidget()
        self.ep_list.currentRowChanged.connect(self.on_episode_changed)
        self.horizontal_splitter.addWidget(self.ep_list)

        # 2. Center: Image and Slider
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        self.image_label = QLabel("No Image")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: black;")
        center_layout.addWidget(self.image_label, stretch=1)

        # Slider and Labels
        slider_container = QWidget()
        slider_vbox = QVBoxLayout(slider_container)
        
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.valueChanged.connect(self.on_slider_changed)
        slider_vbox.addWidget(self.frame_slider)
        
        labels_layout = QHBoxLayout()
        self.frame_label = QLabel("Frame: 0/0")
        self.timestamp_label = QLabel("Timestamp: 0.000s")
        labels_layout.addWidget(self.frame_label)
        labels_layout.addStretch()
        labels_layout.addWidget(self.timestamp_label)
        slider_vbox.addLayout(labels_layout)
        
        center_layout.addWidget(slider_container)
        self.horizontal_splitter.addWidget(center_widget)

        # 3. Right: Hierarchical Selectors and Plots
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        self.right_splitter = QSplitter(Qt.Vertical)
        
        # New Hierarchical Selector (Tree)
        self.feature_tree = QTreeWidget()
        self.feature_tree.setFocusPolicy(Qt.NoFocus) # 关键：不捕获焦点
        self.feature_tree.setHeaderLabel("Features & Dimensions")
        self.feature_tree.itemChanged.connect(self.on_tree_item_changed)
        self.right_splitter.addWidget(self.feature_tree)
        
        self.plot_scroll = QScrollArea()
        self.plot_scroll.setFocusPolicy(Qt.NoFocus)
        self.plot_container = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_container)
        self.plot_layout.setSpacing(10) # 设置图表间距
        self.plot_layout.addStretch() # 底部添加弹簧
        self.plot_scroll.setWidget(self.plot_container)
        self.plot_scroll.setWidgetResizable(True)
        self.right_splitter.addWidget(self.plot_scroll)
        
        # 设置右侧初始比例：特征树更矮，绘图区更大
        self.right_splitter.setStretchFactor(0, 1)
        self.right_splitter.setStretchFactor(1, 6)
        self.right_splitter.setSizes([150, 800])
        
        right_layout.addWidget(self.right_splitter)

        self.horizontal_splitter.addWidget(right_widget)
        # 减小左侧导航栏权重，分配更多给中间和右侧
        self.horizontal_splitter.setStretchFactor(0, 0)
        self.horizontal_splitter.setStretchFactor(1, 2)
        self.horizontal_splitter.setStretchFactor(2, 2)
        self.horizontal_splitter.setSizes([120, 640, 640])
        self.main_splitter.addWidget(self.horizontal_splitter)

        # Bottom Info Panel (保持不变)
        self.info_panel = QGroupBox("Dataset Details")
        self.info_panel_layout = QHBoxLayout(self.info_panel)
        self.info_display = QTextEdit()
        self.info_display.setReadOnly(True)
        self.info_panel_layout.addWidget(self.info_display)
        close_btn = QPushButton("×")
        close_btn.setFixedSize(20, 20)
        close_btn.clicked.connect(self.info_panel.hide)
        self.info_panel_layout.addWidget(close_btn, alignment=Qt.AlignTop)
        self.info_panel.hide()
        self.main_splitter.addWidget(self.info_panel)
        
        self.main_splitter.setStretchFactor(0, 4)
        self.main_splitter.setStretchFactor(1, 1)
        main_layout.addWidget(self.main_splitter)

        self.status_label = QLabel("Ready")
        main_layout.addWidget(self.status_label)

    def on_load_clicked(self):
        repo_id = self.repo_input.text().strip()
        if not repo_id: return
        self.load_btn.setEnabled(False)
        self.status_label.setText(f"Loading {repo_id}...")
        self.loader_thread = LoaderThread(self.processor, repo_id)
        self.loader_thread.finished.connect(self.on_load_finished)
        self.loader_thread.error.connect(self.on_load_error)
        self.loader_thread.start()

    def on_load_finished(self, dataset):
        self.load_btn.setEnabled(True)
        self.save_img_btn.setEnabled(True)
        self.status_label.setText("Loaded")
        
        # Table-based Feature Info for Alignment
        meta = dataset.meta
        sample = dataset[0]
        rows = []
        for k in meta.features.keys():
            val = sample[k]
            type_name = type(val).__name__
            if hasattr(val, 'dtype'): type_name = f"{type_name}[{val.dtype}]"
            
            shape = "1"
            if hasattr(val, 'shape'): shape = str(list(val.shape))
            elif hasattr(val, '__len__'): shape = str(len(val))
            
            # Align columns using table
            rows.append(f"<tr><td width='150'><code>{k}</code></td><td width='200' style='color: #2980b9;'>{type_name}</td><td>shape: {shape}</td></tr>")
        
        info_text = (
            f"<b>Repo ID:</b> {dataset.repo_id}<br>"
            f"<b>Total Episodes:</b> {meta.total_episodes} | <b>Total Frames:</b> {meta.total_frames} | <b>FPS:</b> {meta.fps}<br>"
            f"<b>Robot Type:</b> {meta.robot_type}<br>"
            f"<b>Features:</b><table style='margin-left: 20px;'>{''.join(rows)}</table>"
        )
        self.info_display.setHtml(info_text)
        self.info_panel.show()

        self.ep_list.clear()
        for i in range(dataset.meta.total_episodes):
            self.ep_list.addItem(f"Episode {i}")
        
        self.init_dimension_selectors(dataset)
        if dataset.meta.total_episodes > 0:
            self.ep_list.setCurrentRow(0)

    def init_dimension_selectors(self, dataset):
        """Initializes the hierarchical feature tree."""
        self.feature_tree.clear()
        # Clean up old plots
        for i in reversed(range(self.plot_layout.count())):
            widget = self.plot_layout.itemAt(i).widget()
            if widget: widget.setParent(None)
        self.plots.clear()
        self.plot_curves.clear()
        
        # Keys we want to plot
        target_keys = ['observation.state', 'action', 'next.reward', 'next.done', 'next.success']
        self.vector_keys = [k for k in target_keys if k in dataset.meta.features]
        
        sample = dataset[0]
        for k in self.vector_keys:
            val = sample[k]
            # Convert to numpy to get shape
            v_np = val.numpy() if hasattr(val, 'numpy') else np.array(val)
            dims = v_np.shape[0] if v_np.ndim > 0 else 1
            
            # Create Tree Item
            parent = QTreeWidgetItem(self.feature_tree)
            parent.setText(0, k)
            parent.setCheckState(0, Qt.Checked)
            parent.setExpanded(True)
            
            # Create Plot Widget
            pw = pg.PlotWidget(title=k)
            pw.setBackground('w')
            pw.showGrid(x=True, y=True)
            pw.setMinimumHeight(150)
            pw.setFocusPolicy(Qt.NoFocus)

            v_line = pg.InfiniteLine(pos=0, angle=90, pen='r')
            pw.addItem(v_line)
            self.plots[k] = pw
            self.v_lines[k] = v_line
            self.plot_layout.addWidget(pw)
            
            self.plot_curves[k] = []
            if dims > 1:
                for d in range(dims):
                    child = QTreeWidgetItem(parent)
                    child.setText(0, f"Dimension {d}")
                    child.setCheckState(0, Qt.Checked)
                    child.setData(0, Qt.UserRole, (k, d))

    def on_tree_item_changed(self, item, column):
        """Handles visibility toggles from the tree."""
        key_data = item.data(0, Qt.UserRole)
        is_checked = item.checkState(0) == Qt.Checked
        
        if key_data is None: # Parent item (Feature)
            key = item.text(0)
            if key in self.plots:
                self.plots[key].setVisible(is_checked)
            for i in range(item.childCount()):
                item.child(i).setCheckState(0, item.checkState(0))
        else: # Child item (Dimension)
            key, dim = key_data
            if key in self.plot_curves and dim < len(self.plot_curves[key]):
                self.plot_curves[key][dim].setVisible(is_checked)

    def on_slider_changed(self, value):
        start = self.frame_slider.minimum()
        self.frame_label.setText(f"Frame: {value - start}/{self.frame_slider.maximum() - start}")
        
        # Update timestamp
        try:
            # We need the timestamp from current_ep_data or by fetching frame
            data = self.processor.get_frame(value)
            ts = data.get('timestamp', 0.0)
            self.timestamp_label.setText(f"Timestamp: {ts:.3f}s")
        except: pass
        
        # Update Vertical Lines
        relative_idx = value - start
        for v_line in self.v_lines.values():
            v_line.setPos(relative_idx)
            
        self.update_frame_view(value)

    def on_episode_changed(self, row):
        if row < 0: return
        
        # 显示加载状态
        self.status_label.setText(f"Loading Episode {row}...")
        self.frame_slider.setEnabled(False)
        
        start, end = self.processor.get_episode_range(row)
        self.frame_slider.setRange(start, end - 1)
        
        # 异步加载 episode 数据
        self.episode_loader = EpisodeLoaderThread(self.processor, row, self.vector_keys)
        self.episode_loader.finished.connect(self.on_episode_data_loaded)
        self.episode_loader.error.connect(self.on_episode_load_error)
        self.episode_loader.start()
        
        # 先设置第一帧，不等待全部数据加载
        self.frame_slider.setValue(start)
    
    def on_episode_data_loaded(self, data):
        """Callback when episode data is loaded."""
        self.current_ep_data = data
        start, end = self.frame_slider.minimum(), self.frame_slider.maximum()
        num_frames = end - start + 1
        
        # Update Plot Curves
        for k in self.vector_keys:
            if k not in data:
                continue
            raw_val = data[k]
            # 3. 核心修复：数据转换 (T, D) 或 (T,)
            # 处理布尔值和标量
            if raw_val.dtype == bool:
                plot_data = raw_val.astype(np.float32)
            else:
                plot_data = raw_val.astype(np.float32)
            
            # 确保是 2D 数组 (T, D)
            if plot_data.ndim == 1:
                plot_data = plot_data.reshape(-1, 1)
            pw = self.plots[k]
            # Clear old curves
            for c in self.plot_curves.get(k, []): 
                pw.removeItem(c)
            self.plot_curves[k] = []
            
            # Create new curves for each dimension
            for d in range(plot_data.shape[1]):
                curve = pg.PlotDataItem(plot_data[:, d], pen=pg.mkPen(color=pg.intColor(d), width=1.5))
                pw.addItem(curve)
                self.plot_curves[k].append(curve)
                
            # 统一设置 X 轴范围，确保对齐
            pw.setXRange(0, num_frames, padding=0)
            
            # 如果是 reward/done/success，设置合理的 Y 轴范围
            if any(x in k for x in ['reward', 'done', 'success']):
                pw.setYRange(-0.1, 1.1, padding=0)
        
        self.update_plots_visibility()
        self.frame_slider.setEnabled(True)
        self.status_label.setText("Ready")
    
    def on_episode_load_error(self, err):
        """Callback when episode data loading fails."""
        self.status_label.setText(f"Error loading episode: {err}")
        self.frame_slider.setEnabled(True)

    def update_plots_visibility(self):
        """Updates which curves are shown based on checkboxes in the feature tree."""
        for i in range(self.feature_tree.topLevelItemCount()):
            parent = self.feature_tree.topLevelItem(i)
            key = parent.text(0)
            is_parent_checked = parent.checkState(0) == Qt.Checked
            
            if key in self.plots:
                self.plots[key].setVisible(is_parent_checked)
            
            for j in range(parent.childCount()):
                child = parent.child(j)
                key_data = child.data(0, Qt.UserRole)
                if key_data:
                    k, dim = key_data
                    is_child_checked = child.checkState(0) == Qt.Checked
                    if k in self.plot_curves and dim < len(self.plot_curves[k]):
                        self.plot_curves[k][dim].setVisible(is_child_checked)

    def update_frame_view(self, frame_idx):
        try:
            data = self.processor.get_frame(frame_idx)
            # Find and display image
            img_keys = [k for k in data.keys() if 'image' in k]
            if img_keys:
                img_data = data[img_keys[0]]
                
                # Handle different image formats
                if hasattr(img_data, 'numpy'):  # Torch Tensor
                    img_np = img_data.numpy()
                    # Check if CHW format (channels first)
                    if img_np.ndim == 3 and img_np.shape[0] in [1, 3, 4]:
                        img_np = np.transpose(img_np, (1, 2, 0))  # CHW -> HWC
                    # Normalize to 0-255 uint8
                    if img_np.dtype == np.float32 or img_np.dtype == np.float64:
                        if img_np.max() <= 1.0:
                            img_np = (img_np * 255).astype(np.uint8)
                        else:
                            img_np = img_np.astype(np.uint8)
                elif hasattr(img_data, 'convert'):  # PIL Image
                    img_np = np.array(img_data.convert('RGB'))
                elif isinstance(img_data, dict) and 'path' in img_data:
                    # Video format - need to decode
                    self.status_label.setText("Video format not yet supported in frame view")
                    return
                else:
                    img_np = np.array(img_data)
                
                # Ensure C-contiguous array
                img_np = np.ascontiguousarray(img_np)
                
                # Handle grayscale
                if img_np.ndim == 2:
                    img_np = np.stack([img_np] * 3, axis=-1)
                
                h, w, c = img_np.shape
                bytes_per_line = c * w
                qimg = QImage(img_np.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
                pixmap = QPixmap.fromImage(qimg)
                
                # Scale to fit label while keeping aspect ratio
                scaled_pixmap = pixmap.scaled(
                    self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_label.setPixmap(scaled_pixmap)
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()

    def on_load_error(self, err):
        self.load_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", err)
    
    def save_test_image(self):
        """Save current frame's image to disk for debugging."""
        try:
            frame_idx = self.frame_slider.value()
            data = self.processor.get_frame(frame_idx)
            img_keys = [k for k in data.keys() if 'image' in k]
            if img_keys:
                img_data = data[img_keys[0]]
                
                from PIL import Image as PILImage
                
                if hasattr(img_data, 'numpy'):  # Torch Tensor
                    img_np = img_data.numpy()
                    if img_np.ndim == 3 and img_np.shape[0] in [1, 3, 4]:
                        img_np = np.transpose(img_np, (1, 2, 0))
                    if img_np.dtype in [np.float32, np.float64]:
                        if img_np.max() <= 1.0:
                            img_np = (img_np * 255).astype(np.uint8)
                        else:
                            img_np = img_np.astype(np.uint8)
                    pil_img = PILImage.fromarray(img_np)
                elif hasattr(img_data, 'convert'):  # PIL Image
                    pil_img = img_data.convert('RGB')
                else:
                    pil_img = PILImage.fromarray(np.array(img_data))
                
                filename = f"test_frame_{frame_idx}.png"
                pil_img.save(filename)
                self.status_label.setText(f"Saved to {filename}")
                QMessageBox.information(self, "Success", f"Image saved to {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save: {str(e)}")
            import traceback
            traceback.print_exc()

    def keyPressEvent(self, event):
        # 强制让 slider 或 list 处理，或者直接由 window 处理
        if event.key() in [Qt.Key_W, Qt.Key_Up]:
            self.ep_list.setCurrentRow(max(0, self.ep_list.currentRow() - 1))
        elif event.key() in [Qt.Key_S, Qt.Key_Down]:
            self.ep_list.setCurrentRow(min(self.ep_list.count() - 1, self.ep_list.currentRow() + 1))
        elif event.key() in [Qt.Key_A, Qt.Key_Left]:
            self.frame_slider.setValue(max(self.frame_slider.minimum(), self.frame_slider.value() - 1))
        elif event.key() in [Qt.Key_D, Qt.Key_Right]:
            self.frame_slider.setValue(min(self.frame_slider.maximum(), self.frame_slider.value() + 1))
        else:
            super().keyPressEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = DatasetGui()
    gui.show()
    sys.exit(app.exec())