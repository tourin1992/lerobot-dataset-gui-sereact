import sys
import os
import json
import shutil
import warnings
import torch
from pathlib import Path
from typing import Optional, Dict, List, Set

# Project directory for backup copies
PROJECT_DIR = Path("/home/dzmitry/sandbox/trossen/lerobot-dataset-gui-sereact")
APPROVALS_BACKUP_DIR = PROJECT_DIR / "approvals_backup"

# 过滤 torchvision 和 lerobot 的视频解码相关警告
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision.io._video_deprecation_warning')
warnings.filterwarnings('ignore', message='.*torchcodec.*not available.*')

import numpy as np
import pandas as pd
import pyqtgraph as pg
import pyarrow.parquet as pq
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QImage, QPixmap, QBrush, QColor
from PySide6.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QPushButton, QVBoxLayout, QWidget, QMessageBox, QListWidget,
    QSlider, QSplitter, QScrollArea, QCheckBox, QGroupBox, QTextEdit,
    QTreeWidget, QTreeWidgetItem, QTabWidget, QMenu, QListWidgetItem,
    QGridLayout, QStackedWidget, QFileDialog
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
        self.trim_start_frame: Optional[int] = None
        self.trim_end_frame: Optional[int] = None
        # Multi-camera support
        self.camera_labels: Dict[str, QLabel] = {}
        self.image_keys: List[str] = []
        # Episode approval tracking
        self.approved_episodes: Set[int] = set()
        self.commented_episodes: Dict[int, str] = {}  # episode_idx -> comment
        self.approvals_file_path: Optional[Path] = None
        # Episode tasks (from parquet)
        self.episode_tasks: Dict[int, str] = {}  # episode_idx -> task description
        # Playback timer
        self.playback_timer: Optional[QTimer] = None
        self.playback_speed: float = 1.0  # 1x or 2x
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
        self.choose_dataset_btn = QPushButton("Choose Dataset")
        self.choose_dataset_btn.clicked.connect(self.on_choose_dataset_clicked)
        self.load_btn = QPushButton("Load Dataset")
        self.load_btn.clicked.connect(self.on_load_clicked)
        self.show_info_btn = QPushButton("Show Info")
        self.show_info_btn.clicked.connect(lambda: self.info_panel.show())
        
        top_layout.addWidget(QLabel("Repo ID:"))
        top_layout.addWidget(self.repo_input)
        top_layout.addWidget(self.choose_dataset_btn)
        top_layout.addWidget(self.load_btn)
        top_layout.addWidget(self.show_info_btn)
        main_layout.addLayout(top_layout)

        # Main Vertical Splitter
        self.main_splitter = QSplitter(Qt.Vertical)
        self.horizontal_splitter = QSplitter(Qt.Horizontal)
        
        # 1. Left: Episode List
        self.ep_list = QListWidget()
        self.ep_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.ep_list.customContextMenuRequested.connect(self.show_episode_context_menu)
        self.ep_list.currentRowChanged.connect(self.on_episode_changed)
        self.horizontal_splitter.addWidget(self.ep_list)

        # 2. Center: Image, Slider and Plots (Always Visible)
        self.center_splitter = QSplitter(Qt.Vertical)
        
        # Top part: Image and Slider
        image_slider_container = QWidget()
        image_slider_layout = QVBoxLayout(image_slider_container)
        
        # Stacked widget for single/multi camera views
        self.image_stack = QStackedWidget()
        
        # Single camera view (index 0)
        self.image_label = QLabel("No Image")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: black;")
        self.image_stack.addWidget(self.image_label)
        
        # Multi-camera grid view (index 1)
        self.multi_camera_widget = QWidget()
        self.multi_camera_widget.setStyleSheet("background-color: black;")
        self.multi_camera_layout = QGridLayout(self.multi_camera_widget)
        self.multi_camera_layout.setSpacing(4)
        self.multi_camera_layout.setContentsMargins(0, 0, 0, 0)
        self.image_stack.addWidget(self.multi_camera_widget)
        
        image_slider_layout.addWidget(self.image_stack, stretch=1)

        # Slider and Labels
        slider_container = QWidget()
        slider_vbox = QVBoxLayout(slider_container)
        
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.valueChanged.connect(self.on_slider_changed)
        slider_vbox.addWidget(self.frame_slider)
        
        labels_layout = QHBoxLayout()
        self.frame_label = QLabel("Frame: 0/0")
        self.timestamp_label = QLabel("Timestamp: 0.000s")
        
        # Show All Cameras checkbox
        self.show_all_cameras_cb = QCheckBox("Show All Cameras")
        self.show_all_cameras_cb.setChecked(False)
        self.show_all_cameras_cb.stateChanged.connect(self.on_show_all_cameras_changed)
        
        # Approve episode button
        self.approve_btn = QPushButton("Approve Episode")
        self.approve_btn.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold;")
        self.approve_btn.clicked.connect(self.on_approve_episode_clicked)
        
        # Comment button
        self.comment_btn = QPushButton("Comment")
        self.comment_btn.setStyleSheet("background-color: #f39c12; color: white; font-weight: bold;")
        self.comment_btn.clicked.connect(self.on_comment_episode_clicked)
        
        # Playback buttons
        self.play_1x_btn = QPushButton("▶ 1x")
        self.play_1x_btn.setStyleSheet("background-color: #3498db; color: white; font-weight: bold;")
        self.play_1x_btn.clicked.connect(self.on_play_1x_clicked)
        
        self.play_2x_btn = QPushButton("▶ 2x")
        self.play_2x_btn.setStyleSheet("background-color: #9b59b6; color: white; font-weight: bold;")
        self.play_2x_btn.clicked.connect(self.on_play_2x_clicked)
        
        self.stop_btn = QPushButton("⏹")
        self.stop_btn.setStyleSheet("background-color: #95a5a6; color: white; font-weight: bold;")
        self.stop_btn.clicked.connect(self.on_stop_playback_clicked)
        
        labels_layout.addWidget(self.frame_label)
        labels_layout.addWidget(self.play_1x_btn)
        labels_layout.addWidget(self.play_2x_btn)
        labels_layout.addWidget(self.stop_btn)
        labels_layout.addWidget(self.show_all_cameras_cb)
        labels_layout.addWidget(self.approve_btn)
        labels_layout.addWidget(self.comment_btn)
        labels_layout.addStretch()
        labels_layout.addWidget(self.timestamp_label)
        slider_vbox.addLayout(labels_layout)
        
        image_slider_layout.addWidget(slider_container)
        self.center_splitter.addWidget(image_slider_container)

        # Bottom part: Plots (Now always visible in center)
        self.plot_scroll = QScrollArea()
        self.plot_scroll.setFocusPolicy(Qt.NoFocus)
        self.plot_container = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_container)
        self.plot_layout.setSpacing(10)
        self.plot_layout.addStretch()
        self.plot_scroll.setWidget(self.plot_container)
        self.plot_scroll.setWidgetResizable(True)
        self.center_splitter.addWidget(self.plot_scroll)
        
        # Initial proportions for center
        self.center_splitter.setSizes([400, 500])
        self.horizontal_splitter.addWidget(self.center_splitter)

        # 3. Right: Hierarchical Selectors and Edit Tools (Wrapped in Tabs)
        self.right_tabs = QTabWidget()
        self.horizontal_splitter.addWidget(self.right_tabs)

        # Tab 1: Features (Tree only)
        self.feature_tree = QTreeWidget()
        self.feature_tree.setFocusPolicy(Qt.NoFocus)
        self.feature_tree.setHeaderLabel("Features & Dimensions")
        self.feature_tree.itemChanged.connect(self.on_tree_item_changed)
        self.feature_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.feature_tree.customContextMenuRequested.connect(self.show_feature_tree_context_menu)
        self.right_tabs.addTab(self.feature_tree, "Features")

        # Tab 2: Edit
        self.edit_tab = QWidget()
        self.init_edit_tab()
        self.right_tabs.addTab(self.edit_tab, "Edit")

        # Layout weights: Left is narrow, Center is wide (Image+Plots), Right is medium (Tree/Edit)
        self.horizontal_splitter.setStretchFactor(0, 0)
        self.horizontal_splitter.setStretchFactor(1, 4)
        self.horizontal_splitter.setStretchFactor(2, 1)
        self.horizontal_splitter.setSizes([120, 900, 380])
        self.main_splitter.addWidget(self.horizontal_splitter)

        # Bottom Info Panel (保持不变)
        self.info_panel = QGroupBox("Dataset Details")
        self.info_panel_layout = QHBoxLayout(self.info_panel)
        self.info_display = QTextEdit()
        self.info_display.setReadOnly(True)
        self.info_panel_layout.addWidget(self.info_display)
        close_btn = QPushButton("×")
        close_btn.setFixedSize(24, 24)
        close_btn.clicked.connect(self.info_panel.hide)
        self.info_panel_layout.addWidget(close_btn, alignment=Qt.AlignTop)
        self.info_panel.hide()
        self.main_splitter.addWidget(self.info_panel)
        
        self.main_splitter.setStretchFactor(0, 4)
        self.main_splitter.setStretchFactor(1, 1)
        main_layout.addWidget(self.main_splitter)

        self.status_label = QLabel("Ready")
        main_layout.addWidget(self.status_label)

    def init_edit_tab(self):
        layout = QVBoxLayout(self.edit_tab)
        
        # 1. Pending Operations List
        layout.addWidget(QLabel("<b>Pending Operations:</b>"))
        self.pending_op_list = QListWidget()
        self.pending_op_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.pending_op_list.customContextMenuRequested.connect(self.show_op_context_menu)
        self.pending_op_list.setToolTip("Right click to undo a specific operation")
        layout.addWidget(self.pending_op_list)
        
        self.clear_tasks_btn = QPushButton("Clear All Tasks")
        self.clear_tasks_btn.clicked.connect(self.on_clear_tasks_clicked)
        layout.addWidget(self.clear_tasks_btn)
        
        layout.addSpacing(10)
        
        # 2. Global Operations
        global_group = QGroupBox("Global Operations")
        global_layout = QVBoxLayout(global_group)
        
        self.batch_delete_btn = QPushButton("Batch Delete Episodes...")
        self.batch_delete_btn.clicked.connect(self.on_batch_delete_clicked)
        global_layout.addWidget(self.batch_delete_btn)
        
        self.remove_feature_btn = QPushButton("Remove Features...")
        self.remove_feature_btn.clicked.connect(self.on_remove_feature_clicked)
        global_layout.addWidget(self.remove_feature_btn)
        
        layout.addWidget(global_group)
        
        # 3. Local Operations
        local_group = QGroupBox("Local Operations")
        local_layout = QVBoxLayout(local_group)
        
        self.trim_frames_btn = QPushButton("Trim Selected Range")
        self.trim_frames_btn.setEnabled(False)
        self.trim_frames_btn.setToolTip("Mark start and end on the slider first")
        self.trim_frames_btn.clicked.connect(self.on_trim_frames_clicked)
        local_layout.addWidget(self.trim_frames_btn)
        
        self.edit_frame_btn = QPushButton("Edit Current Frame Features...")
        self.edit_frame_btn.clicked.connect(self.on_edit_frame_clicked)
        local_layout.addWidget(self.edit_frame_btn)
        
        layout.addWidget(local_group)

        layout.addStretch()

        # 4. Export Settings (At the bottom)
        export_group = QGroupBox("Export Settings")
        export_layout = QVBoxLayout(export_group)
        
        export_layout.addWidget(QLabel("New Repo ID:"))
        self.new_repo_input = QLineEdit()
        self.new_repo_input.setPlaceholderText("e.g., lerobot/pusht_modified")
        export_layout.addWidget(self.new_repo_input)
        
        self.save_dataset_btn = QPushButton("Apply Edits && Save Dataset")
        self.save_dataset_btn.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; padding: 10px;")
        self.save_dataset_btn.clicked.connect(self.on_save_dataset_clicked)
        export_layout.addWidget(self.save_dataset_btn)
        
        layout.addWidget(export_group)

    def on_mark_start_clicked(self):
        self.trim_start_frame = self.frame_slider.value()
        # Find local frame index within episode
        start, _ = self.processor.get_episode_range(self.ep_list.currentRow())
        local_idx = self.trim_start_frame - start
        self.status_label.setText(f"Marked start frame: {local_idx}")
        self.update_trim_btn_state()

    def on_mark_end_clicked(self):
        self.trim_end_frame = self.frame_slider.value()
        start, _ = self.processor.get_episode_range(self.ep_list.currentRow())
        local_idx = self.trim_end_frame - start
        self.status_label.setText(f"Marked end frame: {local_idx}")
        self.update_trim_btn_state()

    def update_trim_btn_state(self):
        # Enable trim button if both marks are set and in the same episode
        # (Technically they could be different episodes if we want cross-episode trimming, 
        # but let's keep it simple for now).
        if hasattr(self, 'trim_frames_btn'):
            self.trim_frames_btn.setEnabled(self.trim_start_frame is not None and self.trim_end_frame is not None)

    def on_edit_frame_clicked(self):
        if not self.processor.dataset: return
        
        frame_idx = self.frame_slider.value()
        try:
            data = self.processor.get_frame(frame_idx)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to get frame data: {str(e)}")
            return
            
        # Get episode info
        ep_idx = self.ep_list.currentRow()
        start, _ = self.processor.get_episode_range(ep_idx)
        local_idx = frame_idx - start
        
        from PySide6.QtWidgets import QDialog, QFormLayout, QLineEdit
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Edit Frame {local_idx} in Episode {ep_idx}")
        dialog.resize(400, 500)
        d_layout = QVBoxLayout(dialog)
        
        scroll = QScrollArea()
        scroll_content = QWidget()
        form_layout = QFormLayout(scroll_content)
        
        inputs = {}
        original_data = {}
        # Only show vector/scalar features, skip images/videos
        for key, val in data.items():
            if any(x in key for x in ['task', 'image', 'video', 'index', 'frame_index', 'episode_index', 'timestamp']):
                continue
            
            original_data[key] = val
            # Convert to string for editing
            val_np = val.numpy() if hasattr(val, 'numpy') else np.array(val)
            val_str = np.array2string(val_np, separator=',').replace('[', '').replace(']', '').replace('\n', '')
            
            line_edit = QLineEdit(val_str)
            form_layout.addRow(f"{key}:", line_edit)
            inputs[key] = line_edit
            
        scroll.setWidget(scroll_content)
        scroll.setWidgetResizable(True)
        d_layout.addWidget(scroll)
        
        btns_layout = QHBoxLayout()
        ok_btn = QPushButton("Add to Tasks")
        ok_btn.clicked.connect(dialog.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        btns_layout.addWidget(ok_btn)
        btns_layout.addWidget(cancel_btn)
        d_layout.addLayout(btns_layout)
        
        if dialog.exec() == QDialog.Accepted:
            new_features = {}
            for key, line_edit in inputs.items():
                try:
                    orig_val = original_data[key]
                    txt = line_edit.text().strip()
                    clean_txt = txt.replace('[', '').replace(']', '').replace(' ', '')
                    
                    if not clean_txt:
                        continue
                    
                    parts = [p.strip() for p in clean_txt.split(',') if p.strip()]
                    
                    # Determine target dtype and converter
                    if isinstance(orig_val, torch.Tensor):
                        target_dtype = orig_val.dtype
                        if target_dtype == torch.bool:
                            vals = [p.lower() in ['true', '1', 't', 'y', 'yes'] for p in parts]
                            new_val = torch.tensor(vals, dtype=torch.bool)
                        elif target_dtype in [torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8]:
                            vals = [int(float(p)) for p in parts]
                            new_val = torch.tensor(vals, dtype=target_dtype)
                        else:
                            vals = [float(p) for p in parts]
                            new_val = torch.tensor(vals, dtype=target_dtype)
                        
                        # Handle shape (if original was scalar-like tensor vs vector)
                        if orig_val.ndim == 0 and new_val.numel() == 1:
                            new_val = new_val.squeeze()
                    else:
                        # Fallback for non-tensor types
                        if isinstance(orig_val, bool):
                            new_val = parts[0].lower() in ['true', '1', 't', 'y', 'yes']
                        elif isinstance(orig_val, int):
                            new_val = int(float(parts[0]))
                        else:
                            new_val = float(parts[0])
                    
                    new_features[key] = new_val
                except Exception as e:
                    QMessageBox.warning(self, "Warning", f"Failed to parse {key}: {str(e)}")
            
            if new_features:
                self.processor.add_frame_edit_task(ep_idx, local_idx, new_features)
                self.refresh_edit_ui()
                self.status_label.setText(f"Added edit task for Episode {ep_idx}, Frame {local_idx}")

    def on_trim_frames_clicked(self):
        if self.trim_start_frame is None or self.trim_end_frame is None:
            return
            
        # Get episode index and local frame range
        ep_idx = self.ep_list.currentRow()
        ep_start, _ = self.processor.get_episode_range(ep_idx)
        
        start_local = self.trim_start_frame - ep_start
        end_local = self.trim_end_frame - ep_start
        
        if start_local > end_local:
            start_local, end_local = end_local, start_local
            
        self.processor.add_trim_task(ep_idx, start_local, end_local)
        self.refresh_edit_ui()
        self.status_label.setText(f"Added trim task for Episode {ep_idx}: {start_local}-{end_local}")
        
        # Reset marks
        self.trim_start_frame = None
        self.trim_end_frame = None
        self.update_trim_btn_state()

    def on_clear_tasks_clicked(self):
        self.processor.clear_edit_tasks()
        self.refresh_edit_ui()
        self.status_label.setText("Edit tasks cleared")

    def on_batch_delete_clicked(self):
        from PySide6.QtWidgets import QInputDialog
        text, ok = QInputDialog.getText(self, "Batch Delete Episodes", 
                                       "Enter episode indices (e.g., 0, 2, 5-10):")
        if ok and text:
            try:
                indices = self._parse_indices(text)
                count = 0
                for idx in indices:
                    if idx < self.processor.dataset.meta.total_episodes:
                        self.processor.add_delete_episode_task(idx)
                        count += 1
                self.refresh_edit_ui()
                self.status_label.setText(f"Added {count} episodes to deletion tasks")
            except ValueError as e:
                QMessageBox.critical(self, "Error", f"Invalid input format: {str(e)}")

    def on_remove_feature_clicked(self):
        if not self.processor.dataset: return
        
        # Create a simple dialog with checkboxes
        from PySide6.QtWidgets import QDialog, QListWidgetItem
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Features to Remove")
        dialog.resize(300, 400)
        d_layout = QVBoxLayout(dialog)
        
        list_widget = QListWidget()
        features = sorted(list(self.processor.dataset.meta.features.keys()))
        # Filter out internal/essential features that shouldn't be removed
        internal_keys = ['index', 'frame_index', 'episode_index', 'timestamp', 'task_index']
        features = [f for f in features if not any(k == f or f.endswith(f".{k}") for k in internal_keys)]
        
        for f in features:
            item = QListWidgetItem(f)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            # Pre-check if already in tasks
            item.setCheckState(Qt.Checked if f in self.processor.features_to_remove else Qt.Unchecked)
            list_widget.addItem(item)
        
        d_layout.addWidget(list_widget)
        
        btns_layout = QHBoxLayout()
        ok_btn = QPushButton("Add to Tasks")
        ok_btn.clicked.connect(dialog.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        btns_layout.addWidget(ok_btn)
        btns_layout.addWidget(cancel_btn)
        d_layout.addLayout(btns_layout)
        
        if dialog.exec() == QDialog.Accepted:
            count = 0
            for i in range(list_widget.count()):
                item = list_widget.item(i)
                f_name = item.text()
                if item.checkState() == Qt.Checked:
                    self.processor.add_remove_feature_task(f_name)
                    count += 1
                else:
                    # If it was there but now unchecked, remove from tasks
                    if f_name in self.processor.features_to_remove:
                        self.processor.features_to_remove.remove(f_name)
            
            self.refresh_edit_ui()
            self.status_label.setText(f"Updated feature removal tasks")

    def _parse_indices(self, text: str) -> List[int]:
        """Parses strings like '0, 2, 5-10' into a list of integers."""
        indices = set()
        parts = [p.strip() for p in text.split(",")]
        for part in parts:
            if "-" in part:
                start_str, end_str = part.split("-")
                start, end = int(start_str), int(end_str)
                for i in range(start, end + 1):
                    indices.add(i)
            else:
                indices.add(int(part))
        return sorted(list(indices))

    def refresh_edit_ui(self):
        self.pending_op_list.clear()
        
        # 1. Episode Deletions
        for ep_idx in sorted(self.processor.to_delete_episodes):
            item = QListWidgetItem(f"Delete Episode {ep_idx}")
            item.setData(Qt.UserRole, {"type": "delete_episode", "idx": ep_idx})
            self.pending_op_list.addItem(item)
            
        # 2. Feature Removal
        for f_name in sorted(list(self.processor.features_to_remove)):
            item = QListWidgetItem(f"Remove Feature: {f_name}")
            item.setData(Qt.UserRole, {"type": "remove_feature", "name": f_name})
            self.pending_op_list.addItem(item)
            
        # 3. Trim Tasks
        for i, task in enumerate(self.processor.trim_tasks):
            item = QListWidgetItem(f"Trim Ep {task['episode_index']}: {task['start_frame']}-{task['end_frame']}")
            item.setData(Qt.UserRole, {"type": "trim", "idx": i})
            self.pending_op_list.addItem(item)
            
        # 4. Frame Edits
        for i, task in enumerate(self.processor.frame_edit_tasks):
            item = QListWidgetItem(f"Edit Ep {task['episode_index']} Frame {task['frame_index']}")
            item.setData(Qt.UserRole, {"type": "edit_frame", "idx": i})
            self.pending_op_list.addItem(item)
        
        # Update default new repo id if empty
        if self.processor.dataset and not self.new_repo_input.text():
            self.new_repo_input.setText(f"{self.processor.dataset.repo_id}_modified")

    def show_op_context_menu(self, position):
        item = self.pending_op_list.itemAt(position)
        if not item: return
        
        data = item.data(Qt.UserRole)
        if not data: return
        
        menu = QMenu()
        undo_action = menu.addAction("Undo This Operation")
        
        if data["type"] == "delete_episode":
            undo_action.triggered.connect(lambda: self.undo_delete_task(data["idx"]))
        elif data["type"] == "remove_feature":
            undo_action.triggered.connect(lambda: self.undo_remove_feature_task(data["name"]))
        elif data["type"] == "trim":
            undo_action.triggered.connect(lambda: self.undo_trim_task(data["idx"]))
        elif data["type"] == "edit_frame":
            undo_action.triggered.connect(lambda: self.undo_edit_frame_task(data["idx"]))
            
        menu.exec(self.pending_op_list.mapToGlobal(position))

    def undo_trim_task(self, idx):
        if 0 <= idx < len(self.processor.trim_tasks):
            self.processor.trim_tasks.pop(idx)
            self.refresh_edit_ui()
            self.status_label.setText("Undid trim task")

    def undo_edit_frame_task(self, idx):
        if 0 <= idx < len(self.processor.frame_edit_tasks):
            self.processor.frame_edit_tasks.pop(idx)
            self.refresh_edit_ui()
            self.status_label.setText("Undid frame edit task")

    def undo_remove_feature_task(self, f_name):
        if f_name in self.processor.features_to_remove:
            self.processor.features_to_remove.remove(f_name)
            self.refresh_edit_ui()
            self.status_label.setText(f"Undid removal of feature {f_name}")

    def undo_delete_task(self, ep_idx):
        if ep_idx in self.processor.to_delete_episodes:
            self.processor.to_delete_episodes.remove(ep_idx)
            self.refresh_edit_ui()
            self.status_label.setText(f"Undid deletion of Episode {ep_idx}")

    def on_save_dataset_clicked(self):
        new_repo_id = self.new_repo_input.text().strip()
        if not new_repo_id:
            QMessageBox.warning(self, "Warning", "Please enter a New Repo ID")
            return
            
        # Confirm overwrite if same
        if self.processor.dataset and new_repo_id == self.processor.dataset.repo_id:
            reply = QMessageBox.question(self, "Confirm Overwrite", 
                                       f"New Repo ID is the same as current. Overwrite {new_repo_id}?",
                                       QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No:
                return

        self.status_label.setText("Saving dataset... this may take a while.")
        self.save_dataset_btn.setEnabled(False)
        self.repaint() # Force UI update
        
        # Run in main thread or simple thread to avoid GUI lock
        # For multi-tasking, we should use a proper thread, but let's fix logic first
        try:
            new_dataset = self.processor.apply_edits(new_repo_id)
            QMessageBox.information(self, "Success", f"Dataset saved to {new_dataset.root}")
            self.status_label.setText(f"Dataset saved to {new_repo_id}")
            
            # Reset tasks UI
            self.refresh_edit_ui()
            
            # Switch back to Visualize and load new dataset
            self.right_tabs.setCurrentIndex(0)
            self.repo_input.setText(new_repo_id)
            self.on_load_clicked()
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to save dataset: {str(e)}")
            self.status_label.setText("Save failed")
        finally:
            self.save_dataset_btn.setEnabled(True)

    def on_choose_dataset_clicked(self):
        """Open a folder browser dialog to choose a dataset."""
        # Get the default path based on current username
        username = os.getenv('USER') or os.getenv('USERNAME') or 'user'
        default_path = Path.home() / '.cache' / 'huggingface' / 'lerobot' / 'sereact'
        
        # Create the directory if it doesn't exist (for convenience)
        default_path.mkdir(parents=True, exist_ok=True)
        
        # Open folder dialog
        folder = QFileDialog.getExistingDirectory(
            self,
            "Choose Dataset",
            str(default_path),
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if folder:
            # Convert the selected path to a repo_id format
            folder_path = Path(folder)
            
            # Try to extract repo_id from the path
            # Expected structure: ~/.cache/huggingface/lerobot/{org}/{dataset_name}
            # So we want to get the last two parts as "org/dataset_name"
            try:
                parts = folder_path.parts
                # Find 'lerobot' in path and take the next two parts
                if 'lerobot' in parts:
                    lerobot_idx = parts.index('lerobot')
                    if len(parts) > lerobot_idx + 2:
                        repo_id = f"{parts[lerobot_idx + 1]}/{parts[lerobot_idx + 2]}"
                        self.repo_input.setText(repo_id)
                        return
                # Fallback: just use the last two parts
                if len(parts) >= 2:
                    repo_id = f"{parts[-2]}/{parts[-1]}"
                    self.repo_input.setText(repo_id)
                else:
                    self.repo_input.setText(folder_path.name)
            except Exception:
                # If all else fails, just use the folder name
                self.repo_input.setText(folder_path.name)

    def on_load_clicked(self):
        repo_id = self.repo_input.text().strip()
        if not repo_id: return
        self.load_btn.setEnabled(False)
        self.status_label.setText(f"Loading {repo_id}...")
        self.loader_thread = LoaderThread(self.processor, repo_id)
        self.loader_thread.finished.connect(self.on_load_finished)
        self.loader_thread.error.connect(self.on_load_error)
        self.loader_thread.start()

    def show_episode_context_menu(self, position):
        item = self.ep_list.itemAt(position)
        if not item: return
        
        menu = QMenu()
        ep_idx = self.ep_list.row(item)
        
        mark_action = menu.addAction(f"Mark Episode {ep_idx} for Deletion")
        mark_action.triggered.connect(lambda: self.mark_episode_for_deletion(ep_idx))
        
        menu.exec(self.ep_list.mapToGlobal(position))

    def mark_episode_for_deletion(self, ep_idx):
        self.processor.add_delete_episode_task(ep_idx)
        self.refresh_edit_ui()
        self.status_label.setText(f"Episode {ep_idx} marked for deletion")
        # Switch to Edit tab to show the change
        self.right_tabs.setCurrentIndex(1)

    def on_load_finished(self, dataset):
        self.load_btn.setEnabled(True)
        self.status_label.setText("Loaded")
        
        # Reset tasks and UI on new load
        self.processor.clear_edit_tasks()
        self.new_repo_input.clear()
        self.refresh_edit_ui()
        
        # Try to load task description from episodes parquet
        task_description = self.get_task_description(dataset)
        
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
        
        # Build info text with task at the top
        task_html = ""
        if task_description:
            task_html = f"<b style='color: #e74c3c;'>Task:</b> <i>{task_description}</i><br><br>"
        
        info_text = (
            f"{task_html}"
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
        
        # Load approvals and update colors
        self.load_approvals()
        self.update_episode_list_colors()
        
        self.init_dimension_selectors(dataset)
        self.init_camera_grid(dataset)
        if dataset.meta.total_episodes > 0:
            self.ep_list.setCurrentRow(0)

    def init_camera_grid(self, dataset):
        """Initialize the multi-camera grid based on available image/video keys."""
        # Clear existing widgets from grid layout
        while self.multi_camera_layout.count():
            item = self.multi_camera_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.camera_labels.clear()
        
        # Find all image keys
        sample = dataset[0]
        self.image_keys = sorted([k for k in sample.keys() if 'image' in k])
        
        if not self.image_keys:
            self.show_all_cameras_cb.setEnabled(False)
            return
        
        self.show_all_cameras_cb.setEnabled(True)
        
        # Calculate grid dimensions (as square as possible)
        num_cameras = len(self.image_keys)
        cols = int(np.ceil(np.sqrt(num_cameras)))
        rows = int(np.ceil(num_cameras / cols))
        
        # Create labels for each camera
        for idx, key in enumerate(self.image_keys):
            row = idx // cols
            col = idx % cols
            
            # Create a container widget for each camera with label overlay
            container = QWidget()
            container_layout = QVBoxLayout(container)
            container_layout.setContentsMargins(0, 0, 0, 0)
            container_layout.setSpacing(0)
            
            # Camera name label (extract short name from key)
            short_name = key.split('.')[-1] if '.' in key else key
            name_label = QLabel(short_name)
            name_label.setStyleSheet("background-color: rgba(0, 0, 0, 150); color: white; padding: 2px 6px; font-size: 11px;")
            name_label.setAlignment(Qt.AlignCenter)
            
            # Image label
            img_label = QLabel("No Image")
            img_label.setAlignment(Qt.AlignCenter)
            img_label.setStyleSheet("background-color: #1a1a1a;")
            img_label.setMinimumSize(100, 75)
            
            container_layout.addWidget(name_label)
            container_layout.addWidget(img_label, stretch=1)
            
            self.multi_camera_layout.addWidget(container, row, col)
            self.camera_labels[key] = img_label
        
        # Set equal stretch for all rows and columns
        for r in range(rows):
            self.multi_camera_layout.setRowStretch(r, 1)
        for c in range(cols):
            self.multi_camera_layout.setColumnStretch(c, 1)

    def on_show_all_cameras_changed(self, state):
        """Toggle between single and multi-camera view."""
        if self.show_all_cameras_cb.isChecked():
            self.image_stack.setCurrentIndex(1)  # Multi-camera grid
        else:
            self.image_stack.setCurrentIndex(0)  # Single camera
        
        # Refresh the current frame
        if self.processor.dataset:
            self.update_frame_view(self.frame_slider.value())

    def init_dimension_selectors(self, dataset):
        """Initializes the hierarchical feature tree with Left/Right groupings."""
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
            
            # Get dimension names from metadata if available
            feature_info = dataset.meta.features.get(k, {})
            dim_names = feature_info.get('names', [])
            
            # Create Tree Item
            parent = QTreeWidgetItem(self.feature_tree)
            parent.setText(0, k)
            parent.setCheckState(0, Qt.Checked)
            parent.setExpanded(True)
            # Mark as feature parent
            parent.setData(0, Qt.UserRole, {"type": "feature", "key": k})
            
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
                # Categorize dimensions into left, right, and other
                left_dims = []
                right_dims = []
                other_dims = []
                
                for d in range(dims):
                    if d < len(dim_names):
                        name = dim_names[d]
                    else:
                        name = f"Dimension {d}"
                    
                    name_lower = name.lower()
                    if name_lower.startswith("left"):
                        left_dims.append((d, name))
                    elif name_lower.startswith("right"):
                        right_dims.append((d, name))
                    else:
                        other_dims.append((d, name))
                
                # Create "Left" group if there are left joints
                if left_dims:
                    left_group = QTreeWidgetItem(parent)
                    left_group.setText(0, "🔵 Left")
                    left_group.setCheckState(0, Qt.Checked)
                    left_group.setData(0, Qt.UserRole, {"type": "group", "key": k, "group": "left"})
                    left_group.setExpanded(False)
                    
                    for d, name in left_dims:
                        child = QTreeWidgetItem(left_group)
                        child.setText(0, name)
                        child.setCheckState(0, Qt.Checked)
                        child.setData(0, Qt.UserRole, {"type": "dimension", "key": k, "dim": d})
                
                # Create "Right" group if there are right joints
                if right_dims:
                    right_group = QTreeWidgetItem(parent)
                    right_group.setText(0, "🔴 Right")
                    right_group.setCheckState(0, Qt.Checked)
                    right_group.setData(0, Qt.UserRole, {"type": "group", "key": k, "group": "right"})
                    right_group.setExpanded(False)
                    
                    for d, name in right_dims:
                        child = QTreeWidgetItem(right_group)
                        child.setText(0, name)
                        child.setCheckState(0, Qt.Checked)
                        child.setData(0, Qt.UserRole, {"type": "dimension", "key": k, "dim": d})
                
                # Add "Other" joints directly under parent (or in a group if many)
                if other_dims:
                    if left_dims or right_dims:
                        # Create "Other" group only if we have left/right groups
                        other_group = QTreeWidgetItem(parent)
                        other_group.setText(0, "⚪ Other")
                        other_group.setCheckState(0, Qt.Checked)
                        other_group.setData(0, Qt.UserRole, {"type": "group", "key": k, "group": "other"})
                        other_group.setExpanded(False)
                        
                        for d, name in other_dims:
                            child = QTreeWidgetItem(other_group)
                            child.setText(0, name)
                            child.setCheckState(0, Qt.Checked)
                            child.setData(0, Qt.UserRole, {"type": "dimension", "key": k, "dim": d})
                    else:
                        # No left/right groups, add directly under parent
                        for d, name in other_dims:
                            child = QTreeWidgetItem(parent)
                            child.setText(0, name)
                            child.setCheckState(0, Qt.Checked)
                            child.setData(0, Qt.UserRole, {"type": "dimension", "key": k, "dim": d})

    def on_tree_item_changed(self, item, column):
        """Handles visibility toggles from the tree."""
        item_data = item.data(0, Qt.UserRole)
        is_checked = item.checkState(0) == Qt.Checked
        
        if not item_data:
            return
        
        item_type = item_data.get("type")
        key = item_data.get("key")
        
        if item_type == "feature":
            # Feature parent (action, observation.state, etc.)
            # Propagate check state to all children (groups and dimensions)
            self._set_children_check_state(item, is_checked)
            # Update plot visibility based on whether any dimension is visible
            self._update_plot_visibility(key)
            
        elif item_type == "group":
            # Left/Right/Other group - propagate to children only
            self._set_children_check_state(item, is_checked)
            # Update plot visibility
            self._update_plot_visibility(key)
            
        elif item_type == "dimension":
            # Individual dimension - update curve visibility
            dim = item_data.get("dim")
            if key in self.plot_curves and dim < len(self.plot_curves[key]):
                self.plot_curves[key][dim].setVisible(is_checked)
            # Update plot visibility (show plot if at least one curve is visible)
            self._update_plot_visibility(key)

    def _set_children_check_state(self, item, checked: bool):
        """Recursively set check state for all children."""
        state = Qt.Checked if checked else Qt.Unchecked
        for i in range(item.childCount()):
            child = item.child(i)
            child.setCheckState(0, state)
            # Recurse for nested children (dimensions under groups)
            self._set_children_check_state(child, checked)

    def _update_plot_visibility(self, key: str):
        """Update plot visibility based on whether any of its dimensions are checked."""
        if key not in self.plots:
            return
        
        # Find the feature parent item
        for i in range(self.feature_tree.topLevelItemCount()):
            parent = self.feature_tree.topLevelItem(i)
            parent_data = parent.data(0, Qt.UserRole)
            if parent_data and parent_data.get("key") == key:
                # Check if any dimension is checked
                any_visible = self._has_any_checked_dimension(parent)
                self.plots[key].setVisible(any_visible)
                
                # Also update individual curve visibility
                self._update_curves_visibility(parent, key)
                break

    def _has_any_checked_dimension(self, item) -> bool:
        """Recursively check if any dimension under this item is checked."""
        for i in range(item.childCount()):
            child = item.child(i)
            child_data = child.data(0, Qt.UserRole)
            if child_data:
                if child_data.get("type") == "dimension":
                    if child.checkState(0) == Qt.Checked:
                        return True
                else:
                    # It's a group, recurse
                    if self._has_any_checked_dimension(child):
                        return True
        return False

    def _update_curves_visibility(self, item, key: str):
        """Recursively update curve visibility for all dimensions under this item."""
        for i in range(item.childCount()):
            child = item.child(i)
            child_data = child.data(0, Qt.UserRole)
            if child_data:
                if child_data.get("type") == "dimension":
                    dim = child_data.get("dim")
                    is_checked = child.checkState(0) == Qt.Checked
                    if key in self.plot_curves and dim < len(self.plot_curves[key]):
                        self.plot_curves[key][dim].setVisible(is_checked)
                else:
                    # It's a group, recurse
                    self._update_curves_visibility(child, key)

    def show_feature_tree_context_menu(self, position):
        """Show context menu for feature tree with left/right selection options."""
        item = self.feature_tree.itemAt(position)
        if not item:
            return
        
        # Find the feature parent item (action, observation.state, etc.)
        feature_item = item
        while feature_item.parent():
            feature_item = feature_item.parent()
        
        menu = QMenu()
        
        # Add check/uncheck all
        check_all = menu.addAction("✓ Check all")
        uncheck_all = menu.addAction("✗ Uncheck all")
        
        # Connect actions
        check_all.triggered.connect(lambda: self.set_all_joints(feature_item, True))
        uncheck_all.triggered.connect(lambda: self.set_all_joints(feature_item, False))
        
        menu.exec(self.feature_tree.mapToGlobal(position))

    def set_all_joints(self, parent_item, checked: bool):
        """Check or uncheck all child items recursively."""
        state = Qt.Checked if checked else Qt.Unchecked
        parent_item.setCheckState(0, state)

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
        
        # Stop any ongoing playback
        self.stop_playback()
        
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
        
        # Update approve button state
        self.update_approve_button_state(row)

    def get_approvals_file_path(self) -> Optional[Path]:
        """Get the path to the approvals JSON file in the dataset directory."""
        if not self.processor.dataset:
            return None
        # The dataset root is typically ~/.cache/huggingface/lerobot/{org}/{dataset}
        dataset_root = Path(self.processor.dataset.root)
        return dataset_root / "episode_approvals.json"

    def load_episode_tasks(self, dataset):
        """Load task descriptions for all episodes from episodes parquet files."""
        self.episode_tasks.clear()
        
        try:
            dataset_root = Path(dataset.root)
            # Structure: meta/episodes/chunk-XXX/file-XXX.parquet
            episodes_dir = dataset_root / "meta" / "episodes"
            
            if not episodes_dir.exists():
                print(f"Episodes directory not found: {episodes_dir}")
                return
            
            # Collect all parquet files from chunk directories
            all_dfs = []
            chunk_dir = episodes_dir / "chunk-000"
            
            if chunk_dir.exists() and chunk_dir.is_dir():
                parquet_files = sorted(chunk_dir.glob("*.parquet"))
                for parquet_file in parquet_files:
                    try:
                        table = pq.read_table(parquet_file)
                        df = table.to_pandas()
                        all_dfs.append(df)
                    except Exception as e:
                        print(f"Warning: Could not read {parquet_file}: {e}")
            
            if not all_dfs:
                print("No parquet files found in episodes directory")
                return
            
            # Concatenate all dataframes
            combined_df = pd.concat(all_dfs, ignore_index=True)
            
            # Extract tasks for each episode
            if 'tasks' in combined_df.columns and 'episode_index' in combined_df.columns:
                for _, row in combined_df.iterrows():
                    ep_idx = int(row['episode_index'])
                    tasks = row['tasks']
                    
                    # Tasks is usually a list of strings
                    if isinstance(tasks, (list, np.ndarray)) and len(tasks) > 0:
                        self.episode_tasks[ep_idx] = str(tasks[0])
                    elif isinstance(tasks, str):
                        self.episode_tasks[ep_idx] = tasks
                
                print(f"Loaded tasks for {len(self.episode_tasks)} episodes")
            else:
                print(f"Columns in parquet: {list(combined_df.columns)}")
                
        except Exception as e:
            import traceback
            print(f"Warning: Could not load episode tasks: {e}")
            traceback.print_exc()

    def get_task_description(self, dataset) -> Optional[str]:
        """Get the first task description (for display in info panel)."""
        # Load all episode tasks first
        self.load_episode_tasks(dataset)
        
        # Return the first task if available
        if self.episode_tasks:
            first_ep = min(self.episode_tasks.keys())
            return self.episode_tasks.get(first_ep)
        return None

    def load_approvals(self):
        """Load episode approvals from the JSON file in the dataset directory."""
        self.approved_episodes.clear()
        self.commented_episodes.clear()
        self.approvals_file_path = self.get_approvals_file_path()
        
        if self.approvals_file_path and self.approvals_file_path.exists():
            try:
                with open(self.approvals_file_path, 'r') as f:
                    data = json.load(f)
                    self.approved_episodes = set(data.get('approved_episodes', []))
                    # Load comments (convert string keys back to int)
                    comments_data = data.get('commented_episodes', {})
                    self.commented_episodes = {int(k): v for k, v in comments_data.items()}
                self.status_label.setText(f"Loaded {len(self.approved_episodes)} approved, {len(self.commented_episodes)} commented episodes")
            except Exception as e:
                self.status_label.setText(f"Warning: Could not load approvals: {e}")

    def save_approvals(self):
        """Save episode approvals to the JSON file in the dataset directory and backup."""
        if not self.approvals_file_path:
            self.approvals_file_path = self.get_approvals_file_path()
        
        if self.approvals_file_path:
            try:
                data = {
                    'approved_episodes': sorted(list(self.approved_episodes)),
                    'total_approved': len(self.approved_episodes),
                    'commented_episodes': {str(k): v for k, v in self.commented_episodes.items()},
                    'total_commented': len(self.commented_episodes)
                }
                # Save to dataset directory
                with open(self.approvals_file_path, 'w') as f:
                    json.dump(data, f, indent=2)
                
                # Also save backup copy to project directory
                self.save_approvals_backup()
            except Exception as e:
                QMessageBox.warning(self, "Warning", f"Could not save approvals: {e}")

    def save_approvals_backup(self):
        """Save a backup copy of approvals to the project directory."""
        if not self.approvals_file_path or not self.processor.dataset:
            return
        
        try:
            # Create backup directory if it doesn't exist
            APPROVALS_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
            
            # Create a filename based on the dataset repo_id
            repo_id = self.processor.dataset.repo_id
            safe_name = repo_id.replace('/', '_')
            backup_path = APPROVALS_BACKUP_DIR / f"{safe_name}_approvals.json"
            
            # Copy the file
            shutil.copy2(self.approvals_file_path, backup_path)
        except Exception as e:
            # Don't show error for backup failures, just log to status
            self.status_label.setText(f"Warning: Backup copy failed: {e}")

    def on_approve_episode_clicked(self):
        """Toggle approval status for the current episode."""
        ep_idx = self.ep_list.currentRow()
        if ep_idx < 0:
            return
        
        if ep_idx in self.approved_episodes:
            # Unapprove
            self.approved_episodes.remove(ep_idx)
            self.status_label.setText(f"Episode {ep_idx} unapproved")
        else:
            # Approve - also remove any comment
            self.approved_episodes.add(ep_idx)
            if ep_idx in self.commented_episodes:
                del self.commented_episodes[ep_idx]
            self.status_label.setText(f"Episode {ep_idx} approved")
        
        # Save to file
        self.save_approvals()
        
        # Update UI
        self.update_episode_list_colors()
        self.update_approve_button_state(ep_idx)

    def update_approve_button_state(self, ep_idx: int):
        """Update the approve button text and style based on current episode's status."""
        if ep_idx in self.approved_episodes:
            self.approve_btn.setText("Unapprove Episode")
            self.approve_btn.setStyleSheet("background-color: #e74c3c; color: white; font-weight: bold;")
        else:
            self.approve_btn.setText("Approve Episode")
            self.approve_btn.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold;")

    def update_episode_list_colors(self):
        """Update episode list item colors based on approval/comment status."""
        for i in range(self.ep_list.count()):
            item = self.ep_list.item(i)
            # Build tooltip with task info
            task_info = ""
            if i in self.episode_tasks:
                task_info = f"Task: {self.episode_tasks[i][:80]}..." if len(self.episode_tasks.get(i, "")) > 80 else f"Task: {self.episode_tasks.get(i, '')}"
            
            if i in self.approved_episodes:
                # Green color for approved episodes
                item.setForeground(QBrush(QColor("#27ae60")))
                tooltip = "✓ Approved"
                if task_info:
                    tooltip = f"{tooltip}\n{task_info}"
                item.setToolTip(tooltip)
            elif i in self.commented_episodes:
                # Yellow/orange color for commented episodes
                item.setForeground(QBrush(QColor("#f39c12")))
                tooltip = f"Comment: {self.commented_episodes[i]}"
                if task_info:
                    tooltip = f"{tooltip}\n{task_info}"
                item.setToolTip(tooltip)
            else:
                # Default color (reset to black/default)
                item.setForeground(QBrush(QColor("#000000")))
                item.setToolTip(task_info if task_info else "")

    def on_comment_episode_clicked(self):
        """Open a dialog to add a comment to the current episode."""
        from PySide6.QtWidgets import QDialog, QDialogButtonBox
        
        ep_idx = self.ep_list.currentRow()
        if ep_idx < 0:
            return
        
        # Create comment dialog
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Comment on Episode {ep_idx}")
        dialog.resize(400, 200)
        d_layout = QVBoxLayout(dialog)
        
        d_layout.addWidget(QLabel("Enter your comment:"))
        
        comment_edit = QTextEdit()
        # Pre-fill with existing comment if any
        if ep_idx in self.commented_episodes:
            comment_edit.setText(self.commented_episodes[ep_idx])
        d_layout.addWidget(comment_edit)
        
        btns_layout = QHBoxLayout()
        save_btn = QPushButton("Save Comment")
        save_btn.clicked.connect(dialog.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        clear_btn = QPushButton("Clear Comment")
        clear_btn.clicked.connect(lambda: comment_edit.clear())
        btns_layout.addWidget(save_btn)
        btns_layout.addWidget(clear_btn)
        btns_layout.addWidget(cancel_btn)
        d_layout.addLayout(btns_layout)
        
        if dialog.exec() == QDialog.Accepted:
            comment_text = comment_edit.toPlainText().strip()
            
            if comment_text:
                # Add comment and mark as NOT approved
                self.commented_episodes[ep_idx] = comment_text
                # Remove from approved if it was approved
                if ep_idx in self.approved_episodes:
                    self.approved_episodes.remove(ep_idx)
                self.status_label.setText(f"Episode {ep_idx} commented")
            else:
                # Empty comment - remove comment
                if ep_idx in self.commented_episodes:
                    del self.commented_episodes[ep_idx]
                self.status_label.setText(f"Episode {ep_idx} comment cleared")
            
            # Save to file
            self.save_approvals()
            
            # Update UI
            self.update_episode_list_colors()
            self.update_approve_button_state(ep_idx)

    def on_play_1x_clicked(self):
        """Start playback at 1x speed."""
        self.start_playback(1.0)
        self.play_1x_btn.setStyleSheet("background-color: #2980b9; color: white; font-weight: bold; border: 2px solid white;")
        self.play_2x_btn.setStyleSheet("background-color: #9b59b6; color: white; font-weight: bold;")

    def on_play_2x_clicked(self):
        """Start playback at 2x speed."""
        self.start_playback(2.0)
        self.play_2x_btn.setStyleSheet("background-color: #8e44ad; color: white; font-weight: bold; border: 2px solid white;")
        self.play_1x_btn.setStyleSheet("background-color: #3498db; color: white; font-weight: bold;")

    def on_stop_playback_clicked(self):
        """Stop playback."""
        self.stop_playback()

    def start_playback(self, speed: float):
        """Start playing frames at the given speed multiplier."""
        if not self.processor.dataset:
            return
        
        # Stop any existing playback first
        if self.playback_timer:
            self.playback_timer.stop()
            self.playback_timer.deleteLater()
            self.playback_timer = None
        
        self.playback_speed = speed
        
        # Get FPS from dataset metadata
        fps = self.processor.dataset.meta.fps if hasattr(self.processor.dataset.meta, 'fps') else 30
        
        # Calculate interval in milliseconds
        # At 1x speed, interval = 1000/fps ms
        # At 2x speed, interval = 1000/(fps*2) = 500/fps ms
        interval_ms = max(1, int(1000 / (fps * speed)))
        
        # Create and start timer with parent
        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self.playback_next_frame)
        self.playback_timer.start(interval_ms)
        
        self.status_label.setText(f"Playing at {speed}x speed ({fps} fps, interval: {interval_ms}ms)")

    def stop_playback(self):
        """Stop the playback timer."""
        if self.playback_timer:
            self.playback_timer.stop()
            self.playback_timer.deleteLater()
            self.playback_timer = None
        
        # Reset button styles
        self.play_1x_btn.setStyleSheet("background-color: #3498db; color: white; font-weight: bold;")
        self.play_2x_btn.setStyleSheet("background-color: #9b59b6; color: white; font-weight: bold;")

    def playback_next_frame(self):
        """Advance to the next frame during playback."""
        current = self.frame_slider.value()
        max_val = self.frame_slider.maximum()
        
        if current < max_val:
            self.frame_slider.setValue(current + 1)
        else:
            # Reached end of episode, stop playback
            self.stop_playback()
            self.status_label.setText("Playback finished")
    
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
            parent_data = parent.data(0, Qt.UserRole)
            if not parent_data:
                continue
            
            key = parent_data.get("key")
            if not key:
                continue
            
            # Check if any dimension is visible
            any_visible = self._has_any_checked_dimension(parent)
            
            if key in self.plots:
                self.plots[key].setVisible(any_visible)
            
            # Update all curve visibility recursively
            self._update_curves_visibility(parent, key)

    def _convert_image_to_numpy(self, img_data) -> Optional[np.ndarray]:
        """Convert various image formats to numpy array (HWC, uint8, RGB)."""
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
            # Video format - not yet supported
            return None
        else:
            img_np = np.array(img_data)
        
        # Ensure C-contiguous array
        img_np = np.ascontiguousarray(img_np)
        
        # Handle grayscale
        if img_np.ndim == 2:
            img_np = np.stack([img_np] * 3, axis=-1)
        
        return img_np

    def _display_image_on_label(self, img_np: np.ndarray, label: QLabel):
        """Display a numpy image array on a QLabel."""
        h, w, c = img_np.shape
        bytes_per_line = c * w
        qimg = QImage(img_np.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimg)
        
        # Get label size, use minimum if too small
        label_size = label.size()
        if label_size.width() < 10 or label_size.height() < 10:
            label_size = label.minimumSize()
            if label_size.width() < 10:
                label_size.setWidth(320)
            if label_size.height() < 10:
                label_size.setHeight(240)
        
        # Scale to fit label while keeping aspect ratio
        scaled_pixmap = pixmap.scaled(
            label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled_pixmap)

    def update_frame_view(self, frame_idx):
        try:
            data = self.processor.get_frame(frame_idx)
            img_keys = [k for k in data.keys() if 'image' in k]
            
            if not img_keys:
                return
            
            # Check if showing all cameras
            if self.show_all_cameras_cb.isChecked() and len(self.camera_labels) > 0:
                # Multi-camera view
                for key in self.image_keys:
                    if key in data and key in self.camera_labels:
                        img_np = self._convert_image_to_numpy(data[key])
                        if img_np is not None:
                            self._display_image_on_label(img_np, self.camera_labels[key])
            else:
                # Single camera view (first camera)
                img_np = self._convert_image_to_numpy(data[img_keys[0]])
                if img_np is not None:
                    self._display_image_on_label(img_np, self.image_label)
                elif isinstance(data[img_keys[0]], dict) and 'path' in data[img_keys[0]]:
                    self.status_label.setText("Video format not yet supported in frame view")
                    
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()

    def on_load_error(self, err):
        self.load_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", err)

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
        elif event.key() == Qt.Key_Space:
            # Toggle playback with spacebar
            if self.playback_timer and self.playback_timer.isActive():
                self.stop_playback()
            else:
                self.start_playback(1.0)
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        """Stop playback when closing the application."""
        self.stop_playback()
        super().closeEvent(event)


def run_qt_app():
    """Run the Qt desktop application."""
    app = QApplication(sys.argv)
    gui = DatasetGui()
    gui.show()
    sys.exit(app.exec())


def run_web_app(port: int):
    """Run the web application."""
    from web_app import run_web_server
    run_web_server(port)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='LeRobot Dataset Visualizer')
    parser.add_argument('--web', type=int, metavar='PORT', 
                        help='Run as web server on specified port (e.g., --web 3000)')
    
    args = parser.parse_args()
    
    if args.web:
        run_web_app(args.web)
    else:
        run_qt_app()