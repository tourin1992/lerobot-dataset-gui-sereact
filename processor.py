from pathlib import Path
from typing import Optional, Any, Dict, List

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset


class DatasetProcessor:
    """Handles LeRobot dataset loading and manipulation."""

    def __init__(self):
        self.dataset: Optional[LeRobotDataset] = None
        self.raw_hf_dataset = None
        self.to_delete_episodes: set[int] = set()
        self.features_to_remove: set[str] = set()
        self.trim_tasks: List[Dict] = [] # List of {"episode_index": int, "start_frame": int, "end_frame": int}
        self.frame_edit_tasks: List[Dict] = [] # List of {"episode_index": int, "frame_index": int, "features": Dict}

    def clear_edit_tasks(self):
        """Clears all pending edit tasks."""
        self.to_delete_episodes.clear()
        self.features_to_remove.clear()
        self.trim_tasks.clear()
        self.frame_edit_tasks.clear()

    def add_delete_episode_task(self, episode_idx: int):
        """Adds an episode to the deletion task pool."""
        self.to_delete_episodes.add(episode_idx)

    def add_remove_feature_task(self, feature_name: str):
        """Adds a feature to the removal task pool."""
        self.features_to_remove.add(feature_name)

    def add_trim_task(self, episode_idx: int, start_frame: int, end_frame: int):
        """Adds a trim task (remove frames from start to end within an episode)."""
        self.trim_tasks.append({
            "episode_index": episode_idx,
            "start_frame": start_frame,
            "end_frame": end_frame
        })

    def add_frame_edit_task(self, episode_idx: int, frame_idx: int, features: Dict):
        """Adds a frame edit task (modify features for a specific frame)."""
        self.frame_edit_tasks.append({
            "episode_index": episode_idx,
            "frame_index": frame_idx,
            "features": features
        })

    def apply_edits(self, new_repo_id: str) -> LeRobotDataset:
        """Applies all pending edit tasks and saves to a new dataset."""
        if self.dataset is None:
            raise ValueError("No dataset loaded")
            
        import logging
        import shutil
        from lerobot.datasets.dataset_tools import delete_episodes, remove_feature, merge_datasets
        from lerobot.utils.constants import HF_LEROBOT_HOME
        
        # 确定输出路径，参考 lerobot_edit_dataset.py 的逻辑
        # 默认保存到 lerobot 的主数据目录下
        output_dir = HF_LEROBOT_HOME / new_repo_id
        
        # 如果输出目录已存在且不是当前加载的目录，则备份/清理以便写入新数据
        if output_dir.exists() and output_dir != self.dataset.root:
            old_path = Path(str(output_dir) + "_old")
            logging.info(f"Output directory {output_dir} exists, moving to {old_path}")
            if old_path.exists():
                shutil.rmtree(old_path)
            shutil.move(str(output_dir), str(old_path))

        # 检查是否有编辑任务
        has_edits = bool(self.features_to_remove or self.to_delete_episodes or self.trim_tasks or self.frame_edit_tasks)
        
        # 场景 1: 如果没有任何编辑任务，且用户指定了不同的 repo_id，则执行“另存为”操作
        if not has_edits:
            if new_repo_id == self.dataset.repo_id:
                logging.info("No edits and same repo_id, nothing to save.")
                return self.dataset
                
            logging.info(f"No edits pending. Copying dataset from {self.dataset.repo_id} to {new_repo_id}")
            # 使用 merge_datasets 作为一个高效的拷贝方式，它会处理所有的元数据重索引
            new_dataset = merge_datasets([self.dataset], output_repo_id=new_repo_id, output_dir=output_dir)
            self.clear_edit_tasks()
            return new_dataset

        # 场景 2: 存在编辑任务，按顺序执行并链式处理结果
        current_ds = self.dataset
        
        # 1. 处理特征删除 (Feature Removal)
        if self.features_to_remove:
            logging.info(f"Removing features: {self.features_to_remove}")
            # 判断是否是最后一步，如果是则直接保存到最终目标
            is_last = not (self.to_delete_episodes or self.trim_tasks or self.frame_edit_tasks)
            target_id = new_repo_id if is_last else f"{new_repo_id}_tmp_feat"
            
            current_ds = remove_feature(
                current_ds,
                feature_names=list(self.features_to_remove),
                repo_id=target_id,
                output_dir=output_dir if is_last else None
            )

        # 2. 处理 Episode 删除 (Episode Deletions)
        if self.to_delete_episodes:
            logging.info(f"Deleting episodes: {self.to_delete_episodes}")
            is_last = not (self.trim_tasks or self.frame_edit_tasks)
            target_id = new_repo_id if is_last else f"{new_repo_id}_tmp_del"
            
            current_ds = delete_episodes(
                current_ds,
                episode_indices=list(self.to_delete_episodes),
                repo_id=target_id,
                output_dir=output_dir if is_last else None
            )
            
        # 3. 处理细粒度裁剪和帧编辑 (TODO: 目前仅作为警告，未来可扩展)
        if self.trim_tasks or self.frame_edit_tasks:
            logging.warning("Fine-grained trim and frame edits are recorded but not yet implemented in physical save. Skipping these tasks.")

        # 最终确认：如果中间步骤没有最终同步到 new_repo_id（即存在多步操作且最后一步不是直接保存到目标）
        if current_ds.repo_id != new_repo_id:
            logging.info(f"Finalizing chained edits to {new_repo_id}")
            final_ds = merge_datasets([current_ds], output_repo_id=new_repo_id, output_dir=output_dir)
            current_ds = final_ds

        self.clear_edit_tasks()
        return current_ds

    def load_dataset(self, repo_id: str, root: Optional[Path] = None) -> LeRobotDataset:
        """Loads a LeRobot dataset."""
        self.dataset = LeRobotDataset(repo_id, root=root)
        # Keep a reference to raw dataset (without torch transform) for images
        self.raw_hf_dataset = self.dataset.hf_dataset.with_format(None)
        return self.dataset

    @property
    def metadata(self):
        if self.dataset is None: return None
        return self.dataset.meta

    def get_episode_range(self, episode_idx: int) -> tuple[int, int]:
        """Returns (start_index, end_index) for an episode."""
        if self.dataset is None: return 0, 0
        from_idx = self.dataset.meta.episodes["dataset_from_index"][episode_idx]
        to_idx = self.dataset.meta.episodes["dataset_to_index"][episode_idx]
        return int(from_idx), int(to_idx)

    def get_frame(self, frame_idx: int) -> Dict[str, Any]:
        """Fetches data for a specific global frame index."""
        # Try to get frame with decoded video/images
        # LeRobot dataset with video format needs special handling
        frame_data = self.dataset[frame_idx]
        
        # Check if images are in video format (dict with 'path' key)
        # If so, they should already be decoded by LeRobotDataset
        return frame_data

    def get_episode_data(self, episode_idx: int, keys: List[str]) -> Dict[str, np.ndarray]:
        """Fetches all frames for an episode for specific keys (e.g., state, action)."""
        start, end = self.get_episode_range(episode_idx)
        
        # 优化：使用 select 批量获取，避免逐帧循环
        selected_data = self.dataset.hf_dataset.select(range(start, end))
        
        result = {}
        for key in keys:
            if key in selected_data.features:
                # 批量获取整列数据
                col_data = selected_data[key]
                if isinstance(col_data, list):
                    # 转换列表为 numpy 数组
                    if len(col_data) > 0:
                        if torch.is_tensor(col_data[0]):
                            result[key] = torch.stack(col_data).numpy()
                        else:
                            result[key] = np.array(col_data)
                elif torch.is_tensor(col_data):
                    result[key] = col_data.numpy()
                else:
                    result[key] = np.array(col_data)
        return result