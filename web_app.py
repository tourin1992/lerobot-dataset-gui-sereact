"""
Web UI for LeRobot Dataset Visualizer
Run with: python app.py --web 3000
"""

import io
import json
import base64
import time
import threading
from pathlib import Path
from typing import Optional, Dict, List, Set, Tuple
from collections import OrderedDict

import numpy as np
from flask import Flask, render_template, jsonify, request, send_file
from PIL import Image

from processor import DatasetProcessor

# Project directory for backup copies
PROJECT_DIR = Path(__file__).parent
APPROVALS_BACKUP_DIR = PROJECT_DIR / "approvals_backup"

# Frame cache settings
FRAME_CACHE_SIZE = 50  # Keep up to 50 frames in memory


class LRUCache:
    """Simple LRU cache for frames."""
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()
        self.lock = threading.Lock()
    
    def get(self, key) -> Optional[bytes]:
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                return self.cache[key]
        return None
    
    def put(self, key, value: bytes):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.max_size:
                    self.cache.popitem(last=False)  # Remove oldest
                self.cache[key] = value
    
    def clear(self):
        with self.lock:
            self.cache.clear()


class WebDatasetViewer:
    """Web-based dataset viewer backend."""
    
    def __init__(self):
        self.processor = DatasetProcessor()
        self.approved_episodes: Set[int] = set()
        self.commented_episodes: Dict[int, str] = {}
        self.episode_tasks: Dict[int, str] = {}
        self.approvals_file_path: Optional[Path] = None
        self.current_repo_id: Optional[str] = None
        self.frame_cache = LRUCache(FRAME_CACHE_SIZE)
        self.frame_lock = threading.Lock()  # Protect decoder access
    
    def get_default_dataset_path(self) -> Path:
        """Get the default path for dataset browser."""
        return Path.home() / '.cache' / 'huggingface' / 'lerobot' / 'sereact'
    
    def list_datasets(self) -> List[Dict]:
        """List available datasets in the default directory."""
        datasets = []
        base_path = self.get_default_dataset_path()
        
        if base_path.exists():
            for item in base_path.iterdir():
                if item.is_dir() and (item / "meta").exists():
                    datasets.append({
                        "name": item.name,
                        "repo_id": f"sereact/{item.name}",
                        "path": str(item)
                    })
        
        return sorted(datasets, key=lambda x: x["name"])
    
    def load_dataset(self, repo_id: str) -> Dict:
        """Load a dataset and return metadata."""
        try:
            # Clear frame cache when loading new dataset
            self.frame_cache.clear()
            
            dataset = self.processor.load_dataset(repo_id)
            self.current_repo_id = repo_id
            
            # Load episode tasks
            self._load_episode_tasks(dataset)
            
            # Load approvals
            self._load_approvals()
            
            meta = dataset.meta
            return {
                "success": True,
                "repo_id": repo_id,
                "total_episodes": meta.total_episodes,
                "total_frames": meta.total_frames,
                "fps": meta.fps,
                "robot_type": getattr(meta, 'robot_type', 'unknown'),
                "features": list(meta.features.keys()),
                "task": self.episode_tasks.get(0, ""),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _load_episode_tasks(self, dataset):
        """Load task descriptions from episodes parquet."""
        import pandas as pd
        import pyarrow.parquet as pq
        
        self.episode_tasks.clear()
        
        try:
            dataset_root = Path(dataset.root)
            episodes_dir = dataset_root / "meta" / "episodes"
            
            if not episodes_dir.exists():
                return
            
            chunk_dir = episodes_dir / "chunk-000"
            if chunk_dir.exists() and chunk_dir.is_dir():
                all_dfs = []
                for parquet_file in sorted(chunk_dir.glob("*.parquet")):
                    try:
                        table = pq.read_table(parquet_file)
                        df = table.to_pandas()
                        all_dfs.append(df)
                    except Exception:
                        pass
                
                if all_dfs:
                    combined_df = pd.concat(all_dfs, ignore_index=True)
                    
                    if 'tasks' in combined_df.columns and 'episode_index' in combined_df.columns:
                        for _, row in combined_df.iterrows():
                            ep_idx = int(row['episode_index'])
                            tasks = row['tasks']
                            
                            if isinstance(tasks, (list, np.ndarray)) and len(tasks) > 0:
                                self.episode_tasks[ep_idx] = str(tasks[0])
                            elif isinstance(tasks, str):
                                self.episode_tasks[ep_idx] = tasks
        except Exception as e:
            print(f"Warning: Could not load episode tasks: {e}")
    
    def get_episodes(self) -> List[Dict]:
        """Get list of episodes with their status."""
        if not self.processor.dataset:
            return []
        
        episodes = []
        for i in range(self.processor.dataset.meta.total_episodes):
            status = "normal"
            if i in self.approved_episodes:
                status = "approved"
            elif i in self.commented_episodes:
                status = "commented"
            
            episodes.append({
                "index": i,
                "status": status,
                "comment": self.commented_episodes.get(i, ""),
                "task": self.episode_tasks.get(i, ""),
            })
        
        return episodes
    
    def get_episode_range(self, episode_idx: int) -> Dict:
        """Get frame range for an episode."""
        if not self.processor.dataset:
            return {"start": 0, "end": 0}
        
        start, end = self.processor.get_episode_range(episode_idx)
        return {"start": start, "end": end - 1}
    
    def get_frame_image(self, frame_idx: int, camera_key: Optional[str] = None) -> Optional[bytes]:
        """Get a frame image as JPEG bytes with caching and error handling."""
        if not self.processor.dataset:
            return None
        
        # Create cache key
        cache_key = (frame_idx, camera_key)
        
        # Check cache first
        cached = self.frame_cache.get(cache_key)
        if cached is not None:
            return cached
        
        # Fetch with retry logic
        max_retries = 3
        retry_delay = 0.1  # 100ms
        
        for attempt in range(max_retries):
            try:
                # Use lock to serialize video decoder access
                with self.frame_lock:
                    data = self.processor.get_frame(frame_idx)
                
                img_keys = [k for k in data.keys() if 'image' in k]
                
                if not img_keys:
                    return None
                
                key = camera_key if camera_key and camera_key in img_keys else img_keys[0]
                img_data = data[key]
                
                # Convert to numpy
                if hasattr(img_data, 'numpy'):
                    img_np = img_data.numpy()
                    if img_np.ndim == 3 and img_np.shape[0] in [1, 3, 4]:
                        img_np = np.transpose(img_np, (1, 2, 0))
                    if img_np.dtype == np.float32 or img_np.dtype == np.float64:
                        if img_np.max() <= 1.0:
                            img_np = (img_np * 255).astype(np.uint8)
                        else:
                            img_np = img_np.astype(np.uint8)
                else:
                    img_np = np.array(img_data)
                
                # Convert to PIL and then to bytes
                if img_np.ndim == 2:
                    img_np = np.stack([img_np] * 3, axis=-1)
                
                pil_img = Image.fromarray(img_np)
                buffer = io.BytesIO()
                pil_img.save(buffer, format='JPEG', quality=85)
                buffer.seek(0)
                result = buffer.getvalue()
                
                # Cache the result
                self.frame_cache.put(cache_key, result)
                
                return result
                
            except Exception as e:
                error_msg = str(e)
                if "decoder" in error_msg.lower() or "packet" in error_msg.lower():
                    # Video decoder error - try to recover
                    if attempt < max_retries - 1:
                        print(f"Decoder error (attempt {attempt + 1}/{max_retries}): {e}, retrying...")
                        time.sleep(retry_delay * (attempt + 1))
                        
                        # Try to reset by reloading the dataset on last retry
                        if attempt == max_retries - 2 and self.current_repo_id:
                            print("Attempting to reload dataset to reset decoder...")
                            try:
                                with self.frame_lock:
                                    self.processor.load_dataset(self.current_repo_id)
                            except Exception as reload_error:
                                print(f"Could not reload dataset: {reload_error}")
                    else:
                        print(f"Failed after {max_retries} retries: {e}")
                        return None
                else:
                    print(f"Error getting frame image: {e}")
                    return None
        
        return None
    
    def get_episode_data(self, episode_idx: int) -> Dict:
        """Get episode data for plotting."""
        if not self.processor.dataset:
            return {}
        
        target_keys = ['observation.state', 'action']
        keys = [k for k in target_keys if k in self.processor.dataset.meta.features]
        
        try:
            data = self.processor.get_episode_data(episode_idx, keys)
            result = {}
            
            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    result[k] = v.tolist()
                else:
                    result[k] = v
            
            # Get dimension names
            dim_names = {}
            for k in keys:
                feature_info = self.processor.dataset.meta.features.get(k, {})
                dim_names[k] = feature_info.get('names', [])
            
            # Check if observation.state has 16 dimensions (14 position + 2 torque)
            has_torque = False
            torque_names = []
            if 'observation.state' in result and 'observation.state' in dim_names:
                obs_state_data = result['observation.state']
                obs_state_names = dim_names['observation.state']
                
                # Check if we have 16 dimensions
                if obs_state_data and len(obs_state_data[0]) == 16:
                    has_torque = True
                    # Extract torque data (last 2 dimensions)
                    result['torque'] = [[frame[14], frame[15]] for frame in obs_state_data]
                    torque_names = obs_state_names[14:16] if len(obs_state_names) >= 16 else ['left_torque', 'right_torque']
                    dim_names['torque'] = torque_names
                    
                    # Keep only first 14 dimensions for observation.state
                    result['observation.state'] = [frame[:14] for frame in obs_state_data]
                    dim_names['observation.state'] = obs_state_names[:14] if len(obs_state_names) >= 14 else obs_state_names
            
            return {"data": result, "dim_names": dim_names, "has_torque": has_torque}
        except Exception as e:
            print(f"Error getting episode data: {e}")
            return {}
    
    def get_camera_keys(self) -> List[str]:
        """Get list of camera keys."""
        if not self.processor.dataset:
            return []
        
        sample = self.processor.dataset[0]
        return sorted([k for k in sample.keys() if 'image' in k])
    
    def _get_approvals_file_path(self) -> Optional[Path]:
        """Get the path to the approvals JSON file."""
        if not self.processor.dataset:
            return None
        dataset_root = Path(self.processor.dataset.root)
        return dataset_root / "episode_approvals.json"
    
    def _load_approvals(self):
        """Load episode approvals from file."""
        self.approved_episodes.clear()
        self.commented_episodes.clear()
        self.approvals_file_path = self._get_approvals_file_path()
        
        if self.approvals_file_path and self.approvals_file_path.exists():
            try:
                with open(self.approvals_file_path, 'r') as f:
                    data = json.load(f)
                    self.approved_episodes = set(data.get('approved_episodes', []))
                    comments_data = data.get('commented_episodes', {})
                    self.commented_episodes = {int(k): v for k, v in comments_data.items()}
            except Exception as e:
                print(f"Warning: Could not load approvals: {e}")
    
    def _save_approvals(self):
        """Save episode approvals to file."""
        import shutil
        
        if not self.approvals_file_path:
            self.approvals_file_path = self._get_approvals_file_path()
        
        if self.approvals_file_path:
            try:
                data = {
                    'approved_episodes': sorted(list(self.approved_episodes)),
                    'total_approved': len(self.approved_episodes),
                    'commented_episodes': {str(k): v for k, v in self.commented_episodes.items()},
                    'total_commented': len(self.commented_episodes)
                }
                
                with open(self.approvals_file_path, 'w') as f:
                    json.dump(data, f, indent=2)
                
                # Backup
                APPROVALS_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
                if self.current_repo_id:
                    safe_name = self.current_repo_id.replace('/', '_')
                    backup_path = APPROVALS_BACKUP_DIR / f"{safe_name}_approvals.json"
                    shutil.copy2(self.approvals_file_path, backup_path)
                    
            except Exception as e:
                print(f"Warning: Could not save approvals: {e}")
    
    def approve_episode(self, episode_idx: int) -> Dict:
        """Toggle approval status for an episode."""
        if episode_idx in self.approved_episodes:
            self.approved_episodes.remove(episode_idx)
            status = "unapproved"
        else:
            self.approved_episodes.add(episode_idx)
            if episode_idx in self.commented_episodes:
                del self.commented_episodes[episode_idx]
            status = "approved"
        
        self._save_approvals()
        return {"success": True, "status": status}
    
    def comment_episode(self, episode_idx: int, comment: str) -> Dict:
        """Add a comment to an episode."""
        if comment.strip():
            self.commented_episodes[episode_idx] = comment.strip()
            if episode_idx in self.approved_episodes:
                self.approved_episodes.remove(episode_idx)
            status = "commented"
        else:
            if episode_idx in self.commented_episodes:
                del self.commented_episodes[episode_idx]
            status = "normal"
        
        self._save_approvals()
        return {"success": True, "status": status}


def create_web_app(viewer: WebDatasetViewer) -> Flask:
    """Create the Flask application."""
    app = Flask(__name__, 
                template_folder=str(PROJECT_DIR / "templates"),
                static_folder=str(PROJECT_DIR / "static"))
    
    @app.route('/')
    def index():
        return render_template('index.html')
    
    @app.route('/api/datasets')
    def list_datasets():
        return jsonify(viewer.list_datasets())
    
    @app.route('/api/load', methods=['POST'])
    def load_dataset():
        data = request.get_json()
        repo_id = data.get('repo_id', '')
        return jsonify(viewer.load_dataset(repo_id))
    
    @app.route('/api/episodes')
    def get_episodes():
        return jsonify(viewer.get_episodes())
    
    @app.route('/api/episode/<int:idx>/range')
    def get_episode_range(idx):
        return jsonify(viewer.get_episode_range(idx))
    
    @app.route('/api/episode/<int:idx>/data')
    def get_episode_data(idx):
        return jsonify(viewer.get_episode_data(idx))
    
    @app.route('/api/frame/<int:idx>')
    def get_frame(idx):
        camera = request.args.get('camera')
        img_bytes = viewer.get_frame_image(idx, camera)
        if img_bytes:
            return send_file(
                io.BytesIO(img_bytes),
                mimetype='image/jpeg'
            )
        return '', 404
    
    @app.route('/api/cameras')
    def get_cameras():
        return jsonify(viewer.get_camera_keys())
    
    @app.route('/api/approve/<int:idx>', methods=['POST'])
    def approve_episode(idx):
        return jsonify(viewer.approve_episode(idx))
    
    @app.route('/api/comment/<int:idx>', methods=['POST'])
    def comment_episode(idx):
        data = request.get_json()
        comment = data.get('comment', '')
        return jsonify(viewer.comment_episode(idx, comment))
    
    return app


def run_web_server(port: int = 3000):
    """Run the web server."""
    viewer = WebDatasetViewer()
    app = create_web_app(viewer)
    
    print(f"\n{'='*50}")
    print(f"  LeRobot Dataset Visualizer - Web UI")
    print(f"{'='*50}")
    print(f"  Open in browser: http://localhost:{port}")
    print(f"{'='*50}\n")
    
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)

