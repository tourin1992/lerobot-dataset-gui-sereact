# LeRobot Dataset GUI

A visualization and annotation tool for LeRobot format datasets, featuring both a desktop GUI (PySide6) and a modern web interface (Flask + Plotly).

## Features

### Web Interface (Recommended)

The web interface provides a modern, feature-rich experience for dataset visualization and annotation:

**Video & Camera Visualization**
- Multi-camera grid display with automatic layout
- Real-time frame navigation with slider
- Playback controls (1x, 2x speed)
- Keyboard shortcuts for navigation

**Interactive Plots**
- Action, Observation State, and Torque plots
- Vertical time indicator line synchronized with video
- Resizable plot area (drag the divider)
- Individual plot resizing (drag handle at bottom of each plot)
- Joint visibility tree with Left/Right grouping
- Color-coded joints (black, green, yellow, red, blue, orange, violet)

**Episode Management**
- Episode list with status indicators
- Approval system (mark episodes as approved)
- Comment system for annotations
- Task descriptions from dataset metadata

### Desktop Interface

The desktop GUI provides similar functionality using PySide6:
- Episode navigation with keyboard shortcuts
- Real-time image preview
- Multi-dimensional line charts for State/Action data
- Dimension selector for toggling specific dimensions
- Synchronized timeline across all charts

### Data Editing (Desktop Only)
- **Global Editing** (applies to all episodes)
  - Delete episodes
  - Remove specified features
  - Save to new dataset
- **Local Editing** (for specific episodes)
  - Delete specified frames (trim)
  - Modify features for specific frames

## Installation

### Requirements
- Python 3.10+
- FFmpeg (required for video decoding)

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/lerobot-dataset-gui.git
   cd lerobot-dataset-gui
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or: venv\Scripts\activate  # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install FFmpeg (required for video playback):
   ```bash
   # Ubuntu/Debian
   sudo apt install ffmpeg
   
   # macOS
   brew install ffmpeg
   
   # Or via conda
   conda install -c conda-forge ffmpeg
   ```

## Usage

### Web Interface (Recommended)

Start the web server:
```bash
python web_app.py
```

Then open your browser to `http://localhost:5000`

**Keyboard Shortcuts:**
- `W` / `↑`: Previous episode
- `S` / `↓`: Next episode
- `A` / `←`: Previous frame
- `D` / `→`: Next frame
- `Space`: Toggle playback (1x)

### Desktop Interface

Launch the desktop application:
```bash
python app.py
```

**Keyboard Shortcuts:**
- `↑` / `↓`: Switch between episodes
- `←` / `→`: Navigate frames within current episode

## Project Structure

```
lerobot-dataset-gui/
├── app.py              # Desktop GUI (PySide6 + PyQtGraph)
├── web_app.py          # Web server (Flask)
├── processor.py        # Dataset loading and processing logic
├── templates/
│   └── index.html      # Web interface (HTML/CSS/JS + Plotly)
├── requirements.txt    # Python dependencies
├── LICENSE             # License file
└── README.md           # This file
```

## Tech Stack

- **Desktop GUI**: PySide6, PyQtGraph
- **Web Interface**: Flask, Plotly.js
- **Data Processing**: LeRobot, NumPy, PyTorch, Pandas
- **Image Processing**: Pillow
- **Video Decoding**: FFmpeg, TorchCodec

## Troubleshooting

### FFmpeg not found error
If you see an error about `libavutil.so` or similar FFmpeg libraries:
```
RuntimeError: Could not load libtorchcodec...
```
Install FFmpeg using the commands in the Installation section above.

### Video playback issues
- Ensure FFmpeg is properly installed: `ffmpeg -version`
- Check that the dataset contains valid video files
- Try restarting the web server after installing FFmpeg

## License

See [LICENSE](LICENSE) file for details.

## Contributing

Issues and Pull Requests are welcome!
