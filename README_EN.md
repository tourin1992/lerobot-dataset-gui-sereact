# LeRobot Dataset GUI

English | [中文](README.md)

A PySide6-based graphical interface tool for viewing and editing LeRobot format datasets.

## Features

### Data Viewing
- Load datasets from Hugging Face Hub or local paths via `repo_id`
- Episode navigation bar with keyboard shortcuts (↑/↓)
- Real-time image preview (supports PIL and torch tensor formats)
- Multi-dimensional line charts for State/Action vector data
- Dimension selector to toggle specific dimensions on/off
- Progress bar for frame navigation (←/→ keyboard control)
- Synchronized timeline ruler with linked charts showing current frame position
- Asynchronous data loading to prevent UI freezing

### Data Editing
- **Global Editing** (applies to all episodes)
  - Delete episodes
  - Remove specified features
  - Save to new dataset
  - ⏳ Add specified features (planned)
- **Local Editing** (for specific episodes)
  - Delete specified frames (trim)
  - Modify features for specific frames

## Installation & Usage

### Requirements
- Python 3.10+
- PySide6
- LeRobot 0.4.2

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/shlyang/lerobot-dataset-gui.git
   cd lerobot-dataset-gui
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the application:
   ```bash
   python app.py
   ```

## Usage Guide

### Keyboard Shortcuts
- `↑` / `↓`: Switch between Episodes
- `←` / `→`: Navigate frames within current Episode

### Basic Operations
1. Enter `repo_id` (e.g., `lerobot/pusht`) or local path in the top input field
2. Click `Load Dataset` to load the dataset
3. Left panel displays all Episodes - click or use keyboard to switch
4. Center panel shows the current frame's image
5. Right panel displays State/Action line charts - use checkboxes to select dimensions
6. Drag the progress bar or use keyboard to navigate the timeline

### Editing Features
- Right-click Episode list items to delete episodes
- Use the edit panel to remove features, trim frames, or modify frame data
- All editing operations are recorded as tasks and applied together when saving to a new dataset

## Project Structure
```
lerobot-dataset-gui/
├── app.py              # GUI implementation (PySide6 + PyQtGraph)
├── processor.py        # Dataset loading and processing logic
├── requirements.txt    # Project dependencies
└── README_EN.md       # English documentation
```

## Tech Stack
- **GUI Framework**: PySide6
- **Data Visualization**: PyQtGraph
- **Data Processing**: LeRobot, NumPy, PyTorch
- **Image Processing**: Pillow

## License
See [LICENSE](LICENSE) file for details

## Contributing
Issues and Pull Requests are welcome!

