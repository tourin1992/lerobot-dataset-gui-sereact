# LeRobot Dataset GUI

[English](README_EN.md) | 中文

一个基于 PySide6 的图形化界面工具，用于查看和编辑 LeRobot 格式的数据集。

## 功能特性

### 数据查看
- 支持通过 `repo_id` 从 Hugging Face Hub 或本地加载数据集
- Episode 导航栏，支持键盘快捷键切换（W/S 或 ↑/↓）
- 图像实时预览（支持 PIL 和 torch tensor 格式）
- State/Action 等向量数据的多维度折线图展示
- 维度选择器，可自由开启/关闭特定维度的显示
- 进度条拖动切换帧（A/D 或 ←/→ 键盘控制）
- 同步的时间轴标尺，所有图表联动显示当前帧位置
- 异步数据加载，避免界面卡顿

### 数据编辑
- **全局编辑**（对所有 episode 生效）
  - 删除 episode
  - 删除指定 feature
  - 保存到新数据集
  - ⏳ 增加指定 feature（计划中）
- **局部编辑**（对特定 episode）
  - 删除指定 frames（剪切）
  - 修改指定 frame 的 feature

## 安装与运行

### 环境要求
- Python 3.10+
- PySide6
- LeRobot 0.4.2

### 安装步骤

1. 克隆仓库：
   ```bash
   git clone https://github.com/shlyang/lerobot-dataset-gui.git
   cd lerobot_dataset_processor
   ```

2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

3. 启动应用：
   ```bash
   python app.py
   ```

## 使用说明

### 键盘快捷键
- `↑` / `↓`: 切换 Episode
- `←` / `→`: 在当前 Episode 内切换 Frame

### 基本操作
1. 在顶部输入框输入 `repo_id`（如 `lerobot/pusht`）或本地路径
2. 点击 `Load Dataset` 加载数据集
3. 左侧列表显示所有 Episode，点击或使用键盘切换
4. 中间显示当前帧的图像
5. 右侧显示 State/Action 的折线图，使用复选框选择要显示的维度
6. 拖动进度条或使用键盘浏览时间轴

### 编辑功能
- 右键点击 Episode 列表项可删除 episode
- 在编辑面板中可删除 feature、剪切 frames、修改 frame 数据
- 所有编辑操作会记录为任务，点击保存按钮统一应用到新数据集

## 项目结构
```
lerobot_dataset_processor/
├── app.py              # GUI 界面实现（PySide6 + PyQtGraph）
├── processor.py        # 数据集加载与处理逻辑
├── requirements.txt    # 项目依赖
└── README.md          # 中文文档
```

## 技术栈
- **GUI 框架**: PySide6
- **数据可视化**: PyQtGraph
- **数据处理**: LeRobot, NumPy, PyTorch
- **图像处理**: Pillow

## 许可证
详见 [LICENSE](LICENSE) 文件

## 贡献
欢迎提交 Issue 和 Pull Request！

