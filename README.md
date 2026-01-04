# LeRobot Dataset Processor GUI

一个基于 PySide6 的图形化界面工具，用于查看和编辑 LeRobot 格式的数据集。

## 功能特性
- 支持通过 `repo_id` 从 Hugging Face Hub 或本地加载数据集
- Episode 导航栏，支持键盘快捷键切换（W/S 或 ↑/↓）
- 图像实时预览（支持 PIL 和 torch tensor 格式）
- State/Action 等向量数据的多维度折线图展示
- 维度选择器，可自由开启/关闭特定维度的显示
- 进度条拖动切换帧（A/D 或 ←/→ 键盘控制）
- 同步的时间轴标尺，所有图表联动显示当前帧位置
- 异步数据加载，避免界面卡顿

## 安装与运行

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. 启动应用：
   ```bash
   python app.py
   ```

## 使用说明

### 键盘快捷键
- `W` / `S` 或 `↑` / `↓`: 切换 Episode
- `A` / `D` 或 `←` / `→`: 在当前 Episode 内切换 Frame

### 基本操作
1. 在顶部输入框输入 `repo_id`（如 `lerobot/pusht`）
2. 点击 `Load Dataset` 加载数据集
3. 左侧列表显示所有 Episode，点击或使用键盘切换
4. 中间显示当前帧的图像
5. 右侧显示 State/Action 的折线图，使用复选框选择要显示的维度
6. 拖动进度条或使用键盘浏览时间轴

## 项目结构
- `app.py`: GUI 界面实现（PySide6 + PyQtGraph）
- `processor.py`: 数据集加载与处理逻辑
- `requirements.txt`: 项目依赖

## TODO
- [ ] 增加数据集编辑功能
   - 全局（对所有episode生效）
   - [x] 删除episode
   - [x] 保存到新数据集
   - [x] 删除指定feature
   - [ ] 增加指定feature
   - 局部（对特定episode）
   - [ ] 删除指定frames(剪切？)
   - [ ] 修改指定frame的feature
- [ ] 增加英文版readme

