# Fly0：面向空中视觉语言导航的语义感知与几何规划框架

Fly0 将视觉语言模型的语义目标定位与底层几何路径规划解耦。模型负责从 RGB 图像和自然语言指令中识别目标，系统再结合深度信息进行三维反投影，并由规划器生成可执行、避障的飞行轨迹。

## 当前验证环境

| 组件 | 已验证版本 |
|---|---|
| 操作系统 | Windows 11 Pro 10.0.22631 |
| Unreal Engine | 4.27.2 |
| AirSim Python 客户端 | 1.8.1 |
| Python | 3.8.7，64 位 |
| Ollama | 0.33.2 |
| 视觉语言模型 | qwen3-vl:4b |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| 显存 | 8188 MiB |
| NVIDIA 驱动 | 591.86 |
| NumPy | 1.24.4 |
| SciPy | 1.10.1 |
| OpenCV Contrib | 4.7.0.68 |
| Pillow | 10.4.0 |
| Requests | 2.32.4 |
| msgpack-python | 0.5.6 |
| msgpack-rpc-python | 0.4.1 |
| Tornado | 4.5.3 |

完整环境记录见 [`docs/tested_environment.md`](docs/tested_environment.md)。除上述版本外，其他版本尚未验证为行为等价。

## 项目结构

```text
src/airsim/
├─ main.py                    程序入口
├─ config.json                运行配置
├─ run.bat                    Windows 启动脚本
├─ settings.json              AirSim 相机和 LiDAR 配置
├─ core/control/              无人机控制与 EgoPlanner
├─ core/detection/            视觉目标检测与三维定位
├─ core/navigation/           目标导航流程
├─ core/processing/           自然语言命令处理
└─ sysprompt/                 模型系统提示词
```

## 坐标约定

- Fly0 对外使用 `[x, y, height]`，height 越大表示越高。
- AirSim 内部使用 NED 坐标 `[x, y, z]`，z 越小表示越高。
- UE4 使用厘米单位，Actor 的 Relative Location 不能直接当作世界绝对坐标。

## 安装

### 前置条件

- Windows 11（当前验证版本为 10.0.22631）
- Python 3.8.7，64 位
- Unreal Engine 4.27.2 和 AirSim Multirotor 场景
- Git
- Ollama 0.33.2
- 推荐使用具有至少 8 GB 显存的 NVIDIA GPU

### 1. 克隆仓库

```powershell
git clone https://github.com/Jst599/Fly0.git
cd Fly0
```

### 2. 创建虚拟环境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. 安装 Python 依赖

```powershell
python -m pip install -r env/requirements.txt
python -m pip install airsim==1.8.1 --no-build-isolation
```

AirSim Python 客户端和 UE4 仿真器是两个独立组件。必须将 AirSim Python 客户端安装到实际运行 Fly0 的同一个 Python 解释器中。

### 4. 安装 Ollama 和模型

安装 Ollama 后，在终端执行：

```powershell
ollama serve
ollama pull qwen3-vl:4b
ollama list
```

当前验证配置：Ollama 地址为 `http://127.0.0.1:11434`，模型为 `qwen3-vl:4b`，模型大小约 3.3 GB。

### 5. 准备 UE4/AirSim

按照 [AirSim Windows 安装指南](https://microsoft.github.io/AirSim/build/windows/) 安装 AirSim，并启动 UE4 场景。使用仓库中的 `src/airsim/settings.json` 配置相机和 LiDAR。

运行时必须能够访问：

```text
AirSim RPC：127.0.0.1:41451
```

## 配置

默认配置文件为 `src/airsim/config.json`，当前验证的关键内容为：

```json
{
  "API_TYPE": "ollama",
  "OLLAMA_MODEL": "qwen3-vl:4b",
  "OLLAMA_CTRL_MODEL": "qwen3-vl:4b",
  "vision": {"enabled": true},
  "planner": {"lidar_sensors": ["LidarSensor1"]}
}
```

如需使用其他模型或推理服务，请修改配置，并在实验记录中注明模型、服务地址和版本。

## 启动

启动顺序：

1. 启动 UE4 AirSim 场景；
2. 确认 Ollama 正在运行且已安装 `qwen3-vl:4b`；
3. 启动 Fly0。

推荐使用：

```powershell
cd I:\Fly0-main
$env:FLY0_PYTHON = "C:\path\to\python.exe"
.\src\airsim\run.bat
```

`run.bat` 的 Python 选择顺序为：`FLY0_PYTHON`、项目 `.venv\Scripts\python.exe`、系统 `PATH` 中的 Python。所选解释器必须能够导入 `airsim`、`numpy`、`scipy`、`cv2` 和 `requests`。

也可以直接运行：

```powershell
cd src/airsim
python .\main.py
```

## 交互命令

程序启动后进入飞行模式：

```text
!quit   退出程序
!clear  清除对话历史
!help   显示帮助
```

示例：

```text
起飞到5米高度
飞到[10,8,5]
飞到红色球体上方
```

## AirSim 版本验证

```powershell
I:\fly0_py38\Scripts\python.exe -c "import airsim; print(airsim.__version__); print(airsim.__file__)"
```

当前验证输出为 `1.8.1` 和 `I:\fly0_py38\lib\site-packages\airsim\__init__.py`。如果 `python -c "import airsim"` 失败，请确认使用的是 Fly0 的 Python 解释器，而不是系统默认 Python。

## 安全与实验记录

- 真实碰撞后系统会取消当前路径并执行紧急上升。
- 规划器内部使用 AirSim NED 坐标。
- 运行时生成的检测图片、日志和模型权重不应提交到 Git。
- 实验结果应记录场景、模型、随机种子、动作空间和完整软件环境。

## 许可证

详见 [`LICENSE`](LICENSE)。
