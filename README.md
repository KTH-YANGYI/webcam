# YOLO Webcam App

这个子项目用来把本地 YOLO `.pt` 权重接到摄像头输入上，做实时检测。

## 1. 进入目录

```powershell
cd webcam_app
```

## 2. 安装环境

建议使用 Python 3.12。

```powershell
conda create -n yolo-webcam python=3.12 -y
conda activate yolo-webcam
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

如果你不用 NVIDIA GPU，就把 PyTorch 安装成适合自己环境的版本。

## 3. 配置模型路径

默认配置文件是 [configs/default.yaml](configs/default.yaml)。

把里面的 `model` 改成你自己的权重路径，例如：

```yaml
model: path/to/your/best.pt
```

也可以运行时直接覆盖：

```powershell
python scripts/run_webcam.py --model path\to\your\best.pt
```

## 4. 先检查环境

```powershell
python scripts/env_check.py
```

如果这里过不去，先不要继续跑摄像头。

## 5. 探测摄像头索引

```powershell
python scripts/camera_probe.py --max-index 5 --write-json
```

如果想直接看预览：

```powershell
python scripts/camera_probe.py --max-index 5 --preview --write-json
```

Windows 下通常优先试：

```text
dshow -> msmf -> auto
```

## 6. 运行检测

默认运行：

```powershell
python scripts/run_webcam.py
```

指定摄像头索引和后端：

```powershell
python scripts/run_webcam.py --source 1 --backend dshow
```

保存视频和标签：

```powershell
python scripts/run_webcam.py --source 1 --backend dshow --save --save-txt --name external_cam_run
```

如果想提一点速度：

```powershell
python scripts/run_webcam.py --source 1 --backend dshow --imgsz 512
```

预览窗口里按 `q` 退出。

## 7. 最短流程

```powershell
cd webcam_app
conda activate yolo-webcam
python scripts/env_check.py
python scripts/camera_probe.py --max-index 5 --write-json
python scripts/run_webcam.py --source 1 --backend dshow
```

把最后一条命令里的 `1` 换成你实际探测到的摄像头索引。
