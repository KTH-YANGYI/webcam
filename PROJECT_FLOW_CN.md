# 项目流程与逻辑说明

## 1. 这个项目是干什么的

这个 `webcam_app` 子项目的目标很明确：

- 使用现有训练好的 YOLO 权重做推理
- 从本机或外接 USB UVC 摄像头读取实时画面
- 在画面上绘制检测框
- 按需要保存视频、标签和日志

当前默认模型是：

`../yolo11n_train_v3_datachanged_2/weights/best.pt`

当前默认任务是缺陷检测，模型类别名在本机验证时读到的是：

- `crack`
- `broken`

## 2. 项目结构和每个文件的职责

### `common.py`

这是公共工具模块，负责几件基础但关键的事情：

- 定义项目根目录 `PROJECT_ROOT`
- 定义默认配置路径和输出目录
- 统一解析相对路径
- 统一管理摄像头后端映射
- 创建输出目录
- 生成时间戳和安全文件名

这个文件的作用是把重复逻辑集中起来，避免三个脚本各写一份。

### `configs/default.yaml`

这是默认运行配置。它定义了：

- 默认模型路径
- 默认设备 `device`
- 默认图像尺寸 `imgsz`
- 默认置信度阈值 `conf`
- 默认摄像头索引 `source`
- 默认后端 `backend`
- 默认分辨率和 FPS
- 是否保存视频和标签

这个文件不是死配置。运行脚本时，命令行参数可以覆盖它。

### `scripts/env_check.py`

这是环境检查脚本，不做正式推理，只做“开机体检”。

它的逻辑是：

1. 读取配置文件或命令行传入的模型路径
2. 打印 Python、OpenCV、Torch、CUDA、Ultralytics 版本
3. 调用 `ultralytics.checks()`
4. 尝试加载默认模型
5. 打印模型类别名

这个脚本的目标是尽早发现以下问题：

- Python 环境不对
- GPU 没被 Torch 识别
- 模型路径错误
- 权重不能被 Ultralytics 正常加载

### `scripts/camera_probe.py`

这是摄像头探测脚本，不做 YOLO 推理，只负责确认“哪个摄像头能用”。

它的逻辑是：

1. 遍历摄像头索引，默认是 `0..5`
2. 对每个索引按后端顺序尝试打开
3. 每个组合读取 3 帧
4. 记录是否打开成功、读取是否成功、分辨率、FPS
5. 可选写入 JSON
6. 可选打开预览窗口

Windows 下默认后端顺序是：

- `dshow`
- `msmf`
- `auto`

这个脚本解决的是“系统里到底哪个索引对应外接摄像头”这个问题。

### `scripts/run_webcam.py`

这是正式运行脚本，也是整个项目真正执行检测的入口。

它的逻辑是：

1. 读取 `default.yaml`
2. 用命令行参数覆盖默认配置
3. 把相对路径统一解析成绝对路径
4. 创建日志目录、标签目录、视频目录
5. 检查模型文件是否存在
6. 如果用户要求用 GPU，但 CUDA 不可用，则自动回退到 CPU
7. 加载 YOLO 模型
8. 打开摄像头
9. 逐帧读取图像
10. 每一帧调用 `model(frame, ...)`
11. 用 `results[0].plot()` 生成带框画面
12. 在窗口中显示结果
13. 如果开启保存，就写入视频和标签
14. 按 `q` 退出并释放资源

## 3. 配置是怎么生效的

项目的配置优先级是这样的：

1. 代码里的内置默认值 `DEFAULTS`
2. `configs/default.yaml`
3. 命令行参数

也就是说，最终运行配置不是只看 YAML，而是“默认值 + YAML + CLI 覆盖”的结果。

例如：

```powershell
python scripts/run_webcam.py --source 1 --backend dshow --save
```

这条命令会保留 YAML 里大部分配置，只把：

- `source`
- `backend`
- `save`

改成命令行指定的值。

## 4. 路径是怎么处理的

这个项目有一个重要约定：

所有相对路径都以 `webcam_app/` 根目录为基准，而不是当前命令所在目录，也不是 `configs/` 目录。

例如：

```yaml
model: ../yolo11n_train_v3_datachanged_2/weights/best.pt
```

这个路径最终会被解析成相对于 `webcam_app/` 的绝对路径。

这样做的好处是：

- 配置文件位置不会影响模型路径解释
- 从脚本目录运行和从项目根目录运行都不会错位
- README 中的命令更稳定

## 5. 整个项目的实际运行流程

推荐你按下面顺序理解和使用这个项目：

### 阶段 A：确认环境正确

先运行：

```powershell
python scripts/env_check.py
```

这里验证的是：

- Python 是否正确
- Torch 是否能识别 CUDA
- Ultralytics 是否安装正常
- 默认模型是否能加载

如果这一关不过，后面的摄像头和推理都不应该继续。

### 阶段 B：确认摄像头链路正确

插入摄像头后运行：

```powershell
python scripts/camera_probe.py --max-index 5 --write-json outputs\logs\camera_probe_after_usb.json
```

这里验证的是：

- Windows 是否识别到摄像头
- 哪个索引能打开
- 哪个后端最稳定

如果这一关不过，说明问题在摄像头链路，不在模型。

### 阶段 C：用 Ultralytics CLI 做冒烟测试

例如：

```powershell
yolo detect predict model=../yolo11n_train_v3_datachanged_2/weights/best.pt source=1 show=True device=0 conf=0.25 imgsz=640
```

这里的目的不是长期使用 CLI，而是最快确认“模型 + GPU + 摄像头”能不能一起工作。

### 阶段 D：运行正式脚本

例如：

```powershell
python scripts/run_webcam.py --source 1 --backend dshow
```

这是正式使用路径。只有在前面几关都通过以后，这一步才稳定。

## 6. 运行时的数据流

正式脚本 `run_webcam.py` 的数据流可以理解成下面这样：

```text
default.yaml + CLI 参数
        ↓
  生成最终运行配置
        ↓
    加载 YOLO 模型
        ↓
    打开摄像头
        ↓
    逐帧读取 frame
        ↓
   YOLO 对 frame 做推理
        ↓
  results[0].plot() 画框
        ↓
  显示窗口 / 保存视频 / 保存标签
        ↓
        q 退出
```

其中最核心的一步就是：

- 输入：原始 `frame`
- 输出：带框的 `annotated frame`

## 7. 输出文件是怎么产生的

### 日志

运行正式脚本时，会创建：

- `outputs/logs/run_YYYYMMDD_HHMMSS.log`

日志里会记录：

- 最终配置
- Python 和库版本
- CUDA 信息
- 输出路径
- 运行过程中的错误或退出信息

### 视频

如果传了 `--save`，会生成：

- `outputs/videos/<name>.mp4`

### 标签

如果传了 `--save-txt`，会生成：

- `outputs/labels/<name>_<frame_index>.txt`

也就是每一帧一个标签文件。

## 8. 当前项目的能力边界

当前版本已经支持：

- 摄像头环境检查
- 摄像头索引探测
- Ultralytics 模型加载
- GPU 推理
- 实时显示
- 保存视频
- 保存逐帧标签
- 配置文件 + CLI 覆盖

当前版本还不支持或没有专门做的内容：

- 多摄像头同时推理
- 工业相机 SDK 接入
- ONNX / TensorRT 推理路径
- 图形化界面

其中一个明确限制是：

`run_webcam.py` 目前把 `--source` 定义成了整数，所以它当前只支持摄像头索引，不支持直接传 `mp4` 路径。
如果你要对本地视频做实验，当前需要单独扩展视频输入支持，而不是直接复用摄像头脚本。

## 9. 你实际使用时应该怎么理解这个项目

最简单的理解方式是：

- `env_check.py` 负责先排环境问题
- `camera_probe.py` 负责先排摄像头问题
- `run_webcam.py` 负责正式推理

也就是说，这个项目不是一个“上来就直接跑检测”的黑盒，而是故意拆成三层：

1. 环境层
2. 摄像头层
3. 推理层

这样做的好处是，出问题时你能快速定位到底是：

- 环境坏了
- 摄像头没接对
- 还是模型/推理有问题

## 10. 推荐的最短工作流

如果你只想按最实用的顺序用它，直接按这个流程走：

```powershell
cd C:\Users\18046\Desktop\master\masterthesis\yolo\runs_2633\webcam_app
conda activate yolo-webcam
python scripts/env_check.py
python scripts/camera_probe.py --max-index 5 --write-json outputs\logs\camera_probe_after_usb.json
python scripts/run_webcam.py --source 1 --backend dshow
```

其中最后一条里的 `1` 要换成你探测出来的真实摄像头索引。

如果后面你希望我继续扩展，这个项目最自然的下一步是两种：

1. 给 YOLO 路线增加本地视频文件输入支持
2. 把摄像头和视频输入统一成一个更通用的 `predict_source.py`
