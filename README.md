# Standard-AI-Development

> 这是一个 AI 项目脚手架提示词。下次构建新的 AI 项目时，把本文件作为系统提示/首条指令发给 Claude，并补充【待填项】中的内容，即可快速生成一个结构标准、可直接训练的项目骨架。

---

## 0. 使用方式

把下面这段话连同本文件内容一起发给 Claude：

```
请按照 Standard-AI-Development.md 的规范，帮我创建一个 AI 项目。
任务描述：<这里写你的任务，例如“点云关键点回归 / 图像分类 / 时序预测”>
数据格式：<原始数据放哪、label 长什么样>
评价指标：<我会单独告诉你，见第 6 节>
其他要求：<可选>
```

Claude 应在当前目录下生成完整的 `./data` + `./project` 目录结构，并保证每个文件可直接 `python xxx.py` 运行（在依赖已安装的前提下）。

---

## 1. 项目目录结构

```
.
├── data/                       # 数据区（只读，不写代码）
│   ├── label/                  # 标签：每个样本一个文件（txt/csv/json/ply 等）
│   └── data/                   # 原始数据：点云 / 图像 / 表格 等
└── project/                    # 程序区（所有代码在此）
    ├── config.py               # 全局配置（路径、超参、后处理参数），导出 config 单例
    ├── split.py                # 按 train/val/test 比例划分文件名 -> splits/*.txt
    ├── prepare_pkl.py          # 读取原始数据 + label，打包成 .pkl，供 Dataset 高速加载
    ├── net.py                  # 网络结构定义
    ├── dataset.py              # Dataset 类（从 .pkl 读取，含归一化、下采样、增强）
    ├── loss.py                 # 损失函数（可定义多个版本 V1/V2/V3 便于对比）
    ├── train.py                # 训练入口（tqdm 进度条、保存权重、写 metrics.json）
    ├── test.py                 # 测试入口（加载 best 权重，在 test set 上算指标 + 可视化）
    ├── inference.py            # 单样本/文件夹推理入口（推理 + 后处理 + 保存结果）
    ├── utils/                  # 通用工具函数（可视化、指标计算、IO、几何工具等）
    │   └── ...
    ├── splits/                 # split.py / prepare_pkl.py 生成（运行时产生）
    │   ├── train.txt  val.txt  test.txt
    │   └── train.pkl  val.pkl  test.pkl
    └── runs/                   # 训练输出（运行时产生，每次训练一个 run_xx 子目录）
        └── run_01/
            ├── best_loss.pth       # 验证集最优权重
            ├── epoch_010.pth       # 每 N 个 epoch 存一次的检查点
            └── metrics.json        # 逐 epoch 的 train/val 指标记录
```

说明：
- `./data` 与 `./project` 严格分离：数据只读，代码只写 `./project`。
- `splits/` 与 `runs/` 是运行产物，**不要**预创建内容，脚本里用 `os.makedirs(..., exist_ok=True)`。
- `config.py` 中的数据路径统一用相对 `project/` 的写法，例如 `self.DATA_DIR = "../data/data"`、`self.LABEL_DIR = "../data/label"`。

---

## 2. 各模块职责与约定

### config.py
- 用一个 `Config` 类，实例化后导出全局单例 `config = Config()`，其余脚本 `from config import config as cfg` 引用。
- 字段分组：路径、数据规模（`NUM_POINTS` 等）、训练超参（`BATCH_SIZE / LR / WEIGHT_DECAY / EPOCHS / LR_LIST / GAMMA / NUM_WORKERS`）、后处理参数。
- **多 GPU 由 config 控制**：`GPU_IDS = [0]` 单卡、`GPU_IDS = [0,1,2]` 多卡。训练是否走 `DataParallel` 完全由 `len(cfg.GPU_IDS) > 1` 决定，不要在 `train.py` 里硬编码卡号。
- 不要把实验性的临时参数硬编码进训练脚本，统一收进 `config.py`。

### split.py
- 按比例（默认 0.8 / 0.1 / 0.1）把样本文件名（不含路径、不含后缀）写入 `splits/train.txt`、`val.txt`、`test.txt`。
- 固定随机种子（`SEED=42`），保证可复现。
- `assert abs(TRAIN+VAL+TEST-1.0) < 1e-6`。

### prepare_pkl.py
- 读取 `splits/*.txt` 列出的样本，从 `data/` + `label/` 拼装成 `{特征, 标签, ...}` 的 dict 列表，用 `pickle.dump(..., protocol=HIGHEST_PROTOCOL)` 存为 `splits/{train,val,test}.pkl`。
- 这样训练时一次性载入内存，避免每个 batch 反复读盘。
- 容错：单个样本读取失败要打印警告并跳过，最后打印 `成功/失败` 计数。

### net.py
- 仅定义网络结构。命名导出主类（如 `Net`）。
- 输入/输出 shape 在 docstring 里写清楚。

### dataset.py
- 继承 `torch.utils.data.Dataset`，构造函数接受 `pkl_path, num_points=..., augment=False`。
- `__getitem__` 内完成：读取 → 下采样到 `num_points` → 归一化（中心化 + 缩放到单位球）→ 训练时做增强（旋转、噪声等）→ 返回 `(feat, label, meta)`。
- 归一化要把 `centroid` 和 `scale` 一并返回或可还原，便于推理时还原到世界坐标。

### loss.py
- 每个损失一个 `nn.Module` 子类，`forward(pred, gt)` 返回标量。
- 多版本用 `V1/V2/V3` 后缀区分，便于在 `train.py` 里切换对比。

### train.py
- 入口：`python train.py`，可选 `--resume_ckpt path/to/xxx.pth`。
- **支持多 GPU**：按 `cfg.GPU_IDS` 自动处理。`len(cfg.GPU_IDS) > 1` 时套 `nn.DataParallel(model, device_ids=cfg.GPU_IDS)`，并在日志里打印所用卡号；单卡时不套。保存权重前若是 `DataParallel` 要取 `.module.state_dict()`，保证 resume 时不依赖前缀（或保留剥离 `module.` 的兼容逻辑）。
- 流程：建 DataLoader → 建模型（按上条处理多卡）→ 优化器 + scheduler → 训练循环 → 每 epoch 在 val 上评估 → 写 `metrics.json` → 保存 `best_loss.pth` 与周期性 `epoch_NNN.pth` → 训练结束在 test 上最终评估。
- **进度条格式见第 4 节，必须严格遵守。**
- `metrics.json` 是 list of record，**记录每个 epoch 的训练/验证指标**（`epoch / train_loss / val_loss / 其他metric`），每 epoch 覆盖写整个文件（断点续训时据此恢复 `start_epoch` 和 `best_loss`）。测试期不再写这个文件。
- resume 时若权重带 `module.` 前缀（DataParallel 保存的），要自动剥离再 `load_state_dict`。

### test.py
- 入口：`python test.py --ckpt ./runs/run_01/best_loss.pth [--vis] [--save dir]`。
- 不指定 `--ckpt` 时，自动取 `runs/` 下最新 run 的 `best_loss.pth`。
- 在 test set 上算指标（指标定义见第 6 节），打印汇总；可选可视化/保存。

### inference.py
- 入口：`python inference.py path/to/file_or_dir --ckpt ... --save ...`。
- 对单文件或整个文件夹做推理，跑完后处理（如 NMS 取关键点），保存预测结果（txt/ply/png 等）。
- 自带可视化函数可放本文件，也可下沉到 `utils/`。

### utils/
- 跨脚本复用的小工具：指标计算、可视化（colormap、点云上色）、几何运算、IO 等。
- 每类工具一个文件，保持函数式、无副作用，方便被 train/test/inference 共同调用。

---

## 3. runs/ 组织

- 每次训练新建 `runs/run_XX/`，`XX` 自动递增（扫描已有 `run_*` 目录数 +1，两位补零）。
- 目录内固定产物：
  - `best_loss.pth` —— 验证指标最优时保存。
  - `epoch_NNN.pth` —— 每 10 个 epoch 存一个（周期可配）。
  - `metrics.json` —— 逐 epoch 指标记录，结构为 list。
- `test.py` 默认从 `runs/run_XX/best_loss.pth` 读权重。

---

## 4. 训练进度条格式（必须严格遵守）

训练循环用 `tqdm`，**单行**显示，格式如下：

```
[<current_epoch>/<total_epoch>] | lr=<lr> | <进度条> | <iter>/<total_iter> [已用<剩余, n s/it, loss=xxx, metric=xxx]
```

实现要点：
- `bar = tqdm(train_loader)`
- `bar.set_description(f"[{epoch+1}/{EPOCHS}] | lr={lr:.6f}")`
- 每个 iter 后 `bar.set_postfix(loss=f"{loss:.4f}", metric=f"{metric:.4f}")`
- tqdm 自动渲染 `<进度条>` 与 `[已用<剩余, n s/it, ...]` 部分，无需手写。
- `<metric>` 为当前 batch 累计/瞬时指标，具体名称与计算方式见第 6 节。

每个 epoch 结束后另起一行打印汇总（不进进度条）：

```
[<epoch>/<EPOCHS>] Train <train_loss> | Val <val_loss> | <其他指标>
```

---

## 5. 断点续训

- `--resume_ckpt` 指向某个 `run_XX/*.pth`，脚本据此：
  1. 自动定位同目录的 `metrics.json`，读出 `history`，`start_epoch = len(history)`，并从历史记录里恢复 `best_loss`。
  2. 新 epoch 继续追加到同一 `run_XX/metrics.json`，不新建 run 目录。
- 权重若带 `module.` 前缀需剥离后再加载。

---

## 6. 评价指标（待填）

> 构建项目时，**用户会单独提供**本项目的评价指标。Claude 在生成代码前应主动询问；若用户已给出，则填入下表并在 `test.py` / 训练评估函数中实现。

| 指标名 | 计算方式 | 用在哪 | 备注 |
|--------|----------|--------|------|
| _待填_ | _待填_   | _待填_ | _待填_ |

约定：
- 训练期打印的 `<metric>` 用**轻量、可逐 batch 累计**的指标（如 loss / accuracy），避免每个 iter 都跑昂贵评估。
- 完整指标在 `test.py` 的 test set 上一次性计算，结果同时打印并写入 `runs/run_XX/metrics.json`（或单独的 `test_metrics.json`）。

---

## 7. 代码风格约定

- 中文 docstring + 中文打印，注释说明“为什么”而非“是什么”。
- 所有路径用 `os.path.join`，不要字符串拼接斜杠。
- 设备：`device = "cuda" if torch.cuda.is_available() else "cpu"`，打印一次。
- 多卡：`len(cfg.GPU_IDS) > 1` 时套 `DataParallel`。
- 张量搬运用 `.to(device, non_blocking=True)`；`optimizer.zero_grad(set_to_to_none=True)`。
- 训练加梯度裁剪 `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)`（可配）。
- 数值打印统一保留 4 位小数。

        
