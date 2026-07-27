# Project Name

---
## Overview:

A deep learning framework for ...

> **Author:** xxxx xx
> **Organization:** xxx
> **Created:** 2026-xx-xx
> **Period:** 2026-xx-xx ~ 2026-xx-xx


---
## Contents

- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Usage](#usage)

---
## Installation
```
conda create -n tooth python==3.12
conda activate tooth

pip install -r requirements.txt
```
tip: pytorch (cuda version) installation is independent.
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
```

---
## Dataset Preparation
0. **初始化数据**
>|--Data
>
>|--Project


1. **划分train-val-test**
```
python split.py
```

2. **准备pkl**
```
python prepare_pkl.py
```

---
## Training
- **Before training, you should check config.py**
```
python train.py
```

---
## Inference
- Mode 1: one file
```
python inference.py xxx.ply --ckpt ./runs/run_0x/best_loss.pth --save ./output
```
- Mode 2: one folfer
```
python inference.py path/to/folder/ --ckpt ./runs/_run_0x/best_loss.pth --save ./output
```


