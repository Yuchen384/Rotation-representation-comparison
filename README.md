## Project: Evaluating Rotation Representations for Deep 2D/3D Registration (DRR)

This repo implements a full pipeline to **generate DRR data from CT** and to **train / evaluate**
deep regression models for 3D rotation using three popular rotation representations:

- **Euler angles (XYZ)**
- **Quaternion (normalized 4D, wxyz)**
- **6D continuous representation** (Zhou et al., CVPR 2019)

The project focuses on:

- **Rotation error**: geodesic distance between predicted and ground-truth rotations.  
- **Registration accuracy**: a proxy **2D projection error** under a simple pinhole camera.  
- **Training behaviour**: convergence speed and stability of different rotation parameterizations.

---

### 1. Directory layout

- `prepare_data.py`  
  CLI 脚本，从 NIfTI CT 体积和 `trueposes.csv` 生成 DRR + pose：
  - 读取 femur / tibia `.nii` 和 NIfTI affine；
  - 使用 `data_prep/` 下的模块构建世界坐标系、相机位姿；
  - 用 **物理 ray-casting（GPU）** 渲染 DRR；
  - 输出：
    - `data/images_train.npy`, `data/images_val.npy`  (N, H, W)  
    - `data/poses_train.npy`,  `data/poses_val.npy`   (N, 12) = `[tx, ty, tz, R(9)]`

- `data_prep/`  
  - `volumes.py`：加载 NIfTI，裁剪骨的非零 bbox，做 padding，并更新 affine。  
  - `poses.py`：读取 `trueposes.csv`，在真姿态附近加 SE(3) 噪声，驱动渲染。  
  - `scene.py`：在相机坐标系下构建联合 3D 场景体素（主要给 orth 投影用）。  
  - `ray_drr.py`：**GPU 射线积分 DRR 渲染器**（物理透视投影，支持 512×512）。  

- `data.py`  
  - `DRRRotationDataset`：从 `images_*.npy` / `poses_*.npy` 构建样本：
    - 图像：`(N, H, W)` → `(N, 1, H, W)`，并归一化到 `[-1, 1]`。  
    - 姿态：`pose = [tx, ty, tz, R(9)]` → 提取 `rotmat (3×3)` 和 `t (3,)`。  
  - `build_dataloaders`：返回 train / val 的 `DataLoader`。

- `models.py`  
  - `DRRBackbone`：
    - 预训练 ResNet18，第一层改成单通道输入；  
    - 去掉原来的 FC，接一个新的 `Linear` 输出旋转参数。  
  - `RotationRegressionModule` (PyTorch Lightning)：  
  - `rotation_repr ∈ {euler, quat, 6d, axis_angle}` → 输出维度分别为 3 / 4 / 6 / 3；  
    - 使用 `representations.py` 统一转换为 3×3 rotation matrix；  
    - 损失：**geodesic distance**（弧度）`loss = mean(geodesic)`。

- `representations.py`  
  - Euler ↔ rotation matrix (`euler_xyz_to_matrix`, `matrix_to_euler_xyz`)  
  - Quaternion ↔ rotation matrix (`quat_to_matrix`, `matrix_to_quat`)  
  - 6D 连续表示 ↔ rotation matrix (`cont6d_to_matrix`, `matrix_to_cont6d`)  
  - `geodesic_distance_from_matrices`：so(3) 上的测地线距离。  
  - `project_points`：简单 pinhole 相机投影。

- `metrics.py`  
  - `rotation_metrics`：返回 `geodesic_rad` 和 `geodesic_deg`。  
  - `projection_error`：在给定内参 K 和 (R, t) 下，对 3D 点集计算 2D 均值像素误差。

- `train.py`  
  - 命令行参数：`--rotation-repr {euler, quat, 6d}`, `--max-epochs`, `--devices`, `--data-root` 等。  
  - 使用 PyTorch Lightning `Trainer`：
    - 支持 **多 GPU DDP**：例如 `CUDA_VISIBLE_DEVICES=5,6 --devices 0,1`；  
    - 回调：`ModelCheckpoint`, `LearningRateMonitor`, `TQDMProgressBar`；  
    - Logger：`CSVLogger` 记录 `metrics.csv`，并自动生成 `loss_curve.png`。

- `eval.py`  
  - 命令行参数：`--rotation-reprs euler quat 6d`, `--checkpoints-dir`, `--data-root` 等。  
  - 对每种表示：
    - 加载 best checkpoint（按 `val_loss` 排序）；  
    - 在 val 集上计算：
      - **旋转误差**：geodesic_deg mean / std；  
      - **2D 投影误差**：在单位立方体角点 + 简单 K=I 下的像素误差 mean / std；  
    - 输出结果到终端，并可用 `tee` 保存到 `eval_*.log`。

- `preview_images/`  
  - `npy_preview.py`：将 `images_*.npy` 中的样本保存为 PNG，方便肉眼检查 DRR 外观。

---

### 2. Data generation (DRR)

#### 单次样本生成示例（ray 投影, 512×512）

```bash
cd /home/user/yucqin/rotation
CUDA_VISIBLE_DEVICES=5 conda run -n rotation python prepare_data.py \
  --data-root data \
  --output-root data \
  --train-samples 1 \
  --val-samples 0 \
  --trueposes-csv data/trueposes.csv \
  --anatomy-type bone \
  --projection ray \
  --pad 0 \
  --pad-scene 32 \
  --voxel-size-mm 0.5
```

生成：

- `data/images_train.npy`，形状 `(1, 512, 512)`  
- `data/poses_train.npy`，形状 `(1, 12)`

#### 实验用数据（200 train / 50 val, 512×512）

```bash
CUDA_VISIBLE_DEVICES=5 conda run -n rotation python prepare_data.py \
  --data-root data \
  --output-root data \
  --train-samples 200 \
  --val-samples 50 \
  --trueposes-csv data/trueposes.csv \
  --anatomy-type bone \
  --projection ray \
  --pad 0 \
  --pad-scene 32 \
  --voxel-size-mm 0.5
```

---

### 3. Training setup

#### 训练单一表示（多 GPU）

以 **6D 表示** 为例：

```bash
cd /home/user/yucqin
CUDA_VISIBLE_DEVICES=5,6 conda run -n rotation python -m rotation.train \
  --data-root rotation/data \
  --rotation-repr 6d \
  --max-epochs 50 \
  --devices 0,1 \
  --output-dir rotation/checkpoints
```

- 会在 `rotation/checkpoints/6d/version_0/` 下生成：
  - `metrics.csv`（记录 train/val loss 等）；  
  - `loss_curve.png`（训练 & 验证 loss 曲线）；  
  - 若干 `epoch=XXX-val_loss=YYYY.ckpt`。

对应地，Euler / Quaternion 只需改 `--rotation-repr` 为 `euler` / `quat` 即可。

---

### 4. Evaluation & comparison

#### 评估单一表示（例如 6D）

```bash
CUDA_VISIBLE_DEVICES=5 conda run -n rotation python -m rotation.eval \
  --data-root rotation/data \
  --rotation-reprs 6d \
  --checkpoints-dir rotation/checkpoints \
  --batch-size 32 \
  --num-workers 4 \
  --gpus 1 | tee rotation/eval_6d.log
```

终端和 log 中会打印类似：

- `6d: geodesic_deg mean=1.005, std=0.649; proj_err (px) mean=..., std=...`

#### 一次性对比 Euler / Quat / 6D

```bash
CUDA_VISIBLE_DEVICES=5 conda run -n rotation python -m rotation.eval \
  --data-root rotation/data \
  --rotation-reprs euler quat 6d \
  --checkpoints-dir rotation/checkpoints \
  --batch-size 32 \
  --num-workers 4 \
  --gpus 1 | tee rotation/eval_all.log
```

当前实验的典型输出：

- **Euler**  
  - geodesic mean ≈ **2.40°**，std ≈ 0.83°  
  - proj_err mean ≈ **2.22×10⁶ px**，std ≈ 7.72×10⁵ px
- **Quaternion**  
  - geodesic mean ≈ **1.48°**，std ≈ 0.76°  
  - proj_err mean ≈ **1.48×10⁶ px**，std ≈ 8.31×10⁵ px
- **6D**  
  - geodesic mean ≈ **1.01°**，std ≈ 0.65°  
  - proj_err mean ≈ **9.46×10⁵ px**，std ≈ 6.24×10⁵ px

这些结果表明，在本项目的 DRR–based 2D/3D 注册任务上：

- **6D 表示在旋转误差与 2D 投影误差上均优于 quaternion，且显著优于 Euler**；  
- 训练曲线（`loss_curve.png`）也显示 6D 在收敛速度和稳定性方面表现更好。

---

### 5. 小结：项目目标与现阶段结论

- 已实现：
  - 基于 NIfTI + `trueposes.csv` 的 **物理一致 ray DRR 渲染链路**；  
  - 统一 backbone + 三种 rotation 表示（Euler / Quat / 6D）的回归模型；  
  - 覆盖 **geodesic rotation error** 与 **2D projection error** 的评估代码；  
  - 完整 50-epoch 训练 + 多 GPU DDP 支持 + loss 曲线导出。

- 在当前设置（单病例 CT、ray DRR、512×512、200/50 样本）下：
  - 6D 表示在 **精度**（geodesic / projection error）和 **训练行为**（收敛平滑度）上均表现最佳；  
  - Quaternion 居中；Euler 最差。  

后续可以在更多病例、更大数据量和更丰富的投影配置上重复实验，进一步验证 6D 在医疗 DRR 2D/3D 注册中的优势与局限。 

