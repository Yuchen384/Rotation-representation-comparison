## DRR 2D/3D Registration: Rotation Representation Benchmark

This repo builds an end‑to‑end pipeline to:

- **Generate DRR images** from femur + tibia CT (NIfTI) using a GPU ray‑casting renderer.
- **Train CNN regressors** (ResNet18 backbone) for **3D rotation of both femur and tibia**.
- **Compare four rotation representations**:
  - **Euler** (`euler`)
  - **Quaternion** (`quat`)
  - **6D continuous** (`6d`, Zhou et al. 2019)
  - **Axis‑angle** (`axis_angle`)
- **Evaluate** each representation with:
  - Per‑bone rotation error (**geodesic distance** on \(SO(3)\)) for femur / tibia.
  - Per‑bone **2D projection error** under a simple pinhole camera.
  - A mathematically defined **overall pair** metric that combines femur+tibia:
    - Rotation: \(\sqrt{d_f^2 + d_t^2}\)
    - Projection: RMS \(\sqrt{(e_f^2 + e_t^2)/2}\)

---

### 1. Data generation (DRR)

- Main script: `prepare_data.py`  
  From NIfTI volumes (`.nii`) and `trueposes.csv` it:
  - Loads femur / tibia, crops non‑zero bone region, applies configurable padding.
  - Applies a fixed flip so that **femur is on top, tibia on bottom** in DRR.
  - Adds controlled SE(3) noise around the flipped true pose, independently for both bones.
  - Uses `data_prep/ray_drr.py` to render **512×512 physical DRRs on GPU**.
  - Saves:
    - `data/images_{train,val}.npy`: `(N, H, W)`
    - `data/poses_{train,val}.npy`: `(N, 24)` = `[t_f(3), R_f(9), t_t(3), R_t(9)]`

Typical command (single‑GPU, 200 train / 50 val, recommended settings):

```bash
cd /home/user/qzhao/rotation
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

To visually inspect DRRs and poses:

```bash
conda run -n rotation python npy_preview.py
```

This writes PNGs into `preview_images/` with **femur/tibia Euler angles** in the title.

---

### 2. Training (single or multi‑GPU)

- Main script: `train.py`
- Model: `RotationRegressionModule` in `models.py`
  - Backbone: `DRRBackbone` (ResNet18, 1‑channel input).
  - Output: rotation parameters for **both femur and tibia**; internally converted to rotation matrices.
  - Loss: average geodesic distance (in radians) over available bones.

Single‑GPU training example (6D representation, 50 epochs):

```bash
cd /home/user/qzhao
CUDA_VISIBLE_DEVICES=4 conda run -n rotation python -m rotation.train \
  --data-root rotation/data \
  --rotation-repr 6d \
  --batch-size 8 \
  --max-epochs 50 \
  --devices 0 \
  --output-dir rotation/checkpoints
```

Change `--rotation-repr` to `euler`, `quat`, `6d`, or `axis_angle` to train each representation.  
For multi‑GPU DDP, set e.g. `CUDA_VISIBLE_DEVICES=5,6 --devices 0,1`.

After each run you get:

- `checkpoints/<repr>/version_*/metrics.csv` and `loss_curve.png`
- Several `epoch=XXX-val_loss=YYYY.ckpt` files.

---

### 3. Evaluation & comparison

- Main script: `eval.py`
- For each representation it:
  - Loads the **most recent checkpoint** from `checkpoints/<repr>`.
  - Runs on the validation set and computes:
    - **Femur**: geodesic_deg mean/std, proj_err mean/std.
    - **Tibia**: geodesic_deg mean/std, proj_err mean/std.
    - **Overall pair**:
      - geodesic overall: \(\sqrt{d_f^2 + d_t^2}\)
      - projection overall: \(\sqrt{(e_f^2 + e_t^2)/2}\)

Example: evaluate all four representations and append to a summary log:

```bash
cd /home/user/qzhao
CUDA_VISIBLE_DEVICES=4 conda run -n rotation python -m rotation.eval \
  --data-root rotation/data \
  --rotation-reprs euler quat 6d axis_angle \
  --checkpoints-dir rotation/checkpoints \
  --batch-size 32 \
  --num-workers 4 \
  --gpus 1 >> rotation/@eval_all.log
```

The file `@eval_all.log` is a **human‑readable summary**, organized as:

- Per representation: `femur / tibia / overall pair` metrics.
- Final **ranking** by geodesic_deg mean for femur / tibia / overall pair.

---

### 4. What this repo is for

- A **controlled benchmark** of rotation representations (Euler / quat / 6D / axis‑angle)  
  for deep 2D/3D registration on DRR images.
- Joint regression of **two bones (femur + tibia)** in a realistic DRR setting.
- Quantitative comparison using:
  - Per‑bone 3D rotation error (geodesic on \(SO(3)\)).
  - Per‑bone and overall 2D projection error.
- Ready‑to‑run scripts for:
  - Data generation (`prepare_data.py`)
  - Training (`train.py`)
  - Evaluation & summary logging (`eval.py`, `@eval_all.log`)
