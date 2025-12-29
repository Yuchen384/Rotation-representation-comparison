import os
import matplotlib
print("Matplotlib backend:", matplotlib.get_backend())

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rsc

# 路径按你实际情况来（这里预览最新生成的 5 个 sample）
images_train = np.load("data/images_train.npy")   # (N, H, W)
poses_train = np.load("data/poses_train.npy")     # (N, 24) 或 (N, 12)

print("images_train shape:", images_train.shape)
print("poses_train shape:", poses_train.shape)

# ==== 1. 创建保存图片的文件夹 ====
out_dir = "preview_images"   # 你想要的文件夹名
os.makedirs(out_dir, exist_ok=True)

# ==== 2. 一次性保存前 5 个样本 ====
num_samples = min(5, images_train.shape[0])  # 防止数据少于 5 张

for i in range(num_samples):
    img = images_train[i]
    pose = poses_train[i]

    # 取 femur 的 3x3 旋转矩阵部分：pose = [femur_t(3), femur_R(9), tibia_t(3), tibia_R(9)]
    R_femur_flat = pose[3:12].astype(float)
    R_femur = R_femur_flat.reshape(3, 3)

    # 如果存在 tibia 部分（24 维），则同时解析 tibia 的旋转
    has_tibia = pose.shape[0] >= 24 or len(pose) >= 24
    R_tibia = None
    if has_tibia:
        R_tibia_flat = pose[15:24].astype(float)
        R_tibia = R_tibia_flat.reshape(3, 3)

    r_femur = Rsc.from_matrix(R_femur)
    euler_femur = r_femur.as_euler("xyz", degrees=True)

    if R_tibia is not None:
        r_tibia = Rsc.from_matrix(R_tibia)
        euler_tibia = r_tibia.as_euler("xyz", degrees=True)
        title = (
            f"Image #{i}\n"
            f"Femur Euler (deg): {euler_femur}\n"
            f"Tibia Euler (deg): {euler_tibia}"
        )
    else:
        title = f"Image #{i}\nFemur Euler (deg): {euler_femur}"

    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    # 文件名，例如 image_000.png, image_001.png, ...
    filename = os.path.join(out_dir, f"image_{i:03d}.png")
    plt.savefig(filename, dpi=150)
    plt.close()   # 关闭当前 figure，避免内存占用
    print(f"Saved {filename}")

print(f"\nAll {num_samples} images saved to folder: {out_dir}")
