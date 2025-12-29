from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from .ray_drr import render_drr_with_rays_torch
from .scene import build_joint_scene_from_true_pose
from .volumes import volume_to_drr


def load_true_pose_row(
    csv_path: Path,
    femur_nii_name: str,
    tibia_nii_name: str,
    anatomy_type: str,
) -> dict:
    """
    从 trueposes.csv 中找到一条与 femur/tibia NIfTI 路径匹配、类型匹配的记录，
    并解析出 femur/tibia 的旋转矩阵和平移向量。
    """
    df = pd.read_csv(csv_path)

    anatomy_type = anatomy_type.lower()
    df = df[df["type"].str.lower() == anatomy_type]

    df_sel = df[
        df["femur_nii"].str.contains(femur_nii_name)
        & df["tibia_nii"].str.contains(tibia_nii_name)
    ]

    if df_sel.empty:
        raise ValueError(
            f"No row in {csv_path} matches femur_nii contains '{femur_nii_name}' "
            f"and tibia_nii contains '{tibia_nii_name}' and type='{anatomy_type}'."
        )

    row = df_sel.iloc[0]

    femur_R = np.array(
        [
            [row["femur_rxx"], row["femur_rxy"], row["femur_rxz"]],
            [row["femur_ryx"], row["femur_ryy"], row["femur_ryz"]],
            [row["femur_rzx"], row["femur_rzy"], row["femur_rzz"]],
        ],
        dtype=np.float32,
    )
    femur_t = np.array(
        [row["femur_tx"], row["femur_ty"], row["femur_tz"]], dtype=np.float32
    )

    tibia_R = np.array(
        [
            [row["tibia_rxx"], row["tibia_rxy"], row["tibia_rxz"]],
            [row["tibia_ryx"], row["tibia_ryy"], row["tibia_ryz"]],
            [row["tibia_rzx"], row["tibia_rzy"], row["tibia_rzz"]],
        ],
        dtype=np.float32,
    )
    tibia_t = np.array(
        [row["tibia_tx"], row["tibia_ty"], row["tibia_tz"]], dtype=np.float32
    )

    calib = dict(
        cal_focal_length=float(row["cal_focal_length"]),
        cal_principalp_x=float(row["cal_principalp_x"]),
        cal_principalp_y=float(row["cal_principalp_y"]),
        cal_mm_per_pxl=float(row["cal_mm_per_pxl"]),
    )

    return dict(
        femur_R=femur_R,
        femur_t=femur_t,
        tibia_R=tibia_R,
        tibia_t=tibia_t,
        calib=calib,
    )


def sample_pose_noise(
    translation_range_mm: float,
    rotation_range_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    采样一个 SE(3) 噪声：小平移 + 小旋转（随机轴，角度在 [0, rotation_range] 度内）。
    """
    dt = np.random.uniform(-translation_range_mm, translation_range_mm, size=(3,)).astype(
        np.float32
    )
    # 收紧 x/y 方向的平移噪声，让骨大致保持在视野中心（例如 xy 只有 z 的 20% 幅度）
    dt[0] *= 0.2  # x
    dt[1] *= 0.2  # y

    if rotation_range_deg <= 0:
        dR = np.eye(3, dtype=np.float32)
        return dR, dt

    max_angle_rad = rotation_range_deg * np.pi / 180.0
    angle = np.random.uniform(0.0, max_angle_rad)

    axis = np.random.uniform(-1.0, 1.0, size=(3,))
    norm = np.linalg.norm(axis)
    if norm < 1e-6:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        axis = axis / norm

    ux, uy, uz = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1.0 - c

    dR = np.array(
        [
            [c + ux * ux * C, ux * uy * C - uz * s, ux * uz * C + uy * s],
            [uy * ux * C + uz * s, c + uy * uy * C, uy * uz * C - ux * s],
            [uz * ux * C - uy * s, uz * uy * C + ux * s, c + uz * uz * C],
        ],
        dtype=np.float32,
    )

    return dR, dt


def generate_split_from_true_pose(
    femur_vol: np.ndarray,
    femur_affine: np.ndarray,
    tibia_vol: Optional[np.ndarray],
    tibia_affine: Optional[np.ndarray],
    true_pose: dict,
    n_samples: int,
    trans_noise_mm: float,
    rot_noise_deg: float,
    pad_scene: int,
    voxel_size_mm: float,
    projection: str = "ray",
    desc: str = "Generating",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    在 trueposes.csv 的真实 femur+tibia 位姿附近采样 n_samples 个 noisy pose，
    使用 GPU 射线积分生成 DRR 图像（透视/物理投影）。

    这里会同时为 femur 和 tibia 采样独立的噪声，并且在 pose 向量中
    同时保存两块骨头的姿态，方便后续网络同时回归：

      pose_vec: [femur_t(3), femur_R(9), tibia_t(3), tibia_R(9)]  -> shape (24,)

    如果没有 tibia（极端情况），则只保存 femur 部分 (12,)。

    Returns:
        images: (N, H, W)
        poses:  (N, 24) or (N, 12)
    """
    femur_R_true = true_pose["femur_R"]
    femur_t_true = true_pose["femur_t"]
    tibia_R_true = true_pose.get("tibia_R", None)
    tibia_t_true = true_pose.get("tibia_t", None)
    calib = true_pose.get("calib", {})

    # ---------------------------------------------------------------
    # 把 truepose 先在 3D 里 “flip 一次”，再作为新的基准姿态加噪声：
    #   - 图像里效果等价于把画面绕光轴旋转 180°（上下+左右互换）
    #   - 之后所有噪声都是围绕这个 flip 后的姿态进行
    # ---------------------------------------------------------------
    R_flip = np.array(
        [
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    femur_R_base = R_flip @ femur_R_true
    femur_t_base = R_flip @ femur_t_true

    if tibia_R_true is not None and tibia_t_true is not None:
        tibia_R_base = R_flip @ tibia_R_true
        tibia_t_base = R_flip @ tibia_t_true
    else:
        tibia_R_base = None
        tibia_t_base = None

    images = []
    pose_vecs = []

    for _ in tqdm(range(n_samples), desc=desc):
        # femur 噪声（围绕 flip 后的基准姿态）
        dR_f, dt_f = sample_pose_noise(trans_noise_mm, rot_noise_deg)
        femur_R_noisy = dR_f @ femur_R_base
        femur_t_noisy = femur_t_base + dt_f

        # tibia 噪声（独立，围绕 flip 后的基准姿态）
        if tibia_R_base is not None and tibia_t_base is not None:
            dR_t, dt_t = sample_pose_noise(trans_noise_mm, rot_noise_deg)
            tibia_R_noisy = dR_t @ tibia_R_base
            tibia_t_noisy = tibia_t_base + dt_t
        else:
            tibia_R_noisy = None
            tibia_t_noisy = None

        if projection == "ray":
            # 透视 / 物理射线投影（GPU）
            # 为了获得更清晰的骨边界，这里使用更高的输出分辨率（例如 512x512），
            # 其余几何保持不变。
            img = render_drr_with_rays_torch(
                femur_vol_np=femur_vol,
                femur_affine_np=femur_affine,
                femur_R=femur_R_noisy,
                femur_t=femur_t_noisy,
                tibia_vol_np=tibia_vol,
                tibia_affine_np=tibia_affine,
                tibia_R=tibia_R_noisy,
                tibia_t=tibia_t_noisy,
                calib=calib,
                img_size=512,
                n_steps=256,
            )
        else:
            # 正交 / 简化 sum-axis 投影
            scene_vol = build_joint_scene_from_true_pose(
                femur_vol=femur_vol,
                femur_affine=femur_affine,
                tibia_vol=tibia_vol,
                tibia_affine=tibia_affine,
                femur_R=femur_R_noisy,
                femur_t=femur_t_noisy,
                tibia_R=tibia_R_noisy,
                tibia_t=tibia_t_noisy,
                pad_scene=pad_scene,
                voxel_size_mm=voxel_size_mm,
            )
            img = volume_to_drr(scene_vol)

        images.append(img.astype(np.float32))

        # 构建 pose 向量：优先保存 femur，若存在 tibia 再拼接 tibia 部分
        femur_part = np.concatenate(
            [femur_t_noisy.astype(np.float32), femur_R_noisy.reshape(-1).astype(np.float32)],
            axis=0,
        )  # (12,)

        if tibia_R_noisy is not None and tibia_t_noisy is not None:
            tibia_part = np.concatenate(
                [tibia_t_noisy.astype(np.float32), tibia_R_noisy.reshape(-1).astype(np.float32)],
                axis=0,
            )  # (12,)
            pose_vec = np.concatenate([femur_part, tibia_part], axis=0)  # (24,)
        else:
            pose_vec = femur_part

        pose_vecs.append(pose_vec)

    images_np = np.stack(images, axis=0)
    poses_np = np.stack(pose_vecs, axis=0)
    return images_np, poses_np


