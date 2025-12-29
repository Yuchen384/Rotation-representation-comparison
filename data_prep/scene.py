from typing import Tuple, Optional

import numpy as np
from scipy.ndimage import affine_transform


def build_joint_scene_from_true_pose(
    femur_vol: np.ndarray,
    femur_affine: np.ndarray,
    tibia_vol: Optional[np.ndarray],
    tibia_affine: Optional[np.ndarray],
    femur_R: np.ndarray,
    femur_t: np.ndarray,
    tibia_R: Optional[np.ndarray],
    tibia_t: Optional[np.ndarray],
    pad_scene: float = 32.0,
    voxel_size_mm: float = 0.5,
) -> np.ndarray:
    """
    使用 trueposes.csv 给出的 femur/tibia 姿态 + NIfTI affine，在世界/相机坐标系下
    构建一个真实 3D 场景体素网格，然后对 femur/tibia 分别做仿射重采样，最后在场景中叠加。

    坐标约定：
      - NIfTI affine: x_ct_world = A_b @ [i, j, k, 1]^T
      - true pose:    x_cam      = T_b @ x_ct_world, 其中 T_b = [R_b, t_b; 0, 1]
      - 场景体素:     x_cam      = A_scene @ [d, h, w, 1]^T

    于是，从场景体素坐标到骨体素坐标的映射为：
        [i, j, k, 1]^T = A_b^{-1} T_b^{-1} A_scene [d, h, w, 1]^T
    这正是 scipy.ndimage.affine_transform 所需的输入映射。
    """
    bones = [(femur_vol, femur_affine, femur_R, femur_t)]
    if (
        tibia_vol is not None
        and tibia_affine is not None
        and tibia_R is not None
        and tibia_t is not None
    ):
        bones.append((tibia_vol, tibia_affine, tibia_R, tibia_t))

    # 1) 估计在相机坐标系下的整体 bounding box（mm）
    all_pts = []
    for vol, affine, R_b, t_b in bones:
        D_b, H_b, W_b = vol.shape
        corners = [
            (0, 0, 0),
            (D_b, 0, 0),
            (0, H_b, 0),
            (0, 0, W_b),
            (D_b, H_b, 0),
            (D_b, 0, W_b),
            (0, H_b, W_b),
            (D_b, H_b, W_b),
        ]
        T_b = np.eye(4, dtype=np.float32)
        T_b[:3, :3] = R_b
        T_b[:3, 3] = t_b

        for (i, j, k) in corners:
            v = np.array([i, j, k, 1.0], dtype=np.float32)
            x_ct = affine @ v          # CT 世界坐标
            x_cam = T_b @ x_ct         # 相机坐标
            all_pts.append(x_cam[:3])

    all_pts = np.stack(all_pts, axis=0)  # (N, 3)
    xyz_min = all_pts.min(axis=0) - pad_scene
    xyz_max = all_pts.max(axis=0) + pad_scene

    # 2) 定义场景体素网格大小与 affine（相机坐标 -> 场景索引）
    extent = xyz_max - xyz_min  # (x, y, z) 范围
    D_scene = int(np.ceil(extent[2] / voxel_size_mm))  # depth (沿 z_cam)
    H_scene = int(np.ceil(extent[1] / voxel_size_mm))  # height (沿 y_cam)
    W_scene = int(np.ceil(extent[0] / voxel_size_mm))  # width  (沿 x_cam)

    D_scene = max(D_scene, 1)
    H_scene = max(H_scene, 1)
    W_scene = max(W_scene, 1)

    # A_scene: [d, h, w, 1]^T -> [x_cam, y_cam, z_cam, 1]^T
    vs = float(voxel_size_mm)
    A_scene = np.array(
        [
            [0.0, 0.0, vs, xyz_min[0]],  # x_cam = vs * w + x_min
            [0.0, vs, 0.0, xyz_min[1]],  # y_cam = vs * h + y_min
            [vs, 0.0, 0.0, xyz_min[2]],  # z_cam = vs * d + z_min
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    scene_vol = np.zeros((D_scene, H_scene, W_scene), dtype=np.float32)

    # 3) 对每块骨：用 affine_transform 把 (i,j,k) 采样映射到场景体素上
    for vol, affine, R_b, t_b in bones:
        T_b = np.eye(4, dtype=np.float32)
        T_b[:3, :3] = R_b
        T_b[:3, 3] = t_b
        A_b_inv = np.linalg.inv(affine)
        T_b_inv = np.linalg.inv(T_b)

        # 从场景体素到骨体素的 4x4 映射
        M4 = A_b_inv @ T_b_inv @ A_scene  # shape (4,4)
        M = M4[:3, :3]
        offset = M4[:3, 3]

        contrib = affine_transform(
            vol,
            M,
            offset=offset,
            output_shape=scene_vol.shape,
            order=1,
            mode="constant",
            cval=0.0,
        ).astype(np.float32)
        scene_vol += contrib

    # 4) 归一化到 [0,1]
    vmin, vmax = scene_vol.min(), scene_vol.max()
    if vmax > vmin:
        scene_vol = (scene_vol - vmin) / (vmax - vmin)
    else:
        scene_vol = np.zeros_like(scene_vol, dtype=np.float32)

    return scene_vol.astype(np.float32)


