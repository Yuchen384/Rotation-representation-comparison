import argparse
from pathlib import Path

import numpy as np

from data_prep.volumes import load_volume, crop_volume_around_bone
import data_prep.poses as poses_mod


def sample_pose_noise(translation_range_mm: float,
                      rotation_range_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """
    采样一个 SE(3) 噪声：小平移 + 小旋转（随机轴，角度在 [0, rotation_range] 度内）。
    """
    dt = np.random.uniform(-translation_range_mm,
                           translation_range_mm, size=(3,)).astype(np.float32)

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


# ============ 体积 → 场景 → DRR ============

def rotate_local_volume(vol: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    在 volume 自身坐标系中绕其中心应用旋转 R。
    """
    D_b, H_b, W_b = vol.shape
    center_local = 0.5 * np.array([D_b, H_b, W_b], dtype=np.float32)
    M = R.T.astype(np.float32)
    offset = center_local - M @ center_local

    rotated = affine_transform(
        vol,
        M,
        offset=offset,
        order=1,
        mode="constant",
        cval=0.0,
    ).astype(np.float32)
    return rotated


def place_volume_with_center(
    scene_vol: np.ndarray,
    bone_vol: np.ndarray,
    center_vox: np.ndarray,
) -> None:
    """
    在场景中按给定体素中心 center_vox 摆放一块骨。
    """
    D_b, H_b, W_b = bone_vol.shape
    center_local = 0.5 * np.array([D_b, H_b, W_b], dtype=np.float32)

    start = (center_vox - center_local).astype(int)
    d0, h0, w0 = start
    d1, h1, w1 = d0 + D_b, h0 + H_b, w0 + W_b

    D_s, H_s, W_s = scene_vol.shape

    d0_clip = max(d0, 0)
    h0_clip = max(h0, 0)
    w0_clip = max(w0, 0)
    d1_clip = min(d1, D_s)
    h1_clip = min(h1, H_s)
    w1_clip = min(w1, W_s)

    bd0 = d0_clip - d0
    bh0 = h0_clip - h0
    bw0 = w0_clip - w0
    bd1 = D_b - (d1 - d1_clip)
    bh1 = H_b - (h1 - h1_clip)
    bw1 = W_b - (w1 - w1_clip)

    if d1_clip > d0_clip and h1_clip > h0_clip and w1_clip > w0_clip:
        scene_vol[d0_clip:d1_clip, h0_clip:h1_clip, w0_clip:w1_clip] += bone_vol[
            bd0:bd1, bh0:bh1, bw0:bw1
        ]


def build_joint_scene_from_true_pose(
    femur_vol: np.ndarray,
    femur_affine: np.ndarray,
    tibia_vol: np.ndarray | None,
    tibia_affine: np.ndarray | None,
    femur_R: np.ndarray,
    femur_t: np.ndarray,
    tibia_R: np.ndarray | None,
    tibia_t: np.ndarray | None,
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
    if tibia_vol is not None and tibia_affine is not None and tibia_R is not None and tibia_t is not None:
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


def volume_to_drr(volume: np.ndarray) -> np.ndarray:
    """
    简化版：不再随机旋转，只对联合场景做一次 z 轴积分得到 DRR。
    """
    assert volume.ndim == 3

    drr = volume.sum(axis=0)
    dmin, dmax = drr.min(), drr.max()
    if dmax > dmin:
        drr = (drr - dmin) / (dmax - dmin)
    else:
        drr = np.zeros_like(drr, dtype=np.float32)
    return drr.astype(np.float32)


# ============ 生成数据：在 true pose 上加噪声 ============

def generate_split_from_true_pose(
    femur_vol: np.ndarray,
    femur_affine: np.ndarray,
    tibia_vol: np.ndarray | None,
    tibia_affine: np.ndarray | None,
    true_pose: dict,
    n_samples: int,
    trans_noise_mm: float,
    rot_noise_deg: float,
    pad_scene: int,
    voxel_size_mm: float,
    desc: str = "Generating",
) -> tuple[np.ndarray, np.ndarray]:
    """
    在 trueposes.csv 的真实 femur+tibia 位姿附近采样 n_samples 个 noisy pose，
    为每个 noisy pose 构建联合 3D 场景并生成 DRR。
    """
    femur_R_true = true_pose["femur_R"]
    femur_t_true = true_pose["femur_t"]
    tibia_R_true = true_pose.get("tibia_R", None)
    tibia_t_true = true_pose.get("tibia_t", None)

    images = []
    pose_vecs = []

    for _ in tqdm(range(n_samples), desc=desc):
        dR_f, dt_f = sample_pose_noise(trans_noise_mm, rot_noise_deg)
        femur_R_noisy = dR_f @ femur_R_true
        femur_t_noisy = femur_t_true + dt_f

        if tibia_R_true is not None and tibia_t_true is not None:
            dR_t, dt_t = sample_pose_noise(trans_noise_mm, rot_noise_deg)
            tibia_R_noisy = dR_t @ tibia_R_true
            tibia_t_noisy = tibia_t_true + dt_t
        else:
            tibia_R_noisy = None
            tibia_t_noisy = None

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

        pose_vec = np.concatenate(
            [femur_t_noisy.astype(np.float32), femur_R_noisy.reshape(-1).astype(np.float32)],
            axis=0,
        )
        pose_vecs.append(pose_vec)

    images_np = np.stack(images, axis=0)
    poses_np = np.stack(pose_vecs, axis=0)
    return images_np, poses_np


# ============ CLI ============

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a DRR-like dataset from femur+tibia NIfTI volumes using trueposes.csv + noise."
    )
    parser.add_argument("--data-root", type=str, default="data", help="Directory where NIfTI files live.")
    parser.add_argument("--femur-name", type=str, default="SUBN_02_Femur_RE_Volume.nii")
    parser.add_argument("--tibia-name", type=str, default="SUBN_02_Tibia_RE_Volume.nii")
    parser.add_argument("--trueposes-csv", type=str, default="trueposes.csv", help="CSV with true femur+tibia poses.")
    parser.add_argument("--anatomy-type", type=str, default="bone", help="bone / implant / dupla")

    parser.add_argument("--output-root", type=str, default="data", help="Directory to save numpy arrays.")
    parser.add_argument("--train-samples", type=int, default=500, help="Number of training samples to render.")
    parser.add_argument("--val-samples", type=int, default=100, help="Number of validation samples to render.")

    parser.add_argument("--trans-noise-mm", type=float, default=10.0, help="Max translation noise (mm).")
    parser.add_argument("--rot-noise-deg", type=float, default=10.0, help="Max rotation noise angle (deg).")

    parser.add_argument(
        "--pad",
        type=int,
        default=0,
        help="Padding (in voxels) around each bone before building the scene. "
             "Default 0 = 不额外扩展体积边界，可以减弱 3D 盒子的白框感。",
    )
    parser.add_argument(
        "--pad-scene",
        type=int,
        default=0,
        help="Extra padding (in mm) around the joint scene volume for orth projection. "
             "Default 0 = 不额外扩展场景体积边界。",
    )
    parser.add_argument("--voxel-size-mm", type=float, default=0.5, help="Approximate voxel size (mm).")
    parser.add_argument(
        "--projection",
        type=str,
        default="ray",
        choices=["ray", "orth"],
        help="Projection type: 'ray' for perspective ray integration, 'orth' for sum-axis orthographic.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    femur_path = data_root / args.femur_name
    tibia_path = data_root / args.tibia_name
    if not femur_path.exists():
        raise FileNotFoundError(f"Femur NIfTI volume not found: {femur_path}")

    print(f"Loading femur from {femur_path} ...")
    femur_vol_raw, femur_affine_raw = load_volume(femur_path)
    tibia_vol_raw, tibia_affine_raw = None, None
    if tibia_path.exists():
        print(f"Loading tibia from {tibia_path} ...")
        tibia_vol_raw, tibia_affine_raw = load_volume(tibia_path)

    print("Cropping femur (and tibia if present) around bone (no extra padding by default)...")
    # 对所有投影模式统一使用 args.pad 作为单骨 volume 的 padding；
    # 默认 pad=0，即仅裁到骨的 bounding box，不再额外扩展，减弱白框。
    pad_bone = max(int(args.pad), 0)
    femur_vol, femur_affine = crop_volume_around_bone(
        femur_vol_raw, femur_affine_raw, pad=pad_bone
    )
    tibia_vol, tibia_affine = (
        crop_volume_around_bone(tibia_vol_raw, tibia_affine_raw, pad=pad_bone)
        if tibia_vol_raw is not None
        else (None, None)
    )

    print(f"Femur padded volume shape: {femur_vol.shape}")
    if tibia_vol is not None:
        print(f"Tibia padded volume shape: {tibia_vol.shape}")

    truepose = poses_mod.load_true_pose_row(
        csv_path=Path(args.trueposes_csv),
        femur_nii_name=args.femur_name,
        tibia_nii_name=args.tibia_name,
        anatomy_type=args.anatomy_type,
    )
    print("Loaded true pose from CSV.")

    print("Generating training split (noisy poses around true pose)...")
    images_train, poses_train = poses_mod.generate_split_from_true_pose(
        femur_vol=femur_vol,
        femur_affine=femur_affine,
        tibia_vol=tibia_vol,
        tibia_affine=tibia_affine,
        true_pose=truepose,
        n_samples=args.train_samples,
        trans_noise_mm=args.trans_noise_mm,
        rot_noise_deg=args.rot_noise_deg,
        pad_scene=max(int(args.pad_scene), 0),
        voxel_size_mm=args.voxel_size_mm,
        projection=args.projection,
        desc="Train",
    )
    np.save(output_root / "images_train.npy", images_train)
    np.save(output_root / "poses_train.npy", poses_train)
    print(f"Saved train images to {output_root / 'images_train.npy'} with shape {images_train.shape}")
    print(f"Saved train poses to {output_root / 'poses_train.npy'} with shape {poses_train.shape}")

    if args.val_samples > 0:
        print("Generating validation split...")
        images_val, poses_val = poses_mod.generate_split_from_true_pose(
            femur_vol=femur_vol,
            femur_affine=femur_affine,
            tibia_vol=tibia_vol,
            tibia_affine=tibia_affine,
            true_pose=truepose,
            n_samples=args.val_samples,
            trans_noise_mm=args.trans_noise_mm,
            rot_noise_deg=args.rot_noise_deg,
            pad_scene=max(int(args.pad_scene), 0),
            voxel_size_mm=args.voxel_size_mm,
            projection=args.projection,
            desc="Val",
        )
        np.save(output_root / "images_val.npy", images_val)
        np.save(output_root / "poses_val.npy", poses_val)
        print(f"Saved val images to {output_root / 'images_val.npy'} with shape {images_val.shape}")
        print(f"Saved val poses to {output_root / 'poses_val.npy'} with shape {poses_val.shape}")


if __name__ == "__main__":
    main()
