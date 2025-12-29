from typing import Optional, Dict

import numpy as np
import torch
import torch.nn.functional as F


def _volume_to_torch(vol: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Convert (D, H, W) numpy volume to torch tensor of shape (1, 1, D, H, W).
    """
    t = torch.from_numpy(vol).to(device=device, dtype=torch.float32)
    if t.ndim != 3:
        raise ValueError(f"Expected volume of shape (D,H,W), got {t.shape}")
    return t.unsqueeze(0).unsqueeze(0)


def _compute_rays(
    H: int,
    W: int,
    calib: Dict[str, float],
    device: torch.device,
) -> torch.Tensor:
    """
    为每个像素生成一条相机射线方向 (单位向量)。

    简化假设：
      - 相机位于原点；
      - 成像平面在 z = -f 处（相机朝 -z 看）；
      - 像素中心映射到成像平面的物理坐标后，再归一化得到方向。
    """
    f = float(calib.get("cal_focal_length", 980.0))
    # 探测器物理像素尺寸（mm/px）
    cal_mm_per_pxl = float(calib.get("cal_mm_per_pxl", 0.28))
    # nominal_screen_px 控制“等效探测器宽度”，数值越大 → 视野越大、骨更小；越小 → 视野更窄、骨在图里更大。
    # 这里取一个中间值 1500px，在保证基本完整视野的同时不过于远离，配合更高输出分辨率提高清晰度。
    nominal_screen_px = 1500.0
    s_eff = cal_mm_per_pxl * (nominal_screen_px / float(W))

    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )

    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0

    x = (xs - cx) * s_eff
    y = (ys - cy) * s_eff
    z = torch.full_like(x, -f)  # 成像平面在 z = -f，朝 -z 方向

    dirs = torch.stack([x, y, z], dim=-1)  # (H, W, 3)
    dirs = dirs / torch.linalg.norm(dirs, dim=-1, keepdim=True).clamp_min(1e-6)
    return dirs  # (H, W, 3)


def _sample_along_rays_single_bone(
    vol: torch.Tensor,
    affine_np: np.ndarray,
    R_np: np.ndarray,
    t_np: np.ndarray,
    dirs: torch.Tensor,
    n_steps: int,
    t_min: float,
    t_max: float,
) -> torch.Tensor:
    """
    对单块骨（一个 CT 体积）沿射线做体积积分。

    Args:
        vol: (1, 1, D, H, W) torch tensor on device
        affine_np: (4,4) voxel->world 仿射（numpy）
        R_np, t_np: 3x3 旋转矩阵, 3 维平移（numpy），将 CT world 映射到相机坐标
        dirs: (H, W, 3) 每个像素的射线方向（单位向量）
        n_steps: 沿射线的采样步数
        t_min, t_max: 射线参数范围
    Returns:
        img: (H, W) torch tensor
    """
    device = vol.device
    D, H_vol, W_vol = vol.shape[2], vol.shape[3], vol.shape[4]
    H_img, W_img = dirs.shape[0], dirs.shape[1]

    # 射线参数
    ts = torch.linspace(t_min, t_max, steps=n_steps, device=device)  # (n_steps,)

    # 射线采样点：p(t) = t * d，原点在 (0,0,0)
    # dirs: (H,W,3) -> (1,1,H,W,3)
    dirs_exp = dirs.unsqueeze(0).unsqueeze(0)  # (1,1,H_img,W_img,3)
    ts_exp = ts.view(n_steps, 1, 1, 1)         # (n_steps,1,1,1)

    # points_cam: (n_steps,H,W,3)
    points_cam = ts_exp * dirs_exp.squeeze(0).squeeze(0)  # 广播：ts * dirs, 形状 (n_steps,H_img,W_img,3)

    # 组 4x4 变换矩阵：从相机坐标 -> 骨体素坐标
    A_b = torch.from_numpy(affine_np).to(device=device, dtype=torch.float32)
    T_b = torch.eye(4, device=device, dtype=torch.float32)
    T_b[:3, :3] = torch.from_numpy(R_np).to(device=device, dtype=torch.float32)
    T_b[:3, 3] = torch.from_numpy(t_np).to(device=device, dtype=torch.float32)

    M4 = torch.linalg.inv(A_b) @ torch.linalg.inv(T_b)  # (4,4)

    # 将采样点从相机坐标映射到骨体素坐标
    pts_h = torch.cat(
        [points_cam, torch.ones_like(points_cam[..., :1])], dim=-1
    )  # (n_steps,H_img,W_img,4)
    pts_flat = pts_h.view(-1, 4).T  # (4, N) 其中 N = n_steps*H_img*W_img
    vox_flat = (M4 @ pts_flat).T[:, :3]  # (N,3) -> (z,y,x) 体素索引

    z_idx = vox_flat[:, 0]
    y_idx = vox_flat[:, 1]
    x_idx = vox_flat[:, 2]

    # 归一化到 [-1,1]，注意 grid_sample 期望的顺序是 (x_norm, y_norm, z_norm)
    def _normalize(idx, size):
        return 2.0 * (idx / max(size - 1, 1)) - 1.0

    x_norm = _normalize(x_idx, W_vol)
    y_norm = _normalize(y_idx, H_vol)
    z_norm = _normalize(z_idx, D)

    grid = torch.stack([x_norm, y_norm, z_norm], dim=-1)  # (N,3)
    grid = grid.view(n_steps, H_img, W_img, 3)  # (n_steps,H_img,W_img,3)

    # 使用 grid_sample 进行三线性插值
    # grid_sample 输入为 (N,C,D,H,W) 和 (N,D_out,H_out,W_out,3)
    vals = F.grid_sample(
        vol,
        grid.unsqueeze(0),  # (1,n_steps,H,W,3)
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )  # (1,1,n_steps,H,W)

    vals = vals[0, 0]  # (n_steps,H,W)

    # 简单积分：沿 t 方向求和，再乘以步长
    dt = (t_max - t_min) / max(n_steps - 1, 1)
    img = vals.sum(dim=0) * dt  # (H,W)
    return img


def render_drr_with_rays_torch(
    femur_vol_np: np.ndarray,
    femur_affine_np: np.ndarray,
    femur_R: np.ndarray,
    femur_t: np.ndarray,
    tibia_vol_np: Optional[np.ndarray],
    tibia_affine_np: Optional[np.ndarray],
    tibia_R: Optional[np.ndarray],
    tibia_t: Optional[np.ndarray],
    calib: Dict[str, float],
    img_size: int = 512,
    n_steps: int = 256,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    使用 GPU 上的射线积分生成 DRR。

    这是“正常”的物理投影版本：
      - 不做 2D 裁剪/缩放/连通域等后处理；
      - femur / tibia 各自在各自 CT 体积上做射线积分，然后在 2D 上相加；
      - 唯一的 2D 操作是全局 min-max 归一化到 [0,1]。

    为了减弱条纹（aliasing），我们不再用固定的 [0,2000] 深度范围，而是：
      - 利用 NIfTI affine + R,t 先在相机坐标下估计一遍 femur/tibia 的 z 范围；
      - 只在这个 z 区间附近（加一点 margin）做积分，从而缩短射线路径、减小步长 dt。
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    H = W = img_size

    # 准备体积
    femur_vol = _volume_to_torch(femur_vol_np, device)
    tibia_vol = None
    has_tibia = (
        tibia_vol_np is not None
        and tibia_affine_np is not None
        and tibia_R is not None
        and tibia_t is not None
    )
    if has_tibia:
        tibia_vol = _volume_to_torch(tibia_vol_np, device)

    # === 1) 在相机坐标系下估计骨的 z 范围，用于设置深度积分区间 ===
    def _bone_z_range(
        vol_np: np.ndarray,
        affine_np: np.ndarray,
        R_np: np.ndarray,
        t_np: np.ndarray,
    ) -> tuple[float, float]:
        D_b, H_b, W_b = vol_np.shape
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
        A_b = affine_np.astype(np.float32)
        T_b = np.eye(4, dtype=np.float32)
        T_b[:3, :3] = R_np.astype(np.float32)
        T_b[:3, 3] = t_np.astype(np.float32)

        z_vals = []
        for (i, j, k) in corners:
            v = np.array([i, j, k, 1.0], dtype=np.float32)
            x_ct = A_b @ v
            x_cam = T_b @ x_ct
            z_vals.append(float(x_cam[2]))
        z_vals = np.array(z_vals, dtype=np.float32)
        return float(z_vals.min()), float(z_vals.max())

    zmins = []
    zmaxs = []
    zmin_f, zmax_f = _bone_z_range(femur_vol_np, femur_affine_np, femur_R, femur_t)
    zmins.append(zmin_f)
    zmaxs.append(zmax_f)
    if has_tibia:
        zmin_t, zmax_t = _bone_z_range(tibia_vol_np, tibia_affine_np, tibia_R, tibia_t)
        zmins.append(zmin_t)
        zmaxs.append(zmax_t)

    zmin_all = min(zmins)
    zmax_all = max(zmaxs)

    # 典型情况下 z 是负的（相机在原点，骨在相机前方 ≈ -800mm 左右），
    # 我们只在 [zmin_all, zmax_all] 附近积分，并加一点 margin。
    margin_mm = 50.0
    # 对应中心射线 d≈(0,0,-1) 时，t ≈ -z
    t_near = max(0.0, -zmax_all - margin_mm)  # 较近的一侧
    t_far = -zmin_all + margin_mm             # 较远的一侧
    if t_far <= t_near + 1e-3:
        t_near = 0.0
        t_far = max(1.0, 2.0 * abs(zmin_all - zmax_all))

    t_min = float(t_near)
    t_max = float(t_far)

    # === 2) 射线方向（在最终分辨率平面上定义） ===
    dirs = _compute_rays(H, W, calib, device)  # (H,W,3)

    # === 3) femur 贡献 ===
    img_femur = _sample_along_rays_single_bone(
        femur_vol,
        femur_affine_np,
        femur_R,
        femur_t,
        dirs,
        n_steps=n_steps,
        t_min=t_min,
        t_max=t_max,
    )

    img_total = img_femur

    # === 4) tibia 贡献（如果有的话） ===
    if tibia_vol is not None:
        img_tibia = _sample_along_rays_single_bone(
            tibia_vol,
            tibia_affine_np,
            tibia_R,
            tibia_t,
            dirs,
            n_steps=n_steps,
            t_min=t_min,
            t_max=t_max,
        )
        img_total = img_total + img_tibia

    # === 5) 仅做一次全局归一化到 [0,1]，不再做任何 2D 裁剪/掩膜等后处理 ===
    img_np = img_total.detach().cpu().numpy().astype(np.float32)  # (H,W)
    vmin, vmax = img_np.min(), img_np.max()
    if vmax > vmin:
        img_np = (img_np - vmin) / (vmax - vmin)
    else:
        img_np = np.zeros_like(img_np, dtype=np.float32)

    return img_np.astype(np.float32)


