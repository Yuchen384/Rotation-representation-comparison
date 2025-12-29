from pathlib import Path
from typing import Tuple, Optional

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform


def load_volume(nifti_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a NIfTI volume and its affine.

    Returns:
        vol: (D, H, W) float32 array
        affine: (4, 4) float32 NIfTI affine mapping voxel -> world (mm)
    """
    img = nib.load(str(nifti_path))
    vol = img.get_fdata().astype(np.float32)
    affine = img.affine.astype(np.float32)
    return vol, affine


def crop_volume_around_bone(
    vol: np.ndarray, affine: np.ndarray, pad: int = 32
) -> Tuple[np.ndarray, np.ndarray]:
    """
    对单个体积（femur 或 tibia）找到非零区域的 bounding box，裁剪并在周围 padding。
    同时更新 affine 以保持 voxel->world 映射正确。
    """
    assert vol.ndim == 3
    mask = vol > 0
    if not mask.any():
        # 没有明显骨头，直接返回零体积和原 affine
        return np.zeros_like(vol, dtype=np.float32), affine.copy()

    coords = np.argwhere(mask)
    zmin, ymin, xmin = coords.min(axis=0)
    zmax, ymax, xmax = coords.max(axis=0) + 1

    crop = vol[zmin:zmax, ymin:ymax, xmin:xmax]

    pad_spec = ((pad, pad), (pad, pad), (pad, pad))
    vol_padded = np.pad(crop, pad_spec, mode="constant", constant_values=0.0)

    # 更新 affine：新体素坐标 (i', j', k') 映射回原体素 (i, j, k)
    # i = i' - pad + zmin, j = j' - pad + ymin, k = k' - pad + xmin
    offset_mat = np.array(
        [
            [1, 0, 0, zmin - pad],
            [0, 1, 0, ymin - pad],
            [0, 0, 1, xmin - pad],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    new_affine = affine @ offset_mat

    # 归一化到 [0, 1]
    vmin, vmax = vol_padded.min(), vol_padded.max()
    if vmax > vmin:
        vol_padded = (vol_padded - vmin) / (vmax - vmin)
    else:
        vol_padded = np.zeros_like(vol_padded, dtype=np.float32)

    return vol_padded.astype(np.float32), new_affine


def volume_to_drr(volume: np.ndarray) -> np.ndarray:
    """
    简化版 DRR：沿深度轴做积分，然后在 [0,1] 范围内归一化。

    Args:
        volume: (D, H, W) 场景体积，通常已经在世界/相机坐标系下对齐好。
    Returns:
        drr: (H, W) 投影图像，float32 in [0, 1]
    """
    assert volume.ndim == 3

    drr = volume.sum(axis=0)
    dmin, dmax = drr.min(), drr.max()
    if dmax > dmin:
        drr = (drr - dmin) / (dmax - dmin)
    else:
        drr = np.zeros_like(drr, dtype=np.float32)
    return drr.astype(np.float32)


