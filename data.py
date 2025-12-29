from pathlib import Path
from typing import Optional, Tuple, List

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class DRRRotationDataset(Dataset):
    """
    Simple dataset for DRR-based rotation regression.

    Expected directory structure (can be adapted to your own):

    data_root/
      images/
        sample_000.npy or .nii/.nii.gz
      poses.npy  # shape: (N, 7) [qx, qy, qz, qw, tx, ty, tz] or similar

    For now we assume there is a numpy file with DRR images and a numpy file with
    corresponding ground truth rotations in rotation-matrix form or quaternions.
    You can adapt `_load_image` and `__getitem__` to your actual data format.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        rotation_repr: str = "6d",
        transform: Optional[torch.nn.Module] = None,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split
        self.rotation_repr = rotation_repr.lower()
        self.transform = transform

        # 现在默认使用 prepare_data.py 生成的 images_*.npy 和 poses_*.npy：
        # poses_: (N, 12) 或 (N, 24)，格式为：
        #   - (12): [femur_tx, femur_ty, femur_tz, femur_R(9)]
        #   - (24): [femur_t(3), femur_R(9), tibia_t(3), tibia_R(9)]
        self.images_path = self.data_root / f"images_{split}.npy"
        self.poses_path = self.data_root / f"poses_{split}.npy"

        if not self.images_path.exists() or not self.poses_path.exists():
            raise FileNotFoundError(
                f"Expected {self.images_path} and {self.poses_path} to exist. "
                "Please prepare DRR image and pose numpy arrays."
            )

        self.images = np.load(self.images_path)  # (N, H, W) or (N, 1, H, W)
        self.poses = np.load(self.poses_path)    # (N, 12) or (N, 24)

        if self.images.ndim == 3:
            self.images = self.images[:, None, ...]  # add channel dimension

        assert self.images.shape[0] == self.poses.shape[0], "Mismatched samples between images and poses."

        # 记录 pose 维度，用于区分是否包含 tibia 姿态
        if self.poses.ndim != 2:
            raise ValueError(f"Expected poses to be 2D array, got shape {self.poses.shape}")
        self.pose_dim = self.poses.shape[1]
        if self.pose_dim not in (12, 24):
            raise ValueError(
                f"Unsupported pose dimensionality {self.pose_dim}. "
                "Expected 12 (femur only) or 24 (femur + tibia)."
            )

        if self.transform is None:
            self.transform = T.Compose(
                [
                    T.ConvertImageDtype(torch.float32),
                    T.Normalize(mean=[0.5], std=[0.5]),
                ]
            )

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int):
        img = torch.from_numpy(self.images[idx])
        pose = self.poses[idx]

        # ---------------- femur 姿态 ----------------
        # pose 前 12 维：femur_t(3) + femur_R(9)
        t_femur = torch.from_numpy(pose[0:3].astype(np.float32))
        rot_flat_femur = pose[3:12].astype(np.float32)
        rotmat_femur = torch.from_numpy(rot_flat_femur.reshape(3, 3))

        # ---------------- tibia 姿态（如存在） ----------------
        if self.pose_dim == 24:
            t_tibia = torch.from_numpy(pose[12:15].astype(np.float32))
            rot_flat_tibia = pose[15:24].astype(np.float32)
            rotmat_tibia = torch.from_numpy(rot_flat_tibia.reshape(3, 3))
        else:
            # 若数据集中没有 tibia，使用占位但仍提供字段，方便下游代码统一处理
            t_tibia = torch.zeros_like(t_femur)
            rotmat_tibia = torch.eye(3, dtype=torch.float32)

        img = self.transform(img)

        # Target representation 将在 LightningModule 中转换。
        # 这里同时返回 femur 和 tibia 的 GT，并保留 femur 的别名 rotmat / t
        sample = {
            "image": img,
            # femur
            "rotmat_femur": rotmat_femur,
            "t_femur": t_femur,
            # tibia
            "rotmat_tibia": rotmat_tibia,
            "t_tibia": t_tibia,
            # 向后兼容：保留 femur 作为通用 rotmat / t
            "rotmat": rotmat_femur,
            "t": t_femur,
        }
        return sample


def build_dataloaders(
    data_root: str,
    rotation_repr: str,
    batch_size: int = 16,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = DRRRotationDataset(data_root=data_root, split="train", rotation_repr=rotation_repr)
    val_ds = DRRRotationDataset(data_root=data_root, split="val", rotation_repr=rotation_repr)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


