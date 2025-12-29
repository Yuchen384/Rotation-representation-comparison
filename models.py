from typing import Literal, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import models

from .representations import (
    euler_xyz_to_matrix,
    quat_to_matrix,
    cont6d_to_matrix,
    axis_angle_to_matrix,
    geodesic_distance_from_matrices,
)


RotationRepr = Literal["euler", "quat", "6d", "axis_angle"]


class DRRBackbone(nn.Module):
    """
    Simple CNN backbone for DRR images.
    Uses a pretrained ResNet18 with adjusted first conv layer for single-channel input.
    """

    def __init__(self, out_dim: int):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Adapt first conv to 1-channel inputs
        conv1_weight = resnet.conv1.weight
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            resnet.conv1.weight[:] = conv1_weight.mean(dim=1, keepdim=True)

        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # remove FC
        self.fc = nn.Linear(resnet.fc.in_features, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.feature_extractor(x)
        feats = feats.flatten(1)
        out = self.fc(feats)
        return out


class RotationRegressionModule(pl.LightningModule):
    """
    LightningModule for DRR-based rotation regression with configurable representation.
    """

    def __init__(
        self,
        rotation_repr: RotationRepr = "6d",
        lr: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.rotation_repr = rotation_repr
        # Axis-angle specific hyperparameters (do NOT affect other representations)
        self.max_axis_angle_rad = math.pi  # clamp to at most 180 deg
        self.axis_angle_reg_weight = 1e-4  # very light L2 regularization on angle norm

        # 每种旋转表示对单块骨头的参数维度
        if rotation_repr == "euler":
            base_dim = 3
        elif rotation_repr == "quat":
            base_dim = 4
        elif rotation_repr == "6d":
            base_dim = 6
        elif rotation_repr == "axis_angle":
            # 3D rotation vector: direction = axis, norm = angle (rad)
            base_dim = 3
        else:
            raise ValueError(f"Unsupported rotation representation: {rotation_repr}")

        # 同时回归 femur + tibia：输出维度 = 2 * base_dim
        self.base_dim = base_dim
        out_dim = 2 * base_dim

        self.backbone = DRRBackbone(out_dim=out_dim)
        self.lr = lr

    def _pred_to_rotmat(self, pred: torch.Tensor) -> torch.Tensor:
        if self.rotation_repr == "euler":
            return euler_xyz_to_matrix(pred)
        elif self.rotation_repr == "quat":
            # Normalize quaternions before conversion
            pred_norm = pred / pred.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            return quat_to_matrix(pred_norm)
        elif self.rotation_repr == "6d":
            return cont6d_to_matrix(pred)
        elif self.rotation_repr == "axis_angle":
            # Directly interpret network outputs as rotation vectors, but
            # softly clamp the rotation angle to avoid extremely large norms.
            angle = torch.linalg.norm(pred, dim=-1, keepdim=True)  # (..., 1)
            max_angle = self.max_axis_angle_rad
            # scale <= 1 when angle > max_angle, otherwise = 1
            scale = (max_angle / angle.clamp_min(1e-8)).clamp_max(1.0)
            pred_clamped = pred * scale
            return axis_angle_to_matrix(pred_clamped)
        else:
            raise RuntimeError("Invalid rotation representation")

    def _split_params(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        通过 backbone 预测两块骨头的旋转参数，并按 femur / tibia 拆分。

        Returns:
            params_femur: (B, base_dim)
            params_tibia: (B, base_dim)
        """
        params = self.backbone(img)  # (B, 2*base_dim)
        params_femur, params_tibia = torch.split(params, self.base_dim, dim=-1)
        return params_femur, params_tibia

    def forward(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：返回 femur 与 tibia 的旋转矩阵。
        """
        params_femur, params_tibia = self._split_params(img)
        rot_femur = self._pred_to_rotmat(params_femur)
        rot_tibia = self._pred_to_rotmat(params_tibia)
        return rot_femur, rot_tibia

    def training_step(self, batch, batch_idx):
        img = batch["image"]

        # GT: 默认使用 femur_f/tibia_f 字段，向后兼容 rotmat / t 作为 femur
        rotmat_femur_gt = batch.get("rotmat_femur", batch.get("rotmat"))
        rotmat_tibia_gt = batch.get("rotmat_tibia", None)

        params_femur, params_tibia = self._split_params(img)
        rotmat_femur_pred = self._pred_to_rotmat(params_femur)
        rotmat_tibia_pred = self._pred_to_rotmat(params_tibia)

        geodesic_femur = geodesic_distance_from_matrices(rotmat_femur_pred, rotmat_femur_gt)

        if rotmat_tibia_gt is not None:
            geodesic_tibia = geodesic_distance_from_matrices(rotmat_tibia_pred, rotmat_tibia_gt)
            geodesic = 0.5 * (geodesic_femur + geodesic_tibia)
        else:
            geodesic = geodesic_femur

        loss = geodesic.mean()

        # 对 axis-angle：对两块骨头的旋转向量做一个很轻的 L2 正则
        if self.rotation_repr == "axis_angle":
            angle_norm_f = torch.linalg.norm(params_femur, dim=-1)  # (B,)
            angle_norm_t = torch.linalg.norm(params_tibia, dim=-1)  # (B,)
            angle_norm = 0.5 * (angle_norm_f + angle_norm_t)
            reg = self.axis_angle_reg_weight * angle_norm.mean()
            loss = loss + reg

        self.log("train/loss", loss)
        self.log("train/geodesic_deg", geodesic.mean() * 180.0 / 3.14159265)
        self.log("train/geodesic_femur_deg", geodesic_femur.mean() * 180.0 / 3.14159265)
        if rotmat_tibia_gt is not None:
            self.log("train/geodesic_tibia_deg", geodesic_tibia.mean() * 180.0 / 3.14159265)
        return loss

    def validation_step(self, batch, batch_idx):
        img = batch["image"]
        rotmat_femur_gt = batch.get("rotmat_femur", batch.get("rotmat"))
        rotmat_tibia_gt = batch.get("rotmat_tibia", None)

        params_femur, params_tibia = self._split_params(img)
        rotmat_femur_pred = self._pred_to_rotmat(params_femur)
        rotmat_tibia_pred = self._pred_to_rotmat(params_tibia)

        geodesic_femur = geodesic_distance_from_matrices(rotmat_femur_pred, rotmat_femur_gt)

        if rotmat_tibia_gt is not None:
            geodesic_tibia = geodesic_distance_from_matrices(rotmat_tibia_pred, rotmat_tibia_gt)
            geodesic = 0.5 * (geodesic_femur + geodesic_tibia)
        else:
            geodesic = geodesic_femur

        loss = geodesic.mean()

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/geodesic_deg", geodesic.mean() * 180.0 / 3.14159265, prog_bar=True)
        self.log("val/geodesic_femur_deg", geodesic_femur.mean() * 180.0 / 3.14159265)
        if rotmat_tibia_gt is not None:
            self.log("val/geodesic_tibia_deg", geodesic_tibia.mean() * 180.0 / 3.14159265)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


