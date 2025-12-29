from typing import Dict

import torch

from .representations import geodesic_distance_from_matrices, project_points


def rotation_metrics(rotmat_pred: torch.Tensor, rotmat_gt: torch.Tensor) -> Dict[str, torch.Tensor]:
    geodesic = geodesic_distance_from_matrices(rotmat_pred, rotmat_gt)
    return {
        "geodesic_rad": geodesic,
        "geodesic_deg": geodesic * 180.0 / 3.14159265,
    }


def projection_error(
    K: torch.Tensor,
    R_pred: torch.Tensor,
    t_pred: torch.Tensor,
    R_gt: torch.Tensor,
    t_gt: torch.Tensor,
    pts_3d: torch.Tensor,
) -> torch.Tensor:
    """
    Compute mean 2D projection error between predicted and GT camera poses.

    Returns:
        (...,) tensor of mean pixel error per sample.
    """
    pts_pred = project_points(K, R_pred, t_pred, pts_3d)
    pts_gt = project_points(K, R_gt, t_gt, pts_3d)
    err = torch.linalg.norm(pts_pred - pts_gt, dim=-1).mean(dim=-1)
    return err


