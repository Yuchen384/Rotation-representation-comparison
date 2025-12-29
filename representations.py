import math
from typing import Tuple

import torch


def euler_xyz_to_matrix(euler: torch.Tensor) -> torch.Tensor:
    """
    Convert XYZ Euler angles (in radians) to a rotation matrix.

    Args:
        euler: (..., 3) tensor of angles (rx, ry, rz).

    Returns:
        (..., 3, 3) rotation matrices.
    """
    rx, ry, rz = euler.unbind(-1)
    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)

    rot_x = torch.stack(
        [
            torch.stack([torch.ones_like(cx), torch.zeros_like(cx), torch.zeros_like(cx)], dim=-1),
            torch.stack([torch.zeros_like(cx), cx, -sx], dim=-1),
            torch.stack([torch.zeros_like(cx), sx, cx], dim=-1),
        ],
        dim=-2,
    )
    rot_y = torch.stack(
        [
            torch.stack([cy, torch.zeros_like(cy), sy], dim=-1),
            torch.stack([torch.zeros_like(cy), torch.ones_like(cy), torch.zeros_like(cy)], dim=-1),
            torch.stack([-sy, torch.zeros_like(cy), cy], dim=-1),
        ],
        dim=-2,
    )
    rot_z = torch.stack(
        [
            torch.stack([cz, -sz, torch.zeros_like(cz)], dim=-1),
            torch.stack([sz, cz, torch.zeros_like(cz)], dim=-1),
            torch.stack([torch.zeros_like(cz), torch.zeros_like(cz), torch.ones_like(cz)], dim=-1),
        ],
        dim=-2,
    )

    return rot_z @ rot_y @ rot_x


def matrix_to_euler_xyz(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrices to XYZ Euler angles (radians).
    Uses a standard convention assuming no gimbal lock.
    """
    r11 = matrix[..., 0, 0]
    r21 = matrix[..., 1, 0]
    r31 = matrix[..., 2, 0]
    r32 = matrix[..., 2, 1]
    r33 = matrix[..., 2, 2]

    ry = torch.atan2(-r31, torch.sqrt(r11 ** 2 + r21 ** 2))
    rx = torch.atan2(r32, r33)
    rz = torch.atan2(r21, r11)

    return torch.stack([rx, ry, rz], dim=-1)


def quat_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert normalized quaternions to rotation matrices.

    Args:
        quat: (..., 4) in (w, x, y, z) format.
    """
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = quat.unbind(-1)

    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    m00 = ww + xx - yy - zz
    m01 = 2 * (xy - wz)
    m02 = 2 * (xz + wy)

    m10 = 2 * (xy + wz)
    m11 = ww - xx + yy - zz
    m12 = 2 * (yz - wx)

    m20 = 2 * (xz - wy)
    m21 = 2 * (yz + wx)
    m22 = ww - xx - yy + zz

    row0 = torch.stack([m00, m01, m02], dim=-1)
    row1 = torch.stack([m10, m11, m12], dim=-1)
    row2 = torch.stack([m20, m21, m22], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def matrix_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrices to quaternions (w, x, y, z).
    Implementation based on a standard numerically stable branch.
    """
    m = matrix
    batch_shape = m.shape[:-2]
    m00, m11, m22 = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    trace = m00 + m11 + m22

    qw = torch.empty(batch_shape, device=m.device, dtype=m.dtype)
    qx = torch.empty_like(qw)
    qy = torch.empty_like(qw)
    qz = torch.empty_like(qw)

    # Case 1: trace > 0
    mask1 = trace > 0.0
    s1 = torch.sqrt(trace[mask1] + 1.0) * 2.0
    qw[mask1] = 0.25 * s1
    qx[mask1] = (m[mask1, 2, 1] - m[mask1, 1, 2]) / s1
    qy[mask1] = (m[mask1, 0, 2] - m[mask1, 2, 0]) / s1
    qz[mask1] = (m[mask1, 1, 0] - m[mask1, 0, 1]) / s1

    # Case 2: m00 is largest
    mask2 = (~mask1) & (m00 >= m11) & (m00 >= m22)
    s2 = torch.sqrt(1.0 + m00[mask2] - m11[mask2] - m22[mask2]) * 2.0
    qw[mask2] = (m[mask2, 2, 1] - m[mask2, 1, 2]) / s2
    qx[mask2] = 0.25 * s2
    qy[mask2] = (m[mask2, 0, 1] + m[mask2, 1, 0]) / s2
    qz[mask2] = (m[mask2, 0, 2] + m[mask2, 2, 0]) / s2

    # Case 3: m11 is largest
    mask3 = (~mask1) & (~mask2) & (m11 >= m22)
    s3 = torch.sqrt(1.0 + m11[mask3] - m00[mask3] - m22[mask3]) * 2.0
    qw[mask3] = (m[mask3, 0, 2] - m[mask3, 2, 0]) / s3
    qx[mask3] = (m[mask3, 0, 1] + m[mask3, 1, 0]) / s3
    qy[mask3] = 0.25 * s3
    qz[mask3] = (m[mask3, 1, 2] + m[mask3, 2, 1]) / s3

    # Case 4: m22 is largest
    mask4 = (~mask1) & (~mask2) & (~mask3)
    s4 = torch.sqrt(1.0 + m22[mask4] - m00[mask4] - m11[mask4]) * 2.0
    qw[mask4] = (m[mask4, 1, 0] - m[mask4, 0, 1]) / s4
    qx[mask4] = (m[mask4, 0, 2] + m[mask4, 2, 0]) / s4
    qy[mask4] = (m[mask4, 1, 2] + m[mask4, 2, 1]) / s4
    qz[mask4] = 0.25 * s4

    quat = torch.stack([qw, qx, qy, qz], dim=-1)
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return quat


def cont6d_to_matrix(cont6d: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D continuous representation (Zhou et al., 2019) to rotation matrices.

    Args:
        cont6d: (..., 6) tensor.
    """
    a1 = cont6d[..., 0:3]
    a2 = cont6d[..., 3:6]

    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = torch.nn.functional.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack([b1, b2, b3], dim=-2)


def matrix_to_cont6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrices to 6D continuous representation by taking first two columns.
    """
    return torch.cat([matrix[..., :, 0], matrix[..., :, 1]], dim=-1)


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert axis-angle / rotation-vector representation to rotation matrices.

    Args:
        axis_angle: (..., 3) tensor, where the direction is the rotation axis
                    and the L2 norm is the rotation angle in radians.
    Returns:
        (..., 3, 3) rotation matrices.
    """
    # angle = ||v||, axis = v / ||v||
    angle = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)  # (..., 1)
    # To avoid division by zero, clamp the angle and use where later.
    eps = 1e-8
    axis = axis_angle / angle.clamp_min(eps)

    x, y, z = axis.unbind(-1)
    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)

    ca = torch.cos(angle.squeeze(-1))
    sa = torch.sin(angle.squeeze(-1))
    C = 1.0 - ca

    # Rodrigues' rotation formula components
    m00 = ca + x * x * C
    m01 = x * y * C - z * sa
    m02 = x * z * C + y * sa

    m10 = y * x * C + z * sa
    m11 = ca + y * y * C
    m12 = y * z * C - x * sa

    m20 = z * x * C - y * sa
    m21 = z * y * C + x * sa
    m22 = ca + z * z * C

    row0 = torch.stack([m00, m01, m02], dim=-1)
    row1 = torch.stack([m10, m11, m12], dim=-1)
    row2 = torch.stack([m20, m21, m22], dim=-1)
    rot = torch.stack([row0, row1, row2], dim=-2)

    # For very small angles, fall back to identity to avoid numerical noise.
    if (angle < eps).any():
        eye = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
        rot = torch.where(
            (angle < eps).view(*axis_angle.shape[:-1], 1, 1),
            eye.expand_as(rot),
            rot,
        )
    return rot


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrices to axis-angle / rotation-vector representation.

    Args:
        matrix: (..., 3, 3) rotation matrices.
    Returns:
        (..., 3) tensor v where direction is axis and norm is angle (radians).
    """
    # Based on standard SO(3) log map.
    # Compute angle from trace.
    trace = matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2]
    cos_theta = (trace - 1.0) * 0.5
    cos_theta = cos_theta.clamp(-1.0, 1.0)
    theta = torch.arccos(cos_theta)

    # For small angles, use first-order approximation: v ≈ 0.5 * (R - R^T)^\vee
    eps = 1e-6
    small = theta.abs() < eps

    vx = matrix[..., 2, 1] - matrix[..., 1, 2]
    vy = matrix[..., 0, 2] - matrix[..., 2, 0]
    vz = matrix[..., 1, 0] - matrix[..., 0, 1]

    # Avoid division by zero: for non-small angles, scale by theta / (2 sin theta)
    sin_theta = torch.sin(theta)
    scale = theta / (2.0 * sin_theta.clamp_min(1e-8))
    v = torch.stack([vx, vy, vz], dim=-1) * scale.unsqueeze(-1)

    # Small-angle branch: directly use 0.5 * (R - R^T)^\vee
    v_small = 0.5 * torch.stack([vx, vy, vz], dim=-1)

    v = torch.where(small.unsqueeze(-1), v_small, v)
    return v


def geodesic_distance_from_matrices(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute geodesic distance (in radians) between batches of rotation matrices.

    Args:
        pred: (..., 3, 3)
        target: (..., 3, 3)
    """
    r_rel = pred @ target.transpose(-1, -2)
    trace = r_rel[..., 0, 0] + r_rel[..., 1, 1] + r_rel[..., 2, 2]
    cos = (trace - 1.0) / 2.0
    cos = cos.clamp(-1.0, 1.0)
    return torch.arccos(cos)


def project_points(K: torch.Tensor, R: torch.Tensor, t: torch.Tensor, pts_3d: torch.Tensor) -> torch.Tensor:
    """
    Project 3D points to 2D using camera intrinsics K and extrinsics (R, t).

    Args:
        K: (..., 3, 3) intrinsics
        R: (..., 3, 3) rotation
        t: (..., 3) translation
        pts_3d: (..., N, 3) points
    Returns:
        (..., N, 2) image coordinates
    """
    pts = (R @ pts_3d.transpose(-1, -2) + t.unsqueeze(-1)).transpose(-1, -2)
    pts_cam = (K @ pts.transpose(-1, -2)).transpose(-1, -2)
    x = pts_cam[..., 0] / pts_cam[..., 2].clamp_min(1e-8)
    y = pts_cam[..., 1] / pts_cam[..., 2].clamp_min(1e-8)
    return torch.stack([x, y], dim=-1)


