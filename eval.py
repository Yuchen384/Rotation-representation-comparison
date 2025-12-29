import argparse
from pathlib import Path
from typing import List

import pytorch_lightning as pl
import torch
from tqdm import tqdm

from .data import build_dataloaders
from .models import RotationRegressionModule
from .metrics import rotation_metrics, projection_error


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained models for different rotation representations.")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument(
        "--rotation-reprs",
        type=str,
        nargs="+",
        default=["euler", "quat", "6d"],
        help="List of rotation representations to evaluate.",
    )
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gpus", type=int, default=1)
    return parser.parse_args()


def evaluate_single_repr(
    rotation_repr: str,
    ckpt_path: Path,
    data_root: str,
    batch_size: int,
    num_workers: int,
    gpus: int,
):
    print(f"Evaluating {rotation_repr} from checkpoint: {ckpt_path}")
    model = RotationRegressionModule.load_from_checkpoint(str(ckpt_path))
    model.eval()
    model.freeze()

    _, val_loader = build_dataloaders(
        data_root=data_root,
        rotation_repr=rotation_repr,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    device = torch.device("cuda" if gpus > 0 and torch.cuda.is_available() else "cpu")
    model.to(device)

    # 累积 femur / tibia 以及整体 pair 的误差，方便分别 / 联合统计
    all_geodesic_deg_femur = []
    all_proj_err_femur = []
    all_geodesic_deg_tibia = []
    all_proj_err_tibia = []
    all_geodesic_deg_overall = []
    all_proj_err_overall = []

    # 构造一个简单的 3D 点集（单位立方体的 8 个角点），用于 2D 投影误差评估
    cube_pts = torch.tensor(
        [
            [-0.5, -0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, 0.5, 0.5],
            [0.5, -0.5, -0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, -0.5],
            [0.5, 0.5, 0.5],
        ],
        dtype=torch.float32,
        device=device,
    )  # (8,3)

    # 构造一个简单的相机内参矩阵 K（焦距为 1，主点在 (0,0)），
    # 用于把“角度误差”映射到 2D 像素平面的位移误差。
    K = torch.eye(3, device=device, dtype=torch.float32)[None, :, :]  # (1,3,3)

    # 使用 tqdm 包装验证集迭代，显示评估进度条
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Eval {rotation_repr}", leave=False):
            img = batch["image"].to(device)
            # femur GT（也保留了兼容别名 rotmat / t）
            rotmat_femur_gt = batch.get("rotmat_femur", batch["rotmat"]).to(device)
            t_femur_gt = batch.get("t_femur", batch.get("t", None))
            if t_femur_gt is None:
                t_femur_gt = torch.zeros(rotmat_femur_gt.shape[0], 3, device=device, dtype=torch.float32)
            else:
                t_femur_gt = t_femur_gt.to(device)

            # tibia GT（如果数据里没有，就跳过 tibia 的指标）
            has_tibia = ("rotmat_tibia" in batch)
            if has_tibia:
                rotmat_tibia_gt = batch["rotmat_tibia"].to(device)
                t_tibia_gt = batch.get("t_tibia", None)
                if t_tibia_gt is None:
                    t_tibia_gt = torch.zeros(rotmat_tibia_gt.shape[0], 3, device=device, dtype=torch.float32)
                else:
                    t_tibia_gt = t_tibia_gt.to(device)

            # 使用 LightningModule 的前向接口，获取 femur/tibia 预测
            rotmat_femur_pred, rotmat_tibia_pred = model(img)

            # ===== femur 指标 =====
            mets_f = rotation_metrics(rotmat_femur_pred, rotmat_femur_gt)
            all_geodesic_deg_femur.append(mets_f["geodesic_deg"].cpu())

            bs = rotmat_femur_gt.shape[0]
            pts_3d = cube_pts.unsqueeze(0).expand(bs, -1, -1)  # (B,8,3)
            K_batch = K.expand(bs, -1, -1)                     # (B,3,3)
            proj_err_f = projection_error(
                K_batch,
                R_pred=rotmat_femur_pred,
                t_pred=t_femur_gt,
                R_gt=rotmat_femur_gt,
                t_gt=t_femur_gt,
                pts_3d=pts_3d,
            )  # (B,)
            all_proj_err_femur.append(proj_err_f.cpu())

            # ===== tibia 指标（如果存在）=====
            if has_tibia:
                mets_t = rotation_metrics(rotmat_tibia_pred, rotmat_tibia_gt)
                all_geodesic_deg_tibia.append(mets_t["geodesic_deg"].cpu())

                bs_t = rotmat_tibia_gt.shape[0]
                pts_3d_t = cube_pts.unsqueeze(0).expand(bs_t, -1, -1)
                K_batch_t = K.expand(bs_t, -1, -1)
                proj_err_t = projection_error(
                    K_batch_t,
                    R_pred=rotmat_tibia_pred,
                    t_pred=t_tibia_gt,
                    R_gt=rotmat_tibia_gt,
                    t_gt=t_tibia_gt,
                    pts_3d=pts_3d_t,
                )
                all_proj_err_tibia.append(proj_err_t.cpu())

                # ===== overall: 在 SO(3)xSO(3) 和投影误差上的合并 =====
                # 角度：使用积空间上的欧氏范数 sqrt(df^2 + dt^2)
                geo_pair = torch.sqrt(mets_f["geodesic_deg"] ** 2 + mets_t["geodesic_deg"] ** 2)
                all_geodesic_deg_overall.append(geo_pair.cpu())

                # 投影误差：使用 RMS 合并 sqrt( (ef^2 + et^2) / 2 )
                proj_pair = torch.sqrt(0.5 * (proj_err_f ** 2 + proj_err_t ** 2))
                all_proj_err_overall.append(proj_pair.cpu())

    # ===== femur 聚合 =====
    all_geodesic_deg_femur = torch.cat(all_geodesic_deg_femur, dim=0)
    all_proj_err_femur = torch.cat(all_proj_err_femur, dim=0)

    mean_deg_f = all_geodesic_deg_femur.mean().item()
    std_deg_f = all_geodesic_deg_femur.std().item()
    mean_proj_f = all_proj_err_femur.mean().item()
    std_proj_f = all_proj_err_femur.std().item()

    print(
        f"{rotation_repr} (femur): geodesic_deg mean={mean_deg_f:.3f}, std={std_deg_f:.3f}; "
        f"proj_err (px) mean={mean_proj_f:.3f}, std={std_proj_f:.3f}"
    )

    # ===== tibia 聚合（如果有）=====
    if len(all_geodesic_deg_tibia) > 0:
        all_geodesic_deg_tibia = torch.cat(all_geodesic_deg_tibia, dim=0)
        all_proj_err_tibia = torch.cat(all_proj_err_tibia, dim=0)

        mean_deg_t = all_geodesic_deg_tibia.mean().item()
        std_deg_t = all_geodesic_deg_tibia.std().item()
        mean_proj_t = all_proj_err_tibia.mean().item()
        std_proj_t = all_proj_err_tibia.std().item()

        print(
            f"{rotation_repr} (tibia): geodesic_deg mean={mean_deg_t:.3f}, std={std_deg_t:.3f}; "
            f"proj_err (px) mean={mean_proj_t:.3f}, std={std_proj_t:.3f}"
        )

    # ===== overall pair 聚合（如果有 tibia）=====
    if len(all_geodesic_deg_overall) > 0:
        all_geodesic_deg_overall = torch.cat(all_geodesic_deg_overall, dim=0)
        all_proj_err_overall = torch.cat(all_proj_err_overall, dim=0)

        mean_deg_o = all_geodesic_deg_overall.mean().item()
        std_deg_o = all_geodesic_deg_overall.std().item()
        mean_proj_o = all_proj_err_overall.mean().item()
        std_proj_o = all_proj_err_overall.std().item()

        print(
            f"{rotation_repr} (overall pair): geodesic_deg mean={mean_deg_o:.3f}, std={std_deg_o:.3f}; "
            f"proj_err (px) mean={mean_proj_o:.3f}, std={std_proj_o:.3f}"
        )

    # 为了跟之前的结果格式兼容，这里仍然返回 femur 的统计量
    return mean_deg_f, std_deg_f, mean_proj_f, std_proj_f


def main():
    args = parse_args()
    pl.seed_everything(42, workers=True)

    results = {}
    for repr_name in args.rotation_reprs:
        repr_dir = Path(args.checkpoints_dir) / repr_name
        if not repr_dir.exists():
            print(f"Checkpoint directory {repr_dir} does not exist, skipping {repr_name}.")
            continue

        ckpts = sorted(
            repr_dir.glob("*.ckpt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not ckpts:
            print(f"No checkpoints found in {repr_dir}, skipping {repr_name}.")
            continue

        # Prefer the most recently modified checkpoint (usually latest training run)
        ckpt_path = ckpts[0]
        mean_deg, std_deg, mean_proj, std_proj = evaluate_single_repr(
            rotation_repr=repr_name,
            ckpt_path=ckpt_path,
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            gpus=args.gpus,
        )
        results[repr_name] = (mean_deg, std_deg, mean_proj, std_proj)

    print("==== Summary ====")
    for k, (m, s, mp, sp) in results.items():
        print(
            f"{k}: geodesic mean={m:.3f} deg, std={s:.3f} deg; "
            f"proj_err mean={mp:.3f} px, std={sp:.3f} px"
        )


if __name__ == "__main__":
    main()


