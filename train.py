import argparse
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import CSVLogger

from .data import build_dataloaders
from .models import RotationRegressionModule


def parse_args():
    parser = argparse.ArgumentParser(description="Train DRR rotation regression with different representations.")
    parser.add_argument("--data-root", type=str, default="data", help="Path to data root directory.")
    parser.add_argument(
        "--rotation-repr",
        type=str,
        default="6d",
        choices=["euler", "quat", "6d", "axis_angle"],
        help="Rotation representation to train.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs (0 for CPU). Kept for backward compatibility.",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help="Comma-separated GPU indices, e.g. '0,1'. "
        "If set, overrides --gpus and enables multi-GPU parallel training.",
    )
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Directory to save checkpoints.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_dataloaders(
        data_root=args.data_root,
        rotation_repr=args.rotation_repr,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = RotationRegressionModule(rotation_repr=args.rotation_repr, lr=args.lr)

    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / args.rotation_repr,
            filename="{epoch:03d}-{val_loss:.4f}",
            save_top_k=3,
            monitor="val/loss",
            mode="min",
        ),
        LearningRateMonitor(logging_interval="epoch"),
        # 使用 Lightning 自带的 TQDM 进度条，显示每个 epoch 的 batch 进度
        TQDMProgressBar(refresh_rate=10),
    ]

    # ===== 设备与并行策略配置 =====
    if args.devices:
        # 解析逗号分隔的 GPU id 列表，例如 "5,6"
        device_ids = [int(x) for x in args.devices.split(",") if x.strip() != ""]
        accelerator = "gpu"
        devices = device_ids
    else:
        accelerator = "gpu" if args.gpus > 0 else "cpu"
        devices = args.gpus if args.gpus > 0 else 1

    # 简单选择策略：多 GPU 时用 DDP，单 GPU/CPU 用默认
    if (isinstance(devices, int) and devices > 1) or (
        isinstance(devices, (list, tuple)) and len(devices) > 1
    ):
        strategy = "ddp"
    else:
        strategy = "auto"

    # 使用 CSVLogger 记录每个 step/epoch 的 loss，方便后续画 loss 曲线
    logger = CSVLogger(
        save_dir=str(output_dir),
        name=args.rotation_repr,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        default_root_dir=str(output_dir),
        callbacks=callbacks,
        log_every_n_steps=10,
        logger=logger,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # ===== 训练结束后，根据 CSVLogger 生成一张 loss 曲线图 =====
    try:
        import pandas as pd
        import matplotlib.pyplot as plt

        metrics_path = Path(logger.log_dir) / "metrics.csv"
        if metrics_path.exists():
            df = pd.read_csv(metrics_path)

            plt.figure(figsize=(8, 5))
            if "train/loss" in df.columns:
                plt.plot(df["step"], df["train/loss"], label="train/loss", alpha=0.7)
            if "val/loss" in df.columns:
                # 按 epoch 对 val/loss 进行绘制
                plt.plot(df["epoch"], df["val/loss"], "o-", label="val/loss", alpha=0.9)

            plt.xlabel("step / epoch")
            plt.ylabel("loss")
            plt.title(f"Training/Validation Loss ({args.rotation_repr})")
            plt.legend()
            plt.grid(True, alpha=0.3)

            plot_path = Path(logger.log_dir) / "loss_curve.png"
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved loss curve to {plot_path}")
        else:
            print(f"metrics.csv not found at {metrics_path}, skip loss plotting.")
    except Exception as e:
        print(f"Failed to generate loss curve plot: {e}")


if __name__ == "__main__":
    main()


