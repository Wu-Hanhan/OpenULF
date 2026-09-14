# -*- coding: utf-8 -*-
"""
train.py — Production Training for ULF-MRI Reconstruction
===========================================================

Trains U-Net to reconstruct clean MRI images from noisy sensor data.

Experiments:
  A. squid  — SQUID receiver input → ideal image
  B. coil   — Classical coil input → ideal image
  C. both   — 2-channel (SQUID+Coil) → ideal image

Features:
  - Checkpoint resume (survives SLURM preemption)
  - SIGTERM handler for graceful preemption
  - Mixed precision (AMP) for speed
  - Gradient clipping
  - Reproducible seeds
  - Rich per-epoch logging (CSV + stdout)
  - Incremental log writes (no data loss on crash)

Usage:
    python train.py --experiment squid --epochs 100
    python train.py --experiment coil  --epochs 100
    python train.py --experiment both  --epochs 100
    python train.py --resume checkpoints/squid/checkpoint.pt
"""
from __future__ import annotations
import argparse
import json
import os
import random
import signal
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from dataset import ULFMRIDataset
from dataset_accelerated import AcceleratedMRIDataset
from dataset_mixed import GradualFusionDataset, MixedSingleChannelDataset
from model import UNet, DnCNN, SimpleCNN, ResUNet, NAFNet, CrossFusionNet, PoEFusionVAE


# ===================================================================
# Reproducibility
# ===================================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ===================================================================
# Metrics
# ===================================================================
def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Peak Signal-to-Noise Ratio [dB]. Images assumed in [0, 1]."""
    mse = torch.mean((pred - target) ** 2).item()
    if mse < 1e-10:
        return 50.0
    return 10 * np.log10(1.0 / mse)


def ssim_simple(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Simplified SSIM (global statistics, no windowing).

    Fast approximation for training monitoring.
    For final evaluation, use skimage.metrics.structural_similarity.
    """
    pred = pred.float().clamp(0, 1)
    target = target.float().clamp(0, 1)
    mu_p = pred.mean()
    mu_t = target.mean()
    sigma_p = pred.std()
    sigma_t = target.std()
    sigma_pt = ((pred - mu_p) * (target - mu_t)).mean()
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    num = (2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2)
    den = (mu_p**2 + mu_t**2 + C1) * (sigma_p**2 + sigma_t**2 + C2)
    return (num / den).item()


def nmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Normalized Mean Squared Error."""
    return (torch.sum((pred - target) ** 2) /
            torch.sum(target ** 2)).item()


# ===================================================================
# Training Loop
# ===================================================================
def train_one_epoch(model, loader, optimizer, criterion, scaler, device, epoch=0, use_amp=True):
    model.train()
    total_loss = 0.0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda', dtype=torch.float16, enabled=use_amp):
            if isinstance(model, PoEFusionVAE):
                outputs, mu_f, logvar_f = model(inputs)
                recon_loss = criterion(outputs, targets)
                kl_loss = PoEFusionVAE.kl_divergence(mu_f, logvar_f)
                # β warmup: β_min → β_max over 30 epochs
                # β_min > 0 prevents latent drift during warmup;
                # β_max = 1e-3 gives ~5% KL/recon ratio at convergence
                beta = 1e-5 + (1e-3 - 1e-5) * min(epoch / 30.0, 1.0)
                loss = recon_loss + beta * kl_loss
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * inputs.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp=True):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_nmse = 0.0
    n = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast('cuda', dtype=torch.float16, enabled=use_amp):
            if isinstance(model, PoEFusionVAE):
                outputs, _, _ = model(inputs, deterministic=True)
            else:
                outputs = model(inputs)
            loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)

        # Metrics in float32
        outputs_f = outputs.float().clamp(0, 1)
        targets_f = targets.float()

        for i in range(inputs.size(0)):
            total_psnr += psnr(outputs_f[i], targets_f[i])
            total_ssim += ssim_simple(outputs_f[i], targets_f[i])
            total_nmse += nmse(outputs_f[i], targets_f[i])
            n += 1

    return {
        'loss': total_loss / len(loader.dataset),
        'psnr': total_psnr / n,
        'ssim': total_ssim / n,
        'nmse': total_nmse / n,
    }


# ===================================================================
# Checkpoint Management
# ===================================================================
def save_checkpoint(path, epoch, model, optimizer, scheduler, scaler,
                    best_val_loss, best_val_psnr):
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'best_val_loss': best_val_loss,
        'best_val_psnr': best_val_psnr,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    scaler.load_state_dict(ckpt['scaler'])
    return ckpt['epoch'], ckpt['best_val_loss'], ckpt.get('best_val_psnr', 0)


# ===================================================================
# Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description='ULF-MRI Reconstruction')
    parser.add_argument('--experiment', type=str, required=True,
                        choices=['squid', 'coil', 'both'],
                        help='Experiment type')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--loss', type=str, default='mse',
                        choices=['mse', 'l1'],
                        help='Loss function (mse used in all published experiments)')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--model', type=str, default='unet',
                        choices=['unet', 'dncnn', 'simplecnn', 'resunet', 'nafnet', 'crossfusion', 'poevae'],
                        help='Model architecture')
    parser.add_argument('--base_features', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate (0=off, 0.1-0.3 typical)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--data_dir', type=str, default='data/dataset')
    parser.add_argument('--out_dir', type=str, default='checkpoints')
    parser.add_argument('--acceleration', type=int, default=0,
                        help='Acceleration factor (0=off, 2/4/8=undersampled)')
    parser.add_argument('--fusion_fraction', type=float, default=-1,
                        help='Gradual fusion: fraction of samples with 2nd channel (0-1)')
    parser.add_argument('--fusion_base', type=str, default='squid',
                        choices=['squid', 'coil'],
                        help='Base sensor for gradual fusion')
    parser.add_argument('--mixed_squid_frac', type=float, default=-1,
                        help='Mixed single-channel: fraction of SQUID samples (0-1)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint.pt for resume')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable AMP (fp16) for numerical stability')
    args = parser.parse_args()

    # --- Reproducibility ---
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Device: {device} ({gpu_name}, {gpu_mem:.1f} GB)")
    else:
        print(f"Device: {device}")

    # --- Data ---
    input_type = args.experiment
    if args.fusion_fraction >= 0:
        # Gradual fusion mode
        train_ds = GradualFusionDataset(f'{args.data_dir}/train.h5',
                                         base=args.fusion_base,
                                         secondary_fraction=args.fusion_fraction,
                                         seed=args.seed, augment=True)
        val_ds = GradualFusionDataset(f'{args.data_dir}/val.h5',
                                       base=args.fusion_base,
                                       secondary_fraction=args.fusion_fraction,
                                       seed=args.seed, augment=False)
    elif args.mixed_squid_frac >= 0:
        # Mixed single-channel mode
        train_ds = MixedSingleChannelDataset(f'{args.data_dir}/train.h5',
                                              squid_fraction=args.mixed_squid_frac,
                                              seed=args.seed, augment=True)
        val_ds = MixedSingleChannelDataset(f'{args.data_dir}/val.h5',
                                            squid_fraction=args.mixed_squid_frac,
                                            seed=args.seed, augment=False)
    elif args.acceleration > 0:
        # Accelerated (undersampled) reconstruction mode
        train_ds = AcceleratedMRIDataset(f'{args.data_dir}/train.h5',
                                          input_type, args.acceleration, augment=True)
        val_ds = AcceleratedMRIDataset(f'{args.data_dir}/val.h5',
                                        input_type, args.acceleration, augment=False)
    else:
        train_ds = ULFMRIDataset(f'{args.data_dir}/train.h5', input_type, augment=True)
        val_ds = ULFMRIDataset(f'{args.data_dir}/val.h5', input_type, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, persistent_workers=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True, persistent_workers=True)

    # --- Model ---
    if args.fusion_fraction >= 0:
        in_ch = 2  # gradual fusion: always 2 channels (2nd may be zeros)
    elif args.mixed_squid_frac >= 0:
        in_ch = 1  # mixed single-channel
    elif args.acceleration > 0:
        in_ch = 2 if args.experiment == 'both' else 1
    else:
        in_ch = 2 if args.experiment == 'both' else 1
    if args.model == 'unet':
        model = UNet(in_channels=in_ch, out_channels=1,
                     base_features=args.base_features,
                     dropout=args.dropout).to(device)
    elif args.model == 'dncnn':
        model = DnCNN(in_channels=in_ch, out_channels=1,
                      num_layers=17, features=64).to(device)
    elif args.model == 'simplecnn':
        model = SimpleCNN(in_channels=in_ch, out_channels=1,
                          features=64, num_layers=8).to(device)
    elif args.model == 'resunet':
        model = ResUNet(in_channels=in_ch, out_channels=1,
                        base_features=args.base_features,
                        dropout=args.dropout).to(device)
    elif args.model == 'nafnet':
        # NAFNet at width=16 gives ~7.4M params (comparable to UNet at base=32)
        naf_width = args.base_features // 2 if args.base_features == 32 else args.base_features
        model = NAFNet(in_channels=in_ch, out_channels=1,
                       base_features=naf_width,
                       dropout=args.dropout).to(device)
    elif args.model == 'crossfusion':
        if in_ch != 2:
            raise ValueError("CrossFusionNet requires --experiment both (2-channel input)")
        model = CrossFusionNet(dropout=args.dropout).to(device)
    elif args.model == 'poevae':
        if in_ch != 2:
            raise ValueError("PoEFusionVAE requires --experiment both (2-channel input)")
        model = PoEFusionVAE(dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    # --- Optimizer + Scheduler + AMP ---
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.L1Loss() if args.loss == 'l1' else nn.MSELoss()
    use_amp = not args.no_amp
    scaler = GradScaler('cuda', enabled=use_amp)
    if not use_amp:
        print("  AMP disabled (fp32 training)")

    # --- Output directory ---
    exp_dir = os.path.join(args.out_dir, args.experiment)
    os.makedirs(exp_dir, exist_ok=True)

    # --- Resume from checkpoint ---
    start_epoch = 1
    best_val_loss = float('inf')
    best_val_psnr = 0.0

    ckpt_path = args.resume or os.path.join(exp_dir, 'checkpoint.pt')
    if os.path.exists(ckpt_path):
        print(f"Resuming from: {ckpt_path}")
        start_epoch_loaded, best_val_loss, best_val_psnr = load_checkpoint(
            ckpt_path, model, optimizer, scheduler, scaler, device)
        start_epoch = start_epoch_loaded + 1
        print(f"  Resumed at epoch {start_epoch}, best_loss={best_val_loss:.6f}, "
              f"best_psnr={best_val_psnr:.2f}")

    # --- SIGTERM handler (SLURM preemption) ---
    sigterm_received = [False]

    def sigterm_handler(signum, frame):
        print(f"\n  SIGTERM received — saving checkpoint and exiting...")
        sigterm_received[0] = True

    signal.signal(signal.SIGTERM, sigterm_handler)

    # --- Log file (append mode for resume) ---
    log_path = os.path.join(exp_dir, 'log.csv')
    log_header = "epoch,train_loss,val_loss,val_psnr,val_ssim,val_nmse,lr,time_s"
    if start_epoch == 1:
        with open(log_path, 'w') as f:
            f.write(log_header + '\n')

    # --- Save experiment config ---
    config = {k: v for k, v in vars(args).items()}
    config['n_params'] = n_params
    config['device'] = str(device)
    config['gpu'] = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'
    config['train_samples'] = len(train_ds)
    config['val_samples'] = len(val_ds)
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # --- Print header ---
    print(f"\n{'='*72}")
    print(f"  Experiment: {args.experiment} | {type(model).__name__}(in={in_ch}, base={args.base_features}) | "
          f"{n_params/1e6:.2f}M params")
    print(f"  Epochs: {start_epoch}-{args.epochs} | BS: {args.batch_size} | "
          f"LR: {args.lr} | Seed: {args.seed}")
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | AMP: {'ON' if use_amp else 'OFF'} | "
          f"Grad clip: 1.0 | WD: {args.weight_decay} | Augment: ON")
    print(f"  Early stopping: patience={args.patience}")
    print(f"{'='*72}")

    # --- Training loop ---
    total_train_time = 0.0
    epochs_without_improvement = 0
    for epoch in range(start_epoch, args.epochs + 1):
        if sigterm_received[0]:
            break

        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer,
                                     criterion, scaler, device, epoch=epoch, use_amp=use_amp)
        val = evaluate(model, val_loader, criterion, device, use_amp=use_amp)
        elapsed = time.time() - t0
        total_train_time += elapsed

        scheduler.step(val['loss'])
        lr = optimizer.param_groups[0]['lr']

        # Log to CSV (append, survives crash)
        with open(log_path, 'a') as f:
            f.write(f"{epoch},{train_loss:.6f},{val['loss']:.6f},"
                    f"{val['psnr']:.4f},{val['ssim']:.6f},{val['nmse']:.6f},"
                    f"{lr:.2e},{elapsed:.1f}\n")

        # Save best model + early stopping
        improved = ""
        if val['loss'] < best_val_loss:
            best_val_loss = val['loss']
            best_val_psnr = val['psnr']
            torch.save(model.state_dict(), os.path.join(exp_dir, 'best.pt'))
            improved = " *best*"
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Save checkpoint every 10 epochs and on last epoch
        if epoch % 10 == 0 or epoch == args.epochs:
            save_checkpoint(os.path.join(exp_dir, 'checkpoint.pt'),
                            epoch, model, optimizer, scheduler, scaler,
                            best_val_loss, best_val_psnr)

        # Progress
        eta = elapsed * (args.epochs - epoch)
        print(f"  [{epoch:3d}/{args.epochs}] "
              f"loss: {train_loss:.5f}/{val['loss']:.5f} "
              f"PSNR: {val['psnr']:.2f} SSIM: {val['ssim']:.4f} "
              f"NMSE: {val['nmse']:.4f} "
              f"lr={lr:.1e} {elapsed:.0f}s ETA={eta/60:.0f}m{improved}")

        if sigterm_received[0]:
            break

        # Early stopping
        if epochs_without_improvement >= args.patience:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(no improvement for {args.patience} epochs)")
            break

    # --- Handle SIGTERM: save and exit ---
    if sigterm_received[0]:
        save_checkpoint(os.path.join(exp_dir, 'checkpoint.pt'),
                        epoch, model, optimizer, scheduler, scaler,
                        best_val_loss, best_val_psnr)
        print(f"  Checkpoint saved at epoch {epoch}. Exiting for requeue.")
        sys.exit(0)

    # --- Final save ---
    torch.save(model.state_dict(), os.path.join(exp_dir, 'final.pt'))
    save_checkpoint(os.path.join(exp_dir, 'checkpoint.pt'),
                    epoch, model, optimizer, scheduler, scaler,
                    best_val_loss, best_val_psnr)

    print(f"\n{'='*72}")
    print(f"  Training complete: {args.experiment}")
    print(f"  Best val loss: {best_val_loss:.6f} | Best PSNR: {best_val_psnr:.2f}")
    print(f"  Total time: {total_train_time/60:.1f} min")
    print(f"  Output: {exp_dir}/")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
