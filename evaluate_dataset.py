# -*- coding: utf-8 -*-
"""
evaluate_dataset.py — Generalized Dataset Evaluation
=====================================================

Evaluates all trained models + traditional baselines on any dataset's test set.

Usage:
    python evaluate_dataset.py --data_dir data/dataset --results_dir results_final --out_dir analysis/brainweb
    python evaluate_dataset.py --data_dir data/dataset_abdomen --results_dir results_abdomen --out_dir analysis/abdomen
    python evaluate_dataset.py --data_dir data/dataset_oasis --results_dir results_oasis --out_dir analysis/oasis
"""
from __future__ import annotations
import os
import argparse
import numpy as np
import torch
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_skimage
from skimage.restoration import denoise_nl_means, denoise_tv_chambolle, estimate_sigma
from scipy.ndimage import gaussian_filter

from model import UNet, DnCNN, SimpleCNN, ResUNet
from dataset import ULFMRIDataset


def compute_metrics(preds, targets):
    """PSNR, SSIM, NMSE per sample."""
    psnrs, ssims, nmses = [], [], []
    for i in range(len(preds)):
        p, t = preds[i].clip(0, 1), targets[i]
        mse = np.mean((p - t) ** 2)
        psnrs.append(10 * np.log10(1.0 / max(mse, 1e-10)))
        ssims.append(ssim_skimage(t, p, data_range=1.0))
        nmses.append(np.sum((p - t) ** 2) / max(np.sum(t ** 2), 1e-10))
    return {k: np.array(v) for k, v in
            zip(['psnr', 'ssim', 'nmse'], [psnrs, ssims, nmses])}


@torch.no_grad()
def predict_ml(model, dataset, device, batch_size=32):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds = []
    for inputs, _ in loader:
        out = model(inputs.to(device)).cpu().float().clamp(0, 1)
        preds.append(out.numpy())
    return np.concatenate(preds, axis=0)[:, 0]


MODEL_CLASSES = {
    'unet': (UNet, {'base_features': 32}),
    'unet32': (UNet, {'base_features': 32}),
    'dncnn': (DnCNN, {'num_layers': 17, 'features': 64}),
    'simplecnn': (SimpleCNN, {}),
    'resunet': (ResUNet, {'base_features': 32}),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--results_dir', required=True)
    parser.add_argument('--out_dir', default='analysis/eval')
    parser.add_argument('--skip_traditional', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load test data
    test_path = os.path.join(args.data_dir, 'test.h5')
    with h5py.File(test_path, 'r') as f:
        targets = f['images/ideal'][:].astype(np.float32)
        squid_raw = f['images/squid'][:].astype(np.float32)
        coil_raw = f['images/coil'][:].astype(np.float32)

    N = len(targets)
    print(f"{'='*70}")
    print(f"  Evaluation: {N} test samples from {args.data_dir}")
    print(f"  Models from: {args.results_dir}")
    print(f"  Device: {device}")
    print(f"{'='*70}")

    all_results = {}

    # Raw baselines
    for name, raw in [('SQUID raw', squid_raw), ('Coil raw', coil_raw)]:
        all_results[name] = compute_metrics(raw, targets)
        m = all_results[name]
        print(f"  {name}: PSNR={m['psnr'].mean():.2f}±{m['psnr'].std():.2f}")

    # Traditional methods
    if not args.skip_traditional:
        print(f"\n  Traditional methods (on SQUID):")
        gauss = np.array([gaussian_filter(img, sigma=1.5) for img in squid_raw])
        all_results['SQUID+Gaussian'] = compute_metrics(gauss, targets)
        print(f"    Gaussian: {all_results['SQUID+Gaussian']['psnr'].mean():.2f}")

        tv = np.array([denoise_tv_chambolle(img, weight=0.15) for img in squid_raw])
        all_results['SQUID+TV'] = compute_metrics(tv, targets)
        print(f"    TV:       {all_results['SQUID+TV']['psnr'].mean():.2f}")

        nlm_imgs = []
        for img in squid_raw:
            sig = estimate_sigma(img)
            nlm_imgs.append(denoise_nl_means(img, h=1.2*sig, patch_size=5,
                                              patch_distance=6, fast_mode=True))
        nlm = np.array(nlm_imgs)
        all_results['SQUID+NLM'] = compute_metrics(nlm, targets)
        print(f"    NLM:      {all_results['SQUID+NLM']['psnr'].mean():.2f}")

        try:
            import bm3d
            bm3d_imgs = []
            for img in squid_raw:
                sig = estimate_sigma(img)
                bm3d_imgs.append(bm3d.bm3d(img.astype(np.float64),
                                            sigma_psd=max(sig, 0.01)).clip(0,1).astype(np.float32))
            all_results['SQUID+BM3D'] = compute_metrics(np.array(bm3d_imgs), targets)
            print(f"    BM3D:     {all_results['SQUID+BM3D']['psnr'].mean():.2f}")
        except ImportError:
            print(f"    BM3D: not available")

    # ML models
    print(f"\n  ML models:")
    ml_preds = {}
    for model_name in sorted(os.listdir(args.results_dir)):
        model_dir = os.path.join(args.results_dir, model_name)
        if not os.path.isdir(model_dir):
            continue
        # Determine model class
        key = model_name.lower()
        if key not in MODEL_CLASSES:
            continue
        model_cls, model_kwargs = MODEL_CLASSES[key]

        for exp in ['squid', 'coil', 'both']:
            ckpt = os.path.join(model_dir, exp, 'best.pt')
            if not os.path.exists(ckpt):
                continue
            in_ch = 2 if exp == 'both' else 1
            model = model_cls(in_channels=in_ch, out_channels=1, **model_kwargs).to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
            model.eval()

            ds = ULFMRIDataset(test_path, exp)
            preds = predict_ml(model, ds, device)
            label = f'{model_name}/{exp}'
            ml_preds[label] = preds
            all_results[label] = compute_metrics(preds, targets)
            m = all_results[label]
            print(f"    {label:20s}: PSNR={m['psnr'].mean():.2f}±{m['psnr'].std():.2f}  "
                  f"SSIM={m['ssim'].mean():.4f}  NMSE={m['nmse'].mean():.4f}")

    # Summary table
    print(f"\n{'='*70}")
    print(f"  SUMMARY (sorted by PSNR)")
    print(f"{'='*70}")
    print(f"  {'Method':<22} {'PSNR':>12} {'SSIM':>12} {'NMSE':>12}")
    print(f"  {'-'*58}")
    for name in sorted(all_results.keys(), key=lambda k: all_results[k]['psnr'].mean()):
        m = all_results[name]
        print(f"  {name:<22} {m['psnr'].mean():6.2f}±{m['psnr'].std():.2f}  "
              f"{m['ssim'].mean():7.4f}±{m['ssim'].std():.4f}  "
              f"{m['nmse'].mean():7.4f}±{m['nmse'].std():.4f}")

    # Save results
    results_file = os.path.join(args.out_dir, 'evaluation_results.txt')
    with open(results_file, 'w') as f:
        f.write(f"Dataset: {args.data_dir}\n")
        f.write(f"N_test: {N}\n\n")
        for name in sorted(all_results.keys(), key=lambda k: all_results[k]['psnr'].mean()):
            m = all_results[name]
            f.write(f"{name:<22} PSNR={m['psnr'].mean():.2f}±{m['psnr'].std():.2f}  "
                    f"SSIM={m['ssim'].mean():.4f}±{m['ssim'].std():.4f}  "
                    f"NMSE={m['nmse'].mean():.4f}±{m['nmse'].std():.4f}\n")
    print(f"\n  Saved: {results_file}")

    # Visualization: reconstruction grid
    n_show = min(5, N)
    show_idx = [i * (N // n_show) for i in range(n_show)]

    # Pick best single-channel and best fusion model
    best_methods = []
    if ml_preds:
        squid_models = {k: all_results[k]['psnr'].mean() for k in ml_preds if '/squid' in k}
        coil_models = {k: all_results[k]['psnr'].mean() for k in ml_preds if '/coil' in k}
        both_models = {k: all_results[k]['psnr'].mean() for k in ml_preds if '/both' in k}
        if squid_models:
            best_methods.append(max(squid_models, key=squid_models.get))
        if coil_models:
            best_methods.append(max(coil_models, key=coil_models.get))
        if both_models:
            best_methods.append(max(both_models, key=both_models.get))

    col_data = [('Ideal', targets)]
    col_data.append(('SQUID raw', squid_raw))
    col_data.append(('Coil raw', coil_raw))
    for m in best_methods:
        col_data.append((m, ml_preds[m]))

    n_cols = len(col_data)
    fig, axes = plt.subplots(n_show, n_cols, figsize=(3*n_cols, 3*n_show))
    if n_show == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(show_idx):
        for col, (title, data) in enumerate(col_data):
            ax = axes[row, col]
            ax.imshow(data[idx], cmap='gray', vmin=0, vmax=1)
            if row == 0:
                ax.set_title(title.split('/')[-1] if '/' in title else title, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            if col > 0:
                mse = np.mean((data[idx] - targets[idx])**2)
                psnr = 10 * np.log10(1.0 / max(mse, 1e-10))
                ax.text(3, 8, f'{psnr:.1f}', color='yellow', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'reconstruction_grid.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {args.out_dir}/reconstruction_grid.png")


if __name__ == '__main__':
    main()
