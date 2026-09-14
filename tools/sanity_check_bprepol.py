# -*- coding: utf-8 -*-
"""
sanity_check_bprepol.py — B_prepol Sweep Dataset Sanity Checks
================================================================

Run BEFORE training to verify all datasets are correct and consistent.
"""
import os
import sys
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    bp_configs = [
        (0.01, 'data/dataset_bp_10mT'),
        (0.02, 'data/dataset_bp_20mT'),
        (0.05, 'data/dataset_bp_50mT'),
        (0.10, 'data/dataset'),           # main experiment
        (0.20, 'data/dataset_bp_200mT'),
        (0.30, 'data/dataset_highend'),    # high-end
        (0.50, 'data/dataset_bp_500mT'),
        (1.00, 'data/dataset_bp_1000mT'),
    ]

    print("=" * 75)
    print("  B_prepol Sweep Sanity Checks")
    print("=" * 75)

    all_ok = True
    bp_vals = []
    squid_psnrs = []
    coil_psnrs = []
    ideal_maxes = []
    n_slices_list = []

    for bp, path in bp_configs:
        train_path = os.path.join(path, 'train.h5')
        test_path = os.path.join(path, 'test.h5')

        if not os.path.exists(test_path):
            print(f"\n  ✗ MISSING: {test_path}")
            all_ok = False
            continue

        with h5py.File(test_path, 'r') as f:
            ideal = f['images/ideal'][:].astype(np.float32)
            squid = f['images/squid'][:].astype(np.float32)
            coil = f['images/coil'][:].astype(np.float32)
            attrs_bp = f.attrs.get('B_prepol', None)
            attrs_b0 = f.attrs.get('B0', None)

        with h5py.File(train_path, 'r') as f:
            n_train = f['images/ideal'].shape[0]

        n_test = len(ideal)

        # Check 1: Attribute match
        if attrs_bp is not None and abs(attrs_bp - bp) > 1e-4:
            print(f"\n  ✗ B_prepol MISMATCH at {path}: expected {bp}, got {attrs_bp}")
            all_ok = False
            continue

        # Check 2: B0 consistent
        if attrs_b0 is not None and abs(attrs_b0 - 50e-6) > 1e-8:
            print(f"\n  ✗ B0 MISMATCH at {path}: expected 50e-6, got {attrs_b0}")
            all_ok = False

        # Check 3: Shapes
        assert ideal.shape == squid.shape == coil.shape, f"Shape mismatch at {path}"
        assert ideal.shape[1:] == (128, 128), f"Expected 128x128, got {ideal.shape[1:]}"

        # Check 4: Value ranges
        assert ideal.min() >= 0 and ideal.max() <= 1.0, f"Ideal out of [0,1] at {path}"

        # Check 5: Compute raw PSNR
        mse_s = np.mean((squid - ideal)**2, axis=(1, 2))
        mse_c = np.mean((coil - ideal)**2, axis=(1, 2))
        psnr_s = np.mean(10 * np.log10(1.0 / np.clip(mse_s, 1e-30, None)))
        psnr_c = np.mean(10 * np.log10(1.0 / np.clip(mse_c, 1e-30, None)))

        bp_vals.append(bp)
        squid_psnrs.append(psnr_s)
        coil_psnrs.append(psnr_c)
        ideal_maxes.append(ideal.max())
        n_slices_list.append((n_train, n_test))

        print(f"\n  B_prepol={bp:.2f}T ({path})")
        print(f"    Train={n_train}, Test={n_test}, Shape={ideal.shape[1:]}")
        print(f"    Ideal: max={ideal.max():.4f}, mean={ideal.mean():.4f}")
        print(f"    SQUID raw PSNR: {psnr_s:.2f} dB")
        print(f"    Coil raw PSNR:  {psnr_c:.2f} dB")
        print(f"    Gap (S-C):      {psnr_s - psnr_c:+.2f} dB")

    # Check 6: Slice count consistency
    print(f"\n--- Slice Count Consistency ---")
    train_counts = set(n[0] for n in n_slices_list)
    test_counts = set(n[1] for n in n_slices_list)
    if len(train_counts) == 1 and len(test_counts) == 1:
        print(f"  ✓ All datasets: train={train_counts.pop()}, test={test_counts.pop()}")
    else:
        print(f"  ⚠ INCONSISTENT: train={train_counts}, test={test_counts}")

    # Check 7: SQUID PSNR should increase monotonically with B_prepol
    print(f"\n--- Monotonicity Check ---")
    squid_mono = all(squid_psnrs[i] <= squid_psnrs[i+1]
                     for i in range(len(squid_psnrs)-1))
    print(f"  SQUID PSNR monotonically increasing: {'✓' if squid_mono else '⚠ NOT monotonic'}")

    # Coil PSNR should be roughly constant
    coil_range = max(coil_psnrs) - min(coil_psnrs)
    print(f"  Coil PSNR range: {coil_range:.2f} dB ({'✓ stable' if coil_range < 1.0 else '⚠ varies'})")

    # Check 8: SQUID-Coil gap should be non-negative for all but lowest B_prepol
    print(f"\n--- Gap Consistency ---")
    for bp, ps, pc in zip(bp_vals, squid_psnrs, coil_psnrs):
        gap = ps - pc
        status = '✓' if gap >= -0.1 else '⚠ Coil better than SQUID!'
        print(f"  B_prepol={bp:.2f}T: Gap={gap:+.2f} dB {status}")

    # Generate visualization
    if len(bp_vals) >= 4:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        ax = axes[0]
        ax.plot(bp_vals, squid_psnrs, 'o-', color='#2196F3', linewidth=2, markersize=8, label='SQUID')
        ax.plot(bp_vals, coil_psnrs, 's-', color='#FF9800', linewidth=2, markersize=8, label='Coil')
        ax.set_xlabel('B_prepol (T)'); ax.set_ylabel('Raw Input PSNR (dB)')
        ax.set_title('Input PSNR vs B_prepol'); ax.set_xscale('log')
        ax.legend(); ax.grid(True, alpha=0.2)

        ax = axes[1]
        gaps = [s - c for s, c in zip(squid_psnrs, coil_psnrs)]
        ax.plot(bp_vals, gaps, 'o-', color='#4CAF50', linewidth=2, markersize=8)
        ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
        ax.set_xlabel('B_prepol (T)'); ax.set_ylabel('SQUID - Coil (dB)')
        ax.set_title('Input SQUID Advantage'); ax.set_xscale('log')
        ax.grid(True, alpha=0.2)

        ax = axes[2]
        # Show example images at min, mid, max B_prepol
        for i, bp_show in enumerate([bp_vals[0], bp_vals[len(bp_vals)//2], bp_vals[-1]]):
            idx = bp_vals.index(bp_show)
            bp_path = bp_configs[idx][1]
            with h5py.File(os.path.join(bp_path, 'test.h5'), 'r') as f:
                img = f['images/ideal'][50]
            sub = ax.inset_axes([i*0.33, 0, 0.32, 1.0])
            sub.imshow(img, cmap='gray', vmin=0, vmax=1)
            sub.set_title(f'Bp={bp_show}T', fontsize=9)
            sub.set_xticks([]); sub.set_yticks([])
        ax.axis('off'); ax.set_title('Ideal Images at Different B_prepol')

        plt.tight_layout()
        plt.savefig('analysis/bprepol_sanity_check.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  Saved: analysis/bprepol_sanity_check.png")

    if all_ok:
        print(f"\n{'='*75}")
        print(f"  ✓ ALL SANITY CHECKS PASSED — safe to proceed with training")
        print(f"{'='*75}")
    else:
        print(f"\n{'='*75}")
        print(f"  ✗ SOME CHECKS FAILED — investigate before training")
        print(f"{'='*75}")
        sys.exit(1)


if __name__ == '__main__':
    main()
