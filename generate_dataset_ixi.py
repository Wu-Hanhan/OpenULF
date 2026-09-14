# -*- coding: utf-8 -*-
"""
generate_dataset_ixi.py — Generate ULF-MRI Dataset from IXI Segmentations
===========================================================================

Uses ANTsPy 3-class segmentations (0=BG, 1=CSF, 2=GM, 3=WM) of IXI T1
brain images as tissue maps for our MRI forward model.

Same pipeline as BrainWeb, but:
  - Crisp (discrete) labels instead of fuzzy maps
  - Only 3 tissue classes (CSF/GM/WM) — no skull/fat/muscle
  - ~581 subjects (vs BrainWeb's 20)
  - Different resolution (0.9375mm vs 0.5mm)

Usage:
    python generate_dataset_ixi.py
    python generate_dataset_ixi.py --max_subjects 100
"""
from __future__ import annotations
import os
import sys
import glob
import time
import argparse
import numpy as np
import nibabel as nib
import h5py
from scipy.ndimage import zoom

from constants import GAMMA_H
from forward import SpinEchoParams, MRISignalEngine, _resize
from receiver import ReceiverBase, MultiChannelReceiver, make_squid_array, make_coil_array

# ===================================================================
# Configuration (same as BrainWeb main experiment)
# ===================================================================
SEG_DIR = "data/IXI/segmentations"
OUT_DIR = "data/dataset_ixi"
MATRIX_SIZE = 128
FOV = 0.256

from ulf_params import auto_seq_params
SEQ_PARAMS = auto_seq_params(B0=50e-6, B_prepol=0.1, matrix_size=MATRIX_SIZE, fov=FOV)

# Receiver parameters: same as BrainWeb brain imaging.
# See generate_dataset.py for full citation details.
# sample_sigma=0.4 S/m: Gabriel 1996, head volume-weighted average at ~2 kHz.
# sample_radius=0.09 m: adult head effective radius.
SQUID_PARAMS = dict(
    pickup_radius=0.0185, pickup_turns=1, coupling=0.5,
    gradiometer_order=2, gradiometer_baseline=0.06,
    pickup_standoff=0.043, sample_sigma=0.4, sample_radius=0.09,
)

# Coil: optimized for ULF brain array. See generate_dataset.py for full rationale.
# r=45mm, standoff=15mm (no dewar), spacing=67mm (critical overlap).
COIL_PARAMS = dict(
    coil_radius=0.045, coil_standoff=0.015, coil_turns=0, coil_Q=50,
    wire_diameter=0.51e-3, preamp_en=1.3e-9, preamp_in=0.8e-12,
    sample_sigma=0.4, sample_radius=0.09,
)

MIN_TISSUE_FRACTION = 0.05
SLICE_STRIDE = 4  # every 4th slice to reduce correlation


def load_segmentation(seg_path: str) -> np.ndarray:
    """Load 3-class segmentation volume."""
    nii = nib.load(seg_path)
    seg = np.asarray(nii.dataobj, dtype=np.int8)
    return seg, nii.header.get_zooms()


def downsample_crisp_slice(seg_2d: np.ndarray, voxel_mm: float) -> np.ndarray:
    """Downsample a crisp segmentation slice to target resolution.

    Uses nearest-neighbor interpolation to preserve discrete labels.
    """
    target_voxel = FOV / MATRIX_SIZE * 1000  # ~2.0 mm
    zoom_factor = voxel_mm / target_voxel

    if abs(zoom_factor - 1.0) < 0.05:
        # Already at target resolution, just resize
        return _resize(seg_2d, MATRIX_SIZE)

    ds = zoom(seg_2d.astype(np.float32), zoom_factor, order=0).astype(np.int8)
    return _resize(ds, MATRIX_SIZE)


def process_slice(tissue_map: np.ndarray, params: SpinEchoParams,
                  seed: int, n_channels: int = 7) -> dict:
    """Generate ideal + SQUID + classical images for one crisp slice.

    Uses multi-channel receiver arrays with RSS combination.
    """
    engine = MRISignalEngine(params)
    engine.prepare(tissue_map)

    ideal_mag = np.abs(engine.ideal_image)
    if ideal_mag.max() < 1e-30:
        return None

    # Multi-channel SQUID array
    squid_array = make_squid_array(
        params, n_channels=n_channels, base_seed=seed,
        anatomy='brain', **SQUID_PARAMS)
    img_squid = MultiChannelReceiver.reconstruct_rss(squid_array.acquire(engine))

    # Multi-channel Coil array (independent noise)
    coil_seed = seed + 1_000_000
    coil_array = make_coil_array(
        params, n_channels=n_channels, base_seed=coil_seed,
        anatomy='brain', **COIL_PARAMS)
    img_coil = MultiChannelReceiver.reconstruct_rss(coil_array.acquire(engine))

    def robust_normalize(img):
        p995 = np.percentile(img, 99.5)
        if p995 < 1e-30:
            p995 = img.max()
        return np.clip(img / (p995 + 1e-30), 0, 1).astype(np.float32), np.float32(p995)

    ideal_norm, ideal_max = robust_normalize(ideal_mag)
    squid_norm, squid_max = robust_normalize(img_squid)
    coil_norm, coil_max = robust_normalize(img_coil)

    return {
        'ideal': ideal_norm, 'squid': squid_norm, 'coil': coil_norm,
        'ideal_max': ideal_max, 'squid_max': squid_max, 'coil_max': coil_max,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_subjects', type=int, default=None)
    parser.add_argument('--out_dir', type=str, default=OUT_DIR)
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    params = SpinEchoParams(**SEQ_PARAMS)

    seg_files = sorted(glob.glob(f"{SEG_DIR}/*_seg.nii.gz") +
                       glob.glob(f"{SEG_DIR}/*_ants_seg.nii.gz"))
    if args.max_subjects:
        seg_files = seg_files[:args.max_subjects]

    n_total = len(seg_files)
    n_channels = 7  # Zotev hexagonal array

    print(f"{'='*60}")
    print(f"  IXI Dataset Generation")
    print(f"  B0={params.B0*1e6:.0f}μT, B_prepol={params.B_prepol}T, NEX={params.NEX}")
    print(f"  Array: {n_channels}ch brain (hexagonal)")
    print(f"  {n_total} subjects, stride={SLICE_STRIDE}")
    print(f"{'='*60}")

    # Split: 70/10/20 by subject
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.1)
    splits = {
        'train': seg_files[:n_train],
        'val': seg_files[n_train:n_train + n_val],
        'test': seg_files[n_train + n_val:],
    }

    for split_name, files in splits.items():
        print(f"\n--- {split_name} ({len(files)} subjects) ---")
        collect = {k: [] for k in ['ideal', 'squid', 'coil',
                                    'ideal_max', 'squid_max', 'coil_max']}
        meta = []

        for si, seg_path in enumerate(files):
            t0 = time.time()
            # Extract subject ID from filename like IXI002-Guys-0828-T1_ants_seg.nii.gz
            basename = os.path.basename(seg_path)
            sub_name = basename.split('-')[0]  # e.g., IXI002
            sub_id = int(''.join(c for c in sub_name if c.isdigit()))

            seg_vol, voxels = load_segmentation(seg_path)
            voxel_mm = voxels[0]  # assume isotropic in-plane
            n_slices = seg_vol.shape[2]
            count = 0

            for z in range(0, n_slices, SLICE_STRIDE):
                seg_2d = seg_vol[:, :, z]

                # Skip if mostly background
                tissue_frac = np.sum(seg_2d > 0) / seg_2d.size
                if tissue_frac < MIN_TISSUE_FRACTION:
                    continue

                ds = downsample_crisp_slice(seg_2d, voxel_mm)
                seed = sub_id * 10000 + z
                result = process_slice(ds, params, seed, n_channels=n_channels)
                if result is None:
                    continue

                for key in collect:
                    collect[key].append(result[key])
                meta.append((sub_id, z))
                count += 1

            elapsed = time.time() - t0
            if (si + 1) % 20 == 0 or si == 0:
                print(f"  [{si+1}/{len(files)}] {sub_name}: {count} slices, {elapsed:.1f}s")

        if not collect['ideal']:
            print(f"  WARNING: no slices for {split_name}")
            continue

        n_samples = len(collect['ideal'])
        meta_arr = np.array(meta, dtype=np.int32)

        out_path = os.path.join(out_dir, f"{split_name}.h5")
        with h5py.File(out_path, 'w') as f:
            imgs = f.create_group('images')
            imgs.create_dataset('ideal', data=np.stack(collect['ideal']),
                                compression='gzip', compression_opts=4)
            imgs.create_dataset('squid', data=np.stack(collect['squid']),
                                compression='gzip', compression_opts=4)
            imgs.create_dataset('coil', data=np.stack(collect['coil']),
                                compression='gzip', compression_opts=4)

            norms = f.create_group('norm_factors')
            norms.create_dataset('ideal_max', data=np.array(collect['ideal_max']))
            norms.create_dataset('squid_max', data=np.array(collect['squid_max']))
            norms.create_dataset('coil_max', data=np.array(collect['coil_max']))

            f.create_dataset('meta', data=meta_arr)
            f.attrs['B0'] = params.B0
            f.attrs['B_prepol'] = params.B_prepol
            f.attrs['matrix_size'] = params.matrix_size
            f.attrs['fov'] = params.fov
            f.attrs['NEX'] = params.NEX
            f.attrs['n_channels'] = n_channels
            f.attrs['source'] = 'IXI + ANTsPy Atropos 3-class'
            f.attrs['split'] = split_name

        size_mb = os.path.getsize(out_path) / 1e6
        print(f"  {split_name}: {n_samples} slices, saved {out_path} ({size_mb:.1f} MB)")

    print("\n  Done ✓")


if __name__ == "__main__":
    main()
