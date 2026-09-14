# -*- coding: utf-8 -*-
"""
generate_dataset_dualfield.py — Dual-Field Sequential ULF-MRI Dataset
======================================================================

Simulates a dual-field acquisition protocol:
  Scan 1: B₀ = 50 μT  → SQUID receiver (optimal low-field sensor)
  Scan 2: B₀ = X mT   → Coil receiver  (higher gain at higher frequency)

The two scans see DIFFERENT tissue contrast (T1/T2 depend on B₀),
providing both SNR complementarity and contrast complementarity.

Target: Ideal image at B₀ = 50 μT (standard ULF reconstruction target).

Usage:
    python generate_dataset_dualfield.py --b0_coil 1e-3 --out_dir data/dataset_dualfield_1mT
    python generate_dataset_dualfield.py --b0_coil 10e-3 --out_dir data/dataset_dualfield_10mT
"""
from __future__ import annotations
import os, time, argparse
import numpy as np
import h5py

from forward import SpinEchoParams, MRISignalEngine
from receiver import ReceiverBase, MultiChannelReceiver, make_squid_array, make_coil_array
from ulf_params import auto_seq_params

# ===================================================================
# BrainWeb data loading (reuse from generate_dataset.py)
# ===================================================================
DATA_DIR = "data/brainweb/v2_fuzzy"
MATRIX_SIZE = 128
FOV = 0.256
SLICE_STRIDE = 4
MIN_TISSUE_FRACTION = 0.05

SUBJECT_IDS = [4, 5, 6, 18, 20, 38, 41, 42, 43, 44,
               45, 46, 47, 48, 49, 50, 51, 52, 53, 54]
TRAIN_IDS = [4, 5, 6, 18, 20, 38, 41, 42, 43, 44, 45, 46, 47, 48]
VAL_IDS   = [49, 50]
TEST_IDS  = [51, 52, 53, 54]

# Receiver params (same as main experiments)
SQUID_PARAMS = dict(
    pickup_radius=0.0185, pickup_turns=1, coupling=0.5,
    gradiometer_order=2, gradiometer_baseline=0.06,
    pickup_standoff=0.043, sample_sigma=0.4, sample_radius=0.09,
)
COIL_PARAMS = dict(
    coil_radius=0.045, coil_standoff=0.015, coil_turns=0, coil_Q=50,
    wire_diameter=0.51e-3, preamp_en=1.3e-9, preamp_in=0.8e-12,
    sample_sigma=0.4, sample_radius=0.09,
)

def load_fuzzy_volume(sub_id):
    from generate_dataset import load_fuzzy_volume as _load
    return _load(sub_id)

def downsample_slice(fuzzy_2d):
    from generate_dataset import downsample_slice as _ds
    return _ds(fuzzy_2d)

def build_fuzzy_maps(fuzzy_ds):
    from generate_dataset import build_fuzzy_maps as _bfm
    return _bfm(fuzzy_ds)

def slice_has_tissue(fuzzy_2d):
    tissue_fraction = fuzzy_2d[:, :, 1:].sum() / max(fuzzy_2d.sum(), 1e-6)
    return tissue_fraction > MIN_TISSUE_FRACTION


# ===================================================================
# Dual-field slice processing
# ===================================================================
def process_slice_dualfield(fuzzy_maps: dict, b0_squid: float, b0_coil: float,
                            seed: int, n_channels: int = 7) -> dict:
    """Generate dual-field data: SQUID@b0_squid + Coil@b0_coil.

    Returns:
      ideal:  Ideal image at b0_squid [float32, 0-1]
      squid:  SQUID RSS image at b0_squid [float32, 0-1]
      coil:   Coil RSS image at b0_coil [float32, 0-1]
    """
    # --- Scan 1: SQUID at b0_squid ---
    seq_s = auto_seq_params(b0_squid, matrix_size=MATRIX_SIZE, fov=FOV)
    params_s = SpinEchoParams(**seq_s)
    engine_s = MRISignalEngine(params_s)
    engine_s.prepare_fuzzy(fuzzy_maps)

    ideal_mag = np.abs(engine_s.ideal_image)
    if ideal_mag.max() < 1e-30:
        return None

    squid_array = make_squid_array(
        params_s, n_channels=n_channels, base_seed=seed,
        anatomy='brain', **SQUID_PARAMS)
    img_squid = MultiChannelReceiver.reconstruct_rss(squid_array.acquire(engine_s))

    # --- Scan 2: Coil at b0_coil (different B₀ → different contrast!) ---
    seq_c = auto_seq_params(b0_coil, matrix_size=MATRIX_SIZE, fov=FOV)
    params_c = SpinEchoParams(**seq_c)
    engine_c = MRISignalEngine(params_c)
    engine_c.prepare_fuzzy(fuzzy_maps)

    coil_seed = seed + 1_000_000
    coil_array = make_coil_array(
        params_c, n_channels=n_channels, base_seed=coil_seed,
        anatomy='brain', **COIL_PARAMS)
    img_coil = MultiChannelReceiver.reconstruct_rss(coil_array.acquire(engine_c))

    # --- Normalize ---
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


# ===================================================================
# Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Generate dual-field ULF-MRI dataset (SQUID@50μT + Coil@XmT)')
    parser.add_argument('--b0_coil', type=float, required=True,
                        help='B₀ for Coil scan [T], e.g. 1e-3 for 1 mT')
    parser.add_argument('--b0_squid', type=float, default=50e-6,
                        help='B₀ for SQUID scan [T] (default: 50e-6)')
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--n_channels', type=int, default=7)
    parser.add_argument('--stride', type=int, default=SLICE_STRIDE)
    args = parser.parse_args()

    assert args.b0_coil <= 10e-3, f"B₀_coil={args.b0_coil}T exceeds ULF limit (10 mT)"

    os.makedirs(args.out_dir, exist_ok=True)

    seq_s = auto_seq_params(args.b0_squid, matrix_size=MATRIX_SIZE, fov=FOV)
    seq_c = auto_seq_params(args.b0_coil, matrix_size=MATRIX_SIZE, fov=FOV)

    print("=" * 60)
    print("  Dual-Field ULF-MRI Dataset")
    print(f"  SQUID: B₀={args.b0_squid*1e6:.0f}μT, BW={seq_s['readout_bw']:.0f}Hz, "
          f"TE={seq_s['TE']:.0f}ms, NEX={seq_s['NEX']}")
    print(f"  Coil:  B₀={args.b0_coil*1e3:.1f}mT, BW={seq_c['readout_bw']:.0f}Hz, "
          f"TE={seq_c['TE']:.0f}ms, NEX={seq_c['NEX']}")
    print(f"  Array: {args.n_channels}ch, stride={args.stride}")
    scan_s = MATRIX_SIZE * seq_s['NEX'] * (seq_s['t_pol'] + 0.1 + MATRIX_SIZE/seq_s['readout_bw']) / 60
    scan_c = MATRIX_SIZE * seq_c['NEX'] * (seq_c['t_pol'] + 0.1 + MATRIX_SIZE/seq_c['readout_bw']) / 60
    print(f"  Scan time: SQUID {scan_s:.0f}min + Coil {scan_c:.0f}min = {scan_s+scan_c:.0f}min total")
    print(f"  Note: Different B₀ → different T1/T2 contrast between channels")
    print("=" * 60)

    for split_name, ids in [('train', TRAIN_IDS), ('val', VAL_IDS), ('test', TEST_IDS)]:
        print(f"\n--- {split_name} ({len(ids)} subjects) ---")
        collect = {k: [] for k in ['ideal', 'squid', 'coil',
                                    'ideal_max', 'squid_max', 'coil_max']}
        meta = []

        for sub_id in ids:
            t0 = time.time()
            fuzzy_vol = load_fuzzy_volume(sub_id)
            n_slices = fuzzy_vol.shape[2]
            count = 0

            for z in range(0, n_slices, args.stride):
                fuzzy_2d = fuzzy_vol[:, :, z, :]
                if not slice_has_tissue(fuzzy_2d):
                    continue
                fuzzy_ds = downsample_slice(fuzzy_2d)
                fuzzy_maps = build_fuzzy_maps(fuzzy_ds)
                if len(fuzzy_maps) <= 1:
                    continue

                seed = sub_id * 10000 + z
                result = process_slice_dualfield(
                    fuzzy_maps, args.b0_squid, args.b0_coil,
                    seed, args.n_channels)
                if result is None:
                    continue

                for key in collect:
                    collect[key].append(result[key])
                meta.append((sub_id, z))
                count += 1

            elapsed = time.time() - t0
            print(f"  Subject {sub_id:2d}: {count} slices, {elapsed:.1f}s")

        if not collect['ideal']:
            print(f"  WARNING: no slices for {split_name}")
            continue

        n_samples = len(collect['ideal'])
        out_path = os.path.join(args.out_dir, f"{split_name}.h5")

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

            f.create_dataset('meta', data=np.array(meta, dtype=np.int32))
            f.attrs['B0_squid'] = args.b0_squid
            f.attrs['B0_coil'] = args.b0_coil
            f.attrs['NEX'] = seq_s['NEX']
            f.attrs['n_channels'] = args.n_channels
            f.attrs['mode'] = 'dual-field'
            f.attrs['split'] = split_name

        size_mb = os.path.getsize(out_path) / 1e6
        print(f"  {split_name}: {n_samples} slices ({size_mb:.0f} MB)")

    print("\n  Done ✓")


if __name__ == "__main__":
    main()
