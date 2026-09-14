# -*- coding: utf-8 -*-
"""
generate_dataset_abdomen.py — Synthetic Abdominal ULF-MRI Dataset
===================================================================

Generates synthetic abdominal tissue phantoms with randomized organ
positions and sizes, then runs through our ULF-MRI forward model.

This tests whether the SQUID vs Coil findings generalize beyond
brain imaging to a fundamentally different anatomical context.

Tissue relaxation properties:
  T1/T2 at 3T: de Bazelaire et al., Radiology 230(3):652-659, 2004, Table 2.
  PD: Nyman et al., J Comput Assist Tomogr 10(1):158-161, 1986, Table 1;
      supplemented by Haacke et al., "MRI: Physical Principles" (1999), Ch. 4.
  Conductivity at ~2 kHz: Gabriel et al., Phys Med Biol 41:2271-2293, 1996,
      parametric model evaluated at 2 kHz (ULF Larmor frequency for B0=50μT).
      Specific tissues: IT'IS Foundation Database (itis.swiss/tissue-properties).
  Sample dimensions: Standard adult anthropometry, Visible Human Project.
"""
from __future__ import annotations
import os
import time
import argparse
import numpy as np
import h5py

import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tissue import TissueProperties
from forward import SpinEchoParams, MRISignalEngine
from receiver import ReceiverBase, MultiChannelReceiver, make_squid_array, make_coil_array
import tissue as tissue_module

# Abdominal tissue database (de Bazelaire 2004, Stanisz 2005)
# Abdominal tissue relaxation properties at 3T.
# ┌──────────────┬──────────┬──────────┬──────────────────────────────────────────┐
# │ Property     │ Source   │ Table    │ Notes                                    │
# ├──────────────┼──────────┼──────────┼──────────────────────────────────────────┤
# │ T1, T2 (3T)  │ [1]      │ Table 2  │ In vivo, healthy volunteers, n=10-20     │
# │ PD           │ [2],[3]  │ Table 1  │ Relative to water proton density         │
# │ alpha (T1    │ [4]      │ Fig 3    │ Extrapolated from Bottomley's 1-100 MHz  │
# │  field dep.) │          │          │ data; same exponent used for brain       │
# │ t2star_ratio │ [5]      │ —        │ Estimated from published T2* at 3T       │
# └──────────────┴──────────┴──────────┴──────────────────────────────────────────┘
# [1] de Bazelaire et al., Radiology 230(3):652-659, 2004
# [2] Nyman et al., J Comput Assist Tomogr 10(1):158-161, 1986
# [3] Haacke et al., "MRI: Physical Principles and Sequence Design" (1999)
# [4] Bottomley et al., Med Phys 11(4):425-448, 1984
# [5] Estimated; see Yablonskiy & Haacke, MRM 32:749 (1994)
ABDOMINAL_TISSUE_DB = {
    #                                    PD   T1@3T  T2@3T  alpha  t2*/t2
    0:  TissueProperties("Background",    0, 0.00,    0,    0, 0.0,  1.0),
    1:  TissueProperties("Liver",         1, 0.70,  809,   34, 0.35, 0.70),  # [1] T2 Table 2
    2:  TissueProperties("Kidney",        2, 0.87, 1142,   76, 0.35, 0.70),  # [1] cortex
    3:  TissueProperties("Spleen",        3, 0.80, 1328,   61, 0.35, 0.70),  # [1]
    4:  TissueProperties("Fat",           4, 0.95,  382,   68, 0.30, 0.80),  # [1] subcutaneous
    5:  TissueProperties("Muscle",        5, 0.90,  898,   29, 0.35, 0.70),  # [1] paravertebral
    6:  TissueProperties("Aorta/Blood",   6, 0.90, 1779,  200, 0.35, 0.70),  # Shimada 2012 T1; Zhao 2007 T2
    7:  TissueProperties("Bone/Spine",    7, 0.45,  586,   49, 0.30, 0.60),  # [1] L4 vertebra
}

N = 128
OUT_DIR = "data/dataset_abdomen"

from ulf_params import auto_seq_params
SEQ_PARAMS = auto_seq_params(B0=50e-6, B_prepol=0.1, matrix_size=N, fov=0.40)

# Adapted receiver parameters for abdominal imaging.
# sample_sigma: effective conductivity of abdominal region.
#   Fat ~0.03, Liver ~0.04, Muscle ~0.3, Blood ~0.7 S/m (Gabriel 1996).
#   Volume-weighted average for abdomen ≈ 0.35 S/m.
# sample_radius: ~18 cm for adult abdominal cross-section.
SQUID_PARAMS = dict(
    pickup_radius=0.05, pickup_turns=1, coupling=0.5,
    gradiometer_order=2, gradiometer_baseline=0.08,
    pickup_standoff=0.08,   # dewar 25mm + body wall 30mm + organ depth 25mm
    sample_sigma=0.35, sample_radius=0.18,
)
# Coil: optimized for ULF body array. See receiver.py make_coil_array for rationale.
# r=60mm, standoff=25mm (no dewar, body wall + gap), spacing=90mm (critical overlap).
COIL_PARAMS = dict(
    coil_radius=0.060, coil_standoff=0.025, coil_turns=0, coil_Q=50,
    wire_diameter=0.51e-3, preamp_en=1.3e-9, preamp_in=0.8e-12,
    sample_sigma=0.35, sample_radius=0.18,
)


def generate_abdomen_phantom(seed: int) -> np.ndarray:
    """Generate a randomized synthetic abdominal cross-section."""
    rng = np.random.default_rng(seed)
    phantom = np.zeros((N, N), dtype=int)
    Y, X = np.ogrid[-N//2:N//2, -N//2:N//2]

    # Body wall (elliptical)
    bw_a = rng.uniform(45, 55); bw_b = rng.uniform(35, 45)
    phantom[(X/bw_a)**2 + (Y/bw_b)**2 < 1] = 5  # muscle
    # Subcutaneous fat
    phantom[(X/(bw_a-3))**2 + (Y/(bw_b-3))**2 < 1] = 4
    phantom[(X/(bw_a-6))**2 + (Y/(bw_b-6))**2 < 1] = 0  # clear inside

    # Liver (right side, largest organ)
    lx = rng.uniform(5, 15); ly = rng.uniform(-10, -5)
    la = rng.uniform(16, 24); lb = rng.uniform(12, 18)
    phantom[((X-lx)/la)**2 + ((Y-ly)/lb)**2 < 1] = 1

    # Spleen (left side)
    sx = rng.uniform(-18, -12); sy = rng.uniform(-8, -3)
    sa = rng.uniform(8, 12); sb = rng.uniform(6, 10)
    phantom[((X-sx)/sa)**2 + ((Y-sy)/sb)**2 < 1] = 3

    # Kidneys (bilateral)
    for side in [-1, 1]:
        kx = side * rng.uniform(18, 25); ky = rng.uniform(-3, 3)
        ka = rng.uniform(5, 7); kb = rng.uniform(8, 12)
        phantom[((X-kx)/ka)**2 + ((Y-ky)/kb)**2 < 1] = 2

    # Aorta (center, small circle)
    ax = rng.uniform(-2, 2); ay = rng.uniform(-2, 2)
    ar = rng.uniform(3, 5)
    phantom[(X-ax)**2 + (Y-ay)**2 < ar**2] = 6

    # Spine (posterior)
    spy = rng.uniform(22, 28)
    phantom[((X)/4)**2 + ((Y-spy)/6)**2 < 1] = 7

    return phantom


def process_slice(phantom, params, seed, n_channels=7):
    """Generate ideal + SQUID + coil images.

    Uses multi-channel receiver arrays with RSS combination.
    Abdomen uses 'body' array geometry (larger pickup, wider spacing).
    """
    engine = MRISignalEngine(params)
    engine.prepare(phantom)
    ideal_mag = np.abs(engine.ideal_image)
    if ideal_mag.max() < 1e-30:
        return None

    # Multi-channel SQUID array (body geometry)
    squid_array = make_squid_array(
        params, n_channels=n_channels, base_seed=seed,
        anatomy='body', **SQUID_PARAMS)
    img_squid = MultiChannelReceiver.reconstruct_rss(squid_array.acquire(engine))

    # Multi-channel Coil array (independent noise)
    coil_seed = seed + 1_000_000
    coil_array = make_coil_array(
        params, n_channels=n_channels, base_seed=coil_seed,
        anatomy='body', **COIL_PARAMS)
    img_coil = MultiChannelReceiver.reconstruct_rss(coil_array.acquire(engine))

    def robust_normalize(img):
        p995 = np.percentile(img, 99.5)
        if p995 < 1e-30: p995 = img.max()
        return np.clip(img/(p995+1e-30), 0, 1).astype(np.float32), np.float32(p995)

    ideal_n, ideal_m = robust_normalize(ideal_mag)
    squid_n, squid_m = robust_normalize(img_squid)
    coil_n, coil_m = robust_normalize(img_coil)

    return {
        'ideal': ideal_n, 'squid': squid_n, 'coil': coil_n,
        'ideal_max': ideal_m, 'squid_max': squid_m, 'coil_max': coil_m,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_phantoms', type=int, default=500,
                        help='Number of random abdominal phantoms')
    parser.add_argument('--out_dir', type=str, default=OUT_DIR)
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Temporarily swap tissue database
    original_db = tissue_module.TISSUE_DB
    tissue_module.TISSUE_DB = ABDOMINAL_TISSUE_DB

    params = SpinEchoParams(**SEQ_PARAMS)
    n_total = args.n_phantoms

    n_channels = 7  # hexagonal array, body geometry

    print(f"{'='*60}")
    print(f"  Abdominal ULF-MRI Dataset Generation")
    print(f"  {n_total} random phantoms, B0={params.B0*1e6:.0f}μT, NEX={params.NEX}")
    print(f"  Array: {n_channels}ch body (hexagonal)")
    print(f"  FOV={params.fov*100:.0f}cm (abdominal)")
    print(f"{'='*60}")

    # Split: 70/10/20
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.1)
    splits = {
        'train': range(0, n_train),
        'val': range(n_train, n_train + n_val),
        'test': range(n_train + n_val, n_total),
    }

    for split_name, indices in splits.items():
        collect = {k: [] for k in ['ideal','squid','coil',
                    'ideal_max','squid_max','coil_max']}
        t0 = time.time()

        for idx in indices:
            phantom = generate_abdomen_phantom(seed=idx * 7 + 13)
            result = process_slice(phantom, params, seed=idx * 1000, n_channels=n_channels)
            if result is None:
                continue
            for k in collect:
                collect[k].append(result[k])

        elapsed = time.time() - t0
        n_samples = len(collect['ideal'])

        if n_samples == 0:
            print(f"  {split_name}: 0 phantoms (skipped)")
            continue

        out_path = os.path.join(out_dir, f"{split_name}.h5")
        with h5py.File(out_path, 'w') as f:
            imgs = f.create_group('images')
            imgs.create_dataset('ideal', data=np.stack(collect['ideal']), compression='gzip')
            imgs.create_dataset('squid', data=np.stack(collect['squid']), compression='gzip')
            imgs.create_dataset('coil', data=np.stack(collect['coil']), compression='gzip')
            norms = f.create_group('norm_factors')
            norms.create_dataset('ideal_max', data=np.array(collect['ideal_max']))
            norms.create_dataset('squid_max', data=np.array(collect['squid_max']))
            norms.create_dataset('coil_max', data=np.array(collect['coil_max']))
            f.create_dataset('meta', data=np.zeros((n_samples, 2), dtype=np.int32))
            f.attrs['B0'] = params.B0; f.attrs['B_prepol'] = params.B_prepol
            f.attrs['matrix_size'] = params.matrix_size; f.attrs['fov'] = params.fov
            f.attrs['NEX'] = params.NEX
            f.attrs['n_channels'] = n_channels
            f.attrs['anatomy'] = 'body'
            f.attrs['source'] = 'synthetic_abdomen'
            f.attrs['split'] = split_name

        print(f"  {split_name}: {n_samples} phantoms, {elapsed:.0f}s → {out_path}")

    tissue_module.TISSUE_DB = original_db
    print("\n  Done ✓")


if __name__ == "__main__":
    main()
