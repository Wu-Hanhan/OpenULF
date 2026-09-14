# -*- coding: utf-8 -*-
"""
generate_dataset.py — ULF-MRI Dataset Generator
=================================================

Generates training data for ML reconstruction comparison:
  SQUID quantum receiver vs classical induction coil.

Pipeline per slice:
  BrainWeb fuzzy tissue map
    → downsample to target resolution
    → per-voxel sum normalization (true partial volume fractions)
    → MRI forward engine (prepolarized ULF spin echo)
    → ideal M⊥ image (ground truth)
    → SQUID receiver → noisy k-space → magnitude image
    → Classical coil  → noisy k-space → magnitude image
    → normalize all to [0, 1]
    → save to HDF5

Usage:
    python generate_dataset.py
"""
from __future__ import annotations
import os
import sys
import time
import numpy as np
import nibabel as nib
import h5py
from scipy.ndimage import zoom

from constants import GAMMA_H
from forward import SpinEchoParams, MRISignalEngine
from receiver import ReceiverBase, MultiChannelReceiver, make_squid_array, make_coil_array

# ===================================================================
# Configuration
# ===================================================================
DATA_DIR = "data/brainweb/v2_fuzzy"
OUT_DIR = "data/dataset"
MATRIX_SIZE = 128
FOV = 0.256                   # 25.6 cm
BRAINWEB_VOXEL_MM = 0.5       # BrainWeb V2 resolution
TARGET_VOXEL_MM = FOV / MATRIX_SIZE * 1000  # ~2.0 mm

# Sequence parameters (main experiment)
#
# B0 = 50 μT: chosen to match the ULF-MRI regime used by the two leading
#   experimental groups: Zotev et al. (LANL, B₀=46μT) and Vesanen et al.
#   (Helsinki/Aalto, B₀≈50μT). This is also the approximate magnitude of
#   Earth's magnetic field (~25-65 μT), which is the natural operating point
#   for SQUID-based ULF-MRI. At this field, f_Larmor ≈ 2.1 kHz — far too low
#   for induction coils to be effective, making SQUID sensors essential.
#   Ref: Zotev et al., Supercond. Sci. Technol. 20:S367 (2007);
#        Vesanen et al., Magn. Reson. Med. 69:1795 (2013).
#
# B_prepol = 0.1 T: moderate prepolarization. Zotev et al., JMR 194:115 (2008).
#   Typical range: 20 mT (Vesanen) to 100 mT (Zotev). We use 100 mT.
#
# NEX = 10: signal averages per phase-encode line.
#   Zotev 2008: 6 averages at 64 matrix. We use 10 at 128 matrix to
#   compensate for smaller voxel (2mm vs 3mm). Total scan ~25 min
#   (comparable to Vesanen 2013: 26 min).
#
# BW and TE are auto-computed from B₀ and gradient constraints.
# t_pol=1.0s: ~3×T1 at B_prepol → 96% polarization (Zotev 2008: 1s).
# t_transfer=0.1s: prepol switch-off to readout. Vesanen 2013: 50-200ms.
from ulf_params import auto_seq_params
SEQ_PARAMS = auto_seq_params(
    B0=50e-6, B_prepol=0.1,
    matrix_size=MATRIX_SIZE, fov=FOV,
)

# Matched receiver geometry for brain imaging.
# sample_sigma: effective conductivity of head at ~2 kHz (ULF Larmor frequency).
#   Gabriel et al., Phys Med Biol 41:2271-2293, 1996, parametric model at 2 kHz:
#   GM ~0.10, WM ~0.06, CSF ~2.0, skull ~0.02, scalp ~0.3 S/m.
#   Volume-weighted average for adult head ≈ 0.3-0.4 S/m.
# sample_radius: ~9 cm effective radius for adult head cross-section.
#   Adult head AP ~19 cm, lateral ~15 cm → effective radius ~8-10 cm.
RECEIVER_COMMON = dict(
    sample_sigma=0.4,       # [S/m] Gabriel 1996 volume-weighted average
    sample_radius=0.09,     # [m] adult head effective radius
)

# SQUID parameters.
# pickup_radius: 5cm, matched to Coil radius for fair geometric comparison.
#   Published brain MRI SQUIDs use smaller pickups: Zotev 2008 Ø37mm (r=18.5mm),
#   Vesanen 2013/MEG Ø24mm. SNR ratio is insensitive to pickup size (8-10×
#   range for r=18.5-50mm) because larger area captures more signal AND more noise.
# gradiometer: 2nd-order, 60mm baseline (Zotev 2008).
# standoff: 50mm. For r=50mm this coincides with 1.0×radius. Physically:
#   dewar wall ~25mm (Zotev/LANL measured) + scalp/skull ~13mm + cortex ~10mm.
#   The standoff is set by dewar + anatomy, NOT proportional to pickup radius.
# system_noise_fT: set in SQUIDQuantumReceiver default (2.0 fT/√Hz).
SQUID_PARAMS = dict(
    pickup_radius=0.0185,   # [m] 18.5mm — Zotev 2008 brain MRI (Ø37mm pickup)
    pickup_turns=1,         # single-turn Nb wire (Zotev 2008)
    coupling=0.5,           # input coupling (Clarke & Braginski 2004, §5.5)
    gradiometer_order=2,    # 2nd-order (Zotev 2008; Vrba & Robinson 2001)
    gradiometer_baseline=0.06,  # [m] 6 cm baseline (Zotev 2008)
    pickup_standoff=0.043,  # [m] 43mm: dewar 25mm + scalp/skull 13mm + cortex 5mm
    **RECEIVER_COMMON,
)

# Classical coil: optimized for ULF-MRI at ~2 kHz Larmor frequency.
# coil_radius: 45mm for brain array. Optimized via parameter sweep for best
#   balance of SNR (larger → more flux) and B₁ complementarity with SQUID
#   (different spatial profile). See analysis in receiver.py make_coil_array.
# coil_standoff: 15mm. Coil operates at room temperature (no dewar), so it
#   sits directly on scalp: scalp ~5mm + skull ~7mm + gap ~3mm = ~15mm.
#   This is 3× closer than SQUID (43mm through dewar), giving Coil a
#   sensitivity advantage for shallow cortical tissue.
# Array spacing: 1.5 × coil_radius = 67mm (critical overlap rule,
#   standard practice for mutual inductance minimization in phased arrays).
# Turn count: auto-optimized for INA217 preamp in untuned broadband mode.
# Wire: AWG24 (Matlashov 2011). Skin depth δ=1.4mm >> wire radius at 2 kHz.
# Preamp: INA217 (Matlashov 2011) — validated in ULF-MRI coil literature.
COIL_PARAMS = dict(
    coil_radius=0.045,      # [m] 45mm: optimal for 7ch brain array
    coil_standoff=0.015,    # [m] 15mm: no dewar, scalp+skull+gap
    coil_turns=0,           # 0 = auto-optimize for max SNR (pancake model)
    coil_Q=50,              # body-loaded Q (Edelstein 1986)
    wire_diameter=0.51e-3,  # [m] AWG24 copper (Matlashov 2011)
    preamp_en=1.3e-9,       # [V/√Hz] INA217 typical (TI datasheet)
    preamp_in=0.8e-12,      # [A/√Hz] INA217 (Matlashov 2011)
    **RECEIVER_COMMON,
)

# Subject IDs
SUBJECT_IDS = [4, 5, 6, 18, 20, 38, 41, 42, 43, 44,
               45, 46, 47, 48, 49, 50, 51, 52, 53, 54]

# Train/Val/Test split (by subject, no leakage)
TRAIN_IDS = [4, 5, 6, 18, 20, 38, 41, 42, 43, 44, 45, 46, 47, 48]  # 14
VAL_IDS   = [49, 50]                                                  # 2
TEST_IDS  = [51, 52, 53, 54]                                          # 4

# Minimum tissue fraction to include a slice (skip mostly-empty slices)
MIN_TISSUE_FRACTION = 0.05

# Slice stride: take every Nth slice to reduce inter-slice correlation.
# BrainWeb V2 is 0.5mm/voxel; stride=4 → 2mm spacing → ~20% correlation.
SLICE_STRIDE = 4


# ===================================================================
# Helper Functions
# ===================================================================
def load_fuzzy_volume(subject_id: int) -> np.ndarray:
    """Load and normalize BrainWeb V2 fuzzy volume.

    Returns
    -------
    fuzzy : np.ndarray, shape (H, W, D, 12), float32
        Per-voxel tissue probability maps, sum-normalized so that
        each voxel's 12 channels sum to 1.0.
    """
    path = os.path.join(DATA_DIR, f"brainweb_s{subject_id:02d}_fuzzy.nii")
    if not os.path.exists(path):
        path += ".gz"
    nii = nib.load(path)
    raw = np.asarray(nii.dataobj, dtype=np.float32)  # (362, 434, 362, 12)

    # Per-voxel sum normalization → true partial volume fractions
    total = raw.sum(axis=-1, keepdims=True)
    total[total == 0] = 1.0  # prevent division by zero
    fuzzy = raw / total

    return fuzzy


def downsample_slice(fuzzy_2d: np.ndarray) -> np.ndarray:
    """Downsample a 2D fuzzy slice to target resolution.

    BrainWeb V2: 0.5mm voxels, (362, 434).
    Target: ~2mm voxels within 128×128 matrix at 25.6cm FOV.

    Physical extent of BrainWeb: 181mm × 217mm.
    At 2mm resolution: 90.5 × 108.5 pixels.
    Center-pad to 128×128.

    Parameters
    ----------
    fuzzy_2d : np.ndarray, shape (H, W, 12)

    Returns
    -------
    result : np.ndarray, shape (MATRIX_SIZE, MATRIX_SIZE, 12)
    """
    H, W, n_ch = fuzzy_2d.shape
    zoom_factor = BRAINWEB_VOXEL_MM / TARGET_VOXEL_MM  # 0.5/2.0 = 0.25

    result = np.zeros((MATRIX_SIZE, MATRIX_SIZE, n_ch), dtype=np.float32)

    for ch in range(n_ch):
        channel = fuzzy_2d[:, :, ch]
        if np.max(channel) < 1e-6:
            continue
        # Downsample with bilinear interpolation
        ds = zoom(channel, zoom_factor, order=1).astype(np.float32)
        # Center-pad into target matrix
        h, w = ds.shape
        y0 = (MATRIX_SIZE - h) // 2
        x0 = (MATRIX_SIZE - w) // 2
        # Clip to handle rounding
        h_copy = min(h, MATRIX_SIZE - y0)
        w_copy = min(w, MATRIX_SIZE - x0)
        result[y0:y0+h_copy, x0:x0+w_copy, ch] = ds[:h_copy, :w_copy]

    # Re-normalize after interpolation (zoom can slightly break normalization)
    total = result.sum(axis=-1, keepdims=True)  # (N, N, 1)
    total[total == 0] = 1.0
    result /= total

    return result


def slice_has_tissue(fuzzy_2d: np.ndarray) -> bool:
    """Check if slice has enough non-background tissue."""
    # Channel 0 = background. Sum of all non-background channels.
    tissue_fraction = fuzzy_2d[:, :, 1:].sum() / max(fuzzy_2d.sum(), 1e-6)
    return tissue_fraction > MIN_TISSUE_FRACTION


def build_fuzzy_maps(fuzzy_2d_ds: np.ndarray) -> dict:
    """Convert (N, N, 12) array to {label: (N, N)} dict for engine."""
    fuzzy_maps = {}
    for ch in range(12):
        prob = fuzzy_2d_ds[:, :, ch].astype(np.float64)
        if np.max(prob) > 1e-6:
            fuzzy_maps[ch] = prob
    return fuzzy_maps


def process_slice(fuzzy_maps: dict, params: SpinEchoParams,
                  seed: int, n_channels: int = 7,
                  anatomy: str = 'brain') -> dict:
    """Generate ideal + SQUID + classical data for one slice.

    Uses multi-channel receiver arrays (default 7-channel hexagonal,
    matching Zotev et al. 2007 SQUID system). Each channel has independent
    noise and spatially shifted B₁ sensitivity. Channel images are combined
    via root-sum-of-squares (RSS).

    Parameters
    ----------
    fuzzy_maps : dict
        Tissue probability maps for the forward engine.
    params : SpinEchoParams
        Acquisition parameters (includes NEX for signal averaging).
    seed : int
        RNG seed for noise. Each channel gets seed + channel_index.
    n_channels : int
        Number of receiver channels. Default 7 (Zotev hexagonal).
        Use 1 for single-channel (backward compatible).
    anatomy : str
        'brain' or 'body'. Controls array geometry.

    Returns dict with:
      Images (float32, normalized [0,1]):
        'ideal', 'squid', 'coil'  — each (N, N)
      Normalization factors (for reversibility):
        'ideal_max', 'squid_max', 'coil_max'
    """
    # Forward engine
    engine = MRISignalEngine(params)
    engine.prepare_fuzzy(fuzzy_maps)

    ideal_mag = np.abs(engine.ideal_image)

    # Skip if ideal image is blank
    if ideal_mag.max() < 1e-30:
        return None

    # SQUID multi-channel array
    # Each channel: independent seed, spatially offset pickup coil
    squid_array = make_squid_array(
        params, n_channels=n_channels, base_seed=seed,
        anatomy=anatomy, **SQUID_PARAMS)
    ksp_squid = squid_array.acquire(engine)          # (n_ch, N, N_ro)
    img_squid = MultiChannelReceiver.reconstruct_rss(ksp_squid)  # (N, N)

    # Classical coil multi-channel array
    # Offset seed by 1M to ensure independent noise from SQUID.
    coil_seed = seed + 1_000_000
    coil_array = make_coil_array(
        params, n_channels=n_channels, base_seed=coil_seed,
        anatomy=anatomy, **COIL_PARAMS)
    ksp_coil = coil_array.acquire(engine)
    img_coil = MultiChannelReceiver.reconstruct_rss(ksp_coil)

    # Normalize images to [0, 1] using 99.5th percentile.
    def robust_normalize(img):
        p995 = np.percentile(img, 99.5)
        if p995 < 1e-30:
            p995 = img.max()
        normed = np.clip(img / (p995 + 1e-30), 0, 1)
        return normed.astype(np.float32), np.float32(p995)

    ideal_norm, ideal_max = robust_normalize(ideal_mag)
    squid_norm, squid_max = robust_normalize(img_squid)
    coil_norm, coil_max = robust_normalize(img_coil)

    return {
        # Normalized images for ML training
        'ideal': ideal_norm,
        'squid': squid_norm,
        'coil': coil_norm,
        # Normalization factors
        'ideal_max': np.float32(ideal_max),
        'squid_max': np.float32(squid_max),
        'coil_max': np.float32(coil_max),
    }


# ===================================================================
# Main
# ===================================================================
def generate_split(split_name: str, subject_ids: list,
                   params: SpinEchoParams, out_path: str,
                   n_channels: int = 7, anatomy: str = 'brain'):
    """Generate dataset for one split (train/val/test)."""
    collect = {k: [] for k in ['ideal', 'squid', 'coil',
                                  'ideal_max', 'squid_max', 'coil_max']}
    slice_meta = []

    for sub_id in subject_ids:
        t0 = time.time()
        print(f"  Subject {sub_id:2d}: loading...", end="", flush=True)
        fuzzy_vol = load_fuzzy_volume(sub_id)
        n_slices = fuzzy_vol.shape[2]
        count = 0

        for z in range(0, n_slices, SLICE_STRIDE):
            fuzzy_2d = fuzzy_vol[:, :, z, :]

            if not slice_has_tissue(fuzzy_2d):
                continue

            fuzzy_ds = downsample_slice(fuzzy_2d)
            fuzzy_maps = build_fuzzy_maps(fuzzy_ds)

            if len(fuzzy_maps) <= 1:
                continue

            seed = sub_id * 10000 + z
            result = process_slice(fuzzy_maps, params, seed,
                                   n_channels=n_channels, anatomy=anatomy)
            if result is None:
                continue

            for key in collect:
                collect[key].append(result[key])
            slice_meta.append((sub_id, z))
            count += 1

        elapsed = time.time() - t0
        print(f" {count} slices, {elapsed:.1f}s")

    # Stack and save
    if not collect['ideal']:
        print(f"  WARNING: no slices generated for {split_name}")
        return

    n_samples = len(collect['ideal'])
    meta_arr = np.array(slice_meta, dtype=np.int32)

    print(f"  {split_name}: {n_samples} slices")

    with h5py.File(out_path, 'w') as f:
        # Images: normalized [0,1], float32, for ML training
        imgs = f.create_group('images')
        imgs.create_dataset('ideal', data=np.stack(collect['ideal']),
                            compression='gzip', compression_opts=4)
        imgs.create_dataset('squid', data=np.stack(collect['squid']),
                            compression='gzip', compression_opts=4)
        imgs.create_dataset('coil', data=np.stack(collect['coil']),
                            compression='gzip', compression_opts=4)

        # Normalization factors (to recover original scale)
        norms = f.create_group('norm_factors')
        norms.create_dataset('ideal_max', data=np.array(collect['ideal_max']))
        norms.create_dataset('squid_max', data=np.array(collect['squid_max']))
        norms.create_dataset('coil_max', data=np.array(collect['coil_max']))

        # Metadata
        f.create_dataset('meta', data=meta_arr)

        # Configuration
        f.attrs['B0'] = params.B0
        f.attrs['B_prepol'] = params.B_prepol
        f.attrs['matrix_size'] = params.matrix_size
        f.attrs['fov'] = params.fov
        f.attrs['TE'] = params.TE
        f.attrs['readout_bw'] = params.readout_bw
        f.attrs['NEX'] = params.NEX
        f.attrs['t_pol'] = params.t_pol
        f.attrs['t_transfer'] = params.t_transfer
        f.attrs['n_channels'] = n_channels
        f.attrs['anatomy'] = anatomy
        f.attrs['split'] = split_name
        f.attrs['subject_ids'] = list(subject_ids)

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  Saved: {out_path} ({size_mb:.1f} MB)")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Generate ULF-MRI dataset with SQUID + Classical Coil receivers.')

    # Sequence parameters
    seq = parser.add_argument_group('Sequence')
    seq.add_argument('--b0', type=float, default=None,
                     help='B0 field [T] (default: 50e-6)')
    seq.add_argument('--b_prepol', type=float, default=None,
                     help='Prepolarization field [T] (default: 0.1)')
    seq.add_argument('--nex', type=int, default=None,
                     help='Number of excitations (default: 1)')
    seq.add_argument('--bw', type=float, default=None,
                     help='Readout BW [Hz] (default: auto from B0)')
    seq.add_argument('--te', type=float, default=None,
                     help='Echo time [ms] (default: auto from BW)')

    # Coil parameters
    coil = parser.add_argument_group('Classical Coil')
    coil.add_argument('--coil_radius', type=float, default=None,
                      help='Coil outer radius [m] (default: 0.05)')
    coil.add_argument('--coil_turns', type=int, default=None,
                      help='Coil turns (0=auto-optimize, default: 0)')
    coil.add_argument('--coil_inner_radius', type=float, default=None,
                      help='Coil inner radius [m] (default: radius/5)')
    coil.add_argument('--preamp_en', type=float, default=None,
                      help='Preamp voltage noise [V/rtHz] (default: 1.3e-9)')
    coil.add_argument('--preamp_in', type=float, default=None,
                      help='Preamp current noise [A/rtHz] (default: 0.8e-12)')

    # SQUID parameters
    sq = parser.add_argument_group('SQUID')
    sq.add_argument('--squid_radius', type=float, default=None,
                    help='SQUID pickup radius [m] (default: 0.05)')
    sq.add_argument('--squid_standoff', type=float, default=None,
                    help='SQUID standoff [m] (default: auto)')
    sq.add_argument('--system_noise', type=float, default=None,
                    help='SQUID system noise [fT/rtHz] (default: 2.0)')

    # Array configuration
    arr = parser.add_argument_group('Receiver Array')
    arr.add_argument('--n_channels', type=int, default=7,
                     help='Number of receiver channels per sensor (default: 7, Zotev hexagonal)')
    arr.add_argument('--anatomy', type=str, default='brain',
                     choices=['brain', 'body'],
                     help='Array geometry preset (default: brain)')

    # Output
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--stride', type=int, default=None,
                        help='Slice stride (default: SLICE_STRIDE)')
    args = parser.parse_args()

    out_dir = args.out_dir or OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    # Build sequence params (auto-compute BW/TE from B0 if not specified)
    B0 = args.b0 if args.b0 is not None else 50e-6
    B_prepol = args.b_prepol if args.b_prepol is not None else 0.1
    seq_params = auto_seq_params(B0=B0, B_prepol=B_prepol,
                                  matrix_size=MATRIX_SIZE, fov=FOV)
    if args.bw is not None:
        seq_params['readout_bw'] = args.bw
    if args.te is not None:
        seq_params['TE'] = args.te
    if args.nex is not None:
        seq_params['NEX'] = args.nex

    # Override coil params if specified
    global COIL_PARAMS, SQUID_PARAMS
    if args.coil_radius is not None:
        COIL_PARAMS['coil_radius'] = args.coil_radius
    if args.coil_turns is not None:
        COIL_PARAMS['coil_turns'] = args.coil_turns
    if args.coil_inner_radius is not None:
        COIL_PARAMS['coil_inner_radius'] = args.coil_inner_radius
    if args.preamp_en is not None:
        COIL_PARAMS['preamp_en'] = args.preamp_en
    if args.preamp_in is not None:
        COIL_PARAMS['preamp_in'] = args.preamp_in

    # Override SQUID params if specified
    if args.squid_radius is not None:
        SQUID_PARAMS['pickup_radius'] = args.squid_radius
    if args.squid_standoff is not None:
        SQUID_PARAMS['pickup_standoff'] = args.squid_standoff

    if args.stride is not None:
        global SLICE_STRIDE
        SLICE_STRIDE = args.stride

    params = SpinEchoParams(**seq_params)

    # Create receivers to show auto-computed values
    from classical_coil_receiver import ClassicalCoilReceiver
    _coil_preview = ClassicalCoilReceiver(params, **COIL_PARAMS)

    print("=" * 60)
    print("  ULF-MRI Dataset Generation")
    print(f"  B0={params.B0*1e6:.0f}μT, B_prepol={params.B_prepol}T, NEX={params.NEX}")
    print(f"  f_Larmor={params.f_larmor:.0f} Hz, BW={params.readout_bw:.0f} Hz")
    print(f"  TE={params.TE:.0f}ms, Matrix={params.matrix_size}, FOV={params.fov*100:.0f}cm")
    print(f"  Array: {args.n_channels}ch {args.anatomy} (hexagonal)")
    print(f"  Coil: n={_coil_preview.coil_turns} (r={_coil_preview.coil_radius*100:.0f}cm, "
          f"r_inner={_coil_preview.coil_inner_radius*100:.0f}cm, "
          f"mode={_coil_preview.mode})")
    print(f"  SLICE_STRIDE={SLICE_STRIDE}")
    print("=" * 60)

    # Pass system_noise_fT to SQUID if specified
    if args.system_noise is not None:
        SQUID_PARAMS['system_noise_fT'] = args.system_noise

    for split_name, ids in [('train', TRAIN_IDS),
                             ('val', VAL_IDS),
                             ('test', TEST_IDS)]:
        print(f"\n--- {split_name} ({len(ids)} subjects) ---")
        out_path = os.path.join(out_dir, f"{split_name}.h5")
        generate_split(split_name, ids, params, out_path,
                       n_channels=args.n_channels, anatomy=args.anatomy)

    print("\n  Done ✓")


if __name__ == "__main__":
    main()
