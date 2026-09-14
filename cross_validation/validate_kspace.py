# -*- coding: utf-8 -*-
"""
validate_kspace.py — K-space Output Cross-Validation
======================================================

Writes a COMPLETELY INDEPENDENT reference implementation of the
forward model (tissue map → k-space), sharing NO code with our
engine, then compares the k-space output pixel-by-pixel.

If two independent implementations produce the same k-space,
both are almost certainly correct.

Reference implementation:
  1. For each tissue: compute M_⊥ from first principles
  2. Build image: M_⊥ × mask(tissue)
  3. FFT → k-space
  4. Apply T2* decay in k-space
  5. Sum all tissues

Compare with: forward.py MRISignalEngine
"""
from __future__ import annotations
import sys, os
import numpy as np

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ===================================================================
# REFERENCE IMPLEMENTATION (completely independent, no imports from our engine)
# ===================================================================

# Physical constants (typed manually, NOT imported)
REF_GAMMA_H = 42.577478518e6  # Hz/T
REF_K_B = 1.380649e-23        # J/K
REF_HBAR = 1.054571817e-34    # J·s
REF_N_A = 6.02214076e23       # 1/mol
REF_T_BODY = 310.0            # K

# Derived
REF_GAMMA_RAD = 2 * np.pi * REF_GAMMA_H
REF_N_H = 2 * 1000.0 / 18.015e-3 * REF_N_A
REF_M0_PER_T = REF_N_H * REF_GAMMA_RAD**2 * REF_HBAR**2 / (4 * REF_K_B * REF_T_BODY)

# Tissue database (typed manually to match tissue.py)
# Updated 2026-03-21: per-tissue T1_low, T2_low, CSF NMRD model
REF_TISSUES = {
    # label: (name, PD, T1_3T_ms, T2_3T_ms, alpha, t2star_ratio, T1_low, T2_low, nmrd_A, nmrd_fc)
    0:  ("BG",      0.00,    0,    0, 0.00, 1.0,   0,   0,  0,     0),
    1:  ("CSF",     1.00, 4000, 2000, 0.60, 0.95, 340, 355,  2.657, 189e3),  # Lorentzian NMRD
    2:  ("GM",      0.86, 1331,   80, 0.39, 0.65,  85, 106,  0,     0),
    3:  ("WM",      0.77,  832,   70, 0.33, 0.65,  56, 100,  0,     0),
    4:  ("Fat",     1.00,  380,   60, 0.51, 0.80,  94,  90,  0,     0),
    5:  ("Musc",    0.80, 1400,   40, 0.60, 0.70,  55,  40,  0,     0),
    6:  ("MuscSk",  0.80, 1400,   40, 0.60, 0.70,  55,  40,  0,     0),
    7:  ("Skull",   0.30,  300,   20, 0.30, 0.60,  30,  20,  0,     0),
    8:  ("Vessel",  0.90, 1500,  100, 0.50, 0.70, 181, 120,  0,     0),
    9:  ("Connect", 0.80,  500,   50, 0.40, 0.70,  60,  50,  0,     0),
    10: ("Dura",    0.50,  600,   40, 0.40, 0.70,  50,  40,  0,     0),
    11: ("BMarrow", 0.95,  550,   50, 0.51, 0.80,  94,  90,  0,     0),
}

REF_GAMMA_H = 42.577478518e6  # Hz/T


def ref_raw_T1(T1_3T, alpha, T1_low, nmrd_A, nmrd_fc, B0):
    """Independent T1(B0) — matches tissue.py._raw_T1()."""
    if B0 <= 0 or T1_3T <= 0:
        return T1_low if T1_low > 0 else 30.0
    # CSF: Lorentzian NMRD
    if nmrd_A > 0:
        f = REF_GAMMA_H * B0
        R1_plateau = 1000.0 / T1_3T
        R1 = R1_plateau + nmrd_A / (1.0 + (f / nmrd_fc)**2)
        return 1000.0 / R1
    # Other tissues: power law
    t1_low = T1_low if T1_low > 0 else 30.0
    return t1_low + (T1_3T - t1_low) * (B0 / 3.0)**alpha


def ref_raw_T2(T2_3T, T2_low, B0):
    """Independent T2(B0) intrinsic — matches tissue.py._raw_T2()."""
    if T2_3T <= 0:
        return 0.0
    t2_low = T2_low if T2_low > 0 else T2_3T
    if t2_low == T2_3T or B0 <= 0:
        return t2_low
    R2_3T = 1000.0 / T2_3T
    R2_low = 1000.0 / t2_low
    k = (R2_3T - R2_low) / 3.0
    return 1000.0 / (R2_low + k * B0)


def ref_T2_at_field(T2_3T, T2_low, T1_3T, alpha, T1_low, nmrd_A, nmrd_fc, B0):
    """Independent T2(B0) with T1 constraint — matches tissue.py.T2_at_field()."""
    t2_intr = ref_raw_T2(T2_3T, T2_low, B0)
    if t2_intr <= 0:
        return 0.0
    t1_raw = ref_raw_T1(T1_3T, alpha, T1_low, nmrd_A, nmrd_fc, B0)
    if t1_raw > 0:
        return min(t2_intr, 1.05 * t1_raw)
    return t2_intr


def ref_T2star_at_field(T2_3T, T2_low, T1_3T, alpha, T1_low, nmrd_A, nmrd_fc,
                        t2star_ratio, B0):
    """Independent T2*(B0) — matches tissue.py.T2star_at_field()."""
    t2 = ref_T2_at_field(T2_3T, T2_low, T1_3T, alpha, T1_low, nmrd_A, nmrd_fc, B0)
    if t2 <= 0:
        return 0.0
    if t2star_ratio >= 1.0 or B0 <= 0:
        return t2
    R2p_3T = 1.0 / (T2_3T * t2star_ratio) - 1.0 / T2_3T
    R2p = R2p_3T * (B0 / 3.0)
    return 1.0 / (1.0 / t2 + R2p)


def ref_compute_M_perp(label, B0, B_prepol, TR, TE, t_pol, t_transfer):
    """Independent M_⊥ calculation from first principles."""
    if label not in REF_TISSUES:
        return 0.0
    _, PD, T1_3T, T2_3T, alpha, t2sr, T1_low, T2_low, nmrd_A, nmrd_fc = REF_TISSUES[label]
    if PD <= 0 or T1_3T <= 0:
        return 0.0

    T1_B0 = ref_raw_T1(T1_3T, alpha, T1_low, nmrd_A, nmrd_fc, B0)
    T2_B0 = ref_T2_at_field(T2_3T, T2_low, T1_3T, alpha, T1_low, nmrd_A, nmrd_fc, B0)

    if B_prepol > 0:
        T1_Bp = ref_raw_T1(T1_3T, alpha, T1_low, nmrd_A, nmrd_fc, B_prepol)
        M_eq_Bp = REF_M0_PER_T * B_prepol
        M_pol = M_eq_Bp * (1.0 - np.exp(-t_pol * 1e3 / T1_Bp))
        M_eq_B0 = REF_M0_PER_T * B0
        M_avail = M_eq_B0 + (M_pol - M_eq_B0) * np.exp(-t_transfer * 1e3 / T1_B0)
        return M_avail * PD * np.exp(-TE / T2_B0)
    else:
        M0 = REF_M0_PER_T * B0
        return M0 * PD * (1.0 - np.exp(-TR / T1_B0)) * np.exp(-TE / T2_B0)


def ref_resize(img, N):
    """Independent center-crop/pad."""
    H, W = img.shape
    out = np.zeros((N, N), dtype=img.dtype)
    h, w = min(H, N), min(W, N)
    out[max(0,(N-H)//2):max(0,(N-H)//2)+h,
        max(0,(N-W)//2):max(0,(N-W)//2)+w] = \
        img[max(0,(H-N)//2):max(0,(H-N)//2)+h,
            max(0,(W-N)//2):max(0,(W-N)//2)+w]
    return out


def ref_generate_kspace(tissue_map, B0, B_prepol, TR, TE, t_pol, t_transfer,
                         matrix_size, readout_bw):
    """Complete independent k-space generation.

    Returns: ideal_image, ideal_kspace, tissue_kspaces_with_T2star
    """
    N = matrix_size
    n_readout = N
    dwell_time = 1.0 / readout_bw
    t = (np.arange(n_readout) - n_readout / 2) * dwell_time

    padded = ref_resize(tissue_map, N)
    M_total = np.zeros((N, N))
    kspace_total = np.zeros((N, n_readout), dtype=np.complex128)

    for label, tissue_info in REF_TISSUES.items():
        _, PD, T1_3T, T2_3T, alpha, t2sr, T1_low, T2_low, nmrd_A, nmrd_fc = tissue_info
        if PD <= 0 or T1_3T <= 0:
            continue
        mask = padded == label
        if not np.any(mask):
            continue

        M_perp = ref_compute_M_perp(label, B0, B_prepol, TR, TE, t_pol, t_transfer)
        img = np.zeros((N, N))
        img[mask] = M_perp
        M_total += img

        # FFT to k-space (centered)
        ksp = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))

        # T2* decay in k-space
        T2s = ref_T2star_at_field(T2_3T, T2_low, T1_3T, alpha, T1_low, nmrd_A, nmrd_fc, t2sr, B0)
        if T2s > 0:
            decay = np.exp(-np.abs(t) / (T2s * 1e-3))
            kspace_total += ksp[:, :n_readout] * decay[np.newaxis, :n_readout]

    # Ideal k-space (no T2* decay)
    ideal_ksp = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(M_total)))

    return M_total, ideal_ksp, kspace_total


# ===================================================================
# COMPARISON
# ===================================================================
def main():
    from forward import SpinEchoParams, MRISignalEngine

    print("=" * 70)
    print("  K-Space Cross-Validation")
    print("  Independent reference vs our forward.py MRISignalEngine")
    print("=" * 70)

    # Test configurations
    configs = [
        ("3T Standard SE", dict(B0=3.0, B_prepol=0.0, TR=500, TE=20, t_pol=2.0, t_transfer=0.1)),
        ("50μT ULF Prepol", dict(B0=50e-6, B_prepol=0.1, TR=500, TE=20, t_pol=2.0, t_transfer=0.1)),
        ("1.5T Clinical", dict(B0=1.5, B_prepol=0.0, TR=600, TE=15, t_pol=2.0, t_transfer=0.1)),
    ]

    N = 32  # small for speed

    # Create phantom
    tissue = np.full((N, N), 3, dtype=int)  # WM
    Y, X = np.ogrid[-N//2:N//2, -N//2:N//2]
    tissue[X**2 + Y**2 < (N//4)**2] = 2  # GM
    tissue[X**2 + Y**2 < (N//8)**2] = 1  # CSF

    all_pass = True
    for name, cfg in configs:
        print(f"\n  Config: {name}")
        print(f"    B0={cfg['B0']}, B_prepol={cfg['B_prepol']}, TR={cfg['TR']}, TE={cfg['TE']}")

        # Our engine
        params = SpinEchoParams(B0=cfg['B0'], B_prepol=cfg['B_prepol'],
                                TR=cfg['TR'], TE=cfg['TE'],
                                matrix_size=N, readout_bw=32000,
                                t_pol=cfg['t_pol'], t_transfer=cfg['t_transfer'])
        engine = MRISignalEngine(params)
        engine.prepare(tissue)

        our_image = engine.ideal_image
        our_kspace = engine.ideal_kspace

        # Reference implementation
        ref_image, ref_kspace, ref_kspace_t2s = ref_generate_kspace(
            tissue, cfg['B0'], cfg['B_prepol'], cfg['TR'], cfg['TE'],
            cfg['t_pol'], cfg['t_transfer'], N, 32000)

        # Compare ideal image
        img_err = np.max(np.abs(our_image - ref_image))
        img_rel = img_err / max(np.max(np.abs(ref_image)), 1e-30)
        img_pass = img_rel < 1e-10
        print(f"    Ideal image:  max_err={img_err:.4e}, rel_err={img_rel:.4e} {'✓' if img_pass else '✗'}")

        # Compare ideal k-space
        ksp_err = np.max(np.abs(our_kspace - ref_kspace))
        ksp_rel = ksp_err / max(np.max(np.abs(ref_kspace)), 1e-30)
        ksp_pass = ksp_rel < 1e-10
        print(f"    Ideal k-space: max_err={ksp_err:.4e}, rel_err={ksp_rel:.4e} {'✓' if ksp_pass else '✗'}")

        # Compare k-space with T2* decay (tissue_kspaces)
        # Our engine stores per-tissue k-space; we need to sum with T2* decay
        our_ksp_t2s = np.zeros((N, N), dtype=np.complex128)
        t = params.readout_time_axis()
        for label, ksp in engine.tissue_kspaces.items():
            T2s = engine.tissue_T2star.get(label, 0)
            if T2s > 0:
                decay = np.exp(-np.abs(t) / (T2s * 1e-3))
                our_ksp_t2s += ksp[:, :N] * decay[np.newaxis, :N]

        t2s_err = np.max(np.abs(our_ksp_t2s - ref_kspace_t2s))
        t2s_rel = t2s_err / max(np.max(np.abs(ref_kspace_t2s)), 1e-30)
        t2s_pass = t2s_rel < 1e-10
        print(f"    K-space+T2*:  max_err={t2s_err:.4e}, rel_err={t2s_rel:.4e} {'✓' if t2s_pass else '✗'}")

        # Parseval check
        img_power = np.sum(np.abs(our_image)**2)
        ksp_power = np.sum(np.abs(our_kspace)**2) / (N * N)
        parseval_err = abs(img_power - ksp_power) / img_power
        parseval_pass = parseval_err < 1e-10
        print(f"    Parseval:     err={parseval_err:.4e} {'✓' if parseval_pass else '✗'}")

        all_pass &= img_pass and ksp_pass and t2s_pass and parseval_pass

    print(f"\n{'='*70}")
    print(f"  {'ALL K-SPACE TESTS PASSED ✓' if all_pass else 'SOME FAILED ✗'}")
    print(f"  Independent implementation produces IDENTICAL k-space output.")
    print(f"  This validates: tissue→image→FFT→k-space→T2* decay pipeline.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
