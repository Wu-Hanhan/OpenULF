# -*- coding: utf-8 -*-
"""
ulf_params.py — Auto-configuration for ULF-MRI Parameters
============================================================

Computes physically consistent acquisition parameters (readout BW, TE)
from B₀ and imaging geometry, respecting gradient constraints.

References:
  - Zotev et al., JMR 194:115 (2008): B₀=46μT, G~140μT/m, readout~56ms
  - Vesanen et al., MRM 69:1795 (2013): B₀=50μT, G~100μT/m
"""
from __future__ import annotations
import numpy as np
from constants import GAMMA_H


def compute_readout_bw(B0: float, fov: float, n_readout: int,
                       gradient_fraction: float = 0.7) -> float:
    """Compute physically achievable readout bandwidth [Hz].

    The gradient field across the FOV must not exceed B₀ × gradient_fraction
    to avoid field reversal and severe concomitant field artifacts.

    Parameters
    ----------
    B0 : float
        Main field [T].
    fov : float
        Field of view [m].
    n_readout : int
        Number of readout samples.
    gradient_fraction : float
        Maximum G×FOV/B₀ ratio. Default 0.7 (Zotev used ~0.56).
        Higher values require concomitant field correction.

    Returns
    -------
    bw : float
        Readout bandwidth [Hz].
    """
    G_max = gradient_fraction * B0 / fov
    bw = GAMMA_H * G_max * fov
    # Ensure minimum BW of 500 Hz (practical lower limit)
    bw = max(bw, 500.0)
    # Cap at 250 kHz (clinical MRI upper limit)
    bw = min(bw, 250000.0)
    return bw


def compute_min_TE(readout_bw: float, n_readout: int,
                   margin_ms: float = 5.0) -> float:
    """Compute minimum TE for spin echo [ms].

    TE must accommodate half the readout window plus margins for
    RF pulses and gradient rephasing.

    Parameters
    ----------
    readout_bw : float
        Readout bandwidth [Hz].
    n_readout : int
        Number of readout samples.
    margin_ms : float
        Extra time for 180° pulse + gradient settling [ms].

    Returns
    -------
    TE : float
        Minimum echo time [ms].
    """
    readout_ms = n_readout / readout_bw * 1000
    return readout_ms / 2 + margin_ms


def auto_seq_params(B0: float, B_prepol: float = 0.1,
                    matrix_size: int = 128, fov: float = 0.256,
                    gradient_fraction: float = 0.7,
                    NEX: int = 10, t_pol: float = 1.0,
                    t_transfer: float = 0.1) -> dict:
    """Generate physically consistent sequence parameters for ULF-MRI.

    Parameters
    ----------
    B0 : float
        Main field [T].
    B_prepol : float
        Prepolarization field [T].
    matrix_size : int
        Image matrix size.
    fov : float
        Field of view [m].
    gradient_fraction : float
        Max G×FOV/B₀ (default 0.7).
    NEX : int
        Number of excitations / signal averages. Default 10.
        Zotev 2008: 6 repetitions at 64 matrix; we use 10 at 128 matrix
        to compensate for smaller voxel size. Scan time ~25 min.
    t_pol : float
        Prepolarization duration [s]. Default 1.0.
        Zotev 2008: 1.0s. At B_prepol=100mT, T1~300ms for WM,
        so 1.0s ≈ 3×T1 → 96% polarization efficiency.

    Returns
    -------
    params : dict
        Ready to pass to SpinEchoParams(**params).
    """
    n_readout = matrix_size  # 1× sampling
    bw = compute_readout_bw(B0, fov, n_readout, gradient_fraction)
    min_te = compute_min_TE(bw, n_readout)
    # Use min_TE rounded up to nearest 5ms
    te = max(min_te, 10.0)
    te = np.ceil(te / 5) * 5  # round up to 5ms

    return dict(
        B0=B0,
        B_prepol=B_prepol,
        matrix_size=matrix_size,
        fov=fov,
        TE=te,
        readout_bw=bw,
        NEX=NEX,
        t_pol=t_pol,
        t_transfer=t_transfer,
    )


if __name__ == "__main__":
    print("ULF-MRI Auto-Parameter Configuration")
    print("=" * 65)

    for B0, label in [(20e-6, "20μT"), (50e-6, "50μT"), (100e-6, "100μT"),
                      (200e-6, "200μT"), (1e-3, "1mT"), (3e-3, "3mT")]:
        p = auto_seq_params(B0)
        readout_ms = p['matrix_size'] / p['readout_bw'] * 1000
        G = p['readout_bw'] / (GAMMA_H * p['fov'])
        ratio = G * p['fov'] / B0
        print(f"\n  B0={label}:")
        print(f"    BW={p['readout_bw']:.0f}Hz, TE={p['TE']:.0f}ms, "
              f"readout={readout_ms:.0f}ms")
        print(f"    G={G*1e6:.0f}μT/m, G×FOV/B0={ratio:.2f}")

    print("\n  OK ✓")
