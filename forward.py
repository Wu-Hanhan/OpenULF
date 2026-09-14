# -*- coding: utf-8 -*-
"""
forward.py — MRI Forward Signal Model
=======================================

Generates MRI signal (transverse magnetization M_⊥) from tissue maps
in physical SI units [A/m]. This is the forward model that maps
tissue parameters to the observable signal.

Supports two operating modes:
  Standard:      M_⊥ from thermal equilibrium at B0
  Prepolarized:  3-phase model (polarize → transfer → readout) for ULF-MRI

References:
  [1] Haacke et al., "MRI: Physical Principles" (1999) — signal model
  [2] Vesanen et al., MRM 69, 1795 (2013) — ULF-MRI prepolarization
  [3] Noll et al., IEEE TMI 10(4), 629 (1991) — time-segmented off-resonance
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict

from constants import GAMMA_H, M0_PER_TESLA
from tissue import TissueProperties, TISSUE_DB

# ===================================================================
# GPU Backend Detection
# ===================================================================
try:
    import torch
    _HAS_TORCH = True
    _HAS_CUDA = torch.cuda.is_available()
    _DEFAULT_DEVICE = 'cuda' if _HAS_CUDA else 'cpu'
except ImportError:
    _HAS_TORCH = False
    _HAS_CUDA = False
    _DEFAULT_DEVICE = 'cpu'


# ===================================================================
# Sequence Parameters
# ===================================================================
@dataclass
class SpinEchoParams:
    """Spin echo acquisition parameters with prepolarization model.

    Parameters
    ----------
    B0 : float
        Main magnetic field [T]. Must be >= 0.
        Typical: 50e-6 (ULF), 0.064 (Hyperfine), 1.5, 3.0, 7.0 T.
    B_prepol : float
        Prepolarization field [T]. Default 0 (standard equilibrium).
        ULF-MRI typical: 0.01-0.5 T. Ref: Vesanen et al. 2013.
    TR, TE : float
        Repetition/echo time [ms]. Default 500/20 ms.
    matrix_size : int
        Image matrix dimension. Default 256. Must be > 0.
    fov : float
        Field of view [m]. Default 0.256 (25.6 cm, head).
        Head: 0.20-0.28, Body: 0.30-0.50.
    slice_thickness : float
        Slice thickness [m]. Default 0.005 (5 mm).
    readout_bw : float
        Readout bandwidth [Hz]. Default 32000.
        Clinical 1.5-3T: 16-250 kHz. ULF: 4-64 kHz.
        Determines dwell_time = 1/readout_bw and pixel bandwidth.
    n_readout : int
        Number of readout samples. Default 0 → use matrix_size.
    NEX : int
        Number of excitations (signal averages). Default 1.
        SNR scales as √NEX.
    t_pol : float
        Prepolarization duration [s]. Default 2.0.
    t_transfer : float
        Delay between prepol switch-off and readout [s]. Default 0.1.
    """
    B0: float = 0.1
    B_prepol: float = 0.0
    TR: float = 500.0               # repetition time [ms]
    TE: float = 70.0                # echo time [ms] (ULF: ≥readout/2; clinical: 10-100)
    matrix_size: int = 256
    fov: float = 0.256              # field of view [m]
    slice_thickness: float = 0.005  # [m]
    readout_bw: float = 2000.0      # readout bandwidth [Hz] (ULF: ~2kHz; clinical: 16-250kHz)
    n_readout: int = 0
    NEX: int = 1                    # number of excitations (averages)
    t_pol: float = 2.0              # prepolarization duration [s]
    t_transfer: float = 0.1         # prepol-to-readout delay [s]

    def __post_init__(self):
        if self.B0 < 0:
            raise ValueError(f"B0 must be >= 0, got {self.B0}")
        if self.matrix_size <= 0:
            raise ValueError(f"matrix_size must be > 0, got {self.matrix_size}")
        if self.readout_bw <= 0:
            raise ValueError(f"readout_bw must be > 0, got {self.readout_bw}")
        if self.n_readout <= 0:
            self.n_readout = self.matrix_size
        self.f_larmor = GAMMA_H * self.B0
        self.omega_larmor = 2 * np.pi * self.f_larmor
        self.dx = self.fov / self.matrix_size
        self.dwell_time = 1.0 / self.readout_bw
        self.voxel_volume = self.dx**2 * self.slice_thickness

    def readout_time_axis(self) -> np.ndarray:
        """Time samples relative to echo center [s]."""
        return (np.arange(self.n_readout) - self.n_readout / 2) * self.dwell_time


# ===================================================================
# Signal Engine
# ===================================================================
class MRISignalEngine:
    """MRI signal generator in physical units [A/m].

    Supports two input modes:
      prepare(tissue_map)        — discrete (crisp) label map
      prepare_fuzzy(fuzzy_maps)  — per-tissue probability maps [0,1]
    """

    def __init__(self, params: SpinEchoParams):
        self.p = params
        self._tissue_kspaces: Dict[int, np.ndarray] = {}
        self._tissue_T2star: Dict[int, float] = {}
        self._M_physical: Optional[np.ndarray] = None

    # -----------------------------------------------------------
    # Prepolarization model
    # -----------------------------------------------------------
    def _compute_M_perp(self, tissue: TissueProperties) -> float:
        """Physical M_⊥ [A/m] for one tissue class.

        TWO OPERATING MODES:

        Standard (no prepol):
          Spin-echo steady state at B0:
          M_⊥ = M₀(B0) × PD × (1 - exp(-TR/T1)) × exp(-TE/T2)

        Prepolarized ULF-MRI:
          Each acquisition cycle consists of:
            1. Prepolarize at B_prepol for t_pol
            2. Switch off, wait t_transfer at B0
            3. RF excite (90°) and readout
          The effective cycle time = t_pol + t_transfer + readout.
          TR parameter is NOT used as SE steady-state recovery time,
          because the prepol actively provides magnetization each cycle.
          Reference: Vesanen et al., MRM 69, 1795 (2013).

          M_⊥ = M_avail × PD × exp(-TE/T2)
          where M_avail comes from the 3-phase prepol model:
            Phase 1: M_pol = M_eq(Bp) × (1 - exp(-t_pol / T1(Bp)))
            Phase 2: M_avail = M_eq(B0) + (M_pol - M_eq(B0)) × exp(-t_xfer / T1(B0))
        """
        p = self.p
        T1_B0 = tissue.T1_at_field(p.B0)
        T2 = tissue.T2_at_field(p.B0)

        if p.B_prepol > 0:
            T1_Bp = tissue.T1_at_field(p.B_prepol)
            M_eq_Bp = M0_PER_TESLA * p.B_prepol
            M_pol = M_eq_Bp * (1.0 - np.exp(-p.t_pol * 1e3 / T1_Bp))

            M_eq_B0 = M0_PER_TESLA * p.B0
            M_avail = M_eq_B0 + (M_pol - M_eq_B0) * np.exp(-p.t_transfer * 1e3 / T1_B0)

            return M_avail * tissue.PD * np.exp(-p.TE / T2)
        else:
            M0 = M0_PER_TESLA * p.B0
            return M0 * tissue.PD * (1.0 - np.exp(-p.TR / T1_B0)) * np.exp(-p.TE / T2)

    # -----------------------------------------------------------
    # Crisp (discrete label) preparation
    # -----------------------------------------------------------
    def prepare(self, tissue_map: np.ndarray):
        """Prepare from discrete tissue label map (integer per voxel)."""
        N = self.p.matrix_size
        padded = _resize(tissue_map, N)
        M_total = np.zeros((N, N))
        self._tissue_kspaces = {}
        self._tissue_T2star = {}

        for label, t in TISSUE_DB.items():
            if t.PD <= 0 or t.T1_3T <= 0:
                continue
            mask = padded == label
            if not np.any(mask):
                continue

            M_perp = self._compute_M_perp(t)
            img = np.zeros((N, N))
            img[mask] = M_perp

            self._tissue_kspaces[label] = np.fft.fftshift(
                np.fft.fft2(np.fft.ifftshift(img)))
            self._tissue_T2star[label] = t.T2star_at_field(self.p.B0)
            M_total += img

        self._M_physical = M_total

    # -----------------------------------------------------------
    # Fuzzy (probability map) preparation
    # -----------------------------------------------------------
    def prepare_fuzzy(self, fuzzy_maps: Dict[int, np.ndarray]):
        """Prepare from fuzzy tissue probability maps.

        Parameters
        ----------
        fuzzy_maps : dict
            {label: probability_map} where probability_map has values in [0,1]
            and shape (H, W). Labels correspond to TISSUE_DB keys.
            The maps represent partial volume fractions:
            fuzzy_maps[2][y,x] = 0.7 means voxel (y,x) is 70% grey matter.
        """
        N = self.p.matrix_size
        M_total = np.zeros((N, N))
        self._tissue_kspaces = {}
        self._tissue_T2star = {}

        for label, prob_map in fuzzy_maps.items():
            t = TISSUE_DB.get(label)
            if t is None or t.PD <= 0 or t.T1_3T <= 0:
                continue

            prob = _resize_float(prob_map, N)
            if np.max(prob) < 1e-6:
                continue

            M_perp = self._compute_M_perp(t)
            img = prob * M_perp

            self._tissue_kspaces[label] = np.fft.fftshift(
                np.fft.fft2(np.fft.ifftshift(img)))
            self._tissue_T2star[label] = t.T2star_at_field(self.p.B0)
            M_total += img

        self._M_physical = M_total

    @property
    def ideal_image(self) -> np.ndarray:
        """Ground truth M_⊥ [A/m]."""
        return self._M_physical

    @property
    def ideal_kspace(self) -> np.ndarray:
        """Sum of per-tissue k-space (no T2* decay)."""
        N = self.p.matrix_size
        return sum(self._tissue_kspaces.values(),
                   np.zeros((N, N), dtype=np.complex128))

    @property
    def tissue_kspaces(self) -> Dict[int, np.ndarray]:
        """Per-tissue k-space weighted by B₁-independent M_⊥ [A/m].

        Keys are tissue label integers (matching TISSUE_DB).
        Values are complex (N, N) arrays in image-domain Fourier space.
        Used by ReceiverBase.acquire() to build the reciprocity signal.
        """
        return self._tissue_kspaces

    @property
    def tissue_T2star(self) -> Dict[int, float]:
        """Per-tissue T2* values [ms] at current B0.

        Keys are tissue label integers. Values are T2* in milliseconds.
        """
        return self._tissue_T2star

    @property
    def is_prepared(self) -> bool:
        """Whether prepare() or prepare_fuzzy() has been called."""
        return self._M_physical is not None


# ===================================================================
# Utilities
# ===================================================================
def generate_b0_map(N: int, fov: float, max_hz: float = 20.0,
                     order: int = 2, seed: Optional[int] = None) -> np.ndarray:
    """Generate a smooth B0 off-resonance map for MRI simulation.

    Creates spatially smooth field inhomogeneity using random low-order
    polynomial basis functions. Mimics real B0 variations from imperfect
    shimming (ULF) or tissue susceptibility (high field).

    The generated map has zero mean (net shimmed) with smooth spatial
    structure peaking at ±max_hz.

    Parameters
    ----------
    N : int
        Matrix size.
    fov : float
        Field of view [m].
    max_hz : float
        Peak off-resonance frequency [Hz].
        Typical values:
          ULF 50μT: 5-20 Hz (Helmholtz coil imperfection ~0.5-1%)
          0.1T:     20-50 Hz
          1.5T:     50-100 Hz (shimming residual)
          3T:       100-200 Hz (tissue susceptibility, air-tissue interfaces)
    order : int
        Maximum polynomial order (2=quadratic, 3=cubic, 4=quartic).
        Higher order creates more spatial structure.
        Recommended: 2-3 for realistic shimming residuals.
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    b0_map : np.ndarray, shape (N, N), float64
        Off-resonance frequency [Hz] at each voxel.
        Zero mean, peak-to-peak range approximately 2 × max_hz.
    """
    rng = np.random.default_rng(seed)

    x = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, x)

    b0 = np.zeros((N, N))
    for px in range(order + 1):
        for py in range(order + 1):
            if px + py < 1 or px + py > order:
                continue
            coeff = rng.standard_normal()
            b0 += coeff * (X ** px) * (Y ** py)

    b0 -= np.mean(b0)
    peak = np.max(np.abs(b0))
    if peak > 0:
        b0 *= max_hz / peak

    return b0


def _resize(img: np.ndarray, N: int) -> np.ndarray:
    """Center-crop/pad integer label map to N×N."""
    H, W = img.shape
    out = np.zeros((N, N), dtype=img.dtype)
    h, w = min(H, N), min(W, N)
    out[max(0,(N-H)//2):max(0,(N-H)//2)+h,
        max(0,(N-W)//2):max(0,(N-W)//2)+w] = \
        img[max(0,(H-N)//2):max(0,(H-N)//2)+h,
            max(0,(W-N)//2):max(0,(W-N)//2)+w]
    return out


def _resize_float(img: np.ndarray, N: int) -> np.ndarray:
    """Center-crop/pad float map to N×N with zero padding."""
    H, W = img.shape
    out = np.zeros((N, N), dtype=np.float64)
    h, w = min(H, N), min(W, N)
    out[max(0,(N-H)//2):max(0,(N-H)//2)+h,
        max(0,(N-W)//2):max(0,(N-W)//2)+w] = \
        img[max(0,(H-N)//2):max(0,(H-N)//2)+h,
            max(0,(W-N)//2):max(0,(W-N)//2)+w]
    return out


# ===================================================================
# Self-test
# ===================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  forward.py self-test")
    print("=" * 60)

    print(f"\n  M₀/B = {M0_PER_TESLA:.4e} A/m per Tesla")

    N = 64
    tissue = np.full((N, N), 3, dtype=int)
    Y, X = np.ogrid[-N//2:N//2, -N//2:N//2]
    tissue[X**2 + Y**2 < (N//4)**2] = 2
    tissue[X**2 + Y**2 < (N//8)**2] = 1

    print("\n  [Crisp] Standard:")
    for B0 in [3.0, 0.1]:
        p = SpinEchoParams(B0=B0, matrix_size=N)
        e = MRISignalEngine(p); e.prepare(tissue)
        print(f"    B0={B0}T: M_max={e.ideal_image.max():.4e} A/m")

    print("\n  [Crisp] Prepolarization at 50μT:")
    for t_pol, t_xfer in [(2.0, 0.1), (0.5, 0.1), (2.0, 0.3)]:
        p = SpinEchoParams(B0=50e-6, B_prepol=0.1, matrix_size=N,
                           t_pol=t_pol, t_transfer=t_xfer)
        e = MRISignalEngine(p); e.prepare(tissue)
        print(f"    t_pol={t_pol}s, t_xfer={t_xfer}s: "
              f"M_max={e.ideal_image.max():.4e} A/m")

    print("\n  [Fuzzy] Synthetic partial volume:")
    fuzzy = {}
    fuzzy[3] = np.zeros((N, N)); fuzzy[3][:] = 0.6
    fuzzy[2] = np.zeros((N, N)); fuzzy[2][:] = 0.3
    fuzzy[1] = np.zeros((N, N)); fuzzy[1][:] = 0.1
    p = SpinEchoParams(B0=50e-6, B_prepol=0.1, matrix_size=N)
    e_fuzzy = MRISignalEngine(p); e_fuzzy.prepare_fuzzy(fuzzy)
    print(f"    Uniform 60%WM+30%GM+10%CSF: M_max={e_fuzzy.ideal_image.max():.4e} A/m")

    p_ideal = SpinEchoParams(B0=50e-6, B_prepol=0.1, matrix_size=N,
                              t_pol=100.0, t_transfer=0.0)
    e_i = MRISignalEngine(p_ideal); e_i.prepare(tissue)
    p_real = SpinEchoParams(B0=50e-6, B_prepol=0.1, matrix_size=N,
                             t_pol=2.0, t_transfer=0.1)
    e_r = MRISignalEngine(p_real); e_r.prepare(tissue)
    ratio = e_r.ideal_image.max() / max(e_i.ideal_image.max(), 1e-30)
    print(f"\n  Prepol signal: realistic/ideal = {ratio:.3f} "
          f"({(1-ratio)*100:.1f}% loss)")

    print("\n  OK ✓")
