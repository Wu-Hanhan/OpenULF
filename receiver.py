# -*- coding: utf-8 -*-
"""
receiver.py — Receiver Base Infrastructure
============================================

Unified reciprocity-based acquisition pipeline shared by both
classical induction coil and SQUID quantum receivers.

The Hoult-Richards reciprocity principle provides the core signal model:
  Ψ = ∫ (B₁(r)/I₀) · M_⊥(r) d³r   [Wb]   (reciprocity flux)

This module provides:
  ReceiverBase         — abstract base with full acquisition pipeline
  MultiChannelReceiver — multi-channel array wrapper
  biot_savart_Bz       — exact B-field from circular current loop

References:
  [1] Hoult & Richards, JMR 24, 71 (1976) — reciprocity principle
  [2] Hoult, Concepts Magn. Reson. 12, 173 (2000) — B₁⁻ reciprocity
  [3] Jackson, "Classical Electrodynamics" 3rd ed. — Biot-Savart (§5.5)
  [4] Noll et al., IEEE TMI 10(4), 629 (1991) — time-segmented off-resonance
"""
from __future__ import annotations
import numpy as np
from typing import Optional
from scipy.special import ellipk, ellipe

from constants import MU_0
from forward import (SpinEchoParams, MRISignalEngine, _resize_float,
                     _HAS_TORCH, _HAS_CUDA, _DEFAULT_DEVICE)
if _HAS_TORCH:
    import torch


# ===================================================================
# Receiver Base
# ===================================================================
class ReceiverBase:
    """Unified reciprocity-based acquisition pipeline.

    Subclasses implement:
      compute_B1_per_I(N, fov) → B₁/I₀ map [T/A]
      process_readout_line(s_flux_Wb, rng) → k-space line [V]
    """

    def __init__(self, params: SpinEchoParams, seed: int = 42,
                 adc_bits: Optional[int] = None,
                 adc_fullscale_V: Optional[float] = None):
        """
        Parameters
        ----------
        params : SpinEchoParams
            Acquisition parameters.
        seed : int
            RNG seed for noise generation.
        adc_bits : int, optional
            ADC resolution in bits. If provided, the output voltage is
            quantized to 2^adc_bits uniform levels after all analog processing.
            Typical values: 14-16 (clinical MRI), 24 (SQUID readout).
            Default None = no quantization (infinite precision).
        adc_fullscale_V : float, optional
            ADC full-scale voltage range [V] (peak, not peak-to-peak).
            This is a fixed hardware property: the ADC digitizes the
            range [-adc_fullscale_V, +adc_fullscale_V].
            Signals exceeding this range are clipped (saturation).
            Must be set together with adc_bits to have any effect.
        """
        self.p = params
        self.seed = seed
        self.adc_bits = adc_bits
        self.adc_fullscale_V = adc_fullscale_V

    def compute_B1_per_I(self, N: int, fov: float) -> np.ndarray:
        """Compute receive sensitivity B₁/I₀ map [T/A].

        Must be implemented by subclasses. Returns an (N, N) array
        of B_z field per unit current at each imaging voxel.
        """
        raise NotImplementedError

    def process_readout_line(self, s_flux_Wb: np.ndarray,
                              rng: np.random.Generator) -> np.ndarray:
        """Process one readout line through the receiver chain.

        Must be implemented by subclasses.
        Input: reciprocity flux [Wb]. Output: voltage [V].
        """
        raise NotImplementedError

    def _pre_acquire(self):
        """Hook called before each acquire() or process_kspace() call.

        Subclasses can override to reset per-acquisition diagnostics.
        Default does nothing.
        """
        pass

    # ---------------------------------------------------------------
    # Batch properties (subclasses must implement)
    # ---------------------------------------------------------------
    @property
    def _batch_gain(self) -> float:
        """Scalar gain: flux [Wb] → voltage [V]. Subclasses implement."""
        raise NotImplementedError

    @property
    def _batch_H(self) -> np.ndarray:
        """Transfer function H(f), shape (n_readout,). Subclasses implement."""
        raise NotImplementedError

    @property
    def _batch_noise_psd(self) -> np.ndarray:
        """Noise PSD [V²/Hz], shape (n_readout,). Subclasses implement."""
        raise NotImplementedError

    # ---------------------------------------------------------------
    # Batched processing (GPU-accelerated when torch+CUDA available)
    # ---------------------------------------------------------------
    def _process_lines_batch(self, flux_batch: np.ndarray,
                              seed: int) -> np.ndarray:
        """Process multiple readout lines in one vectorized call.

        Parameters
        ----------
        flux_batch : np.ndarray, shape (N_lines, N_kx), complex
            Input flux [Wb] for all readout lines.
        seed : int
            RNG seed for noise generation.

        Returns
        -------
        output : np.ndarray, shape (N_lines, N_kx), complex128
            Output voltage [V] after receiver chain (and optional ADC).
        """
        N_lines, N_kx = flux_batch.shape
        gain = self._batch_gain
        H = self._batch_H
        S_noise = self._batch_noise_psd
        dt = self.p.dwell_time
        bw_bin = 1.0 / (N_kx * dt)
        noise_amp = np.sqrt(S_noise * N_kx * bw_bin / 2)

        if _HAS_TORCH:
            result = self._batch_torch(flux_batch, gain, H, noise_amp,
                                        seed, _DEFAULT_DEVICE)
        else:
            result = self._batch_numpy(flux_batch, gain, H, noise_amp, seed)

        if self.adc_bits is not None and self.adc_fullscale_V is not None:
            result = self._quantize(result, self.adc_bits, self.adc_fullscale_V)

        return result

    @staticmethod
    def _quantize(signal: np.ndarray, n_bits: int,
                   fullscale_V: float) -> np.ndarray:
        """Quantize complex signal with fixed hardware ADC model.

        Models a real ADC digitizer with fixed voltage range:
          - Input range: [-fullscale_V, +fullscale_V]
          - Resolution: 2^n_bits uniform levels
          - LSB = 2 × fullscale_V / 2^n_bits
          - Signals exceeding the range are CLIPPED (saturation)
          - Real and imaginary parts quantized independently
        """
        LSB = 2 * fullscale_V / (2 ** n_bits)
        re_clip = np.clip(signal.real, -fullscale_V, fullscale_V - LSB)
        im_clip = np.clip(signal.imag, -fullscale_V, fullscale_V - LSB)
        re_q = np.round(re_clip / LSB) * LSB
        im_q = np.round(im_clip / LSB) * LSB
        return re_q + 1j * im_q

    @staticmethod
    def _batch_numpy(flux: np.ndarray, gain: float,
                      H: np.ndarray, noise_amp: np.ndarray,
                      seed: int) -> np.ndarray:
        """Numpy batched path (CPU)."""
        N_lines, N_kx = flux.shape
        rng = np.random.default_rng(seed)

        s_V = flux * gain
        S_filt = np.fft.fft(s_V, axis=-1) * H[np.newaxis, :]

        noise_td = (rng.standard_normal((N_lines, N_kx)) +
                    1j * rng.standard_normal((N_lines, N_kx)))
        noise_fft = np.fft.fft(noise_td, axis=-1) * noise_amp[np.newaxis, :]

        return np.fft.ifft(S_filt + noise_fft, axis=-1)

    @staticmethod
    def _batch_torch(flux: np.ndarray, gain: float,
                      H: np.ndarray, noise_amp: np.ndarray,
                      seed: int, device: str) -> np.ndarray:
        """PyTorch batched path (GPU when device='cuda')."""
        N_lines, N_kx = flux.shape

        flux_t = torch.from_numpy(flux).to(device=device)
        H_t = torch.from_numpy(H.astype(np.complex128)).to(device=device)
        amp_t = torch.from_numpy(noise_amp.astype(np.complex128)).to(device=device)

        s_V = flux_t * gain
        S_filt = torch.fft.fft(s_V, dim=-1) * H_t.unsqueeze(0)

        gen = torch.Generator(device=device).manual_seed(seed)
        noise_re = torch.randn(N_lines, N_kx, generator=gen,
                                device=device, dtype=torch.float64)
        noise_im = torch.randn(N_lines, N_kx, generator=gen,
                                device=device, dtype=torch.float64)
        noise_td = torch.complex(noise_re, noise_im)
        noise_fft = torch.fft.fft(noise_td, dim=-1) * amp_t.unsqueeze(0)

        result = torch.fft.ifft(S_filt + noise_fft, dim=-1)
        return result.cpu().numpy()

    def acquire(self, engine: MRISignalEngine,
                B1_inhomogeneity: Optional[np.ndarray] = None,
                b0_map_hz: Optional[np.ndarray] = None,
                n_b0_segments: int = 8) -> np.ndarray:
        """Acquire k-space via reciprocity principle.

        Parameters
        ----------
        engine : MRISignalEngine
            Prepared signal engine (must call prepare() or prepare_fuzzy() first).
        B1_inhomogeneity : np.ndarray, optional, shape (N, N)
            Spatial B1+ transmit field map, normalized so that 1.0 = nominal
            flip angle. Values typically in range [0.7, 1.3].
            If None, uniform B1 is assumed.
        b0_map_hz : np.ndarray, optional, shape (H, W)
            Off-resonance frequency map [Hz] at each voxel.
            Use generate_b0_map() to create realistic smooth maps.
        n_b0_segments : int
            Number of time segments for off-resonance approximation.
            Reference: Noll et al., IEEE TMI 10(4), 629 (1991).

        Returns
        -------
        kspace : np.ndarray, shape (N, n_readout), complex128
            Acquired k-space in voltage units [V].
        """
        if not engine.is_prepared:
            raise RuntimeError(
                "MRISignalEngine has not been prepared. "
                "Call engine.prepare(tissue_map) or "
                "engine.prepare_fuzzy(fuzzy_maps) first.")

        self._pre_acquire()
        N = self.p.matrix_size
        n_ro = self.p.n_readout
        t = self.p.readout_time_axis()
        vv = self.p.voxel_volume
        NEX = self.p.NEX

        # Step 1: Reciprocity field [T/A]
        B1_I = self.compute_B1_per_I(N, self.p.fov)

        # Step 2: B1-weighted images in image domain
        weighted_images = {}
        for label, ksp in engine.tissue_kspaces.items():
            img = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(ksp)))
            if B1_inhomogeneity is not None:
                img = img * _resize_float(B1_inhomogeneity, N)
            weighted_images[label] = img * B1_I

        # Step 3: Build flux k-space [Wb]
        if b0_map_hz is None:
            flux_Wb = self._build_flux_fast(
                weighted_images, engine.tissue_T2star, t, N, n_ro)
        else:
            b0_hz = _resize_float(np.asarray(b0_map_hz, dtype=np.float64), N)
            flux_Wb = self._build_flux_b0(
                weighted_images, engine.tissue_T2star, t, N, n_ro,
                b0_hz, n_b0_segments)
        flux_Wb *= vv

        # Step 4: Batched receiver processing (GPU-accelerated if available)
        kspace_accum = np.zeros((N, n_ro), dtype=np.complex128)
        for nex_idx in range(NEX):
            kspace_accum += self._process_lines_batch(
                flux_Wb, seed=self.seed + nex_idx)

        return kspace_accum / NEX

    @staticmethod
    def _build_flux_fast(weighted_images, tissue_T2star, t, N, n_ro):
        """Build flux k-space without B0 inhomogeneity (fast path)."""
        flux = np.zeros((N, n_ro), dtype=np.complex128)
        for label, img_w in weighted_images.items():
            T2s = tissue_T2star.get(label, 0)
            if T2s <= 0:
                continue
            ksp_w = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img_w)))
            decay = np.exp(-np.abs(t) / (T2s * 1e-3))
            flux += ksp_w[:, :n_ro] * decay[np.newaxis, :n_ro]
        return flux

    @staticmethod
    def _build_flux_b0(weighted_images, tissue_T2star, t, N, n_ro,
                        b0_hz, n_segments):
        """Build flux k-space with B0 off-resonance (time-segmented).

        Reference: Noll et al., IEEE TMI 10(4), 629 (1991).
        """
        flux = np.zeros((N, n_ro), dtype=np.complex128)
        seg_bounds = np.linspace(0, n_ro, n_segments + 1, dtype=int)

        for seg_idx in range(n_segments):
            k_start = seg_bounds[seg_idx]
            k_end = seg_bounds[seg_idx + 1]
            if k_start >= k_end:
                continue

            k_mid = (k_start + k_end) // 2
            t_center = t[k_mid]
            phase_map = np.exp(-1j * 2 * np.pi * b0_hz * t_center)
            t_seg = t[k_start:k_end]

            for label, img_w in weighted_images.items():
                T2s = tissue_T2star.get(label, 0)
                if T2s <= 0:
                    continue
                img_mod = img_w * phase_map
                ksp_mod = np.fft.fftshift(
                    np.fft.fft2(np.fft.ifftshift(img_mod)))
                decay_seg = np.exp(-np.abs(t_seg) / (T2s * 1e-3))
                flux[:, k_start:k_end] += \
                    ksp_mod[:, k_start:k_end] * decay_seg[np.newaxis, :]

        return flux

    def process_kspace(self, kspace_Wb: np.ndarray,
                        seed: Optional[int] = None,
                        NEX: Optional[int] = None) -> np.ndarray:
        """Pass pre-computed k-space through receiver chain only.

        This is the decoupled receiver interface. It takes k-space data
        in physical flux units [Wb] and applies only the receiver-specific
        signal transduction and noise.

        Supports both 2D and 3D k-space:
          2D: (N_ky, N_kx) — standard single-slice Cartesian
          3D: (N_kz, N_ky, N_kx) — 3D volume Cartesian

        Parameters
        ----------
        kspace_Wb : np.ndarray, shape (..., N_kx), complex
            Input k-space in physical reciprocity flux units [Wb].
        seed : int, optional
            RNG seed. Defaults to self.seed.
        NEX : int, optional
            Number of averages. Defaults to self.p.NEX.

        Returns
        -------
        kspace_V : np.ndarray, same shape as input, complex128
            Output k-space in voltage units [V] after receiver chain.
        """
        if seed is None:
            seed = self.seed
        if NEX is None:
            NEX = self.p.NEX

        original_shape = kspace_Wb.shape
        N_kx = original_shape[-1]

        if N_kx != self.p.n_readout:
            raise ValueError(
                f"Input k-space has {N_kx} readout samples, but receiver "
                f"was configured for n_readout={self.p.n_readout}.")

        self._pre_acquire()
        ksp_2d = kspace_Wb.reshape(-1, N_kx)

        kspace_accum = np.zeros_like(ksp_2d, dtype=np.complex128)
        for nex_idx in range(NEX):
            kspace_accum += self._process_lines_batch(
                ksp_2d, seed=seed + nex_idx)

        return (kspace_accum / NEX).reshape(original_shape)

    @staticmethod
    def reconstruct(kspace: np.ndarray) -> np.ndarray:
        """Magnitude image from k-space."""
        return np.abs(np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(kspace))))


# ===================================================================
# Multi-Channel Receiver
# ===================================================================
class MultiChannelReceiver:
    """Multi-channel receiver array wrapping N single-channel receivers.

    Each channel is an independent ReceiverBase subclass instance
    with its own B₁ sensitivity profile, transfer function, noise,
    and independent RNG seed.

    NOISE CORRELATION NOTE:
      Each channel generates independent noise. In reality, sample
      noise is correlated between channels. For DL training data,
      independent noise per channel is a standard approximation.

    Parameters
    ----------
    channels : list of ReceiverBase
        List of N configured single-channel receivers.
        Each should have a unique seed for independent noise.
    """

    def __init__(self, channels: list):
        if not channels:
            raise ValueError("At least one channel required.")
        self.channels = channels
        self.n_channels = len(channels)
        p0 = channels[0].p
        for i, ch in enumerate(channels[1:], 1):
            if ch.p.n_readout != p0.n_readout:
                raise ValueError(
                    f"Channel {i} has n_readout={ch.p.n_readout}, "
                    f"but channel 0 has {p0.n_readout}.")

    def acquire(self, engine: 'MRISignalEngine',
                B1_inhomogeneity: Optional[np.ndarray] = None,
                b0_map_hz: Optional[np.ndarray] = None,
                n_b0_segments: int = 8) -> np.ndarray:
        """Acquire multi-channel k-space via reciprocity principle.

        Returns
        -------
        kspace : np.ndarray, shape (N_channels, N_ky, N_kx), complex128
        """
        return np.stack([
            ch.acquire(engine, B1_inhomogeneity, b0_map_hz, n_b0_segments)
            for ch in self.channels
        ])

    def process_kspace(self, kspace_Wb: np.ndarray,
                        seed: Optional[int] = None,
                        **kwargs) -> np.ndarray:
        """Pass k-space through all receiver channels.

        Each channel generates independent noise.

        Returns
        -------
        kspace_V : np.ndarray, shape (N_channels, ..., N_kx), complex128
        """
        results = []
        for i, ch in enumerate(self.channels):
            # Spacing of 1000 per channel to avoid overlap with NEX seeds
            ch_seed = (seed + i * 1000) if seed is not None else None
            results.append(ch.process_kspace(kspace_Wb, seed=ch_seed, **kwargs))
        return np.stack(results)

    def sensitivity_maps(self, N: int, fov: float) -> np.ndarray:
        """Compute per-channel B₁/I₀ sensitivity maps [T/A].

        Returns
        -------
        maps : np.ndarray, shape (N_channels, N, N)
        """
        return np.stack([
            ch.compute_B1_per_I(N, fov)
            for ch in self.channels
        ])

    @staticmethod
    def reconstruct_rss(kspaces: np.ndarray) -> np.ndarray:
        """Root-sum-of-squares combination of multi-channel images.

        Parameters
        ----------
        kspaces : np.ndarray, shape (N_channels, N_ky, N_kx)

        Returns
        -------
        image : np.ndarray, shape (N_ky, N_kx)
        """
        per_ch = np.abs(np.fft.ifftshift(
            np.fft.ifft2(np.fft.fftshift(kspaces, axes=(-2,-1)),
                         axes=(-2,-1)),
            axes=(-2,-1)))
        return np.sqrt(np.sum(per_ch**2, axis=0))


# ===================================================================
# Utilities
# ===================================================================
def biot_savart_Bz(r_coil: float, n_turns: int,
                    rho: np.ndarray, z: float) -> np.ndarray:
    """Axial magnetic field B_z/I from a circular current loop [T/A].

    Exact solution using complete elliptic integrals of the first
    and second kind. Valid at all positions (on-axis and off-axis).

    B_z(ρ,z) = (μ₀ n)/(2π) × 1/√((r+ρ)²+z²) ×
               [K(k²) + (r²-ρ²-z²)/((r-ρ)²+z²) × E(k²)]

    where k² = 4rρ / ((r+ρ)²+z²)

    Reference: Jackson, "Classical Electrodynamics" 3rd ed., §5.5.

    Parameters
    ----------
    r_coil : float
        Loop radius [m].
    n_turns : int
        Number of turns.
    rho : np.ndarray
        Radial distance from axis [m]. Shape (N, N) or (N,).
    z : float
        Axial distance from loop plane [m].

    Returns
    -------
    B_z_per_I : np.ndarray
        B_z field per unit current [T/A]. Same shape as rho.
    """
    r = r_coil
    n = n_turns
    rho = np.asarray(rho, dtype=np.float64)
    rho_safe = np.maximum(np.abs(rho), 1e-12)
    z_abs = abs(z)

    alpha2 = (r + rho_safe)**2 + z_abs**2
    beta2 = (r - rho_safe)**2 + z_abs**2

    k2 = 4 * r * rho_safe / alpha2
    k2 = np.clip(k2, 0, 1 - 1e-12)

    K = ellipk(k2)
    E = ellipe(k2)

    Bz = (MU_0 * n / (2 * np.pi)) / np.sqrt(alpha2) * \
         (K + (r**2 - rho_safe**2 - z_abs**2) / np.maximum(beta2, 1e-30) * E)

    return Bz


# ===================================================================
# Array Factory Functions
# ===================================================================
def hexagonal_positions(n_channels: int, spacing: float) -> list:
    """Generate (dx, dy) positions for a hexagonal array.

    Parameters
    ----------
    n_channels : int
        Number of channels. Must be 1, 7, or 19.
        1 = center only. 7 = center + 6 ring. 19 = center + 6 + 12 rings.
    spacing : float
        Center-to-center distance between adjacent channels [m].

    Returns
    -------
    positions : list of (float, float)
        (dx, dy) offsets from FOV center for each channel.
    """
    positions = [(0.0, 0.0)]  # center channel
    if n_channels >= 7:
        for k in range(6):
            angle = k * np.pi / 3
            positions.append((spacing * np.cos(angle),
                              spacing * np.sin(angle)))
    if n_channels >= 19:
        # Second ring: 12 positions at spacing * 2 and spacing * √3
        for k in range(6):
            angle = k * np.pi / 3
            positions.append((2 * spacing * np.cos(angle),
                              2 * spacing * np.sin(angle)))
        for k in range(6):
            angle = k * np.pi / 3 + np.pi / 6
            positions.append((np.sqrt(3) * spacing * np.cos(angle),
                              np.sqrt(3) * spacing * np.sin(angle)))
    return positions[:n_channels]


def make_squid_array(params, n_channels: int = 7,
                     base_seed: int = 42, anatomy: str = 'brain',
                     **squid_overrides) -> MultiChannelReceiver:
    """Create a multi-channel SQUID receiver array.

    Configurations based on published ULF-MRI systems:

    Brain (Zotev et al., Supercond. Sci. Tech. 2007):
      - 7-channel hexagonal, 2nd-order gradiometers
      - Pickup Ø37 mm, spacing 45 mm, baseline 60 mm
      - Standoff ~43 mm (dewar 25 mm + scalp/skull 13 mm + cortex 5 mm)
      - System noise 2.0 fT/√Hz per channel

    Body (extrapolated from brain system, larger geometry):
      - 7-channel hexagonal, larger pickup and spacing
      - Pickup Ø100 mm, spacing 110 mm
      - Standoff ~80 mm (dewar + body wall)

    Parameters
    ----------
    params : SpinEchoParams
        Acquisition parameters.
    n_channels : int
        Number of channels (1, 7, or 19). Default 7.
    base_seed : int
        Base RNG seed. Channel i gets seed = base_seed + i * 1000
        to ensure no overlap with NEX averaging seeds (which use
        seed, seed+1, ..., seed+NEX-1 per channel).
    anatomy : str
        'brain' or 'body'. Selects default geometry.
    **squid_overrides
        Override any SQUIDQuantumReceiver parameter.

    Returns
    -------
    MultiChannelReceiver
        Configured multi-channel SQUID array.
    """
    # Import here to avoid circular dependency
    from quantum_squid_receiver import SQUIDQuantumReceiver

    if anatomy == 'brain':
        defaults = dict(
            pickup_radius=0.0185,       # 18.5 mm (Zotev 2008, Ø37 mm)
            pickup_turns=1,
            coupling=0.5,
            gradiometer_order=2,
            gradiometer_baseline=0.06,  # 60 mm
            pickup_standoff=0.043,      # 43 mm: dewar 25mm + scalp/skull 13mm + cortex 5mm
        )
        spacing = 0.045                 # 45 mm center-to-center
    elif anatomy == 'body':
        defaults = dict(
            pickup_radius=0.050,        # 50 mm (larger for body)
            pickup_turns=1,
            coupling=0.5,
            gradiometer_order=2,
            gradiometer_baseline=0.06,
            pickup_standoff=0.080,      # 80 mm: dewar + body wall + organ depth
        )
        spacing = 0.110                 # 110 mm center-to-center
    else:
        raise ValueError(f"Unknown anatomy: {anatomy}. Use 'brain' or 'body'.")

    defaults.update(squid_overrides)
    positions = hexagonal_positions(n_channels, spacing)

    channels = []
    for i, (dx, dy) in enumerate(positions):
        # Seed spacing of 1000 per channel ensures no overlap with
        # NEX seeds (seed, seed+1, ..., seed+NEX-1) for any NEX ≤ 999
        ch = SQUIDQuantumReceiver(
            params, seed=base_seed + i * 1000,
            pickup_position=(dx, dy),
            **defaults)
        channels.append(ch)

    return MultiChannelReceiver(channels)


def make_coil_array(params, n_channels: int = 7,
                    base_seed: int = 1_000_042, anatomy: str = 'brain',
                    **coil_overrides) -> MultiChannelReceiver:
    """Create a multi-channel classical coil receiver array.

    Induction coils operate at room temperature (no dewar), so they can
    be placed much closer to tissue than SQUIDs. This partially
    compensates for their lower gain (EMF ∝ ω₀).

    Array spacing uses critical overlap rule (1.5 × coil_radius), which
    minimizes mutual inductance between adjacent elements — standard
    practice in clinical phased-array design.

    Configurations optimized for maximum correlation with ideal image
    while maintaining meaningful B₁ spatial complementarity with SQUID:

    Brain: r=45mm, standoff=15mm, spacing=67mm (critical overlap)
      - Larger coils capture more flux (essential at ULF)
      - Close standoff (no dewar) → better shallow-tissue sensitivity
      - B₁ correlation with SQUID ≈ 0.71 → 29% complementary info

    Body:  r=60mm, standoff=25mm, spacing=90mm (critical overlap)
      - Even larger coils for deeper body imaging
      - Standoff accounts for body wall thickness
      - B₁ correlation with SQUID ≈ 0.84 → 16% complementary info

    Parameters
    ----------
    params : SpinEchoParams
        Acquisition parameters.
    n_channels : int
        Number of channels (1, 7, or 19). Default 7.
    base_seed : int
        Base RNG seed. Offset from SQUID seed to ensure independent noise.
    anatomy : str
        'brain' or 'body'. Selects default geometry.
    **coil_overrides
        Override any ClassicalCoilReceiver parameter.

    Returns
    -------
    MultiChannelReceiver
        Configured multi-channel coil array.
    """
    from classical_coil_receiver import ClassicalCoilReceiver

    if anatomy == 'brain':
        defaults = dict(
            coil_radius=0.045,          # 45mm: optimal SNR/complementarity balance
            coil_standoff=0.015,        # 15mm: no dewar, scalp+skull+gap
            coil_turns=0,               # Auto-optimize
            wire_diameter=0.51e-3,      # AWG24 (Matlashov 2011). CRITICAL: default 2mm is wrong
        )
    elif anatomy == 'body':
        defaults = dict(
            coil_radius=0.060,          # 60mm: optimal for body FOV
            coil_standoff=0.025,        # 25mm: body wall + gap
            coil_turns=0,
            wire_diameter=0.51e-3,      # AWG24
        )
    else:
        raise ValueError(f"Unknown anatomy: {anatomy}. Use 'brain' or 'body'.")

    defaults.update(coil_overrides)
    # Critical overlap: spacing = 1.5 × radius (minimizes mutual inductance)
    coil_r = defaults.get('coil_radius', 0.045)
    spacing = 1.5 * coil_r
    positions = hexagonal_positions(n_channels, spacing)

    channels = []
    for i, (dx, dy) in enumerate(positions):
        # Seed spacing of 1000 per channel ensures no overlap with
        # NEX seeds (seed, seed+1, ..., seed+NEX-1) for any NEX ≤ 999
        ch = ClassicalCoilReceiver(
            params, seed=base_seed + i * 1000,
            coil_position=(dx, dy),
            **defaults)
        channels.append(ch)

    return MultiChannelReceiver(channels)
