# -*- coding: utf-8 -*-
"""
classical_coil_receiver.py — Classical Induction Coil (Reciprocity Model)
==========================================================================

Signal chain from reciprocity flux Ψ [Wb] to output voltage:

  Ψ [Wb] (from unified forward model, same input as SQUID)
    → EMF = ω₀ × Ψ  [V]            Faraday induction: EMF = -dΦ/dt
    → H(f): transfer function       Tuned bandpass or untuned flat
    → + V_noise(f)                  Johnson + sample + preamp noise
    → output [V]

Two front-end modes (selected automatically by frequency):
  TUNED (f₀/Q >= BW):   LC bandpass, NF-based preamp noise
  UNTUNED (f₀/Q < BW):  flat transfer, broadband preamp e_n/i_n noise

References:
    [1] Hoult & Richards, JMR 24, 71 (1976) — reciprocity, EMF
    [2] Edelstein et al., MRM 3, 604 (1986) — SNR, sample noise
    [3] Hoult & Lauterbur, JMR 34, 425 (1979) — R_sample formula
    [4] Seton et al., IEEE Trans. Appl. Supercond. 5, 3218 (1995) — untuned ULF
    [5] Myers et al., J. Appl. Phys. 103, 07E912 (2007) — broadband preamp noise
    [6] Grover, "Inductance Calculations" (1946) — coil inductance
    [7] Jackson, "Classical Electrodynamics" 3rd ed. — Biot-Savart (§5.5)
    [8] Friis, Proc. IRE 32, 419 (1944) — noise figure definition
    [9] Pozar, "Microwave Engineering" 4th ed. — skin effect (§1.7)
"""
from __future__ import annotations
import numpy as np
from constants import GAMMA_H, K_B, MU_0
from forward import SpinEchoParams, MRISignalEngine
from receiver import ReceiverBase, biot_savart_Bz


class ClassicalCoilReceiver(ReceiverBase):
    """Classical induction coil receiver with tuned/untuned front-end.

    Parameters
    ----------
    params : SpinEchoParams
        Acquisition parameters.
    coil_radius : float
        Coil radius [m]. Default 0.10 m (10 cm surface coil).
        Typical: 0.03-0.05 (small surface), 0.10-0.15 (head),
        0.20-0.25 (body). Ref: Hoult & Richards 1976.
    coil_turns : int
        Number of turns. Default 2.
        Surface coil: 1-4, Solenoid: 10-100.
        SNR ∝ n for small n, but parasitic capacitance limits high n.
    coil_Q : float
        Unloaded quality factor. Default 50.
        Typical range: unloaded 100-300, body-loaded 20-100.
        Scales roughly with frequency. Ref: Edelstein et al. 1986.
    wire_resistivity : float
        Wire resistivity [Ω·m]. Default 1.68e-8 (copper at 295K).
        Copper: 1.68e-8, Silver: 1.59e-8.
    wire_diameter : float
        Wire diameter [m]. Default 2.0e-3 (2mm).
        Typical: 0.5-3 mm. Affects skin-effect resistance.
    coil_standoff : float or None
        Distance from coil plane to imaging slice center [m].
        Default: 0.5 × coil_radius.
        Room-temperature coil sits on scalp (~5 mm gap) +
        skull/scalp thickness (~15 mm) + depth to cortical slice
        (~5 mm) ≈ 25 mm for a 50 mm radius coil (ratio ~0.5).
    coil_position : tuple of (float, float)
        Coil center offset from FOV center [m]: (dx, dy).
        Default (0, 0). For multi-channel arrays, each coil has
        a different offset.
    T_coil : float
        Coil temperature [K]. Default 295 (room temperature).
        Room temp: 293-298, Cryogenic copper: 77 (LN2) or 4.2 (LHe).
    T_sample : float
        Sample (body) temperature [K]. Default 310 (body, 37°C).
    sample_sigma : float
        Average tissue conductivity [S/m]. Default 0.4.
        Brain: 0.3-0.5, Muscle: 0.5-0.7, Fat: 0.03-0.05.
        Note: real conductivity is frequency-dependent (Gabriel et al.
        1996). Default 0.4 is appropriate for ULF (1-10 kHz).
        At high field (>1T), σ increases to ~0.5-0.7 S/m.
        Ref: Gabriel et al. 1996; Hoult & Lauterbur 1979.
    sample_radius : float
        Effective sample radius [m]. Default 0.09 (head).
        Head: 0.07-0.10, Torso: 0.15-0.20.
        Determines sample noise via R_sample ∝ σ × ω² × r_s^5 / r_c.
        Ref: Hoult & Lauterbur 1979.
    preamp_NF_dB : float
        Preamplifier noise figure [dB] (used in TUNED mode only).
        Default 1.0. Good clinical: 0.5-1.0, Standard: 1.0-2.0.
        Ref: Edelstein et al. 1986.
    preamp_en : float
        Preamplifier voltage noise density [V/√Hz] (UNTUNED mode only).
        Default 1.0e-9 (1 nV/√Hz). White noise floor above 1/f corner.
        Low-noise JFET: 0.5-2 nV/√Hz. Ref: Seton et al. 1995; Myers 2007.
    preamp_in : float
        Preamplifier current noise density [A/√Hz] (UNTUNED mode only).
        Default 2.0e-12 (2 pA/√Hz).
        Low-noise JFET: 1-5 pA/√Hz. Ref: Myers et al. 2007.
    preamp_f1f : float
        Preamplifier 1/f noise corner frequency [Hz] (UNTUNED mode only).
        Default 10.0. Below this frequency, e_n rises as √(f_1f/f).
        Low-noise JFET: 2-50 Hz. Op-amp: 10-1000 Hz.
        At ULF signal frequencies (~2 kHz), 1/f contribution is small
        but included for completeness. Ref: Motchenbacher & Connelly,
        "Low-Noise Electronic System Design" (1993).
    adc_bits : int, optional
        ADC resolution [bits]. Default None (no quantization).
        Clinical MRI: 14-16. Research: 16-24.
        Only effective when adc_fullscale_V is also set.
    adc_fullscale_V : float, optional
        ADC full-scale voltage [V] (positive peak). Default None.
        The ADC digitizes [-fullscale, +fullscale].
        Must match the analog gain chain of the actual hardware.
        Typical after preamp: 0.1-1.0 V.
    seed : int
        RNG seed for reproducible noise.
    """

    def __init__(self, params: SpinEchoParams,
                 coil_radius: float = 0.10, coil_turns: int = 0,
                 coil_Q: float = 50.0,
                 wire_resistivity: float = 1.68e-8,
                 wire_diameter: float = 2.0e-3,
                 coil_standoff: float = None,
                 coil_position: tuple = (0.0, 0.0),
                 coil_inner_radius: float = None,
                 T_coil: float = 295.0, T_sample: float = 310.0,
                 sample_sigma: float = 0.4, sample_radius: float = 0.09,
                 preamp_NF_dB: float = 1.0,
                 preamp_en: float = 1.3e-9,
                 preamp_in: float = 0.8e-12,
                 preamp_f1f: float = 10.0,
                 adc_bits: int = None,
                 adc_fullscale_V: float = None,
                 seed: int = 42):
        super().__init__(params, seed, adc_bits=adc_bits,
                         adc_fullscale_V=adc_fullscale_V)

        # Validate
        if coil_radius <= 0:
            raise ValueError(f"coil_radius must be > 0, got {coil_radius}")
        if coil_Q <= 0:
            raise ValueError(f"coil_Q must be > 0, got {coil_Q}")
        if T_coil <= 0 or T_sample <= 0:
            raise ValueError(f"Temperatures must be > 0 K")

        self.coil_radius = coil_radius  # outer radius
        self.coil_Q_nominal = coil_Q
        self.wire_resistivity = wire_resistivity
        self.wire_diameter = wire_diameter
        self.coil_standoff = coil_standoff if coil_standoff is not None else coil_radius * 0.5
        self.coil_position = coil_position
        self.T_coil = T_coil
        self.T_sample = T_sample
        self.sample_sigma = sample_sigma
        self.sample_radius = sample_radius
        self.preamp_NF_dB = preamp_NF_dB
        self.preamp_en = preamp_en
        self.preamp_in = preamp_in
        self.preamp_f1f = preamp_f1f

        # Pancake coil geometry: inner_radius defaults to radius/5
        # (reasonable for a flat spiral/pancake winding)
        self.coil_inner_radius = coil_inner_radius if coil_inner_radius is not None \
            else coil_radius / 5.0
        self.coil_avg_radius = (self.coil_inner_radius + self.coil_radius) / 2.0

        # Auto-optimize coil_turns if 0
        if coil_turns <= 0:
            self.coil_turns = self._auto_optimize_turns()
        else:
            self.coil_turns = coil_turns

        self._setup()

    # ---------------------------------------------------------------
    # Auto-optimization
    # ---------------------------------------------------------------
    def _auto_optimize_turns(self) -> int:
        """Compute optimal coil turns for maximum SNR.

        Balances signal (∝ n) against noise (e_n + i_n×Z + Johnson).
        Uses pancake inductance model L = μ₀ × n² × r_avg, validated
        against Matlashov 2011 (70 mH measured vs 67.7 mH predicted, 3% error).

        Returns optimal integer number of turns.
        """
        omega = self.p.omega_larmor
        r_avg = self.coil_avg_radius
        a = self.wire_diameter / 2

        best_n = 1; best_snr = 0
        for n in range(1, 5001):
            L = MU_0 * n**2 * r_avg  # pancake inductance
            R = self.wire_resistivity * 2*np.pi*r_avg * n / (np.pi*a**2)
            XL = omega * L
            Z = np.sqrt(R**2 + XL**2)
            noise2 = self.preamp_en**2 + (self.preamp_in * Z)**2 + 4*K_B*self.T_coil*R
            snr = n / np.sqrt(noise2)  # signal ∝ n
            if snr > best_snr:
                best_snr = snr; best_n = n
        return best_n

    # ---------------------------------------------------------------
    # Front-end configuration
    # ---------------------------------------------------------------
    def _setup(self):
        """Select tuned or untuned mode and pre-compute transfer function.

        TUNED (standard MRI, f₀ >> BW):
          LC bandpass at f_larmor with quality factor Q.
          H(f) = 1 / (1 + jQ(f/f₀ - f₀/f)).
          Preamp noise: NF-based (matched impedance).

        UNTUNED (ULF-MRI, f₀ < Q × BW):
          No LC resonance. Coil connects to broadband preamp.
          H(f) = 1 (flat).
          Preamp noise: e_n and i_n (broadband model).
          Reference: Seton 1995, Myers 2007.
        """
        r_avg = self.coil_avg_radius
        n = self.coil_turns; d = self.wire_diameter

        # DC resistance uses average radius (path length = 2π × r_avg per turn)
        self.R_dc = self.wire_resistivity * 2*np.pi*r_avg*n / (np.pi*(d/2)**2)

        # Pancake coil inductance [H]:
        # L = μ₀ × n² × r_avg
        # Validated against Matlashov 2011 measured L=70mH for 1400-turn
        # pancake with r_avg=27.5mm: predicted 67.7mH (3.2% error).
        # Reference: empirical fit to multi-layer pancake geometry.
        self.L_coil = MU_0 * n**2 * r_avg

        f0 = self.p.f_larmor
        bw = self.p.readout_bw
        tuned_3dB = f0 / self.coil_Q_nominal if self.coil_Q_nominal > 0 else 0

        self.mode = "TUNED" if (f0 > 0 and tuned_3dB >= bw) else "UNTUNED"
        self.Q_eff = self.coil_Q_nominal if self.mode == "TUNED" else 0.0

        # Transfer function
        n_ro = self.p.n_readout
        dt = self.p.dwell_time
        self._bb_freqs = np.fft.fftfreq(n_ro, d=dt)

        if self.mode == "TUNED" and f0 > 0:
            f_rf = f0 + self._bb_freqs
            x = f_rf / f0
            with np.errstate(divide='ignore', invalid='ignore'):
                det = self.Q_eff * (x - 1.0 / np.where(x > 0, x, 1e-30))
            self.H = 1.0 / (1.0 + 1j * det)
        else:
            self.H = np.ones(n_ro, dtype=np.complex128)

        self.H_mag2 = np.abs(self.H)**2

        # Pre-compute noise PSD (vectorized)
        freqs_rf = np.abs(f0 + self._bb_freqs)
        self._S_noise = self._compute_noise_psd_vec(freqs_rf)

    # ---------------------------------------------------------------
    # Reciprocity field B₁/I₀ [T/A]
    # ---------------------------------------------------------------
    def compute_B1_per_I(self, N: int, fov: float) -> np.ndarray:
        """B₁ per unit current for pancake surface coil [T/A].

        Computes the total B_z/I from a pancake coil (flat spiral winding)
        by summing contributions from turns distributed between
        coil_inner_radius and coil_radius (outer radius).

        Uses grouped approximation: turns are divided into N_GROUPS
        radial bins, each computed via exact Biot-Savart (elliptic integrals).
        Multi-layer height offset is included.

        GEOMETRY:
          Coil axis perpendicular to imaging plane.
          Turns distributed from r_inner to r_outer in multiple layers.

        WHY FULL B_z (no factor of 1/2):
          In ULF-MRI, B₀ ⊥ coil axis. Full coupling, no B₁⁻ decomposition.
          Reference: Hoult, Concepts Magn. Reson. 12, 173 (2000).
        """
        N_GROUPS = 20  # radial groups for pancake approximation
        x = np.linspace(-fov/2, fov/2, N)
        X, Y = np.meshgrid(x, x)
        dx, dy = self.coil_position
        rho = np.sqrt((X - dx)**2 + (Y - dy)**2)

        r_inner = self.coil_inner_radius
        r_outer = self.coil_radius
        n_total = self.coil_turns
        d_wire = self.wire_diameter * 1.1  # ~10% packing overhead

        # Radial distribution of turns
        n_radial = max(1, int((r_outer - r_inner) / d_wire))
        n_layers = max(1, int(np.ceil(n_total / n_radial)))

        # Group turns into N_GROUPS radial bins
        group_radii = np.linspace(r_inner + d_wire/2, r_outer - d_wire/2, N_GROUPS)
        turns_per_group = n_total / N_GROUPS

        B1_total = np.zeros((N, N))
        for i, r_g in enumerate(group_radii):
            # Average standoff across layers for this radial group
            z_avg = self.coil_standoff + (n_layers - 1) * d_wire / 2
            B1_total += biot_savart_Bz(r_g, turns_per_group, rho, z_avg)

        return B1_total

    # ---------------------------------------------------------------
    # Noise model (vectorized)
    # ---------------------------------------------------------------
    def _coil_R_vec(self, freqs: np.ndarray) -> np.ndarray:
        """Coil resistance with skin effect [Ω], vectorized.

        R_ac = R_dc × max(1, d/(4δ)), where δ = √(2ρ/(2πfμ₀)) is skin depth.
        Reference: Pozar, "Microwave Engineering" 4th ed., §1.7.
        """
        R = np.full_like(freqs, self.R_dc)
        mask = freqs > 0
        delta = np.sqrt(2 * self.wire_resistivity / (2*np.pi*freqs[mask] * MU_0))
        ratio = self.wire_diameter / (4 * delta)
        R[mask] = self.R_dc * np.maximum(1.0, ratio)
        return R

    def _sample_R_vec(self, freqs: np.ndarray) -> np.ndarray:
        """Equivalent sample noise resistance [Ω], vectorized.

        R_sample = n² × (π/30) × ω² × μ₀² × σ × r_s⁵ / r_c²
        Models eddy current losses in a spherical conducting body
        (radius r_s) coupled to an n-turn circular coil (radius r_c).
        The n² factor arises because B ∝ n for fixed current I,
        so eddy current power ∝ B² ∝ n².
        Derived from P_loss = (π/15) σ ω² B₀² r_s⁵ for a sphere in
        a uniform oscillating field, with B₀ from a current loop.
        Reference: Myers et al., J. Magn. Reson. 186, 182 (2007), Eqs. 4-7;
                   Hoult & Lauterbur, JMR 34, 425 (1979).
        Note: small-sphere approximation (r_s << r_c) may not be fully
        satisfied; see SQUID receiver for same caveat.
        At ULF with n=600, n²×R_sample ≈ 0.01Ω — still negligible vs
        R_dc=15.5Ω and preamp noise. Impact on total noise: <0.01%.
        """
        omega = 2 * np.pi * freqs
        return self.coil_turns**2 * (np.pi / 30) * omega**2 * MU_0**2 * \
               self.sample_sigma * self.sample_radius**5 / self.coil_radius**2

    def _compute_noise_psd_vec(self, freqs_rf: np.ndarray) -> np.ndarray:
        """Total noise PSD [V²/Hz] at each frequency bin.

        TUNED mode:
          Thermal noise from coil/sample passes through LC circuit:
            S_thermal_out(f) = (4kT_c R_c + 4kT_b R_s) × |H(f)|²
          Preamp noise referred to input (Friis 1944 noise figure):
            S_preamp = 4 k_B T₀ × (R_c(f₀) + R_s(f₀)) × (NF - 1)
            T₀ = 290 K (IEEE standard reference temperature)
            R_source = R_c + R_s (total source resistance at resonance)
          Total: S_V(f) = S_thermal × |H|² + S_preamp
          References: Edelstein et al., MRM 3, 604 (1986);
                      Friis, Proc. IRE 32, 419 (1944).

        UNTUNED mode:
          No LC shaping. Broadband preamp noise (e_n / i_n model)
          with 1/f voltage noise:
            S_V = 4kT(R_c+R_s) + e_n²×(1 + f_1f/f) + (i_n × |Z_coil|)²
          The (1 + f_1f/f) factor models the 1/f corner of the preamp.
          Reference: Myers et al., J. Appl. Phys. 103, 07E912 (2007).
        """
        Rc = self._coil_R_vec(freqs_rf)
        Rs = self._sample_R_vec(freqs_rf)
        S_thermal = 4*K_B*self.T_coil*Rc + 4*K_B*self.T_sample*Rs

        if self.mode == "TUNED":
            # Thermal noise shaped by tuned circuit
            S_thermal_shaped = S_thermal * self.H_mag2
            # Preamp noise referred to input: S_preamp = 4kT₀ R_source × (NF-1)
            # NF is defined w.r.t. source at standard temperature T₀ = 290K.
            # Source impedance at resonance = R_coil(f₀) + R_sample(f₀).
            # Reference: Friis, Proc. IRE 32, 419 (1944) — noise figure definition;
            #            Edelstein et al., MRM 3, 604 (1986) — MRI application.
            f0 = self.p.f_larmor
            Rc_f0 = self._coil_R_vec(np.array([f0]))[0]
            Rs_f0 = self._sample_R_vec(np.array([f0]))[0]
            T_std = 290.0  # IEEE standard reference temperature [K]
            NF = 10**(self.preamp_NF_dB / 10)
            S_preamp = 4 * K_B * T_std * (Rc_f0 + Rs_f0) * (NF - 1)
            return S_thermal_shaped + S_preamp
        else:
            XL = 2*np.pi*freqs_rf * self.L_coil
            Z_mag = np.sqrt(Rc**2 + XL**2)
            # Preamp voltage noise with 1/f: e_n²×(1 + f_1f/f)
            freqs_safe = np.maximum(np.abs(freqs_rf), 1e-6)
            en2_1f = self.preamp_en**2 * (1.0 + self.preamp_f1f / freqs_safe)
            S_preamp = en2_1f + (self.preamp_in * Z_mag)**2
            return S_thermal + S_preamp

    # ---------------------------------------------------------------
    # Batch properties for vectorized processing
    # ---------------------------------------------------------------
    @property
    def _batch_gain(self) -> float:
        return self.p.omega_larmor

    @property
    def _batch_H(self) -> np.ndarray:
        return self.H

    @property
    def _batch_noise_psd(self) -> np.ndarray:
        return self._S_noise

    # ---------------------------------------------------------------
    # Signal chain: Ψ [Wb] → V [V]
    # ---------------------------------------------------------------
    def process_readout_line(self, s_flux_Wb: np.ndarray,
                              rng: np.random.Generator) -> np.ndarray:
        """Convert reciprocity flux Ψ [Wb] to output voltage [V].

        1. EMF = ω₀ × Ψ  (Faraday induction)
        2. Transfer function H(f) (bandpass or flat)
        3. Add noise in [V]
        """
        n = len(s_flux_Wb)
        dt = self.p.dwell_time

        # Step 1: Faraday EMF [V]
        s_V = s_flux_Wb * self.p.omega_larmor

        # Step 2: Transfer function
        S = np.fft.fft(s_V)
        S_filt = S * self.H[:n]

        # Step 3: Noise
        # Noise PSD S(f) [V²/Hz] → time-domain variance = Σ S(f_k) × Δf
        # For FFT-based generation: noise_fft has E[|X[k]|²] = 2N,
        # and ifft divides by N, so the amplitude scaling must be:
        #   noise_amp = sqrt(S(f_k) × N × Δf / 2) = sqrt(S(f_k) / (2 × dt))
        bw_bin = 1.0 / (n * dt)
        noise_amp = np.sqrt(self._S_noise[:n] * n * bw_bin / 2)
        noise_fft = np.fft.fft(rng.standard_normal(n) +
                                1j * rng.standard_normal(n))

        return np.fft.ifft(S_filt + noise_fft * noise_amp)

    # ---------------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------------
    @property
    def transfer_function(self) -> np.ndarray:
        """Transfer function H(f) (complex). Shape: (n_readout,).

        Tuned mode: LC bandpass. Untuned mode: flat (all ones).
        Frequency bins from np.fft.fftfreq(n_readout, dwell_time).
        """
        return self.H

    @property
    def baseband_freqs(self) -> np.ndarray:
        """Baseband frequency axis [Hz] for transfer function."""
        return self._bb_freqs

    def __repr__(self) -> str:
        b = self.noise_budget()
        return (f"ClassicalCoilReceiver(B0={self.p.B0}, "
                f"mode={self.mode}, "
                f"r={self.coil_radius*100:.0f}cm, "
                f"n={self.coil_turns}, "
                f"Rc={b['R_coil_Ohm']:.4f}Ω, "
                f"Rs={b['R_sample_Ohm']:.4f}Ω)")

    def noise_budget(self) -> dict:
        """Diagnostic summary of receiver noise sources.

        Returns
        -------
        dict with keys:
            B0, mode, Q_eff, R_coil_Ohm, R_sample_Ohm,
            sample_fraction, omega0.
            Untuned mode adds: Z_coil_Ohm, preamp_en_nV, preamp_in_pA.
        """
        f0 = self.p.f_larmor
        fa = abs(f0) if f0 > 0 else 1.0
        Rc = self._coil_R_vec(np.array([fa]))[0]
        Rs = self._sample_R_vec(np.array([fa]))[0]
        result = {"B0": self.p.B0, "mode": self.mode,
                  "Q_eff": self.Q_eff,
                  "R_coil_Ohm": Rc, "R_sample_Ohm": Rs,
                  "sample_fraction": Rs / max(Rc + Rs, 1e-30),
                  "omega0": self.p.omega_larmor}
        if self.mode == "UNTUNED":
            Z = np.sqrt(Rc**2 + (2*np.pi*fa * self.L_coil)**2)
            result["Z_coil_Ohm"] = Z
            result["preamp_en_nV"] = self.preamp_en * 1e9
            result["preamp_in_pA"] = self.preamp_in * 1e12
        return result


# ===================================================================
# Self-test
# ===================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  classical_coil_receiver.py self-test")
    print("=" * 60)

    print("\n  Front-end mode selection:")
    for B0 in [3.0, 1.5, 0.1, 0.01, 200e-6, 50e-6]:
        p = SpinEchoParams(B0=B0, matrix_size=32, readout_bw=32000)
        c = ClassicalCoilReceiver(p)
        b = c.noise_budget()
        extra = ""
        if c.mode == "UNTUNED":
            extra = (f", Z={b['Z_coil_Ohm']:.3f}Ω, "
                     f"e_n={b['preamp_en_nV']:.1f}nV, "
                     f"i_n={b['preamp_in_pA']:.1f}pA")
        print(f"    B0={B0:.5f}T: {c.mode:>7s}, "
              f"Rc={b['R_coil_Ohm']:.4f}Ω, Rs={b['R_sample_Ohm']:.4f}Ω{extra}")

    print("\n  OK ✓")
