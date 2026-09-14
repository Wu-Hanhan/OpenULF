#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimize_coil_turns.py — Optimal induction coil turns for ULF-MRI at B0=50μT
==============================================================================

Sweeps n=1..5000 turns for multiple coil configurations and finds the
n that maximizes SNR. Compares with SQUID receiver SNR.

Physics model matches classical_coil_receiver.py exactly:
  Signal ∝ n × ω₀ × B1/I0(center)  (Faraday EMF via reciprocity)
  Noise = √(Johnson² + Sample² + Preamp_en² + Preamp_in²)

Configurations swept:
  - 2 coil radii: 0.05 m, 0.10 m
  - 3 wire types: AWG12 (2.0mm), AWG24 (0.51mm), Litz equivalent
  - 3 preamp current noise levels: 2.0 pA (default), 0.8 pA (INA217), 0.02 pA (JFET)
"""
import numpy as np
import sys
import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from constants import GAMMA_H, K_B, MU_0, PHI_0, HBAR
from forward import SpinEchoParams
from quantum_squid_receiver import SQUIDQuantumReceiver
from receiver import biot_savart_Bz

# =======================================================================
# Fixed parameters
# =======================================================================
B0 = 50e-6  # T
f_larmor = GAMMA_H * B0  # Hz
omega0 = 2 * np.pi * f_larmor
print(f"B0 = {B0*1e6:.0f} μT, f_Larmor = {f_larmor:.1f} Hz, ω₀ = {omega0:.1f} rad/s")

T_coil = 295.0       # K
T_sample = 310.0     # K
sample_sigma = 0.4   # S/m
sample_radius = 0.09 # m
rho_copper = 1.68e-8 # Ω·m
preamp_en = 1.0e-9   # V/√Hz

# Turns to sweep
n_turns = np.arange(1, 5001)

# =======================================================================
# Configuration grid
# =======================================================================
configs = []

coil_radii = [0.05, 0.10]
wire_configs = [
    ("AWG12 (2.0mm)", 2.0e-3, False),
    ("AWG24 (0.51mm)", 0.51e-3, False),
    ("Litz (0.51mm eff)", 0.51e-3, True),  # Litz eliminates skin effect
]
preamp_configs = [
    ("Default (2.0 pA)", 2.0e-12),
    ("INA217 (0.8 pA)", 0.8e-12),
    ("Best JFET (0.02 pA)", 0.02e-12),
]

for r_coil in coil_radii:
    for wire_name, wire_d, is_litz in wire_configs:
        for preamp_name, preamp_in in preamp_configs:
            configs.append({
                'r_coil': r_coil,
                'wire_name': wire_name,
                'wire_d': wire_d,
                'is_litz': is_litz,
                'preamp_name': preamp_name,
                'preamp_in': preamp_in,
            })


def compute_snr_vs_turns(cfg, n_array):
    """Compute SNR as a function of number of turns.

    Returns dict with arrays for each noise component and total SNR.
    """
    r = cfg['r_coil']
    d = cfg['wire_d']
    a = d / 2  # wire radius
    rho = rho_copper
    i_n = cfg['preamp_in']
    is_litz = cfg['is_litz']

    n = n_array.astype(np.float64)

    # --- Coil inductance [H] ---
    # L = μ₀ × n² × r × (ln(8r/a) - 2)  [Grover 1946]
    L = MU_0 * n**2 * r * (np.log(8 * r / a) - 2)

    # --- DC resistance [Ω] ---
    # R = ρ × (2πr × n) / (π(d/2)²) = ρ × 2πr × n / (πa²)
    R_dc = rho * 2 * np.pi * r * n / (np.pi * a**2)

    # --- AC resistance with skin effect ---
    # Skin depth: δ = √(2ρ/(2πfμ₀))
    delta = np.sqrt(2 * rho / (2 * np.pi * f_larmor * MU_0))
    skin_ratio = d / (4 * delta)  # same for all n

    if is_litz:
        # Litz wire: no skin effect, R_ac = R_dc
        R_ac = R_dc
    else:
        R_ac = R_dc * max(1.0, skin_ratio)

    # --- Signal ∝ n × ω₀ × (B1/I0 at center) ---
    # B1/I0 at center of coil (on-axis, at standoff z = 0.5*r):
    #   B_z(0, z) = μ₀ n r² / (2(r² + z²)^(3/2))
    # For reciprocity, signal = ω₀ × B1_per_I × Ψ_flux
    # The part that depends on n is: n × ω₀
    # But B1/I0 ∝ n (each turn adds), so signal ∝ n × ω₀
    # For comparing coils, we use: S ∝ n × ω₀ × B1_per_I_single_turn(center)
    # where B1_per_I_single_turn = μ₀ r² / (2(r² + z²)^(3/2))
    standoff = 0.5 * r
    # On-axis field for single-turn coil at standoff z:
    B1_single = MU_0 * r**2 / (2 * (r**2 + standoff**2)**1.5)
    # Signal factor (arbitrary units, but consistent for SNR ratio):
    signal = n * omega0 * B1_single

    # --- Johnson noise from coil resistance ---
    # V_J = √(4 k_B T_coil R_ac)  [V/√Hz]
    V_johnson = np.sqrt(4 * K_B * T_coil * R_ac)

    # --- Sample noise ---
    # R_sample = (1/15) × ω² × μ₀² × σ × r_s⁵ / r_c  [Hoult & Lauterbur 1979]
    # This formula is for a SINGLE-TURN coil. For n turns, R_sample ∝ n²
    # (the mutual inductance / coupling scales as n, so the reflected
    # impedance scales as n²). This is analogous to how the coil's own
    # L ∝ n² and R ∝ n. The physical picture: sample noise flux links
    # all n turns, so voltage noise scales as n, and equivalent R ∝ n².
    #
    # Actually, looking at the code in classical_coil_receiver.py line 287:
    #   R_sample = (1/15) * omega**2 * MU_0**2 * sigma * r_s**5 / r_c
    # This does NOT have an n² factor. However, this is R_sample for the
    # MULTI-TURN coil as used in the code, where the coil has n turns.
    # The formula from Hoult & Lauterbur gives R_sample for a single-turn
    # loop. The code uses it directly regardless of n. This seems to be
    # because the formula models the noise resistance contributed by the
    # sample to the coil circuit. For a multi-turn coil, the effective
    # sample noise resistance seen at the coil terminals scales as n²
    # (just like the coil's own inductance). But the Hoult & Lauterbur
    # formula already implicitly accounts for the geometry.
    #
    # Let's look more carefully. In classical_coil_receiver.py, the
    # sample noise voltage is √(4kT_sample × R_sample), where R_sample
    # does NOT depend on n. But the SIGNAL also only scales linearly
    # through omega (gain = omega0, and B1/I0 includes n through
    # biot_savart_Bz which takes n_turns). So in the existing code:
    #   Signal ∝ omega0 × B1_per_I ∝ omega0 × n
    #   Johnson noise ∝ √(R_dc) ∝ √n
    #   Sample noise ∝ √(R_sample) = const (independent of n in code)
    #
    # But this is WRONG physics for the sample noise! The correct physics:
    # sample noise should scale as n because the noise flux from the sample
    # links all n turns. The voltage noise from the sample is:
    #   V_sample = n × ω × (noise flux per turn)
    # Just as the signal is V_signal = n × ω × (signal flux per turn).
    # So R_sample ∝ n² is the correct scaling.
    #
    # However, since we're using the same code for comparison, let's
    # compute BOTH ways and report both.

    # Way 1: Matching existing code (R_sample independent of n)
    omega_f = 2 * np.pi * f_larmor
    R_sample_code = (1/15) * omega_f**2 * MU_0**2 * sample_sigma * sample_radius**5 / r
    V_sample_code = np.sqrt(4 * K_B * T_sample * R_sample_code) * np.ones_like(n)

    # Way 2: Correct physics (R_sample ∝ n²)
    R_sample_1turn = (1/15) * omega_f**2 * MU_0**2 * sample_sigma * sample_radius**5 / r
    R_sample_nturn = R_sample_1turn * n**2
    V_sample_phys = np.sqrt(4 * K_B * T_sample * R_sample_nturn)

    # --- Preamp voltage noise ---
    # e_n = preamp_en [V/√Hz], independent of n
    V_preamp_en = preamp_en * np.ones_like(n)

    # --- Preamp current noise ---
    # i_n × |Z| where Z = R_ac + jωL
    # |Z| = √(R_ac² + (ωL)²)
    XL = omega0 * L
    Z_mag = np.sqrt(R_ac**2 + XL**2)
    V_preamp_in = i_n * Z_mag

    # --- Total noise (code-matching version) ---
    noise_total_code = np.sqrt(V_johnson**2 + V_sample_code**2 +
                                V_preamp_en**2 + V_preamp_in**2)

    # --- Total noise (physics-correct version) ---
    noise_total_phys = np.sqrt(V_johnson**2 + V_sample_phys**2 +
                                V_preamp_en**2 + V_preamp_in**2)

    # --- SNR ---
    snr_code = signal / noise_total_code
    snr_phys = signal / noise_total_phys

    # --- Self-resonance frequency ---
    # Parasitic capacitance: ~1 pF per layer. For single-layer winding,
    # C_parasitic ≈ n × 0.5 pF (interwinding capacitance grows with turns).
    # More conservatively: C_stray ≈ 1-10 pF base + winding capacitance.
    # For a multi-layer solenoid: C_p ≈ C_tt × n_layers where C_tt is
    # turn-to-turn capacitance (~1-5 pF).
    # Simple model: C_p = 2 pF + 0.5 pF × n (for tightly wound)
    # More realistic: C_p = 2 pF × (1 + n/100) for loose single-layer winding
    C_parasitic = 2e-12 * (1 + n / 100)  # pF, grows with turns
    f_sr = 1.0 / (2 * np.pi * np.sqrt(L * C_parasitic))

    return {
        'n': n_array,
        'L': L,
        'R_dc': R_dc,
        'R_ac': R_ac,
        'signal': signal,
        'V_johnson': V_johnson,
        'V_sample_code': V_sample_code,
        'V_sample_phys': V_sample_phys,
        'V_preamp_en': V_preamp_en,
        'V_preamp_in': V_preamp_in,
        'noise_total_code': noise_total_code,
        'noise_total_phys': noise_total_phys,
        'snr_code': snr_code,
        'snr_phys': snr_phys,
        'f_sr': f_sr,
        'C_parasitic': C_parasitic,
        'Z_mag': Z_mag,
        'XL': XL,
        'skin_depth': delta,
        'skin_ratio': skin_ratio,
    }


# =======================================================================
# Run optimization
# =======================================================================
print("\n" + "=" * 120)
print("COIL TURNS OPTIMIZATION RESULTS")
print("=" * 120)

results = []

for cfg in configs:
    res = compute_snr_vs_turns(cfg, n_turns)

    # Find optimal n for both models
    idx_code = np.argmax(res['snr_code'])
    idx_phys = np.argmax(res['snr_phys'])

    n_opt_code = n_turns[idx_code]
    n_opt_phys = n_turns[idx_phys]

    snr_opt_code = res['snr_code'][idx_code]
    snr_opt_phys = res['snr_phys'][idx_phys]

    # Self-resonance check at optimal n
    f_sr_code = res['f_sr'][idx_code]
    f_sr_phys = res['f_sr'][idx_phys]
    sr_ok_code = f_sr_code > 10 * f_larmor
    sr_ok_phys = f_sr_phys > 10 * f_larmor

    # SNR at current default (n=2)
    snr_at_2_code = res['snr_code'][1]  # index 1 = n=2

    results.append({
        'cfg': cfg,
        'res': res,
        'n_opt_code': n_opt_code,
        'n_opt_phys': n_opt_phys,
        'snr_opt_code': snr_opt_code,
        'snr_opt_phys': snr_opt_phys,
        'snr_at_2_code': snr_at_2_code,
        'f_sr_code': f_sr_code,
        'f_sr_phys': f_sr_phys,
        'sr_ok_code': sr_ok_code,
        'sr_ok_phys': sr_ok_phys,
    })

# =======================================================================
# Print summary table
# =======================================================================
print(f"\nSkin depth at {f_larmor:.0f} Hz = {np.sqrt(2*rho_copper/(2*np.pi*f_larmor*MU_0))*1e3:.2f} mm")
print()

# Table header
print(f"{'Config':60s} | {'n_opt':>5s} | {'SNR_opt':>10s} | {'SNR@n=2':>10s} | {'Gain':>6s} | "
      f"{'f_SR[kHz]':>10s} | {'SR>10f₀':>7s} | {'L[H]':>10s} | {'R[Ω]':>8s} | {'Z[Ω]':>10s} | "
      f"{'n_opt_p':>7s} | {'SNR_p':>10s}")
print("-" * 175)

for r in results:
    cfg = r['cfg']
    res = r['res']
    idx_c = r['n_opt_code'] - 1
    idx_p = r['n_opt_phys'] - 1

    label = f"r={cfg['r_coil']*100:.0f}cm, {cfg['wire_name']}, {cfg['preamp_name']}"

    gain = r['snr_opt_code'] / r['snr_at_2_code'] if r['snr_at_2_code'] > 0 else 0

    print(f"{label:60s} | {r['n_opt_code']:5d} | {r['snr_opt_code']:10.4f} | "
          f"{r['snr_at_2_code']:10.4f} | {gain:5.1f}x | "
          f"{r['f_sr_code']/1e3:10.1f} | {'YES' if r['sr_ok_code'] else 'NO!':>7s} | "
          f"{res['L'][idx_c]:10.4e} | {res['R_ac'][idx_c]:8.4f} | {res['Z_mag'][idx_c]:10.4f} | "
          f"{r['n_opt_phys']:7d} | {r['snr_opt_phys']:10.4f}")

# =======================================================================
# Detailed breakdown for key configurations
# =======================================================================
print("\n" + "=" * 120)
print("DETAILED NOISE BREAKDOWN AT OPTIMAL n (code-matching model)")
print("=" * 120)

for r in results:
    cfg = r['cfg']
    res = r['res']
    n_opt = r['n_opt_code']
    idx = n_opt - 1

    label = f"r={cfg['r_coil']*100:.0f}cm, {cfg['wire_name']}, {cfg['preamp_name']}"
    print(f"\n  {label}")
    print(f"    Optimal n = {n_opt}")
    print(f"    L = {res['L'][idx]:.4e} H")
    print(f"    R_dc = {res['R_dc'][idx]:.4f} Ω, R_ac = {res['R_ac'][idx]:.4f} Ω")
    print(f"    XL = ωL = {res['XL'][idx]:.4f} Ω")
    print(f"    |Z| = {res['Z_mag'][idx]:.4f} Ω")
    print(f"    Signal = {res['signal'][idx]:.4e} (arb.)")
    print(f"    Noise breakdown (V/√Hz):")
    print(f"      Johnson (coil):   {res['V_johnson'][idx]:.4e}")
    print(f"      Sample (code):    {res['V_sample_code'][idx]:.4e}")
    print(f"      Preamp e_n:       {res['V_preamp_en'][idx]:.4e}")
    print(f"      Preamp i_n×|Z|:   {res['V_preamp_in'][idx]:.4e}")
    total = res['noise_total_code'][idx]
    print(f"      Total:            {total:.4e}")
    print(f"    Dominant noise: ", end="")
    noises = {
        'Johnson': res['V_johnson'][idx],
        'Sample': res['V_sample_code'][idx],
        'Preamp_en': res['V_preamp_en'][idx],
        'Preamp_in': res['V_preamp_in'][idx],
    }
    dominant = max(noises, key=noises.get)
    for name, val in sorted(noises.items(), key=lambda x: -x[1]):
        pct = (val / total)**2 * 100
        print(f"{name}={pct:.1f}%  ", end="")
    print()
    print(f"    f_SR = {res['f_sr'][idx]/1e3:.1f} kHz (C_p = {res['C_parasitic'][idx]*1e12:.1f} pF)")
    if not r['sr_ok_code']:
        print(f"    *** WARNING: f_SR < 10 × f_Larmor ({10*f_larmor/1e3:.1f} kHz) ***")

# =======================================================================
# SQUID comparison
# =======================================================================
print("\n" + "=" * 120)
print("SQUID vs OPTIMAL COIL COMPARISON")
print("=" * 120)

p = SpinEchoParams(B0=B0, matrix_size=64, readout_bw=32000)
squid = SQUIDQuantumReceiver(p)
squid_budget = squid.noise_budget()

print(f"\nSQUID receiver (default params):")
print(f"  Pickup radius: {squid.pickup_radius*100:.1f} cm")
print(f"  Pickup turns: {squid.pickup_turns}")
print(f"  Standoff: {squid.pickup_standoff*100:.1f} cm (1.0 × radius)")
print(f"  Gradiometer order: {squid.grad_order}")
print(f"  V_Φ_system: {squid.V_phi_system:.4e} V/Wb")
print(f"  Noise budget:")
print(f"    SQUID intrinsic: {squid_budget['squid_noise_fT']:.2f} fT/√Hz")
print(f"    Sample noise:    {squid_budget['sample_noise_fT']:.2f} fT/√Hz")
print(f"    Environmental:   {squid_budget['env_noise_fT']:.2f} fT/√Hz")
print(f"    Total:           {squid_budget['total_noise_fT']:.2f} fT/√Hz")
print(f"    Dewar signal loss: {squid_budget['dewar_signal_loss_pct']:.2f}%")

# Compute SQUID SNR factor using same approach
# SQUID signal: V = V_phi_system × Ψ, where Ψ = B1/I0 × M × voxel_vol
# B1/I0 for SQUID at center (with gradiometer):
squid_standoff = squid.pickup_standoff
squid_r = squid.pickup_radius
# For 2nd order gradiometer:
B1_squid_mag = biot_savart_Bz(squid_r, 1, np.array([0.0]), squid_standoff)[0]
B1_squid_grad1 = biot_savart_Bz(squid_r, 1, np.array([0.0]),
                                  squid_standoff + squid.grad_baseline)[0]
B1_squid_grad2 = biot_savart_Bz(squid_r, 1, np.array([0.0]),
                                  squid_standoff + 2*squid.grad_baseline)[0]
B1_squid = B1_squid_mag - 2*B1_squid_grad1 + B1_squid_grad2
# SQUID signal factor: V_phi_system × B1/I0_gradiometer
squid_signal_factor = squid.V_phi_system * B1_squid

# SQUID total noise [V/√Hz]
squid_total_noise_V = np.sqrt(squid._S_V_sq_at_f0 + squid._S_V_samp + squid._S_V_env)
squid_snr_factor = squid_signal_factor / squid_total_noise_V

print(f"\n  SQUID B1/I0 at center (gradiometer): {B1_squid:.4e} T/A")
print(f"  SQUID signal factor: {squid_signal_factor:.4e}")
print(f"  SQUID total noise V/√Hz: {squid_total_noise_V:.4e}")
print(f"  SQUID SNR factor: {squid_snr_factor:.4e}")

# Compare with coil
print(f"\n  Comparison (same standoff ratio basis):")
print(f"  {'Config':60s} | {'Coil SNR':>10s} | {'SQUID/Coil':>10s}")
print(f"  {'-'*60}-+-{'-'*10}-+-{'-'*10}")

# For fair comparison, normalize both by the same M×voxel terms
# Coil: signal = n × ω₀ × B1_single_turn(at standoff)
# SQUID: signal = V_phi_system × n_pickup × B1_single_turn_squid(at standoff)

for r in results:
    cfg = r['cfg']
    res = r['res']
    n_opt = r['n_opt_code']
    idx = n_opt - 1

    # Coil SNR factor
    coil_snr = r['snr_opt_code']

    # Ratio: SQUID_snr / Coil_snr
    ratio = squid_snr_factor / coil_snr if coil_snr > 0 else float('inf')

    label = f"r={cfg['r_coil']*100:.0f}cm, {cfg['wire_name']}, {cfg['preamp_name']}"
    print(f"  {label:60s} | {coil_snr:10.4e} | {ratio:10.2f}x")

# =======================================================================
# Key physics insights
# =======================================================================
print("\n" + "=" * 120)
print("KEY PHYSICS INSIGHTS")
print("=" * 120)

# Find the best coil config overall
best = max(results, key=lambda r: r['snr_opt_code'])
cfg_best = best['cfg']
print(f"\n1. Best coil configuration: r={cfg_best['r_coil']*100:.0f}cm, "
      f"{cfg_best['wire_name']}, {cfg_best['preamp_name']}")
print(f"   Optimal turns: {best['n_opt_code']}")
print(f"   SNR gain vs n=2: {best['snr_opt_code']/best['snr_at_2_code']:.1f}x")

# Limiting noise source analysis
print(f"\n2. Why the optimum exists:")
print(f"   - Signal ∝ n (each turn adds flux linkage)")
print(f"   - Johnson noise ∝ √n (resistance grows linearly)")
print(f"   - Preamp i_n noise ∝ ωL ∝ n² (inductance grows as n²)")
print(f"   - At low n: dominated by preamp e_n (fixed) → SNR ∝ n")
print(f"   - At high n: dominated by i_n × ωL ∝ n² → SNR ∝ 1/n")
print(f"   - Optimal n: where signal/noise transitions from ∝n to ∝1/n")

# Skin effect
delta = np.sqrt(2 * rho_copper / (2 * np.pi * f_larmor * MU_0))
print(f"\n3. Skin effect at {f_larmor:.0f} Hz:")
print(f"   Skin depth δ = {delta*1e3:.2f} mm")
print(f"   AWG12 (d=2.0mm): d/4δ = {2.0e-3/(4*delta):.3f} → {'NO skin effect' if 2.0e-3/(4*delta) < 1 else 'skin effect active'}")
print(f"   AWG24 (d=0.51mm): d/4δ = {0.51e-3/(4*delta):.3f} → {'NO skin effect' if 0.51e-3/(4*delta) < 1 else 'skin effect active'}")
print(f"   → At 2129 Hz, skin effect is negligible for both wire gauges!")

# What limits SQUID advantage
print(f"\n4. SQUID advantage source:")
print(f"   SQUID detects FLUX directly (no dΦ/dt → no ω₀ penalty)")
print(f"   SQUID noise is frequency-independent (white flux noise)")
print(f"   Coil signal ∝ ω₀ × n, but noise floor limited by preamp")
print(f"   At ULF (2129 Hz), ω₀ is small → coil EMF is tiny")

# =======================================================================
# R_sample scaling analysis
# =======================================================================
print("\n" + "=" * 120)
print("NOTE ON R_SAMPLE SCALING WITH n")
print("=" * 120)
print(f"""
The existing code (classical_coil_receiver.py line 287) computes:
  R_sample = (1/15) × ω² × μ₀² × σ × r_s⁵ / r_c

This does NOT include an n² factor. The Hoult & Lauterbur (1979) formula
gives R_sample for a single-turn loop, and for multi-turn coils, the
reflected sample resistance scales as n² (just like the coil's own
inductance L ∝ n² and resistance R ∝ n).

However, this doesn't change the optimization much because at 2129 Hz,
the sample noise is extremely small compared to the preamp noise:
  R_sample (1-turn, r=10cm) = {(1/15)*(omega0**2)*MU_0**2*0.4*0.09**5/0.10:.4e} Ω
  vs R_dc (1-turn, r=10cm, AWG12) = {rho_copper*2*np.pi*0.10/(np.pi*(1e-3)**2):.4e} Ω

The sample noise is ~{((1/15)*(omega0**2)*MU_0**2*0.4*0.09**5/0.10)/(rho_copper*2*np.pi*0.10/(np.pi*(1e-3)**2))*100:.4f}% of the coil resistance.
At ULF, the coil + preamp noise completely dominates over sample noise.
This is why SQUID provides such a large advantage: the SQUID sees only
sample noise (+ its own intrinsic noise), while the coil adds large
Johnson + preamp noise.
""")

# =======================================================================
# Practical recommendation
# =======================================================================
print("=" * 120)
print("PRACTICAL RECOMMENDATIONS")
print("=" * 120)

# Find best config for each radius
for r_coil in coil_radii:
    subset = [r for r in results if r['cfg']['r_coil'] == r_coil]
    best_sub = max(subset, key=lambda r: r['snr_opt_code'])
    cfg_b = best_sub['cfg']
    print(f"\n  For r = {r_coil*100:.0f} cm coil:")
    print(f"    Best wire: {cfg_b['wire_name']}")
    print(f"    Best preamp: {cfg_b['preamp_name']}")
    print(f"    Optimal turns: {best_sub['n_opt_code']}")
    print(f"    Self-resonance: {best_sub['f_sr_code']/1e3:.1f} kHz "
          f"({'OK' if best_sub['sr_ok_code'] else 'TOO LOW!'})")
    print(f"    SNR improvement vs n=2: {best_sub['snr_opt_code']/best_sub['snr_at_2_code']:.1f}x")

    # SQUID ratio for this config
    ratio = squid_snr_factor / best_sub['snr_opt_code']
    print(f"    SQUID advantage: {ratio:.1f}x (even with optimal coil)")

    # Check what n gives a practical self-resonance (> 100 kHz)
    res = best_sub['res']
    practical_mask = res['f_sr'] > 100e3
    if np.any(practical_mask):
        practical_snr = res['snr_code'].copy()
        practical_snr[~practical_mask] = 0
        idx_practical = np.argmax(practical_snr)
        n_practical = n_turns[idx_practical]
        print(f"    Practical optimal (f_SR > 100 kHz): n = {n_practical}, "
              f"SNR = {practical_snr[idx_practical]:.4e} "
              f"({practical_snr[idx_practical]/best_sub['snr_at_2_code']:.1f}x vs n=2)")

print("\n" + "=" * 120)
print("DONE")
print("=" * 120)
