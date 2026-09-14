#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
numerical_audit.py — Comprehensive Numerical Verification of Signal Chain
==========================================================================

Analytically computes expected values at every stage of the MRI signal chain
and compares with code outputs. Prints PASS/FAIL for each check (>1% = FAIL).

Tests:
  1. Signal levels (M_perp, flux, voltage)
  2. Noise levels (SQUID, Coil PSDs in fT/rtHz)
  3. Image-level SNR
  4. Normalization effect
  5. FFT noise scaling verification
  6. B1/I0 verification (Biot-Savart)
  7. Corrected coil (n=600, INA217)
"""
import sys
import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
from constants import GAMMA_H, GAMMA_RAD, K_B, MU_0, HBAR, PHI_0, N_A, T_BODY, M0_PER_TESLA, N_H_WATER
from tissue import TissueProperties, TISSUE_DB
from forward import SpinEchoParams, MRISignalEngine
from receiver import ReceiverBase, biot_savart_Bz
from classical_coil_receiver import ClassicalCoilReceiver
from quantum_squid_receiver import SQUIDQuantumReceiver
from scipy.special import ellipk, ellipe

TOLERANCE = 0.01  # 1% relative tolerance
pass_count = 0
fail_count = 0

def check(name, computed, expected, tol=TOLERANCE, units=""):
    """Compare computed vs expected, print PASS/FAIL."""
    global pass_count, fail_count
    if expected == 0:
        if computed == 0:
            rel_err = 0
        else:
            rel_err = abs(computed)
    else:
        rel_err = abs(computed - expected) / abs(expected)

    status = "PASS" if rel_err <= tol else "FAIL"
    if status == "PASS":
        pass_count += 1
    else:
        fail_count += 1

    print(f"  [{status}] {name}")
    print(f"         computed = {computed:.6e} {units}")
    print(f"         expected = {expected:.6e} {units}")
    print(f"         rel_err  = {rel_err:.4e} ({rel_err*100:.4f}%)")

def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

# =====================================================================
# Common parameters
# =====================================================================
B0 = 50e-6         # T
B_prepol = 0.1     # T
TE = 20.0           # ms
TR = 500.0          # ms
t_pol = 2.0         # s
t_transfer = 0.1    # s
N = 256
fov = 0.256         # m
dx = fov / N
slice_thickness = 0.005  # m
voxel_volume = dx**2 * slice_thickness
readout_bw = 32000.0  # Hz
dwell_time = 1.0 / readout_bw

# Water tissue (label=1 CSF has PD=1, but let's use pure water props)
# For a clean test, use a tissue with known properties.
# We'll compute for "grey matter" (label 2) since it's the most common brain tissue.
# But for "pure water", let's just manually compute.

# Use Grey Matter (label 2) for the test
tissue_gm = TISSUE_DB[2]  # Grey Matter
PD_gm = tissue_gm.PD  # 0.86

section("0. CONSTANT VERIFICATION")
print(f"  GAMMA_H = {GAMMA_H:.9e} Hz/T")
print(f"  GAMMA_RAD = {GAMMA_RAD:.9e} rad/(s*T)")
print(f"  K_B = {K_B:.9e} J/K")
print(f"  MU_0 = {MU_0:.9e} H/m")
print(f"  HBAR = {HBAR:.9e} J*s")
print(f"  PHI_0 = {PHI_0:.9e} Wb")
print(f"  N_H_WATER = {N_H_WATER:.6e} protons/m^3")
print(f"  M0_PER_TESLA = {M0_PER_TESLA:.6e} A/m per Tesla")

# Verify M0_PER_TESLA independently
M0_check = N_H_WATER * GAMMA_RAD**2 * HBAR**2 / (4 * K_B * T_BODY)
check("M0_PER_TESLA", M0_PER_TESLA, M0_check, units="A/m/T")

# =====================================================================
section("1. SIGNAL LEVELS — Single Pure-Water Voxel at Center")
# =====================================================================

# --- Step 1a: M_perp for GM with prepolarization ---
# T1 at B_prepol = 0.1T
T1_3T_gm = tissue_gm.T1_3T  # 1330 ms
alpha_gm = tissue_gm.alpha   # 0.35
T2_gm = tissue_gm.T2         # 80 ms
T1_min_gm = max(T2_gm * 1.5, 30.0)  # max(120, 30) = 120 ms

T1_Bp = T1_min_gm + (T1_3T_gm - T1_min_gm) * (B_prepol / 3.0)**alpha_gm
T1_B0 = T1_min_gm + (T1_3T_gm - T1_min_gm) * (B0 / 3.0)**alpha_gm

print(f"\n  Grey Matter relaxation at B0={B0*1e6:.0f}uT, B_prepol={B_prepol}T:")
print(f"    T1 at B_prepol={B_prepol}T: {T1_Bp:.2f} ms")
print(f"    T1 at B0={B0*1e6:.0f}uT:    {T1_B0:.2f} ms")
print(f"    T2:                {T2_gm:.1f} ms")

# Verify with code
check("T1_at_field(B_prepol)", tissue_gm.T1_at_field(B_prepol), T1_Bp, units="ms")
check("T1_at_field(B0)", tissue_gm.T1_at_field(B0), T1_B0, units="ms")

# M_eq at B_prepol
M_eq_Bp = M0_PER_TESLA * B_prepol
M_eq_B0 = M0_PER_TESLA * B0

print(f"\n  Equilibrium magnetization:")
print(f"    M_eq(B_prepol={B_prepol}T) = {M_eq_Bp:.6e} A/m")
print(f"    M_eq(B0={B0*1e6:.0f}uT) = {M_eq_B0:.6e} A/m")

# Prepolarization model
# Phase 1: M_pol = M_eq(Bp) * (1 - exp(-t_pol*1e3 / T1_Bp))
M_pol = M_eq_Bp * (1.0 - np.exp(-t_pol * 1e3 / T1_Bp))
print(f"\n  Prepolarization:")
print(f"    M_pol (after t_pol={t_pol}s): {M_pol:.6e} A/m")
print(f"    Fraction of M_eq(Bp): {M_pol/M_eq_Bp:.4f}")

# Phase 2: M_avail = M_eq(B0) + (M_pol - M_eq(B0)) * exp(-t_xfer*1e3 / T1_B0)
M_avail = M_eq_B0 + (M_pol - M_eq_B0) * np.exp(-t_transfer * 1e3 / T1_B0)
print(f"    M_avail (after t_transfer={t_transfer}s): {M_avail:.6e} A/m")

# M_perp = M_avail * PD * exp(-TE/T2)
M_perp_analytical = M_avail * PD_gm * np.exp(-TE / T2_gm)
print(f"\n  M_perp for Grey Matter:")
print(f"    M_perp = M_avail * PD * exp(-TE/T2)")
print(f"    M_perp = {M_avail:.4e} * {PD_gm} * exp(-{TE}/{T2_gm})")
print(f"    M_perp = {M_perp_analytical:.6e} A/m")

# Compare with code
p = SpinEchoParams(B0=B0, B_prepol=B_prepol, TE=TE, TR=TR, matrix_size=N,
                   fov=fov, slice_thickness=slice_thickness, readout_bw=readout_bw,
                   t_pol=t_pol, t_transfer=t_transfer)
engine = MRISignalEngine(p)
M_perp_code = engine._compute_M_perp(tissue_gm)
check("M_perp (GM, prepolarized)", M_perp_code, M_perp_analytical, units="A/m")

# --- Step 1b: Compute flux for SQUID and Coil ---
print(f"\n  --- Flux computation ---")
print(f"  Voxel volume: {voxel_volume:.6e} m^3")

# SQUID: pickup_radius=0.05, standoff=0.05, 1-turn, 2nd-order grad
r_squid = 0.05
z_squid = r_squid * 1.0  # standoff = 1.0 * radius
grad_baseline = 0.06

# B1/I at center for SQUID (rho=0, 2nd-order gradiometer)
# On-axis: B_z/I = mu0 * n * r^2 / (2 * (r^2 + z^2)^(3/2))
def on_axis_B(r, n, z):
    return MU_0 * n * r**2 / (2 * (r**2 + z**2)**1.5)

B1_I_squid_center_mag = on_axis_B(r_squid, 1, z_squid)
# 2nd order gradiometer: B1 - 2*B1(z+b) + B1(z+2b)
B1_I_squid_grad2 = (on_axis_B(r_squid, 1, z_squid)
                     - 2 * on_axis_B(r_squid, 1, z_squid + grad_baseline)
                     + on_axis_B(r_squid, 1, z_squid + 2 * grad_baseline))
print(f"  SQUID B1/I at center (magnetometer): {B1_I_squid_center_mag:.6e} T/A")
print(f"  SQUID B1/I at center (2nd-order grad): {B1_I_squid_grad2:.6e} T/A")
print(f"  Gradiometer signal retention: {B1_I_squid_grad2/B1_I_squid_center_mag*100:.1f}%")

# Compare with code
sq_rcv = SQUIDQuantumReceiver(p, seed=42)
B1_map_squid = sq_rcv.compute_B1_per_I(N, fov)
B1_I_code_squid = B1_map_squid[N//2, N//2]
check("SQUID B1/I at center (code vs analytical)", B1_I_code_squid, B1_I_squid_grad2,
      tol=0.05, units="T/A")  # 5% tol due to off-center discretization

# Flux: Psi = B1/I * M_perp * voxel_volume  [Wb]
Psi_squid = B1_I_squid_grad2 * M_perp_analytical * voxel_volume
print(f"\n  SQUID flux from center voxel: Psi = {Psi_squid:.6e} Wb")

# Coil: coil_radius=0.10, standoff=0.05 (0.5*radius), 2-turn
r_coil = 0.10
z_coil = r_coil * 0.5  # standoff = 0.5 * radius
n_coil = 2

B1_I_coil_center = on_axis_B(r_coil, n_coil, z_coil)
print(f"  Coil B1/I at center: {B1_I_coil_center:.6e} T/A")

coil_rcv = ClassicalCoilReceiver(p, seed=42)
B1_map_coil = coil_rcv.compute_B1_per_I(N, fov)
B1_I_code_coil = B1_map_coil[N//2, N//2]
check("Coil B1/I at center (code vs analytical)", B1_I_code_coil, B1_I_coil_center,
      tol=0.05, units="T/A")

Psi_coil = B1_I_coil_center * M_perp_analytical * voxel_volume
print(f"  Coil flux from center voxel: Psi = {Psi_coil:.6e} Wb")

# --- Step 1c: Output voltages ---
print(f"\n  --- Output voltages ---")
# SQUID: V = V_phi_system * Psi
V_phi_system = sq_rcv.V_phi_system
print(f"  SQUID V_phi_system = {V_phi_system:.6e} V/Wb")
print(f"    V_phi_bare = R_dyn/L_sq = {sq_rcv.R_dyn}/{sq_rcv.L_sq} = {sq_rcv.V_phi_bare:.6e} V/Wb")
print(f"    coupling = {sq_rcv.coupling}")

V_squid = V_phi_system * Psi_squid
print(f"  SQUID output voltage (center voxel): V = {V_squid:.6e} V")

# Coil: V = omega_larmor * Psi
omega_0 = 2 * np.pi * GAMMA_H * B0
print(f"\n  Coil omega_larmor = {omega_0:.6e} rad/s")
print(f"  Coil f_larmor = {GAMMA_H * B0:.2f} Hz")
V_coil = omega_0 * Psi_coil
print(f"  Coil output voltage (center voxel): V = {V_coil:.6e} V")
print(f"\n  Ratio V_squid/V_coil = {V_squid/V_coil:.2f}")

check("SQUID gain (V_phi_system)", sq_rcv._batch_gain, V_phi_system, units="V/Wb")
check("Coil gain (omega_larmor)", coil_rcv._batch_gain, omega_0, units="rad/s")

# =====================================================================
section("2. NOISE LEVELS — SQUID and Coil at B0=50uT")
# =====================================================================

f_larmor = GAMMA_H * B0
print(f"\n  f_larmor = {f_larmor:.2f} Hz")

# --- SQUID noise components ---
print(f"\n  --- SQUID Noise ---")
# SQUID parameters (defaults)
I_c = 10e-6
R_n = 10.0
L_sq = 100e-12
C_j = 0.25e-12
coupling = 0.5
pickup_radius = 0.05
pickup_turns = 1
pickup_area = np.pi * pickup_radius**2 * pickup_turns
T_squid = 4.2
T_sample = 310.0
sample_sigma = 0.4
sample_radius = 0.09
env_field_noise = 1e-12  # T/rtHz
shielding_factor = 100.0
grad_order = 2

R_dyn = R_n / 2
V_phi_bare = R_dyn / L_sq
V_phi_sys = V_phi_bare * coupling

# Quantum correction
omega_c = 2 * np.pi * I_c * R_n / PHI_0
x_q = HBAR * omega_c / (2 * K_B * T_squid)
T_eff = T_squid * x_q / np.tanh(x_q) if x_q > 0.01 else T_squid

# 1. SQUID intrinsic noise
S_phi_squid = 16 * K_B * T_eff * L_sq**2 / R_dyn
flux_noise = np.sqrt(S_phi_squid)
S_V_squid = S_phi_squid * V_phi_bare**2

# Convert to field noise via sensitivity
sensitivity_VT = V_phi_sys * pickup_area
B_noise_squid = np.sqrt(S_V_squid) / sensitivity_VT
print(f"  SQUID intrinsic:")
print(f"    T_eff = {T_eff:.2f} K (quantum correction from T_squid={T_squid}K)")
print(f"    S_phi = {S_phi_squid:.6e} Wb^2/Hz")
print(f"    flux noise = {flux_noise:.6e} Wb/rtHz = {flux_noise/PHI_0*1e6:.2f} uPhi0/rtHz")
print(f"    S_V_squid = {S_V_squid:.6e} V^2/Hz")
print(f"    B_noise_squid = {B_noise_squid*1e15:.3f} fT/rtHz")

check("SQUID intrinsic S_V", sq_rcv._S_V_sq_at_f0, S_V_squid, units="V^2/Hz")

# 2. Sample noise
# S_Phi_sample = 4 * mu0^2 * sigma * kT * r_s^3 / (15 * r_p)
S_phi_samp = (4 * MU_0**2 * sample_sigma * K_B * T_sample *
              sample_radius**3 / (15 * pickup_radius))
S_V_samp = S_phi_samp * V_phi_sys**2
B_noise_samp = np.sqrt(S_V_samp) / sensitivity_VT
print(f"\n  Sample noise:")
print(f"    S_phi_sample = {S_phi_samp:.6e} Wb^2/Hz")
print(f"    S_V_sample = {S_V_samp:.6e} V^2/Hz")
print(f"    B_noise_sample = {B_noise_samp*1e15:.3f} fT/rtHz")

check("SQUID sample S_V", sq_rcv._S_V_samp, S_V_samp, units="V^2/Hz")

# 3. Environmental noise
gr_rejection = 100.0 ** min(grad_order, 2)  # 10000 for 2nd order
env_noise_eff = env_field_noise / shielding_factor / gr_rejection
S_V_env = (env_noise_eff * pickup_area * V_phi_sys)**2
B_noise_env = np.sqrt(S_V_env) / sensitivity_VT
print(f"\n  Environmental noise:")
print(f"    env_field = {env_field_noise*1e12:.1f} pT/rtHz")
print(f"    shielding = {shielding_factor}x, gradiometer rejection = {gr_rejection}x")
print(f"    effective = {env_noise_eff*1e15:.3f} fT/rtHz")
print(f"    S_V_env = {S_V_env:.6e} V^2/Hz")
print(f"    B_noise_env = {B_noise_env*1e15:.6f} fT/rtHz")

check("SQUID env S_V", sq_rcv._S_V_env, S_V_env, units="V^2/Hz")

# Total SQUID noise
B_noise_total = np.sqrt(B_noise_squid**2 + B_noise_samp**2 + B_noise_env**2)
print(f"\n  SQUID total noise: {B_noise_total*1e15:.3f} fT/rtHz")
print(f"  Dominant source: {'sample' if B_noise_samp > B_noise_squid else 'SQUID intrinsic'}")

budget = sq_rcv.noise_budget()
check("SQUID total noise (fT)", budget['total_noise_fT'], B_noise_total*1e15,
      tol=0.02, units="fT/rtHz")

# Literature check
print(f"\n  Literature check:")
print(f"    SQUID total: {B_noise_total*1e15:.3f} fT/rtHz")
if B_noise_total*1e15 < 1.0:
    print(f"    [INFO] Below 1 fT/rtHz -> consistent with sample-noise limited regime")
    print(f"    Published systems: 1-3 fT/rtHz (include more environmental noise)")
else:
    print(f"    [INFO] Above 1 fT/rtHz")

# --- Coil noise components ---
print(f"\n  --- Coil Noise (n={coil_rcv.coil_turns}, default) ---")
# Coil is UNTUNED at 50uT (f_larmor ~ 2.13 kHz, BW=32 kHz)
print(f"  Mode: {coil_rcv.mode}")

r_c = coil_rcv.coil_radius  # 0.10 m
n_c = coil_rcv.coil_turns    # 2
d_w = coil_rcv.wire_diameter  # 2e-3 m
rho_w = coil_rcv.wire_resistivity  # 1.68e-8 Ohm*m
T_c = coil_rcv.T_coil  # 295 K
T_s = coil_rcv.T_sample  # 310 K
sigma_s = coil_rcv.sample_sigma  # 0.4 S/m
r_s = coil_rcv.sample_radius  # 0.09 m
en = coil_rcv.preamp_en  # 1e-9 V/rtHz
i_n = coil_rcv.preamp_in  # 2e-12 A/rtHz
f1f = coil_rcv.preamp_f1f  # 10 Hz

# R_dc
R_dc = rho_w * 2 * np.pi * r_c * n_c / (np.pi * (d_w/2)**2)
print(f"  R_dc = {R_dc:.6f} Ohm")
check("Coil R_dc", coil_rcv.R_dc, R_dc, units="Ohm")

# Skin depth at f_larmor
delta_skin = np.sqrt(2 * rho_w / (2 * np.pi * f_larmor * MU_0))
skin_ratio = d_w / (4 * delta_skin)
R_ac = R_dc * max(1.0, skin_ratio)
print(f"  skin depth at f_larmor: {delta_skin*1e3:.2f} mm")
print(f"  d_wire/(4*delta) = {skin_ratio:.4f}")
print(f"  R_ac = {R_ac:.6f} Ohm (= R_dc, no skin effect at this freq)")

# R_sample for coil
omega_L = 2 * np.pi * f_larmor
R_sample_coil = (1/15) * omega_L**2 * MU_0**2 * sigma_s * r_s**5 / r_c
print(f"  R_sample = {R_sample_coil:.6e} Ohm")

# L_coil
wire_radius = d_w / 2
L_coil = MU_0 * n_c**2 * r_c * (np.log(8*r_c/wire_radius) - 2)
print(f"  L_coil = {L_coil:.6e} H")
check("Coil L_coil", coil_rcv.L_coil, L_coil, units="H")

# Impedance at f_larmor
XL = 2 * np.pi * f_larmor * L_coil
Z_mag = np.sqrt(R_ac**2 + XL**2)
print(f"  XL at f_larmor = {XL:.6e} Ohm")
print(f"  |Z| at f_larmor = {Z_mag:.6e} Ohm")

# Noise PSD (UNTUNED mode)
S_thermal = 4 * K_B * T_c * R_ac + 4 * K_B * T_s * R_sample_coil
en2_1f = en**2 * (1.0 + f1f / f_larmor)
S_preamp = en2_1f + (i_n * Z_mag)**2
S_V_coil_total = S_thermal + S_preamp
print(f"\n  Noise PSD at f_larmor:")
print(f"    S_thermal = {S_thermal:.6e} V^2/Hz")
print(f"      (Johnson coil: {4*K_B*T_c*R_ac:.6e}, sample: {4*K_B*T_s*R_sample_coil:.6e})")
print(f"    S_preamp = {S_preamp:.6e} V^2/Hz")
print(f"      (e_n^2(1+f1f/f) = {en2_1f:.6e}, (i_n*Z)^2 = {(i_n*Z_mag)**2:.6e})")
print(f"    S_total = {S_V_coil_total:.6e} V^2/Hz")

# Find the index closest to f_larmor in the noise PSD array
bb_freqs = np.fft.fftfreq(N, d=dwell_time)
freqs_rf = np.abs(f_larmor + bb_freqs)
idx_f0 = np.argmin(np.abs(bb_freqs))  # DC = f_larmor in RF frame
S_noise_code_at_f0 = coil_rcv._S_noise[idx_f0]
check("Coil noise PSD at f_larmor (code)", S_noise_code_at_f0, S_V_coil_total,
      tol=0.02, units="V^2/Hz")

# Convert to fT/rtHz for coil (using omega*B1/I sensitivity)
# Coil detects via EMF = omega * Phi, so sensitivity = omega * B1/I * Area * something
# Actually for coil: the noise in field units = sqrt(S_V) / (omega * pickup_sensitivity)
# pickup_sensitivity for coil = B1/I, but we need the "effective area"
# For a coil, the field noise is: sqrt(S_V) / (omega * n * A_eff)
# Simpler: B_noise = sqrt(S_V) / (omega * (B1_I_center_per_turn) * A_eff)
# Actually, let's just compute it directly
# The coil EMF from a uniform field B: EMF = omega * n * A * B (for on-axis)
# So B_noise = sqrt(S_V) / (omega * n * pi * r^2)  -- rough on-axis approx
# More precisely, use same formula as SQUID: B = sqrt(S_V) / (gain * B1_I * A)...
# But the gain differs. Let me just use:
# For coil: V = omega * Psi, where Psi = B1_I * M * V_vox
# and B1_I at center ~ mu0*n/(2*z) for on-axis far field
# Actually simplest: use pick area approach
# Let me compute the equivalent field noise for coil
# coil gain = omega_larmor, B1/I at center = B1_I_coil_center
# For a uniform field: flux = B * n * pi * r^2 (rough), voltage = omega * flux
# So B_noise_coil = sqrt(S_V_coil) / (omega * n * pi * r^2)
# This is the standard way
coil_area = np.pi * r_c**2 * n_c  # effective area [m^2]
B_noise_coil_field = np.sqrt(S_V_coil_total) / (omega_L * coil_area)
print(f"\n  Coil field noise equivalent:")
print(f"    effective area = {coil_area:.6e} m^2")
print(f"    B_noise = sqrt(S_V) / (omega * A) = {B_noise_coil_field*1e15:.1f} fT/rtHz")

print(f"\n  Literature comparison (Coil with n=2):")
print(f"    Computed: {B_noise_coil_field*1e15:.1f} fT/rtHz")
print(f"    Expected: ~4800 fT/rtHz (preamp-noise limited)")
# The dominant term is preamp e_n ~ 1 nV/rtHz
# B_noise ~ e_n / (omega * n * pi * r^2) = 1e-9 / (2*pi*2129*2*pi*0.01) ~ ...
print(f"    Dominant noise: {'preamp e_n' if en2_1f > (i_n*Z_mag)**2 and en2_1f > S_thermal else 'other'}")

# --- Coil with n=600 ---
print(f"\n  --- Coil Noise (n=600, broadband) ---")
coil_600 = ClassicalCoilReceiver(p, coil_turns=600, seed=42)
budget_600 = coil_600.noise_budget()
# Compute analytically
R_dc_600 = rho_w * 2 * np.pi * r_c * 600 / (np.pi * (d_w/2)**2)
R_ac_600 = R_dc_600 * max(1.0, skin_ratio)  # same skin ratio
R_sample_600 = R_sample_coil  # R_sample doesn't depend on n (uses r_c)
L_coil_600 = MU_0 * 600**2 * r_c * (np.log(8*r_c/wire_radius) - 2)
XL_600 = 2 * np.pi * f_larmor * L_coil_600
Z_mag_600 = np.sqrt(R_ac_600**2 + XL_600**2)
S_thermal_600 = 4 * K_B * T_c * R_ac_600 + 4 * K_B * T_s * R_sample_600
en2_1f_600 = en**2 * (1.0 + f1f / f_larmor)
S_preamp_600 = en2_1f_600 + (i_n * Z_mag_600)**2
S_V_coil_600 = S_thermal_600 + S_preamp_600
coil_area_600 = np.pi * r_c**2 * 600
B_noise_600 = np.sqrt(S_V_coil_600) / (omega_L * coil_area_600)
print(f"  R_dc_600 = {R_dc_600:.4f} Ohm")
print(f"  L_coil_600 = {L_coil_600:.6e} H")
print(f"  |Z|_600 at f_larmor = {Z_mag_600:.4f} Ohm")
print(f"  S_thermal_600 = {S_thermal_600:.6e} V^2/Hz")
print(f"  S_preamp_600 = {S_preamp_600:.6e} V^2/Hz")
print(f"    (e_n^2 term = {en2_1f_600:.6e}, (i_n*Z)^2 term = {(i_n*Z_mag_600)**2:.6e})")
print(f"  S_total_600 = {S_V_coil_600:.6e} V^2/Hz")
print(f"  B_noise_600 = {B_noise_600*1e15:.1f} fT/rtHz")
print(f"  Dominant at n=600: {'i_n*Z' if (i_n*Z_mag_600)**2 > en2_1f_600 else 'e_n'}")

# =====================================================================
section("3. IMAGE-LEVEL SNR")
# =====================================================================

print(f"\n  Generating test slice with N={N}...")
# Create a simple phantom: circle of GM, background = 0
tissue_map = np.zeros((N, N), dtype=int)
Y, X = np.ogrid[-N//2:N//2, -N//2:N//2]
tissue_map[X**2 + Y**2 < (N//4)**2] = 2  # GM circle, radius N/4

p_img = SpinEchoParams(B0=B0, B_prepol=B_prepol, TE=TE, TR=TR,
                        matrix_size=N, fov=fov, slice_thickness=slice_thickness,
                        readout_bw=readout_bw, t_pol=t_pol, t_transfer=t_transfer)
engine_img = MRISignalEngine(p_img)
engine_img.prepare(tissue_map)

print(f"  M_max in ideal image: {engine_img.ideal_image.max():.6e} A/m")
check("Ideal image max (GM M_perp)", engine_img.ideal_image.max(), M_perp_analytical, units="A/m")

# Acquire with SQUID and Coil
squid_rcv = SQUIDQuantumReceiver(p_img, seed=42)
coil_rcv_img = ClassicalCoilReceiver(p_img, seed=42)

k_squid = squid_rcv.acquire(engine_img)
k_coil = coil_rcv_img.acquire(engine_img)

img_squid = ReceiverBase.reconstruct(k_squid)
img_coil = ReceiverBase.reconstruct(k_coil)

# Define signal and noise regions
signal_mask = (X**2 + Y**2 < (N//8)**2)  # inner circle
noise_mask = (X**2 + Y**2 > (3*N//8)**2)  # outer ring (background)

sig_squid = np.mean(img_squid[signal_mask])
noise_squid = np.std(img_squid[noise_mask])
snr_squid = sig_squid / noise_squid if noise_squid > 0 else float('inf')

sig_coil = np.mean(img_coil[signal_mask])
noise_coil = np.std(img_coil[noise_mask])
snr_coil = sig_coil / noise_coil if noise_coil > 0 else float('inf')

print(f"\n  SQUID image:")
print(f"    Signal (mean in center): {sig_squid:.6e}")
print(f"    Noise (std in background): {noise_squid:.6e}")
print(f"    SNR = {snr_squid:.1f} ({20*np.log10(snr_squid):.1f} dB)")

print(f"\n  Coil image:")
print(f"    Signal (mean in center): {sig_coil:.6e}")
print(f"    Noise (std in background): {noise_coil:.6e}")
print(f"    SNR = {snr_coil:.1f} ({20*np.log10(max(snr_coil, 1e-30)):.1f} dB)")

print(f"\n  SNR ratio SQUID/Coil: {snr_squid/max(snr_coil, 1e-30):.1f}x")
print(f"  SNR advantage (dB): {20*np.log10(snr_squid/max(snr_coil, 1e-30)):.1f} dB")

# =====================================================================
section("4. NORMALIZATION EFFECT")
# =====================================================================

p995_squid = np.percentile(img_squid, 99.5)
p995_coil = np.percentile(img_coil, 99.5)

print(f"\n  Before normalization:")
print(f"    SQUID p99.5: {p995_squid:.6e}")
print(f"    Coil  p99.5: {p995_coil:.6e}")
print(f"    Ratio SQUID/Coil p99.5: {p995_squid/max(p995_coil, 1e-30):.2f}")

img_squid_norm = img_squid / p995_squid
img_coil_norm = img_coil / p995_coil

sig_squid_norm = np.mean(img_squid_norm[signal_mask])
noise_squid_norm = np.std(img_squid_norm[noise_mask])
snr_squid_norm = sig_squid_norm / noise_squid_norm if noise_squid_norm > 0 else float('inf')

sig_coil_norm = np.mean(img_coil_norm[signal_mask])
noise_coil_norm = np.std(img_coil_norm[noise_mask])
snr_coil_norm = sig_coil_norm / noise_coil_norm if noise_coil_norm > 0 else float('inf')

print(f"\n  After normalization (divide by p99.5):")
print(f"    SQUID: signal={sig_squid_norm:.4f}, noise={noise_squid_norm:.6f}, SNR={snr_squid_norm:.1f}")
print(f"    Coil:  signal={sig_coil_norm:.4f}, noise={noise_coil_norm:.6f}, SNR={snr_coil_norm:.1f}")
print(f"    SNR ratio SQUID/Coil after norm: {snr_squid_norm/max(snr_coil_norm, 1e-30):.1f}x")

compression_factor = (snr_squid/max(snr_coil, 1e-30)) / (snr_squid_norm/max(snr_coil_norm, 1e-30))
print(f"\n  Normalization compression of SQUID-Coil difference: {compression_factor:.2f}x")
if abs(compression_factor - 1.0) < 0.1:
    print(f"    [INFO] Minimal compression: normalization preserves SNR ratio")
else:
    print(f"    [INFO] Normalization compresses SNR difference by {compression_factor:.2f}x")
    if snr_coil < 2.0:
        print(f"    [INFO] Coil SNR < 2: noise-dominated, p99.5 mostly captures noise")

# =====================================================================
section("5. FFT NOISE SCALING VERIFICATION")
# =====================================================================

print(f"\n  Testing: generate noise with known flat PSD and measure actual PSD")
# Use a flat PSD S0 = 1e-18 V^2/Hz
S0 = 1e-18
n_samples = 256
dt_test = dwell_time
bw_bin = 1.0 / (n_samples * dt_test)

# The code's formula: noise_amp = sqrt(S * N * bw_bin / 2) = sqrt(S * N / (2 * N * dt)) = sqrt(S / (2*dt))
S_psd = np.full(n_samples, S0)
noise_amp = np.sqrt(S_psd * n_samples * bw_bin / 2)

# Generate many realizations and measure PSD
n_trials = 50000
rng = np.random.default_rng(12345)
measured_psd_sum = np.zeros(n_samples)

for _ in range(n_trials):
    noise_td = rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)
    noise_fft = np.fft.fft(noise_td) * noise_amp
    result_td = np.fft.ifft(noise_fft)
    # Time-domain variance = sum of S(f)*df for each f
    # Measure PSD from time domain: take FFT, compute |X[k]|^2 / (N * df)
    X = np.fft.fft(result_td)
    measured_psd_sum += np.abs(X)**2

# Average PSD: E[|X[k]|^2] / (N * df) should give S(f)
# Actually, PSD = E[|X[k]|^2] * dt / N for one-sided,
# or = E[|X[k]|^2] / (N * bw_bin) for two-sided
measured_psd_avg = measured_psd_sum / n_trials
# For complex noise x[n] with var sigma^2 per component,
# X[k] = sum x[n] e^(-j2pi kn/N) has E[|X[k]|^2] = N * 2 * sigma^2  (for CN(0,2) per sample)
# After scaling by noise_amp: Y[k] = X[k] * A[k], E[|Y[k]|^2] = E[|X[k]|^2] * |A[k]|^2
# Then z[n] = IFFT(Y) has variance = (1/N^2) * sum |A[k]|^2 * E[|X[k]|^2]
# = (1/N^2) * sum A[k]^2 * 2N = (2/N) * sum A[k]^2
# And the PSD is: S(f_k) = E[|Z[k]|^2] * dt where Z[k] = FFT(z)
# Z[k] = Y[k]/1 (ifft then fft gives back same), wait no.
# Actually z = IFFT(Y), and Z' = FFT(z) = Y (since FFT(IFFT(Y)) = Y).
# So E[|Z'[k]|^2] = E[|Y[k]|^2] = |A[k]|^2 * E[|X[k]|^2] = |A[k]|^2 * 2N
# PSD estimate = E[|Z'[k]|^2] * dt^2 / (N * dt) = |A[k]|^2 * 2N * dt / N = |A[k]|^2 * 2 * dt
# But |A[k]|^2 = S[k] * N * bw_bin / 2 = S[k] / (2 * dt)
# So PSD = S[k] / (2*dt) * 2 * dt = S[k].

# But we measured E[|X_out[k]|^2] where X_out = FFT(ifft(Y)).
# ifft(Y)[n] = (1/N) sum Y[k] e^(j2pi kn/N)
# FFT(ifft(Y))[m] = sum_n (1/N) sum_k Y[k] e^(j2pi kn/N) e^(-j2pi mn/N)
#                  = sum_k Y[k] (1/N) sum_n e^(j2pi(k-m)n/N) = Y[m]
# So FFT(IFFT(Y)) = Y. Good.
# measured_psd_avg[k] = E[|Y[k]|^2] = |A[k]|^2 * E[|X[k]|^2]
# E[|X[k]|^2] for X[k] = FFT of (re + j*im) where re, im ~ N(0,1):
#   Each X[k] = sum (re[n]+j*im[n]) e^(-j*2pi*kn/N)
#   Var(Re(X[k])) = N, Var(Im(X[k])) = N
#   E[|X[k]|^2] = 2N
# So measured_psd_avg[k] should be |A[k]|^2 * 2N = S0 * N * bw_bin / 2 * 2N = S0 * N^2 * bw_bin
# PSD in V^2/Hz = measured_psd_avg[k] / (N^2 * bw_bin) should = S0

# Let's try this approach
measured_S = measured_psd_avg / (n_samples**2 * bw_bin)

# Actually, it's simpler. Let's measure the time-domain variance directly.
# If noise has PSD S0 (flat, two-sided), total variance = S0 * total_bandwidth
# total BW = N * bw_bin = 1/dt
# So variance per component should be S0 / (2*dt)  ... no.
# Actually for two-sided PSD S0 over bandwidth from -BW/2 to BW/2:
# variance = integral S0 df = S0 * BW = S0 / dt  ... for complex signal, both Re and Im
# Hmm, let me just empirically verify.

# Simpler approach: measure time-domain variance of the noise output
td_vars = []
for _ in range(10000):
    noise_td2 = rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)
    noise_fft2 = np.fft.fft(noise_td2) * noise_amp
    result_td2 = np.fft.ifft(noise_fft2)
    td_vars.append(np.var(result_td2.real))

mean_var = np.mean(td_vars)
# Expected: sum over all bins of S(f_k) * df where df = bw_bin
# = N * S0 * bw_bin = S0 / dt  ... for one-sided
# Actually: the total two-sided spectral power = sum_k S0 * bw_bin (over positive and negative)
# Since we have N bins each of width bw_bin, total = N * S0 * bw_bin = S0 / dt
# For complex noise, the real part gets half: var(Re) = S0 / (2*dt)
# Wait, let me think more carefully.
#
# The noise generation: z[n] = IFFT( FFT(w[n]) * A[k] )
# w[n] = re[n] + j*im[n], re,im ~ N(0,1)
# W[k] = FFT(w) has E[|W[k]|^2] = 2N
# Y[k] = W[k]*A[k], E[|Y[k]|^2] = |A[k]|^2 * 2N
# z[n] = IFFT(Y), var(z) = (1/N^2) sum |A[k]|^2 * 2N = (2/N) sum |A[k]|^2
# sum |A[k]|^2 = sum S0 * N * bw_bin / 2 = N * S0 / (2*dt)
# var(z) = (2/N) * N * S0 / (2*dt) = S0 / dt
# var(Re(z)) = S0 / (2*dt)

expected_var = S0 / (2 * dt_test)
print(f"  Flat PSD S0 = {S0:.2e} V^2/Hz")
print(f"  N = {n_samples}, dt = {dt_test:.6e} s, bw_bin = {bw_bin:.2f} Hz")
print(f"  Expected time-domain var(Re): S0 / (2*dt) = {expected_var:.6e}")
print(f"  Measured time-domain var(Re): {mean_var:.6e}")
check("FFT noise scaling (variance)", mean_var, expected_var, tol=0.03, units="V^2")

# Also check spectral flatness
# PSD = E[|Y[k]|^2] / (2N) * 1/bw_bin ... hmm, let me just check the ratio
psd_ratio = measured_S / S0
print(f"\n  Spectral flatness check:")
print(f"    Mean PSD / S0 = {np.mean(psd_ratio):.4f} (should be 1.0)")
print(f"    Std PSD / S0 = {np.std(psd_ratio):.4f}")
check("FFT noise PSD recovery (mean)", np.mean(measured_S), S0, tol=0.03, units="V^2/Hz")

# =====================================================================
section("6. B1/I0 VERIFICATION")
# =====================================================================

print(f"\n  --- On-axis Biot-Savart vs analytical formula ---")
# Analytical: B_z = mu0 * I * r^2 / (2 * (r^2 + z^2)^(3/2))
# Code uses elliptic integrals for general rho

test_r = 0.05  # m
test_z = 0.05  # m
test_n = 1

# On-axis (rho=0)
B_analytical = MU_0 * test_n * test_r**2 / (2 * (test_r**2 + test_z**2)**1.5)
B_code = biot_savart_Bz(test_r, test_n, np.array([0.0]), test_z)[0]
print(f"  Single turn, r={test_r}m, z={test_z}m, rho=0:")
print(f"    Analytical: B_z/I = mu0*r^2/(2*(r^2+z^2)^(3/2)) = {B_analytical:.6e} T/A")
check("Biot-Savart on-axis (n=1)", B_code, B_analytical, units="T/A")

# Multi-turn scaling
B_code_5 = biot_savart_Bz(test_r, 5, np.array([0.0]), test_z)[0]
check("Biot-Savart linear in n (n=5 vs 5*n=1)", B_code_5, 5 * B_code, units="T/A")

B_code_100 = biot_savart_Bz(test_r, 100, np.array([0.0]), test_z)[0]
check("Biot-Savart linear in n (n=100 vs 100*n=1)", B_code_100, 100 * B_code, units="T/A")

# Off-axis check at rho = r (worst case for elliptic integrals)
print(f"\n  Off-axis at rho = r (maximum k^2):")
rho_test = test_r * 0.99  # slightly less than r to avoid singularity
# Full elliptic integral formula
r = test_r
n = test_n
z_abs = abs(test_z)
alpha2 = (r + rho_test)**2 + z_abs**2
beta2 = (r - rho_test)**2 + z_abs**2
k2 = 4 * r * rho_test / alpha2
K = ellipk(k2)
E = ellipe(k2)
B_exact = (MU_0 * n / (2 * np.pi)) / np.sqrt(alpha2) * \
          (K + (r**2 - rho_test**2 - z_abs**2) / beta2 * E)
B_code_off = biot_savart_Bz(test_r, test_n, np.array([rho_test]), test_z)[0]
check("Biot-Savart off-axis (rho~r)", B_code_off, B_exact, units="T/A")

# Check different z values
print(f"\n  Distance dependence:")
for z_test in [0.01, 0.03, 0.05, 0.10, 0.20]:
    B_an = MU_0 * test_r**2 / (2 * (test_r**2 + z_test**2)**1.5)
    B_cd = biot_savart_Bz(test_r, 1, np.array([0.0]), z_test)[0]
    err = abs(B_cd - B_an) / B_an * 100
    status = "PASS" if err < 1 else "FAIL"
    print(f"    z={z_test:.2f}m: analytical={B_an:.4e}, code={B_cd:.4e}, err={err:.4f}% [{status}]")
    if err >= 1:
        fail_count += 1
    else:
        pass_count += 1

# =====================================================================
section("7. CORRECTED COIL (n=600, INA217)")
# =====================================================================

# INA217: en = 1.3 nV/rtHz, in = 0.8 pA/rtHz (typical instrumentation amp)
en_ina = 1.3e-9  # V/rtHz
in_ina = 0.8e-12  # A/rtHz
n_600 = 600

p_600 = SpinEchoParams(B0=B0, B_prepol=B_prepol, TE=TE, TR=TR,
                         matrix_size=N, fov=fov, slice_thickness=slice_thickness,
                         readout_bw=readout_bw, t_pol=t_pol, t_transfer=t_transfer)

coil_600 = ClassicalCoilReceiver(p_600, coil_turns=n_600,
                                   preamp_en=en_ina, preamp_in=in_ina,
                                   seed=42)

print(f"\n  Coil parameters: n={n_600}, r={coil_600.coil_radius}m")
print(f"  Preamp: e_n={en_ina*1e9:.1f} nV/rtHz, i_n={in_ina*1e12:.1f} pA/rtHz")
print(f"  Mode: {coil_600.mode}")

# Compute noise analytically
R_dc_600a = rho_w * 2 * np.pi * r_c * n_600 / (np.pi * (d_w/2)**2)
R_ac_600a = R_dc_600a  # no skin effect at 2 kHz
L_coil_600a = MU_0 * n_600**2 * r_c * (np.log(8*r_c/wire_radius) - 2)
XL_600a = 2 * np.pi * f_larmor * L_coil_600a
Z_600a = np.sqrt(R_ac_600a**2 + XL_600a**2)

S_thermal_600a = 4 * K_B * T_c * R_ac_600a + 4 * K_B * T_s * R_sample_coil
en2_1f_600a = en_ina**2 * (1.0 + f1f / f_larmor)
S_preamp_600a = en2_1f_600a + (in_ina * Z_600a)**2
S_total_600a = S_thermal_600a + S_preamp_600a

print(f"\n  Noise breakdown:")
print(f"    R_dc = {R_dc_600a:.4f} Ohm")
print(f"    L = {L_coil_600a:.4e} H")
print(f"    |Z| at f_larmor = {Z_600a:.4f} Ohm")
print(f"    S_thermal = {S_thermal_600a:.6e} V^2/Hz")
print(f"      (Johnson coil: {4*K_B*T_c*R_ac_600a:.6e}, sample: {4*K_B*T_s*R_sample_coil:.6e})")
print(f"    S_preamp = {S_preamp_600a:.6e} V^2/Hz")
print(f"      (e_n^2 = {en2_1f_600a:.6e}, (i_n*Z)^2 = {(in_ina*Z_600a)**2:.6e})")
print(f"    S_total = {S_total_600a:.6e} V^2/Hz")

area_600 = np.pi * r_c**2 * n_600
B_noise_600a = np.sqrt(S_total_600a) / (omega_L * area_600)
print(f"    B_noise_600 = {B_noise_600a*1e15:.1f} fT/rtHz")

# Dominant source
sources = {
    'Johnson coil': 4*K_B*T_c*R_ac_600a,
    'Johnson sample': 4*K_B*T_s*R_sample_coil,
    'preamp e_n': en2_1f_600a,
    'preamp i_n*Z': (in_ina*Z_600a)**2,
}
dominant = max(sources, key=sources.get)
print(f"    Dominant noise source: {dominant} ({sources[dominant]:.6e} V^2/Hz)")

# Generate images
engine_600 = MRISignalEngine(p_600)
engine_600.prepare(tissue_map)

k_600 = coil_600.acquire(engine_600)
img_600 = ReceiverBase.reconstruct(k_600)

sig_600 = np.mean(img_600[signal_mask])
noise_600_val = np.std(img_600[noise_mask])
snr_600 = sig_600 / noise_600_val if noise_600_val > 0 else float('inf')
psnr_600 = 20 * np.log10(img_600.max() / noise_600_val) if noise_600_val > 0 else float('inf')

# Also generate SQUID for comparison
squid_600 = SQUIDQuantumReceiver(p_600, seed=42)
k_sq600 = squid_600.acquire(engine_600)
img_sq600 = ReceiverBase.reconstruct(k_sq600)
sig_sq600 = np.mean(img_sq600[signal_mask])
noise_sq600 = np.std(img_sq600[noise_mask])
snr_sq600 = sig_sq600 / noise_sq600 if noise_sq600 > 0 else float('inf')
psnr_sq600 = 20 * np.log10(img_sq600.max() / noise_sq600) if noise_sq600 > 0 else float('inf')

print(f"\n  Image-level results:")
print(f"    Coil (n=600, INA217):")
print(f"      Signal: {sig_600:.6e}")
print(f"      Noise:  {noise_600_val:.6e}")
print(f"      SNR = {snr_600:.1f} ({20*np.log10(max(snr_600, 1e-30)):.1f} dB)")
print(f"      PSNR = {psnr_600:.1f} dB")
print(f"      Visible: {'YES' if snr_600 > 2 else 'NO'} (SNR {'>' if snr_600 > 2 else '<'} 2)")

print(f"\n    SQUID (reference):")
print(f"      Signal: {sig_sq600:.6e}")
print(f"      Noise:  {noise_sq600:.6e}")
print(f"      SNR = {snr_sq600:.1f} ({20*np.log10(max(snr_sq600, 1e-30)):.1f} dB)")
print(f"      PSNR = {psnr_sq600:.1f} dB")

print(f"\n    Comparison:")
print(f"      SNR ratio SQUID / Coil-600: {snr_sq600/max(snr_600, 1e-30):.2f}x")
print(f"      Coil-600 PSNR {'>' if psnr_600 > psnr_sq600 else '<'} SQUID PSNR")

# Also compare with default coil (n=2)
print(f"\n    vs default Coil (n=2):")
print(f"      Default coil SNR:  {snr_coil:.1f} ({20*np.log10(max(snr_coil, 1e-30)):.1f} dB)")
print(f"      n=600 coil SNR:    {snr_600:.1f} ({20*np.log10(max(snr_600, 1e-30)):.1f} dB)")
print(f"      Improvement:       {20*np.log10(max(snr_600/max(snr_coil, 1e-30), 1e-30)):.1f} dB")


# =====================================================================
section("SUMMARY")
# =====================================================================
print(f"\n  Total checks: {pass_count + fail_count}")
print(f"  PASSED: {pass_count}")
print(f"  FAILED: {fail_count}")
if fail_count == 0:
    print(f"\n  ALL CHECKS PASSED")
else:
    print(f"\n  {fail_count} CHECK(S) FAILED — review above for discrepancies > 1%")
