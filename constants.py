# -*- coding: utf-8 -*-
"""
constants.py — SI Physical Constants for MRI Simulation
========================================================

All values from NIST CODATA 2018 unless otherwise noted.
Derived quantities use the Curie-law equilibrium magnetization
for water protons at body temperature (310 K).

Reference: Haacke et al., "MRI: Physical Principles" (1999), Eq. 4.28.
"""
from __future__ import annotations
import numpy as np

# ===================================================================
# Fundamental Constants (SI)
# ===================================================================
GAMMA_H      = 42.577478518e6   # ¹H gyromagnetic ratio [Hz/T]
GAMMA_RAD    = 2 * np.pi * GAMMA_H  # angular gyromagnetic ratio [rad/(s·T)]
K_B          = 1.380649e-23     # Boltzmann constant [J/K]
MU_0         = 4 * np.pi * 1e-7 # vacuum permeability [H/m]
HBAR         = 1.054571817e-34  # reduced Planck constant [J·s]
PHI_0        = 2.067833848e-15  # magnetic flux quantum [Wb]
N_A          = 6.02214076e23    # Avogadro number [1/mol]
T_BODY       = 310.0            # body temperature [K]

# ===================================================================
# Derived: Equilibrium Magnetization
# ===================================================================
# M₀ = n_H × γ²ℏ² × I(I+1) × B / (3 k_B T), for spin-1/2: I(I+1) = 3/4
# → M₀/B = n_H × γ² × ℏ² / (4 k_B T)
N_H_WATER    = 2 * 1000.0 / 18.015e-3 * N_A  # ~6.69e28 protons/m³
M0_PER_TESLA = N_H_WATER * GAMMA_RAD**2 * HBAR**2 / (4 * K_B * T_BODY)
