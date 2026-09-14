# -*- coding: utf-8 -*-
"""
tissue.py — MRI Tissue Relaxation Database
============================================

Tissue-specific MRI parameters (T1, T2, proton density) with
field-dependent scaling models. Label scheme follows BrainWeb V2.

Literature sources for relaxation parameters at 3 T:
┌─────────────┬──────────────────────────────────────────────────────────────┐
│ Parameter   │ Primary sources                                            │
├─────────────┼──────────────────────────────────────────────────────────────┤
│ T1 (3 T)    │ [1] Wansapura et al., JMRI 9(4):531-538, 1999 — Table 1   │
│             │ [2] Stanisz et al., MRM 54(3):507-512, 2005 — Table 1      │
│             │ [3] Gold et al., AJR 183:343-351, 2004 — Table 1           │
│             │ [4] de Bazelaire et al., Radiology 230(3):652-659, 2004    │
│ T2 (3 T)    │ [1] Wansapura — Table 1, [2] Stanisz — Table 1            │
│             │ [5] Zhao et al., MRM 58(3):592-597, 2007 — blood T2       │
│ PD          │ [6] Collins et al., IEEE TMI 17(3):463-468, 1998 (BrainWeb)│
│             │ [7] Tofts, "Quantitative MRI of the Brain" (2003), Ch. 11  │
│ alpha       │ Re-fitted to ULF data [Z09],[OR22]; original form [8]     │
│ T2*/T2      │ [1] Wansapura — Table 1 (T2* measured at 3 T)             │
│             │ [9] Yablonskiy & Haacke, MRM 32:749-763, 1994             │
│ CSF T1      │ [10] Rooney et al., MRM 57(2):308-318, 2007 — Table 2     │
│ Blood T1    │ [11] Lu et al., MRM 52(4):679-682, 2004                   │
│ Skull/Bone  │ [12] Robson et al., MRM 49:29-37, 2003 — ultrashort TE    │
│             │ [13] Du et al., NMR Biomed 23(10):1086-1100, 2010         │
│ Marrow      │ [14] Shen et al., JMRI 33(5):1236-1239, 2011             │
└─────────────┴──────────────────────────────────────────────────────────────┘

Label scheme follows BrainWeb V2 conventions:
    0=Background, 1=CSF, 2=Grey Matter, 3=White Matter,
    4=Fat, 5=Muscle, 6=Muscle/Skin, 7=Skull,
    8=Vessels, 9=Connective, 10=Dura, 11=Bone Marrow
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass
class TissueProperties:
    """MRI tissue relaxation parameters.

    T1_3T, T2: relaxation times at 3T in [ms].
    PD: proton density relative to water (0-1).
    alpha: T1 field-dependence exponent.
    t2star_ratio: T2*/T2 ratio at high field.
    T1_low: T1 asymptote at very low field [ms]. Per-tissue, from
            measured NMRD data. Replaces the old max(1.5*T2, 30) floor.
    T2_low: T2 at very low field [ms]. At ULF, T2 approaches T2_intrinsic
            (dipolar contribution only, no susceptibility effects).
    """
    name: str;  label: int;  PD: float
    T1_3T: float;  T2: float
    alpha: float = 0.35;  t2star_ratio: float = 0.8
    T1_low: float = 0.0   # 0 = auto (use T2 as floor)
    T2_low: float = 0.0   # 0 = auto (same as T2 at 3T)
    # CSF-specific Lorentzian NMRD parameters (set only for CSF)
    _nmrd_A: float = 0.0    # dispersion amplitude [s⁻¹]
    _nmrd_fc: float = 0.0   # corner frequency [Hz]

    def _raw_T1(self, B0: float) -> float:
        """Raw T1 without any T2 constraint. Used internally."""
        if B0 <= 0 or self.T1_3T <= 0:
            return self._effective_T1_low()
        # CSF: use Lorentzian NMRD model (captures steep ULF dispersion)
        if self._nmrd_A > 0:
            from constants import GAMMA_H
            f = GAMMA_H * B0
            R1_plateau = 1000.0 / self.T1_3T  # [1/s] high-field asymptote
            R1 = R1_plateau + self._nmrd_A / (1.0 + (f / self._nmrd_fc)**2)
            return 1000.0 / R1  # [ms]
        # Other tissues: power law
        return self._effective_T1_low() + (self.T1_3T - self._effective_T1_low()) * (B0 / 3.0)**self.alpha

    def T1_at_field(self, B0: float) -> float:
        """T1 at field B0 [T], returned in [ms].

        For most tissues: power law T1(B0) = T1_low + (T1_3T - T1_low) × (B0/3)^α
        For CSF: Lorentzian NMRD model R1(f) = R1_plateau + A/(1 + (f/f_c)²)

        Parameters fit to measured ULF relaxometry:
          - Zotev et al., IEEE Trans. Appl. Supercond. 19:823 (2009): T1 at 46 μT
          - O'Reilly et al., MRM 87:884 (2022): T1 at 50 mT
          - Inglis et al., PNAS 110:19194 (2013): T1 at 130 μT

        At extreme low field, T1 → T2 (BPP extreme narrowing limit).
        Measured data confirms T1 ≈ T2 within error at 46 μT (Zotev 2009).
        """
        return self._raw_T1(B0)

    def _raw_T2(self, B0: float) -> float:
        """Raw T2 from R2 model without T1 constraint. Used internally."""
        if self.T2 <= 0:
            return 0.0
        t2_low = self._effective_T2_low()
        if t2_low <= 0 or t2_low == self.T2 or B0 <= 0:
            return t2_low
        R2_3T = 1000.0 / self.T2
        R2_low = 1000.0 / t2_low
        k_R2 = (R2_3T - R2_low) / 3.0
        R2_B0 = R2_low + k_R2 * B0
        return 1000.0 / R2_B0

    def T2_at_field(self, B0: float) -> float:
        """T2 at field B0 [T], returned in [ms].

        Two-component model:
          1. R2 intrinsic model: T2 increases at low field as susceptibility
             effects vanish. T2 → T2_low (intrinsic dipolar T2).
          2. T1 constraint: at extreme narrowing, T2 cannot exceed T1.
             T2 = min(T2_intrinsic, 1.05 × T1_raw).

        The 1.05 factor allows T2 to slightly exceed T1, consistent with
        measurements at 46 μT (Zotev 2009: WM T1=75, T2=79; T2/T1≈1.05).

        References:
          - Mitsumori et al., MRM 68:947 (2012): R2 field dependence
          - Zotev et al., IEEE TAS 19:823 (2009): T2 at 46 μT
          - O'Reilly et al., MRM 87:884 (2022): T2 at 50 mT
        """
        t2_intr = self._raw_T2(B0)
        if t2_intr <= 0:
            return 0.0
        # At extreme narrowing, T2 is bounded by T1
        t1_raw = self._raw_T1(B0)
        if t1_raw > 0:
            return min(t2_intr, 1.05 * t1_raw)
        return t2_intr

    def T2star_at_field(self, B0: float) -> float:
        """T2* at field B0 [T], returned in [ms].

        T2* = 1 / (1/T2(B0) + R2'), where R2' is mesoscopic
        susceptibility-induced dephasing (refocused by spin echo).
        R2' ∝ B0 (Yablonskiy & Haacke, MRM 32:749, 1994).
        At ULF: R2' → 0, T2* → T2(B0) → T2_low.
        """
        t2 = self.T2_at_field(B0)
        if t2 <= 0:
            return 0.0
        if self.t2star_ratio >= 1.0 or B0 <= 0:
            return t2
        # R2' at 3T from the 3T T2 and T2* ratio
        R2_prime_3T = (1.0 / (self.T2 * self.t2star_ratio) - 1.0 / self.T2)
        R2_prime = R2_prime_3T * (B0 / 3.0)
        return 1.0 / (1.0 / t2 + R2_prime)

    def _effective_T1_low(self) -> float:
        """T1 low-field asymptote. Uses T1_low if set, else T2 (BPP limit)."""
        if self.T1_low > 0:
            return self.T1_low
        return max(self.T2, 30.0)

    def _effective_T2_low(self) -> float:
        """T2 low-field value. Uses T2_low if set, else T2 (field-independent)."""
        if self.T2_low > 0:
            return self.T2_low
        return self.T2


# Brain tissue relaxation parameters at 3 T.
# ──────────────────────────────────────────────────────────────────────────
# Each value is annotated with its source reference (numbered above).
# Where multiple papers exist, the chosen value and rationale are noted.
#
# PD values for labels 0-7 follow BrainWeb [6] fuzzy tissue model,
# cross-checked with Tofts [7] Ch. 11 (within ±0.02 for GM/WM).
#
# alpha (T1 field-dependence exponent): functional form from Bottomley [8],
# but values RE-FITTED to match measured ULF data from [Z09] and [OR22].
# Original Bottomley values (1-100 MHz fit): GM~0.34, WM~0.38, fat~0.17.
# Our re-fitted values span the full 46μT-3T range and differ significantly.
#
# t2star_ratio (T2*/T2 at 3T) from Wansapura [1] Table 1 where measured.
# GM: T2*=52±6ms / T2=80ms = 0.65. WM: T2*≈45ms / T2≈70ms ≈ 0.64→0.65.
# Other tissues estimated via Yablonskiy [9] static dephasing framework.
# ──────────────────────────────────────────────────────────────────────────
# T1_low and alpha re-fitted to measured ULF relaxometry data:
#   [Z09] Zotev et al., IEEE Trans. Appl. Supercond. 19:823 (2009): T1,T2 at 46 μT
#   [OR22] O'Reilly et al., MRM 87:884 (2022): T1,T2 at 50 mT
#   [I13] Inglis et al., PNAS 110:19194 (2013): T1,T2 at 130 μT
# T2_low from [Z09] T2 at 46 μT (ULF intrinsic T2).
# Muscle PD corrected from 1.0 to 0.80 (literature: 0.75-0.80).
TISSUE_DB: Dict[int, TissueProperties] = {
    #                                    PD   T1@3T  T2@3T  alpha  t2*/t2  T1_low  T2_low  Source notes
    0:  TissueProperties("Background",  0, 0.00,    0,    0, 0.0, 1.0,    0,    0),      # —
    1:  TissueProperties("CSF",         1, 1.00, 4000, 2000, 0.60, 0.95, 340, 355, _nmrd_A=2.657, _nmrd_fc=189e3),  # Lorentzian NMRD: fits 3T(4000), 50mT(3695), 46μT(344) simultaneously. [Z09]+[OR22]. T2_low: [Z09] 355±15.
    2:  TissueProperties("Grey Matter", 2, 0.86, 1331,   80, 0.39, 0.65,  85, 106),     # T1: [1] 1331±13 (exact).     # T1_low: [Z09] 103±5→fit 85. T2_low: [Z09] 106±11. alpha=0.39.
    3:  TissueProperties("White Matter",3, 0.77,  832,   70, 0.33, 0.65,  56, 100),     # T1: [1] 832±10 (exact). T2@3T: [2] Stanisz 69±3→70. T1_low: [Z09] 75±2→fit 56. T2_low: 100 (intrinsic, [OR22] 102±6). At ULF, T2 constrained by min(100, 1.05×T1)≈80, matching [Z09] 79±11.
    4:  TissueProperties("Fat",         4, 1.00,  380,   60, 0.51, 0.80,  94,  90),     # T1_low: [I13] 96±2→fit 94. T2_low: [OR22] 90±13. alpha=0.51.
    5:  TissueProperties("Muscle",      5, 0.80, 1400,   40, 0.60, 0.70,  55,  40),     # T1_low: [OR22] 171@50mT→fit 55. PD: 0.80 (corrected from 1.0). T2_low: [OR22] 39±2→40.
    6:  TissueProperties("Muscle/Skin", 6, 0.80, 1400,   40, 0.60, 0.70,  55,  40),     # Same as Muscle. PD: 0.80 (corrected).
    7:  TissueProperties("Skull",       7, 0.30,  300,   20, 0.30, 0.60,  30,  20),     # T1_low=30 (conservative, no ULF data). T2_low≈T2 (short T2 tissue).
    8:  TissueProperties("Vessels",     8, 0.90, 1500,  100, 0.50, 0.70, 181, 120),     # T1_low: [I13] 190±39→fit 181. T2_low: est ~120ms (less susceptibility at ULF).
    9:  TissueProperties("Connective",  9, 0.80,  500,   50, 0.40, 0.70,  60,  50),     # Estimated. T1_low~T2. Minor BrainWeb label.
   10:  TissueProperties("Dura",       10, 0.50,  600,   40, 0.40, 0.70,  50,  40),     # Estimated. Minor BrainWeb label.
   11:  TissueProperties("Bone Marrow",11, 0.95,  550,   50, 0.51, 0.80,  94,  90),     # Similar to fat (fatty marrow). T1_low from fat analogy.
}
