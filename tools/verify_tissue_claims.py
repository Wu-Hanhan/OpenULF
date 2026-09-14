#!/usr/bin/env python3
"""
verify_tissue_claims.py — Independent verification of T1/T2 claims
===================================================================

Verifies claims 1-7 against the actual published values extracted
directly from the original papers (read/downloaded independently).

Sources verified by reading actual papers:
  - Zotev et al., arXiv:0811.0389 / IEEE TAS 19(3):871-874, 2009
    "SQUID-based Microtesla MRI for In Vivo Relaxometry of the Human Brain"
    Read PDF pages 1-4. Results on page 3.

  - O'Reilly & Webb, MRM 87(2):884-895, 2022 (PMID 34520068)
    "In vivo T1 and T2 relaxation time maps of brain tissue, skeletal
     muscle, and lipid measured in healthy volunteers at 50 mT"
    Extracted from PubMed abstract (which quotes the actual numbers).

  - Vesanen et al., MRM 69(6):1795-1804, 2013 (PMID 22807201)
    "Hybrid ultra-low-field MRI and magnetoencephalography system based
     on a commercial whole-head neuromagnetometer"
    This paper describes the SYSTEM, NOT relaxation measurements.
"""
import numpy as np
import sys

# =====================================================================
# GROUND TRUTH from actual papers
# =====================================================================

# From Zotev et al. 2009, page 3, Section IV Results:
# "We obtained the following T2 values at 46 uT field:
#  106+/-11 ms for gray matter, 79+/-11 ms for white matter,
#  355+/-15 ms for CSF, and 120+/-7 ms for scalp."
#
# "We obtained the following T1 values at Br=46 uT relaxation field:
#  103+/-5 ms for gray matter, 75+/-2 ms for white matter,
#  344+/-9 ms for CSF, and 124+/-7 ms for scalp."

ZOTEV_2009 = {
    "paper": "Zotev et al., IEEE TAS 19(3):871-874, 2009 (arXiv:0811.0389)",
    "B0_uT": 46,
    "T1": {"GM": (103, 5), "WM": (75, 2), "CSF": (344, 9), "Scalp": (124, 7)},
    "T2": {"GM": (106, 11), "WM": (79, 11), "CSF": (355, 15), "Scalp": (120, 7)},
}

# From O'Reilly & Webb 2022, PubMed abstract (PMID 34520068):
# "The average T1 times of gray matter, white matter, and cerebrospinal fluid
#  (CSF) were 327 +/- 10 ms, 275 +/- 5 ms, and 3695 +/- 287 ms, respectively.
#  Corresponding values of T2 were 102 +/- 6 ms, 102 +/- 6 ms, and 1584 +/- 124 ms.
#  T1 times in the calf muscle were measured to be 171 +/- 11 ms and were
#  130 +/- 5 ms in subcutaneous and bone marrow lipid.
#  Corresponding T2 times were 39 +/- 2 ms in muscle and 90 +/- 13 ms in lipid."

OREILLY_2022 = {
    "paper": "O'Reilly & Webb, MRM 87(2):884-895, 2022 (PMID 34520068)",
    "B0_mT": 50,
    "T1": {"GM": (327, 10), "WM": (275, 5), "CSF": (3695, 287),
            "Muscle": (171, 11), "Fat": (130, 5)},
    "T2": {"GM": (102, 6), "WM": (102, 6), "CSF": (1584, 124),
            "Muscle": (39, 2), "Fat": (90, 13)},
}


print("=" * 76)
print("  INDEPENDENT VERIFICATION OF T1/T2 CLAIMS")
print("=" * 76)

n_pass = 0
n_fail = 0
n_warn = 0

def check(claim_id, description, claimed, actual_val, actual_err, source):
    global n_pass, n_fail, n_warn
    match = abs(claimed - actual_val) <= actual_err
    close = abs(claimed - actual_val) <= 2 * actual_err
    if match:
        status = "PASS"
        n_pass += 1
    elif close:
        status = "WARN (within 2sigma)"
        n_warn += 1
    else:
        status = "FAIL"
        n_fail += 1
    print(f"  [{status}] {claim_id}: {description}")
    print(f"         Claimed: {claimed} ms")
    print(f"         Actual:  {actual_val} +/- {actual_err} ms")
    print(f"         Source:  {source}")
    if not match:
        print(f"         Delta:   {claimed - actual_val:+.1f} ms "
              f"({abs(claimed - actual_val)/actual_err:.1f} sigma)")
    return match


# =====================================================================
# CLAIM 1: T1 at 46 uT for Grey Matter = 103 +/- 5 ms (source: "Vesanen 2013")
# =====================================================================
print("\n" + "-" * 76)
print("CLAIM 1: T1 at 46 uT for Grey Matter = 103 +/- 5 ms")
print("  Attributed to: 'Vesanen 2013'")
print("-" * 76)

print("\n  ATTRIBUTION CHECK:")
print("  The Vesanen 2013 paper (MRM 69:1795) describes a HYBRID MEG-MRI SYSTEM.")
print("  It does NOT report T1 or T2 relaxation times for brain tissues.")
print("  The abstract says: 'hybrid MEG-MRI instrumentation based on a commercial")
print("  whole-head MEG device is described' -- no relaxometry data.")
print()
print("  The ACTUAL source of T1 = 103 +/- 5 ms for GM at 46 uT is:")
print("  Zotev et al. 2009 (arXiv:0811.0389 / IEEE TAS)")
print("  Page 3: 'We obtained the following T1 values at Br=46 uT:")
print("  103+/-5 ms for gray matter'")
print()
print("  >>> ATTRIBUTION IS WRONG. Value is correct but source is Zotev 2009, NOT Vesanen 2013.")

actual_T1_GM = ZOTEV_2009["T1"]["GM"]
check("1", "T1(GM) at 46 uT", 103, actual_T1_GM[0], actual_T1_GM[1],
      ZOTEV_2009["paper"])


# =====================================================================
# CLAIM 2: T1 at 46 uT for White Matter = 75 +/- 2 ms (source: "Vesanen 2013")
# =====================================================================
print("\n" + "-" * 76)
print("CLAIM 2: T1 at 46 uT for White Matter = 75 +/- 2 ms")
print("  Attributed to: 'Vesanen 2013'")
print("-" * 76)

print("\n  Same attribution issue as Claim 1.")
print("  Actual source: Zotev 2009, page 3: '75+/-2 ms for white matter'")
print("  >>> ATTRIBUTION IS WRONG. Value is correct, source is Zotev 2009.")

actual_T1_WM = ZOTEV_2009["T1"]["WM"]
check("2", "T1(WM) at 46 uT", 75, actual_T1_WM[0], actual_T1_WM[1],
      ZOTEV_2009["paper"])


# =====================================================================
# CLAIM 3: T2 at 46 uT for Grey Matter = 106 +/- 11 ms (source: "Zotev 2009")
# =====================================================================
print("\n" + "-" * 76)
print("CLAIM 3: T2 at 46 uT for Grey Matter = 106 +/- 11 ms")
print("  Attributed to: 'Zotev et al., IEEE TAS 2009'")
print("-" * 76)

print("\n  ATTRIBUTION CHECK: CORRECT")
print("  Zotev 2009, page 3: '106+/-11 ms for gray matter' (T2)")
print("  Journal: This is from IEEE Trans. Appl. Supercond. (IEEE TAS)")
print("  Also available as arXiv:0811.0389")

actual_T2_GM = ZOTEV_2009["T2"]["GM"]
check("3", "T2(GM) at 46 uT", 106, actual_T2_GM[0], actual_T2_GM[1],
      ZOTEV_2009["paper"])


# =====================================================================
# CLAIM 4: T2 at 46 uT for CSF = 355 +/- 15 ms (source: "Zotev 2008/2009")
# =====================================================================
print("\n" + "-" * 76)
print("CLAIM 4: T2 at 46 uT for CSF = 355 +/- 15 ms")
print("  Attributed to: 'Zotev 2008/2009'")
print("-" * 76)

print("\n  ATTRIBUTION CHECK: CORRECT")
print("  Zotev 2009, page 3: '355+/-15 ms for CSF' (T2)")
print("  Note: The T2 data was also published in Zotev et al. 2008")
print("  (J. Magn. Reson. 194:115-120), referenced as [13] in the 2009 paper.")

actual_T2_CSF = ZOTEV_2009["T2"]["CSF"]
check("4", "T2(CSF) at 46 uT", 355, actual_T2_CSF[0], actual_T2_CSF[1],
      ZOTEV_2009["paper"])


# =====================================================================
# CLAIM 5: T1 at 50 mT from O'Reilly et al. 2022
# =====================================================================
print("\n" + "-" * 76)
print("CLAIM 5: T1 values at 50 mT from O'Reilly et al. 2022")
print("-" * 76)

claims_T1_oreilly = {
    "GM":     327,
    "WM":     275,
    "Muscle": 171,
    "Fat":    130,
}

for tissue, claimed in claims_T1_oreilly.items():
    actual = OREILLY_2022["T1"][tissue]
    check(f"5-{tissue}", f"T1({tissue}) at 50 mT", claimed, actual[0], actual[1],
          OREILLY_2022["paper"])


# =====================================================================
# CLAIM 6: T2 at 50 mT from O'Reilly et al. 2022
# =====================================================================
print("\n" + "-" * 76)
print("CLAIM 6: T2 values at 50 mT from O'Reilly et al. 2022")
print("-" * 76)

claims_T2_oreilly = {
    "GM":     102,
    "WM":     102,
    "Muscle": 39,
    "Fat":    90,
}

for tissue, claimed in claims_T2_oreilly.items():
    actual = OREILLY_2022["T2"][tissue]
    check(f"6-{tissue}", f"T2({tissue}) at 50 mT", claimed, actual[0], actual[1],
          OREILLY_2022["paper"])

print("\n  NOTE on Claim 6 - GM and WM T2 being identical:")
print("  Yes, O'Reilly 2022 actually reports T2 = 102 +/- 6 ms for BOTH")
print("  grey matter and white matter at 50 mT. This is correct from the paper.")
print("  This convergence of T2 at low field is physically expected (T2 has weak")
print("  field dependence; at low field T1->T2 and tissues converge).")


# =====================================================================
# CLAIM 7: Field-dependent model parameters and verification
# =====================================================================
print("\n" + "-" * 76)
print("CLAIM 7: Field-dependent T1 model verification")
print("  Model: T1(B0) = T1_low + (T1_3T - T1_low) * (B0/3)^alpha")
print("-" * 76)

# First, note: the model in tissue.py uses a DIFFERENT form:
# T1(B0) = T1_min + (T1_3T - T1_min) * (B0/3)^alpha
# where T1_min = max(1.5*T2, 30)
# This is NOT the same as the claimed model with T1_low values.
# Let me verify BOTH.

# Claimed parameters
claimed_params = {
    #          T1_low, alpha, T1@50uT (claimed)
    "GM":     (85,     0.39,  102),
    "WM":     (56,     0.33,  77),
    "CSF":    (2000,   0.04,  3288),
    "Fat":    (94,     0.51,  95),
    "Muscle": (55,     0.60,  57),
}

# T1 at 3T from tissue.py
T1_3T = {
    "GM":     1330,
    "WM":     830,
    "CSF":    4000,
    "Fat":    380,
    "Muscle": 1400,
}

# T2 values at 3T from tissue.py
T2_3T = {
    "GM":     80,
    "WM":     70,
    "CSF":    2000,
    "Fat":    60,
    "Muscle": 40,
}

# Actual tissue.py model parameters (T1_min = max(1.5*T2, 30))
tissue_py_alpha = {
    "GM":     0.35,
    "WM":     0.35,
    "CSF":    0.25,
    "Fat":    0.30,
    "Muscle": 0.35,
}

print("\n  Part A: Verify the CLAIMED model T1(B0) = T1_low + (T1_3T-T1_low)*(B0/3)^alpha")
print("  =================================================================================")

print(f"\n  {'Tissue':<8} {'T1_low':>6} {'alpha':>6} {'T1_3T':>6} "
      f"{'T1@50uT':>8} {'T1@50uT':>8} {'T1@50mT':>8} {'T1@50mT':>8}")
print(f"  {'':8} {'':>6} {'':>6} {'':>6} "
      f"{'(claim)':>8} {'(calc)':>8} {'(calc)':>8} {'(O`R22)':>8}")
print("  " + "-" * 70)

for tissue in ["GM", "WM", "CSF", "Fat", "Muscle"]:
    T1_low, alpha, claimed_50uT = claimed_params[tissue]
    T1_3 = T1_3T[tissue]

    # Calculate T1 at 50 uT using claimed model
    calc_50uT = T1_low + (T1_3 - T1_low) * (50e-6 / 3.0) ** alpha

    # Calculate T1 at 50 mT
    calc_50mT = T1_low + (T1_3 - T1_low) * (50e-3 / 3.0) ** alpha

    # Calculate T1 at 3T (should recover T1_3T exactly)
    calc_3T = T1_low + (T1_3 - T1_low) * (3.0 / 3.0) ** alpha

    oreilly = OREILLY_2022["T1"].get(tissue, (None, None))
    oreilly_str = f"{oreilly[0]}" if oreilly[0] else "N/A"

    print(f"  {tissue:<8} {T1_low:>6} {alpha:>6.2f} {T1_3:>6} "
          f"{claimed_50uT:>8} {calc_50uT:>8.1f} {calc_50mT:>8.1f} {oreilly_str:>8}")

print("\n  Verification checks:")
for tissue in ["GM", "WM", "CSF", "Fat", "Muscle"]:
    T1_low, alpha, claimed_50uT = claimed_params[tissue]
    T1_3 = T1_3T[tissue]

    calc_50uT = T1_low + (T1_3 - T1_low) * (50e-6 / 3.0) ** alpha
    calc_3T = T1_low + (T1_3 - T1_low) * (3.0 / 3.0) ** alpha
    calc_50mT = T1_low + (T1_3 - T1_low) * (50e-3 / 3.0) ** alpha

    # Check 1: Recovers T1_3T at 3T?
    err_3T = abs(calc_3T - T1_3)
    ok_3T = err_3T < 0.1

    # Check 2: Matches claimed T1@50uT?
    err_50uT = abs(calc_50uT - claimed_50uT)
    ok_50uT = err_50uT < 1.0

    # Check 3: Matches Zotev at 46 uT for GM/WM?
    calc_46uT = T1_low + (T1_3 - T1_low) * (46e-6 / 3.0) ** alpha
    zotev = ZOTEV_2009["T1"].get(tissue, (None, None))

    # Check 4: Matches O'Reilly at 50 mT?
    oreilly = OREILLY_2022["T1"].get(tissue, (None, None))

    # Check 5: T1 > T2 at all fields down to 50 uT?
    T2 = T2_3T[tissue]
    ok_T1_gt_T2 = calc_50uT > T2

    print(f"\n  {tissue}:")
    print(f"    T1(3T) recovery: calc={calc_3T:.1f}, expected={T1_3} "
          f"[{'OK' if ok_3T else 'FAIL'}]")
    print(f"    T1(50uT) claim: calc={calc_50uT:.1f}, claimed={claimed_50uT} "
          f"[{'OK' if ok_50uT else 'FAIL, delta='+str(err_50uT)+'ms'}]")

    if zotev[0] is not None:
        err_z = abs(calc_46uT - zotev[0])
        print(f"    T1(46uT) vs Zotev: calc={calc_46uT:.1f}, measured={zotev[0]}+/-{zotev[1]} "
              f"[{'OK' if err_z <= zotev[1] else 'MISMATCH, delta='+f'{err_z:.1f}ms'}]")

    if oreilly[0] is not None:
        err_o = abs(calc_50mT - oreilly[0])
        print(f"    T1(50mT) vs O'Reilly: calc={calc_50mT:.1f}, measured={oreilly[0]}+/-{oreilly[1]} "
              f"[{'OK' if err_o <= oreilly[1] else 'MISMATCH, delta='+f'{err_o:.1f}ms'}]")

    print(f"    T1(50uT)={calc_50uT:.1f} > T2={T2} at 50uT? "
          f"[{'OK' if ok_T1_gt_T2 else 'FAIL: T1 < T2!'}]")


print("\n\n  Part B: Verify the ACTUAL tissue.py model")
print("  T1(B0) = T1_min + (T1_3T - T1_min) * (B0/3)^alpha")
print("  where T1_min = max(1.5*T2, 30)")
print("  =================================================================================")

print(f"\n  {'Tissue':<8} {'T1_min':>6} {'alpha':>6} {'T1_3T':>6} "
      f"{'T1@50uT':>8} {'T1@46uT':>8} {'Zotev':>8} {'T1@50mT':>8} {'O`R22':>8} "
      f"{'T1>T2?':>6}")
print("  " + "-" * 80)

for tissue in ["GM", "WM", "CSF", "Fat", "Muscle"]:
    T1_3 = T1_3T[tissue]
    T2 = T2_3T[tissue]
    alpha = tissue_py_alpha[tissue]
    T1_min = max(1.5 * T2, 30.0)

    calc_50uT = T1_min + (T1_3 - T1_min) * (50e-6 / 3.0) ** alpha
    calc_46uT = T1_min + (T1_3 - T1_min) * (46e-6 / 3.0) ** alpha
    calc_50mT = T1_min + (T1_3 - T1_min) * (50e-3 / 3.0) ** alpha

    zotev_T1 = ZOTEV_2009["T1"].get(tissue, (None, None))
    zotev_str = f"{zotev_T1[0]}+/-{zotev_T1[1]}" if zotev_T1[0] else "N/A"

    oreilly_T1 = OREILLY_2022["T1"].get(tissue, (None, None))
    oreilly_str = f"{oreilly_T1[0]}+/-{oreilly_T1[1]}" if oreilly_T1[0] else "N/A"

    t1_gt_t2 = calc_50uT > T2

    print(f"  {tissue:<8} {T1_min:>6.0f} {alpha:>6.2f} {T1_3:>6} "
          f"{calc_50uT:>8.1f} {calc_46uT:>8.1f} {zotev_str:>8} {calc_50mT:>8.1f} {oreilly_str:>8} "
          f"{'YES' if t1_gt_t2 else 'NO!':>6}")


print("\n\n  Part C: Detailed comparison - tissue.py model vs measurements")
print("  =================================================================================")

for tissue in ["GM", "WM", "CSF", "Fat", "Muscle"]:
    T1_3 = T1_3T[tissue]
    T2 = T2_3T[tissue]
    alpha = tissue_py_alpha[tissue]
    T1_min = max(1.5 * T2, 30.0)

    calc_46uT = T1_min + (T1_3 - T1_min) * (46e-6 / 3.0) ** alpha
    calc_50mT = T1_min + (T1_3 - T1_min) * (50e-3 / 3.0) ** alpha

    print(f"\n  {tissue}:")

    zotev_T1 = ZOTEV_2009["T1"].get(tissue, None)
    if zotev_T1 and zotev_T1[0]:
        delta = calc_46uT - zotev_T1[0]
        pct = 100 * delta / zotev_T1[0]
        within = abs(delta) <= zotev_T1[1]
        print(f"    vs Zotev@46uT:   model={calc_46uT:.1f}, measured={zotev_T1[0]}+/-{zotev_T1[1]}, "
              f"delta={delta:+.1f}ms ({pct:+.1f}%) [{'WITHIN ERROR' if within else 'OUTSIDE ERROR'}]")

    oreilly_T1 = OREILLY_2022["T1"].get(tissue, None)
    if oreilly_T1 and oreilly_T1[0]:
        delta = calc_50mT - oreilly_T1[0]
        pct = 100 * delta / oreilly_T1[0]
        within = abs(delta) <= oreilly_T1[1]
        print(f"    vs O'Reilly@50mT: model={calc_50mT:.1f}, measured={oreilly_T1[0]}+/-{oreilly_T1[1]}, "
              f"delta={delta:+.1f}ms ({pct:+.1f}%) [{'WITHIN ERROR' if within else 'OUTSIDE ERROR'}]")


# =====================================================================
# EXTRA: Check the Zotev paper's own cross-validation
# =====================================================================
print("\n\n" + "=" * 76)
print("  ADDITIONAL VERIFICATION: Zotev's own cross-check")
print("=" * 76)
print("\n  Zotev 2009 page 4 states:")
print("  'we estimated T1 for gray and white matter using [the Fischer 1990")
print("  empirical formula]. At f=2 kHz, it provides T1 estimates of 103 ms")
print("  and 76 ms for gray and white matter, respectively.'")
print()
print("  Their measured values: GM=103+/-5, WM=75+/-2")
print("  Fischer formula pred:  GM=103,     WM=76")
print("  Agreement: excellent (within 1 ms)")
print()
print("  NOTE: The Zotev paper explicitly says T1 ~ T2 at 46 uT (within 5%):")
print("    GM:  T1=103, T2=106  (T2/T1 = 1.03)")
print("    WM:  T1=75,  T2=79   (T2/T1 = 1.05)")
print("    CSF: T1=344, T2=355  (T2/T1 = 1.03)")
print("  The paper notes 'T1 values are slightly shorter than T2' but")
print("  says 'this discrepancy is not significant' given experimental errors.")
print()
print("  IMPORTANT: In the Zotev paper, T1 < T2 for GM, WM, and CSF!")
print("  This means a model that enforces T1 >= T2 may need T1_min ~ T2,")
print("  not T1_min = 1.5*T2 as in tissue.py.")


# =====================================================================
# EXTRA: Check CSF T1 at 50 mT - O'Reilly vs model
# =====================================================================
print("\n\n" + "=" * 76)
print("  ADDITIONAL: CSF parameters check")
print("=" * 76)

T1_3_CSF = 4000
T2_CSF = 2000
alpha_CSF_codebase = 0.25
T1_min_CSF = max(1.5 * T2_CSF, 30)  # = 3000!

calc_CSF_50mT = T1_min_CSF + (T1_3_CSF - T1_min_CSF) * (50e-3 / 3.0) ** alpha_CSF_codebase
calc_CSF_46uT = T1_min_CSF + (T1_3_CSF - T1_min_CSF) * (46e-6 / 3.0) ** alpha_CSF_codebase

print(f"\n  tissue.py: CSF T1_3T={T1_3_CSF}, T2={T2_CSF}, alpha={alpha_CSF_codebase}")
print(f"  T1_min = max(1.5*{T2_CSF}, 30) = {T1_min_CSF}")
print(f"  T1(46uT) = {calc_CSF_46uT:.0f} ms  (Zotev measured: 344+/-9)")
print(f"  T1(50mT) = {calc_CSF_50mT:.0f} ms  (O'Reilly measured: 3695+/-287)")
print()

if abs(calc_CSF_46uT - 344) > 100:
    print(f"  >>> PROBLEM: T1_min=3000 forces T1(46uT)={calc_CSF_46uT:.0f}, "
          f"but Zotev measured 344!")
    print(f"      The floor of 1.5*T2 = 3000 ms is WAY too high for CSF.")
    print(f"      Zotev found T1(CSF)=344 ms at 46 uT -- not 3000!")
    n_fail += 1
else:
    print(f"  CSF T1 at 46 uT looks OK")
    n_pass += 1


# =====================================================================
# Verify claim 7 T1_low model predictions match the table exactly
# =====================================================================
print("\n\n" + "=" * 76)
print("  CLAIM 7 TABLE VERIFICATION")
print("=" * 76)

print("\n  Checking: T1(B0) = T1_low + (T1_3T - T1_low) * (B0/3)^alpha")
print("  at B0 = 50 uT:\n")

for tissue in ["GM", "WM", "CSF", "Fat", "Muscle"]:
    T1_low, alpha, claimed = claimed_params[tissue]
    T1_3 = T1_3T[tissue]
    calc = T1_low + (T1_3 - T1_low) * (50e-6 / 3.0) ** alpha
    delta = abs(calc - claimed)
    print(f"  {tissue:<8}: T1_low={T1_low}, alpha={alpha:.2f}, T1_3T={T1_3}")
    print(f"           T1(50uT) = {calc:.2f} ms, claimed = {claimed} ms, "
          f"delta = {delta:.2f} [{'OK' if delta < 1 else 'MISMATCH'}]")


# =====================================================================
# SUMMARY
# =====================================================================
print("\n\n" + "=" * 76)
print("  SUMMARY OF FINDINGS")
print("=" * 76)

print(f"""
  1. CLAIMS 1-2 (T1 at 46 uT for GM and WM):
     VALUES ARE CORRECT (match Zotev 2009 exactly).
     ATTRIBUTION IS WRONG: Source is Zotev et al. 2009 (IEEE TAS / arXiv:0811.0389),
     NOT Vesanen et al. 2013 (MRM 69:1795). Vesanen 2013 describes a MEG-MRI
     system and does NOT report T1/T2 relaxation times.

  2. CLAIMS 3-4 (T2 at 46 uT for GM and CSF):
     VALUES AND ATTRIBUTION ARE CORRECT (Zotev 2009).

  3. CLAIMS 5-6 (T1 and T2 at 50 mT from O'Reilly 2022):
     ALL VALUES ARE CORRECT (verified against PubMed abstract, PMID 34520068).
     Yes, GM and WM T2 are both 102+/-6 ms -- this is what the paper reports.

  4. CLAIM 7 (T1_low / alpha model parameters):
     - The claimed model uses T1_low (tissue-specific floor), NOT the tissue.py
       model which uses T1_min = max(1.5*T2, 30).
     - The claimed T1(50uT) values are arithmetically correct given the parameters.
     - BUT: the tissue.py codebase uses DIFFERENT alpha values and a DIFFERENT
       floor model. The Claim 7 parameters are NOT what's implemented in tissue.py.

  5. CRITICAL ISSUE - CSF in tissue.py:
     tissue.py uses T1_min = max(1.5*T2, 30) = max(3000, 30) = 3000 ms for CSF.
     This means T1(CSF) can NEVER go below 3000 ms in the model.
     But Zotev 2009 measured T1(CSF) = 344 ms at 46 uT!
     This is a factor of ~9x discrepancy. The 1.5*T2 floor is inappropriate
     for CSF because CSF has extremely long T2 (~2000 ms at 3T).

  6. CRITICAL ISSUE - Zotev shows T1 < T2 at 46 uT:
     For GM:  T1=103, T2=106 (ratio 0.97)
     For WM:  T1=75,  T2=79  (ratio 0.95)
     For CSF: T1=344, T2=355 (ratio 0.97)
     The tissue.py floor T1_min = 1.5*T2 enforces T1 >= 1.5*T2, which is
     physically wrong at ultra-low field where T1 ~ T2 (and can be slightly less).

  Verification counts: {n_pass} passed, {n_warn} warnings, {n_fail} failed
""")
