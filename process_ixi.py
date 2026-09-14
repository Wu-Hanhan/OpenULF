# -*- coding: utf-8 -*-
"""
process_ixi.py — IXI Dataset Processing Pipeline
===================================================

Downloads (if needed), segments, and converts IXI T1 brain images
into tissue maps compatible with our MRI forward model.

Pipeline:
  IXI T1 NIfTI → ANTs brain extraction → 3-class segmentation (CSF/GM/WM)
  → tissue map (BrainWeb labels 0-3) → forward model → dataset

Usage:
    python process_ixi.py                    # Process all
    python process_ixi.py --max_subjects 50  # Process first 50
"""
from __future__ import annotations
import os
import sys
import glob
import time
import argparse
import numpy as np
import nibabel as nib

# Directories
IXI_DIR = "data/IXI"
IXI_TAR = os.path.join(IXI_DIR, "IXI-T1.tar")
IXI_NIFTI_DIR = os.path.join(IXI_DIR, "nifti")
IXI_SEG_DIR = os.path.join(IXI_DIR, "segmentations")


def extract_tar():
    """Extract IXI-T1.tar if not already extracted."""
    if os.path.exists(IXI_NIFTI_DIR) and len(glob.glob(f"{IXI_NIFTI_DIR}/*.nii.gz")) > 0:
        n = len(glob.glob(f"{IXI_NIFTI_DIR}/*.nii.gz"))
        print(f"  Already extracted: {n} files in {IXI_NIFTI_DIR}")
        return
    os.makedirs(IXI_NIFTI_DIR, exist_ok=True)
    print(f"  Extracting {IXI_TAR}...")
    import tarfile
    with tarfile.open(IXI_TAR, 'r') as tar:
        tar.extractall(path=IXI_NIFTI_DIR)
    n = len(glob.glob(f"{IXI_NIFTI_DIR}/*.nii.gz"))
    print(f"  Extracted: {n} files")


def segment_brain_ants(t1_path: str, out_path: str) -> bool:
    """Segment a T1 brain image using ANTsPy.

    Returns 3-class tissue map:
      0 = Background
      1 = CSF
      2 = Grey Matter
      3 = White Matter

    Uses ANTs Atropos (Avants et al. 2011) with Kmeans initialization.
    """
    try:
        import ants
    except ImportError:
        print("ANTsPy not available")
        return False

    # Load image
    img = ants.image_read(t1_path)

    # Brain extraction (skull stripping)
    brain = ants.get_mask(img, low_thresh=img.mean() * 0.5, cleanup=2)
    brain_img = img * brain

    # 3-class segmentation via Atropos
    seg = ants.atropos(a=brain_img, m='[0.2,1x1x1]', c='[5,0]',
                       i='kmeans[3]', x=brain)

    # seg['segmentation'] gives labels 1,2,3 (CSF, GM, WM in intensity order)
    # But Atropos orders by intensity: darkest=1, brightest=3
    # In T1: CSF is dark (label 1), GM is mid (label 2), WM is bright (label 3)
    # This matches our convention: 1=CSF, 2=GM, 3=WM ✓
    seg_data = seg['segmentation'].numpy().astype(np.int8)

    # Apply brain mask (set outside brain to 0 = Background)
    seg_data[brain.numpy() == 0] = 0

    # Save as NIfTI
    seg_nii = nib.Nifti1Image(seg_data, nib.load(t1_path).affine)
    nib.save(seg_nii, out_path)
    return True


def process_all(max_subjects: int = None, skip_existing: bool = True):
    """Process all IXI subjects."""
    os.makedirs(IXI_SEG_DIR, exist_ok=True)

    nifti_files = sorted(glob.glob(f"{IXI_NIFTI_DIR}/IXI*.nii.gz"))
    if max_subjects:
        nifti_files = nifti_files[:max_subjects]

    print(f"\n  Processing {len(nifti_files)} IXI subjects...")
    success = 0
    failed = 0

    for i, t1_path in enumerate(nifti_files):
        basename = os.path.basename(t1_path).replace('.nii.gz', '_seg.nii.gz')
        out_path = os.path.join(IXI_SEG_DIR, basename)

        if skip_existing and os.path.exists(out_path):
            success += 1
            continue

        t0 = time.time()
        try:
            ok = segment_brain_ants(t1_path, out_path)
            elapsed = time.time() - t0
            if ok:
                success += 1
                if (i + 1) % 10 == 0 or i == 0:
                    print(f"    [{i+1}/{len(nifti_files)}] {basename} "
                          f"({elapsed:.1f}s)")
            else:
                failed += 1
        except Exception as e:
            failed += 1
            print(f"    [{i+1}/{len(nifti_files)}] FAILED: {basename}: {e}")

    print(f"\n  Done: {success} success, {failed} failed")
    return success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_subjects', type=int, default=None)
    parser.add_argument('--skip_extract', action='store_true')
    args = parser.parse_args()

    print("=" * 60)
    print("  IXI Dataset Processing")
    print("=" * 60)

    # Step 1: Extract tar
    if not args.skip_extract:
        extract_tar()

    # Step 2: Segment all subjects
    n = process_all(max_subjects=args.max_subjects)
    print(f"\n  {n} segmented volumes ready in {IXI_SEG_DIR}")


if __name__ == "__main__":
    main()
