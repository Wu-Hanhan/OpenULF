# Data & Pretrained Models

OpenULF does **not** redistribute any anatomical dataset. The public datasets
below are governed by their own licenses / data-use agreements, so you must
obtain them from their official sources and agree to their terms. OpenULF then
turns each anatomy into paired (ideal, SQUID, coil) ULF-MRI images with the
generation scripts in this repository.

All datasets use **subject-level** train/val/test splits (slices from one
subject never appear in more than one split), so there is no same-source
leakage between training and evaluation.

## Datasets

### BrainWeb  (`generate_dataset.py`)
- Source: BrainWeb Simulated Brain Database, McGill University
  (https://brainweb.bic.mni.mcgill.ca/).
- We use the **fuzzy** (partial-volume) anatomical models.
- Place the volumes under `data/brainweb/v2_fuzzy/` (files
  `brainweb_sNN_fuzzy.nii[.gz]`), then run `python generate_dataset.py`.
- Split: fixed subject IDs (14 train / 2 val / 4 test), see the top of
  `generate_dataset.py`.

### OASIS-1  (`generate_dataset_oasis.py`)
- Source: OASIS-1 cross-sectional MRI (https://www.oasis-brains.org/).
- We use the template-aligned 4-class segmentations (`aligned_seg4.nii.gz`).
- Place subject folders under `data/OASIS/OASIS_OAS1_*/`, then run the script.
- Split: 70 / 10 / 20 % by subject.

### IXI  (`process_ixi.py` → `generate_dataset_ixi.py`)
- Source: IXI dataset (https://brain-development.org/ixi-dataset/).
- `process_ixi.py` segments the T1 volumes into `data/IXI/segmentations/`;
  `generate_dataset_ixi.py` then builds the ULF dataset.
- Split: 70 / 10 / 20 % by subject.

### Synthetic abdomen  (`generate_dataset_abdomen.py`)
- Fully synthetic phantoms — **no external data required**. Just run the script.

## Regenerating a dataset

```bash
# after placing the source anatomy as described above
python generate_dataset.py                 # BrainWeb (default config)
python generate_dataset.py --b_prepol 0.3  # e.g. higher prepolarization
python generate_dataset_oasis.py
python generate_dataset_ixi.py
python generate_dataset_abdomen.py
```

Each run writes `train.h5`, `val.h5`, `test.h5` under the dataset's output
directory (see the `OUT_DIR` constant near the top of each script).

## Pretrained models (774 trained networks)

The trained model checkpoints reported in the paper will be released on
**Zenodo** (with a citable DOI) **upon publication**. This repository ships the
code needed to reproduce them from scratch via `train.py`.
