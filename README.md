# OpenULF

**A modular, open-source co-simulation toolkit for machine-learning image
reconstruction in ultra-low-field (ULF) MRI, with both quantum (SQUID) and
classical (induction-coil) sensors.**

OpenULF couples a Bloch-validated physics simulator, calibrated SQUID and coil
receiver models, and end-to-end ML reconstruction behind a **receiver
abstraction** and a **typed HDF5 contract**, so sensors, models, and datasets
are independently swappable. A make-like sweep engine explores the
B₀ × NEX × channels × model × sensor design space at scale.

> Status: research code accompanying our paper. Pretrained weights and the full 
> sweep outputs are released separately
> (see [DATA.md](DATA.md)).

## Why

At ULF (B₀ < 10 mT), MRI is portable and low-cost but signal-starved. Two
receivers compete: a **quantum SQUID**, sensitive independent of frequency but
requiring cryogenics, and a **classical induction coil**, room-temperature but
with signal ∝ ω₀ = γB₀ and thus very weak at ULF. Under conventional
reconstruction both give low SNR. OpenULF asks — and answers end-to-end — when
machine learning closes the gap, and which sensor is actually worth it.

## Key findings (from the paper)

Training 774 models across four anatomical datasets (BrainWeb, OASIS, IXI, a
synthetic abdomen set) yields three design rules:

1. **The ML–sensor interaction is dataset-dependent.** The same architecture
   compresses the SQUID–coil gap on one anatomy and amplifies it on another;
   raw input SNR does not predict the outcome, so sensors must be judged
   end-to-end.
2. **A SQUID-to-coil crossover near ~450 µT**, fixed by sensor physics (Faraday
   scaling), not ML capacity. Below it the SQUID wins; above it a
   room-temperature coil suffices.
3. **Sensor channels and ML capacity are substitutable** design variables — a
   smaller array with a larger model matches a larger array with a smaller
   model, and model capacity also buys scan time.

## Architecture

```
tissue.py ─► forward.py (MRISignalEngine) ─► ideal M⊥ image + per-tissue k-space
                                              │
receiver.py (ReceiverBase.acquire)  ◄─────────┘
   ├─ classical_coil_receiver.py   (Faraday EMF, ω₀-dependent gain)
   └─ quantum_squid_receiver.py    (flux-locked loop, frequency-independent)
                    │  reciprocity flux Ψ → voltage → H(f) + colored noise
                    ▼
              typed HDF5 (train/val/test .h5)   ← dataset.py
                    ▼
            model.py (SimpleCNN, DnCNN, UNet, ResUNet, NAFNet) ─ train.py ─ evaluate_dataset.py
```

Module dependency: `constants.py → tissue.py → forward.py → receiver.py →
{classical_coil_receiver.py, quantum_squid_receiver.py}`.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # or conda
pip install -r requirements.txt
```

Tested on Python 3.11 with PyTorch 2.6 (CUDA 12.4). A GPU is recommended for
training; the physics simulator and validation tests run on CPU.

## Quick start

```bash
# 1. Obtain a dataset (see DATA.md) and generate paired ULF images:
python generate_dataset.py                 # BrainWeb (default: B0=50 uT, NEX=10)

# 2. Train a model (SQUID / coil / both; unet / dncnn / simplecnn / resunet):
python train.py --experiment squid --model unet --epochs 200 --patience 20

# 3. Evaluate on the held-out test set (ML + traditional baselines):
python evaluate_dataset.py --data_dir data/dataset \
                           --results_dir results --out_dir analysis/brainweb
```

Training reads only `train.h5`/`val.h5` and selects the best checkpoint by
validation loss; the test set is used only in `evaluate_dataset.py`.

## Validation

The physics is checked against closed-form solutions and published anchors:

```bash
python cross_validation/validate_forward.py   # analytical Bloch checks
python cross_validation/validate_kspace.py
python tools/numerical_audit.py               # signal-chain PASS/FAIL (>1% = FAIL)
python tools/verify_tissue_claims.py
python forward.py                             # module self-tests
python quantum_squid_receiver.py
python classical_coil_receiver.py
```

## Repository layout

| File / dir | Purpose |
|---|---|
| `constants.py`, `tissue.py`, `ulf_params.py` | physical constants, tissue relaxation, sequence params |
| `forward.py` | forward MRI signal engine (prepolarization, T2\*, k-space) |
| `receiver.py` | receiver abstract base class + shared acquisition core |
| `classical_coil_receiver.py`, `quantum_squid_receiver.py` | the two sensor models |
| `model.py` | reconstruction networks (SimpleCNN, DnCNN, UNet, ResUNet, NAFNet) |
| `dataset.py`, `dataset_accelerated.py`, `dataset_mixed.py` | PyTorch datasets (typed HDF5) |
| `train.py` | training loop (early stopping, AMP, augmentation) |
| `generate_dataset*.py`, `process_ixi.py` | dataset generators (see DATA.md) |
| `evaluate_dataset.py`, `eval_sigpy_baseline.py` | evaluation + baselines |
| `cross_validation/`, `tools/` | analytical validation and numerical audits |

## Data & pretrained models

See [DATA.md](DATA.md). Datasets are obtained from their official sources
(subject-level splits, no leakage); the 774 trained checkpoints will be
released on Zenodo (DOI) upon publication.

## Citation

If you use OpenULF, please cite:

```bibtex
@misc{wu2026openulf,
  title     = {Quantum or Classical? Co-Design of Sensor and Reconstruction
               for Ultra-Low-Field MRI},
  author    = {Wu, Hanhan and Yuan, Yuchen and Jiang, Weiwen},
  year      = {2026},
}
```

## License

Source code: [MIT](LICENSE). Third-party datasets and any separately released
pretrained weights are governed by their own terms — see [DATA.md](DATA.md).
