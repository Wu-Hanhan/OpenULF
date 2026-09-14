# -*- coding: utf-8 -*-
"""
dataset_accelerated.py — Accelerated MRI Reconstruction Dataset
================================================================

Simulates accelerated (undersampled) MRI acquisition by masking
k-space lines, then trains models to reconstruct full images.

This tests whether SQUID's SNR advantage persists when combined
with scan acceleration (fewer k-space acquisitions).

Pipeline:
  Full noisy k-space → apply undersampling mask → zero-fill → IFFT
  → aliased image (input) → U-Net → clean image (target)

Acceleration factors: 2x, 4x, 8x
Mask types: random uniform (standard in fastMRI literature)
"""
from __future__ import annotations
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset


def create_undersampling_mask(n_lines: int, acceleration: int,
                               center_fraction: float = 0.08,
                               seed: int = None) -> np.ndarray:
    """Create 1D random undersampling mask for k-space.

    Keeps a fully-sampled center region + random lines elsewhere.
    Standard approach from fastMRI (Zbontar et al. 2018).

    Parameters
    ----------
    n_lines : int
        Number of phase-encode lines (k-space rows).
    acceleration : int
        Acceleration factor (2, 4, 8). Keeps ~1/R of total lines.
    center_fraction : float
        Fraction of center k-space to keep (always sampled).
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    mask : np.ndarray, shape (n_lines,), bool
        True for sampled lines, False for skipped.
    """
    rng = np.random.default_rng(seed)
    mask = np.zeros(n_lines, dtype=bool)

    # Always keep center lines (low-frequency content)
    n_center = max(1, int(n_lines * center_fraction))
    center_start = (n_lines - n_center) // 2
    mask[center_start:center_start + n_center] = True

    # Randomly sample remaining lines to reach target sampling rate
    n_target = max(n_center, n_lines // acceleration)
    remaining = np.where(~mask)[0]
    n_extra = n_target - n_center
    if n_extra > 0 and len(remaining) > 0:
        chosen = rng.choice(remaining, min(n_extra, len(remaining)), replace=False)
        mask[chosen] = True

    return mask


class AcceleratedMRIDataset(Dataset):
    """Accelerated MRI reconstruction dataset.

    Applies undersampling to stored k-space, then reconstructs
    zero-filled input images for the model to denoise + reconstruct.

    Parameters
    ----------
    h5_path : str
        Path to HDF5 with k-space data.
    input_type : str
        'squid' or 'coil'.
    acceleration : int
        Acceleration factor (2, 4, 8).
    """

    def __init__(self, h5_path: str, input_type: str = 'squid',
                 acceleration: int = 4, augment: bool = False):
        self.input_type = input_type
        self.acceleration = acceleration
        self.augment = augment

        with h5py.File(h5_path, 'r') as f:
            self.ideal = f['images/ideal'][:].astype(np.float32)
            if 'kspace' not in f:
                raise ValueError(
                    f"AcceleratedMRIDataset requires k-space data in HDF5, "
                    f"but '{h5_path}' has no 'kspace' group. "
                    f"Datasets generated with multi-channel RSS (v3+) do not "
                    f"store per-channel k-space. Use generate_dataset.py with "
                    f"--n_channels 1 to produce single-channel k-space data, "
                    f"or use ULFMRIDataset for image-domain reconstruction.")
            if input_type == 'both':
                # Load both SQUID and Coil k-space for 2-channel reconstruction
                self.kspace_squid = f['kspace/squid'][:].astype(np.complex64)
                self.kspace_coil = f['kspace/coil'][:].astype(np.complex64)
            else:
                # Load single-channel k-space
                self.kspace = f[f'kspace/{input_type}'][:].astype(np.complex64)
                # Load normalization factors
                ksp_key = f'{input_type}_max'
                if f'norm_factors/{ksp_key}' in f:
                    self.norm = f[f'norm_factors/{ksp_key}'][:].astype(np.float32)
                else:
                    self.norm = np.ones(len(self.ideal), dtype=np.float32)

    def __len__(self):
        return len(self.ideal)

    def _undersample_reconstruct(self, ksp, idx):
        """Undersample k-space and reconstruct zero-filled magnitude image."""
        n_lines = ksp.shape[0]
        mask = create_undersampling_mask(
            n_lines, self.acceleration,
            seed=idx * 1000 + self.acceleration)
        ksp_masked = ksp.copy()
        ksp_masked[~mask, :] = 0
        img = np.abs(np.fft.ifftshift(
            np.fft.ifft2(np.fft.fftshift(ksp_masked))))
        p995 = np.percentile(img, 99.5)
        if p995 < 1e-30:
            p995 = img.max()
        return np.clip(img / (p995 + 1e-30), 0, 1).astype(np.float32)

    def __getitem__(self, idx):
        target = torch.from_numpy(self.ideal[idx]).unsqueeze(0)  # (1, H, W)

        if self.input_type == 'both':
            # 2-channel: undersample both SQUID and Coil with same mask
            img_s = self._undersample_reconstruct(self.kspace_squid[idx], idx)
            img_c = self._undersample_reconstruct(self.kspace_coil[idx], idx)
            inp = torch.from_numpy(np.stack([img_s, img_c], axis=0))  # (2, H, W)
        else:
            img = self._undersample_reconstruct(self.kspace[idx], idx)
            inp = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)

        # Data augmentation (same for input and target)
        if self.augment:
            if torch.rand(1).item() > 0.5:
                inp = torch.flip(inp, [-1])
                target = torch.flip(target, [-1])
            if torch.rand(1).item() > 0.5:
                inp = torch.flip(inp, [-2])
                target = torch.flip(target, [-2])
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                inp = torch.rot90(inp, k, [-2, -1])
                target = torch.rot90(target, k, [-2, -1])

        return inp, target


if __name__ == "__main__":
    for acc in [2, 4, 8]:
        ds = AcceleratedMRIDataset('data/dataset/test.h5', 'squid', acc)
        x, y = ds[0]
        mask = create_undersampling_mask(128, acc, seed=0)
        print(f"  R={acc}x: mask keeps {mask.sum()}/{len(mask)} lines "
              f"({mask.sum()/len(mask)*100:.0f}%), "
              f"input={x.shape}, target={y.shape}")
    print("  OK ✓")
