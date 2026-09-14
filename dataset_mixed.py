# -*- coding: utf-8 -*-
"""
dataset_mixed.py — Mixed Channel Dataset for Sensor Fusion Analysis
=====================================================================

Experiments:
  1. Gradual 2ch: SQUID base + f% Coil as 2nd channel (and vice versa)
  2. Mixed single-channel: randomly assign each sample as SQUID or Coil

These answer: "What is the marginal value of adding a second sensor?"
"""
from __future__ import annotations
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset


class GradualFusionDataset(Dataset):
    """2-channel dataset where only a fraction of samples have the 2nd channel.

    For samples without the 2nd channel, it's filled with zeros.
    This simulates: "we have SQUID for all patients, but Coil for only f%."

    Parameters
    ----------
    h5_path : str
        Path to HDF5 file.
    base : str
        Primary sensor: 'squid' or 'coil'. Always available.
    secondary_fraction : float
        Fraction of samples that also have the secondary sensor (0.0 to 1.0).
    seed : int
        Random seed for selecting which samples get the 2nd channel.
    augment : bool
        Data augmentation.
    """

    def __init__(self, h5_path: str, base: str = 'squid',
                 secondary_fraction: float = 1.0, seed: int = 42,
                 augment: bool = True):
        self.base = base
        self.secondary = 'coil' if base == 'squid' else 'squid'
        self.fraction = secondary_fraction
        self.augment = augment

        with h5py.File(h5_path, 'r') as f:
            if 'images' in f:
                self.ideal = f['images/ideal'][:].astype(np.float32)
                self.squid = f['images/squid'][:].astype(np.float32)
                self.coil = f['images/coil'][:].astype(np.float32)
            else:
                self.ideal = f['ideal'][:].astype(np.float32)
                self.squid = f['squid'][:].astype(np.float32)
                self.coil = f['coil'][:].astype(np.float32)

        # Determine which samples get the 2nd channel
        rng = np.random.default_rng(seed)
        n = len(self.ideal)
        n_secondary = int(n * secondary_fraction)
        self.has_secondary = np.zeros(n, dtype=bool)
        if n_secondary > 0:
            chosen = rng.choice(n, n_secondary, replace=False)
            self.has_secondary[chosen] = True

    def __len__(self):
        return len(self.ideal)

    def __getitem__(self, idx):
        target = torch.from_numpy(self.ideal[idx]).unsqueeze(0)

        # Base channel (always available)
        base_data = self.squid[idx] if self.base == 'squid' else self.coil[idx]

        # Secondary channel (available for fraction of samples)
        if self.has_secondary[idx]:
            sec_data = self.coil[idx] if self.base == 'squid' else self.squid[idx]
        else:
            sec_data = np.zeros_like(base_data)  # zero-fill when unavailable

        inp = torch.from_numpy(np.stack([base_data, sec_data], axis=0))  # (2, H, W)

        # Augmentation
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


class MixedSingleChannelDataset(Dataset):
    """Single-channel dataset mixing SQUID and Coil samples.

    Each sample is randomly assigned as either SQUID or Coil input.
    Tests: can one model denoise both sensor types?

    Parameters
    ----------
    h5_path : str
        Path to HDF5 file.
    squid_fraction : float
        Fraction of samples using SQUID (rest use Coil). Default 0.5.
    seed : int
        Random seed.
    augment : bool
        Data augmentation.
    """

    def __init__(self, h5_path: str, squid_fraction: float = 0.5,
                 seed: int = 42, augment: bool = True):
        self.squid_fraction = squid_fraction
        self.augment = augment

        with h5py.File(h5_path, 'r') as f:
            if 'images' in f:
                self.ideal = f['images/ideal'][:].astype(np.float32)
                self.squid = f['images/squid'][:].astype(np.float32)
                self.coil = f['images/coil'][:].astype(np.float32)
            else:
                self.ideal = f['ideal'][:].astype(np.float32)
                self.squid = f['squid'][:].astype(np.float32)
                self.coil = f['coil'][:].astype(np.float32)

        # Assign each sample as SQUID or Coil
        rng = np.random.default_rng(seed)
        self.use_squid = rng.random(len(self.ideal)) < squid_fraction

    def __len__(self):
        return len(self.ideal)

    def __getitem__(self, idx):
        target = torch.from_numpy(self.ideal[idx]).unsqueeze(0)

        if self.use_squid[idx]:
            inp = torch.from_numpy(self.squid[idx]).unsqueeze(0)
        else:
            inp = torch.from_numpy(self.coil[idx]).unsqueeze(0)

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
    path = "data/dataset/test.h5"

    # Test GradualFusionDataset
    for frac in [0.0, 0.1, 0.5, 1.0]:
        ds = GradualFusionDataset(path, base='squid', secondary_fraction=frac, augment=False)
        x, y = ds[0]
        n_with_sec = ds.has_secondary.sum()
        print(f"  GradualFusion(base=squid, f={frac:.1f}): "
              f"x={x.shape} y={y.shape} "
              f"has_secondary={n_with_sec}/{len(ds)}")

    # Test MixedSingleChannelDataset
    for sf in [0.0, 0.5, 1.0]:
        ds = MixedSingleChannelDataset(path, squid_fraction=sf, augment=False)
        x, y = ds[0]
        n_squid = ds.use_squid.sum()
        print(f"  Mixed(squid_frac={sf:.1f}): x={x.shape} n_squid={n_squid}/{len(ds)}")

    print("  OK ✓")
