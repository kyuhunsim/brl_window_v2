"""Temporally correlated exploration processes for continuous control."""

from __future__ import annotations

import numpy as np


class PinkNoise:
    """Online Voss--McCartney approximation of pink (1/f) noise.

    Each sample is zero-mean and temporally correlated.  The generator is
    deliberately independent of SAC's torch RNG so the exploration stream is
    reproducible from the training seed.
    """

    def __init__(self, dim: int, seed: int, octaves: int = 8):
        self.dim = int(dim)
        self.octaves = max(int(octaves), 2)
        self.rng = np.random.default_rng(int(seed))
        self.rows = self.rng.standard_normal((self.octaves, self.dim))
        self.counter = 0

    def reset(self) -> None:
        self.rows = self.rng.standard_normal((self.octaves, self.dim))
        self.counter = 0

    def sample(self) -> np.ndarray:
        self.counter += 1
        value = self.counter
        row = 0
        while row < self.octaves and (value & 1) == 0:
            self.rows[row] = self.rng.standard_normal(self.dim)
            value >>= 1
            row += 1
        # The fastest octave is refreshed every step.
        self.rows[-1] = self.rng.standard_normal(self.dim)
        return np.sum(self.rows, axis=0) / np.sqrt(float(self.octaves))
