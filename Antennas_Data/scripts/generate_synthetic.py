"""Generate a small synthetic dataset for testing the ML pipeline.

Creates `data/synthetic_dataset.csv` with parameter columns and two targets:
- `s11_db` (simulated S11 in dB, lower is better)
- `axial_ratio_db` (axial ratio in dB, lower is better)

This is purely synthetic and only for pipeline validation before connecting
to real CST/HFSS exports.
"""
import os
import numpy as np
import pandas as pd


def make_dataset(path, n=200, seed=0):
    rng = np.random.RandomState(seed)
    # parameters: widths and height (example)
    w1 = rng.uniform(0.5, 2.0, size=n)
    w2 = rng.uniform(0.5, 2.0, size=n)
    h1 = rng.uniform(0.1, 1.0, size=n)

    # synthetic targets (toy functions)
    s11_db = (
        -20
        + (w1 - 1.2) ** 2 * 10
        + (w2 - 1.0) ** 2 * 8
        + rng.normal(0, 0.5, size=n)
    )
    axial_ratio_db = (
        1.0
        + (h1 - 0.6) ** 2 * 6
        + (w2 - 1.1) ** 2 * 3
        + rng.normal(0, 0.2, size=n)
    )

    df = pd.DataFrame({
        'w1': w1,
        'w2': w2,
        'h1': h1,
        's11_db': s11_db,
        'axial_ratio_db': axial_ratio_db,
    })
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print('Wrote synthetic dataset to', path)


if __name__ == '__main__':
    make_dataset(os.path.join('data', 'synthetic_dataset.csv'))
