"""Shared helper. Keep this file in the same folder as train_export.py and app.py."""
import numpy as np


def log_transform(X):
    return np.log1p(np.clip(X, 0, None))
