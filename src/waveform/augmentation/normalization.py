import warnings

import torch


def zero_mean(waveform: torch.Tensor):
    """Centers the waveform to zero mean."""
    return waveform - waveform.mean()


def normalize_by_std(waveform: torch.Tensor):
    """Normalizes the waveform by its standard deviation."""
    std = waveform.std()

    if std == 0:
        warnings.warn('Standard deviation is zero, normalization is not possible.')
        return waveform

    return waveform / std
