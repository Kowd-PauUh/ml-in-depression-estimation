import warnings
import random

import torch
import torchaudio.transforms as T


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


def resample_waveform(waveform: torch.Tensor, orig_sample_rate: int) -> torch.Tensor:
    """
    Resamples the waveform by a random factor uniformly chosen in range [0.8, 1.25].

    Parameters
    ----------
    waveform : torch.Tensor
        The original waveform.
    orig_sample_rate : int
        The original sample rate of the waveform.

    Returns
    -------
    torch.Tensor
        The resampled waveform.
    """
    resample_factor = random.uniform(0.8, 1.25)
    new_sample_rate = int(orig_sample_rate * resample_factor)
    
    resampler = T.Resample(orig_freq=orig_sample_rate, new_freq=new_sample_rate)
    resampled_waveform = resampler(waveform)
    
    return resampled_waveform
