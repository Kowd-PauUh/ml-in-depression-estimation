from typing import Tuple
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


def random_resample(
    waveform: torch.Tensor, 
    orig_sample_rate: int, 
    trim: bool = True,
    pad: bool = True
) -> torch.Tensor:
    """
    Resamples the waveform by a random factor uniformly chosen in range [0.8, 1.25].

    Parameters
    ----------
    waveform : torch.Tensor
        The original waveform.
    orig_sample_rate : int
        The original sample rate of the waveform.
    trim : bool, optional
        Boolean flag on whether to trim the resampled waveform to its original length.
    pad : bool, optional
        Boolean flag on whether to zero pad the resampled waveform to its original length.

    Returns
    -------
    torch.Tensor
        The resampled waveform.
    """
    original_len = waveform.size(1)
    resample_factor = random.uniform(0.8, 1.25)
    new_sample_rate = int(orig_sample_rate * resample_factor)
    
    resampler = T.Resample(orig_freq=orig_sample_rate, new_freq=new_sample_rate)
    resampled_waveform = resampler(waveform)

    # trim if needed
    if resampled_waveform.size(1) > original_len and trim:
        resampled_waveform = resampled_waveform[:, :original_len]

    # zero pad if needed
    if resampled_waveform.size(1) < original_len and pad:
        padding = original_len - resampled_waveform.size(1)
        resampled_waveform = torch.nn.functional.pad(resampled_waveform, (0, padding))
    
    return resampled_waveform


def random_gain(waveform: torch.Tensor) -> torch.Tensor:
    """Multiplies waveform by a random gain chosen uniformly in range [-6.0, +6.0] dB."""
    gain_db = random.uniform(-6.0, 6.0)
    gain_linear = 10 ** (gain_db / 20.0)
    
    waveform_with_gain = waveform * gain_linear
    return waveform_with_gain


def mixup(
    waveform1: torch.Tensor,
    waveform2: torch.Tensor,
    alpha: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Performs mixup on two waveforms.

    Note
    ----
    The longer waveform will be trimmed to the length of shorter one.

    Parameters
    ----------
    waveform1 : torch.Tensor
        The first waveform tensor.
    waveform2 : torch.Tensor
        The second waveform tensor.
    alpha : float, optional
        The mixup interpolation parameter. Defaults to 0.2.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Tuple containing two mixed waveforms.
    """
    # Ensure both waveforms have the same length
    min_length = min(waveform1.size(1), waveform2.size(1))
    waveform1 = waveform1[:, :min_length]
    waveform2 = waveform2[:, :min_length]
    
    # Sample lambda from Beta distribution
    lam = torch.distributions.Beta(alpha, alpha).sample()
    
    # Perform mixup
    mixed_waveform1 = lam * waveform1 + (1 - lam) * waveform2
    mixed_waveform2 = (1 - lam) * waveform1 + lam * waveform2
    
    return mixed_waveform1, mixed_waveform2
