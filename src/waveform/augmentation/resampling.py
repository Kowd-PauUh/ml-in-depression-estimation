from typing import Sequence
import random
from functools import lru_cache

import torch
import torchaudio.transforms as T
import numpy as np


@lru_cache
def _get_resampler(
    orig_freq: int, 
    factors: Sequence[float] = np.linspace(0.8, 1.25, 10),
):
    resampler = T.SpeedPerturbation(orig_freq=orig_freq, factors=factors)
    return resampler


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
    # get input tensor device
    device = waveform.device

    # get original waveform length
    original_len = waveform.size(1)

    # resample waveform
    resampler = _get_resampler(orig_freq=orig_sample_rate).to(device)
    resampled_waveform, _ = resampler(waveform)

    # trim if needed
    if resampled_waveform.size(1) > original_len and trim:
        resampled_waveform = resampled_waveform[:, :original_len]

    # zero pad if needed
    if resampled_waveform.size(1) < original_len and pad:
        padding = original_len - resampled_waveform.size(1)
        resampled_waveform = torch.nn.functional.pad(resampled_waveform, (0, padding))

    return resampled_waveform
