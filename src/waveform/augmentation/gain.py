import random

import torch


def random_gain(waveform: torch.Tensor) -> torch.Tensor:
    """Multiplies waveform by a random gain chosen uniformly in range [-6.0, +6.0] dB."""
    gain_db = random.uniform(-6.0, 6.0)
    gain_linear = 10 ** (gain_db / 20.0)
    
    waveform_with_gain = waveform * gain_linear
    return waveform_with_gain
