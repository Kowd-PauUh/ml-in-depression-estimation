from typing import Tuple

import torch


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
