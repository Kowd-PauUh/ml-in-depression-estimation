from typing import Tuple
from pathlib import Path

import torch
import torchaudio


def load_waveform(
    audio_path: Path | str,
    audio_format: str = 'wav'
) -> Tuple[torch.Tensor, int]:
    """
    Parameters
    ----------
    audio_path : Path or str
        Path to the audio file.
    audio_format : str, optional
        Audio format. Defaults to "wav"

    Returns
    -------
    Tuple[torch.Tensor, int]
        Tuple where the first element is a waveform (Tensor) and the second is a sample rate.
    """
    waveform, sr = torchaudio.load(audio_path, format=audio_format)
    return waveform, sr


def trim_waveform(
    waveform: torch.Tensor,
    sample_rate: int,
    start_time: float | None = None, 
    end_time: float | None = None
) -> torch.Tensor:
    """
    Trims waveform from `start_time` to `end_time`.
    
    Parameters
    ----------
    waveform : torch.Tensor
        Audio waveform as a tensor.
    sample_rate : int
        Audio sample rate (frequency).
    start_time : float, optional
        Time of waveform sample start (in seconds).
    end_time : float, optional
        Time of waveform sample end (in seconds).

    Returns
    -------
    torch.Tensor
        Trimmed waveform.
    """
    start_idx = int(sample_rate * start_time) if start_time is not None else None
    end_idx = int(sample_rate * end_time) if end_time is not None else None

    return waveform[:, start_idx:end_idx]
