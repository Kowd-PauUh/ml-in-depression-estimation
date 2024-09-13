from typing import Tuple
from pathlib import Path

import torch
import torchaudio


def load_waveform(
    audio_path: Path | str,
    audio_format: str = 'wav',
    normalize: bool = False
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
    waveform, sr = torchaudio.load(audio_path, format=audio_format, normalize=normalize)

    # ensure the output waveform is mono
    waveform = torch.mean(waveform.to(torch.float32), dim=0, keepdim=True)
    if not normalize:
        waveform = waveform.to(torch.int16)

    return waveform, sr


def save_waveform(
    waveform: torch.Tensor,
    sample_rate: int,
    save_path: Path | str,
    audio_format: str = 'wav'
):
    """
    Save a waveform to an audio file.

    Parameters
    ----------
    waveform : torch.Tensor
        The waveform to save.
    sample_rate : int
        The sample rate of the waveform.
    save_path : Path or str
        Path where the audio file will be saved.
    audio_format : str, optional
        The format to save the audio file in. Defaults to 'wav'.
    """
    torchaudio.save(
        uri=save_path, 
        src=waveform, 
        sample_rate=sample_rate, 
        format=audio_format
    )


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


def resample_waveform(waveform: torch.Tensor, orig_sample_rate: int, target_sample_rate: int) -> torch.Tensor:
    """
    Resamples the waveform to match the target sample rate.
    
    Parameters
    ----------
    waveform : torch.Tensor
        The waveform tensor to resample.
    orig_sample_rate : int
        The original sample rate of the waveform.
    target_sample_rate : int
        The target sample rate to resample to.
        
    Returns
    -------
    torch.Tensor
        The resampled waveform.
    """
    if orig_sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=orig_sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
    return waveform
