import numpy as np
import torch


def waveform_slope_sign_changes(waveform: torch.Tensor) -> float:
    waveform_np = waveform.numpy().flatten()
    return np.sum(np.diff(np.sign(np.diff(waveform_np))))


def waveform_average_amplitude_change(waveform: torch.Tensor) -> float:
    waveform_np = waveform.numpy().flatten()
    return np.mean(np.abs(np.diff(waveform_np)))


def waveform_difference_absolute_deviation(waveform: torch.Tensor) -> float:
    waveform_np = waveform.numpy().flatten()
    return np.mean(np.abs(np.diff(waveform_np - np.mean(waveform_np))))


def waveform_integrated_absolute_value(waveform: torch.Tensor) -> float:
    waveform_np = waveform.numpy().flatten()
    return np.sum(np.abs(waveform_np))


def waveform_simple_square_integral(waveform: torch.Tensor) -> float:
    waveform_np = waveform.numpy().flatten()
    return np.sum(waveform_np ** 2)


def waveform_length(waveform: torch.Tensor) -> float:
    waveform_np = waveform.numpy().flatten()
    return np.sum(np.abs(np.diff(waveform_np)))
