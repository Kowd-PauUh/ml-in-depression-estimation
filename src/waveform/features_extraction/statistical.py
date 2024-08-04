import numpy as np
import torch
from scipy.stats import skew, kurtosis


def waveform_mean_value(waveform: torch.Tensor):
    waveform_np = waveform.numpy().flatten()
    mean_value = np.mean(waveform_np)
    return mean_value


def waveform_variance(waveform: torch.Tensor):
    waveform_np = waveform.numpy().flatten()
    variance = np.var(waveform_np)
    return variance


def waveform_std_dev(waveform: torch.Tensor):
    waveform_np = waveform.numpy().flatten()
    std_dev = np.std(waveform_np)
    return std_dev


def waveform_skewness(waveform: torch.Tensor):
    waveform_np = waveform.numpy().flatten()
    skewness = skew(waveform_np)
    return skewness


def waveform_kurtosis(waveform: torch.Tensor):
    waveform_np = waveform.numpy().flatten()
    kurt = kurtosis(waveform_np)
    return kurt


def waveform_mean_absolute_value(waveform: torch.Tensor):
    waveform_np = waveform.numpy().flatten()
    return np.mean(np.abs(waveform_np))


def waveform_mean_logarithm_kernel(waveform: torch.Tensor, epsilon: float = 1e-10):
    waveform_np = waveform.numpy().flatten() + epsilon
    return np.mean(np.log(np.abs(waveform_np)))


def waveform_maximum_amplitude(waveform: torch.Tensor):
    waveform_np = waveform.numpy().flatten()
    return np.max(np.abs(waveform_np))


def waveform_minimum_amplitude(waveform: torch.Tensor):
    waveform_np = waveform.numpy().flatten()
    return np.max(np.abs(waveform_np))
