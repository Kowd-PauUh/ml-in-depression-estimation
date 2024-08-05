import numpy as np
import torch
import torchaudio
import librosa
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks


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


def waveform_slope_sign_changes(waveform: torch.Tensor):
    waveform_np = waveform.numpy().flatten()
    return np.sum(np.abs(np.diff(np.sign(np.diff(waveform_np))))) / 2


def waveform_mean_absolute_value(waveform: torch.Tensor):
    waveform_np = waveform.numpy().flatten()
    return np.mean(np.abs(waveform_np))


def waveform_logarithm_detector(waveform: torch.Tensor):
    waveform_np = waveform.numpy().flatten()
    return np.mean(np.log1p(np.abs(waveform_np)))


def waveform_average_amplitude_change(waveform: torch.Tensor):
    waveform_np = waveform.numpy().flatten()
    return np.mean(np.abs(np.diff(waveform_np)))


def waveform_difference_absolute_deviation(waveform: torch.Tensor):
    waveform_np = waveform.numpy().flatten()
    return np.mean(np.abs(np.diff(waveform_np - np.mean(waveform_np))))


def waveform_integrated_absolute_value(waveform: torch.Tensor):
    waveform_np = waveform.numpy().flatten()
    return np.sum(np.abs(waveform_np))


def waveform_mean_logarithm_kernel(waveform: torch.Tensor, epsilon: float = 1e-10):
    waveform_np = waveform.numpy().flatten() + epsilon
    return np.mean(np.log(np.abs(waveform_np)))


def waveform_simple_square_integral(waveform: torch.Tensor):
    waveform_np = waveform.numpy().flatten()
    return np.sum(waveform_np ** 2)


def waveform_moments(waveform: torch.Tensor, order: int):
    waveform_np = waveform.numpy().flatten()
    return np.mean((waveform_np - np.mean(waveform_np)) ** order)


def waveform_maximum_amplitude(waveform: torch.Tensor):
    waveform_np = waveform.numpy().flatten()
    return np.max(np.abs(waveform_np))


def waveform_power_spectrum_ratio(waveform: torch.Tensor):
    stft = np.abs(librosa.stft(waveform.numpy().flatten()))
    power_spectrum = np.sum(stft ** 2, axis=0)
    return np.mean(power_spectrum) / np.std(power_spectrum)


def waveform_peak_frequency(waveform: torch.Tensor):
    stft = np.abs(librosa.stft(waveform.numpy().flatten()))
    frequencies = np.argmax(stft, axis=0)
    return np.mean(frequencies)


def waveform_mean_power(waveform: torch.Tensor):
    waveform_np = waveform.numpy().flatten()
    return np.mean(waveform_np ** 2)


def waveform_total_power(waveform: torch.Tensor):
    waveform_np = waveform.numpy().flatten()
    return np.sum(waveform_np ** 2)


def waveform_variance_of_central_frequency(waveform: torch.Tensor, sr: int):
    stft = np.abs(librosa.stft(waveform.numpy().flatten()))
    frequencies = np.linspace(0, sr / 2, stft.shape[0])[:, np.newaxis]
    central_frequency = np.sum(frequencies * stft, axis=0) / np.sum(stft, axis=0)
    return np.var(central_frequency)


def waveform_hjorth_parameters(waveform: torch.Tensor):
    waveform_np = waveform.numpy().flatten()
    derivative = np.diff(waveform_np)
    second_derivative = np.diff(derivative)
    
    activity = np.var(waveform_np)
    mobility = np.sqrt(np.var(derivative) / activity)
    complexity = np.sqrt(np.var(second_derivative) / np.var(derivative)) / mobility
    
    return activity, mobility, complexity


def waveform_length(waveform: torch.Tensor):
    waveform_np = waveform.numpy().flatten()
    return np.sum(np.abs(np.diff(waveform_np)))


def waveform_glottal_features(waveform: torch.Tensor, sr: int):
    waveform_np = waveform.numpy().flatten()
    peaks, _ = find_peaks(waveform_np)
    gci = np.diff(peaks) / sr
    
    op = np.mean(gci)
    cp = np.median(gci)
    c = np.std(gci)
    
    return op, cp, c


def waveform_tempo_spectral_features(waveform: torch.Tensor, sr: int):
    waveform_np = waveform.numpy().flatten()
    
    tempo, _ = librosa.beat.beat_track(y=waveform_np, sr=sr)
    
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=waveform_np, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=waveform_np, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=waveform_np, sr=sr))
    rmse = np.mean(librosa.feature.rms(y=waveform_np))
    
    return tempo[0], spectral_centroid, spectral_bandwidth, rolloff, rmse


def waveform_formant_features(waveform: torch.Tensor):
    waveform_np = waveform.numpy().flatten()
    lpc = librosa.lpc(waveform_np, order=2)
    
    formants = np.roots(lpc)
    formants = formants[np.imag(formants) >= 0]
    
    f1 = np.mean(np.abs(formants))
    f2 = np.median(np.abs(formants))
    f3 = np.std(np.abs(formants))
    
    return f1, f2, f3


def waveform_other_features(waveform: torch.Tensor, sr: int):
    waveform_np = waveform.numpy().flatten()
    
    pitches, magnitudes = librosa.core.piptrack(y=waveform_np, sr=sr)
    mean_pitch = np.mean(pitches)
    std_pitch = np.std(pitches)
    mean_magnitude = np.mean(magnitudes)
    
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=waveform_np))
    
    voice_portions = np.sum(librosa.effects.split(y=waveform_np, top_db=20)) / len(waveform_np)
    
    return mean_pitch, std_pitch, mean_magnitude, zcr, voice_portions


def waveform_shimmer(waveform: torch.Tensor):
    waveform_np = waveform.numpy().flatten()
    shimmer = np.std(np.diff(waveform_np)) / np.mean(waveform_np)
    return shimmer


def waveform_jitter(waveform: torch.Tensor):
    waveform_np = waveform.numpy().flatten()
    jitter = np.mean(np.abs(np.diff(np.diff(waveform_np)))) / np.mean(np.abs(np.diff(waveform_np)))
    return jitter


def waveform_hnr(waveform: torch.Tensor):
    waveform_np = waveform.numpy().flatten()
    harmonic_to_noise = np.mean(librosa.effects.harmonic(y=waveform_np) / librosa.effects.percussive(y=waveform_np))
    return harmonic_to_noise


def waveform_harmonicity(waveform: torch.Tensor):
    waveform_np = waveform.numpy().flatten()
    harmonicity = np.mean(librosa.effects.harmonic(y=waveform_np))
    return harmonicity


def waveform_voiced_unvoiced_ratio(waveform: torch.Tensor):
    waveform_np = waveform.numpy().flatten()
    voiced_frames = librosa.effects.split(y=waveform_np, top_db=20)
    voiced_duration = np.sum([end - start for start, end in voiced_frames])
    voiced_unvoiced_ratio = voiced_duration / len(waveform_np)
    return voiced_unvoiced_ratio
