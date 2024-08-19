import numpy as np
import torch
import torchaudio
import librosa
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks




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


def waveform_tempo(waveform: torch.Tensor, sr: int):
    waveform_np = waveform.numpy().flatten()
    tempo, _ = librosa.beat.beat_track(y=waveform_np, sr=sr)
    return tempo[0]

def waveform_spectral_centroid(waveform: torch.Tensor, sr: int):
    waveform_np = waveform.numpy().flatten()
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=waveform_np, sr=sr))
    return spectral_centroid

def waveform_spectral_bandwidth(waveform: torch.Tensor, sr: int):
    waveform_np = waveform.numpy().flatten()
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=waveform_np, sr=sr))
    return spectral_bandwidth

def waveform_rolloff(waveform: torch.Tensor, sr: int):
    waveform_np = waveform.numpy().flatten()
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=waveform_np, sr=sr))
    return rolloff

def waveform_rmse(waveform: torch.Tensor):
    waveform_np = waveform.numpy().flatten()
    rmse = np.mean(librosa.feature.rms(y=waveform_np))
    return rmse


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
