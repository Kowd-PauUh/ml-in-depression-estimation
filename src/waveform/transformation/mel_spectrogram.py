import torch
import torchaudio


def mel_spectrogram(
    waveform: torch.Tensor, 
    sample_rate: int,
    num_mel_bins: int = 128, 
    length: int | None = 3072,
    padding: bool = True,
    truncation: bool = True,
):
    """
    Computes MEL spectrogram for given waveform.
    
    Parameters
    ----------
    waveform : torch.Tensor
        Audio waveform as a tensor.
    sample_rate : int
        Audio sample rate (frequency).
    num_mel_bins : int, optional
        Number of bins in MEL spectrogram to be computed. Defaults to 128.
    length : int or None, optional
        Desired length of spectrogram. If specified, spectrograms are 
        truncated and padding is performed if corresponding flags are set.
        Defaults to 3072 which approximately equals to 30s of audio.
    padding : bool, optional
        Boolean flag on whether to perform spectrogram padding. 
        Padding is performed only if `length` parameter is specified.
    truncation : bool, optional
        Boolean flag on whether to perform spectrogram truncation. 
        Truncation is performed only if `length` parameter is specified.

    Returns
    -------
    torch.Tensor
        Tensor of shape (L, num_mel_bins), where L equals `length` if specified,
        otherwise L can be approximated as `100 * len(waveform.shape[1]) / sample_rate`
    """
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, 
        htk_compat=True, 
        sample_frequency=sample_rate, 
        use_energy=False,
        window_type='hanning', 
        num_mel_bins=num_mel_bins, 
        dither=0.0, 
        frame_length=25,  # [ms] 
        frame_shift=10  # [ms]
    )

    if length:
        n_frames = fbank.shape[0]
        p = length - n_frames
        
        if padding and p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        if truncation and p < 0:
            fbank = fbank[0:length, :]

    return fbank.T
