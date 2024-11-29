from typing import Tuple

from tqdm.auto import tqdm
import pandas as pd
from loguru import logger
from torch.utils.data import Dataset
import torch

from src.waveform.utils import load_waveform, trim_waveform, resample_waveform


class FineTuningDataset(Dataset):
    """
    A custom Dataset class for fine-tuning models on audio data.

    This class handles the loading and preprocessing of audio waveforms from
    a pandas DataFrame containing metadata such as file paths, start times, 
    and end times. Depending on the `fast_mode` flag, it can either load all 
    waveforms into memory upfront or load them on-the-fly during training.

    Attributes
    ----------
    df : pd.DataFrame
        DataFrame containing metadata for the audio samples.
    filepath_column_name : str
        Name of the column in `df` that contains file paths to the audio files.
    start_time_column_name : str
        Name of the column in `df` that contains start times (in seconds) for trimming the audio.
    end_time_column_name : str
        Name of the column in `df` that contains end times (in seconds) for trimming the audio.
    fast_mode : bool
        If True, all waveforms are loaded into memory during initialization.
    _waveforms : dict
        Dictionary storing preloaded waveforms and their sample rates when `fast_mode` is True.

    Methods
    -------
    __len__()
        Returns the number of samples in the dataset.
    __getitem__(i)
        Returns the i-th sample and its sample rate from the dataset.
    sizeof_fmt(num, suffix="B")
        Converts a byte size into a human-readable format.

    Example
    -------
    >>> dataset = FineTuningDataset(
    ...     df=df,
    ...     filepath_column_name='source',
    ...     start_time_column_name='start_time',
    ...     end_time_column_name='end_time',
    ...     fast_mode=True  # set to False for RAM-optimised data loading
    ... )
    """
    _waveforms: dict = dict()
    total_size: int = 0

    def __init__(
        self,
        df: pd.DataFrame,
        filepath_column_name: str,
        target_column_name: str,
        start_time_column_name: str,
        end_time_column_name: str,
        fast_mode: bool,
        target_sr: int | None = 16000,
    ):
        """
        Initializes the FineTuningDataset object.

        If `fast_mode` is set to True, all unique waveforms specified in the 
        DataFrame are preloaded into memory and stored in the `_waveforms` 
        attribute. Otherwise, waveforms are loaded from disk during each 
        call to `__getitem__`.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing metadata for the audio samples.
        filepath_column_name : str
            Name of the column in `df` that contains file paths to the audio files.
        target_column_name : str
            Name of the column in `df` that contains target variable value.
        start_time_column_name : str
            Name of the column in `df` that contains start 
            times (in seconds) for trimming the audio.
        end_time_column_name : str
            Name of the column in `df` that contains end 
            times (in seconds) for trimming the audio.
        fast_mode : bool
            If True, all waveforms are preloaded into memory.
        target_sr : int or None, optional
            Sample rate to resample all waveforms to. 
            If "None" no resampling is applied. Defaults to 16000.

        Returns
        -------
        None
        """
        self.df = df.copy().reset_index(drop=True)
        self.filepath_column_name = filepath_column_name
        self.target_column_name = target_column_name
        self.start_time_column_name = start_time_column_name
        self.end_time_column_name = end_time_column_name
        self.fast_mode = fast_mode
        self.target_sr = target_sr

        unique_filepaths_cnt = len(self.df[self.filepath_column_name].unique())
        if self.fast_mode:
            logger.info(
                f'Dataset is initialized in fast mode. All waveforms '
                f'{unique_filepaths_cnt} will be loaded to RAM.'
            )

            filepaths = self.df[self.filepath_column_name].unique()  # all unique filepaths
            pbar = tqdm(filepaths, desc='Loading waveforms')

            for filepath in pbar:
                if filepath in FineTuningDataset._waveforms:
                    continue

                # load waveform
                waveform, sr = load_waveform(audio_path=filepath, normalize=True)

                # resample if needed
                if self.target_sr is not None:
                    waveform = resample_waveform(
                        waveform=waveform,
                        orig_sample_rate=sr,
                        target_sample_rate=self.target_sr
                    )
                    sr = self.target_sr

                # cache waveform
                FineTuningDataset._waveforms[filepath] = (waveform, sr)

                # log to progress bar
                FineTuningDataset.total_size += waveform.element_size() * waveform.nelement()
                pbar.set_postfix(
                    {
                        'files': len(FineTuningDataset._waveforms), 
                        'total_size': self.sizeof_fmt(FineTuningDataset.total_size)
                    }
                )
        else:
            logger.info(
                f'Dataset is initialized in RAM-optimised mode. '
                f'Waveforms ({unique_filepaths_cnt}) '
                f'will be loaded on-the-fly at each training step.'
            )

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples in the dataset.
        """
        return len(self.df)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int, float]:
        """
        Returns the i-th sample from the dataset.

        This method loads the waveform corresponding to the i-th entry 
        in the DataFrame. If `fast_mode` is enabled, the waveform is 
        retrieved from memory; otherwise, it is loaded from disk. The 
        waveform is then trimmed to the specified start and end times.

        Parameters
        ----------
        i : int
            Index of the sample to retrieve.

        Returns
        -------
        Tuple[torch.Tensor, int, float]
            A tuple containing the waveform as a torch tensor, the sample rate and the target value.
        """
        # get information on audio sample
        row = self.df.iloc[i]
        filepath = row[self.filepath_column_name]
        start_time, end_time = row[self.start_time_column_name], row[self.end_time_column_name]
        target_value = float(row[self.target_column_name])

        # load waveform
        if self.fast_mode:
            waveform, sr = FineTuningDataset._waveforms[filepath]
        else:
            waveform, sr = load_waveform(audio_path=filepath, normalize=True)

        # trim waveform into audio sample
        waveform = trim_waveform(
            waveform=waveform,
            start_time=start_time,
            end_time=end_time,
            sample_rate=sr
        )
        return waveform, sr, target_value

    @staticmethod
    def sizeof_fmt(num: int | float, suffix="B") -> str:
        """
        Converts a byte size into a human-readable format.

        Parameters
        ----------
        num : int
            The size in bytes.
        suffix : str, optional
            The suffix to append to the formatted size (default is "B").

        Returns
        -------
        str
            Human-readable string representation of the size.
        """
        for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
            if abs(num) < 1024.0:
                return f"{num:3.1f}{unit}{suffix}"
            num /= 1024.0
        return f"{num:.1f}Yi{suffix}"
