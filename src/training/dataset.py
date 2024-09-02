import pandas as pd
from loguru import logger
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from src.waveform.utils import load_waveform, trim_waveform


class FineTuningDataset(Dataset):
    """
    HELP

    Attributes
    ----------
    HELP

    Methods
    -------
    HELP

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
    def __init__(
        self, 
        df: pd.DataFrame, 
        filepath_column_name: str,
        start_time_column_name: str,
        end_time_column_name: str,
        fast_mode: bool,
    ):
        """
        Initializes `Dataset` object. If `fast_mode` is True, loads 
        all the unique waveforms from `df` to `_waveforms` attribute. 

        Parameters
        ----------
        HELP

        Returns
        -------
        Tuple[torch.Tensor, int]
            Tuple with first element as waveform torch tensor, second - sample rate.
        """
        self.df = df
        self.filepath_column_name = filepath_column_name
        self.start_time_column_name = start_time_column_name
        self.end_time_column_name = end_time_column_name
        self.fast_mode = fast_mode
        self._waveforms = {}

        if self.fast_mode:
            logger.info('Dataset is initialized in fast mode. All waveforms will be loaded to RAM.')

            filepaths = self.df[self.filepath_column_name].unique()  # all unique filepaths
            pbar = tqdm(filepaths, desc='Loading waveforms')
            total_size = 0

            for filepath in pbar:
                # load waveform
                waveform, sr = load_waveform(audio_path=filepath)
                self._waveforms[filepath] = (waveform, sr)
                
                # log to progress bar
                total_size += waveform.element_size() * waveform.nelement()
                pbar.set_postfix({'files': len(self._waveforms), 'total_size': self.sizeof_fmt(total_size)})
        else:
            logger.info('Dataset is initialized in RAM-optimised mode. Waveforms will be loaded at each training step.')
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        """
        HELP 
        """
        # get information on audio sample
        row = self.df.iloc[i]
        filepath = row[self.filepath_column_name]
        start_time, end_time = row[self.start_time_column_name], row[self.end_time_column_name]

        # load waveform
        if self.fast_mode:
            waveform, sr = self._waveforms[filepath]
        else:
            waveform, sr = load_waveform(audio_path=filepath)

        # trim waveform into audio sample
        waveform = trim_waveform(
            waveform=waveform, 
            start_time=start_time, 
            end_time=end_time, 
            sample_rate=sr
        )
        return waveform, sr

    @staticmethod
    def sizeof_fmt(num, suffix="B"):
        # transform bytes number into human readable format
        for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
            if abs(num) < 1024.0:
                return f"{num:3.1f}{unit}{suffix}"
            num /= 1024.0
        return f"{num:.1f}Yi{suffix}"
