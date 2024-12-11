import os
from typing import Dict, List
from pathlib import Path
from tempfile import TemporaryDirectory
from inspect import signature

import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from fire import Fire
from loguru import logger

from disvoice import Articulation, Glottal, Phonation, Prosody, RepLearning

from src.waveform.utils import load_waveform, trim_waveform, save_waveform


PROJECT_DIR = Path(os.environ['PROJECT_DIR'])
PREPROCESSED_DATA_PATH = PROJECT_DIR / 'data/preprocessed_data'
FEATURES_EXTRACTORS_MAPPING = {
    c.__name__: c
    for c in [Articulation, Glottal, Phonation, Prosody, RepLearning]
}
FEATURES_SAVE_DIR = PROJECT_DIR / 'data/processed_data/disvoice_features'
tqdm.pandas()


def extract_waveform_features(
    waveform: torch.Tensor, 
    sample_rate: int, 
    features_extractor_cls
) -> Dict[str, float]:
    with TemporaryDirectory() as temp_dir:
        # save temporary waveform file
        sample_path = Path(temp_dir) / 'sample.wav'
        save_waveform(waveform=waveform, sample_rate=sample_rate, save_path=sample_path)

        # initialize features extractor
        features_extractor = features_extractor_cls(
            **{
                k: v 
                for k, v in {'temp_dir': temp_dir, 'model': 'CAE'}.items()
                if k in signature(features_extractor_cls).parameters
            }
        )

        # extract features
        features = features_extractor.extract_features_file(
            sample_path.as_posix(), 
            static=True, 
            plots=False, 
            fmt="dataframe"
        )

    return features.iloc[0].to_dict()


def extract_group_features(group, features_extractor_cls):
    source = group.iloc[0]['source']
    waveform, sr = load_waveform(source)
    return group.progress_apply(
        lambda row: pd.Series(
            extract_waveform_features(
                waveform=trim_waveform(
                    waveform=waveform, 
                    sample_rate=sr, 
                    start_time=row['start_time'],
                    end_time=row['end_time']
                ), 
                sample_rate=sr,
                features_extractor_cls=features_extractor_cls
            )
        ), 
        axis=1
    )


def extract_features(
    features_extractor_cls_name: str,
    *,
    dataset_path: str | Path = PREPROCESSED_DATA_PATH / 'data.csv',
    split_df_path: str | Path = PREPROCESSED_DATA_PATH / 'split.csv',
    features_save_dir: str | Path = FEATURES_SAVE_DIR,
    match_split_df_on: str = 'participant_id',
    grouping_column_name: str = 'participant_id',
    split_column_name: str = 'split',
    splits: List[str] | None = None,
    downsample_to: int | None = None
):
    if features_extractor_cls_name not in FEATURES_EXTRACTORS_MAPPING:
        raise ValueError(
            f'`features_extractor_cls_name` must be one of: ' + \
            f'{", ".join(list(FEATURES_EXTRACTORS_MAPPING.keys()))}'
        )
    logger.info(f'Using dataset at {dataset_path} with splits defined at {split_df_path}')

    if isinstance(features_save_dir, str):
        features_save_dir = Path(features_save_dir)
    features_save_dir.mkdir(parents=True, exist_ok=True)

    # read dataframe
    df = pd.read_csv(dataset_path)
    if downsample_to:
        logger.info(f'Dataset will be downsampled to {downsample_to} entries.')
        df = df.sample(downsample_to, random_state=42)

    # assign split to each row in dataframe
    split_df = pd.read_csv(split_df_path).set_index(match_split_df_on)
    df[split_column_name] = df[match_split_df_on].apply(lambda i: split_df.loc[i][split_column_name])
    available_splits = list(split_df[split_column_name].unique())
    if not splits:
        splits = available_splits
    logger.info(f'Available splits: {available_splits}. Using user defined: {splits}')

    for split in splits:
        logger.info(f'Processing "{split}" split.')

        # get data split
        data_by_split = df[df[split_column_name] == split]
    
        # extract features and save as CSV file
        features = data_by_split.groupby(grouping_column_name).progress_apply(
            lambda group: extract_group_features(
                group=group,
                features_extractor_cls=FEATURES_EXTRACTORS_MAPPING[features_extractor_cls_name]
            ), 
            include_groups=False
        ).reset_index(drop=True)
        features.to_csv(
            features_save_dir / f"{features_extractor_cls_name}_{split}.csv", 
            index=False
        )


if __name__ == '__main__':
    Fire(extract_features)
