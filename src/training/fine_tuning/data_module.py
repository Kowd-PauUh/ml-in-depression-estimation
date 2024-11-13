import os
from typing import Literal, List
from pathlib import Path

import lightning as L
import pandas as pd
from loguru import logger
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.training.fine_tuning.dataset import FineTuningDataset


PROJECT_DIR = Path(os.environ['PROJECT_DIR'])
PREPROCESSED_DATA_PATH = PROJECT_DIR / 'data/preprocessed_data'


class FineTuningDataModule(L.LightningDataModule):
    def __init__(
        self,
        *,
        # files location
        dataset_path: str | Path = PREPROCESSED_DATA_PATH / 'data.csv',
        split_df_path: str | Path = PREPROCESSED_DATA_PATH / 'split.csv',
        # datasets parameters
        match_split_df_on: str = 'participant_id',
        grouping_column_name: str = 'participant_id',
        split_column_name: str = 'split',
        start_time_column_name: str = 'start_time',
        end_time_column_name: str = 'end_time',
        filepath_column_name: str = 'source',
        target_column_name: Literal['phq_binary', 'phq_score'] = 'phq_binary',
        train_val_split_name: str = 'train',
        test_split_name: str = 'test',
        # preprocessing
        fast_mode: bool = True,
        downsample_to: int | None = None,
        val_size: float = 0.2,
        # training
        batch_size: int = 16,
        val_batch_size: int = 16,
    ):
        super().__init__()

        # datasets paths
        self.dataset_path = dataset_path
        self.split_df_path = split_df_path

        # datasets parameters
        self.match_split_df_on = match_split_df_on
        self.grouping_column_name = grouping_column_name
        self.split_column_name = split_column_name
        self.start_time_column_name = start_time_column_name
        self.end_time_column_name = end_time_column_name
        self.filepath_column_name = filepath_column_name
        self.target_column_name = target_column_name
        self.train_val_split_name = train_val_split_name
        self.test_split_name = test_split_name

        # preprocessing
        self.fast_mode = fast_mode
        self.downsample_to = downsample_to
        self.val_size = val_size
        self.train_size = 1 - val_size
        self.collate_fn = lambda batch: batch

        # batch sizes
        if batch_size % 2 != 0 or val_batch_size % 2 != 0:
            raise ValueError(
                f'Batch sizes must be divisible by two. '
                f'Got {batch_size=}, {val_batch_size=}.'
            )
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size

    def load_df(self) -> pd.DataFrame:
        logger.info(f"Using dataset at {self.dataset_path} with split data at {self.split_df_path}")

        # read dataframe and downsample if needed
        df = pd.read_csv(self.dataset_path)
        if self.downsample_to and self.downsample_to >= len(df):
            logger.warning(
                f'Tried to downsample dataset to {self.downsample_to} ' + \
                f'entries but it contains {len(df)} entries.'
            )
        elif self.downsample_to:
            df = df.sample(self.downsample_to, random_state=42)
            logger.info(f'Dataset is downsampled to {self.downsample_to} entries.')

        # assign split to each row in dataframe
        split_df = pd.read_csv(self.split_df_path).set_index(self.match_split_df_on)
        df[self.split_column_name] = df[self.match_split_df_on].apply(
            lambda i: split_df.loc[i][self.split_column_name]
        )
        available_splits = list(split_df[self.split_column_name].unique())

        # validate split names
        if self.train_val_split_name not in available_splits:
            raise ValueError(
                f'Trying to use {self.train_val_split_name=} but '
                f'available splits are: {available_splits}'
            )
        if self.test_split_name not in available_splits:
            raise ValueError(
                f'Trying to use {self.test_split_name=} but '
                f'available splits are: {available_splits}'
            )

        return df

    def setup(
        self,
        df: pd.DataFrame | None = None,
        train_ids: List[int] | None = None,
        val_ids: List[int] | None = None,
        random_state: int | None = 42,
        stage: str | None = None
    ):
        if df is None:
            df = self.load_df()

        test_df = df[df[self.split_column_name] == self.test_split_name]

        if train_ids is not None and val_ids is not None:
            train_df = df.iloc[train_ids]
            val_df = df.iloc[val_ids]
        else:
            logger.info(
                f'Setting up {self.__class__.__name__} without specifying train and val indices. '
                f'Training data will be split into train ad val split with stratification if possible '
                f'with each group appearing in only one dataset.'
            )

            # group stratified split for train and val df with random_state
            train_val_df = df[df[self.split_column_name] == self.train_val_split_name]
            groups = train_val_df.groupby(self.grouping_column_name).first()

            try:
                train_groups, val_groups = train_test_split(
                    groups.index,
                    test_size=self.val_size,
                    stratify=groups[self.target_column_name],
                    random_state=random_state
                )
            except ValueError:
                logger.warning(
                    f'Failed to stratify splits by "{self.target_column_name}". '
                    f'Splits have no shared groups.'
                )
                train_groups, val_groups = train_test_split(
                    groups.index,
                    test_size=self.val_size,
                    random_state=random_state
                )

            train_df = train_val_df[train_val_df[self.grouping_column_name].isin(train_groups)]
            val_df = train_val_df[train_val_df[self.grouping_column_name].isin(val_groups)]

        # log stratification results
        logger.info(
            f'Mean target value among splits: '
            f'train - {train_df[self.target_column_name].mean():.2f}, '
            f'val - {val_df[self.target_column_name].mean():.2f}, '
            f'test - {test_df[self.target_column_name].mean():.2f}.'
        )

        # initialize fine-tuning datasets
        if stage == 'fit' or stage is None:
            self.train_dataset = FineTuningDataset(
                df=train_df,
                filepath_column_name=self.filepath_column_name,
                target_column_name=self.target_column_name,
                start_time_column_name=self.start_time_column_name,
                end_time_column_name=self.end_time_column_name,
                fast_mode=self.fast_mode
            )
            self.val_dataset = FineTuningDataset(
                df=val_df,
                filepath_column_name=self.filepath_column_name,
                target_column_name=self.target_column_name,
                start_time_column_name=self.start_time_column_name,
                end_time_column_name=self.end_time_column_name,
                fast_mode=self.fast_mode
            )
        if stage == 'test' or stage is None:
            self.test_dataset = FineTuningDataset(
                df=test_df,
                filepath_column_name=self.filepath_column_name,
                target_column_name=self.target_column_name,
                start_time_column_name=self.start_time_column_name,
                end_time_column_name=self.end_time_column_name,
                fast_mode=self.fast_mode
            )

        # log splits statistics
        n_samples = len(df)
        logger.info(
            f'Using dataset with following splits:\n'
            f'Train: {len(train_df)} / {n_samples} ({len(train_df) / n_samples: .4f})\n'
            f'Val: {len(val_df)} / {n_samples} ({len(val_df) / n_samples: .4f})\n'
            f'Test: {len(test_df)} / {n_samples} ({len(test_df) / n_samples: .4f})'
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            collate_fn=self.collate_fn,
            drop_last=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            collate_fn=self.collate_fn,
            drop_last=True,
        )
