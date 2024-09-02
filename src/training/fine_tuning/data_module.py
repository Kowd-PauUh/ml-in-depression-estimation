import os
import random
from collections import defaultdict
from typing import Literal
from pathlib import Path

import lightning as L
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import GroupShuffleSplit

from ds_models.utils import compute_hash
from ds_models.waveform.utils import load_waveform, trim_waveform
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
        target_column_name: str = 'phq-binary',
        train_val_split_name: str = 'train',
        test_split_name: str = 'test',
        # preprocessing
        fast_mode: bool = True,
        downsample_to: int | None = None,
        val_size: float = 0.2,
        # training
        batch_size: int = 16,
        val_batch_size: int = 16,
        augmentation: Literal[None, 'weak', 'moderate', 'strong', 'mixed'] = None,
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

        # training
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.augmentation = augmentation

    def setup(self, stage: str = "train"):
        df = self.load_df()

        df["hash"] = df["query"].apply(compute_hash)
        df["hash_fraction"] = df["hash"].apply(lambda x: (x % 100_001) / 100_000)
        train_df = df[df["hash_fraction"] < self.train_size]
        val_df = df[
            (df["hash_fraction"] >= self.train_size)
            & (df["hash_fraction"] < self.train_size + self.val_size)
        ]
        test_df = df[df["hash_fraction"] > self.train_size + self.val_size]

        # initialize fine-tuning datasets
        self.train_dataset = FineTuningDataset(
            df=train_df,
            filepath_column_name=self.filepath_column_name,
            start_time_column_name=self.start_time_column_name,
            end_time_column_name=self.end_time_column_name,
            fast_mode=self.fast_mode
        )
        self.val_dataset = FineTuningDataset(
            df=val_df,
            filepath_column_name=self.filepath_column_name,
            start_time_column_name=self.start_time_column_name,
            end_time_column_name=self.end_time_column_name,
            fast_mode=self.fast_mode
        )
        self.test_dataset = FineTuningDataset(
            df=test_df,
            filepath_column_name=self.filepath_column_name,
            start_time_column_name=self.start_time_column_name,
            end_time_column_name=self.end_time_column_name,
            fast_mode=self.fast_mode
        )

        # log splits statistics
        n_samples = len(df)
        logger.info(
            f"Using dataset with following splits:\n"
            f"Train: {len(train_df)} / {n_samples} ({len(train_df) / n_samples: .4f})\n"
            f"Val: {len(val_df)} / {n_samples} ({len(val_df) / n_samples: .4f})\n"
            f"Test: {len(test_df)} / {n_samples} ({len(test_df) / n_samples: .4f})"
        )

    # def collate_fn(self, batch) -> tuple[list[dict[str, torch.Tensor]], torch.Tensor]:
    #     texts = [example.texts for example in batch]
    #     tokenizer_outputs = [
    #         self.tokenizer(
    #             sentence,
    #             truncation=True,
    #             padding=self.padding,
    #             return_tensors="pt",
    #             max_length=self.max_length,
    #         )
    #         for sentence in zip(*texts)
    #     ]
    #     labels = torch.tensor([example.label for example in batch])
    #     return tokenizer_outputs, labels

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
