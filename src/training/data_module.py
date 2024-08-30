import os
import random
from collections import defaultdict
from typing import Literal

import lightning as L
import pandas as pd
import torch
from loguru import logger
from minio import Minio
from sentence_transformers import InputExample
from torch.utils.data import Dataset, DataLoader

from ds_models.utils import compute_hash


class ResearchesDataModule(L.LightningDataModule):
    def __init__(
        self,
        # files location
        dataset_path: str | Path = PREPROCESSED_DATA_PATH / 'data.csv',
        split_df_path: str | Path = PREPROCESSED_DATA_PATH / 'split.csv',
        # datasets parameters 
        match_split_df_on: str = 'participant_id',
        grouping_column_name: str = 'participant_id',
        split_column_name: str = 'split',
        train_val_split_name: str = 'train',
        test_split_name: str = 'test',
        # preprocessing
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
        self.train_val_split_name = train_val_split_name
        self.test_split_name = test_split_name

        # preprocessing
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

        self.train_dataset = ResearchesDataset(
            train_df,
            query_field=self.query_field,
            document_field=self.document_field,
            query_prefix=self.query_prefix,
            document_prefix=self.document_prefix,
        )
        self.val_dataset = ResearchesDataset(
            val_df,
            query_field=self.query_field,
            document_field=self.document_field,
            query_prefix=self.query_prefix,
            document_prefix=self.document_prefix,
            samples_per_query=1,
        )
        self.test_dataset = ResearchesDataset(
            test_df,
            query_field=self.query_field,
            document_field=self.document_field,
            query_prefix=self.query_prefix,
            document_prefix=self.document_prefix,
            samples_per_query=1,
        )
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

    def load_df(self):
        logger.info(f"Using dataset at {self.dataset_path} with split data at {self.split_df_path}")

        # read dataframe and optionally downsample
        df = pd.read_csv(dataset_path)
        if downsample_to and downsample_to >= len(df):
            logger.warning(
                f'Tried to downsample dataset to {downsample_to} ' + \
                f'entries but it contains {len(df)} entries.'
            )
        elif downsample_to:
            df = df.sample(downsample_to, random_state=42)
            logger.info(f'Dataset is downsampled to {downsample_to} entries.')

        return pd.read_json(self.local_dataset_path, lines=True)

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


# class ResearchesDataset(Dataset):
#     def __init__(
#         self,
#         df,
#         query_field: str = "query",
#         document_field: str = "positive",
#         query_prefix: str = "",
#         document_prefix: str = "",
#         seed: int = 42,
#         samples_per_query: int | None = None,
#     ):
#         self.query_field = query_field
#         self.document_field = document_field
#         self.df = df[df[document_field].notna() & df[document_field].apply(lambda x: x != "")]
#         n_queries = len(self.df[query_field].unique())
#         logger.info(f"Loaded dataframe with {len(self.df)} examples and {n_queries} unique queries (document_field: "
#                     f"{self.document_field})")
#         self.query_to_samples = defaultdict(list)
#         for query, positive in zip(self.df[query_field], self.df[document_field]):
#             query = query_prefix + query
#             positive = document_prefix + positive
#             self.query_to_samples[query].append(InputExample(texts=[query, positive]))
#         self.samples = list(self.query_to_samples.values())
#         random.seed(seed)
#         self.samples_per_query = samples_per_query

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, i):
#         return random.choice(self.samples[i][: self.samples_per_query])
