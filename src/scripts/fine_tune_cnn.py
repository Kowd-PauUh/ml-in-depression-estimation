import os
import sys
from typing import Literal
from pathlib import Path
import json

import torch
import lightning as L
from lightning.pytorch.loggers import CSVLogger, MLFlowLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import fire
import mlflow
import pandas as pd

from src.training.foundation_models import FOUNDATION_MODELS
from src.training.fine_tuning.training_module import FineTuningTrainingModule
from src.training.fine_tuning.data_module import FineTuningDataModule
from src.training.co2_monitor import CO2Monitor
from src.training.loss_monitor import LossMonitor
from src.training.epoch_logging import EpochLogger, MLFlowLoggerAdapter


EXPERIMENT_NAME = 'CNN fine-tuning'
PROJECT_DIR = Path(os.environ['PROJECT_DIR'])
MODELS_DIR = Path(os.environ.get("MODELS_DIR", PROJECT_DIR/ "data/models")) / 'fine_tuning'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])


def fine_tune_cnn(
    cnn,
    objective: Literal['classification', 'regression'],
    evaluate_on_test_split: bool = True,
    # dataset
    ram_optimized_mode: bool = True,
    downsample_to: int | None = None,
    # spectrogram
    mel_bins: int = 224,
    augmentation: Literal[None, 'weak', 'moderate', 'strong', 'mixed'] = None,
    train_chunking_strategy: Literal['truncate', 'random', 'mean'] = 'truncate',
    eval_chunking_strategy: Literal['truncate', 'mean'] = 'truncate',
    # training
    max_epochs: int = 10,
    min_epochs: int = 1,
    lr_reduction_factor: float = 0.5,
    lr_patience: int = 3,
    lr: float = 1e-5,
    min_delta: float = 1e-3,
    patience: int = 2,
    batch_size: int = 32,
    val_batch_size: int = 32,
    tags: dict = {},
    **kwargs
):
    if objective == 'classification':
        target_column_name = 'phq_binary'
    elif objective == 'regression':
        target_column_name = 'phq_score'
    else:
        raise ValueError(
            f'Supported values for `objective` are '
            f'["classification", "regression"], got "{objective}"'
        )

    model_name = f'{cnn.__class__.__name__}/{objective}'

    # modules
    model = FineTuningTrainingModule(
        cnn=cnn, 
        objective=objective,
        mel_bins=mel_bins,
        augmentation=augmentation,
        train_chunking_strategy=train_chunking_strategy,
        eval_chunking_strategy=eval_chunking_strategy,
        lr=lr,
        lr_reduction_factor=lr_reduction_factor,
        lr_patience=lr_patience,
    ).to(DEVICE)
    data_module = FineTuningDataModule(
        target_column_name=target_column_name,
        downsample_to=downsample_to, 
        batch_size=batch_size, 
        val_batch_size=val_batch_size,
        fast_mode=not ram_optimized_mode,
    )

    # loggers
    csv_logger = CSVLogger(save_dir=MODELS_DIR / model_name, name=None)

    # callbacks
    early_stopping = EarlyStopping(
        monitor="val_loss", mode="min", patience=patience, min_delta=min_delta, verbose=True
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    model_checkpoint = ModelCheckpoint(
        Path(csv_logger.log_dir) / 'checkpoints', 
        save_weights_only=True,
        save_top_k=1, 
        monitor="val_loss", 
        save_last=True, 
        verbose=True
    )
    model_checkpoint.CHECKPOINT_JOIN_CHAR = '_'
    model_checkpoint.CHECKPOINT_EQUALS_CHAR = '_'
    co2_monitor = CO2Monitor()
    loss_monitor = LossMonitor()
    epoch_logger = EpochLogger()

    # save training hyperparams
    train_hparams = {
        'task': 'fine-tuning',
        'cnn': cnn.__class__.__name__,
        'objective': objective,
        'evaluate_on_test_split': evaluate_on_test_split,
        'ram_optimized_mode': ram_optimized_mode,
        'downsample_to': downsample_to,
        'mel_bins': mel_bins,
        'augmentation': augmentation,
        'train_chunking_strategy': train_chunking_strategy,
        'eval_chunking_strategy': eval_chunking_strategy,
        'max_epochs': max_epochs,
        'min_epochs': min_epochs,
        'lr_reduction_factor': lr_reduction_factor,
        'lr_patience': lr_patience,
        'lr': lr,
        'min_delta': min_delta,
        'patience': patience,
        'batch_size': batch_size,
        'val_batch_size': val_batch_size,
        **tags,
        **kwargs
    }
    Path(csv_logger.log_dir).mkdir(parents=True, exist_ok=True)
    train_hparams_path = Path(csv_logger.log_dir) / 'train_hparams.json'
    with open(train_hparams_path, 'w') as f:
        json.dump(train_hparams, f, indent=4)

    # save script execution command
    execution_command_path = Path(csv_logger.log_dir) / 'command.txt'
    with open(execution_command_path, 'w') as f:
        f.write(' '.join(sys.argv))

    # save model signature
    model_signature_path = Path(csv_logger.log_dir) / 'model_signature.txt'
    with open(model_signature_path, 'w') as f:
        f.write(str(model))

    # mlflow run and logger setup
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(EXPERIMENT_NAME)
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.start_run(
        run_name=model_name, 
        tags={
            'experiment_name': EXPERIMENT_NAME,
            'cnn': cnn.__class__.__name__,
            'objective': objective,
            'augmentation': str(augmentation),
            'train_chunking_strategy': train_chunking_strategy,
            'eval_chunking_strategy': eval_chunking_strategy,
            **tags
        }
    )
    mlflow_logger = MLFlowLoggerAdapter(
        mlflow_logger=MLFlowLogger(
            experiment_name=EXPERIMENT_NAME,
            run_id=mlflow.active_run().info.run_id
        )
    )  # this is a workaround for epochs logging

    # trainer
    trainer = L.Trainer(
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        callbacks=[
            early_stopping, 
            model_checkpoint, 
            lr_monitor, 
            co2_monitor, 
            loss_monitor,
            epoch_logger,
        ],
        logger=[csv_logger, mlflow_logger],
        default_root_dir=model_name,
        log_every_n_steps=1,
        **kwargs,
    )

    # log hyperparams and dataset info
    mlflow.log_params(train_hparams)
    for context, data_source in [
        ('Labelled dataset', data_module.dataset_path), 
        ('Split dataset', data_module.split_df_path)
    ]:
        if isinstance(data_source, Path):
            data_source = data_source.as_posix()

        dataset_tags = {'local_dataset_path': data_source, 'type': context}
        mlflow.log_input(
            dataset=mlflow.data.from_pandas(
                df=pd.read_csv(data_source),
                source=mlflow.data.dataset_source.DatasetSource.from_dict(dataset_tags),
                name=Path(data_source).stem,
            ),
            context=context,
            tags=dataset_tags,
        )

    try:
        # training
        trainer.fit(model, data_module)

        # testing
        if evaluate_on_test_split:
            trainer.test(model, data_module)

        # log artifacts
        mlflow.log_artifact(model_signature_path)
        mlflow.log_artifact(execution_command_path)
        mlflow.log_artifact(co2_monitor.artifact_path)
        mlflow.log_artifact(loss_monitor.artifact_path)
        mlflow.log_artifact(train_hparams_path.as_posix())
        mlflow.log_artifact(Path(csv_logger.log_dir, 'metrics.csv').as_posix())

    except Exception as e:
        # create empty file named "failure" on exception
        open(Path(csv_logger.log_dir) / 'failure', 'a').close()
        raise e


def get_model(model_name: str, pretrained: bool):
    for size_category in FOUNDATION_MODELS.values():
        if model_name in size_category:
            return size_category[model_name](pretrained=pretrained)

    raise ValueError(f'Model "{model_name}" is not supported')


def main(
    model_name: str,
    objective: str,
    pretrained: bool = True,
    **kwargs
):
    cnn = get_model(model_name, pretrained)
    fine_tune_cnn(
        cnn=cnn,
        objective=objective,
        tags={
            'model_name': model_name,
            'pretrained': str(pretrained),
        },
        **kwargs
    )

if __name__ == "__main__":
    fire.Fire(main)
