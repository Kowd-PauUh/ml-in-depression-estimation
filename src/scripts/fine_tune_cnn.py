import os
from typing import Literal
from pathlib import Path
import json

import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import fire

from src.training.foundation_models import FOUNDATION_MODELS
from src.training.fine_tuning.training_module import FineTuningTrainingModule
from src.training.fine_tuning.data_module import FineTuningDataModule
from src.training.co2_monitor import CO2Monitor
from src.training.loss_monitor import LossMonitor


MODELS_DIR = Path(os.environ.get("MODELS_DIR", "models"))


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
    chunking_strategy: Literal['truncate', 'random', 'mean'] = 'truncate',
    # training
    max_epochs: int = 10,
    min_epochs: int = 1,
    lr_reduction_factor: float = 0.5,
    lr: float = 1e-5,
    patience: int = 3,
    batch_size: int = 2,
    max_grad_norm: float = 1.0,
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

    model_name = (
        f'{cnn.__class__.__name__}_{objective}_{mel_bins=}_{augmentation=}_'
        f'{chunking_strategy=}_{lr=}_{lr_reduction_factor=}'.replace("'", '')
    ).replace('.', '_').replace('=', '_')

    # modules
    module = FineTuningTrainingModule(cnn=cnn, objective=objective)
    data_module = FineTuningDataModule(
        target_column_name=target_column_name,
        downsample_to=downsample_to, 
        batch_size=batch_size, 
        val_batch_size=batch_size,
        fast_mode=not ram_optimized_mode,
    )

    # loggers
    csv_logger = CSVLogger(save_dir=MODELS_DIR/model_name, name=None)

    # callbacks
    early_stopping = EarlyStopping(
        monitor="val_loss", mode="min", patience=patience, min_delta=1e-3, verbose=True
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

    # save training hyperparams
    train_hparams = {
        'cnn': cnn.__class__.__name__,
        'objective': objective,
        'evaluate_on_test_split': evaluate_on_test_split,
        'ram_optimized_mode': ram_optimized_mode,
        'downsample_to': downsample_to,
        'mel_bins': mel_bins,
        'augmentation': augmentation,
        'chunking_strategy': chunking_strategy,
        'max_epochs': max_epochs,
        'min_epochs': min_epochs,
        'lr_reduction_factor': lr_reduction_factor,
        'lr': lr,
        'patience': patience,
        'batch_size': batch_size,
        'max_grad_norm': max_grad_norm,
        **kwargs
    }
    Path(csv_logger.log_dir).mkdir(parents=True, exist_ok=True)
    train_hparams_path = Path(csv_logger.log_dir) / 'train_hparams.json'
    with open(train_hparams_path, 'w') as f:
        json.dump(train_hparams, f, indent=4)

    # training
    trainer = L.Trainer(
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        callbacks=[early_stopping, model_checkpoint, lr_monitor, co2_monitor, loss_monitor],
        logger=[csv_logger],
        default_root_dir=model_name,
        log_every_n_steps=1,
        gradient_clip_val=max_grad_norm,
        **kwargs,
    )

    try:
        trainer.fit(module, data_module)
    except Exception as e:
        # create empty file named "failure" on exception
        open(Path(csv_logger.log_dir) / 'failure', 'a').close()
        raise e

    # testing
    if evaluate_on_test_split:
        trainer.test(module, data_module)


def get_model(model_name, pretrained):
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
        **kwargs
    )

if __name__ == "__main__":
    fire.Fire(main)
