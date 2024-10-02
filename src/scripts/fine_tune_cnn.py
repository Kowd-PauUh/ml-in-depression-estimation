import os
from pathlib import Path

import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

from src.training.foundation_models import FOUNDATION_MODELS
from src.training.fine_tuning.training_module import FineTuningTrainingModule
from src.training.fine_tuning.data_module import FineTuningDataModule
from src.training.co2_monitor import CO2Monitor
from src.training.loss_monitor import LossMonitor


MODELS_DIR = Path(os.environ.get("MODELS_DIR", "models"))


def fine_tune_cnn(
    cnn,
    objective,
    max_epochs,
    min_epochs,
    test=True,
    ram_optimized_mode=True,
    mel_bins=224,
    augmentation=None,
    chunking_strategy='truncate',
    lr_reduction_factor=0.5,
    lr=1e-5,
    patience=3,
    batch_size = 2,
    downsample_to: int | None = None,
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
            f'["classification", "regression"], got "{self.objective}"'
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
    trainer.fit(module, data_module)

    # testing
    if test:
        trainer.test(module, data_module)
