"""
Workaround for the `lightning` issue described in:
https://github.com/Lightning-AI/pytorch-lightning/issues/3228

Use `EpochLogger` as a callback for the `lightning.Trainer`
to ensure the epoch (not global step) is displayed in MLFlow 
for epoch-wise metrics. Also wrap `MLFlowLogger` with `MLFlowLoggerAdapter`
class to ensure steps are logged correctly (integers only).

Notes
-----
The callback overrides the logged "step" and sets its value to 
`lightning.Trainer.current_epoch`. `MLFlowLoggerAdapter` overrides
`MLFlowLogger.log_metric` method by casting the `step` argument to int.

Examples
--------
>>> import lightning as L
>>> from src.training.epoch_logging import EpochLogger, MLFlowLoggerAdapter
>>> epoch_logger = EpochLogger()
>>> mlflow_logger = MLFlowLogger(experiment_name="my_experiment")
>>> adapted_logger = MLFlowLoggerAdapter(mlflow_logger)
>>> trainer = L.Trainer(
...     callbacks=[epoch_logger],  # `EpochLogger` instance is a `lightning.Callback`
...     logger=adapted_logger,     # `MLFlowLoggerAdapter` instance
... )
"""
import lightning as L
from lightning.pytorch.loggers import MLFlowLogger


class EpochLogger(L.Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_train_epoch_end(self, trainer, pl_module):
        self._log_epoch(trainer, pl_module)

    def on_test_epoch_end(self, trainer, pl_module):
        self._log_epoch(trainer, pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        self._log_epoch(trainer, pl_module)

    def _log_epoch(self, trainer, pl_module):
        pl_module.log('step', trainer.current_epoch)


class MLFlowLoggerAdapter:
    def __init__(self, mlflow_logger: MLFlowLogger):
        self._mlflow_logger = mlflow_logger

    def log_metrics(self, metrics, step: int | None = None) -> None:
        step = int(step) if step is not None else None
        self._mlflow_logger.log_metrics(metrics, step)

    def __getattr__(self, name):
        return getattr(self._mlflow_logger, name)
