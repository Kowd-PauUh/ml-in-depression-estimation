"""
Workaround for the `lightning` issue described in:
https://github.com/Lightning-AI/pytorch-lightning/issues/3228

Use `EpochLogger` as a callback for the `lightning.Trainer`
to ensure the epoch (not global step) is displayed in MLFlow 
for epoch-wise metrics.

Notes
-----
The callback overrides the logged "step" and sets its value to 
`lightning.Trainer.current_epoch`.

Exapmles
--------
>>> import lightning as L
>>> from src.training.epoch_logger import EpochLogger
>>> epoch_logger = EpochLogger()
>>> trainer = L.Trainer(
...     callbacks=[epoch_logger],  # `EpockLogger` instance is a `lightning.Callback`
... )
"""
import lightning as L


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
        step = trainer.current_epoch
        step = int(step) if step is not None else None
        pl_module.log('step', step)
