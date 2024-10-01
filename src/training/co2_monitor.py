import os

import lightning as L
from codecarbon import EmissionsTracker
from loguru import logger


class CO2Monitor(L.Callback):
    def __init__(self):
        super().__init__()
        self.tracker = None
        self.log_dir = None

    def on_train_start(self, trainer, pl_module):
        self.log_dir = trainer.log_dir
        self.tracker = EmissionsTracker(
            log_level='warning', 
            save_to_file=True, 
            output_dir=self.log_dir,
            output_file='co2_emission.csv'
        )
        self.tracker.start()

    def on_train_end(self, trainer, pl_module):
        total_emissions = self.tracker.stop()
        logger.info(f"Total CO2 emissions: {total_emissions} kg")
