import os
from pathlib import Path

import lightning as L
from codecarbon import EmissionsTracker
from loguru import logger


class CO2Monitor(L.Callback):
    def __init__(self, output_file_name: str = 'co2_emission.csv'):
        super().__init__()

        self.output_file_name = output_file_name
        self.tracker = None
        self.log_dir = None
        self.artifact_path = None
        self.total_emissions = None

    def on_train_start(self, trainer, pl_module):
        self.log_dir = trainer.log_dir
        self.tracker = EmissionsTracker(
            log_level='warning', 
            save_to_file=True, 
            output_dir=self.log_dir,
            output_file=self.output_file_name
        )
        self.tracker.start()

    def on_train_end(self, trainer, pl_module):
        self.total_emissions = self.tracker.stop()
        self.artifact_path = Path(self.log_dir, self.output_file_name).as_posix()
        logger.info(f"Total CO2 emissions: {self.total_emissions} kg")
