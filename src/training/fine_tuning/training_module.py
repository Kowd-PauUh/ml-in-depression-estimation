import random

import lightning as L
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from loguru import logger

from src.waveform.augmentation import random_resample, random_gain, mixup
from src.waveform.utils import resample_waveform
from src.waveform.transformation import mel_spectrogram


class FineTuningTrainingModule(L.LightningModule):
    def __init__(
        self,
        # training configuration
        cnn, 
        loss, 
        objective: Literal['classification', 'regression'],
        *,
        # waveform operations
        mel_bins: int = 224,
        augmentation: Literal[None, 'weak', 'moderate', 'strong', 'mixed'] = None,
        chunk_strategy: Literal['truncate', 'random', 'mean'] = 'truncate',
        # learning rate
        lr: float = 3e-5, 
        lr_reduction_factor: float = 0.5,
        lr_patience: int = 3
    ):
        super().__init__()

        # training configuration
        self.cnn = cnn
        self.mel_bins = mel_bins
        self.loss = loss
        self.objective = objective
        if self.objective == 'classification':
            self.metrics_fns: dict = {}
        elif self.objective == 'regression':
            self.metrics_fns: dict = {}

        # waveform operations
        self.augmentation = augmentation
        self.chunk_strategy = chunk_strategy

        # learning rate
        self.lr = lr
        self.lr_reduction_factor = lr_reduction_factor
        self.lr_patience = lr_patience

        # loss monitoring
        self._train_loss_vector = []
        self._val_loss_vector = []
        self._test_loss_vector = []

        # metrics monitoring
        for metric_name in self.metrics_fns:
            for step_name in ['train', 'val', 'test']:
                setattr(self, f'_{step_name}_{metric_name}_vector', [])

        self.validate_init()

    def validate_init(self):
        allowed_objectives = ['classification', 'regression']
        if self.objective not in allowed_objectives:
            raise ValueError(f'Supported values for `objective` are {allowed_objectives}, got "{self.objective}"')

        allowed_augmentations = [None, 'weak', 'moderate', 'strong', 'mixed']
        if self.augmentation not in allowed_augmentations:
            raise ValueError(f'Supported values for `augmentation` are {allowed_augmentations}, got "{self.augmentation}"')

        allowed_chunk_strategies = ['truncate', 'random', 'mean']
        if self.chunk_strategy not in allowed_chunk_strategies:
            raise ValueError(f'Supported values for `augmentation` are {allowed_chunk_strategies}, got "{self.chunk_strategy}"')

    def preprocess_batch(self, batch: List[Tuple[torch.Tensor, int, float]], eval_mode: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        augmented_batch = []

        # iterate through training examples pairs and apply augmentations
        augmentation_strength = [None, 'weak', 'moderate', 'strong', 'mixed'].index(self.augmentation)
        for i in range(0, len(batch), 2):
            if self.augmentation == 'mixed':
                # if augmentation strength is "mixed", randomly choose it from four levels
                augmentation_strength = random.randint(0, 3)

            waveform_1, sr_1, target_value_1 = batch[i]
            waveform_2, sr_2, target_value_2 = batch[i+1]

            if augmentation_strength > 0:
                # apply random gain
                waveform_1 = random_gain(waveform_1)
                waveform_2 = random_gain(waveform_2)
            if augmentation_strength > 1:
                # apply random resample with waveform truncation
                waveform_1 = random_resample(waveform=waveform_1, orig_sample_rate=sr_1, trim=True)
                waveform_2 = random_resample(waveform=waveform_2, orig_sample_rate=sr_2, trim=True)
            if augmentation_strength > 2:
                # resample waveforms if their original sample rates differ
                if sr_1 != sr_2:
                    target_sr = min(sr_1, sr_2)
                    waveform_1 = resample_waveform(waveform=waveform_1, orig_sample_rate=sr_1, target_sample_rate=target_sr)
                    waveform_1 = resample_waveform(waveform=waveform_1, orig_sample_rate=sr_1, target_sample_rate=target_sr)
                    sr_1, sr_2 = target_sr, target_sr

                # apply mixup
                waveform_1, waveform_2 = mixup(waveform_1, waveform_2)

            augmented_batch += [
                (waveform_1, sr_1, target_value_1),
                (waveform_2, sr_2, target_value_2),
            ]

        # calculate MEL spectrograms
        mel_spectrograms = []
        for waveform, sample_rate, _ in augmented_batch:
            mel_spectrograms.append(
                mel_spectrogram(
                    waveform=waveform,
                    sample_rate=sample_rate,
                    num_mel_bins=self.mel_bins,
                    length=3072,  # approximately 30s of audio
                    padding=True,
                    truncation=True
                )
            )

        return torch.cat(mel_spectrograms), torch.tensor([target_value for *_, target_value in augmented_batch]])

    def forward_step(self, batch, eval_mode: bool = True):
        X, y = self.preprocess_batch(batch, eval_mode=eval_mode)
        metrics: dict = {}

        # apply chunking strategy

        # pass input through 
        X = ...

        loss = self.loss(X)
        return loss, metrics

    def store_metrics(self, loss, metrics: Dict[str, int | float], step_name: str, prog_bar: bool = False):
        """
        Logs loss and metrics and stors them into `_{step_name}_{metric_name}_vector` attribute.

        Parameters
        ----------
        loss
            Loss.
        metrics : Dict[str, int | float]
            Dictionary with metrics values.
        step_name : str
            Step name, e.g. "train".
        prog_bar : bool, optional
            Whether to also log to progress bar.
        """
        # log loss and store its value
        getattr(self, f'_{step_name}_loss_vector').append(loss.cpu().detach())
        self.log('{step_name}_loss', loss, prog_bar=prog_bar)

        # log metrics and store their values
        for metric_name, metric_value in metrics.items():
            getattr(self, f'_{step_name}_{metric_name}_vector').append(metric_value)
            self.log(f'{step_name}_{metric_name}', metric_value, prog_bar=prog_bar)

    def training_step(self, batch, _):
        loss, metrics = self.forward_step(batch, eval_mode=False)
        self.store_metrics(loss=loss, metrics=metrics, step_name='train', prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        loss = self.forward_step(batch)
        self.store_metrics(loss=loss, metrics=metrics, step_name='val', prog_bar=True)

    def test_step(self, batch, _):
        loss = self.forward_step(batch)
        self.store_metrics(loss=loss, metrics=metrics, step_name='test')

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), self.lr)
        return {
            'optimizer': optimizer,
            'lr_scheduler': ReduceLROnPlateau(optimizer, patience=self.lr_patience, factor=0.5),
            'monitor': 'val_loss',
        }
