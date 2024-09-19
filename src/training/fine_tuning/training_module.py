import random
from typing import Dict, List, Tuple, Literal

import torch
from torch import optim
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models
import torchmetrics
import lightning as L

from loguru import logger

from src.waveform.augmentation import random_resample, random_gain, mixup
from src.waveform.utils import resample_waveform
from src.waveform.transformation import mel_spectrogram
from src.training.utils import repeat_tensor, truncate_or_pad, get_random_chunk


class FineTuningTrainingModule(L.LightningModule):
    def __init__(
        self,
        # training configuration
        cnn,
        objective: Literal['classification', 'regression'],
        *,
        # waveform operations
        mel_bins: int = 224,
        augmentation: Literal[None, 'weak', 'moderate', 'strong', 'mixed'] = None,
        chunking_strategy: Literal['truncate', 'random', 'mean'] = 'truncate',
        # learning rate
        lr: float = 3e-5,
        lr_reduction_factor: float = 0.5,
        lr_patience: int = 3
    ):
        super().__init__()

        # training configuration
        self.cnn = cnn
        self.mel_bins = mel_bins
        self.objective = objective
        if self.objective == 'classification':
            self.loss_fn = nn.BCEWithLogitsLoss()
            self.metrics_fns = {
                'f2': torchmetrics.FBeta(num_classes=1, beta=2.0, average='binary'),
                'precision': torchmetrics.Precision(num_classes=1, average='binary'),
                'recall': torchmetrics.Recall(num_classes=1, average='binary')
            }
        elif self.objective == 'regression':
            self.loss_fn = nn.MSELoss()
            self.metrics_fns = {
                'mae': torchmetrics.MeanAbsoluteError(),
                'r2': torchmetrics.R2Score()
            }

        # waveform operations
        self.augmentation = augmentation
        self.chunking_strategy = chunking_strategy

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

        self._validate_init()

    def _replace_last_layer(self):
        def replace_conv_layer(classifier, in_channels, out_channels=1):
            """Replaces the last Conv2d layer in a Sequential classifier with specified out_channels."""
            classifier[1] = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1))
            return classifier[:2] if len(classifier) > 2 and isinstance(classifier[2], nn.ReLU) else classifier

        def replace_linear_layer(classifier, in_features, out_features=1):
            """Replaces the last Linear layer in a Sequential classifier with specified out_features."""
            classifier[-1] = nn.Linear(in_features, out_features)
            return classifier

        def add_sigmoid(classifier):
            """Appends a Sigmoid activation to a Sequential classifier."""
            return nn.Sequential(*classifier, nn.Sigmoid())

        if isinstance(self.cnn, models.SqueezeNet):
            self.cnn.classifier = replace_conv_layer(
                classifier=self.cnn.classifier, 
                in_channels=self.cnn.classifier[1].in_channels
            )
        elif isinstance(
            self.cnn, 
            (
                models.ShuffleNetV2, models.MobileNetV2, models.MobileNetV3,
                models.MNASNet, models.EfficientNet, models.AlexNet, models.VGG
            )
        ):
            self.cnn.classifier = replace_linear_layer(
                classifier=self.cnn.classifier, 
                in_features=self.cnn.classifier[-1].in_features
            )
        elif isinstance(self.cnn, models.DenseNet):
            self.cnn.classifier = nn.Linear(self.cnn.classifier.in_features, 1)
        elif isinstance(
            self.cnn, 
            (models.ResNet, models.ResNeXt, models.Wide_ResNet, models.Inception3)
        ):
            self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 1)
        else:
            raise ValueError("Unsupported CNN architecture")

        # Add sigmoid activation if classification objective
        if self.objective == 'classification':
            if isinstance(self.cnn, (models.SqueezeNet, models.ShuffleNetV2, models.MobileNetV2,
                                    models.MobileNetV3, models.MNASNet, models.EfficientNet,
                                    models.AlexNet, models.VGG)):
                self.cnn.classifier = add_sigmoid(self.cnn.classifier)
            elif isinstance(self.cnn, models.DenseNet):
                self.cnn.classifier = nn.Sequential(self.cnn.classifier, nn.Sigmoid())
            elif isinstance(self.cnn, (models.ResNet, models.ResNeXt, models.Wide_ResNet, models.Inception3)):
                self.cnn.fc = nn.Sequential(self.cnn.fc, nn.Sigmoid())

    def _validate_init(self):
        allowed_objectives = ['classification', 'regression']
        if self.objective not in allowed_objectives:
            raise ValueError(
                f'Supported values for `objective` are '
                f'{allowed_objectives}, got "{self.objective}"'
            )

        allowed_augmentations = [None, 'weak', 'moderate', 'strong', 'mixed']
        if self.augmentation not in allowed_augmentations:
            raise ValueError(
                f'Supported values for `augmentation` are '
                f'{allowed_augmentations}, got "{self.augmentation}"'
            )

        allowed_chunk_strategies = ['truncate', 'random', 'mean']
        if self.chunking_strategy not in allowed_chunk_strategies:
            raise ValueError(
                f'Supported values for `augmentation` are '
                f'{allowed_chunk_strategies}, got "{self.chunking_strategy}"'
            )

    def _apply_augmentation(
        self,
        waveforms: List[Tuple[torch.Tensor, int]]
    ) -> List[Tuple[torch.Tensor, int]]:
        augmented_waveforms = []

        # iterate through training examples pairs and apply augmentations
        augmentation_strength = [
            None, 'weak', 'moderate', 'strong', 'mixed'
        ].index(self.augmentation)
        for i in range(0, len(waveforms), 2):
            if self.augmentation == 'mixed':
                # if augmentation strength is "mixed", randomly choose it from four levels
                augmentation_strength = random.randint(0, 3)

            waveform_1, sr_1 = waveforms[i]
            waveform_2, sr_2 = waveforms[i+1]

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
                    waveform_1 = resample_waveform(
                        waveform=waveform_1,
                        orig_sample_rate=sr_1,
                        target_sample_rate=target_sr
                    )
                    waveform_1 = resample_waveform(
                        waveform=waveform_1,
                        orig_sample_rate=sr_1,
                        target_sample_rate=target_sr
                    )
                    sr_1, sr_2 = target_sr, target_sr

                # apply mixup
                waveform_1, waveform_2 = mixup(waveform_1, waveform_2)

            augmented_waveforms += [
                (waveform_1, sr_1),
                (waveform_2, sr_2),
            ]

        return augmented_waveforms

    def _calculate_mel_spectrograms(
        self,
        waveforms: List[Tuple[torch.Tensor, int]]
    ) -> List[torch.Tensor]:
        # calculate MEL spectrograms
        mel_spectrograms = []
        for waveform, sample_rate in waveforms:
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

        return mel_spectrograms

    def _forward_pass(self, X: torch.Tensor):
        if X.shape[2] != X.shape[3]:
            raise ValueError(
                f'{self.__class__.__name__}._forward_pass can only take '
                f'tensors of same length and width. Got {X.shape = }'
            )
        return self.cnn(X)

    def _forward_pass_with_scores_averaging(self, X: torch.Tensor):
        # implementation goes here
        pass

    def forward(self, waveforms: List[Tuple[torch.Tensor, int]], eval_mode: bool = True):
        if eval_mode and self.chunking_strategy == 'random':
            raise ValueError('Random chunking is forbidden in inference mode.')

        # apply aumentations during training
        if not eval_mode:
            waveforms = self._apply_augmentation(waveforms)

        # calculate MEL spectrograms
        mel_spectrograms = self._calculate_mel_spectrograms(waveforms)

        # forward pass with scores averaging
        if self.chunking_strategy == 'mean':
            mel_spectrograms = torch.cat(mel_spectrograms)
            return self._forward_pass_with_scores_averaging(mel_spectrograms)

        # forward pass with truncation
        if self.chunking_strategy == 'truncate':
            mel_spectrograms = [
                repeat_tensor(truncate_or_pad(tensor=t, max_length=self.mel_bins))
                for t in mel_spectrograms
            ]
        elif self.chunking_strategy == 'random':
            mel_spectrograms = [
                repeat_tensor(get_random_chunk(tensor=t, chunk_length=self.mel_bins))
                for t in mel_spectrograms
            ]
        else:
            raise ValueError(f'Unsupported chunking strategy "{self.chunking_strategy}"')

        mel_spectrograms = torch.cat(mel_spectrograms)
        return self._forward_pass(mel_spectrograms)

    def forward_step(self, batch, eval_mode: bool = True):
        waveforms = [(w, sr) for w, sr, _ in batch]
        y = torch.tensor([target_value for *_, target_value in batch])

        # feed waveforms through model and calculate loss
        pred = self(waveforms=waveforms, eval_mode=eval_mode)
        loss = self.loss_fn(pred, y)

        # calculate metrics
        metrics: dict = {}
        for metric_name, metric_fn in self.metrics_fns.items():
            metrics[metric_name] = metric_fn(pred, y)

        return loss, metrics

    def store_metrics(
        self,
        loss,
        metrics: Dict[str, int | float],
        step_name: str,
        prog_bar: bool = False
    ):
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
        loss, metrics = self.forward_step(batch)
        self.store_metrics(loss=loss, metrics=metrics, step_name='val', prog_bar=True)

    def test_step(self, batch, _):
        loss, metrics = self.forward_step(batch)
        self.store_metrics(loss=loss, metrics=metrics, step_name='test')

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), self.lr)
        return {
            'optimizer': optimizer,
            'lr_scheduler': ReduceLROnPlateau(optimizer, patience=self.lr_patience, factor=0.5),
            'monitor': 'val_loss',
        }
