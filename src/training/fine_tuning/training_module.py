import random
from typing import Dict, List, Tuple, Literal
import math

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
        train_chunking_strategy: Literal['truncate', 'random', 'mean', 'gru', 'transformer'] = 'mean',
        eval_chunking_strategy: Literal['truncate', 'mean', 'gru', 'transformer'] = 'mean',
        # learning rate
        lr: float = 3e-5,
        lr_reduction_factor: float = 0.5,
        lr_patience: int = 3,
        # optimizer
        optimizer: Literal['AdamW', 'SGD'] = 'AdamW',
        # compute
        compute_batch_size: int = 256,
    ):
        super().__init__()

        # prepare CNN for either regression / binary classification objective
        # or features extraction for hybrid architectures
        self.cnn = cnn
        cnn_features = 1 if train_chunking_strategy not in ['gru', 'transformer'] else 512
        self._replace_last_cnn_layer(out_features=cnn_features)

        # training configuration
        self.mel_bins = mel_bins
        self.objective = objective
        if self.objective == 'classification':
            self.loss_fn = nn.BCEWithLogitsLoss()
            self.metrics_fns = {
                'f2': torchmetrics.classification.BinaryFBetaScore(beta=2.0),
                'precision': torchmetrics.classification.BinaryPrecision(),
                'recall': torchmetrics.classification.BinaryRecall()
            }
        elif self.objective == 'regression':
            self.loss_fn = nn.MSELoss()
            self.metrics_fns = {
                'mae': torchmetrics.MeanAbsoluteError(),
                'r2': torchmetrics.R2Score()
            }

        # waveform operations
        self.augmentation = augmentation
        self.train_chunking_strategy = train_chunking_strategy
        self.eval_chunking_strategy = eval_chunking_strategy

        # learning rate
        self.lr = lr
        self.lr_reduction_factor = lr_reduction_factor
        self.lr_patience = lr_patience

        # optimizer
        self.optimizer = optimizer

        # compute
        self.compute_batch_size = compute_batch_size

        self._validate_init()

    def _replace_last_cnn_layer(self, out_features: int = 1):
        # models with a sequentional layer (e.g., VGG, AlexNet)
        if hasattr(self.cnn, 'classifier') and isinstance(self.cnn.classifier, nn.Sequential):
            # find the last linear layer in the classifier
            for idx in reversed(range(len(self.cnn.classifier))):
                layer = self.cnn.classifier[idx]

                # models with linear classifiers
                if isinstance(layer, nn.Linear):
                    in_features = layer.in_features

                    # replace the last linear layer
                    self.cnn.classifier[idx] = nn.Linear(in_features, out_features)

                    # remove any layers after the replaced linear layer
                    if idx < len(self.cnn.classifier) - 1:
                        self.cnn.classifier = nn.Sequential(*self.cnn.classifier[:idx+1])
                    break

                # models with convolutional classifiers
                if isinstance(layer, nn.Conv2d):
                    in_channels = layer.in_channels

                    # replace the last Conv2d layer
                    self.cnn.classifier[idx] = nn.Conv2d(in_channels, out_features, kernel_size=1)

                    # remove any layers after the replaced Conv2d layer
                    if idx < len(self.cnn.classifier) - 1:
                        self.cnn.classifier = nn.Sequential(*self.cnn.classifier[:idx+1])
                    break

            else:
                raise ValueError("Neither linear nor Conv2d layer found in self.cnn.classifier.")

        # models with a linear layer (e.g., Densenet)
        elif hasattr(self.cnn, 'classifier') and isinstance(self.cnn.classifier, nn.Linear):
            self.cnn.classifier = nn.Linear(self.cnn.classifier.in_features, out_features)

        # models with a fully connected layer (e.g., ResNet)
        elif hasattr(self.cnn, 'fc') and isinstance(self.cnn.fc, nn.Linear):
            in_features = self.cnn.fc.in_features
            self.cnn.fc = nn.Linear(in_features, out_features)
        else:
            raise ValueError(f"Unsupported CNN architecture: {self.cnn.__class__.__name__}")

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

        allowed_train_chunk_strategies = ['truncate', 'random', 'mean', 'gru', 'transformer']
        if self.train_chunking_strategy not in allowed_train_chunk_strategies:
            raise ValueError(
                f'Supported values for `train_chunking_strategy` are '
                f'{allowed_train_chunk_strategies}, got "{self.train_chunking_strategy}"'
            )

        allowed_eval_chunk_strategies = ['truncate', 'mean', 'gru', 'transformer']
        if self.eval_chunking_strategy not in allowed_eval_chunk_strategies:
            raise ValueError(
                f'Supported values for `eval_chunking_strategy` are '
                f'{allowed_eval_chunk_strategies}, got "{self.eval_chunking_strategy}"'
            )

        chunking_strategy_mismatch_e = ValueError(
            f'When instantiating {self.__class__.__name__} in hybrid setup '
            f'train and eval chunking strategies must match. Got '
            f'train_chunking_strategy={self.train_chunking_strategy}, '
            f'eval_chunking_strategy={self.eval_chunking_strategy}.'
        )
        if self.train_chunking_strategy != self.eval_chunking_strategy:
            if self.train_chunking_strategy in ['gru', 'transformer']:
                raise chunking_strategy_mismatch_e
            if self.eval_chunking_strategy in ['gru', 'transformer']:
                raise chunking_strategy_mismatch_e

        allowed_optimizers = ['AdamW', 'SGD']
        if self.optimizer not in allowed_optimizers:
            raise ValueError(
                f'Supported values for `optimizer` are '
                f'{allowed_optimizers}, got "{self.optimizer}"'
            )

        # save hyperparams
        self.save_hyperparameters(
            'objective', 'mel_bins', 
            'augmentation', 'train_chunking_strategy', 
            'eval_chunking_strategy', 'lr_patience',
            'lr', 'lr_reduction_factor', 'optimizer'
        )
        logger.info(
            f'Training {self.cnn.__class__.__name__} with {self.objective} objective '
            f'and {self.optimizer} optimizer '
            f'(mel_bins = {self.mel_bins}, augmentation = {self.augmentation}, '
            f'train_chunking_strategy = {self.train_chunking_strategy}, '
            f'eval_chunking_strategy = {self.eval_chunking_strategy}, lr = {self.lr}, '
            f'lr_reduction_factor = {self.lr_reduction_factor}, lr_patience = {self.lr_patience})'
        )

    def _apply_augmentation(
        self,
        waveforms: List[Tuple[torch.Tensor, int]]
    ) -> List[Tuple[torch.Tensor, int]]:
        # identity if no augmentation
        if self.augmentation is None:
            return waveforms

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
                    padding=False,
                    truncation=False
                )
            )

        return mel_spectrograms

    def _forward_pass(self, X: torch.Tensor):
        if X.shape[2] != X.shape[3]:
            raise ValueError(
                f'{self.__class__.__name__}._forward_pass can only take '
                f'tensors of same length and width. Got {X.shape = }'
            )
        return self.cnn(X.to(self.device))

    def _forward_pass_with_chunking(self, mel_spectrograms: List[torch.Tensor]):
        all_chunks = []
        chunk_lengths = []

        # for each spectrogram
        for mel_spectrogram in mel_spectrograms:
            # obtain number of chunks
            mel_chunks_cnt = mel_spectrogram.shape[1] // self.mel_bins
            chunk_lengths.append(mel_chunks_cnt)
            
            # split waveform tensor into chunks
            # drop last chunk if it's shorter than `self.mel_bins`
            if mel_spectrogram.shape[1] % self.mel_bins == 0:
                all_chunks += torch.split(mel_spectrogram, self.mel_bins, dim=1)
            else:
                all_chunks += torch.split(mel_spectrogram, self.mel_bins, dim=1)
                all_chunks.pop(-1)  

        # repeat chunks in 3 dimensions
        all_chunks = [repeat_tensor(chunk) for chunk in all_chunks]
        all_chunks = torch.stack(all_chunks, dim=0)

        # feed through features extractor using `self.compute_batch_size`
        forward_pass_result = []
        for i in range(math.ceil(len(all_chunks) / self.compute_batch_size)):
            forward_pass_result.append(
                self._forward_pass(
                    all_chunks[i*self.compute_batch_size:(i + 1)*self.compute_batch_size]
                )
            )

        # combine the results
        forward_pass_result = torch.cat(forward_pass_result, axis=0)
        forward_pass_result_per_chunk = torch.split(forward_pass_result, chunk_lengths)

        return forward_pass_result_per_chunk

    def forward(self, waveforms: List[Tuple[torch.Tensor, int]], eval_mode: bool = True):
        chunking_strategy = self.eval_chunking_strategy if eval_mode else self.train_chunking_strategy

        # apply aumentations during training
        if not eval_mode:
            waveforms = self._apply_augmentation(waveforms)

        # calculate MEL spectrograms
        mel_spectrograms = self._calculate_mel_spectrograms(waveforms)

        # forward pass with scores averaging
        if chunking_strategy == 'mean':
            # feed with chunking through CNN
            scores = self._forward_pass_with_chunking(mel_spectrograms)

            # average scores
            return torch.cat(
                [
                    torch.mean(chunk_scores, dim=0, keepdim=True) 
                    for chunk_scores in scores
                ]
            )

        # forward pass on single chunk
        if chunking_strategy == 'truncate':
            mel_spectrograms = [
                repeat_tensor(truncate_or_pad(tensor=t, max_length=self.mel_bins))
                for t in mel_spectrograms
            ]
        elif chunking_strategy == 'random':
            mel_spectrograms = [
                repeat_tensor(get_random_chunk(tensor=t, chunk_length=self.mel_bins))
                for t in mel_spectrograms
            ]
        else:
            raise ValueError(f'Unsupported chunking strategy "{chunking_strategy}"')

        mel_spectrograms = torch.stack(mel_spectrograms, dim=0)
        return self._forward_pass(mel_spectrograms)

    def forward_step(self, batch, eval_mode: bool = True):
        waveforms = [(w, sr) for w, sr, _ in batch]
        y = torch.tensor([target_value for *_, target_value in batch]).to(self.device)

        # feed waveforms through model and calculate loss
        pred = self(waveforms=waveforms, eval_mode=eval_mode).squeeze()
        loss = self.loss_fn(pred, y)

        # apply sigmoid for classification metrics computation
        if self.objective == 'classification':
            pred = torch.sigmoid(pred)

        # calculate metrics
        metrics: dict = {}
        for metric_name, metric_fn in self.metrics_fns.items():
            metric_fn.to(self.device)
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
        Logs loss and metrics.

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
        self.log(
            f'{step_name}_loss', loss, prog_bar=prog_bar, 
            on_step=True, on_epoch=True
        )

        # log metrics and store their values
        for metric_name, metric_value in metrics.items():
            self.log(
                f'{step_name}_{metric_name}', metric_value, 
                on_step=True, on_epoch=True, 
                prog_bar=prog_bar
            )

    def training_step(self, batch, _):
        loss, metrics = self.forward_step(batch, eval_mode=False)
        self.store_metrics(loss=loss, metrics=metrics, step_name='train', prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        with torch.no_grad():
            loss, metrics = self.forward_step(batch)
            self.store_metrics(loss=loss, metrics=metrics, step_name='val', prog_bar=True)

    def test_step(self, batch, _):
        with torch.no_grad():
            loss, metrics = self.forward_step(batch)
            self.store_metrics(loss=loss, metrics=metrics, step_name='test')

    def configure_optimizers(self):
        if self.optimizer == 'AdamW':
            optimizer = optim.AdamW(self.parameters(), self.lr)
        elif self.optimizer == 'SGD':
            optimizer = optim.SGD(self.parameters(), self.lr)
        else:
            raise ValueError(f'Unsupported optimizer: {self.optimizer}')

        return {
            'optimizer': optimizer,
            'lr_scheduler': ReduceLROnPlateau(optimizer, patience=self.lr_patience, factor=0.5),
            'monitor': 'val_loss',
        }
