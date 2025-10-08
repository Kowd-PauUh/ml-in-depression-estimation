from typing import Dict, List, Literal

import torch
from torch import optim
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
import torchmetrics
import lightning as L
from transformers import AutoTokenizer, AutoModel


class EncoderTrainingModule(L.LightningModule):
    def __init__(
        self,
        # training configuration
        model_name_or_path: str,
        objective: Literal['classification', 'regression'],
        *,
        # model params
        prefix: str = '',
        max_length: int | None = None,
        model_init_kwargs: dict | None = None,
        # learning rate
        scheduler_type: Literal['one_cycle', 'reduce_on_plateau'] = 'one_cycle',
        lr: float = 3e-5,
        lr_reduction_factor: float = 0.5,
        lr_patience: int = 3,
    ):
        super().__init__()
        
        # initialize encoder
        model_init_kwargs = {'add_pooling_layer': False} | (model_init_kwargs or {})
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.encoder = AutoModel.from_pretrained(model_name_or_path, **model_init_kwargs)
        self.max_length = self.tokenizer.model_max_length
        self.prefix = prefix

        # initialize output layer
        n_dim = self.encoder.config.hidden_size
        self.linear = nn.Linear(n_dim, 1)

        # training configuration
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

        # learning rate
        self.scheduler_type = scheduler_type
        self.lr = lr
        self.lr_reduction_factor = lr_reduction_factor
        self.lr_patience = lr_patience

    def forward(self, sentences: list[str]):      
        # add prefixes
        sentences = [self.prefix + s for s in sentences]

        # tokenize sentences
        tokenized_sentences = self.tokenizer(
            sentences,
            truncation=True,
            padding=True,
            return_tensors='pt',
            max_length=self.max_length,
        ).to(self.device)
        
        # feed tokens through encoder
        representations = self.encoder(**tokenized_sentences).last_hidden_state[:, 0, :]
        
        # classification / regression on features from transformer
        return self.linear(representations)

    def forward_step(self, batch):
        sentences = [sentence for _, _, sentence, _ in batch]
        y = torch.tensor([target_value for *_, target_value in batch]).to(self.device)

        pred = self(sentences).squeeze()
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
        batch_size: int,
        prog_bar: bool = False
    ):
        # log loss and store its value
        self.log(
            f'{step_name}_loss', loss, prog_bar=prog_bar, 
            on_step=True, on_epoch=True, batch_size=batch_size
        )

        # log metrics and store their values
        for metric_name, metric_value in metrics.items():
            self.log(
                f'{step_name}_{metric_name}', metric_value, 
                on_step=True, on_epoch=True, batch_size=batch_size, 
                prog_bar=prog_bar
            )

    def training_step(self, batch, _):
        loss, metrics = self.forward_step(batch)
        self.store_metrics(loss=loss, metrics=metrics, step_name='train', batch_size=len(batch), prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        with torch.no_grad():
            loss, metrics = self.forward_step(batch)
            self.store_metrics(loss=loss, metrics=metrics, step_name='val', batch_size=len(batch), prog_bar=True)

    def test_step(self, batch, _):
        with torch.no_grad():
            loss, metrics = self.forward_step(batch)
            self.store_metrics(loss=loss, metrics=metrics, step_name='test', batch_size=len(batch))

    def configure_optimizers(self):
        # optimizer
        optimizer = optim.AdamW(self.parameters(), self.lr)

        # scheduler
        if self.scheduler_type == 'one_cycle':
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.lr,
                total_steps=self.trainer.estimated_stepping_batches
            )
            scheduler = {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        elif self.scheduler_type == 'reduce_on_plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                factor=self.lr_reduction_factor,
                patience=self.lr_patience,
                mode='min'
            )
            scheduler = {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
                'monitor': 'val_loss'
            }
        else:
            raise ValueError(f'Unsupported scheduler_type: {self.scheduler_type}')

        return [optimizer], [scheduler]
