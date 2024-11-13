import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import lightning as L
import torch


class LossMonitor(L.Callback):
    def __init__(self, average_train_loss: bool = True, save_format: str = 'pdf'):
        super().__init__()
        self.average_train_loss = average_train_loss
        self.save_format = save_format
        self.artifact_path = None

    def on_train_end(self, trainer, pl_module):
        log_dir = pl_module.logger.log_dir
        df = pd.read_csv(f'{log_dir}/metrics.csv')

        # plot train loss
        df['epoch'] += 1
        _df = df[~df['train_loss_step'].isna()].copy()
        if not self.average_train_loss:
            _df['step'] = _df['step'] / trainer.num_training_batches

        sns.lineplot(
            data=_df, 
            x='epoch' if self.average_train_loss else 'step', 
            y='train_loss_step', 
            marker='o' if self.average_train_loss else None, 
            label='train_loss'
        )

        # plot val loss
        sns.lineplot(
            data=df[~df['val_loss_step'].isna()].copy(), 
            x='epoch', 
            y='val_loss_step', 
            marker='o', 
            label='val_loss'
        )

        # save plot
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend()
        plt.savefig(
            f'{log_dir}/loss_curve.{self.save_format}', 
            format=self.save_format, 
            bbox_inches='tight'
        )
        plt.close()

        self.artifact_path = f'{log_dir}/loss_curve.{self.save_format}'
