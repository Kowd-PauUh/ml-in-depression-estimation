import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import pytorch_lightning as pl

class LossMonitor(pl.callbacks.Callback):
    def __init__(self, metric_names, plot_by='epoch'):
        super().__init__()
        self.metric_names = metric_names
        self.plot_by = plot_by

    def on_train_end(self, trainer, pl_module):
        log_dir = pl_module.logger.log_dir
        df = pd.read_csv(f"{log_dir}/metrics.csv")

        num_steps_per_epoch = trainer.num_training_batches

        for metric_name in self.metric_names:
            _df = df[~df[metric_name].isna()].copy()

            if self.plot_by == 'step':
                _df['step'] = _df['epoch'] * num_steps_per_epoch + _df['step'] % num_steps_per_epoch
                sns.lineplot(data=_df, x='step', y=metric_name, label=metric_name)
                mean_value = _df[metric_name].mean()
                plt.axhline(mean_value, color='r', linestyle='--', label=f'{metric_name} mean')
            else:
                sns.lineplot(data=_df, x='epoch', y=metric_name, marker='o', label=metric_name)

        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend()
        plt.savefig(f"{log_dir}/loss_plot_{self.plot_by}.png")
        plt.close()
