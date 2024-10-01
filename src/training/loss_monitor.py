import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import lightning as L


class LossMonitor(L.Callback):
    def __init__(self, average_train_loss: bool = True, save_format: str = 'pdf'):
        super().__init__()
        self.average_train_loss = average_train_loss
        self.save_format = save_format

    def on_train_end(self, trainer, pl_module):
        log_dir = pl_module.logger.log_dir
        df = pd.read_csv(f"{log_dir}/metrics.csv")

        # plot train loss
        _df = df[~df['train_loss'].isna()].copy()
        if not self.average_train_loss:
            _df['epoch'] = _df['epoch'] / trainer.num_training_batches
        else:
            _df['epoch'] += 1
            df['epoch'] += 1

        sns.lineplot(
            data=_df, 
            x='epoch', 
            y='train_loss', 
            marker='o', 
            label='train_loss'
        )

        # plot val loss
        sns.lineplot(
            data=df[~df['val_loss'].isna()].copy(), 
            x='epoch', 
            y='val_loss', 
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
            bbox_inches="tight"
        )
        plt.close()
