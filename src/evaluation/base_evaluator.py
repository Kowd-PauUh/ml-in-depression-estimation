from abc import ABC
from typing import List, Callable

import pandas as pd
from sklearn.model_selection import GroupKFold
from tqdm.auto import tqdm, trange


class BaseEvaluator(ABC):
    def shuffle_data(self, *args, **kwargs):
        raise NotImplementedError

    def train_and_evaluate(self, *args, **kwargs):
        raise NotImplementedError

    def cross_validation(
        self,
        n_folds: int,
        iterations: int
    ):
        """
        Performs cross-validation of the given model and evaluator.

        Parameters
        ----------
        evaluator
            Object with implemented `shuffle_data` and `train_and_evaluate` methods.
            `train_and_evaluate` method must return a dictionary with metrics values, for
            example {'f1_score': 0.67459, 'recall': 0.50334}
        n_folds : int
            Folds number.
        iterations : int
            Number of cross-validations to perform, shuffling data before each iteration.

        Returns
        -------
        dict
            Cross-validation results.
        """
        crossval_results = {}
        
        kf = GroupKFold(n_splits=n_folds)
        for _ in trange(iterations, desc='Cross validation'):
            self.shuffle_data()
            
            for train_index, test_index in tqdm(
                list(kf.split(self.data, groups=self.data_groups)), 
                desc='Splits evaluation', 
                leave=False
            ):
                metrics = self.train_and_evaluate(train_index, test_index)

                for metric_name, metric_value in metrics.items():
                    if metric_name not in crossval_results:
                        crossval_results[metric_name] = []
                    crossval_results[metric_name].append(metric_value)
        
        return crossval_results

