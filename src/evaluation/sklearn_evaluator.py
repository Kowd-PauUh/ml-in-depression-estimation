from typing import List, Callable

import pandas as pd

from src.evaluation.abs_evaluator import Evaluator


class SklearnEvaluator(Evaluator):
    def __init__(
        self, 
        model, 
        inference_fn: Callable,
        X: pd.DataFrame, 
        Y: pd.DataFrame, 
        score_fns: List[Callable]
    ):
        super().__init__()

        self.model = model
        self.inference_fn = inference_fn
        self.data = pd.concat([X.copy(), Y.copy()], axis=1)
        self.target = Y.columns[0]

        self.score_fns = score_fns

    def shuffle_data(self):
        self.data = self.data.sample(frac=1).reset_index(drop=True)

    def train_and_evaluate(self, train_index, test_index, **kwargs):
        metrics = {fn.__name__: [] for fn in self.score_fns}

        self.model.fit(
            self.data.iloc[train_index].drop(self.target, axis=1), 
            self.data.iloc[train_index][self.target]
        )

        y_test = self.data.iloc[test_index][self.target].to_numpy()
        y_pred = self.inference_fn(self.data.iloc[test_index].drop(self.target, axis=1))

        for fn in self.score_fns:
            metrics[fn.__name__].append(fn(y_test, y_pred, **kwargs))
    
        return metrics
