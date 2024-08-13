from typing import List, Callable

import pandas as pd

from src.evaluation.base_evaluator import BaseEvaluator


class SklearnEvaluator(BaseEvaluator):
    def __init__(
        self, 
        model, 
        preprocessing_fn: Callable,
        inference_fn: Callable,
        X: pd.DataFrame, 
        Y: pd.DataFrame, 
        data_groups: pd.Series,
        score_fns: List[Callable]
    ):
        super().__init__()

        self.model = model
        self.preprocessing_fn = preprocessing_fn
        self.inference_fn = inference_fn
        self.data = pd.concat([X.copy(), Y.copy()], axis=1)
        self.data_groups = data_groups
        self.target = Y.columns[0]

        self.score_fns = score_fns

    def shuffle_data(self):
        self.data = self.data.sample(frac=1).reset_index(drop=True)

    def train_and_evaluate(self, train_index, test_index, **kwargs):
        metrics = {fn.__name__: [] for fn in self.score_fns}

        X_train, X_test = self.preprocessing_fn(
            data=self.data.drop(self.target, axis=1),
            train_index=train_index,
            test_index=test_index
        )

        self.model.fit(
            X_train, 
            self.data.iloc[train_index][self.target]
        )

        y_test = self.data.iloc[test_index][self.target].to_numpy()
        y_pred = self.inference_fn(X_test)

        for fn in self.score_fns:
            metrics[fn.__name__].append(fn(y_test, y_pred, **kwargs))
    
        return metrics
