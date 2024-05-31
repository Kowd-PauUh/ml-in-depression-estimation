from typing import List, Callable

import pandas as pd
from sklearn.model_selection import KFold
from tqdm.auto import tqdm, trange


def cross_validate_model(
    model,
    X: pd.DataFrame,
    Y: pd.DataFrame,
    n_folds: int,
    iterations: int,
    score_fns: List[Callable],
    **kwargs
):
    metrics = {fn.__name__: [] for fn in score_fns}
    data = pd.concat([X.copy(), Y.copy()], axis=1)
    target = Y.columns[0]
    
    n_splits = 5
    kf = KFold(n_splits=n_folds)
    for _ in trange(5, desc='Cross validation'):
        shuffle = data.sample(frac=1).reset_index(drop=True)
        for train_index, test_index in tqdm(list(kf.split(shuffle)), desc='Splits evaluation', leave=False):
            model.fit(
                shuffle.iloc[train_index].drop(target, axis=1), 
                shuffle.iloc[train_index][target]
            )

            y_test = shuffle.iloc[test_index][target].to_numpy()
            y_pred = model.predict(shuffle.iloc[test_index].drop(target, axis=1))

            for fn in score_fns:
                metrics[fn.__name__].append(fn(y_test, y_pred, **kwargs))
    
    return metrics
