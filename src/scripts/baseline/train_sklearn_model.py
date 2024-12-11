import os
from typing import List, Literal
from pathlib import Path
from time import time

import sklearn
from sklearn.utils import all_estimators
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support, mean_squared_error, r2_score
from tqdm.auto import tqdm
import pandas as pd
import fire
import mlflow


EXPERIMENT_NAME = 'Sklearn model repeated'

PROJECT_DIR = Path(os.environ['PROJECT_DIR'])
PREPROCESSED_DATA_PATH = PROJECT_DIR / 'data/preprocessed_data'
PROCESSED_DATA_PATH = PROJECT_DIR / 'data/processed_data'

mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
timestamp = str(time()).replace('.', '')


def standardize_columns(df):
    means = df.mean()
    stds = df.std()
    return (df - means) / stds, means, stds


def train_sklearn_model(
    # training
    model_name: str,
    objective: str,
    features: List[
        Literal['Glottal', 'Phonation', 'Prosody', 'Other', 'Articulation', 'RepLearning']
    ] = ['Glottal', 'Phonation', 'Prosody', 'Other', 'Articulation'],
    tags: dict = {},
    # cross-validation
    n_folds: int = 5,
    n_repetitions: int = 2,
    # sklearn model kwargs
    **kwargs
):
    # objective-dependent params 
    if objective == 'regression':
        dependent_variable = 'phq_score'
        type_filter = 'regressor'
    elif objective == 'classification':
        dependent_variable = 'phq_binary'
        type_filter = 'classifier'
    else:
        raise ValueError(f'Supported objectives are: "regression", "classification". Got "{objective}"')

    # model
    sklearn_model = [
        estimator for estimator_name, estimator in all_estimators(type_filter=type_filter) 
        if estimator_name == model_name
    ]
    if sklearn_model:
        sklearn_model = sklearn_model[0](**kwargs)
    else:
        raise ValueError(f'Sklearn estimator "{model_name}" ({type_filter=}) does not exist.')

    # prepare splits
    df = pd.read_csv(PREPROCESSED_DATA_PATH / 'data.csv')
    split_df = pd.read_csv(PREPROCESSED_DATA_PATH / 'split.csv').set_index('participant_id')
    df['split'] = df['participant_id'].apply(lambda i: split_df.loc[i]['split'])

    # split data into train-val and test datasets
    train_val_df = df[df['split'] == 'train'].reset_index(drop=True)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)

    # get features paths
    features_paths = list(PROCESSED_DATA_PATH.rglob('*.csv'))
    features_paths = {
        path.stem.split('_')[0]: {
            p.stem.split('_')[-1]: p 
            for p in features_paths 
            if p.stem.startswith(path.stem.split('_')[0])
        } 
        for path in features_paths
        if path.stem.split('_')[0] in features
    }

    # load features and merge them horizontally
    train_val_features_df = pd.DataFrame()
    test_features_df = pd.DataFrame()

    for features_name in features_paths.keys():
        # train features
        path = features_paths[features_name]['train']
        features = pd.read_csv(path).reset_index(drop=True)
        train_val_features_df = pd.concat([train_val_features_df, features], axis=1)

        # test features
        path = features_paths[features_name]['test']
        features = pd.read_csv(path).reset_index(drop=True)
        test_features_df = pd.concat([test_features_df, features], axis=1)

    # some of the features may be missing, we fill them with train dataset median
    train_val_features_df = train_val_features_df.fillna(train_val_features_df.median())
    test_features_df = test_features_df.fillna(train_val_features_df.median())

    for seed in range(n_repetitions):
        train_hparams = {
            'model_name': model_name,
            'objective': objective,
            'n_folds': n_folds,
            'n_repetitions': n_repetitions,
            'seed': seed,
            **kwargs
        }

        # group stratified split for train and val df
        groups = train_val_df.groupby('participant_id').first()
        groups = groups.sample(frac=1, random_state=seed)

        # split groups into folds
        kfold = KFold(n_splits=n_folds, shuffle=False)
        folds = kfold.split(groups.index)

        for train_groups_idx, val_groups_idx in folds:
            # split dataset by 'participant_id' according to fold
            train_groups, val_groups = groups.index[train_groups_idx], groups.index[val_groups_idx]
            train_df = train_val_df[train_val_df['participant_id'].isin(train_groups)]
            val_df = train_val_df[train_val_df['participant_id'].isin(val_groups)]

            # train dataset
            X_train = train_val_features_df.loc[train_df.index].reset_index(drop=True)
            X_train, X_train_means, X_train_stds = standardize_columns(X_train)  # standardize
            y_train = train_val_df.loc[train_df.index][dependent_variable].to_numpy().astype(int)

            # val dataset
            X_val = train_val_features_df.loc[val_df.index].reset_index(drop=True)
            X_val = (X_val - X_train_means) / X_train_stds  # standardize using train means and stds
            y_val = train_val_df.loc[val_df.index][dependent_variable].to_numpy().astype(int)

            # test dataset
            X_test = test_features_df
            X_test = (X_test - X_train_means) / X_train_stds  # standardize using train means and stds
            y_test = test_df[dependent_variable].to_numpy().astype(int)

            # train the model
            sklearn_model.fit(X_train, y_train)
            y_train_pred = sklearn_model.predict(X_train)
            if objective == 'classification':
                train_precision, train_recall, train_f_score, _ = precision_recall_fscore_support(
                    y_train, y_train_pred, beta=2, average='binary'
                )
            elif objective == 'regression':
                train_mse = mean_squared_error(y_train, y_train_pred)
                train_r2 = r2_score(y_train, y_train_pred)

            # validate the model
            y_val_pred = sklearn_model.predict(X_val)
            if objective == 'classification':
                val_precision, val_recall, val_f_score, _ = precision_recall_fscore_support(
                    y_val, y_val_pred, beta=2, average='binary'
                )
            elif objective == 'regression':
                val_mse = mean_squared_error(y_val, y_val_pred)
                val_r2 = r2_score(y_val, y_val_pred)

            # test the model
            y_test_pred = sklearn_model.predict(X_test)
            if objective == 'classification':
                test_precision, test_recall, test_f_score, _ = precision_recall_fscore_support(
                    y_test, y_test_pred, beta=2, average='binary'
                )
            elif objective == 'regression':
                test_mse = mean_squared_error(y_test, y_test_pred)
                test_r2 = r2_score(y_test, y_test_pred)

            # log metrics to MLFlow
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(
                run_name=model_name, 
                tags={
                    'experiment_name': EXPERIMENT_NAME,
                    'cross_validation': timestamp,
                    **tags,
                }
            ):
                mlflow.log_params(train_hparams)

                if objective == 'classification':
                    mlflow.log_metric('train_precision_epoch', train_precision)
                    mlflow.log_metric('train_recall_epoch', train_recall)
                    mlflow.log_metric('train_f2_epoch', train_f2)
                    mlflow.log_metric('val_precision_epoch', val_precision)
                    mlflow.log_metric('val_recall_epoch', val_recall)
                    mlflow.log_metric('val_f2_epoch', val_f_score)
                    mlflow.log_metric('test_precision_epoch', test_precision)
                    mlflow.log_metric('test_recall_epoch', test_recall)
                    mlflow.log_metric('test_f2_epoch', test_f_score)
                elif objective == 'regression':
                    mlflow.log_metric('train_mse_epoch', train_mse)
                    mlflow.log_metric('train_r2_epoch', train_r2)
                    mlflow.log_metric('val_mse_epoch', val_mse)
                    mlflow.log_metric('val_r2_epoch', val_r2)
                    mlflow.log_metric('test_mse_epoch', test_mse)
                    mlflow.log_metric('test_r2_epoch', test_r2)


if __name__ == "__main__":
    fire.Fire(train_sklearn_model)
