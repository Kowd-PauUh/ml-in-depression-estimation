from typing import  List, Callable

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency


def cramers_v(vector_a, vector_b):
    """Cramer's V test for two feature vectors"""
    confusion_matrix = pd.crosstab(vector_a, vector_b)
    chi2 = chi2_contingency(confusion_matrix)[0]
    
    n = confusion_matrix.to_numpy().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape

    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def _corr_plot(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    method: str | Callable,
):
    plt.figure(figsize=(5, 3))

    corr = df[features + [target]].corr(method=method).iloc[:-1]
    mask = np.triu(np.ones_like(corr, dtype=bool))
    mask[:, -1] = ~mask[:, -1]

    sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt='.2f')
    return plt.gcf()


def continuous_features_corr_plot(
    df: pd.DataFrame,
    features: List[str],
    target: str
):
    _corr_plot(df=df, features=features, target=target, method='pearson')
    plt.title("Pearson's correllation matrix (continuous features)")

    return plt.gcf()


def categorical_features_corr_plot(
    df: pd.DataFrame,
    features: List[str],
    target: str
):
    _corr_plot(df=df, features=features, target=target, method=cramers_v)
    plt.title("Cramer's V correllation matrix (categorical features)")
    
    return plt.gcf()