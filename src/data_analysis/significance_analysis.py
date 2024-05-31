from typing import List, Literal

from scipy.stats import mannwhitneyu, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def boxplot_with_significance(
    data: List[List], 
    title: str, 
    ylabel: str, 
    xticklabels: List[str],
    variable_type: Literal['continuous', 'categorical']
):
    """
    Plots boxplots for given groups in `data` argument and performs
    statistical tests to assess differences among groups. Statistically
    significant differences are then then highlighted on the plot with
    *, ** or *** symbols, depending on significance level.

    Parameters
    ----------
    data : List[List]
        List of vectors (groups) to analyse.
    title : str
        Plot title.
    ylabel : str
        Plot Y-axis label.
    xticklabels : List[str]
        Groups names such that `xticklabels` and `data` lengths match.
    variable_type : Literal['continuous', 'categorical']
        Variable type (continuous or categorical) in given groups. 
        Variable type determines which statistical test will be performed.
        For continuos variable type a Mann–Whitney U test is performed.
        For categorical variable type a Chi-squared test is performed.
    """
    ax = plt.axes()
    bp = ax.boxplot(data, widths=0.6, patch_artist=True)
    ax.set_title(title, fontsize=14)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(xticklabels)
    ax.tick_params(axis='x', which='major', length=0)
    xticks = [0.5] + [x + 0.5 for x in ax.get_xticks()]
    ax.set_xticks(xticks, minor=True)
    ax.tick_params(axis='x', which='minor', length=3, width=1)

    colors = sns.color_palette('pastel')
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    plt.setp(bp['medians'], color='k')

    significant_combinations = []
    ls = list(range(1, len(data) + 1))
    combinations = [(ls[x], ls[x + y]) for y in reversed(ls) for x in range((len(ls) - y))]
    for c in combinations:
        data1 = data[c[0] - 1]
        data2 = data[c[1] - 1]
        
        # Significance
        if variable_type == 'continuous':
            _, p = mannwhitneyu(data1, data2)
        elif variable_type == 'categorical':
            categories = set(data1 + data2)
            p = chi2_contingency(
                [
                    [data1.count(c) for c in categories], 
                    [data2.count(c) for c in categories]
                ]
            ).pvalue
        else:
            raise ValueError('`variable_type` argument can take only two values: "continuous" and "categorical"')
            
        if p < 0.05:
            significant_combinations.append([c, p])

    bottom, top = ax.get_ylim()
    yrange = top - bottom

    # Significance bars
    for i, significant_combination in enumerate(significant_combinations):
        x1 = significant_combination[0][0]
        x2 = significant_combination[0][1]

        level = len(significant_combinations) - i
        bar_height = (yrange * 0.08 * level) + top
        bar_tips = bar_height - (yrange * 0.02)
        plt.plot(
            [x1, x1, x2, x2],
            [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k')

        p = significant_combination[1]
        if p < 0.001:
            sig_symbol = '***'
        elif p < 0.01:
            sig_symbol = '**'
        elif p < 0.05:
            sig_symbol = '*'
        text_height = bar_height + (yrange * 0.01)
        plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', c='k')

    bottom, top = ax.get_ylim()
    yrange = top - bottom
    ax.set_ylim(bottom - 0.02 * yrange, top)

    # Annotate sample size below each box
    for i, dataset in enumerate(data):
        sample_size = len(dataset)
        ax.text(i + 1, bottom, fr'n = {sample_size}', ha='center', size='x-small')


def feature_importance_plot(
    df: pd.DataFrame, 
    feature: str,
    target: str,
    variable_type: Literal['continuous', 'categorical']
):
    """
    Plots boxplots for `target` grouped by 'feature' and performs 
    statistical tests to assess differences among these groups. 
    Statistically significant differences are then then highlighted on the 
    plot with *, ** or *** symbols, depending on significance level.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing features and target.
    feature : str
        Name of the feature in `df` used to group target.
    target : str
        Name of the target in `df` used to assess `feature` impact on target.
    variable_type : Literal['continuous', 'categorical']
        Variable type (continuous or categorical) of the target. 
        Variable type determines which statistical test will be performed.
        Mann–Whitney U test is performed for continuos variable target.
        Chi-squared test is performed for categorical target.
    """
    feature_groups = df.groupby(feature).apply(
        lambda group: group[target].to_list(), 
        include_groups=False
    ).to_dict()
    boxplot_with_significance(
        data=list(feature_groups.values()), 
        title=f"'{target}' grouped by '{feature}'", 
        ylabel=target, 
        xticklabels=list(feature_groups.keys()),
        variable_type=variable_type
    )
