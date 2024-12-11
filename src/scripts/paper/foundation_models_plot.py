import os
from pathlib import Path
from copy import deepcopy

from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from src.training.foundation_models import FOUNDATION_MODELS


# set plot save dir
PROJECT_DIR = Path(os.environ['PROJECT_DIR'])
PAPER_DIR = PROJECT_DIR / 'paper'


if __name__ == '__main__':
    # count each model parameters
    sizes = deepcopy(FOUNDATION_MODELS)
    for size in tqdm(FOUNDATION_MODELS, desc='Processing models'):
        for model_name in tqdm(FOUNDATION_MODELS[size], leave=False):
            model = FOUNDATION_MODELS[size][model_name]()
            sizes[size][model_name] = sum(p.numel() for p in model.parameters())

    colors = {
        'tiny': 'green', 
        'small': 'blue', 
        'medium': 'orange', 
        'large': 'red', 
        'huge': 'purple'
    }

    model_names = []
    sizes_millions = []
    color_list = []

    for size_group, models in sizes.items():
        for model, size in models.items():
            model_names.append(model)
            sizes_millions.append(size / 1e6)
            color_list.append(colors[size_group])

    # create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(sizes_millions, model_names, c=color_list)

    # add labels
    for i, size in enumerate(sizes_millions):
        ax.text(size + 1, i, f"{size:.1f}M", fontsize=9, va='center')

    # set grid, labels, and title
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    ax.set_xlabel('Model Parameters (in Millions)', fontsize=12)

    legend_handles = [
        plt.Line2D(
            [0], [0], marker='o', color='w', 
            markerfacecolor=colors[key], markersize=10, 
            label=key.capitalize()
        ) 
        for key in colors
    ]
    ax.legend(handles=legend_handles, title="Model Size")
    ax.spines[['right', 'top']].set_visible(False)

    # save plot
    plt.savefig(PAPER_DIR / 'figures/foundation-models-by-size.pdf', bbox_inches='tight')
