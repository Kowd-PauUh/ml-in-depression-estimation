import os
from pathlib import Path
from typing import Tuple
import warnings

import pandas as pd
from pandas.errors import SettingWithCopyWarning
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from loguru import logger


# data source & destination specification
PROJECT_DIR = Path(os.environ['PROJECT_DIR'])
RAW_DATA_PATH = PROJECT_DIR / 'data/raw_data'
PREPROCESSED_DATA_PATH = PROJECT_DIR / 'data/preprocessed_data'
PREPROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

PLOT = False  # whether to create data-related plots 

# lower and upper bounds for audio filtering
MIN_AUDIO_DURATION = 10  # [s]
MAX_AUDIO_DURATION = 30  # [s]

if __name__ == '__main__':
    # disable warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

    # gather split files and combine them into single one
    lables_paths = RAW_DATA_PATH.rglob('*split.csv')
    labels = pd.DataFrame({'split': []})
    for path in sorted(lables_paths):
        labels = pd.concat([labels, pd.read_csv(path)], axis=0)
        labels.fillna(path.stem.split('_')[0], inplace=True)
    labels.columns = [c.lower() for c in labels.columns]

    # fill missing values (there are two of them) in the 'PTSD Severity' column with its median depending on binary  PCL-C score
    labels['ptsd severity'][labels['ptsd severity'] == 'test'] = labels[labels['ptsd severity'] == 'test'].apply(
        lambda row: labels['ptsd severity'][(labels['ptsd severity'] != 'test') & (labels['pcl-c (ptsd)'] == row['pcl-c (ptsd)'])].median(),
        axis=1
    )

    # clean the 'gender' label
    labels['gender'] = labels['gender'].apply(str.strip)

    # log unique labels
    for label in ['gender', 'phq_binary', 'phq_score', 'pcl-c (ptsd)', 'ptsd severity']:
        logger.info(f'{label}: {labels[label].unique()}')

    # gather transcripts
    transcripts_paths = RAW_DATA_PATH.rglob('*Transcript.csv')
    transcripts = pd.DataFrame({'id': []})
    for path in sorted(transcripts_paths):
        transcripts = pd.concat([transcripts, pd.read_csv(path)], axis=0)
        transcripts.fillna(int(path.stem.split('_')[0]), inplace=True)
    transcripts.columns = [c.lower() for c in transcripts.columns]

    # obtain duration and words number for each recording
    transcripts['words_cnt'] = transcripts['text'].apply(lambda text: len(text.strip().split()))
    transcripts['duration'] = transcripts.apply(lambda row: row['end_time'] - row['start_time'], axis=1)

    # filter transcripts
    filtered_transcripts = transcripts[
        (transcripts['duration'] > MIN_AUDIO_DURATION) & 
        (transcripts['duration'] < MAX_AUDIO_DURATION) & 
        (transcripts['words_cnt'] > 10)
    ]
    logger.info(f'Total patients preserved: {100 * len(set(filtered_transcripts["id"].unique())) / len(labels):.2f}%')
    logger.info(f'{filtered_transcripts["duration"].sum() / 3600:.2f}h of recodrings in total')

    if PLOT:
        # plot words count histogram
        sns.histplot(filtered_transcripts['words_cnt'])
        plt.show()

        # plot words_cnt=f(duration) scatter
        sns.scatterplot(data=filtered_transcripts, x='duration', y='words_cnt')
        plt.show()

        # plot preserved data distribution vs original one
        for label in ['gender', 'phq_binary', 'phq_score', 'pcl-c (ptsd)', 'ptsd severity']:
            plt.figure(figsize=(6, 2.5))
            sns.histplot(labels[label], bins=10, label='Original dataset')
            sns.histplot(
                labels[label][labels['participant_id'].isin(filtered_transcripts['id'].unique())],
                bins=10,
                label='Filtered dataset'
            )
            plt.legend()
            plt.show()

    # create final df
    df = filtered_transcripts.merge(
        labels, 
        left_on='id', 
        right_on='participant_id', 
        validate='m:1'
    ).reset_index(drop=True)
    df.drop('id', axis=1, inplace=True)
    df['source'] = df['participant_id'].apply(
        lambda p_id: (RAW_DATA_PATH / f'{int(p_id)}_P/{int(p_id)}_AUDIO.wav').as_posix()
    )

    # log splits statistics
    logger.info(f'Final df split statistics:\n{df.split.value_counts() / len(df)}')

    # save dataset
    save_path = PREPROCESSED_DATA_PATH / 'data.csv'
    df.to_csv(save_path, index=False)
    logger.info(f'Dataset was saved to {save_path}')
