import os
from pathlib import Path
import warnings

import pandas as pd
from pandas.errors import SettingWithCopyWarning
import seaborn as sns
import matplotlib.pyplot as plt
from loguru import logger


EDAIC = False  # if False, only participants from DAIC-WOZ are preserved
DAIC_WOZ_PIDS = [
    302, 307, 331, 335, 346, 367, 377, 381, 382, 388, 389, 390, 395, 403,
    404, 406, 413, 417, 418, 420, 422, 436, 439, 440, 451, 458, 472, 476,
    477, 482, 483, 484, 489, 490, 492, 303, 304, 305, 310, 312, 313, 315,
    316, 317, 318, 319, 320, 321, 322, 324, 325, 326, 327, 328, 330, 333,
    336, 338, 339, 340, 341, 343, 344, 345, 347, 348, 350, 351, 352, 353,
    355, 356, 357, 358, 360, 362, 363, 364, 366, 368, 369, 370, 371, 372,
    374, 375, 376, 379, 380, 383, 385, 386, 391, 392, 393, 397, 400, 401,
    402, 409, 412, 414, 415, 416, 419, 423, 425, 426, 427, 428, 429, 430,
    433, 434, 437, 441, 443, 444, 445, 446, 447, 448, 449, 454, 455, 456,
    457, 459, 463, 464, 468, 471, 473, 474, 475, 478, 479, 485, 486, 487,
    488, 491, 300, 301, 306, 308, 309, 311, 314, 323, 329, 332, 334, 337,
    349, 354, 359, 361, 365, 373, 378, 384, 387, 396, 399, 405, 407, 408,
    410, 411, 421, 424, 431, 432, 435, 438, 442, 450, 452, 453, 461, 462,
    465, 466, 467, 469, 470, 480, 481
]

# data source & destination specification
PROJECT_DIR = Path(os.environ['PROJECT_DIR'])
RAW_DATA_PATH = PROJECT_DIR / 'data/raw_data'
PREPROCESSED_DATA_PATH = PROJECT_DIR / 'data/preprocessed_data'
PREPROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

PLOT = True  # whether to create data-related plots 

# lower and upper bounds for audio filtering
MIN_AUDIO_DURATION = 10  # [s]
MAX_AUDIO_DURATION = 30  # [s]
MIN_WORDS_CNT = 10

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

    # filter out E-DAIC participants
    if not EDAIC:
        labels = labels[labels['participant_id'].isin(DAIC_WOZ_PIDS)]

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

    # filter out E-DAIC participants
    if not EDAIC:
        transcripts = transcripts[transcripts['id'].isin(DAIC_WOZ_PIDS)]

    # obtain duration and words number for each recording
    transcripts['words_cnt'] = transcripts['text'].apply(lambda text: len(text.strip().split()))
    transcripts['duration'] = transcripts.apply(lambda row: row['end_time'] - row['start_time'], axis=1)

    # filter transcripts
    filtered_transcripts = transcripts[
        (transcripts['duration'] > MIN_AUDIO_DURATION) & 
        (transcripts['duration'] < MAX_AUDIO_DURATION) & 
        (transcripts['words_cnt'] > MIN_WORDS_CNT)
    ]
    logger.info(f'Total patients preserved: {100 * len(set(filtered_transcripts["id"].unique())) / len(labels):.2f}%')
    logger.info(f'{filtered_transcripts["duration"].sum() / 3600:.2f}h of recodrings in total')

    if PLOT:
        plots_dir = PROJECT_DIR / 'data/figures'
        os.makedirs(plots_dir, exist_ok=True)

        # plot words count histogram
        sns.histplot(filtered_transcripts['words_cnt'])
        plt.savefig(plots_dir / '00. words_cnt histogram.png')
        plt.show()

        # plot words_cnt=f(duration) scatter
        sns.scatterplot(data=filtered_transcripts, x='duration', y='words_cnt')
        plt.savefig(plots_dir / '01. words_cnt vs duration.png')
        plt.show()

        # plot preserved data distribution vs original one
        for i, label in enumerate(['gender', 'phq_binary', 'phq_score', 'pcl-c (ptsd)', 'ptsd severity'], start=2):
            plt.figure(figsize=(6, 2.5))
            sns.histplot(labels[label], bins=10, label='Original dataset')
            sns.histplot(
                labels[label][labels['participant_id'].isin(filtered_transcripts['id'].unique())],
                bins=10,
                label='Filtered dataset'
            )
            plt.legend()
            plt.savefig(plots_dir / f'0{i}. preserved {label}.png')
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
