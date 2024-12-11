import os
from pathlib import Path
import warnings

import pandas as pd
from pandas.errors import SettingWithCopyWarning
from loguru import logger


# data source & destination specification
PROJECT_DIR = Path(os.environ['PROJECT_DIR'])
PREPROCESSED_DATA_PATH = PROJECT_DIR / 'data/preprocessed_data'

# split params
TEST_PERCENTAGE = 0.2
RANDOM_STATE = 0

if __name__ == '__main__':
    # disable warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

    # load preprocessed dataset
    df = pd.read_csv(PREPROCESSED_DATA_PATH / 'data.csv')
    df = df.drop(['start_time', 'end_time', 'text', 'source', 'confidence', 'split'], axis=1)
    df = df.set_index('participant_id')

    # get unique participants df
    participants_df = pd.read_csv(PREPROCESSED_DATA_PATH / 'data.csv').drop_duplicates(subset='participant_id')

    # split with stratification over 'phq_binary'
    phq_binary_counts = participants_df['phq_binary'].value_counts()
    test_positive_diagnosys_ids = participants_df['participant_id'][participants_df['phq_binary'] == 1].sample(
        int(phq_binary_counts[1] * TEST_PERCENTAGE), 
        random_state=RANDOM_STATE,
    )
    test_negative_diagnosys_ids = participants_df['participant_id'][participants_df['phq_binary'] == 0].sample(
        int(phq_binary_counts[0] * TEST_PERCENTAGE), 
        random_state=RANDOM_STATE,
    )
    test_ids = pd.concat([test_positive_diagnosys_ids, test_negative_diagnosys_ids]).to_list()

    # assign splits
    df['split'] = None
    df['split'][df.index.isin(test_ids)] = 'test'
    df['split'][~df.index.isin(test_ids)] = 'train'

    # log participants recordings percentage
    original_positives_percentage = 100 * df['phq_binary'].sum() / len(df)
    train_positives_percentage = 100 * df['phq_binary'][df['split'] == 'train'].sum() / len(df[df['split'] == 'train'])
    test_positives_percentage = 100 * df['phq_binary'][df['split'] == 'test'].sum() / len(df[df['split'] == 'test'])
    logger.info(f'{original_positives_percentage = :.2f}\n{train_positives_percentage = :.2f}\n{test_positives_percentage = :.2f}')

    # save split dataset
    save_path = PREPROCESSED_DATA_PATH / 'split.csv'
    df[['split']][~df.index.duplicated(keep='first')].to_csv(save_path)
    logger.info(f'Dataset was saved to {save_path}')
