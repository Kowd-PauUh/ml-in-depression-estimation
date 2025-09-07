import os
from pathlib import Path

import pandas as pd


# read data
split_path = Path(os.getenv('PROJECT_DIR')) / 'data/preprocessed_data/split.csv'
data_path = Path(os.getenv('PROJECT_DIR')) / 'data/preprocessed_data/data.csv'

# merge two dataframes
data_df = pd.read_csv(data_path).drop('split', axis=1)
split_df = pd.read_csv(split_path)
merged_df = data_df.merge(split_df, on='participant_id', how='left')

# compute mean predictor baseline
merged_df['mean_prediction'] = merged_df[merged_df['split'] == 'train']['phq_score'].mean()

# compute mean absolute error
merged_df['error'] = abs(merged_df['mean_prediction'] - merged_df['phq_score'])
mae = merged_df[['split', 'error']].groupby('split').mean()['error'].to_dict()

print('Baseline mean absolute error (MAE):')
print(mae)
