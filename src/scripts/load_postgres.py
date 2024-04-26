import os
from pathlib import Path
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from src.db import make_session, create_client, create_campaign


PROJECT_DIR = Path(os.environ['PROJECT_DIR'])
POSTGRES_HOST = os.environ['POSTGRES_HOST']
POSTGRES_PORT = os.environ['POSTGRES_PORT']
POSTGRES_USER = os.environ['POSTGRES_USER']
POSTGRES_PASSWORD = os.environ['POSTGRES_PASSWORD']
DB_NAME = os.environ['POSTGRES_DB_NAME']

if __name__ == '__main__':
    prev_camp_path = PROJECT_DIR / 'data/raw_data/previous_campaigns.csv'
    curr_camp_path = PROJECT_DIR / 'data/raw_data/current_campaign.csv'

    prev_camp_df = pd.read_csv(prev_camp_path, sep=';')
    curr_camp_df = pd.read_csv(curr_camp_path, sep=';')

    # data preprocessing
    prev_camp_df['contact_date'] = prev_camp_df['contact_date'].apply(
        lambda s: datetime.strptime(s, '%Y-%m-%d').date()
    )
    bool_mapping = {
        'yes': True,
        'no': False
    }
    curr_camp_df = curr_camp_df.map(
        lambda val: bool_mapping.get(val, val) if isinstance(val, str) else val
    )

    # drop inconsistent entries
    prev_camp_df = prev_camp_df[prev_camp_df['client_id'].isin(curr_camp_df['client_id'])]

    session = make_session(
        postgres_user=POSTGRES_USER,
        postgres_password=POSTGRES_PASSWORD,
        postgres_host=POSTGRES_HOST,
        postgres_port=POSTGRES_PORT,
        db_name=DB_NAME
    )

    try:
        for _, row in tqdm(curr_camp_df.iterrows(), desc='Adding campaigns'):
            values = row.to_dict()
            values['age'] = int(values['age']) if values['age'] > 0 else None
            create_client(
                session=session,
                **values
            )
        session.commit()

        for _, row in tqdm(prev_camp_df.iterrows(), desc='Adding clients'):
            create_campaign(
                session=session,
                **row.to_dict()
            )
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()
