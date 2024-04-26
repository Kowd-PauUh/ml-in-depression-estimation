import os

import psycopg2
import pandas as pd


conn = psycopg2.connect(
    dbname=os.environ['POSTGRES_DB_NAME'], 
    user=os.environ['POSTGRES_USER'], 
    password=os.environ['POSTGRES_PASSWORD'], 
    host=os.environ['POSTGRES_HOST'], 
    port=os.environ['POSTGRES_PORT']
)
try:
    for table_name in ['campaigns', 'clients']:
        pd.read_sql(
            f"""
            SELECT * 
            FROM marketing_campaign.{table_name}
            """,
            conn
        )
        print(f'{table_name} is available')
except Exception as e:
    print(f'Database is not abailable: {e}')