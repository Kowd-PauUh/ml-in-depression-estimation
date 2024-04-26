from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()


def make_session(
    postgres_user: str, 
    postgres_password: str, 
    postgres_host: str, 
    postgres_port: int, 
    db_name: str
):
    engine = create_engine(
        f"postgresql+psycopg2://{postgres_user}:{postgres_password}" + \
        f"@{postgres_host}:{postgres_port}/{db_name}"
    )

    result = sessionmaker(bind=engine)

    return result()
