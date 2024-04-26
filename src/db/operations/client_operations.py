from datetime import date

from sqlalchemy.orm import Session

from src.db.models import (
    Client
)


def create_client(
    session: Session,
    client_id: int,
    age: int,
    job: str,
    marital: str,
    education: str,
    default: bool,
    balance: int,
    housing: bool,
    loan: bool,
    contact: str,
    day: int,
    month: int,
    duration: int,
    campaign: int,
    y: bool
) -> Client:
    client = Client(
        client_id=client_id,
        age=age,
        job=job,
        marital=marital,
        education=education,
        default=default,
        balance=balance,
        housing=housing,
        loan=loan,
        contact=contact,
        day=day,
        month=month,
        duration=duration,
        campaign=campaign,
        y=y
    )
    session.add(client)

    return client
