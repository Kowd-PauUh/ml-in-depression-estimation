from datetime import date

from sqlalchemy.orm import Session

from src.db.models import (
    Campaign
)


def create_campaign(
    session: Session,
    client_id: int,
    contact_date: date,
    contact_outcome: str
) -> Campaign:
    campaign = Campaign(
        client_id=client_id,
        contact_date=contact_date,
        contact_outcome=contact_outcome
    )
    session.add(campaign)

    return campaign
