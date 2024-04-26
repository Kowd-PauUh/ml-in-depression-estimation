from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship


from src.db.helpers import Base


class Campaign(Base):
    __schema__ = 'marketing_campaign'
    __tablename__ = 'campaigns'

    contact_id = Column(Integer, primary_key=True)
    client_id = Column(Integer, ForeignKey('marketing_campaign.clients.client_id'))
    contact_date = Column(DateTime(timezone=False), nullable=False)
    contact_outcome = Column(String(256), nullable=False)

    __table_args__ = {"schema": __schema__}
