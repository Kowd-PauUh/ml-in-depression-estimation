from sqlalchemy import Column, Integer, String, Boolean
from sqlalchemy.orm import relationship


from src.db.helpers import Base


class Client(Base):
    __schema__ = 'marketing_campaign'
    __tablename__ = 'clients'

    client_id = Column(Integer, primary_key=True)
    
    age = Column(Integer)
    job = Column(String(50))
    marital = Column(String(50))
    education = Column(String(50))
    default = Column(Boolean)
    balance = Column(Integer)
    housing = Column(Boolean)
    loan = Column(Boolean)
    contact = Column(String(50))
    day = Column(Integer)
    month = Column(String(50))
    duration = Column(Integer)
    campaign = Column(Integer)
    y = Column(Boolean)

    __table_args__ = {"schema": __schema__}
