from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from pydantic import BaseModel
from typing import List, Optional

Base = declarative_base()


class AnalysisRecord(Base):
    __tablename__ = "analysis_records"

    id = Column(Integer, primary_key=True, index=True)
    created_date = Column(DateTime(timezone=True), server_default=func.now())
    title = Column(String, nullable=True)
    topics = Column(
        String, nullable=False, default=""
    )  # USe string instead of Array just for simple usage case for this demo
    sentiment = Column(String, nullable=True)  # positive, neutral, negative
    keywords = Column(String, nullable=False, default="")


class AnalysisRecordCreate(BaseModel):
    title: Optional[str] = None
    topics: List[str]
    sentiment: Optional[str] = None
    keywords: List[str]


class AnalysisRecordResponse(AnalysisRecordCreate):
    id: int
    created_date: str
