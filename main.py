import sys
import os
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware

# Add the current directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import engine, get_db
from models import AnalysisRecord, Base
from llm.client import OpenRouterClient, OpenRouterError
from utils.noun_extractor import extract_keywords

# Create database tables if they don't exist (skip during testing)
if not os.getenv("PYTEST_CURRENT_TEST"):
    Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Jouster Backend API", description="Text analysis API using OpenRouter"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request model for the analyze endpoint
class AnalyzeRequest(BaseModel):
    text: str


# Initialize OpenRouter client
openrouter_client = OpenRouterClient()


@app.post("/analyze", response_model=dict)
def analyze_text(request: AnalyzeRequest, db: Session = Depends(get_db)):
    # Validate that the input text is a valid string with content
    if (
        not request.text
        or not isinstance(request.text, str)
        or len(request.text.strip()) == 0
    ):
        raise HTTPException(
            status_code=400, detail="Input text must be a non-empty string"
        )

    try:
        # Get analysis from OpenRouter
        ai_response = openrouter_client.analyze_text(request.text)

        keywords = extract_keywords(request.text, top_n=3)

        # Convert list to comma-separated string for storage
        topics_str = ",".join(ai_response.topics) if ai_response.topics else ""
        keywords_str = ",".join(keywords) if keywords else ""

        # Create database record
        record = AnalysisRecord(
            title=ai_response.title,
            topics=topics_str,
            sentiment=ai_response.sentiment,
            keywords=keywords_str,
        )

        db.add(record)
        db.commit()
        db.refresh(record)

        response_data = {
            "id": record.id,
            "created_date": record.created_date.isoformat(),
            "summary": ai_response.summary,
            "title": ai_response.title,
            "topics": record.topics,
            "sentiment": ai_response.sentiment,
            "keywords": record.keywords,
        }

        return response_data
    except OpenRouterError as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"OpenRouter error: {str(e)}")
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error analyzing text: {str(e)}")


@app.get("/search", response_model=List[dict])
def search_records(topic: str, db: Session = Depends(get_db)):
    # Validate that the topic is a valid string with content
    if not topic or not isinstance(topic, str) or len(topic.strip()) == 0:
        raise HTTPException(
            status_code=400, detail="Topic parameter must be a non-empty string"
        )

    try:
        search_topic = topic.lower()
        records = (
            db.query(AnalysisRecord)
            .filter(AnalysisRecord.topics.ilike(f"%{search_topic}%"))
            .all()
        )

        result = []
        for record in records:
            result.append(
                {
                    "id": record.id,
                    "created_date": record.created_date.isoformat(),
                    "title": record.title,
                    "topics": record.topics,
                    "sentiment": record.sentiment,
                    "keywords": record.keywords,
                }
            )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error searching records: {str(e)}"
        )


@app.get("/records", response_model=List[dict])
def list_all_records(db: Session = Depends(get_db)):
    try:
        records = db.query(AnalysisRecord).all()

        result = []
        for record in records:
            result.append(
                {
                    "id": record.id,
                    "created_date": record.created_date.isoformat(),
                    "title": record.title,
                    "topics": record.topics,
                    "sentiment": record.sentiment,
                    "keywords": record.keywords,
                }
            )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing records: {str(e)}")


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
