from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from datetime import datetime
import json

# Use DATABASE_URL from environment or default to a local PostgreSQL instance
DATABASE_URL = "postgresql+psycopg2://admin:1234@localhost:5432/resume_db"


engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Resume(Base):
    __tablename__ = "resumes"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    email = Column(String)
    phone = Column(String)
    education = Column(JSON)  # List of education entries
    keywords = Column(JSON)   # List of keywords
    experience = Column(JSON) # List of experience roles
    years_experience = Column(Integer)
    profile_metadata = Column(JSON)   # Dict of extra metadata (scores, etc.)
    rule_score = Column(Float)
    semantic_score = Column(Float)
    combined_score = Column(Float)
    rejection_reason = Column(String)
    processed_date = Column(DateTime, default=datetime.utcnow)
    resume_link = Column(String)
    job_role = Column(String)
    required_skills = Column(JSON)
    optional_skills = Column(JSON)


def init_db():
    """Create all tables in the database."""
    Base.metadata.create_all(bind=engine)

def save_resume_to_db(profile, job_role, required_skills, optional_skills):
    db = SessionLocal()
    try:
        # Ensure all JSON fields are Python objects, not strings
        def ensure_list(val):
            if isinstance(val, str):
                try:
                    return json.loads(val)
                except Exception:
                    return []
            return val
        def ensure_dict(val):
            if isinstance(val, str):
                try:
                    return json.loads(val)
                except Exception:
                    return {}
            return val
        resume = Resume(
            name=profile.get("name"),
            email=profile.get("email"),
            phone=profile.get("phone"),
            education=ensure_list(profile.get("education")),
            keywords=ensure_list(profile.get("keywords")),
            experience=ensure_list(profile.get("experience")),
            years_experience=profile.get("years_experience"),
            profile_metadata=ensure_dict(profile.get("metadata")),
            rule_score=profile["metadata"].get("rule_score"),
            semantic_score=profile["metadata"].get("semantic_score"),
            combined_score=profile["metadata"].get("combined_score"),
            rejection_reason=profile["metadata"].get("rejection_reason"),
            processed_date=datetime.strptime(profile["metadata"].get("processed_date"), "%Y-%m-%d %H:%M:%S") if profile["metadata"].get("processed_date") else datetime.utcnow(),
            resume_link=profile["metadata"].get("resume_link"),
            job_role=job_role,
            required_skills=ensure_list(required_skills),
            optional_skills=ensure_list(optional_skills)
        )
        db.add(resume)
        db.commit()
    except Exception as e:
        print(f"Error saving to database: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    try:
        # Try to connect and create tables
        init_db()
        with engine.connect() as conn:
            print("Database connection successful!")
    except Exception as e:
        print(f"Database connection failed: {e}")
