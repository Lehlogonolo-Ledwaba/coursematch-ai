from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import httpx
import json
import os

app = FastAPI(title="CourseMatch AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

COURSES = [
    {"id": "c1",  "title": "Machine Learning Specialization",     "provider": "Stanford / DeepLearning.AI", "level": "Intermediate",         "topics": ["Machine Learning", "Math & Stats", "Data Science", "AI & NLP"]},
    {"id": "c2",  "title": "Google Data Analytics Certificate",   "provider": "Google / Coursera",          "level": "Beginner",             "topics": ["Data Science", "Business", "Math & Stats"]},
    {"id": "c3",  "title": "Full-Stack Web Development Bootcamp", "provider": "The App Brewery",            "level": "Beginner-Intermediate", "topics": ["Web Development", "Mobile Apps"]},
    {"id": "c4",  "title": "Deep Learning Specialization",        "provider": "DeepLearning.AI",            "level": "Advanced",             "topics": ["Machine Learning", "AI & NLP", "Math & Stats"]},
    {"id": "c5",  "title": "Google UX Design Certificate",        "provider": "Google / Coursera",          "level": "Beginner",             "topics": ["Design", "Business"]},
    {"id": "c6",  "title": "AWS Cloud Practitioner",              "provider": "Amazon Web Services",        "level": "Beginner",             "topics": ["Cloud Computing", "Cybersecurity"]},
    {"id": "c7",  "title": "IBM Cybersecurity Analyst",           "provider": "IBM / Coursera",             "level": "Intermediate",         "topics": ["Cybersecurity", "Cloud Computing"]},
    {"id": "c8",  "title": "Business Analytics Specialization",   "provider": "Wharton / Coursera",         "level": "Intermediate",         "topics": ["Business", "Data Science", "Math & Stats"]},
    {"id": "c9",  "title": "iOS & Android App Development",       "provider": "Meta / Coursera",            "level": "Intermediate",         "topics": ["Mobile Apps", "Web Development"]},
    {"id": "c10", "title": "NLP Specialization",                  "provider": "DeepLearning.AI",            "level": "Advanced",             "topics": ["AI & NLP", "Machine Learning", "Math & Stats"]},
]


class StudentProfile(BaseModel):
    name: Optional[str] = "Student"
    goal: str
    interests: List[str]
    level: str
    learning_style: str
    weekly_hours: int
    background: Optional[str] = ""


class CourseScore(BaseModel):
    id: str
    title: str
    provider: str
    level: str
    topics: List[str]
    score: int
    reason: str


class RecommendationResponse(BaseModel):
    summary: str
    courses: List[CourseScore]


@app.get("/")
def root():
    return {"message": "CourseMatch AI API is live", "version": "1.0.0"}


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(profile: StudentProfile):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="API key not configured on server.")

    course_list = "\n".join(
        f"- {c['title']} ({c['provider']}) | Level: {c['level']} | Topics: {', '.join(c['topics'])}"
        for c in COURSES
    )

    prompt = f"""You are an expert AI education counselor. Analyse the student profile and score each course.

STUDENT PROFILE:
- Name: {profile.name}
- Goal: {profile.goal}
- Interests: {', '.join(profile.interests)}
- Skill level: {profile.level}
- Learning style: {profile.learning_style}
- Weekly hours: {profile.weekly_hours}
- Background: {profile.background or 'None'}

COURSES:
{course_list}

Return ONLY valid JSON, no markdown:
{{"summary":"2 sentence profile summary and strategy","courses":[{{"id":"c1","score":85,"reason":"one sentence explanation"}},all 10 courses]}}"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "max_tokens": 1400,
        "temperature": 0.35,
        "messages": [{"role": "user", "content": prompt}],
    }

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            res = await client.post(GROQ_URL, headers=headers, json=payload)
            res.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"Groq error: {e.response.text}")
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Cannot reach Groq: {str(e)}")

    raw = res.json()["choices"][0]["message"]["content"].strip()
    clean = raw.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="AI returned invalid JSON. Try again.")

    course_map = {c["id"]: c for c in COURSES}
    enriched = []
    for item in parsed.get("courses", []):
        meta = course_map.get(item["id"], {})
        enriched.append(CourseScore(
            id=item["id"],
            title=meta.get("title", ""),
            provider=meta.get("provider", ""),
            level=meta.get("level", ""),
            topics=meta.get("topics", []),
            score=item["score"],
            reason=item["reason"],
        ))

    enriched.sort(key=lambda x: x.score, reverse=True)
    return RecommendationResponse(summary=parsed.get("summary", ""), courses=enriched)