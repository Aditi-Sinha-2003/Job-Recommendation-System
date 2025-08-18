from fastapi import APIRouter
import json
from app.models import SkillTrend

router = APIRouter()

@router.get("/", response_model=list[SkillTrend])
def get_skills():
    with open("app/data/skills.json") as f:
        skills = json.load(f)
    return skills
