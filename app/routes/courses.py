from fastapi import APIRouter
import json
from app.models import Course

router = APIRouter()

@router.get("/", response_model=list[Course])
def get_courses():
    with open("app/data/courses.json") as f:
        courses = json.load(f)
    return courses
