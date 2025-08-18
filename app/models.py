from pydantic import BaseModel
from typing import List

class Job(BaseModel):
    id: int
    title: str
    company: str
    location: str
    salary: str
    match_score: float
    skills: List[str]
    description: str

class Course(BaseModel):
    id: int
    title: str
    provider: str
    link: str
    rating: float
    skills: List[str]

class SkillTrend(BaseModel):
    skill: str
    demand_score: float
