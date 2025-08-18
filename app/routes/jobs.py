from fastapi import APIRouter
import json
from app.models import Job

router = APIRouter()

@router.get("/", response_model=list[Job])
def get_jobs():
    with open("app/data/jobs.json") as f:
        jobs = json.load(f)
    return jobs

@router.get("/{job_id}", response_model=Job)
def get_job(job_id: int):
    with open("app/data/jobs.json") as f:
        jobs = json.load(f)
    job = next((j for j in jobs if j["id"] == job_id), None)
    if not job:
        return {"error": "Job not found"}
    return job
