# backend/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.data import jobs, courses, skills
from app.realtime_jobs import router as realtime_router 


app = FastAPI()
app.include_router(realtime_router)

# Allow frontend to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],  # your frontend URL
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/jobs")
async def get_jobs():
    return jobs

@app.get("/courses")
async def get_courses():
    return courses

@app.get("/skills")
async def get_skills():
    return skills
