# backend/app/realtime_jobs.py
import requests
from bs4 import BeautifulSoup
from fastapi import APIRouter

router = APIRouter()

def fetch_naukri_jobs(keyword="python", location="bangalore"):
    url = f"https://www.naukri.com/{keyword}-jobs-in-{location}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    jobs = []
    for job_card in soup.find_all("div", class_="jobTuple"):
        title = job_card.find("a", class_="title").text.strip()
        company = job_card.find("a", class_="subTitle").text.strip()
        location = job_card.find("li", class_="location").text.strip()
        jobs.append({"title": title, "company": company, "location": location})
    return jobs

@router.get("/realtime-jobs")
async def get_realtime_jobs(keyword: str = "python", location: str = "bangalore"):
    naukri_jobs = fetch_naukri_jobs(keyword, location)
    return {"jobs": naukri_jobs}
