import requests
from fastapi import APIRouter, File, UploadFile, Query
from sentence_transformers import SentenceTransformer, util
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
import numpy as np
import html

router = APIRouter()

API_URL = "https://jsearch.p.rapidapi.com/search"
API_KEY = "#"

HEADERS = {
    "x-rapidapi-key": API_KEY,
    "x-rapidapi-host": "jsearch.p.rapidapi.com"
}

# ✅ Helper to clean text
def clean_text(text: str) -> str:
    if not text:
        return ""
    try:
        # Decode HTML entities & fix encoding issues
        return html.unescape(text).encode("latin1").decode("utf-8", errors="ignore")
    except Exception:
        return text.strip()  # fallback

# Load model once
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# ✅ Fallback job fetch (RemoteOK with cleaning)
# -----------------------------
async def fetch_remoteok_jobs(keyword: str, limit: int = 10):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get("https://remoteok.io/api", headers=headers)
        data = res.json()

        jobs = []
        # ✅ skip first element (metadata row)
        for j in data[1:]:
            if isinstance(j, dict) and "position" in j:
                if keyword.lower() in j["position"].lower() or not keyword:
                    desc = j.get("description") or f"{j.get('position')} at {j.get('company')}"
                    jobs.append({
                        "title": clean_text(j.get("position")),
                        "company": clean_text(j.get("company")),
                        "location": "Remote",
                        "apply_link": j.get("url"),
                        "description": clean_text(desc)[:300],
                        "source": "RemoteOK"
                    })

        # ✅ fallback: return some jobs even if no match
        if not jobs:
            for j in data[1:limit+1]:
                if isinstance(j, dict) and "position" in j:
                    jobs.append({
                        "title": clean_text(j.get("position")),
                        "company": clean_text(j.get("company")),
                        "location": "Remote",
                        "apply_link": j.get("url"),
                        "description": clean_text(j.get("description") or j.get("position"))[:300],
                        "source": "RemoteOK"
                    })
        return jobs[:limit]
    except Exception as e:
        print("RemoteOK fetch failed:", e)
        return []

# -----------------------------
# ✅ Logistic Regression Training (Synthetic Example)
# -----------------------------
def train_logistic_regression():
    samples = [
        ("I am a Python developer with ML experience", "Python Developer working on ML models", 1),
        ("Frontend React developer", "Python Backend Developer", 0),
        ("Data Scientist skilled in NLP", "NLP Engineer with Transformers", 1),
        ("Java Developer Spring Boot", "Frontend Angular Developer", 0),
    ]

    X, y = [], []
    for resume, job, label in samples:
        r_emb = model.encode(resume)
        j_emb = model.encode(job)
        pair_emb = np.abs(r_emb - j_emb)
        X.append(pair_emb)
        y.append(label)

    clf = LogisticRegression()
    clf.fit(X, y)
    return clf

clf = train_logistic_regression()

# -----------------------------
# ✅ Resume-based Recommendation Endpoint
# -----------------------------
@router.post("/resume-jobs")
async def recommend_jobs(
    file: UploadFile = File(...),
    keyword: str = Query("python developer"),
    location: str = Query("Bangalore"),
    limit: int = Query(10)
):
    """Recommend jobs using Cosine Similarity, Logistic Regression & KMeans clustering"""

    # 1️⃣ Read resume
    resume_text = (await file.read()).decode("utf-8", errors="ignore")

    # 2️⃣ Try RapidAPI (may fail due to quota)
    querystring = {"query": f"{keyword} in {location}", "num_pages": 1}
    jobs, job_descriptions = [], []
    try:
        response = requests.get(API_URL, headers=HEADERS, params=querystring)
        data = response.json()
        for job in data.get("data", [])[:limit]:
            desc = f"{job.get('job_title', '')} {job.get('job_description', '')}"
            jobs.append({
                "title": clean_text(job.get("job_title")),
                "company": clean_text(job.get("employer_name")),
                "location": clean_text(job.get("job_city")),
                "apply_link": job.get("job_apply_link"),
                "description": clean_text(desc)[:300]
            })
            job_descriptions.append(clean_text(desc))
    except Exception:
        pass

    # 3️⃣ Fallback to RemoteOK if no jobs
    if not jobs:
        fallback_jobs = await fetch_remoteok_jobs(keyword, limit)
        for job in fallback_jobs:
            desc = job.get("description") or f"{job['title']} at {job['company']}"
            jobs.append(job)
            job_descriptions.append(clean_text(desc))

    if not job_descriptions:
        return {"jobs": [], "message": "No jobs found"}

    # 4️⃣ Compute embeddings
    resume_emb = model.encode(resume_text, convert_to_tensor=True)
    job_embs = model.encode(job_descriptions, convert_to_tensor=True)

    # ✅ Method 1: Cosine Similarity
    similarities = util.pytorch_cos_sim(resume_emb, job_embs)[0]
    cosine_ranked = [
        {**job, "score": float(sim)}
        for job, sim in sorted(zip(jobs, similarities), key=lambda x: x[1], reverse=True)
    ]

    # ✅ Method 2: Logistic Regression
    pair_features = [np.abs(resume_emb.cpu().numpy() - j.cpu().numpy()) for j in job_embs]
    probs = clf.predict_proba(pair_features)[:, 1]
    logistic_ranked = [
        {**job, "probability": float(prob)}
        for job, prob in sorted(zip(jobs, probs), key=lambda x: x[1], reverse=True)
    ]

    # ✅ Method 3: KMeans Clustering
    clustered_jobs = []
    try:
        job_embs_np = job_embs.cpu().numpy().astype(np.float64)
        kmeans = KMeans(n_clusters=min(3, len(job_embs_np)), random_state=42, n_init=10)
        clusters = kmeans.fit_predict(job_embs_np)
        resume_cluster = kmeans.predict([resume_emb.cpu().numpy().astype(np.float64)])[0]
        clustered_jobs = [job for job, c in zip(jobs, clusters) if c == resume_cluster]
    except Exception as e:
        print("KMeans clustering failed:", e)

    return {
        "jobs": cosine_ranked,
        "cosine_similarity": cosine_ranked,
        "logistic_regression": logistic_ranked,
        "clustered_jobs": clustered_jobs,
        "success": True,
        "sources": ["RapidAPI JSearch", "RemoteOK (fallback)", "Arbeitnow (fallback)"]
    }

# -----------------------------
# ✅ NEW: Realtime Jobs Endpoint
# -----------------------------
@router.get("/realtime-jobs")
async def realtime_jobs(keyword: str = "python", limit: int = 20):
    jobs = await fetch_remoteok_jobs(keyword, limit)

    if not jobs:
        try:
            res = requests.get("https://arbeitnow.com/api/job-board-api")
            data = res.json().get("data", [])
            for job in data[:limit]:
                if keyword.lower() in job["title"].lower() or not keyword:
                    jobs.append({
                        "title": clean_text(job.get("title")),
                        "company": clean_text(job.get("company_name")),
                        "location": clean_text(job.get("location")),
                        "apply_link": job.get("url"),
                        "description": clean_text(job.get("description") or job.get("title"))[:300],
                        "source": "Arbeitnow"
                    })
        except Exception as e:
            print("Arbeitnow failed:", e)

    return {"jobs": jobs[:limit], "success": True, "sources": ["RemoteOK", "Arbeitnow"]}

