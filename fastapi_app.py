"""
FastAPI server inside Hugging Face Space
POST /predict  ->  zero-shot subject prediction + save to TiDB
"""
import os
import time
from contextlib import asynccontextmanager

import mysql.connector
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------- load model ONCE ----------
MODEL_NAME = "MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33"
LABELS = [
    "Mathematics", "Physics", "Chemistry", "Biology",
    "History", "Geography", "Literature", "Computer-Science"
]

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # load at startup
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    ml_models["tokenizer"] = tokenizer
    ml_models["model"] = model
    yield
    # shutdown
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

# ---------- DB helper ----------
def get_conn():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", 4000)),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASS"),
        database=os.getenv("DB_NAME"),
        ssl_ca=os.getenv("DB_SSL_CA_PATH") or None
    )

# ---------- request schema ----------
class PredictRequest(BaseModel):
    student_id: str
    text: str

# ---------- API endpoint ----------
@app.post("/predict")
def predict(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(400, "Empty text")
    tok = ml_models["tokenizer"](
        req.text,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    if torch.cuda.is_available():
        tok = {k: v.cuda() for k, v in tok.items()}
    with torch.no_grad():
        logits = ml_models["model"](**tok).logits
        probs = torch.softmax(logits, dim=-1)[0]
        idx = int(torch.argmax(probs))
        subject = LABELS[idx]

    # save to DB
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO log_table (student_id, input_sample, subject, prediction_time) "
            "VALUES (%s, %s, %s, %s)",
            (req.student_id, req.text, subject, time.strftime('%Y-%m-%d %H:%M:%S'))
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print("DB error:", e)
        raise HTTPException(500, "DB write failed")

    return {"subject": subject}

@app.get("/")
def root():
    return {"message": "Subject predictor is running"}