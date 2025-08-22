# 🎓 Personalise AI Tutor & Performance Tracker

> **One-liner**: Adaptive learning platform that predicts each student’s performance, delivers real-time AI tutoring tailored to their learning style, and logs every interaction to build personalized study paths.

---

## 🚀 Features

- 🔮 **Automated Performance Prediction** – fetch student data from **TiDB** → PCA models return **score %**, **pass/fail**, **drop-out risk**  
- 🔧 **Manual Input Mode** – sliders & toggles for “what-if” testing without database records  
- 📑 **Multi-Page Navigation** – one Streamlit app with three pages: **Dashboard**, **AI Tutor**, **Digit Recognizer**  
- ⚖️ **Model Consistency Check** – auto-corrects pass/fail when predicted score ≥ 60 but model flags fail  
- 🤖 **Adaptive AI Tutor** – login via **TiDB student_id**, **Gemini 2.0-Flash** streaming answers, prompt dynamically tuned to **preferred learning style**  
- 🕵️ **Real-Time Topic Tracking** – zero-shot subject labeling via **DeBERTa-v3-large** on Hugging Face Space, logged for analytics  
- 🔢 **Digit Recognition** – upload or webcam digit → CNN → **digit + confidence**

---

## 📊 Dataset Overview (TiDB)

| Column | Short Description |
|---|---|
| `student_id` | Unique ID (e.g. STD001) |
| `grade_level` | Grade 1–12 |
| `age` | Grade + 5–6 yrs |
| `gender` | Male / Female / Other |
| `attendance_rate` | % attended (50–100) |
| `avg_daily_study_time` | Hours/day |
| `homework_completion_rate` | % HW done (40–100) |
| `past_score` | Previous score (30–100) |
| `motivation_level` | Low / Medium / High |
| `use_ed_tech` | Uses EdTech (T/F) |
| `video_time` | Minutes on videos |
| `quiz_time` | Minutes on quizzes |
| `homework_time` | Minutes on HW |
| `preferred_learning_style` | Visual / Auditory / etc. |
| `quiz_accuracy` | Predicted quiz % (20–100) |

---

## 🧼 Data & Model Pipeline

1. **Data Cleaning** – validate TiDB columns, handle missing values  
2. **PCA Transformation** – reduce 8 features → dense vectors  
3. **Prediction** – Ridge + calibrated classifiers → score, pass/fail, drop-out  
4. **Topic Logging** – zero-shot DeBERTa labels every chat message  
5. **Dynamic Prompting** – Gemini prompt rewritten per `preferred_learning_style`

---
## 🔄 Project Workflow

1. **Data Extraction** – pull student records from **TiDB**  
2. **Model Prediction** – PCA → Ridge → calibrated classifiers  
3. **Interactive Dashboard** – visualize risk & progress  

4. **AI Tutor Chat**  
   • user logs in → Gemini streaming chat  
   • **every message is POST-ed** to **HF-Space `/predict`**  
   • HF-Space runs **DeBERTa-v3-large** → returns **subject label**  
   • label + message **saved** in `log_table` for analytics  

5. **Digit Recognition** – micro-assessment via CNN  
6. **Analytics Loop** – logged topics drive next content suggestions


## 🧰 Technologies Used

- `Python 3`
- `Streamlit` (multi-page app)
- `Pandas, scikit-learn` (PCA, calibrated classifiers)
- `Google Gemini 2.0-Flash` (tutor LLM)
- `Transformers + DeBERTa-v3-large` (topic prediction)
- `TiDB Cloud` (student data & logs)
- `FastAPI` (HF-Space micro-service)

---

## 📦 Installation Instructions

1. **Clone**
   ```bash
   git clone https://github.com/Adchayakumar/friendly_tutor.git
   cd friendly_tutor

2.**Install deps**
  ```bash
  python -m venv env && source env/bin/activate  # Windows: env\Scripts\activate
  pip install -r requirements.txt

3. **Set env vars**
  create env file
  'GOOGLE_API_KEY'
  'HF_TOKEN'

4. **Launch**
  ```bash

  streamlit run app.py

