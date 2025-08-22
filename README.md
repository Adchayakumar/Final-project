# 🎓 Personalized AI Tutor & Performance Tracker

> **One-liner**: Adaptive learning platform that predicts each student’s performance, delivers real-time AI tutoring tailored to their learning style, and logs every interaction to build personalized study paths.

---

## Project Overview
### Description
Developed as part of the GUVI Data Science course, this project is a Personalized AI Tutor & Performance Tracker that leverages machine learning and AI to enhance student learning. It predicts performance metrics (score %, pass/fail, dropout risk) using PCA-transformed data from TiDB, offers real-time tutoring via Gemini 2.0-Flash, and includes features like digit recognition and topic tracking. The system is deployed as a multi-page Streamlit app for interactive use.

### Features
- 🔮 **Automated Performance Prediction** – Fetch student data from **TiDB** → PCA models return **score %**, **pass/fail**, **drop-out risk**.
- 🔧 **Manual Input Mode** – Sliders & toggles for “what-if” testing without database records.
- 📑 **Multi-Page Navigation** – One Streamlit app with three pages: **Dashboard**, **AI Tutor**, **Digit Recognizer**.
- ⚖️ **Model Consistency Check** – Auto-corrects pass/fail when predicted score ≥ 60 but model flags fail.
- 🤖 **Adaptive AI Tutor** – Login via **TiDB student_id**, **Gemini 2.0-Flash** streaming answers, prompt dynamically tuned to **preferred learning style**.
- 🕵️ **Real-Time Topic Tracking** – Zero-shot subject labeling via **DeBERTa-v3-large** on Hugging Face Space, logged for analytics.
- 🔢 **Digit Recognition** – Upload or webcam digit → CNN → **digit + confidence**.

## Dataset
### Source and Overview
- **Source**: Stored in **TiDB Cloud**, a synthetic dataset with student performance records (`/content/synthetic_student_data_final.csv`).
- **Columns**:
  | Column | Short Description |
  |---|---|
  | `student_id` | Unique ID (e.g., STD001) |
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
  | `predict_score` | Predicted score (0–100) |
  | `pass_fail` | Pass/Fail (1/0) |
  | `dropout_risk` | Dropout risk (1/0) |

### Preprocessing Details
- Loaded data with `on_bad_lines='skip'` to handle parsing errors.
- Converted `use_ed_tech` (Yes/No) and `pass_fail` (Pass/Fail) to binary (1/0).
- Handled `dropout_risk` by coercing errors to NaN, filling with 0, and converting to int.
- Scaled features with `StandardScaler` and applied PCA to reduce 8 key features to 5 components explaining 97.29% variance.
- One-hot encoded categorical variables (e.g., `gender`, `preferred_learning_style`).

## Machine Learning Pipeline
### Techniques Applied
- **Score Prediction**: Ridge regression with L2 regularization.
- **Pass/Fail and Dropout Risk**: Calibrated Logistic Regression with `CalibratedClassifierCV`.
- **Dimensionality Reduction**: PCA to address multicollinearity (original VIFs > 100).
- **Hyperparameter Tuning**: GridSearchCV with 5-fold CV to optimize `alpha` (Ridge) and `C` (Logistic).

### Model Training and Evaluation
- **Score Prediction**: Best alpha = 1, CV score = 0.8948, Train/Test R2 = 0.89, MAE = 2.44, RMSE = 3.06.
- **Pass/Fail**: Accuracy = 0.90, F1 = 0.89-0.91, consistency = 0.86016.
- **Dropout Risk**: Accuracy = 0.95, F1 = 0.89 for "Risk".

## Learning Style Determination
### Technique
- **Clustering**: KMeans clustering with the Elbow method determined 4 optimal clusters based on features like `attendance_rate`, `avg_daily_study_time`, `past_score`, and `quiz_accuracy`.
- **Features**: `attendance_rate`, `avg_daily_study_time`, `homework_completion_rate`, `past_score`, `motivation_level`, `video_time`, `quiz_time`, `homework_time`, `quiz_accuracy`, `use_ed_tech` (encoded).
- **Process**: Data scaled with `StandardScaler`, Elbow curve analyzed, and final clustering saved to `clustered_students.csv`.

### Cluster Mapping
- **Cluster 0: Hard Worker** – High `attendance_rate` (>85%), `avg_daily_study_time` (>4 hours), and `homework_completion_rate` (>90%), reflecting diligent effort.
- **Cluster 1: Smart Worker** – High `past_score` (>80%) and `quiz_accuracy` (>85%) with moderate `study_time` (<3 hours), indicating efficient learning.
- **Cluster 2: Regular Student** – Average values across metrics (e.g., `attendance_rate` ~70-80%), balanced effort.
- **Cluster 3 : Less Active** Student who study less and get the less marks .

## Workflow
1. **Data Extraction** – Pull student records from **TiDB**.
2. **Model Prediction** – PCA → Ridge → calibrated classifiers.
3. **Interactive Dashboard** – Visualize risk & progress.
4. **AI Tutor Chat**:
   - User logs in → Gemini streaming chat.
   - Every message is POST-ed to **HF-Space `/predict`**.
   - **DeBERTa-v3-large** returns subject label.
   - Label + message saved in `log_table` for analytics.
5. **Digit Recognition** – Micro-assessment via CNN.
6. **Analytics Loop** – Logged topics drive next content suggestions.

## Technologies Used
- **Programming**: Python 3
- **Libraries**: `Streamlit` (multi-page app), `Pandas`, `scikit-learn` (PCA, classifiers), `Transformers` (DeBERTa-v3-large), `joblib`
- **AI Models**: `Google Gemini 2.0-Flash` (tutor), `DeBERTa-v3-large` (topic labeling), CNN (digit recognition)
- **Database**: `TiDB Cloud` (student data & logs)
- **Micro-service**: `FastAPI` (HF-Space)

## Limitations and Future Work

### Limitations
- **Consistency Accuracy**: The pass/fail consistency accuracy (0.86016) indicates some contradictions between predicted scores and pass/fail classifications, warranting further refinement.
- **Synthetic Data**: Reliance on synthetic data from TiDB may not fully reflect real-world student patterns, potentially affecting model generalizability.


### Future Work
- **Enhanced AI Model**: Integrate the Google Gemini Flash 2.0 experimental model for improved tutoring capabilities.
- **Voice Integration**: Add voice input/output options to the Streamlit app for a more accessible user experience.
- **Interactive Features**: Implement quiz-making buttons to facilitate seamless interaction with the LLM.
- **Manual Mode**: Introduce a manual mode for unpredictable students, allowing LLM control to tailor responses (e.g., short answers for hard workers, detailed explanations for less active students).
- **Real Data Testing**: Validate models with real TiDB data to enhance accuracy and relevance.


## Installation and Usage
### Prerequisites
- Python 3.11+
- Required libraries: Install via `pip install -r requirements.txt`.

### Installation
1. **Clone**
   ```bash
   git clone https://github.com/Adchayakumar/friendly_tutor.git
   cd friendly_tutor

2.**Install deps**
  ```bash
  python -m venv env && source env/bin/activate  # Windows: env\Scripts\activate
  pip install -r requirements.txt

3. create .env file 

  store values of 
  'GOOGLE_API_KEY'
  'HF_TOKEN'

4. **Launch**
  
  streamlit run app.py

