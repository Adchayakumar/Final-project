
import streamlit as st
import pandas as pd
import mysql.connector
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import numpy as np

# ------------------- Page Config -------------------
st.set_page_config(layout="wide", page_title="Student Performance Dashboard",page_icon=":slightly_smiling_face:")
st.markdown("""
<style>
.stApp { background-color: #0E1220; }
.block-container { background-color: #1C1F2E; padding: 2rem; border-radius: 10px; }
[data-testid="stMetricValue"] { color: #00E5FF; font-weight: bold; }
[data-testid="stMetricLabel"] { color: #FFFFFF; }
body, .stMarkdown, .stText { color: #FFFFFF; }
.progress-label { color: #FFFFFF; font-size: 18px; font-weight: 600; margin-bottom: 6px; }
.neon-bar { background-color: rgba(30,34,45,1); border-radius: 10px; overflow: hidden; height: 26px; margin-bottom: 14px; box-shadow: inset 0 0 0 1px rgba(255,255,255,0.02); }
.neon-fill { height: 100%; text-align: center; color: #000000; font-size: 13px; font-weight: 700; line-height: 26px; }
.neon-card { background-color:#232731; border-radius:12px; padding:14px; text-align:center; box-shadow: 0 0 18px rgba(0,229,255,0.08); margin-bottom:8px; }
.neon-card h4 { color:#ffffff; margin:0; font-size:14px; font-weight:600; }
.neon-card h2 { margin:6px 0 0 0; font-size:28px; font-weight:800; }
.status-row { display:flex; align-items:center; justify-content:center; margin-top:8px; }
.status-dot { width:12px; height:12px; border-radius:50%; margin-right:10px; box-shadow:0 0 10px rgba(0,0,0,0.6); }
.status-text { color:white; font-weight:700; font-size:16px; }
.css-1d391kg { padding-top: 0px; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 1)  NEW CACHED LOADER  – clustering pipeline
# =========================================================
@st.cache_resource
def load_clustering_assets():
    """
    Returns scaler, PCA and K-means trained on
    ['attendance_rate', 'homework_completion_rate', 'past_score',
     'motivation_level', 'video_time', 'quiz_time', 'homework_time',
     'quiz_accuracy', 'use_ed_tech']
    """
    try:
        cl_scaler = joblib.load('/content/scaler_object.pkl')
        cl_pca    = joblib.load('/content/pca_object.pkl')
        kmeans    = joblib.load('/content/mnkn_model.pkl')
        return cl_scaler, cl_pca, kmeans
    except FileNotFoundError as e:
        st.error(f"Clustering asset missing: {e}")
        return None, None, None

cl_scaler, cl_pca, kmeans = load_clustering_assets()

# =========================================================
# 2)  CLUSTER FEATURE LIST & MAPPING
# =========================================================
cluster_features = [
    'attendance_rate', 'homework_completion_rate', 'past_score',
    'motivation_level', 'video_time', 'quiz_time', 'homework_time',
    'quiz_accuracy', 'use_ed_tech'
]
cluster_map = {
    0: "Regular student",
    1: "Less active",
    2: "Quick learner"
}

# =========================================================
# 3)  HELPER – predict learning-style cluster
# =========================================================
def predict_cluster(user_dict):
    """
    user_dict must contain the 9 cluster_features (floats or ints).
    Returns the mapped cluster name.
    """
    try:
        df = pd.DataFrame([{f: float(user_dict[f]) for f in cluster_features}]) # order the features and convert to dataframe
        X_scaled = cl_scaler.transform(df)  
        X_pca    = cl_pca.transform(X_scaled)
        cluster  = int(kmeans.predict(X_pca)[0])
        return cluster_map.get(cluster, "Unknown")
    except Exception as e:
        st.error(f"Cluster prediction error: {e}")
        return "Unknown"

# ------------------- Load Models, Scaler & PCA -------------------
@st.cache_resource
def load_models():
    try:
        ridge = joblib.load('/content/ridge_model_pca.pkl')
        calibrated_pass = joblib.load('/content/pass_fail_model_pca.pkl')
        calibrated_dropout = joblib.load('/content/dropout_risk_model_pca.pkl')
        scaler = joblib.load('/content/scaler.pkl')
        pca = joblib.load('/content/pca.pkl')
        return ridge, calibrated_pass, calibrated_dropout, scaler, pca
    except FileNotFoundError as e:
        st.error(f"Model, scaler, or PCA file not found: {e}")
        return None, None, None, None, None

ridge, calibrated_pass, calibrated_dropout, scaler, pca = load_models()

if ridge is None:
    st.stop()

# ------------------- Unified Feature List (for predictions) -------------------
unified_features = [
    'grade_level', 'attendance_rate', 'avg_daily_study_time',
    'homework_completion_rate', 'past_score', 'motivation_level', 'use_ed_tech',
    'quiz_accuracy'
]

# ------------------- Encoding helper -------------------
def encode_row(row: dict) -> dict:
    """
    Convert categorical columns to numeric values.
    """
    out = dict(row)
    try:
        out['grade_level'] = int(float(out.get('grade_level', 0)))
    except (ValueError, TypeError):
        out['grade_level'] = 0
    try:
        out['motivation_level'] = int(float(out.get('motivation_level', 2)))
    except (ValueError, TypeError):
        out['motivation_level'] = 2
    val = str(out.get('use_ed_tech', '')).strip().lower()
    out['use_ed_tech'] = 1 if val in ('true', 'yes', '1') else 0
    return out

# ------------------- Database helper -------------------
def get_student_data(student_id):
    try:
        conn = mysql.connector.connect(
            host="gateway01.ap-southeast-1.prod.aws.tidbcloud.com",
            port=4000,
            user="4V44XYoMA7okY9v.root",
            password="aW2CrSwcTgjFhNAb",
            database="final_project",
            ssl_verify_cert=True,
            ssl_verify_identity=True
        )
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM student_performance WHERE student_id = %s", (student_id,))
        row = cursor.fetchone()
        conn.close()
        return row
    except mysql.connector.Error as e:
        st.error(f"Database error: {e}")
        return None

# ------------------- Safe preprocessing -------------------
def preprocess_input(raw_dict, scaler, pca):
    enc = encode_row(raw_dict) # Encode the cat column to num
    df = pd.DataFrame([enc])
    for col in unified_features:
        if col not in df.columns:
            df[col] = 0.0
    df = df[unified_features].astype(float)
    scaled = scaler.transform(df)
    pca_transformed = pca.transform(scaled)
    return pca_transformed

# ------------------- Run all three models -------------------
def run_models(student_dict):
    try:
        X_pca = preprocess_input(student_dict, scaler, pca) #Scale and pca

        score = min(round(ridge.predict(X_pca)[0], 2), 99)  # Score prediction

        pf_raw = calibrated_pass.predict(X_pca)[0]   # pass/fail prediction
        pf_num = 1 if pf_raw == 1 else 0

        dropout_risk = calibrated_dropout.predict(X_pca)[0]  # dropout risk prediction

        # Consistency check
        if score >= 60 and pf_num == 0:
            pf_num = 1
            st.warning("Contradiction detected: High score but predicted Fail. Overriding to Pass.")

        return {
            "Pass/Fail": pf_num,
            "Dropout Risk": int(dropout_risk),
            "Predicted Score": score
        }
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return {
            "Pass/Fail": "N/A",
            "Dropout Risk": "N/A",
            "Predicted Score": "N/A"
        }

# ------------------- UI helpers -------------------
def neon_progress_label(label, percent, color):
    percent = max(0, min(int(percent), 100))
    st.markdown(f"""
    <div class="progress-label">{label}</div>
    <div class="neon-bar">
        <div class="neon-fill" style="width:{percent}%; background-color:{color};">{percent}%</div>
    </div>
    """, unsafe_allow_html=True)

def neon_card_html(title, value, glow_color):
    return f"""
    <div class="neon-card" style="box-shadow:0 0 20px {glow_color}33;">
        <h4>{title}</h4>
        <h2 style="color:{glow_color};">{value}</h2>
    </div>
    """


# =========================================================
# Combined UI – keeps PCA pipeline, borrows V2 manual polish
# =========================================================

st.title("📊 Student Performance Dashboard")
st.info(
    "Enter a Student ID for automatic prediction from the database "
    "or toggle Manual Mode for custom input. Ensure all inputs are valid."
)

mode = st.toggle("Manual Mode", value=False)

if not mode:
    # ---------- DATABASE MODE (unchanged from V1) ----------
    student_id = st.text_input("Enter Student ID", help="Enter a valid student ID.")
    if student_id:
        with st.spinner("Fetching student data…"):
            student_data = get_student_data(student_id)

        if student_data:
            prediction = run_models(student_data)

            student_name = student_data.get("name", "UNKNOWN")
            gender = student_data.get("gender", "N/A")
            dropout_risk = prediction.get("Dropout Risk", "N/A")

            col1, col2 = st.columns([3, 2])
            with col1:
                st.markdown(f"**Student ID:** {student_id}")
            with col2:
                if dropout_risk == 1:
                    st.markdown(
                        "<span style='color:red; font-weight:bold;'>🔴 Dropout risk is high</span>",
                        unsafe_allow_html=True,
                    )

            st.markdown(
                f"**Name:** {student_name}<br>"
                f"**Age:** {student_data.get('age', 'N/A')}<br>"
                f"**Grade:** {student_data.get('grade_level', 'N/A')}<br>"
                f"**Gender:** {gender}",
                unsafe_allow_html=True,
            )

            left_side, right_side = st.columns([1, 2])

            with left_side:
                st.markdown(
                    "<h3 style='color:white; margin-bottom:10px;'>Progress</h3>",
                    unsafe_allow_html=True,
                )

                mot_val = student_data.get("motivation_level", 0)
                quiz_acc = student_data.get("quiz_accuracy", 0)
                hw_completion = student_data.get("homework_completion_rate", 0)
                attendance = student_data.get("attendance_rate", 0)

                homework_time = student_data.get("homework_time", 0)
                quiz_time = student_data.get("quiz_time", 0)
                video_time = student_data.get("video_time", 0)
                avg_study_time=student_data.get("avg_daily_study_time",0)
                total_time = homework_time + quiz_time + video_time
                video_percent = (video_time / total_time) * 100 if total_time > 0 else 0

                neon_progress_label(
                    "Motivation Level",
                    (mot_val / 3) * 100 if mot_val else 0,
                    "#00E5FF",
                )
                neon_progress_label("Quiz Accuracy", quiz_acc, "#8E2DE2")
                neon_progress_label("Video Completion", video_percent, "#FF00FF")
                neon_progress_label("Homework Completion", hw_completion, "#00E5FF")

                edtech_str = "Yes" if student_data.get("use_ed_tech", 0) == 1 else "No"
                pref_style = student_data.get("learning_style", "N/A")
                st.markdown(
                    "<div style='margin-top:8px;'><h4 style='margin:0 0 6px 0;'>Additional Info</h4></div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div style='color:#00E5FF; font-weight:600;'>EdTech Usage: "
                    f"<span style='color:white; font-weight:400;'>{edtech_str}</span></div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div style='color:#8E2DE2; font-weight:600;'>Preferred Learning Style: "
                    f"<span style='color:white; font-weight:400;'>{pref_style}</span></div>",
                    unsafe_allow_html=True,
                )

            with right_side:
                fig = go.Figure(
                    data=[
                        go.Pie(
                            labels=["Homework", "Quiz", "Video"],
                            values=[homework_time, quiz_time, video_time],
                            hole=0.6,
                            textinfo="label+percent",
                            marker=dict(colors=["#8E2DE2", "#00E5FF", "#FF00FF"]),
                            showlegend=False,
                        )
                    ]
                )
                fig.update_layout(
                    annotations=[
                        dict(
                            text=f"{round(avg_study_time,1)} Hrs",
                            x=0.5,
                            y=0.5,
                            font_size=18,
                            font_color="#FFFFFF",
                            showarrow=False,
                        )
                    ],
                    paper_bgcolor="#1C1F2E",
                    plot_bgcolor="#1C1F2E",
                    margin=dict(t=0, b=0, l=0, r=0),
                    height=320,
                )
                st.plotly_chart(fig, use_container_width=True)

                c1, c2 = st.columns(2)
                with c1:
                    past_score_display = (
                        f"{student_data.get('past_score', 'N/A')}%"
                        if student_data.get("past_score") is not None
                        else "N/A"
                    )
                    st.markdown(
                        neon_card_html("Past Score", past_score_display, "#00E5FF"),
                        unsafe_allow_html=True,
                    )
                with c2:
                    pred_score_display = (
                        f"{prediction['Predicted Score']}%"
                        if prediction.get("Predicted Score") is not None
                        else "N/A"
                    )
                    st.markdown(
                        neon_card_html("Predicted Score", pred_score_display, "#8E2DE2"),
                        unsafe_allow_html=True,
                    )

                pass_status = prediction.get("Pass/Fail", "N/A")
                dot_color = "#29FF87" if pass_status == 1 else "#FF5C7A"
                status_text = "PASS" if pass_status == 1 else "FAIL"
                st.markdown(
                    f"""
                    <div class="status-row">
                        <div class="status-dot"
                             style="background-color:{dot_color};
                                    box-shadow:0 0 12px {dot_color}88;"></div>
                        <div class="status-text">{status_text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div style='margin-top:8px; text-align:center; color:#CCCCCC;'>"
                    f"Dropout Risk: <b style='color:#FFFFFF'>{dropout_risk}</b></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.warning("Student ID not found!")

else:
  # =========================================================
  # 🔧 NEW WIZARD-STYLE MANUAL MODE
  # =========================================================
  st.subheader("🔧 Manual Prediction Wizard")
  st.info("Step-by-step: enter data → choose outputs → get predictions.")

  # ---------- STEP 1 – collect ALL inputs ----------
  st.markdown("### 1️⃣  Enter Student Data")

  # 9 features for the 3 performance models
  perf_features = unified_features
  # 9 features for clustering (identical here, but kept separate for clarity)
  clust_features = cluster_features

  # Build one flat dict so we can loop once
  all_features = list(dict.fromkeys(perf_features + clust_features))  # preserve order, no dupes
  user_data = {}

  cols = st.columns(3)
  for idx, feat in enumerate(all_features):
      with cols[idx % 3]:
          label = feat.replace("_", " ").title()
          if feat == "motivation_level":
              user_data[feat] = st.slider(label, 1, 3, value=2, key=f"slider_{feat}")
          elif feat == "avg_stuudy_time":
              user_data[feat] = st.slider(label, 2, 5, value=3, key=f"slider_{feat}")
          elif feat == "use_ed_tech":
              user_data[feat] = 1 if st.radio(label, ["Yes", "No"], key=f"radio_{feat}") == "Yes" else 0
          else:
              user_data[feat] = st.number_input(label, value=0.0, key=f"num_{feat}")

  # ---------- STEP 2 – choose what to predict ----------
  st.markdown("### 2️⃣  Choose Prediction Targets")
  targets = st.multiselect(
      "Select one or more:",
      ["Score", "Pass/Fail", "Drop-out Risk", "Learning Style"],
      default=["Score"]
  )

  # ---------- STEP 3 – compute & display ----------
  if st.button("🚀  Run Prediction"):
      if not targets:
          st.warning("Please select at least one prediction target.")
          st.stop()

      results = {}

      # ---- 1) SCORE ----
      if "Score" in targets:
          try:
              X_perf = preprocess_input(user_data, scaler, pca)  # re-uses unified pipeline
              score = min(round(ridge.predict(X_perf)[0], 2), 99)
              results["Score"] = f"{score} %"
          except Exception as e:
              results["Score"] = f"Error: {e}"

      # ---- 2) PASS/FAIL ----
      if "Pass/Fail" in targets:
          try:
              X_perf = preprocess_input(user_data, scaler, pca)
              pf_raw = calibrated_pass.predict(X_perf)[0]
              results["Pass/Fail"] = "PASS" if pf_raw == 1 else "FAIL"
          except Exception as e:
              results["Pass/Fail"] = f"Error: {e}"

      # ---- 3) DROP-OUT RISK ----
      if "Drop-out Risk" in targets:
          try:
              X_perf = preprocess_input(user_data, scaler, pca)
              risk_raw = calibrated_dropout.predict(X_perf)[0]
              results["Drop-out Risk"] = "Yes" if risk_raw == 1 else "No"
          except Exception as e:
              results["Drop-out Risk"] = f"Error: {e}"

      # ---- 4) LEARNING STYLE ----
      if "Learning Style" in targets:
          try:
              learning_style = predict_cluster(user_data)  # uses clustering pipeline
              results["Learning Style"] = learning_style
          except Exception as e:
              results["Learning Style"] = f"Error: {e}"

      # ---------- DISPLAY ----------
      st.markdown("### 3️⃣  Results")
      cols = st.columns(len(results))
      for col, (title, val) in zip(cols, results.items()):
          color = "#29FF87" if "PASS" in str(val) or "No" in str(val) else "#FF5C7A"
          if title == "Learning Style":
              color = "#00E5FF"
          col.markdown(
              f"""
              <div class="neon-card" style="box-shadow:0 0 20px {color}33;">
                  <h4>{title}</h4>
                  <h2 style="color:{color};">{val}</h2>
              </div>
              """,
              unsafe_allow_html=True,
          )