# Technical Report: Personalized AI Tutor & Performance Tracker

## Executive Summary
This project develops a Personalized AI Tutor & Performance Tracker as part of the GUVI Data Science course. The core problem is addressing diverse student learning needs in education, where traditional methods often fail to adapt to individual styles, speeds, and risks. The proposed AI solution uses machine learning models to predict student performance (score, pass/fail, dropout risk) and clustering for learning styles, integrated into a Streamlit app with Gemini 2.0-Flash for tutoring and DeBERTa-v3-large for topic tracking.

The solution processes synthetic student data (1,000,000 rows) with PCA to handle multicollinearity, achieving high model performance (R2 = 0.89 for score prediction, accuracy = 0.90-0.95 for classifications). Challenges like consistency issues (0.86 accuracy) were mitigated with overrides, paving the way for real-world enhancements like voice integration.

## Exploratory Data Analysis (EDA)
EDA was conducted using Pandas Profiling on the synthetic dataset (`synthetic_student_data_final.csv`), revealing key insights into data quality, distributions, and relationships.

### Key Statistics and Insights
- **Dataset Size**: 1,000,000 observations, 18 variables, no missing values or duplicates.
- **Variable Types**: 1 text (`student_id`), 11 numeric (e.g., `attendance_rate`, `past_score`), 4 categorical (e.g., `gender`, `preferred_learning_style`), 2 boolean (`pass_fail`, `dropout_risk`).
- **Distributions**:
  - `grade_level`: Bimodal, peaking at 6-8 (~13.3% each), lower for 1-5 (~4%) and 9-12 (~10%).
  - `attendance_rate`: Mean = 83.5, skewed toward higher values (50-100).
  - `avg_daily_study_time`: Mean = 2.45 hours, clipped 0.5-5.
  - `past_score`: Mean = 65, clipped 40-100.
  - `predict_score`: Mean ~60-70, highly correlated with `pass_fail` (0.964).
- **Correlations and Multicollinearity**:
  - High correlations flagged (17 alerts): e.g., `age` vs. `grade_level` (0.984), `avg_daily_study_time` vs. `quiz_time` (0.979).
  - Performance metrics (e.g., `past_score`, `quiz_accuracy`) strongly linked to targets, but time features (e.g., `video_time`) show multicollinearity.
  - Insight: Synthetic data patterns (e.g., derived time splits) cause high VIFs (>100), addressed via PCA in modeling.
- **Visualizations**: Histograms show skewed distributions (e.g., attendance_rate kurtosis -0.53). Correlation heatmap highlights clusters among time and performance vars. No missing values, but class imbalance in `dropout_risk` (~21% True).

EDA confirmed the data's suitability for modeling but highlighted multicollinearity, guiding PCA use.

## Model Training & Evaluation
Models were trained on 80% of the data (800,000 rows) with 5-fold K-fold CV, GridSearchCV for tuning, and PCA (5 components, 97.29% variance) for dimensionality reduction.

### Score Prediction (Ridge Regression)
- **Training**: GridSearchCV tuned `alpha` = 1, CV R2 = 0.8948.
- **Evaluation**:
  - Train: MAE = 2.43, MSE = 9.29, RMSE = 3.05, R2 = 0.89.
  - Test: MAE = 2.44, MSE = 9.33, RMSE = 3.06, R2 = 0.89.
- **Comparison**: Outperforms linear regression baseline (R2 ~0.85) due to regularization.

### Pass/Fail Prediction (Calibrated Logistic Regression)
- **Training**: GridSearchCV tuned `C` and class weights, calibrated with sigmoid method.
- **Evaluation**: Accuracy = 0.90, Precision/Recall/F1 ~0.89-0.91, consistency = 0.86016.
- **Comparison**: Better than uncalibrated logistic (accuracy 0.85), balanced for slight imbalance.

### Dropout Risk Prediction (Calibrated Logistic Regression)
- **Training**: Similar to pass/fail, tuned for F1-score.
- **Evaluation**: Accuracy = 0.95, F1 = 0.89 for "Risk" class.
- **Comparison**: Handles imbalance better than naive Bayes (accuracy 0.92 but recall 0.80).

Overall, models show strong performance but consistency needs refinement.

## Challenges Faced & Improvements
- **Challenge**: High multicollinearity (VIF >100 for `past_score`, `quiz_accuracy`), causing model instability and contradictions (e.g., score 100, fail).
  - **Improvement**: Applied PCA to reduce to 5 components (97.29% variance), lowering effective VIF to ~1-5, and added consistency overrides (threshold 0.7).
- **Challenge**: Choosing the right LLM for educational purposes was difficult. Free models like Olam and Mistral 7B required significant computational power, which was impractical.
  - **Improvement**: Utilized Google Gemini 2.0-Flash via API access, providing efficient and scalable tutoring capabilities.
- **Challenge**: Finding suitable datasets for topic prediction on Kaggle, Hugging Face, and Google was unsuccessful, and synthetic data led to unsatisfactory ML model performance for this task.
  - **Improvement**: Adopted zero-shot classification with DeBERTa-v3-large, which is easy to implement. Messages are POST-ed to HF-Space `/predict`, labeled, and stored in TiDB’s `log_table` for future preprocessing and BERT model training.
- **Challenge**: Running the app locally failed due to terminal crashes. Streamlit Cloud was tried, but code changes and pushes were time-consuming, especially for debugging errors.
  - **Improvement**: Switched to Colab + Pyngrok setup, creating a Ngrok account for quick testing and stable public URLs.
- **Challenge**: Creating the synthetic dataset initially resulted in a random dataset unsuitable for ML training. Finding the right algorithm matching real-world data was hard and time-consuming.
  - **Improvement**: Evaluated 5 algorithms, selected the best dataset, and trained models. High VIF led to contradictions, so retrained all models with the same scaling, PCA, and feature order to minimize issues. Added app verification: if predicted score ≥60, show "Pass" to further arrest 
- **Challenge**: Imbalanced classes in `dropout_risk` (~21% True), leading to low recall (0.88).
  - **Improvement**: Applied `class_weight='balanced'` and F1 scoring in GridSearchCV.
- **Challenge**: Lower consistency accuracy (0.86).
  - **Improvement**: Derived pass/fail from scores (≥60) and overrode low-probability predictions.

These fixes improved R2 from ~0.30 (earlier attempts) to 0.89 and reduced contradictions.

## Future Enhancements
- **Enhanced AI Model**: Integrate the Google Gemini Flash 2.0 experimental model for improved tutoring capabilities.
- **Voice Integration**: Add voice input/output options to the Streamlit app for a more accessible user experience.
- **Interactive Features**: Implement quiz-making buttons to facilitate seamless interaction with the LLM.
- **Manual Mode**: Introduce a manual mode for unpredictable students, allowing LLM control to tailor responses (e.g., short answers for hard workers, detailed explanations for less active students).
- **Real Data Testing**: Validate models with real TiDB data to enhance accuracy and relevance.
- **BERT Training**: Preprocess saved topic-labeled data from TiDB and train a BERT model for improved topic prediction.
- **App Scalability**: Deploy on cloud platforms like AWS or GCP with real-time TiDB syncing.