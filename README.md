# ❤️ Heart Failure Prediction System

A complete, end-to-end Machine Learning web application that predicts the risk of heart failure based on 11 clinical features. Built with **Streamlit** for the interactive UI and **Scikit-Learn** for the ML pipeline — all in a single cohesive `app.py` file.

---

## 🖥️ Live Application Preview

| Tab | Description |
|-----|-------------|
| 📊 Exploratory Data Analysis | Interactive charts, class distribution, correlation heatmap |
| 🔮 Heart Failure Risk Prediction | Clinical input form, risk probability gauge, key risk flags |
| 📈 Model Evaluation Metrics | Accuracy, confusion matrix, feature importance, classification report |

---

## 📁 Project Structure

```
Heart_Failure_Prediction/
├── app.py                              # Main Streamlit application
├── heart.csv                           # Dataset (918 records, 12 columns)
├── requirements.txt                    # Python dependencies
├── README.md                           # This documentation file
└── python_script_to_generate_docx.py  # Script to export the technical report
```

---

## 📊 Dataset Overview

**Source:** UCI Machine Learning Repository — Heart Failure Prediction Dataset
**Kaggle Dataset:** [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
**Records:** 918 | **Target:** `HeartDisease` (0 = Normal, 1 = Heart Disease)
**Disease Prevalence:** ~55.3%

### Feature Reference Table

| Feature | Type | Description | Values / Range |
|---------|------|-------------|----------------|
| `Age` | Numerical | Patient age in years | 28 – 77 |
| `Sex` | Categorical | Biological sex | M, F |
| `ChestPainType` | Categorical | Type of chest pain experienced | TA, ATA, NAP, ASY |
| `RestingBP` | Numerical | Resting blood pressure (mm Hg) | 0 – 200 |
| `Cholesterol` | Numerical | Serum cholesterol (mg/dL) | 0 – 603 |
| `FastingBS` | Numerical | Fasting blood sugar > 120 mg/dL | 0 (No), 1 (Yes) |
| `RestingECG` | Categorical | Resting ECG results | Normal, ST, LVH |
| `MaxHR` | Numerical | Maximum heart rate achieved (bpm) | 60 – 202 |
| `ExerciseAngina` | Categorical | Exercise-induced angina | N, Y |
| `Oldpeak` | Numerical | ST depression induced by exercise | -2.6 – 6.2 |
| `ST_Slope` | Categorical | Slope of peak exercise ST segment | Up, Flat, Down |
| `HeartDisease` | Target | Presence of heart disease | 0 (Normal), 1 (Disease) |

**Categorical Value Descriptions:**
- **ChestPainType:** TA = Typical Angina · ATA = Atypical Angina · NAP = Non-Anginal Pain · ASY = Asymptomatic
- **RestingECG:** Normal · ST = ST-T wave abnormality · LVH = Left Ventricular Hypertrophy
- **ST_Slope:** Up (upsloping) · Flat · Down (downsloping)

---

## 🏗️ Architecture & ML Pipeline

```
heart.csv
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Data Loading (@st.cache_data)                       │
│  pandas.read_csv("heart.csv")                        │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  Preprocessing (ColumnTransformer)                   │
│  ├─ Numerical:   passthrough (Age, BP, …)            │
│  └─ Categorical: OneHotEncoder (Sex, CPT, …)         │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  Model Training (@st.cache_resource)                 │
│  RandomForestClassifier                              │
│  ├─ n_estimators = 200                               │
│  ├─ random_state = 42                                │
│  ├─ class_weight = "balanced"                        │
│  └─ train/test split = 80% / 20% (stratified)        │
└──────────────────────┬──────────────────────────────┘
                       │
              ┌────────┴────────┐
              ▼                 ▼
    ┌──────────────────┐  ┌─────────────────────┐
    │  Prediction API  │  │  Evaluation Metrics  │
    │  predict()       │  │  accuracy, CM,       │
    │  predict_proba() │  │  report, feat_imp    │
    └──────────────────┘  └─────────────────────┘
              │
              ▼
    Streamlit UI Tabs (EDA | Prediction | Metrics)
```

**Key design choices:**
- `@st.cache_resource` ensures the model is trained **once** at startup and reused across all sessions.
- `@st.cache_data` caches the CSV read for fast re-renders.
- The full Scikit-Learn `Pipeline` object (preprocessor + classifier) is stored so that new prediction inputs are transformed consistently.

---

## ⚙️ Installation & Execution

### Prerequisites
- Python **3.9 – 3.12** (recommended: 3.11)
- `pip` package manager

### 1 — Clone / download the project

```bash
git clone <your-repo-url>
cd Heart_Failure_Prediction
```

### 2 — (Optional) Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### 4 — Run the Streamlit application

```bash
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.
---

## 🤖 Model Performance (on 20% hold-out test set)

| Metric | Value |
|--------|-------|
| **Accuracy** | ~89% |
| **Precision (Disease)** | ~90% |
| **Recall (Disease)** | ~91% |
| **F1-Score (Disease)** | ~90% |

> Actual values displayed live in the **Model Evaluation Metrics** tab.

---

## 🔑 Top Predictive Features

Based on Random Forest feature importance scores:

1. `ST_Slope_Flat` / `ST_Slope_Up`
2. `Oldpeak`
3. `ChestPainType_ASY`
4. `MaxHR`
5. `ExerciseAngina_Y`
6. `Age`
7. `Sex_M`

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | 1.35.0 | Web UI framework |
| pandas | 2.2.2 | Data manipulation |
| numpy | 1.26.4 | Numerical computing |
| scikit-learn | 1.5.0 | ML pipeline & model |
| plotly | 5.22.0 | Interactive charts |
| seaborn | 0.13.2 | Statistical heatmaps |
| matplotlib | 3.9.0 | Static plot backend |
| python-docx | 1.1.2 | DOCX report generation |

---

## ⚠️ Disclaimer

This application is built for **educational and research purposes only**. It is **not** a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical decisions.

---

## 📜 License

MIT License — free to use, modify, and distribute with attribution.
