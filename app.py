import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report
)

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Failure Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #c0392b;
        text-align: center;
        padding: 0.4rem 0 0.2rem 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #555;
        text-align: center;
        margin-bottom: 1.4rem;
    }
    .metric-card {
        background: #f7f8fa;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border-left: 5px solid #c0392b;
    }
    .risk-high {
        background: #fdecea;
        border-radius: 10px;
        padding: 1.2rem;
        border: 2px solid #e74c3c;
        text-align: center;
    }
    .risk-low {
        background: #eafaf1;
        border-radius: 10px;
        padding: 1.2rem;
        border: 2px solid #27ae60;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
NUMERICAL_FEATURES   = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
CATEGORICAL_FEATURES = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
TARGET               = "HeartDisease"

FEATURE_DESCRIPTIONS = {
    "Age":           "Age of the patient (years)",
    "Sex":           "Patient sex (M / F)",
    "ChestPainType": "Chest pain type (TA / ATA / NAP / ASY)",
    "RestingBP":     "Resting blood pressure (mm Hg)",
    "Cholesterol":   "Serum cholesterol (mg/dL)",
    "FastingBS":     "Fasting blood sugar > 120 mg/dL (1 = True, 0 = False)",
    "RestingECG":    "Resting ECG result (Normal / ST / LVH)",
    "MaxHR":         "Maximum heart rate achieved",
    "ExerciseAngina":"Exercise-induced angina (Y / N)",
    "Oldpeak":       "ST depression induced by exercise",
    "ST_Slope":      "Slope of peak exercise ST segment (Up / Flat / Down)",
}

# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("heart.csv")
    return df

# ──────────────────────────────────────────────
# Model training
# ──────────────────────────────────────────────
@st.cache_resource
def train_model(df: pd.DataFrame):
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy  = accuracy_score(y_test, y_pred)
    cm        = confusion_matrix(y_test, y_pred)
    report    = classification_report(y_test, y_pred, output_dict=True)

    # Feature importance names
    cat_encoder = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    cat_names   = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    feat_names  = NUMERICAL_FEATURES + cat_names

    importances = pipeline.named_steps["classifier"].feature_importances_
    feat_imp_df = pd.DataFrame({"Feature": feat_names, "Importance": importances})
    feat_imp_df = feat_imp_df.sort_values("Importance", ascending=False).head(12)

    return pipeline, accuracy, cm, report, feat_imp_df, X_test, y_test, y_pred

# ──────────────────────────────────────────────
# App header
# ──────────────────────────────────────────────
st.markdown('<div class="main-header">❤️ Heart Failure Prediction System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Machine Learning-powered clinical decision support · '
    'Random Forest · 918 patient records</div>',
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# Load data & model
# ──────────────────────────────────────────────
try:
    df = load_data()
except FileNotFoundError:
    st.error("❌ `heart.csv` not found. Please place it in the same directory as `app.py`.")
    st.stop()

pipeline, accuracy, cm, report, feat_imp_df, X_test, y_test, y_pred = train_model(df)

# ──────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊 Exploratory Data Analysis",
    "🔮 Heart Failure Risk Prediction",
    "📈 Model Evaluation Metrics",
])

# ══════════════════════════════════════════════
# TAB 1 – EDA Dashboard
# ══════════════════════════════════════════════
with tab1:
    st.subheader("Dataset Overview")

    disease_count  = int(df[TARGET].sum())
    normal_count   = int(len(df) - disease_count)
    disease_rate   = round(disease_count / len(df) * 100, 1)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🧾 Total Records",    len(df))
    col2.metric("🧬 Features",         len(NUMERICAL_FEATURES) + len(CATEGORICAL_FEATURES))
    col3.metric("💔 Heart Disease",    f"{disease_count} ({disease_rate}%)")
    col4.metric("✅ Normal",            normal_count)

    st.divider()

    # Row 1: Pie + Age-MaxHR scatter
    row1_left, row1_right = st.columns(2)

    with row1_left:
        st.markdown("#### Class Distribution")
        fig_pie = px.pie(
            values=[normal_count, disease_count],
            names=["Normal (0)", "Heart Disease (1)"],
            color_discrete_sequence=["#27ae60", "#e74c3c"],
            hole=0.45,
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20), showlegend=True)
        st.plotly_chart(fig_pie, use_container_width=True)

    with row1_right:
        st.markdown("#### Age vs. Max Heart Rate (by Diagnosis)")
        fig_scatter = px.scatter(
            df, x="Age", y="MaxHR",
            color=df[TARGET].map({0: "Normal", 1: "Heart Disease"}),
            color_discrete_map={"Normal": "#27ae60", "Heart Disease": "#e74c3c"},
            opacity=0.65,
            labels={"color": "Diagnosis"},
        )
        fig_scatter.update_layout(margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Row 2: Chest Pain Type + ST_Slope
    row2_left, row2_right = st.columns(2)

    with row2_left:
        st.markdown("#### Chest Pain Type vs. Heart Disease")
        cp_counts = (
            df.groupby(["ChestPainType", TARGET])
            .size()
            .reset_index(name="Count")
        )
        cp_counts["Diagnosis"] = cp_counts[TARGET].map({0: "Normal", 1: "Heart Disease"})
        fig_cp = px.bar(
            cp_counts, x="ChestPainType", y="Count", color="Diagnosis",
            barmode="group",
            color_discrete_map={"Normal": "#27ae60", "Heart Disease": "#e74c3c"},
            labels={"ChestPainType": "Chest Pain Type"},
        )
        fig_cp.update_layout(margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_cp, use_container_width=True)

    with row2_right:
        st.markdown("#### ST_Slope vs. Heart Disease")
        st_counts = (
            df.groupby(["ST_Slope", TARGET])
            .size()
            .reset_index(name="Count")
        )
        st_counts["Diagnosis"] = st_counts[TARGET].map({0: "Normal", 1: "Heart Disease"})
        fig_st = px.bar(
            st_counts, x="ST_Slope", y="Count", color="Diagnosis",
            barmode="stack",
            color_discrete_map={"Normal": "#27ae60", "Heart Disease": "#e74c3c"},
            labels={"ST_Slope": "ST Slope"},
        )
        fig_st.update_layout(margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_st, use_container_width=True)

    # Row 3: Correlation heatmap
    st.markdown("#### Numerical Feature Correlation Heatmap")
    corr_df = df[NUMERICAL_FEATURES + [TARGET]].corr()
    fig_heat, ax_heat = plt.subplots(figsize=(9, 4))
    sns.heatmap(
        corr_df, annot=True, fmt=".2f", cmap="RdYlGn_r",
        linewidths=0.5, ax=ax_heat,
    )
    ax_heat.set_title("Correlation Matrix", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig_heat, use_container_width=True)
    plt.close(fig_heat)

# ══════════════════════════════════════════════
# TAB 2 – Prediction
# ══════════════════════════════════════════════
with tab2:
    st.subheader("Enter Patient Clinical Data")

    with st.form("prediction_form"):
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            age          = st.slider("Age (years)",          20, 100, 50)
            resting_bp   = st.slider("Resting BP (mm Hg)",   80, 200, 120)
            cholesterol  = st.slider("Cholesterol (mg/dL)",   0, 600, 220)
            fasting_bs   = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1], format_func=lambda x: "Yes (1)" if x else "No (0)")

        with col_b:
            max_hr       = st.slider("Max Heart Rate",        60, 220, 150)
            oldpeak      = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            sex          = st.selectbox("Sex", ["M", "F"])
            chest_pain   = st.selectbox("Chest Pain Type", ["ASY", "ATA", "NAP", "TA"],
                                        help="ASY=Asymptomatic, ATA=Atypical Angina, NAP=Non-Anginal, TA=Typical Angina")

        with col_c:
            resting_ecg  = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"],
                                        help="ST=ST-T wave abnormality, LVH=Left Ventricular Hypertrophy")
            ex_angina    = st.selectbox("Exercise-Induced Angina", ["N", "Y"])
            st_slope     = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

        submitted = st.form_submit_button("🔍 Predict Heart Failure Risk", use_container_width=True)

    if submitted:
        input_data = pd.DataFrame([{
            "Age":            age,
            "RestingBP":      resting_bp,
            "Cholesterol":    cholesterol,
            "FastingBS":      fasting_bs,
            "MaxHR":          max_hr,
            "Oldpeak":        oldpeak,
            "Sex":            sex,
            "ChestPainType":  chest_pain,
            "RestingECG":     resting_ecg,
            "ExerciseAngina": ex_angina,
            "ST_Slope":       st_slope,
        }])

        prediction   = pipeline.predict(input_data)[0]
        probabilities = pipeline.predict_proba(input_data)[0]
        risk_prob    = round(float(probabilities[1]) * 100, 1)

        st.divider()
        res_left, res_right = st.columns([1, 1])

        with res_left:
            if prediction == 1:
                st.markdown(
                    f'<div class="risk-high">'
                    f'<h2 style="color:#c0392b;">⚠️ HIGH RISK</h2>'
                    f'<p style="font-size:1.1rem;">Predicted: <strong>Heart Disease</strong></p>'
                    f'<p style="font-size:1.6rem; font-weight:700; color:#c0392b;">{risk_prob}% risk probability</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="risk-low">'
                    f'<h2 style="color:#27ae60;">✅ LOW RISK</h2>'
                    f'<p style="font-size:1.1rem;">Predicted: <strong>Normal</strong></p>'
                    f'<p style="font-size:1.6rem; font-weight:700; color:#27ae60;">{risk_prob}% risk probability</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        with res_right:
            st.markdown("#### Risk Probability Gauge")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_prob,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Heart Disease Risk (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": "#e74c3c" if risk_prob >= 50 else "#27ae60"},
                    "steps": [
                        {"range": [0,  40], "color": "#d5f5e3"},
                        {"range": [40, 70], "color": "#fdebd0"},
                        {"range": [70, 100],"color": "#fadbd8"},
                    ],
                    "threshold": {
                        "line":  {"color": "#c0392b", "width": 4},
                        "thickness": 0.75,
                        "value": 50,
                    },
                },
            ))
            fig_gauge.update_layout(height=280, margin=dict(t=30, b=10, l=10, r=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

        # Clinical risk factor highlights
        st.divider()
        st.markdown("#### 🔑 Key Clinical Risk Indicators")
        flags = []
        if oldpeak > 2.0:
            flags.append("🔴 **Oldpeak** is elevated (>2.0) — strong predictor of ischemia")
        if chest_pain == "ASY":
            flags.append("🔴 **Asymptomatic chest pain** is the highest-risk chest pain type")
        if st_slope in ("Flat", "Down"):
            flags.append(f"🔴 **ST Slope ({st_slope})** is associated with higher cardiac risk")
        if ex_angina == "Y":
            flags.append("🟡 **Exercise-induced angina** present — warrants further evaluation")
        if max_hr < 120:
            flags.append("🟡 **Low Max Heart Rate** (<120 bpm) may indicate reduced cardiac reserve")
        if age > 60:
            flags.append("🟡 **Age > 60** — cardiovascular risk increases with age")
        if cholesterol > 300:
            flags.append("🟡 **High Cholesterol** (>300 mg/dL) — dyslipidemia risk factor")
        if fasting_bs == 1:
            flags.append("🟡 **Elevated Fasting Blood Sugar** — potential diabetic risk")

        if flags:
            for flag in flags:
                st.markdown(f"- {flag}")
        else:
            st.success("No critical individual risk factors identified in this profile.")

# ══════════════════════════════════════════════
# TAB 3 – Model Evaluation
# ══════════════════════════════════════════════
with tab3:
    st.subheader("Model Evaluation Metrics")

    # Summary metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🎯 Accuracy",  f"{accuracy * 100:.1f}%")
    m2.metric("🔵 Precision (Dis.)", f"{report['1']['precision']*100:.1f}%")
    m3.metric("🟢 Recall (Dis.)",    f"{report['1']['recall']*100:.1f}%")
    m4.metric("⚖️ F1-Score (Dis.)",  f"{report['1']['f1-score']*100:.1f}%")

    st.divider()

    eval_left, eval_right = st.columns(2)

    with eval_left:
        st.markdown("#### Confusion Matrix")
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Reds",
            xticklabels=["Predicted Normal", "Predicted Disease"],
            yticklabels=["Actual Normal",    "Actual Disease"],
            ax=ax_cm, linewidths=0.5,
        )
        ax_cm.set_title("Confusion Matrix", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig_cm, use_container_width=True)
        plt.close(fig_cm)

    with eval_right:
        st.markdown("#### Feature Importance (Top 12)")
        fig_fi = px.bar(
            feat_imp_df,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Reds",
        )
        fig_fi.update_layout(
            yaxis={"autorange": "reversed"},
            coloraxis_showscale=False,
            margin=dict(t=20, b=20, l=10, r=10),
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    # Classification report table
    st.divider()
    st.markdown("#### Classification Report")
    report_rows = []
    for label_key, label_name in [("0", "Normal"), ("1", "Heart Disease"),
                                   ("macro avg", "Macro Avg"),
                                   ("weighted avg", "Weighted Avg")]:
        if label_key in report:
            row = report[label_key]
            report_rows.append({
                "Class":     label_name,
                "Precision": f"{row['precision']:.3f}",
                "Recall":    f"{row['recall']:.3f}",
                "F1-Score":  f"{row['f1-score']:.3f}",
                "Support":   int(row["support"]) if "support" in row else "—",
            })
    st.dataframe(pd.DataFrame(report_rows), use_container_width=True, hide_index=True)

    # Algorithm note
    st.divider()
    st.info(
        "**Model:** Random Forest Classifier (200 trees, balanced class weights)  \n"
        "**Preprocessing:** OneHotEncoding for categorical features, passthrough for numerical  \n"
        "**Split:** 80% train / 20% test, stratified  \n"
        "**Dataset:** 918 records, 11 clinical features"
    )
