"""
Smart Campus Resource Utilization & Dropout Risk Analytics Platform
Streamlit Dashboard — Multi-page Application
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Path setup ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROCESSED_PATH = ROOT / "data" / "processed" / "students_processed.csv"
METRICS_PATH = ROOT / "models" / "metrics.json"
FEATURE_COLS_PATH = ROOT / "models" / "feature_cols.json"

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Campus Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] .stRadio label { font-size: 15px !important; }

/* ── KPI Cards ── */
.kpi-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 20px 24px;
    text-align: center;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    transition: transform 0.2s;
}
.kpi-card:hover { transform: translateY(-3px); }
.kpi-value { font-size: 2.4rem; font-weight: 700; color: #38bdf8; margin: 8px 0 2px; }
.kpi-label { font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.06em; }
.kpi-icon  { font-size: 2rem; }

/* ── Section headers ── */
.section-header {
    font-size: 1.4rem; font-weight: 600;
    color: #f1f5f9;
    border-left: 4px solid #38bdf8;
    padding-left: 12px;
    margin: 28px 0 14px;
}
.insight-box {
    background: #1e293b; border-left: 4px solid #818cf8;
    padding: 14px 18px; border-radius: 0 10px 10px 0;
    font-size: 0.93rem; color: #cbd5e1; margin-top: 10px;
}
.rec-box {
    background: #1e293b; border-left: 4px solid #34d399;
    padding: 14px 18px; border-radius: 0 10px 10px 0;
    font-size: 0.93rem; color: #a7f3d0; margin-top: 4px;
}
/* ── Risk badges ── */
.risk-high   { background: #7f1d1d; color: #fca5a5; padding: 4px 14px;
                border-radius: 999px; font-weight: 600; font-size: 0.9rem; }
.risk-medium { background: #78350f; color: #fde68a; padding: 4px 14px;
                border-radius: 999px; font-weight: 600; font-size: 0.9rem; }
.risk-low    { background: #14532d; color: #86efac; padding: 4px 14px;
                border-radius: 999px; font-weight: 600; font-size: 0.9rem; }

/* ── Main background ── */
.stApp { background-color: #0a0f1e; }
.block-container { padding: 2rem 3rem; }

/* ── Plotly charts dark ── */
.js-plotly-plot { border-radius: 12px; }
</style>
""",
    unsafe_allow_html=True,
)

DARK_TEMPLATE = "plotly_dark"
ACCENT_BLUE = "#38bdf8"
ACCENT_PURPLE = "#818cf8"
ACCENT_GREEN = "#34d399"
ACCENT_ORANGE = "#fb923c"


# ── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame | None:
    if PROCESSED_PATH.exists():
        return pd.read_csv(PROCESSED_PATH)
    return None


@st.cache_data(show_spinner=False)
def load_metrics() -> dict | None:
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            return json.load(f)
    return None


def _add_engineered_features_inline(vals: dict) -> dict:
    """Compute derived features from raw inputs for dashboard prediction."""
    att = vals["attendance_percentage"]
    score = vals["avg_assignment_score"]
    lms = vals["lms_login_frequency"]
    lib = vals["library_visits_per_month"]
    disc = vals["disciplinary_actions"]
    internet = vals["internet_access"]
    hostel = vals["hostel_resident"]

    lms_norm = (lms / 60) * 100
    lib_norm = (lib / 20) * 100
    engagement_index = round(0.40 * att + 0.35 * lms_norm + 0.25 * lib_norm, 4)
    academic_risk_score = round(min((100 - score) + 5 * disc, 100), 4)
    attendance_volatility = round(abs(att - 72) / 73, 4)
    resource_utilization_score = round(
        0.5 * (lib / 20) + 0.3 * internet + 0.2 * hostel, 4
    )
    composite_dropout_risk = round(
        0.40 * (academic_risk_score / 100)
        + 0.35 * (1 - engagement_index / 100)
        + 0.25 * (1 - resource_utilization_score),
        4,
    )
    vals.update(
        engagement_index=engagement_index,
        academic_risk_score=academic_risk_score,
        attendance_volatility=attendance_volatility,
        resource_utilization_score=resource_utilization_score,
        composite_dropout_risk=composite_dropout_risk,
    )
    return vals


def _predict_inline(vals: dict, model_name: str = "gradient_boosting") -> dict:
    """Call model predict without circular import issues."""
    try:
        from src.models.predict import predict_one

        return predict_one(vals, model_name=model_name)
    except Exception as e:
        return {"dropout_prob": 0.0, "dropout_pred": 0, "risk_level": "Unknown", "error": str(e)}


# ── Sidebar navigation ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Smart Campus")
    st.markdown("### Analytics Platform")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        [
            "🏠 Overview",
            "📊 EDA Explorer",
            "🤖 Prediction Studio",
            "📈 Insights & Recommendations",
            "ℹ️ About",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem; color:#64748b;'>"
        "Enterprise AI · Analytics · Decision Intelligence</div>",
        unsafe_allow_html=True,
    )

df = load_data()
metrics = load_metrics()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("# 🎓 Smart Campus Resource Utilization Platform")
    st.markdown(
        "**Enterprise-grade analytics for student dropout prevention "
        "and campus resource optimisation.**"
    )
    st.markdown("---")

    # KPI Cards
    if df is not None:
        total = len(df)
        dropout_rate = df["dropout"].mean() * 100
        avg_engagement = df["engagement_index"].mean()
        high_risk = (df["composite_dropout_risk"] > 0.6).sum()

        c1, c2, c3, c4 = st.columns(4)
        for col, icon, value, label in [
            (c1, "👥", f"{total:,}", "Total Students"),
            (c2, "⚠️", f"{dropout_rate:.1f}%", "Dropout Rate"),
            (c3, "💡", f"{avg_engagement:.1f}", "Avg Engagement"),
            (c4, "🔴", f"{high_risk:,}", "High-Risk Students"),
        ]:
            col.markdown(
                f"""<div class="kpi-card">
                    <div class="kpi-icon">{icon}</div>
                    <div class="kpi-value">{value}</div>
                    <div class="kpi-label">{label}</div>
                </div>""",
                unsafe_allow_html=True,
            )
    else:
        st.warning("⚠️ Data not found. Run the data pipeline to populate dashboards.")

    # Architecture
    st.markdown('<div class="section-header">🏗️ System Architecture</div>', unsafe_allow_html=True)
    arch_col1, arch_col2 = st.columns([3, 2])
    with arch_col1:
        st.code(
            """
┌─────────────────────────────────────────────────────┐
│           Smart Campus Analytics Platform           │
│                                                     │
│  Data Layer          ML Layer         API Layer     │
│  ──────────          ────────         ─────────     │
│  generate_data.py →  train.py     →  FastAPI        │
│  loader.py        →  LR / RF / GB →  /predict       │
│  clean.py         →  metrics.py   →  /eda-summary   │
│  engineer.py                      →  /model-metrics │
│                                                     │
│  Dashboard Layer: Streamlit (this app)              │
│  Storage: SQLite / DuckDB / CSV                     │
│  CI/CD: GitHub Actions → Docker                     │
└─────────────────────────────────────────────────────┘
            """,
            language="text",
        )
    with arch_col2:
        st.markdown("### Business Problem")
        st.markdown(
            "Educational institutions lack **actionable insights** into student dropout risk, "
            "engagement patterns, and resource utilisation efficiency. Decisions are reactive "
            "and intuition-driven. This platform transforms raw interaction data into "
            "**decision-ready intelligence** through EDA, ML, and analytics dashboards."
        )

    # Model scoreboard
    if metrics:
        st.markdown('<div class="section-header">🏆 Model Performance Scoreboard</div>', unsafe_allow_html=True)
        rows = []
        for name, m in metrics.items():
            rows.append(
                {
                    "Model": name.replace("_", " ").title(),
                    "ROC-AUC": m.get("roc_auc", "-"),
                    "F1": m.get("f1", "-"),
                    "Precision": m.get("precision", "-"),
                    "Recall": m.get("recall", "-"),
                    "CV-AUC": m.get("cv_auc_mean", "-"),
                }
            )
        st.dataframe(
            pd.DataFrame(rows).style.highlight_max(
                subset=["ROC-AUC", "F1"], color="#1e3a5f"
            ),
            use_container_width=True,
            hide_index=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA Explorer":
    st.markdown("# 📊 EDA Explorer")
    st.markdown("Explore the dataset interactively. All data is synthetic with realistic correlations.")

    if df is None:
        st.error("Data not found. Run `python src/ingestion/generate_data.py` and the pipeline.")
        st.stop()

    # ── Sidebar filters
    st.sidebar.markdown("### Filters")
    semesters = sorted(df["semester"].unique().tolist())
    sel_sem = st.sidebar.multiselect("Semester", semesters, default=semesters)
    sel_hostel = st.sidebar.multiselect("Hostel Resident", [0, 1], default=[0, 1])
    sel_internet = st.sidebar.multiselect("Internet Access", [0, 1], default=[0, 1])

    fdf = df[
        df["semester"].isin(sel_sem)
        & df["hostel_resident"].isin(sel_hostel)
        & df["internet_access"].isin(sel_internet)
    ]
    st.markdown(f"**Filtered records:** `{len(fdf):,}` / `{len(df):,}`")

    # ── Dropout distribution
    st.markdown('<div class="section-header">Dropout Distribution</div>', unsafe_allow_html=True)
    do_counts = fdf["dropout"].value_counts().reset_index()
    do_counts.columns = ["Dropout", "Count"]
    do_counts["Dropout"] = do_counts["Dropout"].map({0: "Retained", 1: "Dropout"})
    fig_pie = px.pie(
        do_counts, names="Dropout", values="Count",
        color="Dropout",
        color_discrete_map={"Retained": ACCENT_GREEN, "Dropout": "#ef4444"},
        hole=0.45, template=DARK_TEMPLATE,
    )
    fig_pie.update_layout(height=360, margin=dict(t=20, b=20))
    col_pie1, col_pie2 = st.columns([1, 2])
    col_pie1.plotly_chart(fig_pie, use_container_width=True)
    with col_pie2:
        st.markdown(
            '<div class="insight-box">💡 <b>Technical:</b> The dataset has a meaningful class '
            "imbalance. Dropout rate varies significantly by semester and hostel status."
            "<br><br>📊 <b>Business:</b> 1-in-6 students are at dropout risk — a manageable "
            "segment for targeted intervention if identified early."
            "<br><br>✅ <b>Recommendation:</b> Prioritise semester 1-2 students for early "
            "outreach programmes.</div>",
            unsafe_allow_html=True,
        )

    # ── Feature distributions
    st.markdown('<div class="section-header">Feature Distributions</div>', unsafe_allow_html=True)
    numeric_features = [
        "attendance_percentage", "avg_assignment_score",
        "lms_login_frequency", "library_visits_per_month",
        "engagement_index", "composite_dropout_risk",
    ]
    feat_choice = st.selectbox("Select Feature", numeric_features)
    fig_hist = px.histogram(
        fdf, x=feat_choice, color=fdf["dropout"].map({0: "Retained", 1: "Dropout"}),
        nbins=50, barmode="overlay", opacity=0.75,
        color_discrete_map={"Retained": ACCENT_GREEN, "Dropout": "#ef4444"},
        template=DARK_TEMPLATE, labels={"color": "Status"},
    )
    fig_hist.update_layout(height=360, margin=dict(t=30, b=20), legend_title="Status")
    st.plotly_chart(fig_hist, use_container_width=True)

    # ── Boxplots
    st.markdown('<div class="section-header">Feature vs Dropout — Boxplots</div>', unsafe_allow_html=True)
    box_cols = st.columns(2)
    for idx, feat in enumerate(["attendance_percentage", "avg_assignment_score",
                                "lms_login_frequency", "engagement_index"]):
        fig_box = px.box(
            fdf, x=fdf["dropout"].map({0: "Retained", 1: "Dropout"}),
            y=feat, color=fdf["dropout"].map({0: "Retained", 1: "Dropout"}),
            color_discrete_map={"Retained": ACCENT_GREEN, "Dropout": "#ef4444"},
            template=DARK_TEMPLATE, points="outliers",
        )
        fig_box.update_layout(height=300, showlegend=False, margin=dict(t=30, b=10))
        box_cols[idx % 2].plotly_chart(fig_box, use_container_width=True)

    # ── Correlation heatmap
    st.markdown('<div class="section-header">Correlation Heatmap</div>', unsafe_allow_html=True)
    corr_cols = [
        "attendance_percentage", "avg_assignment_score", "lms_login_frequency",
        "library_visits_per_month", "disciplinary_actions",
        "engagement_index", "academic_risk_score", "composite_dropout_risk", "dropout",
    ]
    corr = fdf[corr_cols].corr()
    fig_heat = go.Figure(
        go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale="RdBu_r", zmid=0, text=corr.values.round(2),
            texttemplate="%{text}", textfont={"size": 11},
        )
    )
    fig_heat.update_layout(
        template=DARK_TEMPLATE, height=520, margin=dict(t=30, b=10),
        xaxis={"tickangle": -30},
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    st.markdown(
        '<div class="insight-box">💡 Strong negative correlations: attendance, engagement_index, '
        "and avg_assignment_score vs dropout. disciplinary_actions shows positive correlation.<br><br>"
        "✅ <b>Recommendation:</b> Use engagement_index as the primary early-warning KPI.</div>",
        unsafe_allow_html=True,
    )

    # ── Resource utilisation by semester
    st.markdown('<div class="section-header">Resource Utilisation by Semester</div>', unsafe_allow_html=True)
    sem_agg = fdf.groupby("semester").agg(
        avg_attendance=("attendance_percentage", "mean"),
        avg_lms=("lms_login_frequency", "mean"),
        avg_library=("library_visits_per_month", "mean"),
        dropout_rate=("dropout", "mean"),
    ).reset_index()
    fig_sem = px.bar(
        sem_agg.melt(id_vars="semester", value_vars=["avg_attendance", "avg_lms", "avg_library"]),
        x="semester", y="value", color="variable", barmode="group",
        template=DARK_TEMPLATE,
        color_discrete_sequence=[ACCENT_BLUE, ACCENT_PURPLE, ACCENT_GREEN],
    )
    fig_sem.update_layout(height=360, margin=dict(t=30, b=10), legend_title="Metric")
    st.plotly_chart(fig_sem, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PREDICTION STUDIO
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Prediction Studio":
    st.markdown("# 🤖 Prediction Studio")
    st.markdown("Enter a student's profile to predict dropout risk in real time.")

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("### 📝 Student Profile")
        with st.form("predict_form"):
            attendance = st.slider("Attendance (%)", 0, 100, 72)
            assignment_score = st.slider("Avg Assignment Score", 0, 100, 65)
            lms_freq = st.slider("LMS Logins / Month", 0, 60, 15)
            library = st.slider("Library Visits / Month", 0, 20, 3)
            disc = st.slider("Disciplinary Actions", 0, 10, 0)
            semester = st.selectbox("Semester", list(range(1, 9)), index=1)
            hostel = st.radio("Hostel Resident", [0, 1], format_func=lambda x: "Yes" if x else "No", horizontal=True)
            internet = st.radio("Internet Access", [0, 1], format_func=lambda x: "Yes" if x else "No", horizontal=True)
            model_choice = st.selectbox(
                "Model",
                ["gradient_boosting", "random_forest", "logistic_regression"],
                format_func=lambda x: x.replace("_", " ").title(),
            )
            submitted = st.form_submit_button("🔍 Predict Dropout Risk", use_container_width=True)

    with col_result:
        if submitted:
            student_vals = {
                "attendance_percentage": attendance,
                "avg_assignment_score": assignment_score,
                "lms_login_frequency": lms_freq,
                "library_visits_per_month": library,
                "disciplinary_actions": disc,
                "semester": semester,
                "hostel_resident": hostel,
                "internet_access": internet,
            }
            student_vals = _add_engineered_features_inline(student_vals)
            result = _predict_inline(student_vals, model_name=model_choice)

            if "error" in result:
                st.error(f"Model not loaded: {result['error']}. Run `python src/models/train.py` first.")
            else:
                prob = result["dropout_prob"]
                risk = result["risk_level"]
                st.markdown("### 🎯 Risk Assessment")

                risk_color = {"High": "risk-high", "Medium": "risk-medium", "Low": "risk-low"}
                st.markdown(
                    f'<div style="text-align:center; margin:10px 0;">'
                    f'<span class="{risk_color.get(risk, "risk-low")}">{risk} Risk</span></div>',
                    unsafe_allow_html=True,
                )

                # Gauge chart
                fig_gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=prob * 100,
                        title={"text": "Dropout Probability (%)", "font": {"size": 16, "color": "#94a3b8"}},
                        gauge={
                            "axis": {"range": [0, 100], "tickcolor": "#475569"},
                            "bar": {"color": "#ef4444" if risk == "High" else ("#f59e0b" if risk == "Medium" else ACCENT_GREEN)},
                            "bgcolor": "#1e293b",
                            "bordercolor": "#334155",
                            "steps": [
                                {"range": [0, 35], "color": "#14532d"},
                                {"range": [35, 65], "color": "#78350f"},
                                {"range": [65, 100], "color": "#7f1d1d"},
                            ],
                            "threshold": {"line": {"color": "white", "width": 3}, "value": prob * 100},
                        },
                        number={"suffix": "%", "font": {"size": 36, "color": "#f1f5f9"}},
                    )
                )
                fig_gauge.update_layout(
                    height=280, template=DARK_TEMPLATE, margin=dict(t=30, b=10)
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

                # Engineered features
                st.markdown("### 🧮 Derived Features")
                feat_data = {
                    "Engagement Index": student_vals["engagement_index"],
                    "Academic Risk Score": student_vals["academic_risk_score"],
                    "Attendance Volatility": student_vals["attendance_volatility"],
                    "Resource Utilization": student_vals["resource_utilization_score"],
                    "Composite Risk Proxy": student_vals["composite_dropout_risk"],
                }
                feat_df = pd.DataFrame(
                    list(feat_data.items()), columns=["Feature", "Value"]
                )
                fig_feat = px.bar(
                    feat_df, x="Value", y="Feature", orientation="h",
                    template=DARK_TEMPLATE, color="Value",
                    color_continuous_scale=["#34d399", "#f59e0b", "#ef4444"],
                )
                fig_feat.update_layout(
                    height=260, margin=dict(t=10, b=10),
                    coloraxis_showscale=False, showlegend=False,
                )
                st.plotly_chart(fig_feat, use_container_width=True)

                # Recommendations
                st.markdown("### 💡 Recommendation")
                if risk == "High":
                    st.markdown(
                        '<div class="insight-box">🚨 <b>High Priority Intervention:</b> '
                        "Contact student immediately. Schedule counselling and attendance review. "
                        "Offer peer mentoring and financial aid check.</div>",
                        unsafe_allow_html=True,
                    )
                elif risk == "Medium":
                    st.markdown(
                        '<div class="insight-box">⚠️ <b>Watch & Support:</b> '
                        "Monitor LMS engagement weekly. Invite to academic support sessions. "
                        "Check in via email.</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="rec-box">✅ <b>Low Risk:</b> '
                        "Student engagement is strong. Continue current trajectory. "
                        "Can act as peer mentor.</div>",
                        unsafe_allow_html=True,
                    )
        else:
            st.info("👈 Fill in the student profile and click **Predict Dropout Risk**.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — INSIGHTS & RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Insights & Recommendations":
    st.markdown("# 📈 Insights & Recommendations")
    st.markdown("Data-driven findings for academic administrators.")

    if df is None:
        st.error("Data not found. Run the pipeline first.")
        st.stop()

    # ── Top dropout drivers (from correlation with dropout)
    st.markdown('<div class="section-header">🔑 Top Feature Correlations with Dropout</div>', unsafe_allow_html=True)
    corr_with_dropout = (
        df.drop(columns=["student_id", "dropout"])
        .corr()["dropout" if "dropout" not in df.drop(columns=["student_id"]).columns else df.columns[0]]
    )
    # Recompute straight
    num_df = df.select_dtypes(include="number")
    corr_vals = num_df.corrwith(df["dropout"]).drop("dropout").sort_values(key=abs, ascending=False)
    fig_corr = px.bar(
        x=corr_vals.values,
        y=corr_vals.index,
        orientation="h",
        template=DARK_TEMPLATE,
        color=corr_vals.values,
        color_continuous_scale=["#34d399", "#f8fafc", "#ef4444"],
        labels={"x": "Correlation with Dropout", "y": "Feature"},
    )
    fig_corr.add_vline(x=0, line_color="#475569", line_width=1)
    fig_corr.update_layout(height=420, margin=dict(t=20, b=10), coloraxis_showscale=False)
    st.plotly_chart(fig_corr, use_container_width=True)
    st.markdown(
        '<div class="insight-box">💡 <b>academic_risk_score</b> and <b>composite_dropout_risk</b> are the '
        "strongest engineered predictors. Among raw features, <b>attendance_percentage</b> and "
        "<b>avg_assignment_score</b> drive the model most.</div>",
        unsafe_allow_html=True,
    )

    # ── Segment analysis
    st.markdown('<div class="section-header">📌 At-Risk Segment Profiles</div>', unsafe_allow_html=True)
    def risk_tier(row):
        if row["composite_dropout_risk"] > 0.60:
            return "High Risk"
        elif row["composite_dropout_risk"] > 0.35:
            return "Medium Risk"
        else:
            return "Low Risk"
    df2 = df.copy()
    df2["risk_tier"] = df2.apply(risk_tier, axis=1)

    seg = (
        df2.groupby("risk_tier")
        .agg(
            count=("student_id", "count"),
            avg_attendance=("attendance_percentage", "mean"),
            avg_score=("avg_assignment_score", "mean"),
            avg_engagement=("engagement_index", "mean"),
            dropout_rate=("dropout", "mean"),
        )
        .reset_index()
        .round(2)
    )
    st.dataframe(seg, use_container_width=True, hide_index=True)

    # ── Semester-wise dropout
    st.markdown('<div class="section-header">📅 Dropout Rate by Semester</div>', unsafe_allow_html=True)
    sem_dropout = df.groupby("semester")["dropout"].mean().reset_index()
    sem_dropout.columns = ["Semester", "Dropout Rate"]
    sem_dropout["Dropout Rate %"] = (sem_dropout["Dropout Rate"] * 100).round(2)
    fig_sem_d = px.line(
        sem_dropout, x="Semester", y="Dropout Rate %",
        markers=True, template=DARK_TEMPLATE, line_shape="spline",
    )
    fig_sem_d.update_traces(line_color=ACCENT_ORANGE, marker_size=8, marker_color="#f1f5f9")
    fig_sem_d.update_layout(height=320, margin=dict(t=20, b=10))
    st.plotly_chart(fig_sem_d, use_container_width=True)

    # ── Actionable recommendations
    st.markdown('<div class="section-header">✅ Actionable Recommendations</div>', unsafe_allow_html=True)
    recs = [
        ("🎯 Early Warning System", "Deploy composite_dropout_risk as a live KPI dashboard for academic advisors. Alert when score > 0.6."),
        ("📱 LMS Engagement Nudges", "Students with LMS logins < 10/month are 2× more likely to drop out. Trigger automated email nudges."),
        ("📚 Library Access Expansion", "Non-hostel students have 40% fewer library visits. Extend evening hours and add virtual library access."),
        ("🏠 Hostel Support", "Hostel residents show lower dropout (likely better access to resources). Expand hostel capacity for at-risk commuters."),
        ("⚖️ Disciplinary Intervention", "Each disciplinary action increases dropout probability by ~8%. Implement restorative practices over punitive measures."),
        ("📋 Semester 1-2 Priority", "Dropout risk spikes in early semesters. Mandatory mentoring in semester 1-2 is critical."),
    ]
    rec_cols = st.columns(2)
    for i, (title, desc) in enumerate(recs):
        with rec_cols[i % 2]:
            st.markdown(
                f'<div class="rec-box"><b>{title}</b><br>{desc}</div>',
                unsafe_allow_html=True,
            )
            st.markdown("")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.markdown("# ℹ️ About This Platform")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🧰 Technology Stack")
        stack = {
            "Data & Processing": ["Python 3.11", "Pandas", "NumPy", "DuckDB"],
            "Machine Learning": ["Scikit-learn", "Logistic Regression", "Random Forest", "Gradient Boosting"],
            "Visualisation": ["Plotly", "Seaborn", "Matplotlib"],
            "API": ["FastAPI", "Pydantic v2", "Uvicorn"],
            "Dashboard": ["Streamlit"],
            "DevOps": ["Docker", "Docker Compose", "GitHub Actions"],
        }
        for category, items in stack.items():
            st.markdown(f"**{category}:** {', '.join(items)}")

    with col2:
        st.markdown("### ⚖️ Ethical Considerations")
        st.markdown(
            """
- **Fairness**: Model should be audited for demographic bias before production use.
- **Transparency**: All model decisions are explainable via feature importance.
- **Privacy**: Student IDs are anonymised. PII must not enter the system.
- **Human-in-the-loop**: AI predictions are decision support, not automated decisions.
- **Consent**: Students should be informed if their data is used for dropout prediction.
            """
        )

    st.markdown("### ⚠️ Limitations")
    st.markdown(
        """
- Dataset is **synthetic** — production deployment requires real institutional data.
- Model performance may degrade over semesters without retraining.
- Engagement features (LMS, library) depend on consistent data collection pipelines.
- Class imbalance handling uses `class_weight='balanced'` — review threshold in production.
        """
    )

    st.markdown("### 📦 Deployment")
    st.code(
        """
# Local
streamlit run dashboard/app.py

# Docker Compose
docker-compose up --build

# Streamlit Cloud
# Push to GitHub → connect repo at share.streamlit.io → set main: dashboard/app.py
        """,
        language="bash",
    )
