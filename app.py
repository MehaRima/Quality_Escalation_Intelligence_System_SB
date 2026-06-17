import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from data_utils import generate_synthetic_quality_data, prepare_quality_data, date_filter, rule_risk, recommendations

st.set_page_config(page_title="Quality Escalation Intelligence", page_icon="🚦", layout="wide")

st.markdown("""
<style>
.block-container {
    padding-top: 0.75rem;
}

.headercard {
    background: #f0fdf4;
    color: #14532d;
    padding: 14px 18px;
    border-radius: 16px;
    border: 1px solid #bbf7d0;
    margin-bottom: 14px;
}

.headercard h1 {
    font-size: 2.05rem;
    line-height: 1.15;
    margin: 0 0 6px 0;
    font-weight: 700;
}

.headercard p {
    font-size: 1rem;
    margin: 0;
}

.headercard * {
    color: #14532d !important;
}

div[data-testid="stMetric"] {
    background: #f7fee7;
    border: 1px solid #d9f99d;
    border-radius: 14px;
    padding: 10px 12px;
    min-height: 92px;
}

div[data-testid="stMetric"] label,
div[data-testid="stMetric"] div,
div[data-testid="stMetric"] p {
    color: #1a2e05 !important;
}

div[data-testid="stMetricValue"] {
    font-size: 1.85rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="headercard"><h1>🚦 Quality Escalation Intelligence System</h1><p>Prioritize quality issues, estimate escalation risk, and support corrective-action planning.</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Quality Data")
    up=st.file_uploader("Upload quality issue CSV", type=["csv"])
    n=st.slider("Demo issues", 500, 8000, 3000, step=500)
    if up:
        try: df=prepare_quality_data(pd.read_csv(up)); src="Uploaded CSV"
        except Exception as e: st.error(e); df=generate_synthetic_quality_data(n); src="Synthetic Demo"
    else:
        df=generate_synthetic_quality_data(n); src="Synthetic Demo"
    df=rule_risk(df)
    st.success(src)
    period=st.selectbox("Issue window", ["All Data","Last 30 Days","Last 90 Days","Last 180 Days"])
    df=date_filter(df, period)
    team_opts=["All Teams"]+sorted(df["owner_team"].unique().tolist())
    team=st.selectbox("Owner team", team_opts)
    if team!="All Teams": df=df[df["owner_team"]==team]
    st.info(f"{len(df):,} issues in view")

if df.empty:
    st.warning("No quality issues available for this selection.")
    st.stop()

a,b,c,d=st.columns(4)
a.metric("Quality Issues", f"{len(df):,}")
b.metric("High Risk", f"{(df['risk_band'].astype(str)=='High').sum():,}")
c.metric("Escalation Rate", f"{df['escalated'].mean()*100:.1f}%")
d.metric("Median Age", f"{df['age_days'].median():.1f} days")

left,right=st.columns([1,1])
with left:
    st.subheader("Escalation Risk Queue")
    queue=df.sort_values("escalation_risk_score", ascending=False).head(25)
    st.dataframe(queue, use_container_width=True, hide_index=True)
with right:
    st.subheader("Risk Distribution")
    fig=px.histogram(df, x="escalation_risk_score", color="risk_band", nbins=25, title="Escalation Risk Score Distribution")
    st.plotly_chart(fig, use_container_width=True)

t1,t2,t3=st.tabs(["Risk Drivers","Escalation Model","Corrective Actions"])
with t1:
    st.subheader("Quality Risk Drivers")
    g=df.groupby("issue_type").agg(issues=("issue_type","count"), avg_risk=("escalation_risk_score","mean"), escalation_rate=("escalated","mean")).reset_index()
    fig=px.scatter(g, x="issues", y="avg_risk", size="escalation_rate", hover_name="issue_type", title="Issue Volume vs Escalation Risk")
    st.plotly_chart(fig, use_container_width=True)
with t2:
    st.subheader("Simple Escalation Model")
    model_df=pd.get_dummies(df[["severity","issue_type","channel","owner_team","age_days","repeat_issue","customer_impact","escalated"]].dropna(), drop_first=True)
    if model_df["escalated"].nunique() > 1 and len(model_df)>100:
        X=model_df.drop(columns=["escalated"]); y=model_df["escalated"]
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,random_state=4,stratify=y)
        clf=RandomForestClassifier(n_estimators=120, random_state=4, max_depth=6).fit(X_train,y_train)
        auc=roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
        st.metric("Validation ROC-AUC", f"{auc:.3f}")
        imp=pd.DataFrame({"feature":X.columns,"importance":clf.feature_importances_}).sort_values("importance", ascending=False).head(12)
        fig=px.bar(imp, x="importance", y="feature", orientation="h", title="Model Feature Importance")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need enough labeled escalated/non-escalated examples to train the model.")
with t3:
    st.subheader("Corrective-Action Recommendations")
    for i,r in enumerate(recommendations(df),1): st.write(f"**{i}.** {r}")
    st.write("- Define ownership for aging high-risk issues.")
    st.write("- Review recurring issues separately from one-time defects.")
    st.write("- Use impact score and age together, not severity alone.")
