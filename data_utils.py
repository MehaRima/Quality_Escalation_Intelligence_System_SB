import pandas as pd
import numpy as np

TYPES=["Content Error","Process Gap","Data Mismatch","Access Issue","Communication Gap","Review Delay","System Defect"]
TEAMS=["QA","Assessment","Operations","Content","Support","Training"]

def generate_synthetic_quality_data(n=3000, seed=12):
    rng=np.random.default_rng(seed)
    sev=rng.choice(["Low","Medium","High","Critical"], n, p=[.28,.42,.22,.08])
    age=np.clip(rng.gamma(2,6,n)+np.where(sev=="Critical",8,0)+np.where(sev=="High",4,0),0,90).round(1)
    repeat=rng.binomial(1, np.where(np.isin(sev,["High","Critical"]),.32,.12))
    impact=np.clip(rng.normal(35,20,n)+np.where(sev=="Critical",35,0)+np.where(sev=="High",20,0),0,100).round(1)
    prob=.08 + (sev=="High")*.22 + (sev=="Critical")*.45 + repeat*.18 + (age>14)*.12 + (impact>70)*.18
    escalated=rng.binomial(1, np.clip(prob,0,0.9))
    return pd.DataFrame({
        "issue_date":pd.Timestamp.today().normalize()-pd.to_timedelta(rng.integers(0,365,n),unit="D"),
        "issue_type":rng.choice(TYPES,n),
        "severity":sev,
        "channel":rng.choice(["Portal","Email","Audit","Review Queue","Stakeholder"],n),
        "owner_team":rng.choice(TEAMS,n),
        "status":rng.choice(["Open","In Review","Resolved","Blocked"],n,p=[.22,.28,.42,.08]),
        "age_days":age,"repeat_issue":repeat,"customer_impact":impact,"escalated":escalated
    })

def prepare_quality_data(df):
    df=df.copy(); cols={c.lower().strip().replace(" ","_"):c for c in df.columns}
    def pick(ns):
        for n in ns:
            if n in cols: return cols[n]
        return None
    n=len(df); out=pd.DataFrame()
    out["issue_date"]=pd.to_datetime(df[pick(["issue_date","date","created_date"])], errors="coerce") if pick(["issue_date","date","created_date"]) else pd.Timestamp.today()
    out["issue_type"]=df[pick(["issue_type","category","type"])].astype(str) if pick(["issue_type","category","type"]) else "Process Gap"
    out["severity"]=df[pick(["severity","priority"])].astype(str) if pick(["severity","priority"]) else "Medium"
    out["channel"]=df[pick(["channel","source"])].astype(str) if pick(["channel","source"]) else "Portal"
    out["owner_team"]=df[pick(["owner_team","team","department"])].astype(str) if pick(["owner_team","team","department"]) else "QA"
    out["status"]=df[pick(["status","state"])].astype(str) if pick(["status","state"]) else "Open"
    out["age_days"]=pd.to_numeric(df[pick(["age_days","age","days_open"])], errors="coerce") if pick(["age_days","age","days_open"]) else np.random.gamma(2,6,n)
    out["repeat_issue"]=pd.to_numeric(df[pick(["repeat_issue","repeat","recurring"])], errors="coerce") if pick(["repeat_issue","repeat","recurring"]) else 0
    out["customer_impact"]=pd.to_numeric(df[pick(["customer_impact","impact_score","impact"])], errors="coerce") if pick(["customer_impact","impact_score","impact"]) else 40
    out["escalated"]=pd.to_numeric(df[pick(["escalated","is_escalated","escalation"])], errors="coerce") if pick(["escalated","is_escalated","escalation"]) else 0
    out["issue_date"]=out["issue_date"].fillna(pd.Timestamp.today())
    out["age_days"]=out["age_days"].fillna(out["age_days"].median()).clip(0,365)
    out["repeat_issue"]=out["repeat_issue"].fillna(0).clip(0,1).astype(int)
    out["customer_impact"]=out["customer_impact"].fillna(40).clip(0,100)
    out["escalated"]=out["escalated"].fillna(0).clip(0,1).astype(int)
    return out

def date_filter(df, mode):
    if mode=="All Data": return df
    days={"Last 30 Days":30,"Last 90 Days":90,"Last 180 Days":180}[mode]
    return df[df["issue_date"] >= df["issue_date"].max()-pd.Timedelta(days=days)]

def rule_risk(df):
    sev_map={"Low":10,"Medium":35,"High":65,"Critical":90}
    s=df["severity"].map(sev_map).fillna(35)
    age=(df["age_days"].rank(pct=True)*25)
    rep=df["repeat_issue"]*15
    impact=df["customer_impact"]*.25
    df=df.copy()
    df["escalation_risk_score"]=(s*.45+age+rep+impact).clip(0,100).round(1)
    df["risk_band"]=pd.cut(df["escalation_risk_score"], bins=[-1,40,70,101], labels=["Low","Watch","High"])
    return df

def recommendations(df):
    rec=[]
    high=df[df["risk_band"].astype(str)=="High"]
    if len(high):
        t=high["issue_type"].value_counts().index[0]
        rec.append(f"Prioritize **{t}** issues in the escalation review queue.")
    if (df["repeat_issue"]==1).mean()>0.18:
        rec.append("Repeat issue rate is elevated. Add corrective-action tracking for recurring causes.")
    if df["age_days"].median()>10:
        rec.append("Median issue age is high. Review aging open items twice weekly.")
    rec.append("Use high-risk queue as a daily triage list for owner teams.")
    return rec
