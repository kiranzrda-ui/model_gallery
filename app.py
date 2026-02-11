import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import datetime
import csv
import random
import re

# --- CONFIG & COMPACT STYLING ---
st.set_page_config(page_title="Model Hub", layout="wide")

st.markdown("""
    <style>
    :root { --accent: #6200EE; --success: #2E7D32; --warning: #F9A825; --critical: #C62828; }
    .model-card {
        border: 1px solid #e0e0e0; border-top: 4px solid var(--accent);
        padding: 15px; background-color: #ffffff; margin-bottom: 20px;
        min-height: 420px; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        display: flex; flex-direction: column; justify-content: space-between;
    }
    .model-title { font-size: 1.1rem; font-weight: 700; color: #111; margin-bottom: 2px; }
    .status-pill { font-size: 0.65rem; padding: 2px 8px; border-radius: 10px; color: white; font-weight: bold; float: right; }
    .Healthy { background-color: var(--success); }
    .Warning { background-color: var(--warning); }
    .Critical { background-color: var(--critical); }
    .meta-table { font-size: 0.75rem; width: 100%; margin: 10px 0; border-collapse: collapse; }
    .meta-table td { padding: 4px 0; border-bottom: 1px solid #f9f9f9; }
    .meta-label { color: #888; width: 45%; }
    .meta-val { color: #333; font-weight: 500; text-align: right; }
    .use-case-text { font-size: 0.8rem; color: #555; margin: 10px 0; line-height: 1.4; height: 60px; overflow: hidden; }
    .metric-grid { display: flex; justify-content: space-between; background: #fafafa; padding: 10px; border-radius: 4px; border: 1px solid #eee; }
    .metric-item { text-align: center; }
    .metric-label { font-size: 0.65rem; color: #999; display: block; text-transform: uppercase; }
    .metric-num { font-size: 0.85rem; font-weight: 700; color: var(--accent); }
    .stButton>button { background-color: #111; color: #fff; border-radius: 4px; border: none; width: 100%; height: 35px; font-size: 0.8rem; }
    .stButton>button:hover { background-color: var(--accent); color: #fff; }
    </style>
    """, unsafe_allow_html=True)

# --- PATHS & DATA ---
REG_PATH = "model_registry_v3.csv"
REQ_PATH = "requests_v3.csv"
LOG_PATH = "search_efficacy_v3.csv"

def init_files():
    if not os.path.exists(REQ_PATH):
        pd.DataFrame(columns=["model_name", "requester", "status", "timestamp"]).to_csv(REQ_PATH, index=False)
    if not os.path.exists(LOG_PATH):
        pd.DataFrame(columns=["query", "found", "timestamp"]).to_csv(LOG_PATH, index=False)

init_files()
df_master = pd.read_csv(REG_PATH)
req_log = pd.read_csv(REQ_PATH)
search_logs = pd.read_csv(LOG_PATH)

# --- SEARCH LOGIC ---
def process_search(query, df):
    if not query: return df
    q = query.lower()
    # 1. Numerical Filtering (e.g., >90 accuracy)
    match = re.search(r'([><=]=?)\s*(\d+)', q)
    if match:
        op, val = match.groups()
        val = float(val)/100 if float(val) > 1 else float(val)
        if '>' in op: df = df[df['accuracy'] >= val]
        elif '<' in op: df = df[df['accuracy'] <= val]

    # 2. Textual Ranking
    df_search = df.copy()
    df_search['blob'] = df_search.fillna('').astype(str).apply(' '.join, axis=1)
    vec = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
    mtx = vec.fit_transform(df_search['blob'].tolist() + [query])
    scores = cosine_similarity(mtx[-1], mtx[:-1])[0]
    df_search['relevance'] = scores
    
    results = df_search[df_search['relevance'] > 0.01].sort_values('relevance', ascending=False)
    
    # Log efficacy
    new_log = pd.DataFrame([{"query": query, "found": len(results), "timestamp": str(datetime.datetime.now())}])
    pd.concat([search_logs, new_log]).to_csv(LOG_PATH, index=False)
    return results

# --- UI COMPONENTS ---
def render_model_tile(row):
    return f"""
    <div class="model-card">
        <div>
            <span class="status-pill {row['monitoring_status']}">{row['monitoring_status']}</span>
            <div class="model-title">{row['name']}</div>
            <div class="version-tag">{row['model_version']} | {row['type']}</div>
            <table class="meta-table">
                <tr><td class="meta-label">Domain</td><td class="meta-val">{row['domain']}</td></tr>
                <tr><td class="meta-label">Stage</td><td class="meta-val">{row['model_stage']}</td></tr>
                <tr><td class="meta-label">SLA Tier</td><td class="meta-val">{row['sla_tier']}</td></tr>
                <tr><td class="meta-label">Team</td><td class="meta-val">{row['model_owner_team']}</td></tr>
                <tr><td class="meta-label">Source</td><td class="meta-val">{row['training_data_source']}</td></tr>
            </table>
            <div class="use-case-text"><b>Use Case:</b> {row['use_cases']}</div>
        </div>
        <div class="metric-grid">
            <div class="metric-item"><span class="metric-label">Acc</span><span class="metric-num">{int(row['accuracy']*100)}%</span></div>
            <div class="metric-item"><span class="metric-label">Lat</span><span class="metric-num">{row['latency']}ms</span></div>
            <div class="metric-item"><span class="metric-label">Drift</span><span class="metric-num">{row['data_drift']}</span></div>
        </div>
    </div>
    """

# --- NAVIGATION ---
with st.sidebar:
    st.title("Hub")
    user = st.selectbox("Role", ["John Doe", "Jane Nu", "Sam King", "Nat Patel (Leader)", "Admin"])

# --- VIEW: DATA SCIENTIST ---
if user in ["John Doe", "Jane Nu", "Sam King"]:
    t1, t2, t3 = st.tabs(["Gallery", "Ingest", "My Portfolio"])
    
    with t1:
        query = st.text_input("💬 Smart Search (Keywords or Metrics like '>90')", placeholder="Search...")
        results = process_search(query, df_master)
        for i in range(0, min(len(results), 30), 3):
            cols = st.columns(3)
            for j in range(3):
                if i+j < len(results):
                    row = results.iloc[i+j]
                    with cols[j]:
                        st.markdown(render_model_tile(row), unsafe_allow_html=True)
                        if st.button("Request Access", key=f"req_{i+j}_{row['name']}"):
                            new_req = pd.DataFrame([{"model_name": row['name'], "requester": user, "status": "Pending", "timestamp": str(datetime.datetime.now())}])
                            pd.concat([req_log, new_req]).to_csv(REQ_PATH, index=False)
                            st.toast(f"Request sent for {row['name']}")

    with t2:
        with st.form("ingest"):
            n = st.text_input("Name*")
            d = st.text_input("Domain")
            desc = st.text_area("Detailed Description (Keywords like 'Shadow' help tagging)*")
            if st.form_submit_button("Submit"):
                if n and desc:
                    new_row = pd.DataFrame([{"name": n, "model_version": "v1.0", "domain": d, "type": "Community", "accuracy": 0.85, "latency": 45, "use_cases": desc, "contributor": user, "monitoring_status": "Healthy", "model_stage": "Shadow", "usage": 0, "data_drift": 0.0, "model_owner_team": "Strategy AI", "sla_tier": "Bronze", "training_data_source": "Input", "model_owner_team": "Data Science Hub"}])
                    pd.concat([df_master, new_row]).to_csv(REG_PATH, index=False)
                    st.success("Ingested!")

    with t3:
        my_m = df_master[df_master['contributor'] == user]
        if not my_m.empty:
            sel = st.selectbox("Inspect Asset", my_m['name'].unique())
            m_dat = my_m[my_m['name'] == sel].iloc[0]
            fig = go.Figure(go.Scatterpolar(r=[m_dat['accuracy']*100, 100-m_dat['data_drift']*100, 100-m_dat['cpu_util'], 100-m_dat['error_rate']*10], theta=['Accuracy','Stability','Efficiency','Reliability'], fill='toself'))
            st.plotly_chart(fig, use_container_width=True)
        else: st.info("No assets yet.")

# --- VIEW: LEADER ---
elif user == "Nat Patel (Leader)":
    st.subheader("Approval Gateway")
    # Refresh log
    current_reqs = pd.read_csv(REQ_PATH)
    pending = current_reqs[current_reqs['status'] == "Pending"]
    if not pending.empty:
        for idx, r in pending.iterrows():
            c1, c2 = st.columns([3, 1])
            c1.write(f"💼 **{r['requester']}** requested access to **{r['model_name']}**")
            if c2.button("Approve", key=f"ap_{idx}"):
                current_reqs.at[idx, 'status'] = "Approved"
                current_reqs.to_csv(REQ_PATH, index=False)
                st.rerun()
    else: st.success("Queue clear.")

# --- VIEW: ADMIN ---
else:
    k1, k2, k3 = st.columns(3)
    k1.metric("Global Usage", df_master['usage'].sum())
    if len(search_logs) > 0:
        eff = (len(search_logs[search_logs['found'] > 0]) / len(search_logs)) * 100
        k2.metric("Search Conversion", f"{int(eff)}%")
    k3.metric("Inventory", len(df_master))

    st.divider()
    selected_model = st.selectbox("Highlight Asset", ["None"] + list(df_master['name'].unique()))
    
    # Theme: Pale Yellow vs Red
    cv = [1.0 if name == selected_model else 0.0 for name in df_master['name']]
    cs = [[0, 'rgba(255, 249, 196, 0.4)'], [1, '#B71C1C']] # Pale Yellow vs Deep Red
    
    fig = go.Figure(data=go.Parcoords(
        labelfont=dict(size=10, color='black'),
        tickfont=dict(size=8, color='gray'),
        line=dict(color=cv, colorscale=cs, showscale=False),
        dimensions=list([
            dict(range=[0.7, 1.0], label='Accuracy', values=df_master['accuracy']),
            dict(range=[0, 200], label='Latency', values=df_master['latency']),
            dict(range=[0, 20000], label='Usage', values=df_master['usage']),
            dict(range=[0, 0.25], label='Drift', values=df_master['data_drift'])
        ])
    ))
    fig.update_layout(margin=dict(t=80, b=40, l=60, r=60), height=500)
    st.plotly_chart(fig, use_container_width=True)
