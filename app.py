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
search_logs = pd.read_csv(LOG_PATH)

# --- ADVANCED SEARCH ENGINE ---
def process_intelligent_search(query, df):
    if not query: return df
    
    q = query.lower()
    filtered_df = df.copy()

    # 1. Descriptive Magnitude Parsing (Keywords to Logic)
    logic_maps = {
        r"low latency": filtered_df['latency'] < 40,
        r"high latency": filtered_df['latency'] > 90,
        r"high accuracy": filtered_df['accuracy'] > 0.95,
        r"low accuracy": filtered_df['accuracy'] < 0.80,
        r"usage (very )?high": filtered_df['usage'] > 10000,
        r"low usage": filtered_df['usage'] < 1000,
        r"high drift": filtered_df['data_drift'] > 0.1,
        r"low drift": filtered_df['data_drift'] < 0.03,
    }
    for phrase, mask in logic_maps.items():
        if re.search(phrase, q):
            filtered_df = filtered_df[mask]

    # 2. Numerical Operator Parsing (e.g., latency < 50, cpu > 80)
    # Pattern: [column name] [operator] [value]
    patterns = {
        'latency': r'latency\s*([<>]=?)\s*(\d+)',
        'accuracy': r'accuracy\s*([<>]=?)\s*(\d+)',
        'usage': r'usage\s*([<>]=?)\s*(\d+)',
        'cpu_util': r'cpu\s*utilization\s*([<>]=?)\s*(\d+)|cpu\s*([<>]=?)\s*(\d+)',
        'data_drift': r'drift\s*([<>]=?)\s*(0\.\d+|\d+)'
    }

    for col, pattern in patterns.items():
        match = re.search(pattern, q)
        if match:
            # Handle groups because CPU has two possible regex matches
            groups = [g for g in match.groups() if g is not None]
            op, val = groups[0], float(groups[1])
            
            # Normalize percentage for accuracy
            if col == 'accuracy' and val > 1: val = val / 100
            
            if op == '<': filtered_df = filtered_df[filtered_df[col] < val]
            elif op == '>': filtered_df = filtered_df[filtered_df[col] > val]
            elif op == '<=': filtered_df = filtered_df[filtered_df[col] <= val]
            elif op == '>=': filtered_df = filtered_df[filtered_df[col] >= val]

    if filtered_df.empty: return filtered_df

    # 3. Semantic Keyword Search (TF-IDF)
    filtered_df['blob'] = filtered_df.fillna('').astype(str).apply(' '.join, axis=1)
    vec = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
    mtx = vec.fit_transform(filtered_df['blob'].tolist() + [query])
    scores = cosine_similarity(mtx[-1], mtx[:-1])[0]
    filtered_df['relevance'] = scores
    
    results = filtered_df[filtered_df['relevance'] > 0.001].sort_values('relevance', ascending=False)
    
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
                <tr><td class="meta-label">Endpoint</td><td class="meta-val">{row['inference_endpoint_id']}</td></tr>
            </table>
            <div class="use-case-text"><b>Case:</b> {row['use_cases']}</div>
        </div>
        <div class="metric-grid">
            <div class="metric-item"><span class="metric-label">Acc</span><span class="metric-num">{int(row['accuracy']*100)}%</span></div>
            <div class="metric-item"><span class="metric-label">Lat</span><span class="metric-num">{row['latency']}ms</span></div>
            <div class="metric-item"><span class="metric-label">CPU</span><span class="metric-num">{row['cpu_util']}%</span></div>
        </div>
    </div>
    """

# --- NAVIGATION ---
with st.sidebar:
    st.title("Marketplace")
    user = st.selectbox("Current User", ["John Doe", "Jane Nu", "Sam King", "Nat Patel (Leader)", "Admin"])
    st.divider()
    st.caption("Search Examples:")
    st.info("'latency < 50ms'\n'usage very high'\n'Finance high accuracy'\n'cpu < 30%'")

# --- VIEW: DATA SCIENTIST ---
if user in ["John Doe", "Jane Nu", "Sam King"]:
    t1, t2, t3 = st.tabs(["🏛 Unified Gallery", "🚀 Ingest Asset", "📊 My Portfolio"])
    
    with t1:
        query = st.text_input("💬 Ask the Hub (e.g., 'Finance low latency and cpu < 50%')", placeholder="Type query...")
        results = process_intelligent_search(query, df_master)
        
        st.write(f"Showing {len(results)} results")
        for i in range(0, min(len(results), 30), 3):
            cols = st.columns(3)
            for j in range(3):
                if i+j < len(results):
                    row = results.iloc[i+j]
                    with cols[j]:
                        st.markdown(render_model_tile(row), unsafe_allow_html=True)
                        if st.button("Request Access", key=f"req_{i+j}_{row['name']}"):
                            # Direct write to ensure Nat Patel sees it
                            new_req = pd.DataFrame([{"model_name": row['name'], "requester": user, "status": "Pending", "timestamp": str(datetime.datetime.now())}])
                            new_req.to_csv(REQ_PATH, mode='a', header=False, index=False)
                            st.toast(f"Request for {row['name']} sent to Nat Patel.")

    with t2:
        with st.form("ingest_form", clear_on_submit=True):
            st.subheader("Asset Ingestion")
            n = st.text_input("Model Name*")
            d = st.text_input("Domain")
            desc = st.text_area("Detailed Use Case (Mention stage, team, hardware)*")
            if st.form_submit_button("Publish"):
                if n and desc:
                    new_row = pd.DataFrame([{
                        "name": n, "model_version": "v1.0", "domain": d, "type": "Community",
                        "accuracy": 0.88, "latency": 45, "use_cases": desc, "contributor": user,
                        "monitoring_status": "Healthy", "model_stage": "Experimental", "usage": 0, "data_drift": 0.0,
                        "cpu_util": 20, "mem_util": 4, "throughput": 100, "error_rate": 0.01,
                        "model_owner_team": "Internal", "sla_tier": "Bronze", "training_data_source": "Manual",
                        "inference_endpoint_id": f"ep-{random.randint(100,999)}"
                    }])
                    new_row.to_csv(REG_PATH, mode='a', header=False, index=False)
                    st.success("Successfully Ingested and Persisted!")

    with t3:
        my_m = df_master[df_master['contributor'] == user]
        if not my_m.empty:
            sel = st.selectbox("Inspect Asset Telemetry", my_m['name'].unique())
            m_dat = my_m[my_m['name'] == sel].iloc[0]
            # RADAR CHART
            fig = go.Figure(go.Scatterpolar(
                r=[m_dat['accuracy']*100, 100-m_dat['data_drift']*100, 100-m_dat['cpu_util'], 100-m_dat['error_rate']*10],
                theta=['Accuracy','Stability','Efficiency','Reliability'],
                fill='toself', line_color='#6200EE'
            ))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, height=450)
            st.plotly_chart(fig, use_container_width=True)
        else: st.info("You have not contributed any assets yet.")

# --- VIEW: LEADER (NAT PATEL) ---
elif user == "Nat Patel (Leader)":
    st.subheader("Leader Approval Gateway")
    if os.path.exists(REQ_PATH):
        req_df = pd.read_csv(REQ_PATH)
        pending = req_df[req_df['status'] == "Pending"]
        if not pending.empty:
            for idx, r in pending.iterrows():
                c1, c2 = st.columns([3, 1])
                c1.write(f"💼 **{r['requester']}** requested **{r['model_name']}**")
                if c2.button("Approve", key=f"ap_{idx}"):
                    req_df.at[idx, 'status'] = "Approved"
                    req_df.to_csv(REQ_PATH, index=False)
                    st.rerun()
        else: st.success("Approval queue is clear.")

# --- VIEW: ADMIN ---
else:
    k1, k2, k3 = st.columns(3)
    k1.metric("API Consumption", f"{df_master['usage'].sum():,}")
    if not search_logs.empty:
        eff = (len(search_logs[search_logs['found'] > 0]) / len(search_logs)) * 100
        k2.metric("Search Efficacy", f"{int(eff)}%")
    k3.metric("Inventory", len(df_master))

    st.divider()
    selected_model = st.selectbox("Solo Inspector (Highlight Line)", ["None"] + list(df_master['name'].unique()))
    
    cv = [1.0 if name == selected_model else 0.0 for name in df_master['name']]
    cs = [[0, 'rgba(255, 249, 196, 0.4)'], [1, '#B71C1C']] # Pale Yellow vs Deep Red
    
    fig = go.Figure(data=go.Parcoords(
        labelfont=dict(size=11, color='black'),
        tickfont=dict(size=9, color='#666'),
        line=dict(color=cv, colorscale=cs, showscale=False),
        dimensions=list([
            dict(range=[0.7, 1.0], label='Accuracy', values=df_master['accuracy']),
            dict(range=[0, 200], label='Latency', values=df_master['latency']),
            dict(range=[0, 25000], label='Usage', values=df_master['usage']),
            dict(range=[0, 100], label='CPU %', values=df_master['cpu_util']),
            dict(range=[0, 0.25], label='Drift', values=df_master['data_drift'])
        ])
    ))
    fig.update_layout(margin=dict(t=80, b=40, l=70, r=70), height=500)
    st.plotly_chart(fig, use_container_width=True)
