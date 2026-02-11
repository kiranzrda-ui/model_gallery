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
st.set_page_config(page_title="Model Gallery", layout="wide")

st.markdown("""
    <style>
    :root { --accent: #6200EE; --success: #2E7D32; --warning: #F9A825; --critical: #C62828; }
    
    .model-card {
        border: 1px solid #e0e0e0;
        border-top: 4px solid var(--accent);
        padding: 15px;
        background-color: #ffffff;
        margin-bottom: 20px;
        min-height: 400px;
        border-radius: 8px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    }
    .model-title { font-size: 1.1rem; font-weight: 700; color: #111; margin-bottom: 2px; }
    .status-pill { font-size: 0.65rem; padding: 2px 8px; border-radius: 10px; color: white; font-weight: bold; float: right; }
    .Healthy { background-color: var(--success); }
    .Warning { background-color: var(--warning); }
    .Critical { background-color: var(--critical); }

    .meta-table { font-size: 0.75rem; width: 100%; margin: 10px 0; border-collapse: collapse; }
    .meta-table td { padding: 4px 0; border-bottom: 1px solid #f9f9f9; }
    .meta-label { color: #888; width: 40%; }
    .meta-val { color: #333; font-weight: 500; text-align: right; }

    .use-case-text { font-size: 0.8rem; color: #555; margin: 10px 0; line-height: 1.4; height: 50px; overflow: hidden; }
    
    .metric-grid { display: flex; justify-content: space-between; background: #fafafa; padding: 10px; border-radius: 4px; border: 1px solid #eee; }
    .metric-item { text-align: center; }
    .metric-label { font-size: 0.65rem; color: #999; display: block; text-transform: uppercase; }
    .metric-num { font-size: 0.9rem; font-weight: 700; color: var(--accent); }

    .stButton>button { background-color: #111; color: #fff; border-radius: 4px; border: none; width: 100%; height: 35px; }
    .stButton>button:hover { background-color: var(--accent); color: #fff; }
    </style>
    """, unsafe_allow_html=True)

# --- DATA LAYER ---
REG_PATH = "model_registry_v3.csv"
REQ_PATH = "requests_v3.csv"
LOG_PATH = "search_efficacy_v3.csv"

def init_files():
    if not os.path.exists(REG_PATH):
        # Initial Seed
        pd.DataFrame(columns=[
            "name", "model_version", "domain", "type", "accuracy", "latency", "clients", 
            "use_cases", "contributor", "usage", "data_drift", "pred_drift", "cpu_util", 
            "mem_util", "throughput", "error_rate", "model_owner_team", "last_retrained_date", 
            "model_stage", "training_data_source", "approval_status", "monitoring_status", 
            "sla_tier", "feature_store_dependency", "inference_endpoint_id"
        ]).to_csv(REG_PATH, index=False)

init_files()
df_master = pd.read_csv(REG_PATH)
req_log = pd.read_csv(REQ_PATH) if os.path.exists(REQ_PATH) else pd.DataFrame(columns=["model_name", "requester", "status", "timestamp"])

# --- SEARCH & PARSING LOGIC ---
def natural_language_query(query, df):
    """Filters data based on keywords and math symbols in the search bar."""
    if not query: return df
    
    # 1. Metric Filtering (e.g., >90 accuracy)
    q = query.lower()
    match = re.search(r'([><=]=?)\s*(\d+)', q)
    if match:
        op, val = match.groups()
        val = float(val)/100 if float(val) > 1 else float(val)
        if '>' in op: df = df[df['accuracy'] >= val]
        elif '<' in op: df = df[df['accuracy'] <= val]
    
    # 2. Text Search
    df['blob'] = df.fillna('').astype(str).apply(' '.join, axis=1)
    vec = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
    mtx = vec.fit_transform(df['blob'].tolist() + [query])
    scores = cosine_similarity(mtx[-1], mtx[:-1])[0]
    df['relevance'] = scores
    return df[df['relevance'] > 0.01].sort_values('relevance', ascending=False)

# --- NAVIGATION ---
with st.sidebar:
    st.title("Settings")
    user = st.selectbox("Switch User", ["John Doe", "Jane Nu", "Sam King", "Nat Patel (Leader)", "Admin"])

# --- DATA SCIENTIST VIEW ---
if user in ["John Doe", "Jane Nu", "Sam King"]:
    st.subheader("Model Gallery")
    t1, t2, t3 = st.tabs(["Search Gallery", "Contribute", "My Assets"])

    with t1:
        query = st.text_input("💬 Search anything (e.g. 'Finance Prod >90 accuracy')", placeholder="Keywords or metrics...")
        results = natural_language_query(query, df_master)

        for i in range(0, min(len(results), 30), 3):
            cols = st.columns(3)
            for j in range(3):
                if i+j < len(results):
                    row = results.iloc[i+j]
                    with cols[j]:
                        # HTML string built carefully to avoid markdown parsing errors
                        card_html = f"""
                        <div class="model-card">
                            <div>
                                <span class="status-pill {row['monitoring_status']}">{row['monitoring_status']}</span>
                                <div class="model-title">{row['name']}</div>
                                <div class="version-tag">{row['model_version']} | {row['type']}</div>
                                
                                <table class="meta-table">
                                    <tr><td class="meta-label">Domain</td><td class="meta-val">{row['domain']}</td></tr>
                                    <tr><td class="meta-label">Stage</td><td class="meta-val">{row['model_stage']}</td></tr>
                                    <tr><td class="meta-label">Team</td><td class="meta-val">{row['model_owner_team']}</td></tr>
                                    <tr><td class="meta-label">SLA Tier</td><td class="meta-val">{row['sla_tier']}</td></tr>
                                    <tr><td class="meta-label">Data Source</td><td class="meta-val">{row['training_data_source']}</td></tr>
                                </table>
                                
                                <div class="use-case-text"><b>Use Case:</b> {row['use_cases']}</div>
                            </div>
                            
                            <div>
                                <div class="metric-grid">
                                    <div class="metric-item"><span class="metric-label">Acc</span><span class="metric-num">{int(row['accuracy']*100)}%</span></div>
                                    <div class="metric-item"><span class="metric-label">Lat</span><span class="metric-num">{row['latency']}ms</span></div>
                                    <div class="metric-item"><span class="metric-label">Drift</span><span class="metric-num">{row['data_drift']}</span></div>
                                </div>
                                <div style="height:15px;"></div>
                            </div>
                        </div>
                        """
                        st.markdown(card_html, unsafe_allow_html=True)
                        if st.button("Request Access", key=f"req_{i+j}_{row['name']}"):
                            st.toast(f"Request sent for {row['name']}")

    with t2:
        st.write("### Register a New Model")
        with st.form("ingest", clear_on_submit=True):
            n = st.text_input("Model Name*")
            d = st.text_input("Domain")
            desc = st.text_area("Detailed Description (Include Stage, SLA, Team info)*")
            if st.form_submit_button("Submit"):
                # Basic inference from text
                stage = "Shadow" if "shadow" in desc.lower() else "Prod"
                new_row = pd.DataFrame([{
                    "name": n, "model_version": "v1.0", "domain": d, "type": "Community",
                    "accuracy": 0.85, "latency": 50, "use_cases": desc, "contributor": user,
                    "monitoring_status": "Healthy", "model_stage": stage, "usage": 0, "data_drift": 0,
                    "model_owner_team": "Internal", "sla_tier": "Bronze", "training_data_source": "Manual"
                }])
                pd.concat([df_master, new_row]).to_csv(REG_PATH, index=False)
                st.success("Model submitted and searchable!")

    with t3:
        st.dataframe(df_master[df_master['contributor'] == user])

# --- ADMIN VIEW ---
elif user == "Admin":
    st.subheader("Global Metrics & Governance")
    k1, k2 = st.columns(2)
    k1.metric("Total Assets", len(df_master))
    k2.metric("Total Invocations", df_master['usage'].sum())

    st.divider()
    st.write("### Multi-Dimensional Performance Inspector")
    
    # Selection logic for highlighting
    selected_model = st.selectbox("Highlight Specific Asset", ["None"] + list(df_master['name'].unique()))
    
    plot_df = df_master.copy()
    colors = [1.0 if name == selected_model else 0.0 for name in plot_df['name']]
    
    fig = go.Figure(data=go.Parcoords(
        labelfont=dict(size=11, color='black'), # Smaller font to fix smudging
        tickfont=dict(size=9, color='gray'),
        line=dict(color=colors, colorscale=[[0, 'rgba(161,0,255,0.1)'], [1, 'red']], showscale=False),
        dimensions=list([
            dict(range=[0.7, 1.0], label='Accuracy', values=plot_df['accuracy']),
            dict(range=[0, 200], label='Latency', values=plot_df['latency']),
            dict(range=[0, 20000], label='Usage', values=plot_df['usage']),
            dict(range=[0, 0.3], label='Drift', values=plot_df['data_drift']),
            dict(range=[0, 100], label='CPU %', values=plot_df['cpu_util'])
        ])
    ))
    
    fig.update_layout(margin=dict(t=80, b=40, l=50, r=50), height=450)
    st.plotly_chart(fig, use_container_width=True)

# --- LEADER VIEW ---
else:
    st.write("### Approval Queue")
    st.info("No pending requests.")
