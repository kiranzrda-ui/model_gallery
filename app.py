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

# --- CONFIG & STYLING ---
st.set_page_config(page_title="Model Hub", layout="wide")

st.markdown("""
    <style>
    :root { --accent: #6200EE; --success: #2E7D32; --warning: #F9A825; --critical: #C62828; }
    .model-card {
        border: 1px solid #e0e0e0; border-top: 4px solid var(--accent);
        padding: 15px; background-color: #ffffff; margin-bottom: 20px;
        min-height: 440px; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        display: flex; flex-direction: column; justify-content: space-between;
    }
    .model-title { font-size: 1.1rem; font-weight: 700; color: #111; margin-bottom: 2px; }
    .version-tag { font-size: 0.75rem; color: #666; font-family: monospace; }
    .status-pill { font-size: 0.65rem; padding: 2px 8px; border-radius: 10px; color: white; font-weight: bold; float: right; }
    .Healthy { background-color: var(--success); }
    .Warning { background-color: var(--warning); }
    .Critical { background-color: var(--critical); }
    
    .meta-table { font-size: 0.75rem; width: 100%; margin: 10px 0; border-collapse: collapse; }
    .meta-table td { padding: 4px 0; border-bottom: 1px solid #f9f9f9; }
    .meta-label { color: #888; width: 45%; }
    .meta-val { color: #333; font-weight: 500; text-align: right; }
    
    .use-case-text { font-size: 0.8rem; color: #555; margin: 10px 0; line-height: 1.4; height: 50px; overflow: hidden; }
    
    .metric-grid { display: flex; justify-content: space-between; background: #fafafa; padding: 8px; border-radius: 4px; border: 1px solid #eee; margin-bottom: 10px;}
    .metric-item { text-align: center; }
    .metric-label { font-size: 0.6rem; color: #999; display: block; }
    .metric-num { font-size: 0.85rem; font-weight: 700; color: var(--accent); }

    /* Button Styling */
    .stButton>button { background-color: #111; color: #fff; border-radius: 4px; font-size: 0.8rem; height: 35px; }
    .stButton>button:hover { background-color: var(--accent); color: #fff; }
    </style>
    """, unsafe_allow_html=True)

# --- PATHS & DATA ---
REG_PATH = "model_registry_v3.csv"
REQ_PATH = "requests_v3.csv"
LOG_PATH = "search_efficacy_v3.csv"

def get_df():
    if not os.path.exists(REG_PATH):
        return pd.DataFrame()
    return pd.read_csv(REG_PATH)

def save_df(df):
    df.to_csv(REG_PATH, index=False, quoting=csv.QUOTE_ALL)

# --- SEARCH ENGINE (REFINED) ---
def process_intelligent_search(query, df):
    if not query or df.empty: return df
    
    q = query.lower()
    # Clean NaNs before searching to avoid TF-IDF exceptions
    df = df.fillna('N/A')
    filtered_df = df.copy()

    # 1. Descriptive magnitude parsing
    logic_maps = {
        r"low latency": filtered_df['latency'] < 40,
        r"high latency": filtered_df['latency'] > 90,
        r"high accuracy": filtered_df['accuracy'] > 0.95,
        r"usage (very )?high": filtered_df['usage'] > 10000,
        r"low usage": filtered_df['usage'] < 1000,
    }
    for phrase, mask in logic_maps.items():
        if re.search(phrase, q): filtered_df = filtered_df[mask]

    # 2. Numerical Operator Parsing (e.g., latency < 50)
    patterns = {
        'latency': r'latency\s*([<>]=?)\s*(\d+)',
        'accuracy': r'accuracy\s*([<>]=?)\s*(\d+)',
        'usage': r'usage\s*([<>]=?)\s*(\d+)',
        'cpu_util': r'cpu\s*([<>]=?)\s*(\d+)',
    }
    for col, pattern in patterns.items():
        match = re.search(pattern, q)
        if match:
            op, val = match.group(1), float(match.group(2))
            if col == 'accuracy' and val > 1: val = val / 100
            if op == '<': filtered_df = filtered_df[filtered_df[col] < val]
            elif op == '>': filtered_df = filtered_df[filtered_df[col] > val]
            elif op == '<=': filtered_df = filtered_df[filtered_df[col] <= val]
            elif op == '>=': filtered_df = filtered_df[filtered_df[col] >= val]

    if filtered_df.empty: return filtered_df

    # 3. TF-IDF Semantic Search
    # We join all technical columns into one string per row
    cols_to_index = ['name', 'clients', 'use_cases', 'domain', 'model_owner_team', 'model_stage', 'training_data_source']
    filtered_df['search_text'] = filtered_df[cols_to_index].astype(str).apply(' '.join, axis=1)
    
    vec = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
    mtx = vec.fit_transform(filtered_df['search_text'].tolist() + [query])
    scores = cosine_similarity(mtx[-1], mtx[:-1])[0]
    filtered_df['relevance'] = scores
    
    return filtered_df[filtered_df['relevance'] > 0.001].sort_values('relevance', ascending=False)

# --- NAVIGATION ---
with st.sidebar:
    st.title("Marketplace")
    user = st.selectbox("Role", ["John Doe", "Jane Nu", "Sam King", "Nat Patel (Leader)", "Admin"])
    st.divider()
    st.caption("Search Tips:")
    st.info("Try: 'latency < 50'\nTry: 'shadow finance'\nTry: 'cpu < 40%'")

# Reload data every rerun
df_master = get_df()

# --- VIEWS ---
if user in ["John Doe", "Jane Nu", "Sam King"]:
    t1, t2, t3 = st.tabs(["🏛 Unified Gallery", "🚀 Ingest Asset", "📊 My Portfolio"])
    
    with t1:
        query = st.text_input("💬 Smart Chat Search", placeholder="e.g. 'Finance low latency and >90 accuracy'")
        results = process_intelligent_search(query, df_master)
        
        st.write(f"Matches: {len(results)}")
        for i in range(0, min(len(results), 30), 3):
            cols = st.columns(3)
            for j in range(3):
                if i+j < len(results):
                    row = results.iloc[i+j]
                    with cols[j]:
                        st.markdown(f"""
                        <div class="model-card">
                            <div>
                                <span class="status-pill {row['monitoring_status']}">{row['monitoring_status']}</span>
                                <div class="model-title">{row['name']}</div>
                                <div class="version-tag">{row['model_version']} | {row['type']}</div>
                                <table class="meta-table">
                                    <tr><td class="meta-label">Domain</td><td class="meta-val">{row['domain']}</td></tr>
                                    <tr><td class="meta-label">Stage</td><td class="meta-val">{row['model_stage']}</td></tr>
                                    <tr><td class="meta-label">Team</td><td class="meta-val">{row['model_owner_team']}</td></tr>
                                </table>
                                <div class="use-case-text"><b>Case:</b> {row['use_cases']}</div>
                            </div>
                            <div class="metric-grid">
                                <div class="metric-item"><span class="metric-label">Acc</span><span class="metric-num">{int(float(row['accuracy'])*100)}%</span></div>
                                <div class="metric-item"><span class="metric-label">Lat</span><span class="metric-num">{row['latency']}ms</span></div>
                                <div class="metric-item"><span class="metric-label">CPU</span><span class="metric-num">{row['cpu_util']}%</span></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        btn_col1, btn_col2 = st.columns(2)
                        with btn_col1:
                            if st.button("Access Request", key=f"req_{i+j}_{row['name']}"):
                                new_req = pd.DataFrame([{"model_name": row['name'], "requester": user, "status": "Pending", "timestamp": str(datetime.datetime.now())}])
                                new_req.to_csv(REQ_PATH, mode='a', header=not os.path.exists(REQ_PATH), index=False)
                                st.toast("Sent to Nat Patel")
                        
                        with btn_col2:
                            with st.popover("More Details"):
                                st.markdown(f"### {row['name']} Technical Detail")
                                c_a, c_b = st.columns(2)
                                c_a.write(f"**Version:** {row['model_version']}")
                                c_a.write(f"**SLA Tier:** {row['sla_tier']}")
                                c_a.write(f"**Endpoint ID:** `{row['inference_endpoint_id']}`")
                                c_b.write(f"**Data Source:** {row['training_data_source']}")
                                c_b.write(f"**Feature Store:** {row['feature_store_dependency']}")
                                c_b.write(f"**Last Retrained:** {row['last_retrained_date']}")
                                st.divider()
                                st.write("**Clients Using This:**")
                                st.info(row['clients'])
                                st.write("**Detailed Description:**")
                                st.write(row['use_cases'])

    with t2:
        with st.form("ingest_v4", clear_on_submit=True):
            st.subheader("Model Ingestion")
            n = st.text_input("Model Name*")
            d = st.text_input("Business Domain (e.g. Risk, Finance)")
            cl = st.text_input("Clients (Separate with ;)")
            desc = st.text_area("Full Description*")
            
            if st.form_submit_button("Publish Model"):
                if n and desc:
                    # Creating a complete row matching the master schema exactly
                    new_row = pd.DataFrame([{
                        "name": n, "model_version": "v1.0.0", "domain": d if d else "General",
                        "type": "Community", "accuracy": 0.85, "latency": 50, "clients": cl if cl else "N/A",
                        "use_cases": desc, "contributor": user, "usage": 0, "data_drift": 0.0,
                        "pred_drift": 0.0, "cpu_util": 0, "mem_util": 0, "throughput": 0, "error_rate": 0.0,
                        "model_owner_team": "Data Science Hub", "last_retrained_date": str(datetime.date.today()),
                        "model_stage": "Experimental", "training_data_source": "Contributor Input",
                        "approval_status": "Pending", "monitoring_status": "Healthy", "sla_tier": "Bronze",
                        "feature_store_dependency": "None", "inference_endpoint_id": f"ep-{random.randint(100,999)}"
                    }])
                    df_master = pd.concat([df_master, new_row], ignore_index=True)
                    save_df(df_master)
                    st.success(f"Success! Model '{n}' is now live.")
                else:
                    st.error("Name and Description are mandatory.")

    with t3:
        my_m = df_master[df_master['contributor'] == user]
        if not my_m.empty:
            sel = st.selectbox("Select Asset to Inspect", my_m['name'].unique())
            m_dat = my_m[my_m['name'] == sel].iloc[0]
            fig = go.Figure(go.Scatterpolar(
                r=[float(m_dat['accuracy'])*100, 100-(float(m_dat['data_drift'])*100), 100-float(m_dat['cpu_util']), 100-float(m_dat['error_rate'])*10],
                theta=['Accuracy','Stability','Efficiency','Reliability'], fill='toself'
            ))
            st.plotly_chart(fig, use_container_width=True)
        else: st.info("No contributions yet.")

elif user == "Nat Patel (Leader)":
    st.subheader("Leader Approval Queue")
    if os.path.exists(REQ_PATH):
        req_df = pd.read_csv(REQ_PATH)
        pend = req_df[req_df['status'] == "Pending"]
        if not pend.empty:
            for idx, r in pend.iterrows():
                c1, c2 = st.columns([3, 1])
                c1.write(f"💼 **{r['requester']}** requested **{r['model_name']}**")
                if c2.button("Approve", key=f"ap_{idx}"):
                    req_df.at[idx, 'status'] = "Approved"
                    req_df.to_csv(REQ_PATH, index=False)
                    st.rerun()
        else: st.success("Queue clear.")

else: # ADMIN VIEW
    k1, k2, k3 = st.columns(3)
    k1.metric("API Volume", f"{int(df_master['usage'].sum()):,}")
    k2.metric("Fleet Inventory", len(df_master))
    k3.metric("Uptime", "99.98%")

    st.divider()
    hl = st.selectbox("Highlight Line", ["None"] + list(df_master['name'].unique()))
    cv = [1.0 if name == hl else 0.0 for name in df_master['name']]
    cs = [[0, 'rgba(255, 249, 196, 0.4)'], [1, '#B71C1C']]
    
    fig = go.Figure(data=go.Parcoords(
        labelfont=dict(size=11, color='black'), tickfont=dict(size=9, color='#666'),
        line=dict(color=cv, colorscale=cs, showscale=False),
        dimensions=list([
            dict(range=[0.7, 1.0], label='Accuracy', values=df_master['accuracy']),
            dict(range=[0, 200], label='Latency', values=df_master['latency']),
            dict(range=[0, 25000], label='Usage', values=df_master['usage']),
            dict(range=[0, 100], label='CPU %', values=df_master['cpu_util']),
            dict(range=[0, 0.3], label='Drift', values=df_master['data_drift'])
        ])
    ))
    fig.update_layout(margin=dict(t=80, b=40, l=70, r=70), height=500)
    st.plotly_chart(fig, use_container_width=True)
