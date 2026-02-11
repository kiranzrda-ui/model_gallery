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

# --- CONFIG & STYLING ---
st.set_page_config(page_title="Accenture AI Marketplace | Ultimate Registry", layout="wide")

st.markdown("""
    <style>
    :root { --accent: #A100FF; --success: #2E7D32; --warning: #F9A825; --critical: #C62828; }
    
    .model-card {
        border: 1px solid #e0e0e0; border-top: 4px solid var(--accent);
        padding: 12px; background-color: #ffffff; margin-bottom: 15px;
        min-height: 480px; display: flex; flex-direction: column; justify-content: space-between;
        border-radius: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .model-title { font-size: 1.1rem; font-weight: 700; color: #000; margin-bottom: 0px; }
    .version-tag { font-size: 0.7rem; color: #666; font-family: monospace; }
    .client-tag { font-size: 0.75rem; color: var(--accent); font-weight: 600; margin: 5px 0; }
    
    .meta-table { font-size: 0.7rem; width: 100%; border-collapse: collapse; margin-bottom: 8px;}
    .meta-table td { padding: 2px 0; border-bottom: 1px solid #f0f0f0; color: #444; }
    .meta-label { color: #888; width: 50%; }
    
    .status-pill { font-size: 0.65rem; padding: 2px 8px; border-radius: 10px; color: white; font-weight: bold; float: right; }
    .status-healthy { background-color: var(--success); }
    .status-warning { background-color: var(--warning); }
    .status-critical { background-color: var(--critical); }

    .metric-box { background: #f8f9fa; padding: 6px; border-radius: 4px; display: flex; justify-content: space-between; font-size: 0.7rem; border: 1px solid #eee; }
    .stButton>button { background-color: #000; color: #fff; border-radius: 0; font-size: 0.8rem; height: 32px; width: 100%; }
    .stButton>button:hover { background-color: var(--accent); }
    </style>
    """, unsafe_allow_html=True)

# --- SCHEMA DEFINITION ---
CSV_COLUMNS = [
    "name", "model_version", "domain", "type", "accuracy", "latency", 
    "clients", "use_cases", "contributor", "usage", "data_drift", "pred_drift", 
    "cpu_util", "mem_util", "throughput", "error_rate", "model_owner_team", 
    "last_retrained_date", "model_stage", "training_data_source", "approval_status", 
    "monitoring_status", "sla_tier", "feature_store_dependency", "inference_endpoint_id"
]

REG_PATH = "model_registry_v3.csv"
REQ_PATH = "requests_v3.csv"
LOG_PATH = "search_efficacy_v3.csv"

def init_files():
    if not os.path.exists(REG_PATH):
        # Generate initial data with full schema
        data = []
        teams = ["Core AI", "Risk Science", "Growth Analytics", "Ops Intelligence"]
        stages = ["Prod", "Shadow", "Canary", "Retired"]
        sources = ["Snowflake", "AWS S3", "Azure Data Lake", "Google BigQuery"]
        mon_status = ["Healthy", "Warning", "Critical"]
        sla = ["Tier 1 (Mission Critical)", "Tier 2 (High)", "Tier 3 (Standard)"]
        
        for i in range(100):
            d = random.choice(["Finance", "HR", "Supply Chain", "IT Ops", "Legal"])
            data.append({
                "name": f"{d}-Asset-{i+100}", "model_version": f"v{random.randint(1,5)}.{random.randint(0,9)}",
                "domain": d, "type": random.choice(["Official", "Community"]),
                "accuracy": round(random.uniform(0.75, 0.99), 3), "latency": random.randint(15, 120),
                "clients": "Fortune 500 Clients", "use_cases": f"Optimization of {d} workflows",
                "contributor": random.choice(["John Doe", "Jane Nu", "Sam King"]), "usage": random.randint(100, 15000),
                "data_drift": round(random.uniform(0, 0.2), 3), "pred_drift": 0.05,
                "cpu_util": random.randint(10, 90), "mem_util": random.randint(4, 32),
                "throughput": random.randint(100, 2000), "error_rate": round(random.uniform(0, 4), 2),
                "model_owner_team": random.choice(teams), 
                "last_retrained_date": (datetime.datetime.now() - datetime.timedelta(days=random.randint(0,60))).strftime("%Y-%m-%d"),
                "model_stage": random.choice(stages), "training_data_source": random.choice(sources),
                "approval_status": "Approved", "monitoring_status": random.choice(mon_status),
                "sla_tier": random.choice(sla), "feature_store_dependency": random.choice(["Feast-FS-01", "None"]),
                "inference_endpoint_id": f"ep-{random.randint(1000, 9999)}"
            })
        pd.DataFrame(data).to_csv(REG_PATH, index=False, quoting=csv.QUOTE_ALL)
    
    if not os.path.exists(REQ_PATH): pd.DataFrame(columns=["model_name", "requester", "status", "timestamp"]).to_csv(REQ_PATH, index=False)
    if not os.path.exists(LOG_PATH): pd.DataFrame(columns=["query", "found", "timestamp"]).to_csv(LOG_PATH, index=False)

init_files()
df_master = pd.read_csv(REG_PATH)
req_log = pd.read_csv(REQ_PATH)
search_logs = pd.read_csv(LOG_PATH)

# --- SEARCH LOGIC ---
def advanced_search(query, df):
    if not query: return df
    search_df = df.copy()
    # Create search blob from all searchable meta-data
    search_df['blob'] = search_df.fillna('').astype(str).apply(' '.join, axis=1)
    vec = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
    mtx = vec.fit_transform(search_df['blob'].tolist() + [query])
    scores = cosine_similarity(mtx[-1], mtx[:-1])[0]
    search_df['score'] = scores
    results = search_df[search_df['score'] > 0.02].sort_values('score', ascending=False)
    
    new_log = pd.DataFrame([{"query": query, "found": len(results), "timestamp": str(datetime.datetime.now())}])
    pd.concat([search_logs, new_log]).to_csv(LOG_PATH, index=False)
    return results

# --- AUTH & NAVIGATION ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/Accenture.svg/2560px-Accenture.svg.png", width=120)
    current_user = st.selectbox("Switch Identity", ["John Doe", "Jane Nu", "Sam King", "Nat Patel (Leader)", "Admin"])
    st.divider()
    st.caption("Filters")
    acc_sel = st.multiselect("Accuracy Class", ["High (>98%)", "Medium (80-97%)", "Low (<80%)"], default=["High (>98%)", "Medium (80-97%)"])

# --- VIEW: DATA SCIENTIST ---
if current_user in ["John Doe", "Jane Nu", "Sam King"]:
    st.title(f"Marketplace Hub | {current_user}")
    t1, t2, t3 = st.tabs(["🏛 Unified Gallery", "🚀 Ingest Asset", "📊 My Portfolio"])

    with t1:
        q = st.text_input("💬 Smart Search: Try 'v2.1', 'Tier 1', 'Snowflake', or 'Healthy'", placeholder="e.g. 'Finance Prod Tier 1'")
        display_df = advanced_search(q, df_master)
        
        for i in range(0, min(len(display_df), 30), 3):
            cols = st.columns(3)
            for j in range(3):
                if i+j < len(display_df):
                    row = display_df.iloc[i+j]
                    mon_class = "status-healthy" if row['monitoring_status']=="Healthy" else "status-warning" if row['monitoring_status']=="Warning" else "status-critical"
                    with cols[j]:
                        st.markdown(f"""
                        <div class="model-card">
                            <div>
                                <span class="status-pill {mon_class}">{row['monitoring_status']}</span>
                                <div class="model-title">{row['name']} <span class="version-tag">{row['model_version']}</span></div>
                                <div class="client-tag">Team: {row['model_owner_team']}</div>
                                
                                <table class="meta-table">
                                    <tr><td class="meta-label">Domain:</td><td>{row['domain']}</td></tr>
                                    <tr><td class="meta-label">Stage:</td><td>{row['model_stage']}</td></tr>
                                    <tr><td class="meta-label">SLA Tier:</td><td>{row['sla_tier']}</td></tr>
                                    <tr><td class="meta-label">Data Source:</td><td>{row['training_data_source']}</td></tr>
                                    <tr><td class="meta-label">Dependency:</td><td>{row['feature_store_dependency']}</td></tr>
                                </table>
                                
                                <div style="font-size:0.75rem; color:#444; margin-bottom:10px;"><b>Use Case:</b> {row['use_cases']}</div>
                            </div>
                            <div>
                                <div class="metric-box">
                                    <span><b>ACC:</b> {int(row['accuracy']*100)}%</span>
                                    <span><b>LAT:</b> {row['latency']}ms</span>
                                    <span><b>DRIFT:</b> {row['data_drift']}</span>
                                </div>
                                <div style="height:8px;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        if st.button("Request Access", key=f"r_{i+j}_{row['name']}"):
                            n_req = pd.DataFrame([{"model_name": row['name'], "requester": current_user, "status": "Pending", "timestamp": str(datetime.datetime.now())}])
                            pd.concat([req_log, n_req]).to_csv(REQ_PATH, index=False)
                            st.toast("Request Sent!")

    with t2:
        with st.form("ingest", clear_on_submit=True):
            st.subheader("Model Meta-Data Ingestion")
            c1, c2, c3 = st.columns(3)
            i_name = c1.text_input("Model Name")
            i_ver = c2.text_input("Version", "v1.0")
            i_dom = c3.selectbox("Domain", ["Finance", "HR", "Supply Chain", "IT Ops", "Legal"])
            
            i_desc = st.text_area("Use Case & Description (Free text for indexing)")
            
            c4, c5 = st.columns(2)
            i_team = c4.text_input("Owner Team")
            i_source = c5.text_input("Data Source (Snowflake/S3)")
            
            c6, c7 = st.columns(2)
            i_stage = c6.selectbox("Stage", ["Prod", "Shadow", "Canary", "Experimental"])
            i_sla = c7.selectbox("SLA Tier", ["Tier 1", "Tier 2", "Tier 3"])

            if st.form_submit_button("Publish to Registry"):
                new_row = pd.DataFrame([{
                    "name": i_name, "model_version": i_ver, "domain": i_dom, "type": "Community",
                    "accuracy": 0.85, "latency": 50, "clients": "Internal", "use_cases": i_desc,
                    "contributor": current_user, "usage": 0, "data_drift": 0.0, "pred_drift": 0.0,
                    "cpu_util": 0, "mem_util": 0, "throughput": 0, "error_rate": 0,
                    "model_owner_team": i_team, "last_retrained_date": datetime.date.today().strftime("%Y-%m-%d"),
                    "model_stage": i_stage, "training_data_source": i_source, "approval_status": "Pending",
                    "monitoring_status": "Healthy", "sla_tier": i_sla, "feature_store_dependency": "None",
                    "inference_endpoint_id": "TBD"
                }])
                pd.concat([df_master, new_row]).to_csv(REG_PATH, index=False, quoting=csv.QUOTE_ALL)
                st.success("Asset Published Successfully!")

    with t3:
        my_m = df_master[df_master['contributor'] == current_user]
        if not my_m.empty:
            st.subheader(f"Portfolio of {current_user}")
            st.dataframe(my_m[["name", "model_version", "model_stage", "monitoring_status", "usage"]])
            sel = st.selectbox("Deep Dive Radar", my_m['name'].unique())
            m_dat = my_m[my_m['name'] == sel].iloc[0]
            fig = go.Figure(go.Scatterpolar(r=[m_dat['accuracy']*100, 100-m_dat['data_drift']*100, 100-m_dat['cpu_util'], 100-m_dat['error_rate']*10], theta=['Accuracy','Stability','Efficiency','Reliability'], fill='toself'))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No assets found.")

elif current_user == "Nat Patel (Leader)":
    st.title("Approval Gateway")
    pend = req_log[req_log['status'] == "Pending"]
    if not pend.empty:
        for idx, r in pend.iterrows():
            st.write(f"💼 **{r['requester']}** requested access to **{r['model_name']}**")
            if st.button("Approve", key=f"ap_{idx}"):
                req_log.at[idx, 'status'] = "Approved"
                req_log.to_csv(REQ_PATH, index=False)
                st.rerun()
    else:
        st.success("Approval queue clear.")

else: # ADMIN
    st.title("Admin Governance")
    k1, k2, k3 = st.columns(3)
    k1.metric("Usage", df_master['usage'].sum())
    if len(search_logs) > 0:
        eff = (len(search_logs[search_logs['found'] > 0]) / len(search_logs)) * 100
        k2.metric("Search Efficacy", f"{int(eff)}%")
    k3.metric("Inventory", len(df_master))

    st.divider()
    hl = st.selectbox("Solo Inspector", ["None"] + list(df_master['name'].unique()))
    cv = [1.0 if name == hl else 0.0 for name in df_master['name']]
    cs = [[0, 'rgba(255, 249, 196, 0.3)'], [1, '#B71C1C']]
    fig = go.Figure(data=go.Parcoords(
        labelfont=dict(size=12, color='black'),
        line=dict(color=cv, colorscale=cs, showscale=False),
        dimensions=list([
            dict(range=[0.7, 1.0], label='Accuracy', values=df_master['accuracy']),
            dict(range=[0, 150], label='Latency', values=df_master['latency']),
            dict(range=[0, 20000], label='Usage', values=df_master['usage']),
            dict(range=[0, 0.3], label='Drift', values=df_master['data_drift'])
        ])
    ))
    fig.update_layout(margin=dict(t=100, b=50, l=80, r=80), paper_bgcolor='white', height=500)
    st.plotly_chart(fig, use_container_width=True)
