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
st.set_page_config(page_title="Enterprise Model Gallery", layout="wide")

st.markdown("""
    <style>
    :root { --accent: #6200EE; --success: #2E7D32; --warning: #F9A825; --critical: #C62828; }
    
    /* Model Card CSS */
    .model-card {
        border: 1px solid #e0e0e0; border-top: 4px solid var(--accent);
        padding: 15px; background-color: #ffffff; margin-bottom: 20px;
        min-height: 420px; display: flex; flex-direction: column; justify-content: space-between;
        border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .model-title { font-size: 1.1rem; font-weight: 700; color: #1a1a1a; margin-bottom: 0px; }
    .version-tag { font-size: 0.75rem; color: #666; font-family: monospace; }
    
    .status-pill { font-size: 0.65rem; padding: 3px 10px; border-radius: 12px; color: white; font-weight: bold; float: right; }
    .status-Healthy { background-color: var(--success); }
    .status-Warning { background-color: var(--warning); }
    .status-Critical { background-color: var(--critical); }

    .meta-table { font-size: 0.75rem; width: 100%; border-collapse: collapse; margin: 12px 0; }
    .meta-table td { padding: 4px 0; border-bottom: 1px solid #f0f0f0; }
    .meta-label { color: #777; width: 45%; font-weight: 500; }
    .meta-value { color: #333; font-weight: 400; text-align: right; }

    .use-case-box { font-size: 0.8rem; color: #444; margin-bottom: 15px; line-height: 1.4; border-left: 3px solid #eee; padding-left: 10px; }
    
    .metric-container { display: flex; justify-content: space-between; background: #fdfdfd; padding: 8px; border-radius: 4px; font-size: 0.7rem; border: 1px solid #eee; }
    .metric-item { text-align: center; flex: 1; }
    .metric-val { font-weight: 700; color: var(--accent); display: block; font-size: 0.85rem; }

    /* Buttons */
    .stButton>button { background-color: #222; color: #fff; border-radius: 4px; font-size: 0.8rem; height: 35px; width: 100%; transition: 0.3s; }
    .stButton>button:hover { background-color: var(--accent); border-color: var(--accent); }
    </style>
    """, unsafe_allow_html=True)

# --- SCHEMA & PATHS ---
REG_PATH = "model_registry_v3.csv"
REQ_PATH = "requests_v3.csv"
LOG_PATH = "search_efficacy_v3.csv"

# --- HELPER FUNCTIONS ---
def init_files():
    if not os.path.exists(REG_PATH):
        data = []
        for i in range(100):
            d = random.choice(["Finance", "IT Ops", "HR", "Healthcare", "Risk"])
            data.append({
                "name": f"{d}-Model-{i+100}", "model_version": f"v{random.randint(1,3)}.{random.randint(0,9)}",
                "domain": d, "type": random.choice(["Official", "Community"]),
                "accuracy": round(random.uniform(0.70, 0.99), 3), "latency": random.randint(15, 150),
                "clients": "Fortune 500", "use_cases": f"Capacity Forecasting for {d}",
                "contributor": random.choice(["John Doe", "Jane Nu", "Sam King"]), "usage": random.randint(100, 15000),
                "data_drift": round(random.uniform(0, 0.1), 3), "pred_drift": 0.05,
                "cpu_util": random.randint(10, 80), "mem_util": random.randint(4, 32),
                "throughput": random.randint(100, 2000), "error_rate": round(random.uniform(0, 3), 2),
                "model_owner_team": random.choice(["Risk Analytics", "Strategy AI", "Growth Team"]), 
                "last_retrained_date": (datetime.datetime.now() - datetime.timedelta(days=random.randint(0,90))).strftime("%Y-%m-%d"),
                "model_stage": random.choice(["Prod", "Shadow", "Canary"]), "training_data_source": "Regulatory Filings",
                "approval_status": "Approved", "monitoring_status": random.choice(["Healthy", "Warning", "Critical"]),
                "sla_tier": random.choice(["Gold", "Silver", "Bronze"]), "feature_store_dependency": "sagemaker_fs",
                "inference_endpoint_id": f"ep-{random.randint(1000, 9999)}"
            })
        pd.DataFrame(data).to_csv(REG_PATH, index=False, quoting=csv.QUOTE_ALL)
    if not os.path.exists(REQ_PATH): pd.DataFrame(columns=["model_name", "requester", "status", "timestamp"]).to_csv(REQ_PATH, index=False)
    if not os.path.exists(LOG_PATH): pd.DataFrame(columns=["query", "found", "timestamp"]).to_csv(LOG_PATH, index=False)

def extract_metadata_from_text(desc):
    """Infers metadata from contributor description using regex."""
    desc = desc.lower()
    stage = "Prod"
    if "shadow" in desc: stage = "Shadow"
    elif "canary" in desc: stage = "Canary"
    elif "experimental" in desc: stage = "Experimental"
    
    sla = "Bronze"
    if "gold" in desc or "tier 1" in desc: sla = "Gold"
    elif "silver" in desc or "tier 2" in desc: sla = "Silver"
    
    team = "General AI"
    if "risk" in desc: team = "Risk Analytics"
    elif "it" in desc or "ops" in desc: team = "IT Ops Team"
    
    return stage, sla, team

def parse_metrics_from_query(query, df):
    """Filters dataframe based on numerical constraints in search query (e.g. '>90 accuracy')."""
    query = query.lower()
    filtered_df = df.copy()
    
    # Accuracy patterns: >90, 95%, 0.95
    acc_match = re.search(r'([><=]=?)\s*(\d+)', query)
    if acc_match:
        op, val = acc_match.groups()
        val = float(val) / 100 if float(val) > 1 else float(val)
        if op == '>': filtered_df = filtered_df[filtered_df['accuracy'] > val]
        elif op == '<': filtered_df = filtered_df[filtered_df['accuracy'] < val]
        elif op == '>=': filtered_df = filtered_df[filtered_df['accuracy'] >= val]
    
    return filtered_df

def advanced_search(query, df):
    if not query: return df
    
    # 1. First apply numerical/metric filters inferred from query
    search_df = parse_metrics_from_query(query, df)
    if search_df.empty: return search_df
    
    # 2. Textual search ranking
    search_df['blob'] = search_df.fillna('').astype(str).apply(' '.join, axis=1)
    vec = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
    mtx = vec.fit_transform(search_df['blob'].tolist() + [query])
    scores = cosine_similarity(mtx[-1], mtx[:-1])[0]
    search_df['relevance'] = scores
    
    results = search_df[search_df['relevance'] > 0.01].sort_values('relevance', ascending=False)
    
    # Log Efficacy
    new_log = pd.DataFrame([{"query": query, "found": len(results), "timestamp": str(datetime.datetime.now())}])
    pd.concat([pd.read_csv(LOG_PATH), new_log]).to_csv(LOG_PATH, index=False)
    
    return results

# --- DATA INITIALIZATION ---
init_files()
df_master = pd.read_csv(REG_PATH)
req_log = pd.read_csv(REQ_PATH)
search_logs = pd.read_csv(LOG_PATH)

# --- SIDEBAR ---
with st.sidebar:
    st.title("Navigation")
    current_user = st.selectbox("Current User", ["John Doe", "Jane Nu", "Sam King", "Nat Patel (Leader)", "Admin"])
    st.info("Tip: You can search for metrics directly in the search bar. E.g., 'Finance >90 accuracy'")

# --- VIEW: DATA SCIENTIST ---
if current_user in ["John Doe", "Jane Nu", "Sam King"]:
    st.header(f"Data Scientist Workspace: {current_user}")
    t1, t2, t3 = st.tabs(["🏛 Unified Gallery", "🚀 Ingest Asset", "📊 My Portfolio"])

    with t1:
        q = st.text_input("💬 Search Models (Keywords, Stage, Team, or Metrics like '>90')", placeholder="e.g. 'Shadow Risk >85'")
        display_df = advanced_search(q, df_master)
        
        # Rendering results as Tiles
        for i in range(0, min(len(display_df), 30), 3):
            cols = st.columns(3)
            for j in range(3):
                if i+j < len(display_df):
                    row = display_df.iloc[i+j]
                    with cols[j]:
                        # Using unsafe_allow_html=True to fix the formatting of the card
                        st.markdown(f"""
                        <div class="model-card">
                            <div>
                                <span class="status-pill status-{row['monitoring_status']}">{row['monitoring_status']}</span>
                                <div class="model-title">{row['name']}</div>
                                <div class="version-tag">{row['model_version']} | {row['type']}</div>
                                
                                <table class="meta-table">
                                    <tr><td class="meta-label">Domain:</td><td class="meta-value">{row['domain']}</td></tr>
                                    <tr><td class="meta-label">Team:</td><td class="meta-value">{row['model_owner_team']}</td></tr>
                                    <tr><td class="meta-label">Stage:</td><td class="meta-value">{row['model_stage']}</td></tr>
                                    <tr><td class="meta-label">SLA Tier:</td><td class="meta-value">{row['sla_tier']}</td></tr>
                                    <tr><td class="meta-label">Source:</td><td class="meta-value">{row['training_data_source']}</td></tr>
                                    <tr><td class="meta-label">Dependency:</td><td class="meta-value">{row['feature_store_dependency']}</td></tr>
                                </table>
                                
                                <div class="use-case-box">
                                    <b>Use Case:</b><br>{row['use_cases']}
                                </div>
                            </div>
                            
                            <div>
                                <div class="metric-container">
                                    <div class="metric-item"><span class="metric-val">{int(row['accuracy']*100)}%</span>Accuracy</div>
                                    <div class="metric-item"><span class="metric-val">{row['latency']}ms</span>Latency</div>
                                    <div class="metric-item"><span class="metric-val">{row['data_drift']}</span>Drift</div>
                                </div>
                                <div style="height:10px;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        if st.button("Request Access", key=f"req_{i+j}_{row['name']}"):
                            n_req = pd.DataFrame([{"model_name": row['name'], "requester": current_user, "status": "Pending", "timestamp": str(datetime.datetime.now())}])
                            pd.concat([req_log, n_req]).to_csv(REQ_PATH, index=False)
                            st.toast("Request Sent!")

    with t2:
        st.subheader("Contribute New Model Asset")
        with st.form("ingest_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            i_name = col1.text_input("Model Name*")
            i_ver = col2.text_input("Version", "v1.0")
            i_desc = st.text_area("Description & Metadata*", help="Include stage (Shadow/Prod), team name, and data source here.")
            
            if st.form_submit_button("Ingest Asset"):
                if i_name and i_desc:
                    stage, sla, team = extract_metadata_from_text(i_desc)
                    new_row = pd.DataFrame([{
                        "name": i_name, "model_version": i_ver, "domain": "Uncategorized", "type": "Community",
                        "accuracy": 0.85, "latency": 50, "clients": "Internal", "use_cases": i_desc,
                        "contributor": current_user, "usage": 0, "data_drift": 0.0, "pred_drift": 0.0,
                        "cpu_util": 0, "mem_util": 0, "throughput": 0, "error_rate": 0,
                        "model_owner_team": team, "last_retrained_date": datetime.date.today().strftime("%Y-%m-%d"),
                        "model_stage": stage, "training_data_source": "Contributor Input", "approval_status": "Pending",
                        "monitoring_status": "Healthy", "sla_tier": sla, "feature_store_dependency": "TBD",
                        "inference_endpoint_id": "TBD"
                    }])
                    pd.concat([df_master, new_row]).to_csv(REG_PATH, index=False, quoting=csv.QUOTE_ALL)
                    st.success(f"Asset Ingested with inferred stage: {stage}")
                else:
                    st.error("Missing required fields.")

    with t3:
        st.subheader("Your Impact & Submissions")
        my_m = df_master[df_master['contributor'] == current_user]
        if not my_m.empty:
            st.dataframe(my_m[["name", "model_version", "model_stage", "usage", "accuracy"]])
            sel = st.selectbox("Inspect Asset Radar", my_m['name'].unique())
            m_dat = my_m[my_m['name'] == sel].iloc[0]
            fig = go.Figure(go.Scatterpolar(r=[m_dat['accuracy']*100, 100-m_dat['data_drift']*100, 100-m_dat['cpu_util'], 100-m_dat['error_rate']*10], theta=['Accuracy','Stability','Efficiency','Reliability'], fill='toself'))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No assets contributed yet.")

elif current_user == "Nat Patel (Leader)":
    st.header("Model Approval Gateway")
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

else: # ADMIN VIEW
    st.header("Admin Governance & Efficacy")
    k1, k2, k3 = st.columns(3)
    k1.metric("Total Model Consumption", df_master['usage'].sum())
    if len(search_logs) > 0:
        hit_rate = (len(search_logs[search_logs['found'] > 0]) / len(search_logs)) * 100
        k2.metric("Search Conversion (Efficacy)", f"{int(hit_rate)}%")
    k3.metric("Global Asset Inventory", len(df_master))

    st.divider()
    st.subheader("Interactive Portfolio Inspector")
    
    # Selection logic for clean lines
    hl = st.selectbox("Solo Inspector (Select one to clean plot)", ["None"] + list(df_master['name'].unique()))
    
    # Chart styling fixes (smudge prevention)
    plot_df = df_master.copy()
    cv = [1.0 if name == hl else 0.0 for name in plot_df['name']]
    cs = [[0, 'rgba(161, 0, 255, 0.1)'], [1, '#B71C1C']] # Semi-transparent purple vs Dark Red
    
    fig = go.Figure(data=go.Parcoords(
        labelfont=dict(size=10, color='black'), # Smaller font to prevent smudging
        tickfont=dict(size=8, color='gray'),
        line=dict(color=cv, colorscale=cs, showscale=False),
        dimensions=list([
            dict(range=[0.7, 1.0], label='Accuracy', values=plot_df['accuracy']),
            dict(range=[0, 200], label='Latency', values=plot_df['latency']),
            dict(range=[0, 15000], label='Usage', values=plot_df['usage']),
            dict(range=[0, 0.2], label='Drift', values=plot_df['data_drift']),
            dict(range=[0, 100], label='CPU %', values=plot_df['cpu_util']),
            dict(range=[0, 10], label='Error %', values=plot_df['error_rate'])
        ])
    ))
    # Add padding and adjust margins to stop smudging
    fig.update_layout(
        margin=dict(t=80, b=40, l=60, r=60),
        paper_bgcolor='white', 
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
