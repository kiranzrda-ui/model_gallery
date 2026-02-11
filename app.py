import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import datetime
import random

# --- CONFIG & STYLING ---
st.set_page_config(page_title="Accenture AI Marketplace", layout="wide")

st.markdown("""
    <style>
    :root { --accent: #A100FF; --light: #F3E5F5; }
    span[data-baseweb="tag"] { background-color: var(--light) !important; color: var(--accent) !important; }
    
    .model-card {
        border: 1px solid #e0e0e0; border-top: 4px solid var(--accent);
        padding: 12px; background-color: #ffffff; margin-bottom: 15px;
        min-height: 380px; display: flex; flex-direction: column; justify-content: space-between;
        border-radius: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .model-title { font-size: 1.05rem; font-weight: 700; color: #000; margin-bottom: 2px; }
    .client-tag { font-size: 0.7rem; color: var(--accent); font-weight: 600; margin-bottom: 8px; }
    .model-desc { font-size: 0.8rem; color: #444; line-height: 1.3; margin-bottom: 10px; 
                  display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
    
    .metric-box { background: #f8f9fa; padding: 8px; border-radius: 4px; display: flex; justify-content: space-between; font-size: 0.75rem; }
    .status-indicator { height: 8px; width: 8px; border-radius: 50%; display: inline-block; }
    
    .stButton>button { background-color: #000; color: #fff; border-radius: 0; font-size: 0.8rem; height: 35px; width: 100%; }
    .stButton>button:hover { background-color: var(--accent); }
    </style>
    """, unsafe_allow_html=True)

# --- PERSISTENCE ---
REG_PATH = "model_registry_v3.csv"
LOG_PATH = "search_efficacy_logs.csv"
REQ_PATH = "requests_v3.csv"

def init_files():
    if not os.path.exists(REG_PATH):
        # Generate varied data if file doesn't exist
        doms = ["Finance", "HR", "Procurement", "Supply Chain", "IT", "Legal", "Marketing"]
        cls = ["Apple", "NASA", "Amazon", "Coca-Cola", "BMW", "Samsung", "Walmart"]
        data = []
        for i in range(55):
            d = random.choice(doms)
            acc = random.uniform(0.70, 0.99)
            lat = random.randint(20, 100)
            data.append({
                "name": f"{d}-Engine-{i+100}", "domain": d, "type": "Official" if i < 20 else "Community",
                "accuracy": round(acc, 2), "latency": lat, "clients": f"{random.choice(cls)}, {random.choice(cls)}",
                "use_cases": f"Scalable {d} Automation", "description": f"Advanced {d} neural network optimized for high-volume enterprise workloads and low-latency inference.",
                "contributor": random.choice(["John Doe", "Jane Nu", "Sam King"]) if i >= 20 else "System",
                "usage": random.randint(100, 10000), "data_drift": round(random.uniform(0, 0.15), 3),
                "cpu_util": random.randint(20, 90), "mem_util": random.randint(4, 32), "throughput": random.randint(100, 2000), "error_rate": round(random.uniform(0, 5), 2)
            })
        pd.DataFrame(data).to_csv(REG_PATH, index=False)
    if not os.path.exists(LOG_PATH): pd.DataFrame(columns=["query", "results_found", "timestamp"]).to_csv(LOG_PATH, index=False)
    if not os.path.exists(REQ_PATH): pd.DataFrame(columns=["model_name", "requester", "status", "timestamp"]).to_csv(REQ_PATH, index=False)

init_files()
df = pd.read_csv(REG_PATH)
search_logs = pd.read_csv(LOG_PATH)
requests_log = pd.read_csv(REQ_PATH)

# --- AUTH ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/Accenture.svg/2560px-Accenture.svg.png", width=120)
    user = st.selectbox("Current User", ["John Doe", "Jane Nu", "Sam King", "Nat Patel (Leader)", "Admin"])
    st.divider()
    st.subheader("Performance Filters")
    acc_filter = st.multiselect("Accuracy Class", ["High (>98%)", "Medium (80-97%)", "Low (<80%)"], default=["High (>98%)", "Medium (80-97%)"])
    lat_filter = st.multiselect("Latency Class", ["Low (<40ms)", "Med (41-60ms)", "High (>60ms)"], default=["Low (<40ms)", "Med (41-60ms)", "High (>60ms)"])

def apply_filters(input_df):
    m = pd.Series([False] * len(input_df), index=input_df.index)
    if "High (>98%)" in acc_filter: m |= (input_df['accuracy'] >= 0.98)
    if "Medium (80-97%)" in acc_filter: m |= (input_df['accuracy'] >= 0.80) & (input_df['accuracy'] < 0.98)
    if "Low (<80%)" in acc_filter: m |= (input_df['accuracy'] < 0.80)
    input_df = input_df[m]
    
    m_lat = pd.Series([False] * len(input_df), index=input_df.index)
    if "Low (<40ms)" in lat_filter: m_lat |= (input_df['latency'] < 40)
    if "Med (41-60ms)" in lat_filter: m_lat |= (input_df['latency'] >= 41) & (input_df['latency'] <= 60)
    if "High (>60ms)" in lat_filter: m_lat |= (input_df['latency'] > 60)
    return input_df[m_lat]

# --- SEARCH LOGIC ---
def run_search(query, search_df):
    if not query: return search_df
    search_df['blob'] = search_df.astype(str).apply(' '.join, axis=1)
    vec = TfidfVectorizer(stop_words='english')
    mtx = vec.fit_transform(search_df['blob'].tolist() + [query])
    scores = cosine_similarity(mtx[-1], mtx[:-1])[0]
    search_df['score'] = scores
    results = search_df[search_df['score'] > 0.05].sort_values('score', ascending=False)
    
    # Log Efficacy
    new_log = pd.DataFrame([{"query": query, "results_found": len(results), "timestamp": str(datetime.datetime.now())}])
    pd.concat([search_logs, new_log]).to_csv(LOG_PATH, index=False)
    return results

# --- UI VIEWS ---
if user in ["John Doe", "Jane Nu", "Sam King"]:
    st.title(f"Marketplace Hub | {user}")
    t1, t2, t3 = st.tabs(["🏛 Unified Gallery", "🚀 Ingest Model", "👤 My Dashboard"])
    
    with t1:
        q = st.text_input("💬 Open Search: Ask for any model, client, or use-case...", placeholder="e.g. 'NASA high accuracy' or 'Supply chain FedEx'")
        display_df = run_search(q, apply_filters(df))
        
        for i in range(0, len(display_df), 3):
            cols = st.columns(3)
            for j in range(3):
                if i+j < len(display_df):
                    row = display_df.iloc[i+j]
                    drift_color = "#2E7D32" if row['data_drift'] < 0.05 else "#EF6C00"
                    with cols[j]:
                        st.markdown(f"""
                        <div class="model-card">
                            <div>
                                <div style="display:flex; justify-content:space-between; align-items:center;">
                                    <span style="font-size:0.65rem; color:#666;">{row['domain']}</span>
                                    <span class="status-indicator" style="background:{drift_color};"></span>
                                </div>
                                <div class="model-title">{row['name']}</div>
                                <div class="client-tag">Clients: {row['clients']}</div>
                                <div class="model-desc">{row['description']}</div>
                            </div>
                            <div>
                                <div class="metric-box">
                                    <span><b>ACC:</b> {int(row['accuracy']*100)}%</span>
                                    <span><b>LAT:</b> {row['latency']}ms</span>
                                    <span><b>USE:</b> {row['usage']}</span>
                                </div>
                                <div style="height:10px;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        if st.button("Request Access", key=f"r_{row['name']}"):
                            n_req = pd.DataFrame([{"model_name": row['name'], "requester": user, "status": "Pending", "timestamp": str(datetime.datetime.now())}])
                            pd.concat([requests_log, n_req]).to_csv(REQ_PATH, index=False)
                            st.toast("Request Sent!")

    with t2:
        with st.form("ingest"):
            st.subheader("Model Metadata Contribution")
            c1, c2 = st.columns(2)
            n_name = c1.text_input("Name")
            n_dom = c2.selectbox("Domain", ["Finance", "HR", "Supply Chain", "IT", "Legal"])
            n_clients = st.text_input("Clients (Comma separated)")
            n_desc = st.text_area("Full Description")
            if st.form_submit_button("Publish"):
                new_row = pd.DataFrame([{"name": n_name, "domain": n_dom, "type": "Community", "accuracy": 0.90, "latency": 40, "clients": n_clients, "description": n_desc, "contributor": user, "usage": 0, "data_drift": 0.01, "cpu_util": 20, "mem_util": 8, "throughput": 500, "error_rate": 0.1}])
                pd.concat([df, new_row]).to_csv(REG_PATH, index=False)
                st.success("Ingested!")

    with t3:
        my_m = df[df['contributor'] == user]
        if not my_m.empty:
            st.subheader("Your Impact & Telemetry")
            sel = st.selectbox("Inspect Model", my_m['name'])
            m_dat = my_m[my_m['name'] == sel].iloc[0]
            col_a, col_b = st.columns(2)
            with col_a:
                fig_radar = go.Figure(go.Scatterpolar(r=[m_dat['accuracy']*100, 100-m_dat['data_drift']*100, 100-m_dat['cpu_util'], 100-m_dat['error_rate']*10], theta=['Accuracy', 'Stability', 'Efficiency', 'Reliability'], fill='toself', line_color='#A100FF'))
                st.plotly_chart(fig_radar, use_container_width=True)
            with col_b:
                st.metric("Total Usage", m_dat['usage'])
                st.metric("Inference Throughput", f"{m_dat['throughput']} req/s")
        else:
            st.info("No contributions yet.")

elif user == "Nat Patel (Leader)":
    st.title("Approval Gateway")
    pend = requests_log[requests_log['status'] == "Pending"]
    st.dataframe(pend, use_container_width=True)
    for idx, r in pend.iterrows():
        if st.button(f"Approve {r['requester']} for {r['model_name']}"):
            requests_log.at[idx, 'status'] = "Approved"
            requests_log.to_csv(REQ_PATH, index=False)
            st.rerun()

else: # ADMIN
    st.title("Marketplace Governance Dashboard")
    
    # METRICS ROW
    k1, k2, k3 = st.columns(3)
    k1.metric("Total Model Usage", df['usage'].sum())
    # Search Efficacy: % of searches that returned results
    if len(search_logs) > 0:
        eff = (len(search_logs[search_logs['results_found'] > 0]) / len(search_logs)) * 100
        k2.metric("Search Efficacy", f"{int(eff)}%")
    k3.metric("Pending Approvals", len(requests_log[requests_log['status'] == "Pending"]))

    # WOW CHART: Interactive Parallel Coordinates
    st.subheader("Interactive Portfolio Inspector")
    st.write("Highlight a line to see details. High Accuracy models are Purple.")
    
    # Create a numeric ID for the Parallel Coordinates mapping
    df_plot = df.copy()
    df_plot['id'] = range(len(df_plot))
    
    fig_para = px.parallel_coordinates(df_plot, 
                                      color="accuracy",
                                      dimensions=['accuracy', 'latency', 'usage', 'data_drift', 'cpu_util'],
                                      color_continuous_scale=px.colors.sequential.Purples)
    st.plotly_chart(fig_para, use_container_width=True)
    
    # Table to allow "Highlighting" specific models from the plot
    st.subheader("Model Drill-down")
    st.dataframe(df[['name', 'domain', 'usage', 'accuracy', 'latency', 'data_drift', 'contributor']])
