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
    :root { --accent: #A100FF; --highlight: #FFD700; --light: #F3E5F5; }
    span[data-baseweb="tag"] { background-color: var(--light) !important; color: var(--accent) !important; }
    
    .model-card {
        border: 1px solid #e0e0e0; border-top: 4px solid var(--accent);
        padding: 12px; background-color: #ffffff; margin-bottom: 15px;
        min-height: 350px; display: flex; flex-direction: column; justify-content: space-between;
        border-radius: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .model-title { font-size: 1rem; font-weight: 700; color: #000; margin-bottom: 2px; }
    .client-tag { font-size: 0.75rem; color: var(--accent); font-weight: 600; margin-bottom: 5px; }
    .model-desc { font-size: 0.8rem; color: #444; line-height: 1.3; margin-bottom: 8px; 
                  display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
    
    .metric-box { background: #f8f9fa; padding: 6px; border-radius: 4px; display: flex; justify-content: space-between; font-size: 0.7rem; }
    
    .stButton>button { background-color: #000; color: #fff; border-radius: 0; font-size: 0.8rem; height: 32px; width: 100%; }
    .stButton>button:hover { background-color: var(--accent); }
    </style>
    """, unsafe_allow_html=True)

# --- PERSISTENCE ---
REG_PATH = "model_registry_v4.csv"
LOG_PATH = "search_logs_v4.csv"
REQ_PATH = "requests_v4.csv"

def init_files():
    if not os.path.exists(REG_PATH):
        doms = ["Finance", "HR", "Procurement", "Supply Chain", "IT", "Legal", "Marketing"]
        cls = ["Apple, NASA", "Amazon, BMW", "Coca-Cola, Shell", "JP Morgan, HSBC", "Samsung, Google", "Walmart, FedEx", "Meta, Microsoft"]
        data = []
        for i in range(60):
            d = random.choice(doms)
            acc = random.uniform(0.75, 0.99)
            lat = random.randint(15, 120)
            data.append({
                "name": f"{d}-Asset-{i+100}", "domain": d, "type": "Official" if i < 20 else "Community",
                "accuracy": round(acc, 3), "latency": lat, "clients": random.choice(cls),
                "use_cases": f"Scalable {d} processing", "description": f"Proprietary {d} architecture designed for high-concurrency enterprise workloads. Optimized for P99 latency and cross-domain data drift resilience.",
                "contributor": random.choice(["John Doe", "Jane Nu", "Sam King"]) if i >= 20 else "System",
                "usage": random.randint(100, 15000), "data_drift": round(random.uniform(0, 0.20), 3),
                "cpu_util": random.randint(10, 95), "mem_util": random.randint(4, 64), "throughput": random.randint(50, 2500), "error_rate": round(random.uniform(0, 4), 2)
            })
        pd.DataFrame(data).to_csv(REG_PATH, index=False)
    if not os.path.exists(LOG_PATH): pd.DataFrame(columns=["query", "found", "timestamp"]).to_csv(LOG_PATH, index=False)
    if not os.path.exists(REQ_PATH): pd.DataFrame(columns=["model_name", "requester", "status", "timestamp"]).to_csv(REQ_PATH, index=False)

init_files()
df = pd.read_csv(REG_PATH)
search_logs = pd.read_csv(LOG_PATH)
req_log = pd.read_csv(REQ_PATH)

# --- SIDEBAR & AUTH ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/Accenture.svg/2560px-Accenture.svg.png", width=120)
    user_role = st.selectbox("Current User", ["John Doe", "Jane Nu", "Sam King", "Nat Patel (Leader)", "Admin"])
    st.divider()
    st.subheader("Decision Filters")
    acc_sel = st.multiselect("Accuracy", ["High (>98%)", "Medium (80-97%)", "Low (<80%)"], default=["High (>98%)", "Medium (80-97%)"])
    lat_sel = st.multiselect("Latency", ["Low (<40ms)", "Med (41-60ms)", "High (>60ms)"], default=["Low (<40ms)", "Med (41-60ms)", "High (>60ms)"])

def filter_registry(input_df):
    m_acc = pd.Series([False] * len(input_df), index=input_df.index)
    if "High (>98%)" in acc_sel: m_acc |= (input_df['accuracy'] >= 0.98)
    if "Medium (80-97%)" in acc_sel: m_acc |= (input_df['accuracy'] >= 0.80) & (input_df['accuracy'] < 0.98)
    if "Low (<80%)" in acc_sel: m_acc |= (input_df['accuracy'] < 0.80)
    input_df = input_df[m_acc]
    
    m_lat = pd.Series([False] * len(input_df), index=input_df.index)
    if "Low (<40ms)" in lat_sel: m_lat |= (input_df['latency'] < 40)
    if "Med (41-60ms)" in lat_sel: m_lat |= (input_df['latency'] >= 41) & (input_df['latency'] <= 60)
    if "High (>60ms)" in lat_sel: m_lat |= (input_df['latency'] > 60)
    return input_df[m_lat]

# --- CONSUMER VIEW ---
if user_role in ["John Doe", "Jane Nu", "Sam King"]:
    st.title(f"Consumer Marketplace | {user_role}")
    t1, t2, t3 = st.tabs(["🏛 Unified Gallery", "🚀 Contribute", "📊 My Impact"])
    
    with t1:
        q = st.text_input("💬 Chat Search: Find by client, task, or performance (e.g. 'NASA high accuracy')", placeholder="e.g. 'Supply Chain'")
        display_df = filter_registry(df)
        if q:
            display_df['blob'] = display_df.astype(str).apply(' '.join, axis=1)
            vec = TfidfVectorizer(stop_words='english')
            mtx = vec.fit_transform(display_df['blob'].tolist() + [q])
            display_df['score'] = cosine_similarity(mtx[-1], mtx[:-1])[0]
            display_df = display_df[display_df['score'] > 0.05].sort_values('score', ascending=False)
            # Log Search Efficacy
            new_log = pd.DataFrame([{"query": q, "found": len(display_df), "timestamp": str(datetime.datetime.now())}])
            pd.concat([search_logs, new_log]).to_csv(LOG_PATH, index=False)

        for i in range(0, len(display_df), 3):
            cols = st.columns(3)
            for j in range(3):
                if i+j < len(display_df):
                    row = display_df.iloc[i+j]
                    with cols[j]:
                        st.markdown(f"""
                        <div class="model-card">
                            <div>
                                <div style="display:flex; justify-content:space-between; font-size:0.6rem; color:gray;">
                                    <span>{row['domain']}</span><span>{row['type']}</span>
                                </div>
                                <div class="model-title">{row['name']}</div>
                                <div class="client-tag">Clients: {row['clients']}</div>
                                <div class="model-desc">{row['description']}</div>
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
                        if st.button("Request Access", key=f"btn_{row['name']}"):
                            req = pd.DataFrame([{"model_name": row['name'], "requester": user_role, "status": "Pending", "timestamp": str(datetime.datetime.now())}])
                            pd.concat([req_log, req]).to_csv(REQ_PATH, index=False)
                            st.toast("Request Sent!")

    with t2:
        with st.form("ingest_form", clear_on_submit=True):
            st.subheader("Model Metadata Ingestion")
            c1, c2 = st.columns(2)
            in_name = c1.text_input("Model Name")
            in_dom = c2.selectbox("Domain", ["Finance", "HR", "Supply Chain", "IT", "Legal"])
            in_cls = st.text_input("Clients Used In")
            in_desc = st.text_area("Full Description")
            if st.form_submit_button("Publish"):
                new_row = pd.DataFrame([{"name": in_name, "domain": in_dom, "type": "Community", "accuracy": 0.88, "latency": 35, "clients": in_cls, "description": in_desc, "contributor": user_role, "usage": 0, "data_drift": 0.01, "cpu_util": 15, "mem_util": 4, "throughput": 100, "error_rate": 0.01}])
                pd.concat([df, new_row]).to_csv(REG_PATH, index=False)
                st.success("Successfully Ingested")

    with t3:
        my_m = df[df['contributor'] == user_role]
        if not my_m.empty:
            st.metric("Total Views of Your Assets", my_m['usage'].sum())
            st.dataframe(my_m[['name', 'domain', 'accuracy', 'latency', 'usage']], use_container_width=True)
        else:
            st.info("No contributions yet.")

# --- LEADER VIEW ---
elif user_role == "Nat Patel (Leader)":
    st.title("Approval Gateway")
    pend = req_log[req_log['status'] == "Pending"]
    if not pend.empty:
        for idx, r in pend.iterrows():
            st.write(f"**{r['requester']}** wants access to **{r['model_name']}**")
            if st.button("Approve", key=f"ap_{idx}"):
                req_log.at[idx, 'status'] = "Approved"
                req_log.to_csv(REQ_PATH, index=False)
                st.rerun()
    else:
        st.success("No pending approvals.")

# --- ADMIN VIEW ---
else:
    st.title("Admin Governance Dashboard")
    
    # KPIs
    k1, k2, k3 = st.columns(3)
    k1.metric("Global Usage", df['usage'].sum())
    if len(search_logs) > 0:
        hit_rate = (len(search_logs[search_logs['found'] > 0]) / len(search_logs)) * 100
        k2.metric("Search Efficacy", f"{int(hit_rate)}%")
    k3.metric("Inventory Size", len(df))

    # INTERACTIVE PARALLEL PLOT WITH SEARCH & HIGHLIGHT
    st.divider()
    st.subheader("Interactive Portfolio Inspector")
    
    # 1. Search to Filter Plot
    admin_q = st.text_input("🔍 Search to filter the lines in the plot below...", placeholder="e.g. 'Finance' or 'NASA'")
    plot_df = df.copy()
    if admin_q:
        plot_df['blob'] = plot_df.astype(str).apply(' '.join, axis=1)
        v = TfidfVectorizer(stop_words='english')
        m = v.fit_transform(plot_df['blob'].tolist() + [admin_q])
        plot_df['score'] = cosine_similarity(m[-1], m[:-1])[0]
        plot_df = plot_df[plot_df['score'] > 0.05]
    
    # 2. Dropdown to Highlight specific model
    highlight_name = st.selectbox("🎯 Highlight a specific model line:", ["None"] + list(plot_df['name'].unique()))
    
    # 3. Create Custom Graph Object for Highlighting
    # Define colors: standard lines are semi-transparent purple, highlighted is Neon Gold
    colors = []
    line_widths = []
    for name in plot_df['name']:
        if name == highlight_name:
            colors.append("#FFD700") # Gold
            line_widths.append(6)
        else:
            colors.append("rgba(161, 0, 255, 0.3)") # Faded Purple
            line_widths.append(1)

    fig = go.Figure(data=go.Parcoords(
        line = dict(color = range(len(plot_df)), 
                   colorscale = [[0, 'rgba(161,0,255,0.2)'], [1, 'rgba(161,0,255,0.2)']], # Default
                   ),
        dimensions = list([
            dict(range = [0,1], label = 'Accuracy', values = plot_df['accuracy']),
            dict(range = [0,120], label = 'Latency (ms)', values = plot_df['latency']),
            dict(range = [0,15000], label = 'Usage', values = plot_df['usage']),
            dict(range = [0,0.2], label = 'Drift', values = plot_df['data_drift']),
            dict(range = [0,100], label = 'CPU %', values = plot_df['cpu_util'])
        ])
    ))
    
    # Overwrite the colors to apply highlighting
    fig.data[0].line.color = colors
    fig.data[0].line.width = line_widths

    st.plotly_chart(fig, use_container_width=True)
    
    # Data Table Drill-down
    st.subheader("Audit Table")
    st.dataframe(plot_df[['name', 'domain', 'accuracy', 'latency', 'usage', 'contributor']], use_container_width=True)
