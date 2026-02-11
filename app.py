import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import datetime
import random
import csv

# --- CONFIG & STYLING ---
st.set_page_config(page_title="Accenture AI Marketplace | Enterprise Hub", layout="wide")

st.markdown("""
    <style>
    :root { --accent: #A100FF; --selected-red: #B71C1C; --pale-yellow: #FFFDE7; }
    span[data-baseweb="tag"] { background-color: #F3E5F5 !important; color: var(--accent) !important; }
    .model-card {
        border: 1px solid #e0e0e0; border-top: 4px solid var(--accent);
        padding: 12px; background-color: #ffffff; margin-bottom: 15px;
        min-height: 400px; display: flex; flex-direction: column; justify-content: space-between;
        border-radius: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .model-title { font-size: 1.1rem; font-weight: 700; color: #000; margin-bottom: 2px; }
    .client-tag { font-size: 0.75rem; color: var(--accent); font-weight: 600; margin-bottom: 5px; }
    .model-desc { font-size: 0.85rem; color: #444; line-height: 1.4; margin-bottom: 8px; 
                  display: -webkit-box; -webkit-line-clamp: 4; -webkit-box-orient: vertical; overflow: hidden; }
    .metric-box { background: #f8f9fa; padding: 6px; border-radius: 4px; display: flex; justify-content: space-between; font-size: 0.75rem; }
    .stButton>button { background-color: #000; color: #fff; border-radius: 0; font-size: 0.8rem; height: 35px; width: 100%; }
    .stButton>button:hover { background-color: var(--accent); }
    </style>
    """, unsafe_allow_html=True)

# --- REALISTIC SEED DATA ---
REAL_MODELS = [
    {"name": "BERT-Sentiment-Pro", "domain": "Marketing", "type": "Official", "accuracy": 0.94, "latency": 25, "clients": "Coca-Cola, Samsung", "use_cases": "Brand Sentiment Analysis", "description": "A fine-tuned BERT model for real-time sentiment analysis of social media feeds and customer reviews. High precision in detecting sarcasm and brand-specific emotions.", "contributor": "System"},
    {"name": "Prophet-Cashflow-V4", "domain": "Finance", "type": "Official", "accuracy": 0.88, "latency": 110, "clients": "JP Morgan, HSBC", "use_cases": "Forecasting Cash Reserves", "description": "Time-series forecasting model using the Prophet library. Designed to predict quarterly cash flow cycles with seasonality adjustments for global banking holidays.", "contributor": "System"},
    {"name": "XGBoost-Fraud-Shield", "domain": "Finance", "type": "Official", "accuracy": 0.99, "latency": 12, "clients": "Barclays, Shell", "use_cases": "Credit Card Fraud Detection", "description": "Ultra-low latency gradient boosting model for transactional fraud detection. Trained on millions of anonymized payment records to identify high-risk anomalies in real-time.", "contributor": "System"},
    {"name": "Supply-Chain-Twin", "domain": "Supply Chain", "type": "Official", "accuracy": 0.91, "latency": 65, "clients": "FedEx, Walmart", "use_cases": "Inventory Optimization", "description": "Digital twin simulator powered by reinforcement learning. Optimizes stock levels across multi-node distribution centers while reducing logistics costs.", "contributor": "System"},
    {"name": "Legal-Clause-Extractor", "domain": "Legal", "type": "Official", "accuracy": 0.85, "latency": 450, "clients": "Apple, Toyota", "use_cases": "Contract Risk Analysis", "description": "NLP transformer model specializing in Named Entity Recognition (NER) for legal documents. Automatically flags non-standard liability and termination clauses in vendor contracts.", "contributor": "System"},
    {"name": "HR-Retention-Pulse", "domain": "HR", "type": "Official", "accuracy": 0.82, "latency": 35, "clients": "Google, Meta", "use_cases": "Employee Churn Prediction", "description": "Predictive classifier that analyzes engagement scores and work patterns to identify high-risk flight profiles. Built with ethical AI constraints to remove demographic bias.", "contributor": "System"}
]

# --- PERSISTENCE ENGINE ---
REG_PATH = "model_registry_v3.csv"
LOG_PATH = "search_logs_v3.csv"
REQ_PATH = "requests_v3.csv"

def init_files():
    """Ensures CSVs exist and are populated with 1000 models."""
    if not os.path.exists(REG_PATH):
        doms = ["Finance", "HR", "Procurement", "Supply Chain", "IT", "Legal", "Marketing", "ESG", "R&D"]
        clients = ["Apple", "NASA", "Amazon", "Coca-Cola", "BMW", "Samsung", "Walmart", "FedEx", "Meta", "Microsoft", "JP Morgan", "Shell", "Toyota", "AstraZeneca"]
        users = ["John Doe", "Jane Nu", "Sam King"]

        data = []
        # Add real seeds
        for m in REAL_MODELS:
            m.update({"usage": random.randint(5000, 20000), "data_drift": 0.02, "pred_drift": 0.02, "cpu_util": 35, "mem_util": 8, "throughput": 500, "error_rate": 0.01})
            data.append(m)

        # Generate up to 1000
        for i in range(len(data), 1000):
            d = random.choice(doms)
            c_list = ", ".join(random.sample(clients, 2))
            contributor = users[i % 3]
            data.append({
                "name": f"{random.choice(['Neural', 'Core', 'Alpha', 'Quantum'])}-{d}-v{i}",
                "domain": d, "type": "Community", "accuracy": round(random.uniform(0.70, 0.99), 3),
                "latency": random.randint(10, 250), "clients": c_list, "use_cases": f"Optimization for {d}",
                "description": f"Standard {d} optimization model built for enterprise scale. Trained on global datasets for {c_list}.",
                "contributor": contributor, "usage": random.randint(10, 5000), "data_drift": round(random.uniform(0, 0.25), 3),
                "pred_drift": 0.05, "cpu_util": random.randint(5, 95), "mem_util": random.randint(2, 64),
                "throughput": random.randint(10, 2000), "error_rate": round(random.uniform(0, 5), 2)
            })
        pd.DataFrame(data).to_csv(REG_PATH, index=False, quoting=csv.QUOTE_ALL)
    
    if not os.path.exists(LOG_PATH): pd.DataFrame(columns=["query", "found", "timestamp"]).to_csv(LOG_PATH, index=False)
    if not os.path.exists(REQ_PATH): pd.DataFrame(columns=["model_name", "requester", "status", "timestamp"]).to_csv(REQ_PATH, index=False)

init_files()
df_master = pd.read_csv(REG_PATH)
search_logs = pd.read_csv(LOG_PATH)
req_log = pd.read_csv(REQ_PATH)

# --- SIDEBAR & AUTH ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/Accenture.svg/2560px-Accenture.svg.png", width=120)
    current_user = st.selectbox("Login Profile", ["John Doe", "Jane Nu", "Sam King", "Nat Patel (Leader)", "Admin"])
    st.divider()
    st.subheader("Performance Tiers")
    acc_sel = st.multiselect("Accuracy Class", ["High (>98%)", "Medium (80-97%)", "Low (<80%)"], default=["High (>98%)", "Medium (80-97%)"])
    lat_sel = st.multiselect("Latency Class", ["Low (<40ms)", "Med (41-60ms)", "High (>60ms)"], default=["Low (<40ms)", "Med (41-60ms)", "High (>60ms)"])

def filter_registry(df):
    m_acc = pd.Series([False] * len(df), index=df.index)
    if "High (>98%)" in acc_sel: m_acc |= (df['accuracy'] >= 0.98)
    if "Medium (80-97%)" in acc_sel: m_acc |= (df['accuracy'] >= 0.80) & (df['accuracy'] < 0.98)
    if "Low (<80%)" in acc_sel: m_acc |= (df['accuracy'] < 0.80)
    df = df[m_acc]
    
    m_lat = pd.Series([False] * len(df), index=df.index)
    if "Low (<40ms)" in lat_sel: m_lat |= (df['latency'] < 40)
    if "Med (41-60ms)" in lat_sel: m_lat |= (df['latency'] >= 41) & (df['latency'] <= 60)
    if "High (>60ms)" in lat_sel: m_lat |= (df['latency'] > 60)
    return df[m_lat]

# --- VIEWS ---
if current_user in ["John Doe", "Jane Nu", "Sam King"]:
    st.title(f"Consumer Hub | {current_user}")
    t1, t2, t3 = st.tabs(["🏛 Unified Gallery", "🚀 Contribute Asset", "👤 My Dashboard"])
    
    with t1:
        q = st.text_input("💬 Chat Search: Describe a client name or model description keyword...", placeholder="e.g. 'NASA high accuracy' or 'sentiment'")
        display_df = filter_registry(df_master)
        if q:
            display_df['blob'] = display_df.astype(str).apply(' '.join, axis=1)
            vec = TfidfVectorizer(stop_words='english')
            mtx = vec.fit_transform(display_df['blob'].tolist() + [q])
            display_df['score'] = cosine_similarity(mtx[-1], mtx[:-1])[0]
            display_df = display_df[display_df['score'] > 0.05].sort_values('score', ascending=False)
            new_log = pd.DataFrame([{"query": q, "found": len(display_df), "timestamp": str(datetime.datetime.now())}])
            pd.concat([search_logs, new_log]).to_csv(LOG_PATH, index=False)

        st.caption(f"Showing {len(display_df)} models matched.")
        for i in range(0, min(len(display_df), 30), 3):
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
                                <div style="font-size:0.65rem; color:#888;">By: {row['contributor']}</div>
                            </div>
                            <div class="metric-box">
                                <span><b>ACC:</b> {int(row['accuracy']*100)}%</span>
                                <span><b>LAT:</b> {row['latency']}ms</span>
                                <span><b>USE:</b> {row['usage']}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        if st.button("Request Access", key=f"r_{row['name']}"):
                            req = pd.DataFrame([{"model_name": row['name'], "requester": current_user, "status": "Pending", "timestamp": str(datetime.datetime.now())}])
                            pd.concat([req_log, req]).to_csv(REQ_PATH, index=False)
                            st.toast("Access request sent.")

    with t2:
        with st.form("ingest_form", clear_on_submit=True):
            st.subheader("Model Metadata Ingestion")
            c1, c2 = st.columns(2)
            in_name = c1.text_input("Asset Name", placeholder="e.g. Sales-Forecaster-v1")
            in_dom = c2.selectbox("Domain", ["Finance", "HR", "Supply Chain", "IT", "Legal", "Marketing"])
            in_cls = st.text_input("Client Projects", placeholder="e.g. Shell, Walmart")
            in_desc = st.text_area("Detailed Description (Keywords entered here help search ranking)", help="Example: This model uses XGBoost for predicting invoice fraud.")
            if st.form_submit_button("Publish to Marketplace"):
                if in_name and in_desc:
                    new_row = pd.DataFrame([{
                        "name": in_name, "domain": in_dom, "type": "Community", "accuracy": 0.88, "latency": 35, 
                        "clients": in_cls, "description": in_desc, "contributor": current_user, "usage": 0, 
                        "data_drift": 0.01, "pred_drift": 0.01, "cpu_util": 15, "mem_util": 4, "throughput": 100, "error_rate": 0.01,
                        "use_cases": "New Entry"
                    }])
                    # APPEND TO MASTER AND SAVE TO CSV
                    df_master = pd.concat([df_master, new_row], ignore_index=True)
                    df_master.to_csv(REG_PATH, index=False, quoting=csv.QUOTE_ALL)
                    st.success(f"Success! '{in_name}' has been added to the registry and is now searchable.")
                else:
                    st.error("Model Name and Description are required.")

    with t3:
        my_m = df_master[df_master['contributor'] == current_user]
        if not my_m.empty:
            st.subheader(f"Portfolio Metrics for {current_user}")
            k1, k2, k3 = st.columns(3)
            k1.metric("Assets Contributed", len(my_m))
            k2.metric("Aggregate Usage", f"{my_m['usage'].sum():,}")
            k3.metric("Avg Portfolio Accuracy", f"{int(my_m['accuracy'].mean()*100)}%")
            
            st.divider()
            sel_m = st.selectbox("Deep-Dive Inspection", my_m['name'])
            m_dat = my_m[my_m['name'] == sel_m].iloc[0]
            
            col_r, col_t = st.columns([2, 1])
            with col_r:
                fig_radar = go.Figure(go.Scatterpolar(
                    r=[m_dat['accuracy']*100, 100-m_dat['data_drift']*100, 100-m_dat['cpu_util'], 100-m_dat['error_rate']*10],
                    theta=['Accuracy', 'Stability', 'Efficiency', 'Reliability'],
                    fill='toself', line_color='#A100FF'
                ))
                fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, title="Model Health Radar")
                st.plotly_chart(fig_radar, use_container_width=True)
            with col_t:
                st.write("**Runtime Telemetry**")
                st.info(f"Throughput: {m_dat['throughput']} req/s")
                st.info(f"CPU: {m_dat['cpu_util']}% | Mem: {m_dat['mem_util']}GB")
        else:
            st.info("No contributions found. Use the 'Contribute' tab to start.")

elif current_user == "Nat Patel (Leader)":
    st.title("Leader Approval Gateway")
    pend = req_log[req_log['status'] == "Pending"]
    if not pend.empty:
        for idx, r in pend.iterrows():
            st.write(f"💼 **{r['requester']}** requested access to **{r['model_name']}**")
            if st.button("Approve Access", key=f"ap_{idx}"):
                req_log.at[idx, 'status'] = "Approved"
                req_log.to_csv(REQ_PATH, index=False)
                st.rerun()
    else:
        st.success("No pending approvals.")

else: # ADMIN VIEW
    st.title("Admin Governance Dashboard")
    k1, k2, k3 = st.columns(3)
    k1.metric("Global Usage Volume", f"{df_master['usage'].sum():,}")
    if len(search_logs) > 0:
        eff = (len(search_logs[search_logs['found'] > 0]) / len(search_logs)) * 100
        k2.metric("Search Efficacy", f"{int(eff)}%")
    k3.metric("Inventory Size", len(df_master))
    
    st.divider()
    st.subheader("Interactive Portfolio Inspector")
    admin_q = st.text_input("🔍 Search to filter fleet view...", placeholder="e.g. 'NASA'")
    plot_df = df_master.copy()
    if admin_q:
        plot_df['blob'] = plot_df.astype(str).apply(' '.join, axis=1)
        v = TfidfVectorizer(stop_words='english')
        m = v.fit_transform(plot_df['blob'].tolist() + [admin_q])
        plot_df['score'] = cosine_similarity(m[-1], m[:-1])[0]
        plot_df = plot_df[plot_df['score'] > 0.05]
    
    hl_name = st.selectbox("🎯 Solo Line Inspector:", ["None"] + list(plot_df['name'].unique()))
    
    color_vals, c_scale = [], []
    if hl_name == "None":
        color_vals = [0.5] * len(plot_df)
        c_scale = [[0, 'rgba(161, 0, 255, 0.4)'], [1, 'rgba(161, 0, 255, 0.4)']]
    else:
        color_vals = [1.0 if name == hl_name else 0.0 for name in plot_df['name']]
        c_scale = [[0, 'rgba(255, 249, 196, 0.2)'], [1, '#B71C1C']]

    fig_para = go.Figure(data=go.Parcoords(
        labelfont=dict(size=14, color='black', family='Arial'),
        tickfont=dict(size=10, color='#666'),
        line=dict(color=color_vals, colorscale=c_scale, showscale=False),
        dimensions=list([
            dict(range=[0.7, 1.0], label='Accuracy', values=plot_df['accuracy']),
            dict(range=[0, 250], label='Latency', values=plot_df['latency']),
            dict(range=[0, 25000], label='Usage', values=plot_df['usage']),
            dict(range=[0, 0.3], label='Drift', values=plot_df['data_drift']),
            dict(range=[0, 100], label='CPU %', values=plot_df['cpu_util'])
        ])
    ))
    fig_para.update_layout(margin=dict(t=100, b=50, l=100, r=100), paper_bgcolor='white', height=550)
    st.plotly_chart(fig_para, use_container_width=True)
