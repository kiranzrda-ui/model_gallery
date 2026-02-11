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
        min-height: 380px; display: flex; flex-direction: column; justify-content: space-between;
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
REG_PATH = "model_registry_final.csv"
LOG_PATH = "search_logs_final.csv"
REQ_PATH = "requests_final.csv"

def init_files():
    if not os.path.exists(REG_PATH):
        doms = ["Finance", "HR", "Procurement", "Supply Chain", "IT", "Legal", "Marketing"]
        cls = ["Apple, NASA", "Amazon, BMW", "Coca-Cola, Shell", "JP Morgan, HSBC", "Samsung, Google", "Walmart, FedEx", "Meta, Microsoft"]
        data = []
        for i in range(60):
            d = random.choice(doms)
            data.append({
                "name": f"{d}-Asset-{i+100}", "domain": d, "type": "Official" if i < 20 else "Community",
                "accuracy": round(random.uniform(0.75, 0.99), 3), "latency": random.randint(15, 120),
                "clients": random.choice(cls), "use_cases": f"Scalable {d} processing",
                "description": f"Proprietary {d} architecture designed for high-concurrency enterprise workloads. Optimized for P99 latency and cross-domain data drift resilience.",
                "contributor": random.choice(["John Doe", "Jane Nu", "Sam King"]) if i >= 20 else "System",
                "usage": random.randint(100, 15000), "data_drift": round(random.uniform(0, 0.20), 3),
                "cpu_util": random.randint(10, 95), "mem_util": random.randint(4, 64), "throughput": random.randint(50, 2500), "error_rate": round(random.uniform(0, 4), 2)
            })
        pd.DataFrame(data).to_csv(REG_PATH, index=False)
    if not os.path.exists(LOG_PATH): pd.DataFrame(columns=["query", "found", "timestamp"]).to_csv(LOG_PATH, index=False)
    if not os.path.exists(REQ_PATH): pd.DataFrame(columns=["model_name", "requester", "status", "timestamp"]).to_csv(REQ_PATH, index=False)

init_files()
df_master = pd.read_csv(REG_PATH)
search_logs = pd.read_csv(LOG_PATH)
req_log = pd.read_csv(REQ_PATH)

# --- AUTH & FILTERS ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/Accenture.svg/2560px-Accenture.svg.png", width=120)
    user_role = st.selectbox("Current User", ["John Doe", "Jane Nu", "Sam King", "Nat Patel (Leader)", "Admin"])
    st.divider()
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
    t1, t2, t3 = st.tabs(["🏛 Unified Gallery", "🚀 Contribute", "📊 My Impact Dashboard"])
    
    with t1:
        q = st.text_input("💬 AI Search (Task, Client, or Logic)", placeholder="e.g. 'NASA high accuracy'")
        display_df = filter_registry(df_master)
        if q:
            display_df['blob'] = display_df.astype(str).apply(' '.join, axis=1)
            vec = TfidfVectorizer(stop_words='english')
            mtx = vec.fit_transform(display_df['blob'].tolist() + [q])
            display_df['score'] = cosine_similarity(mtx[-1], mtx[:-1])[0]
            display_df = display_df[display_df['score'] > 0.05].sort_values('score', ascending=False)
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
                            <div class="metric-box">
                                <span><b>ACC:</b> {int(row['accuracy']*100)}%</span>
                                <span><b>LAT:</b> {row['latency']}ms</span>
                                <span><b>USE:</b> {row['usage']}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        if st.button("Request Access", key=f"btn_{row['name']}"):
                            req = pd.DataFrame([{"model_name": row['name'], "requester": user_role, "status": "Pending", "timestamp": str(datetime.datetime.now())}])
                            pd.concat([req_log, req]).to_csv(REQ_PATH, index=False)
                            st.toast("Request Sent!")

    with t2:
        with st.form("ingest_form", clear_on_submit=True):
            st.subheader("Ingest New Asset")
            c1, c2 = st.columns(2)
            in_name = c1.text_input("Model Name")
            in_dom = c2.selectbox("Domain", ["Finance", "HR", "Supply Chain", "IT", "Legal"])
            in_cls = st.text_input("Clients Used In")
            in_desc = st.text_area("Full Description")
            if st.form_submit_button("Publish"):
                new_row = pd.DataFrame([{"name": in_name, "domain": in_dom, "type": "Community", "accuracy": 0.88, "latency": 35, "clients": in_cls, "description": in_desc, "contributor": user_role, "usage": 0, "data_drift": 0.01, "cpu_util": 15, "mem_util": 4, "throughput": 100, "error_rate": 0.01}])
                pd.concat([df_master, new_row]).to_csv(REG_PATH, index=False)
                st.success("Successfully Ingested")

    with t3:
        my_m = df_master[df_master['contributor'] == user_role]
        if not my_m.empty:
            st.subheader("Asset Performance Radar")
            sel_m = st.selectbox("Inspect Model", my_m['name'])
            m_dat = my_m[my_m['name'] == sel_m].iloc[0]
            
            # SPIDER CHART
            fig_radar = go.Figure(go.Scatterpolar(
                r=[m_dat['accuracy']*100, 100-m_dat['data_drift']*100, 100-m_dat['cpu_util'], 100-m_dat['error_rate']*10],
                theta=['Accuracy', 'Stability', 'Efficiency', 'Reliability'],
                fill='toself', line_color='#A100FF'
            ))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, height=400)
            st.plotly_chart(fig_radar, use_container_width=True)
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
    
    k1, k2, k3 = st.columns(3)
    k1.metric("Global Usage", df_master['usage'].sum())
    if len(search_logs) > 0:
        hit_rate = (len(search_logs[search_logs['found'] > 0]) / len(search_logs)) * 100
        k2.metric("Search Efficacy", f"{int(hit_rate)}%")
    k3.metric("Inventory Size", len(df_master))

    st.divider()
    st.subheader("Interactive Portfolio Inspector")
    
    admin_q = st.text_input("🔍 Filter Model Fleet (Search)", placeholder="e.g. 'NASA'")
    plot_df = df_master.copy()
    if admin_q:
        plot_df['blob'] = plot_df.astype(str).apply(' '.join, axis=1)
        v = TfidfVectorizer(stop_words='english')
        m = v.fit_transform(plot_df['blob'].tolist() + [admin_q])
        plot_df['score'] = cosine_similarity(m[-1], m[:-1])[0]
        plot_df = plot_df[plot_df['score'] > 0.05]
    
    hl_name = st.selectbox("🎯 Solo Line Inspector (Select Model):", ["None"] + list(plot_df['name'].unique()))
    
    # DATA & COLOR LOGIC TO PREVENT CRASH
    # Use a numeric list for line colors. 
    # If a model is selected, other lines are invisible.
    color_vals = []
    if hl_name == "None":
        color_vals = plot_df['accuracy'].tolist()
        c_scale = [[0, 'rgba(161, 0, 255, 0.2)'], [1, 'rgba(161, 0, 255, 0.8)']]
    else:
        # 1 for selected model, 0 for others
        color_vals = [1 if name == hl_name else 0 for name in plot_df['name']]
        # colorscale: 0 is transparent, 1 is Gold
        c_scale = [[0, 'rgba(0,0,0,0)'], [1, 'rgba(255, 215, 0, 1)']]

    fig_para = go.Figure(data=go.Parcoords(
        labelfont=dict(size=14, color='black'),
        tickfont=dict(size=10, color='gray'),
        line=dict(
            color=color_vals,
            colorscale=c_scale,
            showscale=False
        ),
        dimensions=list([
            dict(range=[0.7, 1.0], label='Accuracy', values=plot_df['accuracy']),
            dict(range=[0, 130], label='Latency', values=plot_df['latency']),
            dict(range=[0, 16000], label='Usage', values=plot_df['usage']),
            dict(range=[0, 0.25], label='Drift', values=plot_df['data_drift']),
            dict(range=[0, 100], label='CPU %', values=plot_df['cpu_util'])
        ])
    ))
    
    fig_para.update_layout(margin=dict(t=80, b=50, l=80, r=80), paper_bgcolor='white')
    st.plotly_chart(fig_para, use_container_width=True)
    
    st.dataframe(plot_df[['name', 'domain', 'accuracy', 'usage', 'contributor']], use_container_width=True)
