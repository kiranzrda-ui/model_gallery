import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import datetime
import random

# --- CONFIGURATION ---
st.set_page_config(page_title="Accenture AI Marketplace | Telemetry", layout="wide")

# CUSTOM CSS
st.markdown("""
    <style>
    :root { --accent-purple: #A100FF; --light-purple: #F3E5F5; --success-green: #2E7D32; --warning-orange: #EF6C00; --error-red: #C62828; }
    span[data-baseweb="tag"] { background-color: var(--light-purple) !important; color: var(--accent-purple) !important; }
    
    .model-card {
        border: 1px solid #e0e0e0; border-top: 3px solid var(--accent-purple);
        padding: 12px; background-color: #ffffff; margin-bottom: 10px;
        min-height: 320px; display: flex; flex-direction: column; justify-content: space-between;
        border-radius: 4px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .status-dot { height: 10px; width: 10px; border-radius: 50%; display: inline-block; margin-right: 5px; }
    .dot-green { background-color: var(--success-green); }
    .dot-yellow { background-color: var(--warning-orange); }
    .dot-red { background-color: var(--error-red); }
    
    .model-title { font-size: 1rem; font-weight: 700; color: #000; margin: 4px 0; }
    .compact-metrics { display: flex; justify-content: space-between; background: #f8f9fa; padding: 6px; border-radius: 4px; font-size: 0.7rem; margin-top: 5px; }
    .telemetry-tag { font-size: 0.65rem; color: #666; font-family: monospace; }
    
    .stButton>button { background-color: #000; color: white; border-radius: 0px; font-size: 0.75rem; width: 100%; }
    .stButton>button:hover { background-color: var(--accent-purple); }
    </style>
    """, unsafe_allow_html=True)

# --- PERSISTENCE LAYER ---
REG_PATH = "model_registry_v2.csv"
REQ_PATH = "requests_v2.csv"

def init_data():
    if not os.path.exists(REG_PATH):
        data = []
        doms = ["Finance", "HR", "Procurement", "Supply Chain", "IT", "Legal"]
        users = ["John Doe", "Jane Nu", "Sam King"]
        for i in range(45):
            drift = round(random.uniform(0, 0.15), 3)
            data.append({
                "name": f"{random.choice(doms)}-{i+100}", "domain": random.choice(doms),
                "type": "Official" if i < 15 else "Community", "accuracy": random.choice([0.99, 0.92, 0.75]),
                "latency": random.choice([30, 50, 80]), "contributor": "System" if i < 15 else random.choice(users),
                "usage": random.randint(500, 10000), "description": "Enterprise-scale predictive asset.",
                "data_drift": drift, "pred_drift": round(drift * 1.2, 3),
                "cpu_util": random.randint(10, 80), "mem_util": random.randint(2, 16),
                "throughput": random.randint(50, 1000), "error_rate": round(random.uniform(0, 2), 2)
            })
        pd.DataFrame(data).to_csv(REG_PATH, index=False)
    if not os.path.exists(REQ_PATH):
        pd.DataFrame(columns=["model_name", "requester", "status", "timestamp"]).to_csv(REQ_PATH, index=False)

init_data()
registry = pd.read_csv(REG_PATH)
requests = pd.read_csv(REQ_PATH)

# --- SIDEBAR FILTERS ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/Accenture.svg/2560px-Accenture.svg.png", width=120)
    user_role = st.selectbox("Identity Switch", ["John Doe", "Jane Nu", "Sam King", "Nat Patel (Leader)", "Admin"])
    st.divider()
    st.subheader("Filter Performance")
    acc_sel = st.multiselect("Accuracy", ["High (>98%)", "Medium (80-97%)", "Low (<80%)"], default=["High (>98%)", "Medium (80-97%)"])
    lat_sel = st.multiselect("Latency", ["Low (<40ms)", "Med (41-60ms)", "High (>60ms)"], default=["Low (<40ms)", "Med (41-60ms)", "High (>60ms)"])

# Filter Logic
def apply_logic(df):
    mask = pd.Series([False] * len(df), index=df.index)
    if "High (>98%)" in acc_sel: mask |= (df['accuracy'] >= 0.98)
    if "Medium (80-97%)" in acc_sel: mask |= (df['accuracy'] >= 0.80) & (df['accuracy'] < 0.98)
    if "Low (<80%)" in acc_sel: mask |= (df['accuracy'] < 0.80)
    df = df[mask]
    
    mask_lat = pd.Series([False] * len(df), index=df.index)
    if "Low (<40ms)" in lat_sel: mask_lat |= (df['latency'] < 40)
    if "Med (41-60ms)" in lat_sel: mask_lat |= (df['latency'] >= 41) & (df['latency'] <= 60)
    if "High (>60ms)" in lat_sel: mask_lat |= (df['latency'] > 60)
    return df[mask_lat]

# --- UI LOGIC ---

if user_role in ["John Doe", "Jane Nu", "Sam King"]:
    st.title(f"Marketplace Hub | {user_role}")
    t_gal, t_con, t_my = st.tabs(["🏛 Unified Gallery", "🚀 Ingest Model", "📊 My Portfolio"])

    with t_gal:
        q = st.text_input("🔍 Search by task, client, or keyword...", placeholder="e.g. 'Finance' or 'Drift < 0.05'")
        display_df = apply_logic(registry)
        if q:
            display_df['blob'] = display_df.astype(str).apply(' '.join, axis=1)
            vectorizer = TfidfVectorizer(stop_words='english')
            matrix = vectorizer.fit_transform(display_df['blob'].tolist() + [q])
            display_df['score'] = cosine_similarity(matrix[-1], matrix[:-1])[0]
            display_df = display_df[display_df['score'] > 0].sort_values('score', ascending=False)

        # 3-Column Grid
        for i in range(0, len(display_df), 3):
            cols = st.columns(3)
            for j in range(3):
                if i+j < len(display_df):
                    row = display_df.iloc[i+j]
                    drift_status = "dot-green" if row['data_drift'] < 0.05 else "dot-yellow" if row['data_drift'] < 0.1 else "dot-red"
                    with cols[j]:
                        st.markdown(f"""
                        <div class="model-card">
                            <div>
                                <div style="display:flex; justify-content:space-between;">
                                    <span class="domain-tag">{row['domain']}</span>
                                    <span class="status-dot {drift_status}" title="Health Status"></span>
                                </div>
                                <div class="model-title">{row['name']}</div>
                                <div class="telemetry-tag">⚡ Drift: {row['data_drift']} | 📈 Err: {row['error_rate']}%</div>
                                <div class="model-desc">{row['description']}</div>
                            </div>
                            <div>
                                <div class="compact-metrics">
                                    <span><b>ACC:</b> {int(row['accuracy']*100)}%</span>
                                    <span><b>LAT:</b> {row['latency']}ms</span>
                                    <span><b>TPUT:</b> {row['throughput']}/s</span>
                                </div>
                                <div style="height:10px;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        if st.button("Request Access", key=f"req_{row['name']}"):
                            new_req = pd.DataFrame([{"model_name": row['name'], "requester": user_role, "status": "Pending", "timestamp": str(datetime.datetime.now())}])
                            pd.concat([requests, new_req]).to_csv(REQ_PATH, index=False)
                            st.toast("Request Sent to Nat Patel")

    with t_con:
        with st.form("ingest_form", clear_on_submit=True):
            st.subheader("Model Metadata Ingestion")
            c1, c2 = st.columns(2)
            n = c1.text_input("Model Name")
            d = c2.selectbox("Domain", ["Finance", "HR", "Supply Chain", "IT", "Legal"])
            desc = st.text_area("Functionality Description")
            st.divider()
            st.caption("Runtime Simulation (Auto-generated for prototype)")
            if st.form_submit_button("Publish to Marketplace"):
                new_m = pd.DataFrame([{
                    "name": n, "domain": d, "type": "Community", "accuracy": 0.85, "latency": 45,
                    "contributor": user_role, "usage": 0, "description": desc,
                    "data_drift": 0.02, "pred_drift": 0.02, "cpu_util": 30, "mem_util": 4,
                    "throughput": 100, "error_rate": 0.05
                }])
                pd.concat([registry, new_m]).to_csv(REG_PATH, index=False)
                st.success("Asset Ingested Successfully")

    with t_my:
        my_mods = registry[registry['contributor'] == user_role]
        if not my_mods.empty:
            st.subheader("Live Telemetry: Your Assets")
            selected_mod = st.selectbox("Select Model to Inspect", my_mods['name'])
            m_data = my_mods[my_mods['name'] == selected_mod].iloc[0]
            
            col_a, col_b = st.columns([1, 2])
            with col_a:
                # Wow Factor 1: Radar Chart
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=[m_data['accuracy']*100, 100-(m_data['data_drift']*100), 100-m_data['cpu_util'], 100-m_data['error_rate']*10],
                    theta=['Accuracy', 'Data Stability', 'CPU Efficiency', 'Reliability'],
                    fill='toself', name=selected_mod, line_color='#A100FF'
                ))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, title="Health Radar")
                st.plotly_chart(fig, use_container_width=True)
            
            with col_b:
                # Wow Factor 2: Resource Gauges
                g1, g2 = st.columns(2)
                with g1:
                    fig_g = go.Figure(go.Indicator(mode="gauge+number", value=m_data['cpu_util'], title={'text': "CPU %"}, gauge={'bar': {'color': "#A100FF"}}))
                    fig_g.update_layout(height=250)
                    st.plotly_chart(fig_g, use_container_width=True)
                with g2:
                    fig_g2 = go.Figure(go.Indicator(mode="gauge+number", value=m_data['throughput'], title={'text': "Req/Sec"}, gauge={'bar': {'color': "#000000"}}))
                    fig_g2.update_layout(height=250)
                    st.plotly_chart(fig_g2, use_container_width=True)
        else:
            st.info("No models found in your portfolio.")

elif user_role == "Nat Patel (Leader)":
    st.title("Approval Gateway | Nat Patel")
    st.markdown("---")
    pending = requests[requests['status'] == "Pending"]
    if not pending.empty:
        for idx, row in pending.iterrows():
            with st.container():
                c1, c2, c3 = st.columns([3, 2, 1])
                c1.write(f"💼 **{row['requester']}** requested **{row['model_name']}**")
                c2.write(f"🕒 {row['timestamp'][:16]}")
                if c3.button("Approve", key=f"app_{idx}"):
                    requests.at[idx, 'status'] = "Approved"
                    requests.to_csv(REQ_PATH, index=False)
                    st.rerun()
    else:
        st.success("Approval queue is empty.")

else: # ADMIN VIEW
    st.title("Enterprise Governance Dashboard")
    st.markdown("---")
    
    # Wow Factor 3: High-dimensional Analysis
    st.subheader("Global Model Portfolio Analysis")
    fig = px.parallel_coordinates(registry, color="accuracy", 
                             dimensions=['accuracy', 'latency', 'data_drift', 'cpu_util', 'usage'],
                             color_continuous_scale=px.colors.sequential.Purples)
    st.plotly_chart(fig, use_container_width=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("System Resource Heatmap")
        fig_heat = px.density_heatmap(registry, x="cpu_util", y="mem_util", z="usage", 
                                     color_continuous_scale="Purples", title="Compute vs Popularity")
        st.plotly_chart(fig_heat, use_container_width=True)
    with c2:
        st.subheader("Audit Logs")
        st.dataframe(requests.sort_values('timestamp', ascending=False), use_container_width=True)
