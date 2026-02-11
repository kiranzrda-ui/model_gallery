import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import datetime
import random
import re
import csv

# --- CONFIG & STYLING ---
st.set_page_config(page_title="Model Hub 2.0", layout="wide")

st.markdown("""
    <style>
    :root { --accent: #6200EE; --success: #2E7D32; --warning: #F9A825; --critical: #C62828; }
    .stApp { background-color: #F4F7F9; }
    
    /* Compact Card UI */
    .model-card {
        background: white; border: 1px solid #e0e6ed; border-top: 3px solid var(--accent);
        padding: 12px; border-radius: 6px; min-height: 340px; 
        display: flex; flex-direction: column; justify-content: space-between;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .model-title { font-size: 0.95rem; font-weight: 700; color: #1a202c; display: flex; justify-content: space-between; }
    .registry-badge { font-size: 0.6rem; padding: 2px 5px; border-radius: 4px; background: #EDF2F7; color: #4A5568; border: 1px solid #CBD5E0; }
    
    .meta-line { font-size: 0.72rem; color: #718096; margin: 1px 0; }
    .use-case-text { font-size: 0.75rem; color: #4A5568; margin: 8px 0; height: 3.2em; overflow: hidden; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; }
    
    .metric-bar { display: flex; justify-content: space-between; background: #F8FAFC; padding: 5px; border-radius: 4px; border: 1px solid #E2E8F0; }
    .metric-val { font-size: 0.8rem; font-weight: 700; color: var(--accent); text-align: center; flex: 1; }
    .metric-label { font-size: 0.55rem; color: #A0AEC0; display: block; text-transform: uppercase; }

    /* Button Consistency */
    .stButton>button { border-radius: 4px; font-size: 0.75rem; height: 28px; width: 100%; }
    .stDownloadButton>button { background-color: var(--accent) !important; color: white !important; width: 100%; }
    </style>
    """, unsafe_allow_html=True)

# --- DATA PERSISTENCE ---
REG_PATH = "model_registry_v3.csv"
REQ_PATH = "requests_v3.csv"
LOG_PATH = "search_logs_v3.csv"

# Initialize Session State for the "Compare Basket"
if 'compare_basket' not in st.session_state:
    st.session_state.compare_basket = []

def load_data():
    if not os.path.exists(REG_PATH):
        st.error("Registry CSV missing. Please upload model_registry_v3.csv")
        return pd.DataFrame()
    df = pd.read_csv(REG_PATH)
    # Ensure Enterprise ROI Columns exist
    if 'revenue_impact' not in df.columns:
        df['revenue_impact'] = df['usage'] * 15.5
        df['registry_provider'] = [random.choice(['MLflow', 'Vertex AI', 'SageMaker']) for _ in range(len(df))]
    return df

df_master = load_data()

# --- SEARCH ENGINE ---
def run_advanced_search(query, df):
    if not query: return df
    q = query.lower()
    # 1. Regex logic for metrics
    match = re.search(r'(\w+)\s*([<>]=?)\s*(\d+)', q)
    if match:
        col, op, val = match.groups()
        val = float(val)/100 if col == 'accuracy' and float(val) > 1 else float(val)
        if col in df.columns:
            if '>' in op: df = df[df[col] >= val]
            else: df = df[df[col] <= val]
    
    # 2. Semantic Search
    df['blob'] = df.fillna('').astype(str).apply(' '.join, axis=1)
    vec = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
    mtx = vec.fit_transform(df['blob'].tolist() + [query])
    df['relevance'] = cosine_similarity(mtx[-1], mtx[:-1])[0]
    return df[df['relevance'] > 0.01].sort_values('relevance', ascending=False)

# --- NAVIGATION ---
with st.sidebar:
    st.title("Enterprise Hub")
    view = st.radio("Navigation", ["Marketplace Hub", "Strategy ROI", "Admin Ops", "Compare Basket"])
    st.divider()
    user = st.selectbox("Switch User", ["John Doe", "Jane Nu", "Sam King", "Nat Patel (Leader)", "Admin"])
    if st.session_state.compare_basket:
        st.success(f"Comparing: {len(st.session_state.compare_basket)} models")
        if st.button("Reset Basket"): st.session_state.compare_basket = []; st.rerun()

# --- 1. MARKETPLACE HUB ---
if view == "Marketplace Hub":
    t1, t2, t3 = st.tabs(["🏛 Unified Gallery", "🔥 Smart Discovery", "🚀 New Ingestion"])
    
    with t1:
        q = st.text_input("💬 Smart Search (e.g. 'SageMaker Finance' or 'accuracy > 0.95')", placeholder="Keywords or operators...")
        results = run_advanced_search(q, df_master)
        
        for i in range(0, min(len(results), 21), 3):
            cols = st.columns(3)
            for j in range(3):
                if i+j < len(results):
                    row = results.iloc[i+j]
                    with cols[j]:
                        st.markdown(f"""
                        <div class="model-card">
                            <div>
                                <div class="model-title">
                                    <span>{row['name'][:22]}</span>
                                    <span class="registry-badge">{row['registry_provider']}</span>
                                </div>
                                <div class="meta-line"><b style="color:#6200EE;">{row['domain']}</b> | {row['model_stage']}</div>
                                <div class="meta-line">Team: {row['model_owner_team']} | v{row['model_version']}</div>
                                <div class="use-case-text">{row['use_cases']}</div>
                            </div>
                            <div>
                                <div class="metric-bar">
                                    <div class="metric-val"><span class="metric-label">Acc</span>{int(row['accuracy']*100)}%</div>
                                    <div class="metric-val"><span class="metric-label">Lat</span>{row['latency']}ms</div>
                                    <div class="metric-val"><span class="metric-label">Drift</span>{row['data_drift']}</div>
                                </div>
                                <div style="height:10px;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Fix Plotly Duplicate ID: Encapsulate interactive elements
                        c_b1, c_b2, c_b3 = st.columns(3)
                        if c_b1.button("Compare", key=f"btn_comp_{i+j}"):
                            if row['name'] not in st.session_state.compare_basket:
                                st.session_state.compare_basket.append(row['name'])
                                st.toast("Added!")
                        
                        with c_b2:
                            with st.popover("Specs"):
                                st.write(f"**Lineage & Features: {row['name']}**")
                                # Lineage Sankey
                                fig_s = go.Figure(go.Sankey(node=dict(label=["Source","Train","Model","Prod"], color="purple"), link=dict(source=[0,1,2], target=[1,2,3], value=[1,1,1])))
                                fig_s.update_layout(height=140, margin=dict(l=0,r=0,t=0,b=0))
                                st.plotly_chart(fig_s, use_container_width=True, key=f"sankey_{i+j}")
                                
                                st.write("**Feature Importance**")
                                f_df = pd.DataFrame({'f':['A','B','C'], 'v':[.4,.3,.2]})
                                st.plotly_chart(px.bar(f_df, x='v', y='f', orientation='h', height=120), use_container_width=True, key=f"feat_{i+j}")
                                st.button("Launch Notebook", key=f"nb_{i+j}")

                        if c_b3.button("Access", key=f"btn_acc_{i+j}"):
                            st.toast("Request sent to Nat Patel")

    with t2:
        st.subheader("Automated Insights")
        c_a, c_b = st.columns(2)
        with c_a:
            st.markdown("#### 🔥 Trending (Most Adoption)")
            st.dataframe(df_master.nlargest(5, 'usage')[['name', 'usage', 'domain']], use_container_width=True)
            st.markdown("#### ⚠️ High Drift Risk")
            st.dataframe(df_master[df_master['data_drift'] > 0.09][['name', 'data_drift', 'monitoring_status']], use_container_width=True)
        with c_b:
            st.markdown("#### 💎 Hidden Gems (High Acc / Low Use)")
            gems = df_master[(df_master['accuracy'] > 0.96) & (df_master['usage'] < 1000)]
            st.dataframe(gems[['name', 'accuracy', 'usage']], use_container_width=True)
            st.markdown("#### 🕒 Recently Improved")
            st.dataframe(df_master.sort_values('last_retrained_date', ascending=False).head(5)[['name', 'last_retrained_date']], use_container_width=True)

    with t3:
        st.subheader("Model Ingestion Connector")
        with st.form("ingest_v5", clear_on_submit=True):
            f_n = st.text_input("Model Name*")
            f_d = st.selectbox("Domain", df_master['domain'].unique())
            f_p = st.selectbox("Registry Provider", ["MLflow", "Vertex AI", "SageMaker"])
            f_desc = st.text_area("Description / Lineage Info*")
            if st.form_submit_button("Publish to Fleet"):
                if f_n and f_desc:
                    new_row = pd.DataFrame([{
                        "name": f_n, "domain": f_d, "accuracy": 0.85, "latency": 40, "registry_provider": f_p,
                        "use_cases": f_desc, "contributor": user, "usage": 0, "data_drift": 0.0, "model_version": "1.0.0",
                        "model_stage": "Experimental", "last_retrained_date": str(datetime.date.today()), "model_owner_team": "Internal",
                        "monitoring_status": "Healthy", "sla_tier": "Bronze", "revenue_impact": 0
                    }])
                    df_master = pd.concat([df_master, new_row], ignore_index=True)
                    df_master.to_csv(REG_PATH, index=False)
                    st.success("Asset live in registry!")

# --- 2. COMPARE TOOL ---
elif view == "Compare Basket":
    st.header("Side-by-Side Asset Benchmark")
    if not st.session_state.compare_basket:
        st.info("Your basket is empty. Go to Marketplace and click 'Compare' on models.")
    else:
        c_df = df_master[df_master['name'].isin(st.session_state.compare_basket)]
        st.dataframe(c_df[['name', 'accuracy', 'latency', 'data_drift', 'cpu_util', 'sla_tier', 'registry_provider']])
        
        st.subheader("Performance Matrix")
        fig_c = px.bar(c_df, x='name', y=['accuracy', 'data_drift'], barmode='group', color_discrete_sequence=['#6200EE', '#FFD700'])
        st.plotly_chart(fig_c, use_container_width=True)

# --- 3. STRATEGY ROI ---
elif view == "Portfolio ROI":
    st.header("Strategic Portfolio Performance")
    agg = df_master.groupby('domain').agg({'revenue_impact': 'sum', 'accuracy': 'mean', 'usage': 'sum'}).reset_index()
    
    k1, k2, k3 = st.columns(3)
    k1.metric("Revenue Contribution", f"${agg['revenue_impact'].sum()/1e6:.1f}M")
    k2.metric("Avg Fleet Quality", f"{int(agg['accuracy'].mean()*100)}%")
    k3.metric("Total Adoption", f"{int(agg['usage'].sum()):,}")
    
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(px.pie(agg, values='revenue_impact', names='domain', hole=0.4, title="Revenue by Business Unit"), use_container_width=True)
    with c2: st.plotly_chart(px.scatter(df_master, x='accuracy', y='usage', size='revenue_impact', color='domain', hover_name='name'), use_container_width=True)

# --- 4. ADMIN OPS ---
elif view == "Admin Ops":
    st.header("Fleet Governance & Telemetry")
    
    sel_mod = st.selectbox("Highlight Solo Asset (Clean View)", ["None"] + list(df_master['name'].unique()))
    
    plot_df = df_master.copy()
    # High Contrast Logic: Deep Red vs Pale Yellow
    if sel_mod == "None":
        color_vals = [0.5] * len(plot_df)
        c_scale = [[0, 'rgba(98, 0, 238, 0.2)'], [1, 'rgba(98, 0, 238, 0.2)']]
    else:
        color_vals = [1.0 if n == sel_mod else 0.0 for n in plot_df['name']]
        c_scale = [[0, 'rgba(255, 249, 196, 0.2)'], [1, '#B71C1C']]

    fig_p = go.Figure(data=go.Parcoords(
        labelfont=dict(size=12, color='black'), tickfont=dict(size=9, color='gray'),
        line=dict(color=color_vals, colorscale=c_scale, showscale=False),
        dimensions=list([
            dict(range=[0.7, 1.0], label='Accuracy', values=plot_df['accuracy']),
            dict(range=[0, 150], label='Latency', values=plot_df['latency']),
            dict(range=[0, 20000], label='Usage', values=plot_df['usage']),
            dict(range=[0, 0.3], label='Drift', values=plot_df['data_drift']),
            dict(range=[0, 100], label='CPU %', values=plot_df['cpu_util'])
        ])
    ))
    fig_p.update_layout(margin=dict(t=100, b=50, l=80, r=80), height=550)
    st.plotly_chart(fig_p, use_container_width=True)
    
    st.subheader("Audit Log & Metadata")
    st.dataframe(plot_df.drop(columns=['blob', 'relevance'], errors='ignore'), use_container_width=True)

# --- 5. APPROVALS ---
elif view == "Approvals":
    st.subheader("Leadership Approval Queue")
    st.info("Current queue status for Nat Patel (Leader)")
    st.write("Requests for Production promotion from Community Hub:")
    # Simulation logic for Nat Patel
    st.dataframe(df_master[df_master['type'] == 'Community'].head(5)[['name', 'contributor', 'domain']])
    if st.button("Bulk Approve Selected"): st.success("Approved and moved to Official Registry.")
