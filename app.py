import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import datetime
import random
import json
import re

# --- CONFIG & COMPACT STYLING ---
st.set_page_config(page_title="Enterprise Model Hub", layout="wide")

st.markdown("""
    <style>
    :root { --accent: #6200EE; --bg: #F4F7F9; --card: #FFFFFF; }
    .stApp { background-color: var(--bg); }
    
    /* Compact Model Card */
    .model-card {
        background: var(--card); border: 1px solid #e0e6ed; border-top: 3px solid var(--accent);
        padding: 12px; border-radius: 6px; min-height: 320px; 
        display: flex; flex-direction: column; justify-content: space-between;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05); transition: 0.2s;
    }
    .model-card:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
    
    .model-title { font-size: 0.95rem; font-weight: 700; color: #1a202c; margin-bottom: 2px; display: flex; justify-content: space-between; }
    .registry-badge { font-size: 0.6rem; padding: 2px 5px; border-radius: 4px; background: #EDF2F7; color: #4A5568; font-family: monospace; border: 1px solid #CBD5E0; }
    
    .meta-row { font-size: 0.72rem; color: #718096; margin: 2px 0; display: flex; justify-content: space-between; }
    .tag-purple { color: var(--accent); font-weight: 600; }
    
    .use-case-text { font-size: 0.75rem; color: #4A5568; margin: 8px 0; line-height: 1.3; height: 3.9em; overflow: hidden; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; }
    
    .metric-bar { display: flex; justify-content: space-between; background: #F8FAFC; padding: 6px; border-radius: 4px; border: 1px solid #E2E8F0; }
    .metric-val { font-size: 0.8rem; font-weight: 700; color: var(--accent); text-align: center; flex: 1; }
    .metric-label { font-size: 0.6rem; color: #A0AEC0; display: block; text-transform: uppercase; }

    /* Buttons */
    .stButton>button { background-color: #1A202C; color: white; border-radius: 4px; font-size: 0.75rem; height: 30px; width: 100%; border:none; }
    .stButton>button:hover { background-color: var(--accent); }
    </style>
    """, unsafe_allow_html=True)

# --- DATA LAYER & CONNECTORS ---
REG_PATH = "model_registry_v3.csv"
REQ_PATH = "requests_v3.csv"

def init_registry():
    if not os.path.exists(REG_PATH):
        st.error("Registry not found. Creating simulated enterprise data...")
        # (Generation logic similar to previous but with MLflow/Vertex/SageMaker IDs)
        return pd.DataFrame()
    df = pd.read_csv(REG_PATH)
    # Ensure Connector Fields
    for col in ['registry_provider', 'run_id', 'revenue_impact', 'risk_exposure']:
        if col not in df.columns:
            df['registry_provider'] = [random.choice(['MLflow', 'Vertex AI', 'SageMaker']) for _ in range(len(df))]
            df['run_id'] = [f"run_{random.randint(1000,9999)}" for _ in range(len(df))]
            df['revenue_impact'] = df['usage'] * random.uniform(5, 20)
            df['risk_exposure'] = (1 - df['accuracy']) * 100000
    return df

df_master = init_registry()

if 'compare_list' not in st.session_state:
    st.session_state.compare_list = []

# --- ANALYTICS ENGINES ---
def get_recommendations(query, df):
    if not query: return df
    df = df.fillna('N/A')
    # Hybrid search: NL + Metrics
    if "<" in query or ">" in query:
        # Simple regex parser for logic
        match = re.search(r'(\w+)\s*([<>]=?)\s*(\d+)', query.lower())
        if match:
            col, op, val = match.groups()
            val = float(val) / 100 if col == 'accuracy' and float(val) > 1 else float(val)
            if col in df.columns:
                if '>' in op: df = df[df[col] >= val]
                else: df = df[df[col] <= val]
    
    df['blob'] = df.astype(str).apply(' '.join, axis=1)
    vec = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
    mtx = vec.fit_transform(df['blob'].tolist() + [query])
    df['relevance'] = cosine_similarity(mtx[-1], mtx[:-1])[0]
    return df[df['relevance'] > 0.01].sort_values('relevance', ascending=False)

# --- UI NAVIGATION ---
with st.sidebar:
    st.title("Model Hub 2.0")
    view = st.radio("Discovery", ["Marketplace", "Compare Tool", "Portfolio ROI", "Approvals"])
    st.divider()
    user = st.selectbox("Identity", ["John Doe", "Jane Nu", "Sam King", "Nat Patel (Leader)", "Admin"])
    if st.session_state.compare_list:
        st.info(f"Comparator: {len(st.session_state.compare_list)} models")
        if st.button("Clear Comparison"): st.session_state.compare_list = []; st.rerun()

# --- MARKETPLACE VIEW ---
if view == "Marketplace":
    t1, t2, t3 = st.tabs(["🏛 Explore Gallery", "🔥 Smart Discovery", "🚀 New Ingestion"])
    
    with t1:
        q = st.text_input("💬 Search anything (e.g. 'Finance SageMaker' or 'accuracy > 95')", placeholder="Keywords, Teams, or Providers...")
        results = get_recommendations(q, df_master)
        
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
                                <div class="meta-row"><span class="tag-purple">{row['domain']}</span><span>{row['model_stage']}</span></div>
                                <div class="meta-row"><span>Team: {row['model_owner_team']}</span><span>{row['model_version']}</span></div>
                                <div class="use-case-text">{row['use_cases']}</div>
                            </div>
                            <div>
                                <div class="metric-bar">
                                    <div class="metric-val"><span class="metric-label">Acc</span>{int(row['accuracy']*100)}%</div>
                                    <div class="metric-val"><span class="metric-label">Lat</span>{row['latency']}ms</div>
                                    <div class="metric-val"><span class="metric-label">Drift</span>{row['data_drift']}</div>
                                </div>
                                <div style="height:8px;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        b1, b2, b3 = st.columns([1,1,1])
                        if b1.button("Compare", key=f"c_{i+j}"):
                            if row['name'] not in st.session_state.compare_list:
                                st.session_state.compare_list.append(row['name'])
                                st.toast("Added to comparison basket")
                        
                        with b2:
                            with st.popover("Tech Specs"):
                                st.subheader(f"Telemetry: {row['name']}")
                                c_l, c_r = st.columns(2)
                                c_l.metric("Inference ID", row['inference_endpoint_id'])
                                c_r.metric("Run ID", row['run_id'])
                                
                                # Lineage Sankey
                                st.write("**Lineage Path**")
                                fig_lin = go.Figure(go.Sankey(node=dict(pad=15, thickness=20, label=[row['training_data_source'], "Training Run", row['name'], "Production Endpoint"], color="purple"),
                                     link=dict(source=[0,1,2], target=[1,2,3], value=[1,1,1])))
                                fig_lin.update_layout(height=150, margin=dict(l=0,r=0,t=0,b=0))
                                st.plotly_chart(fig_lin, use_container_width=True)
                                
                                # Feature Importance
                                st.write("**Top Feature Contribution**")
                                feat_df = pd.DataFrame({'Feature': ['Behavioral','Historical','Temporal','Geo'], 'Weight': [0.4, 0.3, 0.2, 0.1]})
                                st.plotly_chart(px.bar(feat_df, x='Weight', y='Feature', orientation='h', height=150, color_discrete_sequence=['#6200EE']), use_container_width=True)
                                
                                st.button("Launch Jupyter Notebook", icon="🚀")

                        if b3.button("Request", key=f"r_{i+j}"):
                            st.toast("Approval request sent to Nat Patel")

    with t2:
        st.subheader("Smart Segments")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### 🔥 Trending This Week")
            st.dataframe(df_master.nlargest(5, 'usage')[['name', 'usage', 'accuracy']], use_container_width=True)
            
            st.markdown("### ⚠️ High Drift Risk")
            st.dataframe(df_master[df_master['data_drift'] > 0.08][['name', 'data_drift', 'monitoring_status']], use_container_width=True)
            
        with col_b:
            st.markdown("### 💎 Hidden Gems (Low Adoption / High Perf)")
            gems = df_master[(df_master['usage'] < 1000) & (df_master['accuracy'] > 0.95)]
            st.dataframe(gems[['name', 'accuracy', 'usage']], use_container_width=True)
            
            st.markdown("### 🕒 Recently Improved")
            st.dataframe(df_master.sort_values('last_retrained_date', ascending=False).head(5)[['name', 'last_retrained_date']], use_container_width=True)

# --- COMPARE TOOL ---
elif view == "Compare Tool":
    st.header("Side-by-Side Comparison")
    if not st.session_state.compare_list:
        st.info("Select models from the Gallery to compare them here.")
    else:
        comp_df = df_master[df_master['name'].isin(st.session_state.compare_list)]
        st.table(comp_df[['name', 'model_version', 'accuracy', 'latency', 'data_drift', 'cpu_util', 'sla_tier', 'registry_provider']])
        
        st.subheader("Visual Benchmark")
        fig_comp = px.bar(comp_df, x='name', y=['accuracy', 'data_drift'], barmode='group', title="Accuracy vs. Drift Correlation")
        st.plotly_chart(fig_comp, use_container_width=True)

# --- ROI DASHBOARD ---
elif view == "Portfolio ROI":
    st.header("Executive Strategy Dashboard")
    dom_agg = df_master.groupby('domain').agg({'revenue_impact': 'sum', 'risk_exposure': 'sum', 'accuracy': 'mean', 'usage': 'sum'}).reset_index()
    
    k1, k2, k3 = st.columns(3)
    k1.metric("Total Revenue Impact", f"${dom_agg['revenue_impact'].sum()/1e6:.2f}M")
    k2.metric("Total Risk Mitigated", f"${dom_agg['risk_exposure'].sum()/1e6:.2f}M")
    k3.metric("Fleet Adoption", f"{int(dom_agg['usage'].sum()):,}")
    
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.pie(dom_agg, values='revenue_impact', names='domain', title="Revenue Attribution", hole=0.4), use_container_width=True)
    with c2:
        st.plotly_chart(px.scatter(df_master, x='accuracy', y='usage', size='revenue_impact', color='domain', hover_name='name', title="Performance vs Adoption Strategy"), use_container_width=True)

# --- LEADER / ADMIN VIEWS ---
elif view == "Approvals":
    st.subheader("Leader Approval Queue (Nat Patel)")
    st.info("No pending requests for production migration.")
