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

# --- CONFIG & COMPACT STYLING ---
st.set_page_config(page_title="Model Hub 2.0", layout="wide")

st.markdown("""
    <style>
    :root { --accent: #6200EE; --lite-purple: #F3E5F5; --deep-purple: #7B1FA2; }
    .stApp { background-color: #F8FAFC; font-size: 0.85rem; }
    
    /* Compact Card UI */
    .model-card {
        background: white; border: 1px solid #e2e8f0; border-top: 3px solid var(--accent);
        padding: 10px; border-radius: 6px; min-height: 300px; 
        display: flex; flex-direction: column; justify-content: space-between;
    }
    .model-title { font-size: 0.9rem; font-weight: 700; color: #1a202c; margin-bottom: 2px; }
    .meta-line { font-size: 0.7rem; color: #64748b; margin: 1px 0; }
    .use-case-text { font-size: 0.7rem; color: #475569; height: 3em; overflow: hidden; margin: 5px 0; }
    
    .metric-bar { display: flex; justify-content: space-between; background: #F1F5F9; padding: 4px; border-radius: 4px; }
    .metric-val { font-size: 0.75rem; font-weight: 700; color: var(--accent); text-align: center; flex: 1; }
    .metric-label { font-size: 0.5rem; color: #94a3b8; display: block; text-transform: uppercase; }

    /* Small Lite Purple Buttons */
    .stButton>button { 
        background-color: var(--lite-purple); color: var(--deep-purple); 
        border: 1px solid var(--deep-purple); border-radius: 4px; 
        font-size: 0.65rem; height: 24px; padding: 0 5px; 
    }
    .stButton>button:hover { background-color: var(--deep-purple); color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- DATA PERSISTENCE ---
REG_PATH = "model_registry_v3.csv"

def load_data():
    if not os.path.exists(REG_PATH):
        st.error("Registry CSV missing.")
        return pd.DataFrame()
    df = pd.read_csv(REG_PATH)
    # Ensure numeric columns for ROI math
    numeric_cols = ['usage', 'accuracy', 'latency', 'data_drift', 'revenue_impact', 'risk_exposure']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            df[col] = 0.0
    return df

df_master = load_data()

# --- HYBRID SEARCH ENGINE (Math + Text) ---
def run_intelligent_search(query, df):
    if not query: return df
    df = df.copy().fillna('')
    q = query.lower()
    
    # 1. Math Parser (e.g., latency < 30, drift < 0.5)
    patterns = {
        'latency': r'latency\s*([<>]=?)\s*(\d+)',
        'accuracy': r'accuracy\s*([<>]=?)\s*(\d+)',
        'usage': r'usage\s*([<>]=?)\s*(\d+)',
        'data_drift': r'drift\s*([<>]=?)\s*(0\.\d+|\d+)'
    }
    for col, pattern in patterns.items():
        match = re.search(pattern, q)
        if match:
            op, val = match.groups()
            val = float(val) / 100 if col == 'accuracy' and float(val) > 1 else float(val)
            if op == '<': df = df[df[col] < val]
            elif op == '>': df = df[df[col] > val]
            elif op == '<=': df = df[df[col] <= val]
            elif op == '>=': df = df[df[col] >= val]

    if df.empty: return df

    # 2. Text Search
    df['blob'] = df.astype(str).apply(' '.join, axis=1)
    vec = TfidfVectorizer(stop_words='english')
    mtx = vec.fit_transform(df['blob'].tolist() + [query])
    df['relevance'] = cosine_similarity(mtx[-1], mtx[:-1])[0]
    return df[df['relevance'] > 0.01].sort_values('relevance', ascending=False)

# --- UI NAVIGATION ---
with st.sidebar:
    st.title("Model Hub")
    view = st.radio("Navigation", ["Marketplace", "Portfolio ROI", "Admin Ops"])
    user = st.selectbox("Switch User", ["John Doe", "Jane Nu", "Sam King", "Nat Patel", "Admin"])
    if 'compare_list' not in st.session_state: st.session_state.compare_list = []

# --- 1. MARKETPLACE ---
if view == "Marketplace":
    q = st.text_input("💬 Search (e.g. 'Finance latency < 50')", placeholder="Search...")
    results = run_intelligent_search(q, df_master)
    
    for i in range(0, min(len(results), 21), 3):
        cols = st.columns(3)
        for j in range(3):
            if i+j < len(results):
                row = results.iloc[i+j]
                with cols[j]:
                    st.markdown(f"""
                    <div class="model-card">
                        <div>
                            <div class="model-title">{row['name'][:25]}</div>
                            <div class="meta-line"><b>{row['domain']}</b> | {row['model_stage']}</div>
                            <div class="meta-line">Team: {row['model_owner_team']}</div>
                            <div class="use-case-text">{row['use_cases']}</div>
                        </div>
                        <div class="metric-bar">
                            <div class="metric-val"><span class="metric-label">Acc</span>{int(row['accuracy']*100)}%</div>
                            <div class="metric-val"><span class="metric-label">Lat</span>{int(row['latency'])}ms</div>
                            <div class="metric-val"><span class="metric-label">Drift</span>{row['data_drift']}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    c1, c2, c3 = st.columns(3)
                    if c1.button("Compare", key=f"c_{i+j}"):
                        st.session_state.compare_list.append(row['name'])
                        st.toast("Added")
                    
                    with c2:
                        with st.popover("Specs"):
                            st.write(f"**SHAP Feature Importance: {row['name']}**")
                            # Simulated SHAP Summary Plot
                            shap_data = pd.DataFrame({
                                'Feature': ['Age', 'Income', 'Credit_Score', 'Location', 'Last_Purchase'],
                                'Impact': [0.45, -0.32, 0.28, 0.15, -0.05],
                                'Color': ['#6200EE', '#EF4444', '#6200EE', '#6200EE', '#EF4444']
                            }).sort_values('Impact')
                            fig_shap = px.bar(shap_data, x='Impact', y='Feature', orientation='h', 
                                            color='Color', color_discrete_map="identity")
                            fig_shap.update_layout(height=200, margin=dict(l=0,r=0,t=0,b=0), showlegend=False)
                            st.plotly_chart(fig_shap, use_container_width=True, key=f"shap_{i+j}")
                            st.caption("Positive impact (Purple) vs Negative impact (Red)")
                    
                    if c3.button("Request", key=f"r_{i+j}"): st.toast("Sent")

# --- 2. STRATEGY ROI ---
elif view == "Portfolio ROI":
    st.header("Domain Strategy Dashboard")
    agg = df_master.groupby('domain').agg({
        'revenue_impact': 'sum', 
        'risk_exposure': 'sum', 
        'accuracy': 'mean', 
        'usage': 'sum'
    }).reset_index()
    
    if not agg.empty:
        k1, k2, k3 = st.columns(3)
        k1.metric("Rev Contribution", f"${agg['revenue_impact'].sum()/1e6:.1f}M")
        k2.metric("Risk Exposure", f"${agg['risk_exposure'].sum()/1e6:.1f}M")
        k3.metric("Fleet Adoption", f"{int(agg['usage'].sum()):,}")
        
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(px.pie(agg, values='revenue_impact', names='domain', hole=0.4, title="Revenue by Unit"))
        with c2: st.plotly_chart(px.bar(agg, x='domain', y='accuracy', title="Avg Accuracy by Domain"))
    else:
        st.warning("No data available for ROI calculation.")

# --- 3. ADMIN OPS ---
elif view == "Admin Ops":
    st.header("Fleet Governance")
    sel = st.selectbox("Focus Asset", ["None"] + list(df_master['name'].unique()))
    
    # Solo Line Logic
    cv = [1.0 if n == sel else 0.0 for n in df_master['name']]
    cs = [[0, 'rgba(255, 249, 196, 0.2)'], [1, '#B71C1C']]

    fig_p = go.Figure(data=go.Parcoords(
        labelfont=dict(size=10, color='black'), tickfont=dict(size=8),
        line=dict(color=cv, colorscale=cs, showscale=False),
        dimensions=list([
            dict(range=[0.7, 1.0], label='Accuracy', values=df_master['accuracy']),
            dict(range=[0, 150], label='Latency', values=df_master['latency']),
            dict(range=[0, 20000], label='Usage', values=df_master['usage']),
            dict(range=[0, 0.3], label='Drift', values=df_master['data_drift'])
        ])
    ))
    fig_p.update_layout(margin=dict(t=80, b=40, l=80, r=80), height=500)
    st.plotly_chart(fig_p, use_container_width=True)
