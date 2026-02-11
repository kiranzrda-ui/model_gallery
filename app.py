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

# --- CONFIG & STYLING ---
st.set_page_config(page_title="Enterprise AI Strategy Hub", layout="wide")

st.markdown("""
    <style>
    :root { --accent: #6200EE; --success: #2E7D32; --warning: #F9A825; --critical: #C62828; --gold: #FFD700; }
    .model-card {
        border: 1px solid #e0e0e0; border-top: 4px solid var(--accent);
        padding: 15px; background-color: #ffffff; margin-bottom: 20px;
        min-height: 420px; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        display: flex; flex-direction: column; justify-content: space-between;
    }
    .ribbon-header { font-size: 1.4rem; font-weight: 700; color: #333; margin: 20px 0 10px 0; border-left: 5px solid var(--accent); padding-left: 15px; }
    .kpi-card { background: #f8f9fa; padding: 20px; border-radius: 10px; border: 1px solid #eee; text-align: center; }
    .kpi-val { font-size: 1.8rem; font-weight: 800; color: var(--accent); display: block; }
    .kpi-label { font-size: 0.8rem; color: #666; text-transform: uppercase; }
    
    /* Similarity Styling */
    .similar-box { background: #f3e5f5; padding: 10px; border-radius: 5px; margin-top: 10px; font-size: 0.8rem; border: 1px solid #ce93d8; }
    
    .status-pill { font-size: 0.65rem; padding: 2px 8px; border-radius: 10px; color: white; font-weight: bold; float: right; }
    .Healthy { background-color: var(--success); }
    .Warning { background-color: var(--warning); }
    .Critical { background-color: var(--critical); }
    
    .metric-grid { display: flex; justify-content: space-between; background: #fafafa; padding: 8px; border-radius: 4px; border: 1px solid #eee; }
    .metric-num { font-size: 0.85rem; font-weight: 700; color: var(--accent); }
    </style>
    """, unsafe_allow_html=True)

# --- DATA LAYER ---
REG_PATH = "model_registry_v3.csv"
REQ_PATH = "requests_v3.csv"

def load_data():
    if not os.path.exists(REG_PATH):
        st.error("Please ensure model_registry_v3.csv is in your Git repo.")
        return pd.DataFrame()
    df = pd.read_csv(REG_PATH)
    # Inject synthetic ROI metrics if missing for the Strategy Dashboard
    if 'revenue_saved' not in df.columns:
        df['revenue_saved'] = df['usage'] * random.uniform(10, 50)
        df['risk_reduction_score'] = df['accuracy'] * 100
        df['uptime'] = [random.uniform(98.5, 99.99) for _ in range(len(df))]
    return df

df_master = load_data()

# --- SIMILARITY ENGINE ---
def get_similar_models(model_name, df, top_n=3):
    try:
        df_sim = df.copy().fillna('')
        df_sim['blob'] = df_sim['use_cases'] + " " + df_sim['domain']
        vec = TfidfVectorizer(stop_words='english')
        mtx = vec.fit_transform(df_sim['blob'])
        idx = df_sim[df_sim['name'] == model_name].index[0]
        scores = cosine_similarity(mtx[idx], mtx).flatten()
        df_sim['sim_score'] = scores
        return df_sim[df_sim['name'] != model_name].sort_values('sim_score', ascending=False).head(top_n)
    except:
        return pd.DataFrame()

# --- SEARCH ENGINE ---
def smart_search(query, df):
    if not query: return df
    q = query.lower()
    # Apply hard filters based on query keywords
    if "low latency" in q: df = df[df['latency'] < 40]
    if "high accuracy" in q: df = df[df['accuracy'] > 0.96]
    if "drift" in q and "high" in q: df = df[df['data_drift'] > 0.1]
    
    df['blob'] = df.fillna('').astype(str).apply(' '.join, axis=1)
    vec = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
    mtx = vec.fit_transform(df['blob'].tolist() + [query])
    scores = cosine_similarity(mtx[-1], mtx[:-1])[0]
    df['relevance'] = scores
    return df[df['relevance'] > 0.01].sort_values('relevance', ascending=False)

# --- NAVIGATION ---
with st.sidebar:
    st.title("Enterprise Hub")
    view_mode = st.radio("Switch Dashboard", ["📦 Marketplace Hub", "📈 Strategy & ROI Dashboard"])
    st.divider()
    user = st.selectbox("Role", ["John Doe", "Jane Nu", "Sam King", "Nat Patel (Leader)", "Admin"])

# --- VIEW 1: MARKETPLACE HUB (Data Scientist Focus) ---
if view_mode == "📦 Marketplace Hub":
    t1, t2, t3 = st.tabs(["🏛 Unified Gallery", "🚀 Ingest", "📊 My Impact"])

    with t1:
        query = st.text_input("💬 Search 1,000+ Models...", placeholder="e.g. 'high performance low adoption' or 'shadow finance'")
        
        if not query:
            # DYNAMIC DISCOVERY SECTIONS
            st.markdown('<div class="ribbon-header">🔥 Trending This Week</div>', unsafe_allow_html=True)
            trending = df_master.sort_values('usage', ascending=False).head(3)
            cols = st.columns(3)
            for idx, row in enumerate(trending.to_dict('records')):
                with cols[idx]:
                    st.markdown(f"""<div class="model-card">
                        <div>
                            <span class="status-pill Healthy">Trending</span>
                            <div style="font-weight:700;">{row['name']}</div>
                            <div style="font-size:0.8rem; color:#666;">{row['use_cases']}</div>
                        </div>
                        <div class="metric-grid">
                            <div class="metric-item"><span class="metric-num">{row['usage']}</span>Views</div>
                            <div class="metric-item"><span class="metric-num">{int(row['accuracy']*100)}%</span>Acc</div>
                        </div>
                    </div>""", unsafe_allow_html=True)

            st.markdown('<div class="ribbon-header">💎 Hidden Gems (Low Adoption / High Perf)</div>', unsafe_allow_html=True)
            gems = df_master[(df_master['usage'] < 1000) & (df_master['accuracy'] > 0.97)].head(3)
            cols_gems = st.columns(3)
            for idx, row in enumerate(gems.to_dict('records')):
                if idx < 3:
                    with cols_gems[idx]:
                        st.markdown(f'<div class="model-card"><b>{row["name"]}</b><br><small>{row["domain"]}</small><br>Acc: {row["accuracy"]}</div>', unsafe_allow_html=True)

            st.markdown('<div class="ribbon-header">⚠️ High Drift Risk (Action Required)</div>', unsafe_allow_html=True)
            drifting = df_master[df_master['data_drift'] > 0.12].head(3)
            cols_drift = st.columns(3)
            for idx, row in enumerate(drifting.to_dict('records')):
                if idx < 3:
                    with cols_drift[idx]:
                        st.error(f"**{row['name']}**\n\nDrift: {row['data_drift']}")
        
        else:
            # SEARCH RESULTS
            results = smart_search(query, df_master)
            st.write(f"Results: {len(results)}")
            for i in range(0, min(len(results), 21), 3):
                cols = st.columns(3)
                for j in range(3):
                    if i+j < len(results):
                        row = results.iloc[i+j]
                        with cols[j]:
                            st.markdown(f"""<div class="model-card">
                                <div>
                                    <span class="status-pill {row['monitoring_status']}">{row['monitoring_status']}</span>
                                    <div style="font-weight:700;">{row['name']}</div>
                                    <div style="font-size:0.75rem; color:gray;">{row['model_owner_team']} | {row['model_stage']}</div>
                                    <div class="use-case-text">{row['use_cases']}</div>
                                </div>
                                <div class="metric-grid">
                                    <div class="metric-item"><span class="metric-num">{int(row['accuracy']*100)}%</span>Acc</div>
                                    <div class="metric-item"><span class="metric-num">{row['latency']}ms</span>Lat</div>
                                    <div class="metric-item"><span class="metric-num">{row['usage']}</span>Use</div>
                                </div>
                            </div>""", unsafe_allow_html=True)
                            with st.popover("More & Similar"):
                                st.write(f"**Description:** {row['description']}")
                                st.write(f"**SLA:** {row['sla_tier']}")
                                st.markdown('<div class="similar-box"><b>Similar Models:</b>', unsafe_allow_html=True)
                                sim_mods = get_similar_models(row['name'], df_master)
                                for s_name in sim_mods['name']:
                                    st.write(f"🔗 {s_name}")
                                st.markdown('</div>', unsafe_allow_html=True)
                                if st.button("Request Access", key=f"q_{i+j}"):
                                    st.toast("Request Logged")

# --- VIEW 2: STRATEGY & ROI (Executive Focus) ---
else:
    st.title("Strategic Model Portfolio Insights")
    
    # Global KPI Row
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.markdown(f'<div class="kpi-card"><span class="kpi-val">${df_master["revenue_saved"].sum()/1e6:.1f}M</span><span class="kpi-label">Revenue Saved</span></div>', unsafe_allow_html=True)
    with k2: st.markdown(f'<div class="kpi-card"><span class="kpi-val">{df_master["uptime"].mean():.2f}%</span><span class="kpi-label">Avg Uptime</span></div>', unsafe_allow_html=True)
    with k3: st.markdown(f'<div class="kpi-card"><span class="kpi-val">{len(df_master[df_master["monitoring_status"]=="Healthy"])}</span><span class="kpi-label">Healthy Assets</span></div>', unsafe_allow_html=True)
    with k4: st.markdown(f'<div class="kpi-card"><span class="kpi-val">82%</span><span class="kpi-label">SLA Adherence</span></div>', unsafe_allow_html=True)

    st.divider()
    
    # Domain Level Breakdown
    domain_summary = df_master.groupby('domain').agg({
        'usage': 'sum',
        'revenue_saved': 'sum',
        'accuracy': 'mean',
        'uptime': 'mean'
    }).reset_index()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("ROI Attribution by Domain")
        fig_roi = px.bar(domain_summary, x='domain', y='revenue_saved', color='revenue_saved', color_continuous_scale='Purples')
        st.plotly_chart(fig_roi, use_container_width=True)
    
    with c2:
        st.subheader("Adoption vs Performance")
        fig_adopt = px.scatter(df_master, x='accuracy', y='usage', color='domain', size='revenue_saved', hover_name='name')
        st.plotly_chart(fig_adopt, use_container_width=True)

    st.subheader("Domain Governance Maturity")
    st.table(domain_summary.style.format({'revenue_saved': '${:,.0f}', 'accuracy': '{:.2%}', 'uptime': '{:.2f}%'}))
