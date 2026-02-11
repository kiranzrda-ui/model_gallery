import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os, datetime, random, re, csv

# --- CONFIG & STYLING ---
st.set_page_config(page_title="Enterprise Model Hub", layout="wide")

st.markdown("""
    <style>
    :root { --accent: #6200EE; --lite-purple: #F3E5F5; --deep-purple: #7B1FA2; }
    .stApp { background-color: #F8FAFC; font-size: 0.82rem; }
    
    /* Compact Card UI */
    .model-card {
        background: white; border: 1px solid #e2e8f0; border-top: 3px solid var(--accent);
        padding: 10px; border-radius: 6px; min-height: 320px; 
        display: flex; flex-direction: column; justify-content: space-between;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .model-title { font-size: 0.85rem; font-weight: 700; color: #1e293b; margin-bottom: 2px; }
    .tag-row { display: flex; justify-content: space-between; margin-bottom: 4px; }
    .registry-badge { font-size: 0.55rem; padding: 1px 4px; border-radius: 3px; background: #f1f5f9; border: 1px solid #cbd5e1; color: #475569; font-family: monospace; }
    .meta-line { font-size: 0.65rem; color: #64748b; margin: 0; }
    .use-case-text { font-size: 0.68rem; color: #334155; height: 2.8em; overflow: hidden; margin: 4px 0; line-height: 1.2; }
    
    .metric-bar { display: flex; justify-content: space-between; background: #F1F5F9; padding: 4px; border-radius: 4px; margin-bottom: 8px; }
    .metric-val { font-size: 0.72rem; font-weight: 700; color: var(--accent); text-align: center; flex: 1; }
    .metric-label { font-size: 0.5rem; color: #94a3b8; display: block; text-transform: uppercase; }

    /* Small Lite Purple Buttons */
    .stButton>button { 
        background-color: var(--lite-purple); color: var(--deep-purple); 
        border: 1px solid var(--deep-purple); border-radius: 4px; 
        font-size: 0.62rem; height: 22px; padding: 0 4px; width: 100%;
    }
    .stButton>button:hover { background-color: var(--deep-purple); color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- DOMAIN SHAP MAPS ---
SHAP_FEATURES = {
    "Finance": ["Credit_Score", "Annual_Income", "Debt_Ratio", "Trans_Freq", "Collateral"],
    "Healthcare": ["Age", "BMI", "Blood_Pressure", "Glucose", "Genetic_Score"],
    "Retail": ["Recency", "Frequency", "Monetary_Val", "Seasonality", "Customer_Age"],
    "HR": ["Tenure", "Engagement_Score", "Salary_Delta", "Education_Lvl", "Travel_Freq"],
    "Supply Chain": ["Lead_Time", "Inventory_Lvl", "Distance", "Supplier_Rating", "Demand_Vol"],
    "IT Ops": ["CPU_Load", "Latency_P99", "Disk_Usage", "Packet_Loss", "Memory_Pressure"]
}

# --- DATA LAYER ---
REG_PATH = "model_registry_v3.csv"
REQ_PATH = "requests_v3.csv"

def get_data():
    if not os.path.exists(REG_PATH): return pd.DataFrame()
    df = pd.read_csv(REG_PATH)
    # Patch numerical columns for ROI math
    nums = ['usage', 'accuracy', 'latency', 'data_drift', 'revenue_impact', 'risk_exposure', 'cpu_util', 'error_rate', 'throughput']
    for c in nums:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    return df

df_master = get_data()

# --- STATE ---
if 'basket' not in st.session_state: st.session_state.basket = []

# --- SEARCH ENGINE (Regex + Text) ---
def advanced_query(query, df):
    if not query: return df
    q = query.lower()
    work_df = df.copy().fillna('N/A')
    
    # 1. Math Filter
    pats = {'latency': r'latency\s*([<>]=?)\s*(\d+)', 'accuracy': r'accuracy\s*([<>]=?)\s*(\d+)', 'usage': r'usage\s*([<>]=?)\s*(\d+)'}
    for col, pat in pats.items():
        match = re.search(pat, q)
        if match:
            op, val = match.groups()
            val = float(val)/100 if col=='accuracy' and float(val)>1 else float(val)
            if '>' in op: work_df = work_df[work_df[col] >= val]
            else: work_df = work_df[work_df[col] <= val]

    if work_df.empty: return work_df

    # 2. Semantic Search
    work_df['blob'] = work_df.astype(str).apply(' '.join, axis=1)
    vec = TfidfVectorizer(stop_words='english')
    mtx = vec.fit_transform(work_df['blob'].tolist() + [query])
    work_df['relevance'] = cosine_similarity(mtx[-1], mtx[:-1])[0]
    return work_df[work_df['relevance'] > 0.01].sort_values('relevance', ascending=False)

# --- NAVIGATION ---
with st.sidebar:
    st.title("Model Marketplace")
    view = st.radio("Navigate", ["Gallery Hub", "Compare Tool", "Domain ROI", "My Portfolio", "Admin Ops"])
    st.divider()
    role = st.selectbox("Switch User", ["John Doe", "Jane Nu", "Sam King", "Nat Patel", "Admin"])
    if st.session_state.basket:
        st.info(f"Basket: {len(st.session_state.basket)} items")
        if st.button("Clear Basket"): st.session_state.basket = []; st.rerun()

# --- VIEW: GALLERY HUB ---
if view == "Gallery Hub":
    t1, t2, t3 = st.tabs(["🏛 Marketplace", "🚀 Ingest Asset", "💡 Approvals"])
    
    with t1:
        query = st.text_input("💬 Hybrid Search (e.g. 'Finance Prod >90 accuracy')", placeholder="Keywords or Logic...")
        results = advanced_query(query, df_master)
        
        for i in range(0, min(len(results), 21), 3):
            cols = st.columns(3)
            for j in range(3):
                if i+j < len(results):
                    row = results.iloc[i+j]
                    with cols[j]:
                        st.markdown(f"""
                        <div class="model-card">
                            <div>
                                <div class="tag-row">
                                    <span class="registry-badge">{row.get('registry_provider','Vertex AI')}</span>
                                    <span style="font-size:0.6rem; color:#2E7D32;">● Healthy</span>
                                </div>
                                <div class="model-title">{row['name'][:24]}</div>
                                <div class="meta-line"><b>{row['domain']}</b> | {row['model_stage']}</div>
                                <div class="meta-line">Team: {row['model_owner_team']} | v{row['model_version']}</div>
                                <div class="use-case-text">{row['use_cases']}</div>
                            </div>
                            <div class="metric-bar">
                                <div class="metric-val"><span class="metric-label">Acc</span>{int(row['accuracy']*100)}%</div>
                                <div class="metric-val"><span class="metric-label">Lat</span>{int(row['latency'])}ms</div>
                                <div class="metric-val"><span class="metric-label">Drift</span>{row['data_drift']}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        b1, b2, b3 = st.columns(3)
                        if b1.button("Compare", key=f"c_{i+j}"):
                            st.session_state.basket.append(row['name']); st.toast("Added")
                        with b2:
                            with st.popover("Specs"):
                                st.write(f"**SHAP Summary: {row['name']}**")
                                feats = SHAP_FEATURES.get(row['domain'], ["F1","F2","F3","F4","F5"])
                                shap_df = pd.DataFrame({'Feature': feats, 'Impact': [random.uniform(-0.5, 0.5) for _ in range(5)]}).sort_values('Impact')
                                shap_df['Color'] = ['#EF4444' if x < 0 else '#6200EE' for x in shap_df['Impact']]
                                fig_shap = px.bar(shap_df, x='Impact', y='Feature', orientation='h', color='Color', color_discrete_map="identity", height=180)
                                fig_shap.update_layout(margin=dict(l=0,r=0,t=0,b=0), showlegend=False)
                                st.plotly_chart(fig_shap, use_container_width=True, key=f"sh_{i+j}")
                                st.caption("Inference ID: " + str(row.get('inference_endpoint_id','N/A')))
                        if b3.button("Request", key=f"r_{i+j}"): st.toast("Sent to Leader")

    with t2:
        with st.form("ingest"):
            st.subheader("Ingest to Marketplace")
            fn = st.text_input("Asset Name*")
            fd = st.selectbox("Business Domain", list(SHAP_FEATURES.keys()))
            fe = st.text_area("Description (Lineage/Stage/Use Case)*")
            if st.form_submit_button("Publish Model"):
                if fn and fe:
                    new = pd.DataFrame([{"name":fn,"domain":fd,"use_cases":fe,"contributor":role,"accuracy":0.85,"latency":40,"usage":0,"model_stage":"Shadow","model_owner_team":"New Team","registry_provider":"MLflow","revenue_impact":0,"risk_exposure":0}])
                    pd.concat([df_master, new]).to_csv(REG_PATH, index=False)
                    st.success("Asset live!")

    with t3:
        st.subheader("Nat Patel's Approval Queue")
        st.dataframe(df_master[df_master['model_stage'] == "Shadow"].head(5)[['name','domain','contributor']])

# --- VIEW: COMPARE TOOL ---
elif view == "Compare Tool":
    st.header("Side-by-Side Comparison")
    if not st.session_state.basket: st.info("Basket empty.")
    else:
        cdf = df_master[df_master['name'].isin(st.session_state.basket)]
        st.dataframe(cdf[['name','accuracy','latency','data_drift','sla_tier','model_stage']])
        fig_bench = px.bar(cdf, x='name', y='accuracy', color='domain', title="Accuracy Benchmark")
        st.plotly_chart(fig_bench, use_container_width=True)

# --- VIEW: DOMAIN ROI ---
elif view == "Domain ROI":
    st.header("Strategic Domain Insights")
    agg = df_master.groupby('domain').agg({'revenue_impact':'sum','risk_exposure':'sum','usage':'sum','accuracy':'mean'}).reset_index()
    k1, k2, k3 = st.columns(3)
    k1.metric("Revenue Contribution", f"${agg['revenue_impact'].sum()/1e6:.1f}M")
    k2.metric("Avg Quality", f"{int(agg['accuracy'].mean()*100)}%")
    k3.metric("Total Adoption", f"{int(agg['usage'].sum()):,}")
    
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(px.pie(agg, values='revenue_impact', names='domain', hole=0.4, title="Revenue Attribution"))
    with c2: st.plotly_chart(px.bar(agg, x='domain', y='revenue_impact', title="ROI by Business Unit"))

# --- VIEW: MY PORTFOLIO ---
elif view == "My Portfolio":
    my_m = df_master[df_master['contributor'] == role]
    if not my_m.empty:
        st.header(f"Portfolio: {role}")
        sel = st.selectbox("Inspect Model", my_m['name'].unique())
        m = my_m[my_m['name'] == sel].iloc[0]
        fig_r = go.Figure(go.Scatterpolar(r=[m['accuracy']*100, 100-(m['data_drift']*100), m['cpu_util'], 100-m['error_rate']*10], theta=['Accuracy','Stability','Efficiency','Reliability'], fill='toself'))
        st.plotly_chart(fig_r, use_container_width=True)
    else: st.info("No assets found.")

# --- VIEW: ADMIN OPS ---
elif view == "Admin Ops":
    st.header("Fleet Telemetry")
    sel = st.selectbox("Focus Solo Line", ["None"] + list(df_master['name'].unique()))
    cv = [1.0 if n == sel else 0.0 for n in df_master['name']]
    fig_p = go.Figure(data=go.Parcoords(
        labelfont=dict(size=10, color='black'), tickfont=dict(size=8),
        line=dict(color=cv, colorscale=[[0,'rgba(98,0,238,0.1)'], [1,'#B71C1C']], showscale=False),
        dimensions=[dict(range=[0.7,1], label='Accuracy', values=df_master['accuracy']),
                    dict(range=[0,150], label='Latency', values=df_master['latency']),
                    dict(range=[0,25000], label='Usage', values=df_master['usage']),
                    dict(range=[0,0.3], label='Drift', values=df_master['data_drift'])]))
    fig_p.update_layout(margin=dict(t=80, b=40, l=80, r=80), height=500)
    st.plotly_chart(fig_p, use_container_width=True)
