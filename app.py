import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os, datetime, random, re, csv

# --- CONFIG & STYLING ---
st.set_page_config(page_title="Enterprise Model Marketplace", layout="wide")

st.markdown("""
    <style>
    :root { --accent: #6200EE; --lite-purple: #F3E5F5; --deep-purple: #7B1FA2; --pale-yellow: #FFF9C4; }
    .stApp { background-color: #F8FAFC; font-size: 0.82rem; }
    .model-card {
        background: white; border: 1px solid #e2e8f0; border-top: 3px solid var(--accent);
        padding: 10px; border-radius: 6px; min-height: 330px; 
        display: flex; flex-direction: column; justify-content: space-between;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .model-title { font-size: 0.85rem; font-weight: 700; color: #1e293b; margin-bottom: 2px; }
    .registry-badge { font-size: 0.55rem; padding: 1px 4px; border-radius: 3px; background: #f1f5f9; border: 1px solid #cbd5e1; color: #475569; font-family: monospace; }
    .use-case-text { font-size: 0.68rem; color: #334155; height: 2.8em; overflow: hidden; margin: 4px 0; line-height: 1.2; }
    .metric-bar { display: flex; justify-content: space-between; background: #F1F5F9; padding: 4px; border-radius: 4px; margin-bottom: 8px; }
    .metric-val { font-size: 0.72rem; font-weight: 700; color: var(--accent); text-align: center; flex: 1; }
    .metric-label { font-size: 0.5rem; color: #94a3b8; display: block; text-transform: uppercase; }
    .stButton>button { 
        background-color: var(--lite-purple); color: var(--deep-purple); 
        border: 1px solid var(--deep-purple); border-radius: 4px; 
        font-size: 0.62rem; height: 22px; padding: 0 4px; width: 100%;
    }
    .stButton>button:hover { background-color: var(--deep-purple); color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- CONSTANTS ---
REG_PATH = "model_registry_v3.csv"
SHAP_FEATURES = {
    "Finance": ["Credit_Score", "Annual_Income", "Debt_Ratio", "Trans_Freq", "Collateral"],
    "Healthcare": ["Age", "BMI", "Blood_Pressure", "Glucose", "Genetic_Score"],
    "Retail": ["Recency", "Frequency", "Monetary_Val", "Seasonality", "Customer_Age"],
    "HR": ["Tenure", "Engagement_Score", "Salary_Delta", "Education_Lvl", "Travel_Freq"],
    "Supply Chain": ["Lead_Time", "Inventory_Lvl", "Distance", "Supplier_Rating", "Demand_Vol"],
    "IT Ops": ["CPU_Load", "Latency_P99", "Disk_Usage", "Packet_Loss", "Memory_Pressure"],
    "Risk": ["Exposure_Index", "Counterparty_Score", "Market_Vol", "Liquidity_Gap", "Default_Prob"]
}

# --- SELF-HEALING DATA ENGINE ---
def load_and_sanitize_data():
    if not os.path.exists(REG_PATH): return pd.DataFrame()
    df = pd.read_csv(REG_PATH)
    
    # Define required columns and types
    schema = {
        'accuracy': 0.85, 'latency': 40, 'data_drift': 0.01, 'usage': 100,
        'revenue_impact': 5000, 'risk_exposure': 1000, 'cpu_util': 30, 'error_rate': 0.01,
        'approval_status': 'Approved', 'domain': 'Finance', 'contributor': 'System',
        'model_stage': 'Prod', 'registry_provider': 'MLflow', 'inference_endpoint_id': 'ep-001'
    }
    
    for col, default in schema.items():
        if col not in df.columns:
            df[col] = default
        if col in ['accuracy', 'latency', 'data_drift', 'usage', 'revenue_impact', 'risk_exposure', 'cpu_util']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default)
            
    return df

df_master = load_and_sanitize_data()

if 'basket' not in st.session_state: st.session_state.basket = []

# --- HYBRID SEARCH ENGINE ---
def hybrid_search(query, df):
    if not query: return df
    q = query.lower()
    df_res = df.copy()
    
    # 1. Math Filters (e.g. latency < 50)
    pats = {'latency': r'latency\s*([<>]=?)\s*(\d+)', 'accuracy': r'accuracy\s*([<>]=?)\s*(\d+)', 'drift': r'drift\s*([<>]=?)\s*(0\.\d+|\d+)', 'usage': r'usage\s*([<>]=?)\s*(\d+)'}
    for key, pat in pats.items():
        match = re.search(pat, q)
        if match:
            op, val = match.groups()
            val = float(val)/100 if (key=='accuracy' and float(val)>1) else float(val)
            col = 'data_drift' if key=='drift' else key
            if '>' in op: df_res = df_res[df_res[col] >= val]
            else: df_res = df_res[df_res[col] <= val]

    if df_res.empty: return df_res
    
    # 2. Textual Similarity
    df_res['blob'] = df_res.fillna('').astype(str).apply(' '.join, axis=1)
    vec = TfidfVectorizer(stop_words='english')
    mtx = vec.fit_transform(df_res['blob'].tolist() + [query])
    df_res['relevance'] = cosine_similarity(mtx[-1], mtx[:-1])[0]
    return df_res[df_res['relevance'] > 0.01].sort_values('relevance', ascending=False)

# --- SIDEBAR ---
with st.sidebar:
    st.title("Model Hub 3.0")
    view = st.radio("Navigation", ["Marketplace", "Compare Tool", "Insights & Trends", "Domain ROI", "My Portfolio", "Admin Ops"])
    role = st.selectbox("Current User", ["John Doe", "Jane Nu", "Sam King", "Nat Patel", "Admin"])
    if st.session_state.basket:
        st.success(f"Basket: {len(st.session_state.basket)} models")
        if st.button("Reset Basket"): st.session_state.basket = []; st.rerun()

# --- 1. MARKETPLACE ---
if view == "Marketplace":
    t1, t2 = st.tabs(["🏛 Model Gallery", "🚀 Ingest Asset"])
    with t1:
        query = st.text_input("💬 Search Keywords or Logic (e.g. 'Finance >90 accuracy latency < 50')", placeholder="Search...")
        results = hybrid_search(query, df_master)
        for i in range(0, min(len(results), 21), 3):
            cols = st.columns(3)
            for j in range(3):
                if i+j < len(results):
                    row = results.iloc[i+j]
                    with cols[j]:
                        st.markdown(f"""
                        <div class="model-card">
                            <div>
                                <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                                    <span class="registry-badge">{row['registry_provider']}</span>
                                    <span style="color:#2E7D32; font-size:0.6rem;">● Healthy</span>
                                </div>
                                <div class="model-title">{row['name'][:24]}</div>
                                <div class="meta-line"><b>{row['domain']}</b> | {row['model_stage']}</div>
                                <div class="meta-line">Team: {row.get('model_owner_team','Strategy AI')}</div>
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
                            if row['name'] not in st.session_state.basket: st.session_state.basket.append(row['name']); st.toast("Added")
                        with b2:
                            with st.popover("Specs"):
                                st.write(f"**SHAP Summary: {row['name']}**")
                                feats = SHAP_FEATURES.get(row['domain'], ["Feature_A","Feature_B","Feature_C","Feature_D","Feature_E"])
                                shap_df = pd.DataFrame({'Feature': feats, 'Impact': [random.uniform(-0.5, 0.5) for _ in range(5)]}).sort_values('Impact')
                                shap_df['Color'] = ['#EF4444' if x < 0 else '#6200EE' for x in shap_df['Impact']]
                                fig = px.bar(shap_df, x='Impact', y='Feature', orientation='h', color='Color', color_discrete_map="identity", height=180)
                                fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), showlegend=False)
                                st.plotly_chart(fig, use_container_width=True, key=f"sh_{i+j}")
                        if b3.button("Request", key=f"r_{i+j}"):
                            df_master.loc[df_master['name'] == row['name'], 'approval_status'] = 'Pending'
                            df_master.to_csv(REG_PATH, index=False); st.toast("Sent to Nat Patel")

    with t2:
        with st.form("ingest"):
            st.subheader("Register Asset")
            fn = st.text_input("Name*"); fd = st.selectbox("Domain", list(SHAP_FEATURES.keys())); fe = st.text_area("Description*")
            if st.form_submit_button("Publish"):
                new = pd.DataFrame([{"name":fn,"domain":fd,"use_cases":fe,"contributor":role,"accuracy":0.85,"latency":40,"usage":0,"model_stage":"Shadow","approval_status":"Pending"}])
                pd.concat([df_master, new]).to_csv(REG_PATH, index=False); st.success("Asset live!")

# --- 2. COMPARE TOOL ---
elif view == "Compare Tool":
    st.header("Side-by-Side Comparison")
    if not st.session_state.basket: st.info("Add models from Marketplace.")
    else:
        cdf = df_master[df_master['name'].isin(st.session_state.basket)]
        st.dataframe(cdf[['name','accuracy','data_drift','latency','usage','domain']], use_container_width=True)
        st.plotly_chart(px.bar(cdf, x='name', y=['accuracy','data_drift','latency','usage'], barmode='group', title="Comparison Matrix"))

# --- 3. INSIGHTS & TRENDS ---
elif view == "Insights & Trends":
    st.header("Strategic "Wow" Dashboard")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("🔥 Trending")
        st.table(df_master.nlargest(5, 'usage')[['name','usage']])
    with c2:
        st.subheader("💎 Hidden Gems")
        st.table(df_master[(df_master['accuracy'] > 0.96) & (df_master['usage'] < 1000)].head(5)[['name','accuracy']])
    with c3:
        st.subheader("⚠️ High Drift Action")
        st.table(df_master.nlargest(5, 'data_drift')[['name','data_drift']])

# --- 4. DOMAIN ROI ---
elif view == "Domain ROI":
    st.header("Strategic Domain Value")
    agg = df_master.groupby('domain').agg({'revenue_impact':'sum','risk_exposure':'sum','usage':'sum','accuracy':'mean'}).reset_index()
    k1, k2 = st.columns(2)
    k1.metric("Rev Contribution", f"${agg['revenue_impact'].sum()/1e6:.1f}M")
    k2.metric("Fleet Quality", f"{int(agg['accuracy'].mean()*100)}%")
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(px.pie(agg, values='revenue_impact', names='domain', hole=0.4))
    with c2: st.plotly_chart(px.bar(agg, x='domain', y='revenue_impact', color='accuracy'))

# --- 5. MY PORTFOLIO ---
elif view == "My Portfolio":
    my_m = df_master[df_master['contributor'] == role]
    if not my_m.empty:
        st.header(f"Contributor: {role}")
        sel = st.selectbox("Inspect Asset", my_m['name'].unique())
        m = my_m[my_m['name'] == sel].iloc[0]
        fig = go.Figure(go.Scatterpolar(r=[m['accuracy']*100, 100-(m['data_drift']*100), m.get('cpu_util',50), 100-m.get('error_rate',0)*10], theta=['Accuracy','Stability','Efficiency','Reliability'], fill='toself'))
        st.plotly_chart(fig, use_container_width=True)
    else: st.info("No assets found.")

# --- 6. ADMIN OPS / NAT PATEL ---
elif view == "Admin Ops":
    if role == "Nat Patel":
        st.header("Leader Approval Gateway")
        pend = df_master[df_master['approval_status'] == 'Pending']
        if not pend.empty:
            st.dataframe(pend[['name','domain','contributor','accuracy']])
            if st.button("Bulk Approve All"):
                df_master.loc[df_master['approval_status'] == 'Pending', 'approval_status'] = 'Approved'
                df_master.to_csv(REG_PATH, index=False); st.rerun()
        else: st.success("Queue clear.")
    else:
        st.header("Fleet Governance")
        sel = st.selectbox("Focus Asset", ["None"] + list(df_master['name'].unique()))
        cv = [1.0 if n == sel else 0.0 for n in df_master['name']]
        fig = go.Figure(data=go.Parcoords(
            labelfont=dict(size=10, color='black'), tickfont=dict(size=8),
            line=dict(color=cv, colorscale=[[0, '#FFF9C4'], [1, '#B71C1C']], showscale=False),
            dimensions=[dict(range=[0.7,1], label='Accuracy', values=df_master['accuracy']),
                        dict(range=[0,150], label='Latency', values=df_master['latency']),
                        dict(range=[0,25000], label='Usage', values=df_master['usage']),
                        dict(range=[0,0.3], label='Drift', values=df_master['data_drift'])]))
        fig.update_layout(margin=dict(t=80, b=40, l=80, r=80), height=500)
        st.plotly_chart(fig, use_container_width=True)
        if sel != "None": st.table(df_master[df_master['name'] == sel][['name','domain','usage','accuracy','latency','data_drift']])
