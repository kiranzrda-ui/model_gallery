import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import os, datetime, random, re, csv, json

# --- CONFIG & STYLING ---
st.set_page_config(page_title="Enterprise Model Hub 6.0", layout="wide")

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

# --- SCHEMA & CONSTANTS ---
MASTER_COLUMNS = [
    "name", "domain", "type", "accuracy", "latency", "clients", "use_cases", 
    "contributor", "usage", "data_drift", "revenue_impact", "risk_exposure", 
    "approval_status", "model_stage", "model_version", "model_owner_team",
    "cpu_util", "error_rate", "throughput", "training_data_source", "sla_tier"
]
REG_PATH = "model_registry_v3.csv"
REQ_PATH = "requests_v3.csv"

SHAP_FEATURES = {
    "Finance": ["Credit_Score", "Annual_Income", "Debt_Ratio", "Trans_Freq", "Collateral"],
    "Healthcare": ["Age", "BMI", "Blood_Pressure", "Glucose", "Genetic_Risk"],
    "Retail": ["Recency", "Frequency", "Monetary_Val", "Seasonality", "Customer_Age"],
    "HR": ["Tenure", "Engagement_Score", "Salary_Delta", "Education", "Travel_Freq"],
    "Supply Chain": ["Lead_Time", "Route_Risk", "Fuel_Index", "Carrier_Rating", "Inventory"],
    "IT Ops": ["CPU_Load", "P99_Latency", "Error_Count", "Throughput", "Uptime"],
    "Risk": ["Exposure_Index", "Volatility", "Compliance_Score", "Liquidity_Gap", "Default_Prob"]
}

# --- DATA ENGINE (Self-Healing Fix) ---
def load_sanitized_data():
    if not os.path.exists(REG_PATH): return pd.DataFrame(columns=MASTER_COLUMNS)
    df = pd.read_csv(REG_PATH)
    
    # 1. Fill missing columns
    for col in MASTER_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0 if col in ["accuracy", "revenue_impact", "usage", "risk_exposure"] else "N/A"
    
    # 2. Force numeric conversion
    nums = ["accuracy", "latency", "usage", "data_drift", "revenue_impact", "risk_exposure"]
    for c in nums:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce').fillna(0.0)
    
    # 3. IF DATA IS EMPTY OR ZEROS, GENERATE PROTOTYPE ROI VALUES
    if df['revenue_impact'].sum() == 0:
        # Business Logic: Revenue Impact = Usage * random factor, or accuracy-based value
        df['revenue_impact'] = df['usage'].apply(lambda x: x * random.uniform(50, 200) if x > 0 else random.uniform(1000, 5000))
    if df['risk_exposure'].sum() == 0:
        df['risk_exposure'] = df['accuracy'].apply(lambda x: (1-x) * 50000 if x > 0 else 5000)
    
    if 'approval_status' not in df.columns: df['approval_status'] = 'Approved'
    return df

df_master = load_sanitized_data()

# --- GEMINI INTEGRATION ---
def call_gemini(prompt, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        return model.generate_content(prompt).text
    except Exception as e: return f"AI Error: {str(e)}"

# --- SEARCH ENGINE ---
def hybrid_search(query, df):
    if not query: return df
    q = query.lower()
    df_res = df.copy().fillna('N/A')
    
    # Math Filters (e.g. latency < 30)
    pats = {'latency': r'latency\s*([<>]=?)\s*(\d+)', 'accuracy': r'accuracy\s*([<>]=?)\s*(\d+)', 'drift': r'drift\s*([<>]=?)\s*(0\.\d+|\d+)'}
    for key, pat in pats.items():
        match = re.search(pat, q)
        if match:
            op, val = match.groups()
            val = float(val)/100 if (key=='accuracy' and float(val)>1) else float(val)
            col = 'data_drift' if key=='drift' else key
            if '>' in op: df_res = df_res[df_res[col] >= val]
            else: df_res = df_res[df_res[col] <= val]

    if df_res.empty: return df_res
    df_res['blob'] = df_res.astype(str).apply(' '.join, axis=1)
    vec = TfidfVectorizer(stop_words='english')
    mtx = vec.fit_transform(df_res['blob'].tolist() + [query])
    df_res['relevance'] = cosine_similarity(mtx[-1], mtx[:-1])[0]
    return df_res[df_res['relevance'] > 0.01].sort_values('relevance', ascending=False)

# --- UI COMPONENTS ---
def render_tile(row, key_prefix):
    st.markdown(f"""
    <div class="model-card">
        <div>
            <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                <span class="registry-badge">v{row.get('model_version','1.0')}</span>
                <span style="color:#2E7D32; font-size:0.6rem;">● Healthy</span>
            </div>
            <div class="model-title">{row['name']}</div>
            <div class="meta-line"><b>{row['domain']}</b> | {row['contributor']}</div>
            <div class="use-case-text">{row['use_cases']}</div>
        </div>
        <div class="metric-bar">
            <div class="metric-val"><span class="metric-label">Acc</span>{int(row['accuracy']*100)}%</div>
            <div class="metric-val"><span class="metric-label">Lat</span>{int(row['latency'])}ms</div>
            <div class="metric-val"><span class="metric-label">Value</span>${row['revenue_impact']/1000:.1f}k</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Compare", key=f"c_{key_prefix}_{row['name']}"):
            if row['name'] not in st.session_state.basket:
                st.session_state.basket.append(row['name'])
                st.toast("Added to basket")
    with c2:
        with st.popover("Specs"):
            st.write(f"**SHAP Summary: {row['name']}**")
            # DYNAMIC FEATURES BASED ON DOMAIN
            feats = SHAP_FEATURES.get(row['domain'], ["F1","F2","F3","F4","F5"])
            shap_df = pd.DataFrame({
                'Feature': feats, 
                'Impact': [random.uniform(-0.5, 0.5) for _ in range(len(feats))]
            }).sort_values('Impact')
            shap_df['Color'] = ['#EF4444' if x < 0 else '#6200EE' for x in shap_df['Impact']]
            fig = px.bar(shap_df, x='Impact', y='Feature', orientation='h', color='Color', color_discrete_map="identity", height=180)
            fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), showlegend=False, xaxis_title="Feature Impact (SHAP)")
            st.plotly_chart(fig, use_container_width=True, key=f"sh_{key_prefix}_{row['name']}")
    with c3:
        if st.button("Request", key=f"r_{key_prefix}_{row['name']}"):
            df_master.loc[df_master['name'] == row['name'], 'approval_status'] = 'Pending'
            df_master.to_csv(REG_PATH, index=False)
            st.toast("Sent to Nat Patel")

# --- NAVIGATION ---
with st.sidebar:
    st.title("Hub Controls")
    api_key = st.text_input("Gemini API Key", type="password")
    nav = st.radio("Navigation", ["WorkBench Companion", "Model Gallery", "AI Business Value", "Approval"])
    st.divider()
    role = st.selectbox("Switch User", ["John Doe", "Jane Nu", "Sam King", "Nat Patel", "Admin"])
    st.session_state.role = role
    if 'basket' not in st.session_state: st.session_state.basket = []
    if st.session_state.basket:
        st.success(f"Basket: {len(st.session_state.basket)}")
        if st.button("Clear Basket"): st.session_state.basket = []; st.rerun()

# --- 1. WORKBENCH COMPANION ---
if nav == "WorkBench Companion":
    st.header("🤖 WorkBench Companion")
    if "messages" not in st.session_state: st.session_state.messages = []
    if "last_context" not in st.session_state: st.session_state.last_context = None

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "df" in msg:
                cols = st.columns(min(len(msg["df"]), 3))
                for i, r in enumerate(msg["df"].head(3).to_dict('records')):
                    with cols[i]: render_tile(r, f"chat_h_{i}")
            if "chart" in msg: st.plotly_chart(msg["chart"], use_container_width=True)

    if prompt := st.chat_input("E.g., 'Compare Cash App and Ris-Model-001'"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            q = prompt.lower()
            res_txt, res_df, res_chart = "", pd.DataFrame(), None

            # INTENT: SUBMISSION (LLM Entity Extraction)
            if any(x in q for x in ["submit", "register"]):
                raw_json = call_gemini(f"Extract JSON (name, domain, accuracy(0-1), latency(int)) from: {prompt}. Return ONLY raw JSON.", api_key)
                try:
                    data = json.loads(re.search(r"\{.*\}", raw_json, re.DOTALL).group())
                    new_row = {c: "N/A" for c in MASTER_COLUMNS}
                    new_row.update({"name": data.get('name','Asset'), "domain": data.get('domain','Finance'), "accuracy": data.get('accuracy',0.9), "latency": data.get('latency',30), "contributor": role, "approval_status": "Pending", "model_stage": "Prod", "usage": 10, "revenue_impact": 5000})
                    df_master = pd.concat([df_master, pd.DataFrame([new_row])], ignore_index=True)
                    df_master.to_csv(REG_PATH, index=False)
                    res_txt = f"✅ I've extracted the details for **{data.get('name')}** and submitted it to **Nat Patel**."
                except: res_txt = "Could not parse model details. Please provide Name, Domain, Accuracy, and Latency."

            # INTENT: TRENDS / GEMS
            elif "trending" in q or "popular" in q:
                res_txt = "Most active models this week:"
                res_df = df_master.nlargest(3, 'usage')
            elif "gems" in q or "hidden" in q:
                res_txt = "High performance assets with low adoption:"
                res_df = df_master[(df_master['accuracy'] > 0.96) & (df_master['usage'] < 1000)].head(3)

            # INTENT: ROI / REVENUE
            elif "revenue" in q or "impact" in q:
                dom = next((d for d in df_master['domain'].unique() if d.lower() in q), st.session_state.last_context)
                if dom:
                    st.session_state.last_context = dom
                    subset = df_master[df_master['domain'] == dom]
                    val = subset['revenue_impact'].sum()
                    res_txt = f"The **{dom}** domain has generated **${val/1e6:.2f}M** in impact."
                    if "top" in q or "contributor" in q:
                        res_df = subset.nlargest(3, 'revenue_impact')
                else: res_txt = "Which domain should I analyze (e.g., Finance, Risk)?"

            # INTENT: COMPARE
            elif "compare" in q:
                names = [n for n in df_master['name'] if n.lower() in q]
                if len(names) >= 2:
                    res_df = df_master[df_master['name'].isin(names)]
                    res_chart = px.bar(res_df, x='name', y=['accuracy','latency','data_drift','usage'], barmode='group', title="Side-by-Side Comparison")
                    res_txt = "Generated comparison chart."
                else: res_txt = "Please provide two exact model names."

            # DEFAULT: SEARCH
            else:
                res_txt = call_gemini(prompt, api_key) if api_key else "Search results:"
                res_df = hybrid_search(prompt, df_master)

            st.markdown(res_txt)
            if res_chart: st.plotly_chart(res_chart)
            if not res_df.empty and len(res_df) < len(df_master):
                cols = st.columns(min(len(res_df), 3))
                for i, r in enumerate(res_df.head(3).to_dict('records')):
                    with cols[i]: render_tile(r, f"chat_{i}")
            
            st.session_state.messages.append({"role": "assistant", "content": res_txt, "df": res_df.head(3), "chart": res_chart})

# --- 2. MODEL GALLERY ---
elif nav == "Model Gallery":
    q_gal = st.text_input("💬 Hybrid Search (e.g. 'latency < 50' or 'Risk Prod')")
    res = hybrid_search(q_gal, df_master)
    for i in range(0, min(len(res), 15), 3):
        cols = st.columns(3)
        for j in range(3):
            if i+j < len(res):
                with cols[j]: render_tile(res.iloc[i+j], f"gal_{i+j}")

# --- 3. AI BUSINESS VALUE ---
elif nav == "AI Business Value":
    st.header("Executive ROI Dashboard")
    # ENFORCED AGGREGATION
    agg = df_master.groupby('domain').agg({'revenue_impact':'sum','usage':'sum','accuracy':'mean'}).reset_index()
    k1, k2 = st.columns(2)
    k1.metric("Total Rev Impact", f"${agg['revenue_impact'].sum()/1e6:.1f}M")
    k2.metric("Portfolio Quality", f"{int(agg['accuracy'].mean()*100)}%")
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(px.pie(agg, values='revenue_impact', names='domain', hole=0.4, title="Revenue Share"))
    with c2: st.plotly_chart(px.bar(agg, x='domain', y='revenue_impact', color='accuracy', title="ROI vs Performance"))

# --- 4. APPROVAL / ADMIN ---
elif nav == "Approval":
    if role == "Nat Patel" or role == "Admin":
        st.subheader("Leader Approval Queue")
        pend = df_master[df_master['approval_status'] == 'Pending']
        if not pend.empty:
            st.table(pend[['name','domain','contributor','accuracy']])
            if st.button("Approve All"):
                df_master.loc[df_master['approval_status']=='Pending','approval_status']='Approved'
                df_master.to_csv(REG_PATH, index=False); st.rerun()
        else: st.success("Queue Clear.")
        
        st.divider()
        st.subheader("Fleet Governance (High-Contrast)")
        sel = st.selectbox("Highlight Line", ["None"] + list(df_master['name'].unique()))
        cv = [1.0 if n == sel else 0.0 for n in df_master['name']]
        fig_p = go.Figure(data=go.Parcoords(
            labelfont=dict(size=10, color='black'), tickfont=dict(size=8),
            line=dict(color=cv, colorscale=[[0, '#FFF9C4'], [1, '#B71C1C']], showscale=False),
            dimensions=[dict(range=[0.7,1], label='Accuracy', values=df_master['accuracy']),
                        dict(range=[0,150], label='Latency', values=df_master['latency']),
                        dict(range=[0,25000], label='Usage', values=df_master['usage']),
                        dict(range=[0,0.3], label='Drift', values=df_master['data_drift'])]))
        fig_p.update_layout(margin=dict(t=80, b=40, l=80, r=80), height=550)
        st.plotly_chart(fig_p, use_container_width=True)
    else: st.warning("Restricted Access.")
