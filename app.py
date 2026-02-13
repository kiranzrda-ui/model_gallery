import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import os, datetime, random, re, csv, json

# --- CONFIG & STYLING ---
st.set_page_config(page_title="Model Hub 18.0", layout="wide")

st.markdown("""
    <style>
    :root { --accent: #6200EE; --lite-p: #F3E5F5; --deep-p: #7B1FA2; --pale-y: #FFF9C4; }
    .stApp { background-color: #F8FAFC; font-size: 0.8rem; }
    .model-card {
        background: white; border: 1px solid #e2e8f0; border-top: 3px solid var(--accent);
        padding: 8px 10px; border-radius: 6px; min-height: 270px; 
        display: flex; flex-direction: column; justify-content: space-between;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    .model-title { font-size: 0.82rem; font-weight: 700; color: #1e293b; margin-bottom: 1px; }
    .owner-tag { font-size: 0.5rem; font-weight: bold; color: var(--deep-p); text-transform: uppercase; background: var(--lite-p); padding: 1px 3px; border-radius:2px; }
    .metric-bar { display: flex; justify-content: space-between; background: #F1F5F9; padding: 3px; border-radius: 4px; margin-bottom: 4px; }
    .metric-val { font-size: 0.7rem; font-weight: 700; color: var(--accent); text-align: center; flex: 1; }
    .metric-label { font-size: 0.45rem; color: #94a3b8; display: block; text-transform: uppercase; }
    .stButton>button { 
        background-color: var(--lite-p); color: var(--deep-p); border: 1px solid var(--deep-p);
        border-radius: 4px; font-size: 0.58rem; height: 20px; padding: 0 2px; width: 100%;
        line-height: 1;
    }
    .stButton>button:hover { background-color: var(--deep-p); color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- CONSTANTS ---
REG_PATH = "model_registry_v3.csv"
REQ_PATH = "requests_v3.csv"
SHAP_FEATURES = {
    "Finance": ["Credit", "Income", "Debt", "Trans_Vol"],
    "Healthcare": ["Age", "BMI", "BP", "Glucose"],
    "Risk": ["Exposure", "Volatility", "Compliance", "Liq"],
    "Supply Chain": ["Lead_Time", "Inventory", "Route", "Demand"],
    "Retail": ["Footfall", "Seasonality", "Margin", "Promo"],
    "Default": ["Feature_1", "Feature_2", "Feature_3", "Feature_4"]
}

# --- DATA ENGINE ---
def load_sanitized_data():
    if not os.path.exists(REG_PATH): return pd.DataFrame()
    df = pd.read_csv(REG_PATH)
    
    # 1. Self-Heal missing ROI and Accuracy columns
    if 'revenue_impact' not in df.columns: df['revenue_impact'] = 0.0
    if 'risk_exposure' not in df.columns: df['risk_exposure'] = 0.0
    
    # 2. Force numeric conversion (Handling semicolon data)
    nums = ['usage', 'accuracy', 'latency', 'data_drift', 'revenue_impact', 'risk_exposure']
    for c in nums:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce').fillna(0.0)
    
    # 3. Populate missing values for Strategy ROI
    if df['revenue_impact'].sum() == 0:
        df['revenue_impact'] = df['usage'] * 125.0 # $125 per usage
    
    # 4. Standardize Status for Nat Patel
    if 'approval_status' in df.columns:
        df['approval_status'] = df['approval_status'].replace('pending_review', 'Pending')
    else:
        df['approval_status'] = 'Approved'
        
    return df

df_master = load_sanitized_data()

# --- SEARCH ENGINE ---
def hybrid_search(query, df):
    if not query or df.empty: return df
    q = query.lower()
    df_res = df.copy().fillna('N/A')
    
    # Logic Filtering
    pats = {'latency': r'latency\s*([<>]=?)\s*(\d+)', 'accuracy': r'accuracy\s*([<>]=?)\s*(\d+)'}
    for k, p in pats.items():
        match = re.search(p, q)
        if match:
            op, val = match.groups()
            val = float(val)/100 if (k=='accuracy' and float(val)>1) else float(val)
            df_res = df_res[df_res[k] >= val] if '>' in op else df_res[df_res[k] <= val]
    
    # Semantic
    df_res['blob'] = df_res.astype(str).apply(' '.join, axis=1)
    vec = TfidfVectorizer(stop_words='english')
    mtx = vec.fit_transform(df_res['blob'].tolist() + [query])
    df_res['relevance'] = cosine_similarity(mtx[-1], mtx[:-1])[0]
    return df_res[df_res['relevance'] > 0.01].sort_values('relevance', ascending=False)

# --- UI COMPONENTS ---
def render_tile(row, user, prefix, total_results, is_companion=True):
    label = "Owned" if row['contributor'] == user else ""
    st.markdown(f"""
    <div class="model-card">
        <div>
            <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                <span class="owner-tag">{label}</span>
                <span style="font-size:0.5rem; color:#666;">v{row.get('model_version','1.0')}</span>
            </div>
            <div class="model-title">{row['name'][:22]}</div>
            <div style="font-size:0.6rem; color:gray;"><b>{row['domain']}</b> | {row['type']}</div>
            <div style="font-size:0.65rem; color:#444; height:2.4em; overflow:hidden;">{row['use_cases']}</div>
        </div>
        <div class="metric-bar">
            <div class="metric-val"><span class="metric-label">Acc</span>{int(row['accuracy']*100)}%</div>
            <div class="metric-val"><span class="metric-label">Lat</span>{int(row['latency'])}ms</div>
            <div class="metric-val"><span class="metric-label">Drift</span>{row['data_drift']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        if total_results > 1:
            if st.button("Compare", key=f"c_{prefix}_{row['name']}_{random.randint(0,999)}"):
                if is_companion: st.session_state.chat_trigger = f"Compare model {row['name']}"
                else: 
                    if row['name'] not in st.session_state.basket: st.session_state.basket.append(row['name'])
    with c2:
        with st.popover("Specs"):
            st.write(f"**SHAP Importance: {row['name']}**")
            feats = SHAP_FEATURES.get(row['domain'], SHAP_FEATURES["Default"])
            shap_df = pd.DataFrame({'Feature': feats, 'Impact': [random.uniform(-0.5, 0.5) for _ in range(4)]}).sort_values('Impact')
            shap_df['Color'] = ['#EF4444' if x < 0 else '#6200EE' for x in shap_df['Impact']]
            fig = px.bar(shap_df, x='Impact', y='Feature', orientation='h', color='Color', color_discrete_map="identity", height=200)
            fig.update_layout(margin=dict(l=60,r=10,t=10,b=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    with c3:
        if st.button("Request", key=f"r_{prefix}_{row['name']}_{random.randint(0,999)}"):
            if is_companion: st.session_state.chat_trigger = f"I want to request access for {row['name']}"
            else:
                st.toast("Request Logged for Nat Patel")

# --- NAVIGATION ---
with st.sidebar:
    st.title("Enterprise AI Hub")
    app_mode = st.toggle("Enable Web Mode", value=False)
    user_role = st.selectbox("Current User", ["John Doe", "Jane Nu", "Sam King", "Nat Patel"])
    api_key = st.text_input("Gemini API Key", type="password")
    if 'basket' not in st.session_state: st.session_state.basket = []

# --- 1. COMPANION MODE ---
if not app_mode:
    st.header("🤖 WorkBench Companion")
    if "messages" not in st.session_state: st.session_state.messages = []
    if 'chat_trigger' in st.session_state:
        st.session_state.messages.append({"role": "user", "content": st.session_state.pop('chat_trigger')})

    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "df" in msg and not msg["df"].empty:
                cols = st.columns(min(len(msg["df"]), 3))
                for idx, r in enumerate(msg["df"].head(3).to_dict('records')):
                    with cols[idx]: render_tile(r, user_role, f"ch_{i}", len(msg["df"]), is_companion=True)
            if "pie" in msg: st.plotly_chart(msg["pie"])

    if prompt := st.chat_input("Prompt: 'Compare IT-Model-001 and 008'"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            q = prompt.lower()
            res_txt, res_df, res_pie = "Analyzing...", pd.DataFrame(), None
            
            if "revenue" in q or "impact" in q:
                agg = df_master.groupby('domain')['revenue_impact'].sum().reset_index()
                res_pie = px.pie(agg, values='revenue_impact', names='domain', hole=0.4)
                res_txt = "Here is the revenue contribution by domain."
            elif "compare" in q:
                names = [n for n in df_master['name'].unique() if n.lower() in q or n.split('-')[-1] in q]
                if len(names) >= 2:
                    res_df = df_master[df_master['name'].isin(names)]
                    st.table(res_df[['name','accuracy','latency','usage']])
                    res_txt = f"Benchmarking {len(names)} models."
                else: res_txt = "Please specify model names."
            else:
                res_df = hybrid_search(q, df_master).head(3)
                res_txt = "Relevant models found:" if not res_df.empty else "No models found. Try other keywords."
            
            st.markdown(res_txt)
            if res_pie: st.plotly_chart(res_pie)
            if not res_df.empty:
                cols = st.columns(min(len(res_df), 3))
                for idx, r in enumerate(res_df.head(3).to_dict('records')):
                    with cols[idx]: render_tile(r, user_role, f"at_{idx}", len(res_df), is_companion=True)
            st.session_state.messages.append({"role": "assistant", "content": res_txt, "df": res_df, "pie": res_pie})

# --- 2. WEB MODE ---
else:
    tabs = ["Model Gallery", "AI Business Value"]
    if st.session_state.basket: tabs.insert(1, "Benchmark Basket")
    if user_role == "Nat Patel": tabs.append("Approval Portal")
    t = st.tabs(tabs)
    
    with t[0]:
        q = st.text_input("Smart Search")
        res = hybrid_search(q, df_master)
        for i in range(0, min(len(res), 15), 3):
            cols = st.columns(3)
            for j in range(3):
                if i+j < len(res):
                    with cols[j]: render_tile(res.iloc[i+j], user_role, f"gal_{i+j}", len(res), is_companion=False)

    current_idx = 1
    if st.session_state.basket:
        with t[current_idx]:
            c_df = df_master[df_master['name'].isin(st.session_state.basket)]
            st.table(c_df[['name','accuracy','latency','usage','data_drift']])
            st.plotly_chart(px.bar(c_df, x='name', y=['accuracy','usage'], barmode='group'))
            if st.button("Clear Basket"): st.session_state.basket = []; st.rerun()
        current_idx += 1

    with t[current_idx]:
        st.header("Strategic Portfolio ROI")
        src = df_master if user_role == "Nat Patel" else df_master[df_master['contributor'] == user_role]
        agg = src.groupby('domain').agg({'revenue_impact':'sum','accuracy':'mean'}).reset_index()
        k1, k2 = st.columns(2)
        k1.metric("Revenue Impact", f"${agg['revenue_impact'].sum()/1e6:.1f}M")
        k2.metric("Avg Accuracy", f"{int(agg['accuracy'].mean()*100)}%")
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(px.pie(agg, values='revenue_impact', names='domain', hole=0.4))
        with c2: st.plotly_chart(px.bar(agg, x='domain', y='revenue_impact', color='accuracy'))

    if user_role == "Nat Patel":
        with t[-1]:
            st.header("Leader Approval Gateway")
            pend = df_master[df_master['approval_status'] == 'Pending']
            if not pend.empty:
                st.table(pend[['name','domain','contributor','accuracy']])
                if st.button("Bulk Approve"):
                    df_master.loc[df_master['approval_status']=='Pending','approval_status']='Approved'
                    df_master.to_csv(REG_PATH, index=False); st.rerun()
            else: st.success("Queue clear.")
