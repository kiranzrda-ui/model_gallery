import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import os, datetime, random, re, csv, json

# --- CONFIG & STYLING ---
st.set_page_config(page_title="Model Hub 6.0", layout="wide")

st.markdown("""
    <style>
    :root { --accent: #6200EE; --lite-purple: #F3E5F5; --deep-purple: #7B1FA2; --pale-yellow: #FFF9C4; }
    .stApp { background-color: #F8FAFC; font-size: 0.82rem; }
    
    /* Compact Card UI */
    .model-card {
        background: white; border: 1px solid #e2e8f0; border-top: 3px solid var(--accent);
        padding: 10px; border-radius: 6px; min-height: 310px; 
        display: flex; flex-direction: column; justify-content: space-between;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .model-title { font-size: 0.85rem; font-weight: 700; color: #1e293b; margin-bottom: 2px; }
    .metric-bar { display: flex; justify-content: space-between; background: #F1F5F9; padding: 4px; border-radius: 4px; margin-bottom: 5px; }
    .metric-val { font-size: 0.72rem; font-weight: 700; color: var(--accent); text-align: center; flex: 1; }
    .metric-label { font-size: 0.5rem; color: #94a3b8; display: block; text-transform: uppercase; }
    
    .stButton>button { 
        background-color: var(--lite-purple); color: var(--deep-purple); 
        border: 1px solid var(--deep-purple); border-radius: 4px; 
        font-size: 0.62rem; height: 22px; width: 100%;
    }
    .stButton>button:hover { background-color: var(--deep-purple); color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- SCHEMA & DATA ---
MASTER_COLUMNS = [
    "name", "domain", "type", "accuracy", "latency", "clients", "use_cases", 
    "contributor", "usage", "data_drift", "revenue_impact", "risk_exposure", 
    "approval_status", "model_stage", "model_version", "model_owner_team"
]
REG_PATH = "model_registry_v3.csv"
SHAP_MAP = {
    "Finance": ["Credit", "Income", "Debt", "History"],
    "Risk": ["Exposure", "Vol", "Comp", "Liq"],
    "Healthcare": ["Age", "BMI", "BP", "Glucose"],
    "Default": ["F1", "F2", "F3", "F4"]
}

def load_data():
    if not os.path.exists(REG_PATH): return pd.DataFrame(columns=MASTER_COLUMNS)
    df = pd.read_csv(REG_PATH)
    for col in MASTER_COLUMNS:
        if col not in df.columns: df[col] = 0.0 if col in ["accuracy","usage","revenue_impact"] else "N/A"
    nums = ["accuracy", "latency", "usage", "data_drift", "revenue_impact", "risk_exposure"]
    for c in nums:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce').fillna(0.0)
    return df

df_master = load_data()

# --- GEMINI INTEGRATION ---
def call_gemini(prompt, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        return model.generate_content(prompt).text
    except Exception as e: return f"AI Logic: {str(e)}"

# --- SEARCH & SIMILARITY ---
def hybrid_search(query, df):
    if not query or df.empty: return df
    df_res = df.copy().fillna('N/A')
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
                <span style="font-size:0.55rem; color:#666;">v{row.get('model_version','1.0')}</span>
                <span style="color:#2E7D32; font-size:0.6rem;">● {row['model_stage']}</span>
            </div>
            <div class="model-title">{row['name']}</div>
            <div style="font-size:0.65rem; color:gray;"><b>{row['domain']}</b> | {row['contributor']}</div>
            <div style="font-size:0.68rem; color:#444; height:3em; overflow:hidden; margin:5px 0;">{row['use_cases']}</div>
        </div>
        <div class="metric-bar">
            <div class="metric-val"><span class="metric-label">Acc</span>{int(row['accuracy']*100)}%</div>
            <div class="metric-val"><span class="metric-label">Lat</span>{int(row['latency'])}ms</div>
            <div class="metric-val"><span class="metric-label">Drift</span>{row['data_drift']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    if c1.button("Compare", key=f"c_{key_prefix}_{row['name']}"):
        st.session_state.basket.append(row['name']); st.toast("Added to Basket")
    with c2:
        with st.popover("Specs"):
            st.write(f"**Specs: {row['name']}**")
            feats = SHAP_MAP.get(row['domain'], SHAP_MAP["Default"])
            shap_df = pd.DataFrame({'Feature': feats, 'Impact': [random.uniform(-0.5, 0.5) for _ in range(4)]})
            st.plotly_chart(px.bar(shap_df, x='Impact', y='Feature', orientation='h', height=150), use_container_width=True, key=f"sh_{key_prefix}_{row['name']}")
    if c3.button("Access", key=f"r_{key_prefix}_{row['name']}"):
        st.toast("Request Sent")

# --- NAVIGATION ---
with st.sidebar:
    st.title("Enterprise AI")
    api_key = st.text_input("Gemini API Key", type="password")
    nav = st.radio("Navigation", ["WorkBench Companion", "Model Gallery", "AI Business Value", "Approval"])
    st.divider()
    role = st.selectbox("Role", ["John Doe", "Jane Nu", "Sam King", "Nat Patel", "Admin"])
    if 'basket' not in st.session_state: st.session_state.basket = []

# --- 1. WORKBENCH COMPANION (CONVERSATIONAL INSIGHTS) ---
if nav == "WorkBench Companion":
    st.header("🤖 WorkBench Companion")
    if "messages" not in st.session_state: st.session_state.messages = []
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "df" in msg:
                cols = st.columns(min(len(msg["df"]), 3))
                for i, r in enumerate(msg["df"].head(3).to_dict('records')):
                    with cols[i]: render_tile(r, f"chat_h_{i}")

    if prompt := st.chat_input("Ask about trends, gems, or specific model specs..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            q = prompt.lower()
            res_df = pd.DataFrame()
            # CONVERSATIONAL INTENTS
            if "trending" in q or "popular" in q:
                res_txt = "Here are the most adopted models this week:"
                res_df = df_master.nlargest(3, 'usage')
            elif "gems" in q or "hidden" in q:
                res_txt = "Found these low-adoption, high-performance models:"
                res_df = df_master[(df_master['accuracy'] > 0.96) & (df_master['usage'] < 1000)].head(3)
            elif "drift" in q or "risk" in q:
                res_txt = "The following models have high data drift and need attention:"
                res_df = df_master.nlargest(3, 'data_drift')
            elif "spec" in q or "detail" in q or "tell me about" in q:
                res_txt = "Fetching specific model telemetry..."
                res_df = hybrid_search(q, df_master).head(1)
            else:
                res_txt = call_gemini(prompt, api_key) if api_key else "I found these relevant models:"
                res_df = hybrid_search(prompt, df_master)
            
            st.markdown(res_txt)
            if not res_df.empty and len(res_df) < len(df_master):
                cols = st.columns(min(len(res_df), 3))
                for i, r in enumerate(res_df.head(3).to_dict('records')):
                    with cols[i]: render_tile(r, f"chat_new_{i}")
                st.session_state.messages.append({"role": "assistant", "content": res_txt, "df": res_df.head(3)})
            else:
                st.session_state.messages.append({"role": "assistant", "content": res_txt})

# --- 2. MODEL GALLERY (SPECS & COMPARE) ---
elif nav == "Model Gallery":
    t1, t2 = st.tabs(["🏛 Unified Gallery", "⚖️ Comparison Basket"])
    with t1:
        q_gal = st.text_input("Search Models")
        res_gal = hybrid_search(q_gal, df_master)
        for i in range(0, min(len(res_gal), 15), 3):
            cols = st.columns(3)
            for j in range(3):
                if i+j < len(res_gal):
                    with cols[j]: render_tile(res_gal.iloc[i+j], f"gal_{i+j}")
    with t2:
        if not st.session_state.basket: st.info("Basket is empty. Add models from the Gallery.")
        else:
            comp_df = df_master[df_master['name'].isin(st.session_state.basket)]
            st.dataframe(comp_df[['name','accuracy','latency','data_drift','usage']])
            st.plotly_chart(px.bar(comp_df, x='name', y=['accuracy','latency','data_drift'], barmode='group'))

# --- 3. AI BUSINESS VALUE (ROI) ---
elif nav == "AI Business Value":
    st.header("Strategic ROI Dashboard")
    agg = df_master.groupby('domain').agg({'revenue_impact':'sum','usage':'sum','accuracy':'mean'}).reset_index()
    k1, k2 = st.columns(2)
    k1.metric("Total Rev Impact", f"${agg['revenue_impact'].sum()/1e6:.1f}M")
    k2.metric("Fleet Adoption", f"{int(agg['usage'].sum()):,}")
    st.plotly_chart(px.bar(agg, x='domain', y='revenue_impact', color='accuracy', title="ROI by Domain"))

# --- 4. APPROVAL (NAT PATEL) ---
elif nav == "Approval":
    if role == "Nat Patel" or role == "Admin":
        st.subheader("Leader Approval Queue")
        pend = df_master[df_master['approval_status'] == 'Pending']
        if not pend.empty:
            st.table(pend[['name','domain','contributor']])
            if st.button("Bulk Approve"):
                df_master.loc[df_master['approval_status']=='Pending','approval_status']='Approved'
                df_master.to_csv(REG_PATH, index=False); st.rerun()
        else: st.success("Queue clear.")
    else: st.warning("Access Restricted to Nat Patel.")
