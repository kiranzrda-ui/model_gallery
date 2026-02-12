import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import os, datetime, random, re, csv

# --- CONFIG & STYLING ---
st.set_page_config(page_title="Enterprise Model Hub 4.0", layout="wide")

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
REQ_PATH = "requests_v3.csv"
SHAP_FEATURES = {
    "Finance": ["Credit_Score", "Income", "Debt_Ratio", "Trans_Count", "History"],
    "Healthcare": ["Age", "BMI", "Blood_Pressure", "Glucose", "Genetic_Risk"],
    "Retail": ["Recency", "Frequency", "Monetary", "Location", "App_Usage"],
    "Supply Chain": ["Lead_Time", "Route_Risk", "Fuel_Index", "Carrier_Rating", "Inventory"],
    "IT Ops": ["CPU_Load", "Memory_Free", "P99_Latency", "Error_Count", "Uptime"],
    "Risk": ["Exposure", "Volatility", "Compliance_Score", "Liquidity", "Geopolitical"]
}

# --- DATA ENGINE ---
def load_sanitized_data():
    if not os.path.exists(REG_PATH): return pd.DataFrame()
    df = pd.read_csv(REG_PATH)
    nums = ['usage', 'accuracy', 'latency', 'data_drift', 'revenue_impact', 'risk_exposure', 'cpu_util', 'error_rate', 'throughput']
    for c in nums:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce').fillna(0)
    if 'approval_status' not in df.columns: df['approval_status'] = 'Approved'
    return df

df_master = load_sanitized_data()

# --- SEARCH ENGINE ---
def hybrid_search(query, df):
    if not query: return df
    q = query.lower()
    work_df = df.copy().fillna('N/A')
    pats = {'latency': r'latency\s*([<>]=?)\s*(\d+)', 'accuracy': r'accuracy\s*([<>]=?)\s*(\d+)', 'drift': r'drift\s*([<>]=?)\s*(0\.\d+|\d+)'}
    for key, pat in pats.items():
        match = re.search(pat, q)
        if match:
            op, val = match.groups()
            val = float(val)/100 if (key=='accuracy' and float(val)>1) else float(val)
            col = 'data_drift' if key=='drift' else key
            if '>' in op: work_df = work_df[work_df[col] >= val]
            else: work_df = work_df[work_df[col] <= val]
    if work_df.empty: return work_df
    work_df['blob'] = work_df.astype(str).apply(' '.join, axis=1)
    vec = TfidfVectorizer(stop_words='english')
    mtx = vec.fit_transform(work_df['blob'].tolist() + [query])
    work_df['relevance'] = cosine_similarity(mtx[-1], mtx[:-1])[0]
    return work_df[work_df['relevance'] > 0.01].sort_values('relevance', ascending=False)

# --- GEMINI AI (ROBUST MODEL VERSIONING) ---
def get_ai_response(prompt, api_key):
    try:
        genai.configure(api_key=api_key)
        # Using a safer model identifier
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        context = f"Enterprise Model Hub Assistant. DB size: {len(df_master)}. User: {st.session_state.get('role','User')}."
        return model.generate_content(context + "\nUser: " + prompt).text
    except Exception as e:
        return f"AI Logic Active (Offline Mode). Found models based on: {prompt}"

# --- UI COMPONENTS ---
def render_tile(row, key_prefix):
    st.markdown(f"""
    <div class="model-card">
        <div>
            <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                <span class="registry-badge">{row.get('registry_provider','Vertex AI')}</span>
                <span style="color:#2E7D32; font-size:0.6rem;">● Healthy</span>
            </div>
            <div class="model-title">{row['name'][:24]}</div>
            <div class="meta-line"><b>{row['domain']}</b> | {row['model_stage']}</div>
            <div class="use-case-text">{row['use_cases']}</div>
        </div>
        <div class="metric-bar">
            <div class="metric-val"><span class="metric-label">Acc</span>{int(row['accuracy']*100)}%</div>
            <div class="metric-val"><span class="metric-label">Lat</span>{int(row['latency'])}ms</div>
            <div class="metric-val"><span class="metric-label">Drift</span>{row['data_drift']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Request", key=f"btn_{key_prefix}_{row['name']}"):
        st.toast("Request Sent")

# --- NAVIGATION ---
with st.sidebar:
    st.title("Enterprise AI")
    api_key = st.text_input("Gemini API Key", type="password")
    nav = st.radio("Navigate", ["AI Copilot", "Model Gallery", "Domain ROI", "My Portfolio", "Admin Ops"])
    role = st.selectbox("Current User", ["John Doe", "Jane Nu", "Sam King", "Nat Patel", "Admin"])
    st.session_state.role = role

# --- 1. AI COPILOT (WITH INTENT DETECTION) ---
if nav == "AI Copilot":
    st.header("🤖 Intelligent Assistant")
    if "messages" not in st.session_state: st.session_state.messages = []
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "df" in msg and not msg["df"].empty:
                cols = st.columns(min(len(msg["df"]), 3))
                for idx, r in enumerate(msg["df"].head(3).to_dict('records')):
                    with cols[idx]: render_tile(r, f"chat_hist_{idx}")

    if prompt := st.chat_input("E.g., 'Submit model Cash App with 99% accuracy'"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        with st.chat_message("assistant"):
            # A. Intent Detection for Submission
            if "submit" in prompt.lower() or "register" in prompt.lower():
                # Regex Extraction for John Doe's request
                name_match = re.search(r"model name is ([\w\s]+),", prompt, re.I) or re.search(r"submit ([\w\s]+) with", prompt, re.I)
                acc_match = re.search(r"(\d+)%", prompt)
                lat_match = re.search(r"(\d+)ms", prompt)
                
                m_name = name_match.group(1).strip() if name_match else "New Asset"
                m_acc = float(acc_match.group(1))/100 if acc_match else 0.85
                m_lat = int(lat_match.group(1)) if lat_match else 50
                
                new_row = pd.DataFrame([{
                    "name": m_name, "domain": "Finance" if "finance" in prompt.lower() else "General",
                    "use_cases": prompt, "contributor": role, "accuracy": m_acc, "latency": m_lat,
                    "usage": 0, "model_stage": "Prod", "approval_status": "Pending", "data_drift": 0.0
                }])
                df_master = pd.concat([df_master, new_row], ignore_index=True)
                df_master.to_csv(REG_PATH, index=False)
                
                response_text = f"✅ **Request Processed!** I have registered the model **{m_name}** and submitted it to **Nat Patel** for approval. It is currently in 'Pending' status."
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            
            # B. Standard Search Intent
            else:
                response_text = get_ai_response(prompt, api_key)
                st.markdown(response_text)
                res = hybrid_search(prompt, df_master)
                if not res.empty and len(res) < len(df_master):
                    cols = st.columns(min(len(res), 3))
                    for idx, r in enumerate(res.head(3).to_dict('records')):
                        with cols[idx]: render_tile(r, f"chat_new_{idx}")
                    st.session_state.messages.append({"role": "assistant", "content": response_text, "df": res.head(3)})
                else:
                    st.session_state.messages.append({"role": "assistant", "content": response_text})

# --- OTHER VIEWS (ROI, PORTFOLIO, ADMIN) ---
elif nav == "Model Gallery":
    q = st.text_input("Smart Search")
    res = hybrid_search(q, df_master)
    for i in range(0, min(len(res), 15), 3):
        cols = st.columns(3)
        for j in range(3):
            if i+j < len(res):
                with cols[j]: render_tile(res.iloc[i+j], f"gal_{i+j}")

elif nav == "Domain ROI":
    agg = df_master.groupby('domain').agg({'revenue_impact':'sum','usage':'sum','accuracy':'mean'}).reset_index()
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.pie(agg, values='revenue_impact', names='domain', hole=0.4))
    c2.plotly_chart(px.bar(agg, x='domain', y='revenue_impact', color='accuracy'))

elif nav == "My Portfolio":
    my_m = df_master[df_master['contributor'] == role]
    if not my_m.empty:
        sel = st.selectbox("Inspect Model", my_m['name'].unique())
        m = my_m[my_m['name'] == sel].iloc[0]
        fig = go.Figure(go.Scatterpolar(r=[m['accuracy']*100, 100-(m['data_drift']*100), 50, 95], theta=['Accuracy','Stability','Efficiency','Reliability'], fill='toself'))
        st.plotly_chart(fig, use_container_width=True)

elif nav == "Admin Ops":
    if role == "Nat Patel":
        st.subheader("Leader Approval Queue")
        pend = df_master[df_master['approval_status'] == 'Pending']
        if not pend.empty:
            st.table(pend[['name','domain','contributor','accuracy']])
            if st.button("Bulk Approve"):
                df_master.loc[df_master['approval_status'] == 'Pending', 'approval_status'] = 'Approved'
                df_master.to_csv(REG_PATH, index=False); st.rerun()
        else: st.success("Queue clear.")
    else:
        sel = st.selectbox("Solo Asset Inspector", ["None"] + list(df_master['name'].unique()))
        cv = [1.0 if n == sel else 0.0 for n in df_master['name']]
        fig_p = go.Figure(data=go.Parcoords(
            labelfont=dict(size=10), tickfont=dict(size=8),
            line=dict(color=cv, colorscale=[[0, '#FFF9C4'], [1, '#B71C1C']], showscale=False),
            dimensions=[dict(range=[0.7,1], label='Acc', values=df_master['accuracy']),
                        dict(range=[0,150], label='Lat', values=df_master['latency']),
                        dict(range=[0,25000], label='Use', values=df_master['usage'])]))
        st.plotly_chart(fig_p, use_container_width=True)
