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
        font-size: 0.62rem; height: 22px; width: 100%;
    }
    .stButton>button:hover { background-color: var(--deep-purple); color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- CONSTANTS ---
REG_PATH = "model_registry_v3.csv"
REQ_PATH = "requests_v3.csv"

# --- DATA ENGINE (Self-Healing to prevent ROI Errors) ---
def load_sanitized_data():
    if not os.path.exists(REG_PATH): return pd.DataFrame()
    df = pd.read_csv(REG_PATH)
    
    # 1. Ensure Critical Columns Exist
    if 'revenue_impact' not in df.columns:
        df['revenue_impact'] = pd.to_numeric(df.get('usage', 0), errors='coerce').fillna(0) * 12.5
    if 'risk_exposure' not in df.columns:
        df['risk_exposure'] = (1 - pd.to_numeric(df.get('accuracy', 0), errors='coerce').fillna(0.8)) * 50000
    
    # 2. Strict numeric cleaning
    nums = ['usage', 'accuracy', 'latency', 'data_drift', 'revenue_impact', 'risk_exposure', 'cpu_util']
    for c in nums:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce').fillna(0)
            
    if 'approval_status' not in df.columns: df['approval_status'] = 'Approved'
    return df

df_master = load_sanitized_data()

# --- SEARCH ENGINE ---
def hybrid_search(query, df):
    if not query or df.empty: return df
    q = query.lower()
    work_df = df.copy().fillna('N/A')
    pats = {'latency': r'latency\s*([<>]=?)\s*(\d+)', 'accuracy': r'accuracy\s*([<>]=?)\s*(\d+)'}
    for key, pat in pats.items():
        match = re.search(pat, q)
        if match:
            op, val = match.groups()
            val = float(val)/100 if (key=='accuracy' and float(val)>1) else float(val)
            if '>' in op: work_df = work_df[work_df[key] >= val]
            else: work_df = work_df[work_df[key] <= val]
    
    if work_df.empty: return work_df
    work_df['blob'] = work_df.astype(str).apply(' '.join, axis=1)
    vec = TfidfVectorizer(stop_words='english')
    mtx = vec.fit_transform(work_df['blob'].tolist() + [query])
    work_df['relevance'] = cosine_similarity(mtx[-1], mtx[:-1])[0]
    return work_df[work_df['relevance'] > 0.01].sort_values('relevance', ascending=False)

# --- GEMINI AI CONFIG ---
def get_ai_response(prompt, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        context = f"Enterprise Model Assistant. Database has {len(df_master)} models. Current user: {st.session_state.get('role','User')}."
        return model.generate_content(context + "\nUser: " + prompt).text
    except Exception as e:
        return f"AI Logic Active. Details: {str(e)}"

# --- UI COMPONENTS ---
def render_tile(row, key_prefix):
    st.markdown(f"""
    <div class="model-card">
        <div>
            <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                <span class="registry-badge">v{row.get('model_version','1.0')}</span>
                <span style="color:#2E7D32; font-size:0.6rem;">● {row.get('model_stage','Prod')}</span>
            </div>
            <div class="model-title">{row['name']}</div>
            <div class="meta-line"><b>{row['domain']}</b> | {row['contributor']}</div>
            <div class="use-case-text">{row['use_cases']}</div>
        </div>
        <div class="metric-bar">
            <div class="metric-val"><span class="metric-label">Acc</span>{int(row['accuracy']*100)}%</div>
            <div class="metric-val"><span class="metric-label">Lat</span>{int(row['latency'])}ms</div>
            <div class="metric-val"><span class="metric-label">Drift</span>{row['data_drift']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Request Access", key=f"btn_{key_prefix}_{random.randint(0,9999)}"):
        st.toast("Access Request Logged")

# --- NAVIGATION ---
with st.sidebar:
    st.title("Model Hub")
    api_key = st.text_input("Gemini API Key", type="password")
    nav = st.radio("Navigation", ["AI Copilot", "Model Gallery", "Strategy ROI", "Admin Ops"])
    st.divider()
    role = st.selectbox("Login As", ["John Doe", "Jane Nu", "Sam King", "Nat Patel", "Admin"])
    st.session_state.role = role

# --- 1. AI COPILOT (The Brain) ---
if nav == "AI Copilot":
    st.header("🤖 Intelligent Model Copilot")
    if "messages" not in st.session_state: st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "chart" in msg: st.plotly_chart(msg["chart"], use_container_width=True)

    if prompt := st.chat_input("E.g. 'Compare Cash App and Ris-Model-001' or 'Submit a new model'"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            # A. SUBMISSION INTENT
            if "submit" in prompt.lower() or "register" in prompt.lower():
                name_match = re.search(r"name is ([\w\s]+),", prompt, re.I)
                new_row = pd.DataFrame([{"name": name_match.group(1) if name_match else "Chat-Asset", "domain": "Finance", "accuracy": 0.99, "latency": 27, "contributor": role, "approval_status": "Pending", "model_stage": "Prod", "use_cases": prompt}])
                df_master = pd.concat([df_master, new_row], ignore_index=True)
                df_master.to_csv(REG_PATH, index=False)
                res_txt = "✅ **Model submitted!** I have added your asset and notified **Nat Patel** for approval."
                st.markdown(res_txt)
                st.session_state.messages.append({"role": "assistant", "content": res_txt})

            # B. COMPARISON INTENT
            elif "compare" in prompt.lower():
                names = [n for n in df_master['name'] if n.lower() in prompt.lower()]
                if len(names) >= 2:
                    comp_df = df_master[df_master['name'].isin(names)]
                    fig = px.bar(comp_df, x='name', y=['accuracy', 'latency', 'data_drift'], barmode='group', title="Direct Comparison")
                    st.plotly_chart(fig, use_container_width=True)
                    res_txt = f"I've generated a performance comparison between **{', '.join(names)}**."
                    st.markdown(res_txt)
                    st.session_state.messages.append({"role": "assistant", "content": res_txt, "chart": fig})
                else:
                    st.markdown("I couldn't find two specific model names to compare. Please use exact names.")

            # C. GENERAL SEARCH
            else:
                res_txt = get_ai_response(prompt, api_key)
                st.markdown(res_txt)
                results = hybrid_search(prompt, df_master)
                if not results.empty and len(results) < len(df_master):
                    cols = st.columns(min(len(results), 3))
                    for idx, r in enumerate(results.head(3).to_dict('records')):
                        with cols[idx]: render_tile(r, f"chat_{idx}")
                st.session_state.messages.append({"role": "assistant", "content": res_txt})

# --- 2. MODEL GALLERY ---
elif nav == "Model Gallery":
    q = st.text_input("Smart Search (Text or Logic)")
    res = hybrid_search(q, df_master)
    for i in range(0, min(len(res), 15), 3):
        cols = st.columns(3)
        for j in range(3):
            if i+j < len(res):
                with cols[j]: render_tile(res.iloc[i+j], f"gal_{i+j}")

# --- 3. STRATEGY ROI (Fixed KeyError) ---
elif nav == "Strategy ROI":
    st.header("Domain Strategy & Value Dashboard")
    # Clean group by
    agg = df_master.groupby('domain').agg({
        'revenue_impact': 'sum',
        'risk_exposure': 'sum',
        'usage': 'sum',
        'accuracy': 'mean'
    }).reset_index()
    
    k1, k2 = st.columns(2)
    k1.metric("Total Rev Impact", f"${agg['revenue_impact'].sum()/1e6:.1f}M")
    k2.metric("Total Usage", f"{int(agg['usage'].sum()):,}")
    
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(px.pie(agg, values='revenue_impact', names='domain', hole=0.4, title="Revenue by Domain"))
    with c2: st.plotly_chart(px.bar(agg, x='domain', y='revenue_impact', color='accuracy', title="Performance vs. Value"))

# --- 4. ADMIN OPS / NAT PATEL ---
elif nav == "Admin Ops":
    if role == "Nat Patel":
        st.subheader("Approval Queue")
        pend = df_master[df_master['approval_status'] == 'Pending']
        st.table(pend[['name','domain','contributor','accuracy']])
        if st.button("Approve All Requests"):
            df_master['approval_status'] = 'Approved'
            df_master.to_csv(REG_PATH, index=False); st.rerun()
    else:
        sel = st.selectbox("Highlight Asset", ["None"] + list(df_master['name'].unique()))
        cv = [1.0 if n == sel else 0.0 for n in df_master['name']]
        fig = go.Figure(data=go.Parcoords(
            labelfont=dict(size=10, color='black'),
            line=dict(color=cv, colorscale=[[0, '#FFF9C4'], [1, '#B71C1C']], showscale=False),
            dimensions=[dict(range=[0.7,1], label='Accuracy', values=df_master['accuracy']),
                        dict(range=[0,150], label='Latency', values=df_master['latency']),
                        dict(range=[0,25000], label='Usage', values=df_master['usage'])]))
        fig.update_layout(margin=dict(t=80, b=40, l=80, r=80), height=550)
        st.plotly_chart(fig, use_container_width=True)
