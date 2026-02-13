import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import os, datetime, random, re, csv, json

# --- CONFIG & STYLING ---
st.set_page_config(page_title="Model Hub 13.0", layout="wide")

st.markdown("""
    <style>
    :root { --accent: #6200EE; --lite-p: #F3E5F5; --deep-p: #7B1FA2; }
    .model-card {
        background: white; border: 1px solid #e2e8f0; border-top: 3px solid var(--accent);
        padding: 10px; border-radius: 6px; min-height: 250px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .model-title { font-size: 0.9rem; font-weight: 700; color: #1e293b; }
    .metric-bar { display: flex; justify-content: space-between; background: #F8FAFC; padding: 4px; border-radius: 4px; margin-top: 10px; }
    .metric-val { font-size: 0.75rem; font-weight: 700; color: var(--accent); text-align: center; }
    .stButton>button { font-size: 0.65rem; height: 26px; width: 100%; border-radius: 4px; }
    </style>
    """, unsafe_allow_html=True)

# --- DATA ENGINE ---
REG_PATH = "model_registry_v3.csv"
REQ_PATH = "requests_v3.csv"

def load_data():
    if not os.path.exists(REG_PATH): return pd.DataFrame()
    df = pd.read_csv(REG_PATH)
    # Ensure numerical columns for ROI
    for col in ['accuracy', 'usage', 'revenue_impact', 'latency']:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df

df_master = load_data()

# --- STATE INITIALIZATION ---
if 'messages' not in st.session_state: st.session_state.messages = []
if 'basket' not in st.session_state: st.session_state.basket = []
if 'trigger_action' in st.session_state:
    action = st.session_state.pop('trigger_action')
    st.session_state.messages.append({"role": "user", "content": action})

# --- UI: RENDER TILE (Buttons send text back to Chat) ---
def render_tile(row, prefix):
    st.markdown(f"""
    <div class="model-card">
        <div class="model-title">{row['name']}</div>
        <div style="font-size:0.65rem; color:gray;">{row['domain']} | {row['model_stage']}</div>
        <div style="font-size:0.7rem; height:2.5em; overflow:hidden; margin: 5px 0;">{row['use_cases']}</div>
        <div class="metric-bar">
            <div class="metric-val">ACC<br>{int(row['accuracy']*100)}%</div>
            <div class="metric-val">LAT<br>{int(row['latency'])}ms</div>
            <div class="metric-val">DRIFT<br>{row['data_drift']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    if c1.button("Compare", key=f"c_{prefix}_{row['name']}"):
        st.session_state.trigger_action = f"Compare {row['name']}"
        st.rerun()
    with c2:
        with st.popover("Specs"):
            st.write(f"### {row['name']} SHAP Summary")
            fig = px.bar(pd.DataFrame({'F': ['A','B','C','D'], 'V': [0.1, 0.5, -0.3, 0.2]}), x='V', y='F', orientation='h', height=200)
            fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)
    if c3.button("Request", key=f"r_{prefix}_{row['name']}"):
        st.session_state.trigger_action = f"Request access for {row['name']}"
        st.rerun()

# --- SIDEBAR ---
with st.sidebar:
    st.title("Hub Controls")
    mode = st.toggle("Enable Web Mode")
    user = st.selectbox("Current User", ["John Doe", "Jane Nu", "Sam King", "Nat Patel"])
    
    # Navigation logic based on toggle
    if mode:
        nav = st.radio("Navigation", ["Model Gallery", "AI Business Value"] + (["Approval Portal"] if user == "Nat Patel" else []))
    else:
        nav = "WorkBench Companion"

# --- INTENT HANDLER (The "Brain") ---
def handle_intent(prompt):
    q = prompt.lower()
    # Submission
    if "submit" in q or "register" in q:
        return "✅ Model registered. Nat Patel will review shortly.", pd.DataFrame()
    # Comparison
    elif "compare" in q:
        names = [n for n in df_master['name'] if n.lower() in q]
        return f"Comparing {len(names)} models.", df_master[df_master['name'].isin(names)]
    # Request
    elif "request" in q:
        m = re.search(r"access for ([\w\-\s]+)", q)
        return f"Request logged for {m.group(1) if m else 'Asset'}.", pd.DataFrame()
    # Default
    return "Searching for relevant models...", hybrid_search(prompt, df_master).head(3)

# --- 1. WORKBENCH COMPANION ---
if nav == "WorkBench Companion":
    st.header("🤖 WorkBench Companion")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "df" in msg and not msg["df"].empty:
                cols = st.columns(min(len(msg["df"]), 3))
                for i, r in enumerate(msg["df"].head(3).to_dict('records')):
                    with cols[i]: render_tile(r, f"ch_{i}")

    if prompt := st.chat_input("Prompt: 'Compare [Name1] and [Name2]' or 'Submit model...'"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        with st.chat_message("assistant"):
            txt, df = handle_intent(prompt)
            st.markdown(txt)
            if not df.empty:
                cols = st.columns(min(len(df), 3))
                for i, r in enumerate(df.head(3).to_dict('records')):
                    with cols[i]: render_tile(r, f"ch_res_{i}")
            st.session_state.messages.append({"role": "assistant", "content": txt, "df": df})

# --- 2. MODEL GALLERY ---
elif nav == "Model Gallery":
    st.header("🏛 Model Gallery")
    q = st.text_input("Smart Search")
    res = hybrid_search(q, df_master)
    for i in range(0, min(len(res), 12), 3):
        cols = st.columns(3)
        for j in range(3):
            if i+j < len(res):
                with cols[j]: render_tile(res.iloc[i+j], f"gal_{i+j}")

# --- 3. AI BUSINESS VALUE ---
elif nav == "AI Business Value":
    st.header("Domain Strategy ROI")
    src = df_master if user == "Nat Patel" else df_master[df_master['contributor'] == user]
    agg = src.groupby('domain').agg({'revenue_impact':'sum','accuracy':'mean'}).reset_index()
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(px.pie(agg, values='revenue_impact', names='domain', title="Revenue Attribution"))
    with c2: st.plotly_chart(px.bar(agg, x='domain', y='accuracy', title="Avg Accuracy"))

# --- 4. APPROVAL PORTAL ---
elif nav == "Approval Portal":
    st.header("Leader Approval Gateway")
    pend = df_master[df_master['approval_status'] == 'Pending']
    if not pend.empty:
        st.table(pend[['name','domain','accuracy']])
        if st.button("Bulk Approve"):
            df_master.loc[df_master['approval_status']=='Pending','approval_status']='Approved'
            df_master.to_csv(REG_PATH, index=False); st.rerun()
    else: st.success("Queue empty.")
