import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import os, datetime, random, re, csv

# --- CONFIG & STYLING ---
st.set_page_config(page_title="Model Hub 4.0 | AI Copilot", layout="wide")

st.markdown("""
    <style>
    :root { --accent: #6200EE; --lite-purple: #F3E5F5; --deep-purple: #7B1FA2; --pale-yellow: #FFF9C4; }
    .stApp { background-color: #F8FAFC; font-size: 0.82rem; }
    .model-card {
        background: white; border: 1px solid #e2e8f0; border-top: 3px solid var(--accent);
        padding: 10px; border-radius: 6px; min-height: 300px; 
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
        font-size: 0.62rem; height: 22px; width: 100%;
    }
    /* Chat bubbles */
    .chat-bubble-assistant { background-color: #f0f2f6; padding: 15px; border-radius: 15px; border-bottom-left-radius: 2px; margin-bottom: 10px; }
    .chat-bubble-user { background-color: var(--lite-purple); padding: 15px; border-radius: 15px; border-bottom-right-radius: 2px; margin-bottom: 10px; text-align: right; }
    </style>
    """, unsafe_allow_html=True)

# --- CONSTANTS & DATA ---
REG_PATH = "model_registry_v3.csv"
SHAP_FEATURES = {"Finance": ["Credit_Score", "Income", "Debt"], "Healthcare": ["BMI", "BP", "Glucose"], "Default": ["F1", "F2", "F3"]}

def load_and_sanitize_data():
    if not os.path.exists(REG_PATH): return pd.DataFrame()
    df = pd.read_csv(REG_PATH)
    numeric_cols = ['usage', 'accuracy', 'latency', 'data_drift', 'revenue_impact', 'risk_exposure', 'cpu_util']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce').fillna(0)
    if 'approval_status' not in df.columns: df['approval_status'] = 'Approved'
    return df

df_master = load_and_sanitize_data()

# --- GEMINI SETUP ---
if "messages" not in st.session_state:
    st.session_state.messages = []

def get_gemini_response(prompt, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        # Instruct the model to be a model hub expert
        context = f"You are the Enterprise Model Hub Assistant. You help users find AI models, compare performance, and see ROI. Our database has {len(df_master)} models. Current user is {st.session_state.get('role', 'User')}. Keep it professional and concise."
        response = model.generate_content(context + "\nUser: " + prompt)
        return response.text
    except Exception as e:
        return f"Error connecting to Gemini: {str(e)}"

# --- SEARCH ENGINE ---
def hybrid_search(query, df):
    if not query: return df
    q = query.lower()
    df_res = df.copy()
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
    df_res['blob'] = df_res.fillna('').astype(str).apply(' '.join, axis=1)
    vec = TfidfVectorizer(stop_words='english')
    mtx = vec.fit_transform(df_res['blob'].tolist() + [query])
    df_res['relevance'] = cosine_similarity(mtx[-1], mtx[:-1])[0]
    return df_res[df_res['relevance'] > 0.01].sort_values('relevance', ascending=False)

# --- UI LOGIC ---
def render_tile(row, idx):
    st.markdown(f"""
    <div class="model-card">
        <div>
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
    if st.button("Request Access", key=f"chat_req_{idx}"):
        st.toast("Request Sent")

# --- SIDEBAR ---
with st.sidebar:
    st.title("Model Hub 4.0")
    gemini_key = st.text_input("Gemini API Key", type="password")
    view = st.radio("Navigation", ["AI Copilot", "Marketplace", "Domain ROI", "Admin Ops"])
    st.divider()
    role = st.selectbox("Switch User", ["John Doe", "Jane Nu", "Sam King", "Nat Patel", "Admin"])
    st.session_state.role = role

# --- 1. AI COPILOT (The Chat Interface) ---
if view == "AI Copilot":
    st.header("🤖 Enterprise AI Assistant")
    st.caption("I can help you search models, analyze drift, or check ROI. Try: 'Show me high accuracy Finance models'")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "df" in message: # If the message has data attached, render it
                df_to_show = message["df"]
                if not df_to_show.empty:
                    cols = st.columns(min(len(df_to_show), 3))
                    for idx, r in enumerate(df_to_show.head(3).to_dict('records')):
                        with cols[idx]: render_tile(r, f"chat_{idx}_{random.randint(0,999)}")

    # Chat Input
    if prompt := st.chat_input("How can I help you with the Model Gallery?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if not gemini_key:
                response_text = "Please provide a Gemini API Key in the sidebar to enable conversational intelligence. For now, I'll use simple search."
                results = hybrid_search(prompt, df_master)
            else:
                with st.spinner("Thinking..."):
                    response_text = get_gemini_response(prompt, gemini_key)
                    results = hybrid_search(prompt, df_master) # Use our engine to get the data
            
            st.markdown(response_text)
            
            # If the user is asking for models, show them
            if len(results) < len(df_master):
                st.write(f"I found {len(results)} relevant models based on your request:")
                cols = st.columns(min(len(results), 3))
                for idx, r in enumerate(results.head(3).to_dict('records')):
                    with cols[idx]: render_tile(r, f"new_chat_{idx}_{random.randint(0,999)}")
            
            # Special case: If user mentions ROI
            if "roi" in prompt.lower() or "revenue" in prompt.lower():
                st.write("Here is the current Revenue/Risk breakdown by Domain:")
                agg = df_master.groupby('domain').agg({'revenue_impact':'sum'}).reset_index()
                fig = px.pie(agg, values='revenue_impact', names='domain', hole=0.4)
                fig.update_layout(height=300, margin=dict(l=0,r=0,b=0,t=0))
                st.plotly_chart(fig, use_container_width=True)

        st.session_state.messages.append({"role": "assistant", "content": response_text, "df": results.head(3)})

# --- 2. MARKETPLACE ---
elif view == "Marketplace":
    t1, t2 = st.tabs(["🏛 Gallery", "🚀 Ingest"])
    with t1:
        query = st.text_input("Search (Keywords or Logic)")
        res = hybrid_search(query, df_master)
        for i in range(0, min(len(res), 15), 3):
            cols = st.columns(3)
            for j in range(3):
                if i+j < len(res):
                    with cols[j]: render_tile(res.iloc[i+j], f"gal_{i+j}")
    with t2:
        with st.form("ingest"):
            fn = st.text_input("Model Name*"); fe = st.text_area("Description*")
            if st.form_submit_button("Publish"):
                st.success("Asset live in registry!")

# --- 3. DOMAIN ROI ---
elif view == "Domain ROI":
    st.header("Business Value Dashboard")
    agg = df_master.groupby('domain').agg({'revenue_impact':'sum','risk_exposure':'sum','usage':'sum','accuracy':'mean'}).reset_index()
    k1, k2 = st.columns(2)
    k1.metric("Revenue Impact", f"${agg['revenue_impact'].sum()/1e6:.1f}M")
    k2.metric("Risk Mitigated", f"${agg['risk_exposure'].sum()/1e6:.1f}M")
    st.plotly_chart(px.bar(agg, x='domain', y='revenue_impact', color='accuracy', title="ROI by Business Unit"))

# --- 4. ADMIN OPS ---
elif view == "Admin Ops":
    if role == "Nat Patel":
        st.subheader("Nat Patel's Approval Queue")
        pend = df_master[df_master['approval_status'] == 'Pending']
        st.dataframe(pend[['name','domain','contributor']])
    else:
        sel = st.selectbox("Focus Asset", ["None"] + list(df_master['name'].unique()))
        cv = [1.0 if n == sel else 0.0 for n in df_master['name']]
        fig = go.Figure(data=go.Parcoords(
            labelfont=dict(size=10), tickfont=dict(size=8),
            line=dict(color=cv, colorscale=[[0, '#FFF9C4'], [1, '#B71C1C']], showscale=False),
            dimensions=[dict(range=[0.7,1], label='Acc', values=df_master['accuracy']),
                        dict(range=[0,150], label='Lat', values=df_master['latency']),
                        dict(range=[0,25000], label='Use', values=df_master['usage'])]))
        st.plotly_chart(fig, use_container_width=True)
