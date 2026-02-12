import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import os, datetime, random, re, csv, json

# --- CONFIG & STYLING ---
st.set_page_config(page_title="Enterprise Model Hub 5.0", layout="wide")

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
        font-size: 0.62rem; height: 22px; width: 100%;
    }
    .stButton>button:hover { background-color: var(--deep-purple); color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- DATA PERSISTENCE ---
REG_PATH = "model_registry_v3.csv"
REQ_PATH = "requests_v3.csv"

def load_sanitized_data():
    if not os.path.exists(REG_PATH): return pd.DataFrame()
    df = pd.read_csv(REG_PATH)
    # Ensure ROI columns exist
    if 'revenue_impact' not in df.columns:
        df['revenue_impact'] = pd.to_numeric(df.get('usage', 0), errors='coerce').fillna(0) * 12.5
    if 'risk_exposure' not in df.columns:
        df['risk_exposure'] = (1 - pd.to_numeric(df.get('accuracy', 0), errors='coerce').fillna(0.8)) * 50000
    
    nums = ['usage', 'accuracy', 'latency', 'data_drift', 'revenue_impact', 'risk_exposure']
    for c in nums:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce').fillna(0)
    if 'approval_status' not in df.columns: df['approval_status'] = 'Approved'
    return df

df_master = load_sanitized_data()

# --- GEMINI INTEGRATION ---
def call_gemini(prompt, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        return model.generate_content(prompt).text
    except Exception as e:
        return f"Error: {str(e)}"

# --- INTENT 1: SUBMISSION EXTRACTION ---
def extract_model_entities(user_input, api_key):
    prompt = f"""
    Act as a data extractor. Extract model details from the user text into a JSON object.
    Fields: name, domain, accuracy (as float 0-1), latency (as int ms), use_cases.
    Text: "{user_input}"
    Return ONLY valid JSON. If a field is missing, use null.
    """
    response = call_gemini(prompt, api_key)
    try:
        # Clean potential markdown from LLM
        clean_json = re.search(r"\{.*\}", response, re.DOTALL).group()
        return json.loads(clean_json)
    except:
        return None

# --- INTENT 2: ROI ANALYTICS ---
def get_roi_stats(query_text):
    q = query_text.lower()
    domains = df_master['domain'].unique()
    target_domain = next((d for d in domains if d.lower() in q), None)
    
    if target_domain:
        subset = df_master[df_master['domain'] == target_domain]
        rev = subset['revenue_impact'].sum()
        risk = subset['risk_exposure'].sum()
        count = len(subset)
        top_models = subset.nlargest(3, 'revenue_impact')[['name', 'revenue_impact']].to_dict('records')
        return {"domain": target_domain, "revenue": rev, "risk": risk, "count": count, "top": top_models}
    return None

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
    if st.button("Request Access", key=f"btn_{key_prefix}_{random.randint(0,99999)}"):
        st.toast("Access Request Logged")

# --- NAVIGATION ---
with st.sidebar:
    st.title("Model Hub 5.0")
    api_key = st.text_input("Gemini API Key", type="password")
    nav = st.radio("Navigation", ["AI Copilot", "Model Gallery", "Strategy ROI", "Admin Ops"])
    st.divider()
    role = st.selectbox("Current User", ["John Doe", "Jane Nu", "Sam King", "Nat Patel", "Admin"])
    st.session_state.role = role

# --- 1. AI COPILOT ---
if nav == "AI Copilot":
    st.header("🤖 Intelligent Assistant")
    if "messages" not in st.session_state: st.session_state.messages = []
    if "last_context" not in st.session_state: st.session_state.last_context = None

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "df" in msg:
                cols = st.columns(min(len(msg["df"]), 3))
                for idx, r in enumerate(msg["df"].head(3).to_dict('records')):
                    with cols[idx]: render_tile(r, f"chat_h_{idx}")

    if prompt := st.chat_input("E.g. 'How much revenue did Finance create?' or 'Show my models'"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            # A. INTENT: SUBMISSION
            if "submit" in prompt.lower() or "register" in prompt.lower():
                entities = extract_model_entities(prompt, api_key)
                if entities:
                    new_row = pd.DataFrame([{
                        "name": entities.get('name', 'Chat-Asset'),
                        "domain": entities.get('domain', 'General'),
                        "accuracy": entities.get('accuracy', 0.85),
                        "latency": entities.get('latency', 50),
                        "use_cases": entities.get('use_cases', prompt),
                        "contributor": role, "approval_status": "Pending", "model_stage": "Prod", "usage": 0
                    }])
                    df_master = pd.concat([df_master, new_row], ignore_index=True)
                    df_master.to_csv(REG_PATH, index=False)
                    res = f"✅ I've extracted the details for **{entities.get('name')}** and submitted it to **Nat Patel**."
                else:
                    res = "I couldn't extract the model details. Please specify Name, Domain, Accuracy, and Latency."
                st.markdown(res)

            # B. INTENT: PERSONAL PORTFOLIO
            elif "my contributions" in prompt.lower() or "my models" in prompt.lower():
                user_df = df_master[df_master['contributor'] == role]
                res = f"You have contributed **{len(user_df)}** models, {role}. Here are the top ones:"
                st.markdown(res)
                cols = st.columns(min(len(user_df), 3))
                for idx, r in enumerate(user_df.head(3).to_dict('records')):
                    with cols[idx]: render_tile(r, f"port_{idx}")
                st.session_state.messages.append({"role": "assistant", "content": res, "df": user_df.head(3)})

            # C. INTENT: ROI ANALYTICS
            elif "revenue" in prompt.lower() or "impact" in prompt.lower() or "contributor" in prompt.lower():
                stats = get_roi_stats(prompt) or st.session_state.last_context
                if stats:
                    st.session_state.last_context = stats
                    if "top" in prompt.lower() or "which models" in prompt.lower():
                        res = f"The top contributors in **{stats['domain']}** are: " + ", ".join([m['name'] for m in stats['top']])
                    else:
                        res = f"The **{stats['domain']}** domain has created a revenue impact of **${stats['revenue']/1e6:.2f}M** across {stats['count']} models."
                else:
                    res = "Which domain (Finance, HR, IT Ops) are you interested in?"
                st.markdown(res)

            # D. DEFAULT SEARCH
            else:
                res = call_gemini(prompt, api_key) if api_key else "I found these models for you:"
                st.markdown(res)
            
            st.session_state.messages.append({"role": "assistant", "content": res})

# --- 2. MODEL GALLERY ---
elif nav == "Model Gallery":
    q = st.text_input("Smart Search")
    # (Existing hybrid_search logic here)
    res = df_master # Placeholder for gallery logic
    for i in range(0, 9, 3):
        cols = st.columns(3)
        for j in range(3):
            if i+j < len(res):
                with cols[j]: render_tile(res.iloc[i+j], f"gal_{i+j}")

# --- 3. STRATEGY ROI ---
elif nav == "Strategy ROI":
    st.header("Strategic Domain Dashboard")
    agg = df_master.groupby('domain').agg({'revenue_impact':'sum','usage':'sum','accuracy':'mean'}).reset_index()
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.pie(agg, values='revenue_impact', names='domain', hole=0.4))
    c2.plotly_chart(px.bar(agg, x='domain', y='revenue_impact', color='accuracy'))

# --- 4. ADMIN OPS / NAT PATEL ---
elif nav == "Admin Ops":
    if role == "Nat Patel":
        st.subheader("Nat Patel's Approval Queue")
        pend = df_master[df_master['approval_status'] == 'Pending']
        if not pend.empty:
            st.table(pend[['name','domain','contributor','accuracy']])
            if st.button("Bulk Approve"):
                df_master.loc[df_master['approval_status'] == 'Pending', 'approval_status'] = 'Approved'
                df_master.to_csv(REG_PATH, index=False); st.rerun()
        else: st.success("No pending approvals.")
    else:
        sel = st.selectbox("Solo Inspector", ["None"] + list(df_master['name'].unique()))
        cv = [1.0 if n == sel else 0.0 for n in df_master['name']]
        fig = go.Figure(data=go.Parcoords(
            labelfont=dict(size=10), line=dict(color=cv, colorscale=[[0, '#FFF9C4'], [1, '#B71C1C']], showscale=False),
            dimensions=[dict(range=[0.7,1], label='Acc', values=df_master['accuracy']),
                        dict(range=[0,150], label='Lat', values=df_master['latency']),
                        dict(range=[0,25000], label='Use', values=df_master['usage'])]))
        st.plotly_chart(fig, use_container_width=True)
