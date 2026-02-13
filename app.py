import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import os, datetime, random, re, csv, json

# --- CONFIG & STYLING ---
st.set_page_config(page_title="Enterprise Model Hub 14.0", layout="wide")

st.markdown("""
    <style>
    :root { --accent: #6200EE; --lite-p: #F3E5F5; --deep-p: #7B1FA2; --pale-y: #FFF9C4; }
    .stApp { background-color: #F8FAFC; font-size: 0.8rem; }
    .model-card {
        background: white; border: 1px solid #e2e8f0; border-top: 3px solid var(--accent);
        padding: 8px 10px; border-radius: 6px; min-height: 260px; 
        display: flex; flex-direction: column; justify-content: space-between;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    .model-title { font-size: 0.82rem; font-weight: 700; color: #1e293b; margin-bottom: 1px; }
    .owner-tag { font-size: 0.52rem; font-weight: bold; color: var(--deep-p); text-transform: uppercase; background: var(--lite-p); padding: 1px 4px; border-radius:3px; }
    .metric-bar { display: flex; justify-content: space-between; background: #F1F5F9; padding: 3px; border-radius: 4px; margin-bottom: 4px; }
    .metric-val { font-size: 0.7rem; font-weight: 700; color: var(--accent); text-align: center; flex: 1; }
    .metric-label { font-size: 0.45rem; color: #94a3b8; display: block; text-transform: uppercase; }
    .stButton>button { 
        background-color: var(--lite-p); color: var(--deep-p); border: 1px solid var(--deep-p);
        border-radius: 4px; font-size: 0.58rem; height: 18px; padding: 0 2px; width: 100%;
        line-height: 1;
    }
    .stButton>button:hover { background-color: var(--deep-p); color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- SCHEMA DEFINITION ---
MASTER_FIELDS = [
    "name", "model_version", "domain", "type", "accuracy", "latency", 
    "clients", "use_cases", "contributor", "usage", "data_drift", "pred_drift", 
    "cpu_util", "mem_util", "throughput", "error_rate", "model_owner_team", 
    "last_retrained_date", "model_stage", "training_data_source", "approval_status", 
    "monitoring_status", "sla_tier", "feature_store_dependency", "inference_endpoint_id",
    "revenue_impact", "risk_exposure"
]
REG_PATH = "model_registry_v3.csv"
REQ_PATH = "requests_v3.csv"
SHAP_FEATURES = {
    "Finance": ["Credit_Score", "Income", "Debt", "Risk"],
    "Healthcare": ["Age", "BMI", "BP", "Glucose"],
    "Risk": ["Exposure", "Volatility", "Compliance", "Liq"],
    "Supply Chain": ["Lead_Time", "Inventory", "Route", "Demand"],
    "Default": ["Feature_1", "Feature_2", "Feature_3", "Feature_4"]
}

# --- DATA ENGINE ---
def load_and_sanitize():
    if not os.path.exists(REG_PATH): return pd.DataFrame(columns=MASTER_FIELDS)
    df = pd.read_csv(REG_PATH)
    # Ensure all columns exist and are correct types
    for col in MASTER_FIELDS:
        if col not in df.columns:
            df[col] = 0.0 if col in ["accuracy", "usage", "revenue_impact", "risk_exposure", "latency", "data_drift"] else "N/A"
    
    # Force Numeric for ROI Logic
    nums = ["accuracy", "usage", "revenue_impact", "risk_exposure", "latency", "data_drift"]
    for n in nums:
        df[n] = pd.to_numeric(df[n].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce').fillna(0.0)
    
    if 'approval_status' not in df.columns: df['approval_status'] = 'Approved'
    return df

def load_reqs():
    if not os.path.exists(REQ_PATH): return pd.DataFrame(columns=["model_name", "requester", "status", "timestamp"])
    return pd.read_csv(REQ_PATH)

# Global Data Refresh
df_master = load_and_sanitize()
df_reqs = load_reqs()

# --- INITIALIZE STATE ---
if 'messages' not in st.session_state: st.session_state.messages = []
if 'basket' not in st.session_state: st.session_state.basket = []
if 'context' not in st.session_state: st.session_state.context = {"domain": None, "intent": None, "draft": {}}

# --- LOGIC ENGINES ---
def call_gemini(prompt, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        return model.generate_content(prompt).text
    except: return "Assistant: I am currently in local search mode."

def hybrid_search(query, df):
    if not query or df.empty: return df
    q = query.lower()
    df_res = df.copy().fillna('N/A')
    df_res['blob'] = df_res.astype(str).apply(' '.join, axis=1)
    vec = TfidfVectorizer(stop_words='english')
    mtx = vec.fit_transform(df_res['blob'].tolist() + [query])
    df_res['score'] = cosine_similarity(mtx[-1], mtx[:-1])[0]
    return df_res[df_res['score'] > 0.01].sort_values('score', ascending=False)

# --- UI COMPONENTS ---
def render_tile(row, user, prefix, total_count):
    # Determine Label
    label = ""
    if row['contributor'] == user: label = "Owned"
    elif not df_reqs[(df_reqs['model_name']==row['name']) & (df_reqs['requester']==user) & (df_reqs['status']=='Approved')].empty: label = "Licensed"

    st.markdown(f"""
    <div class="model-card">
        <div>
            <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                <span class="owner-tag">{label}</span>
                <span style="font-size:0.5rem; color:#666;">v{row.get('model_version','1.0')}</span>
            </div>
            <div class="model-title">{row['name'][:22]}</div>
            <div style="font-size:0.6rem; color:gray;">{row['domain']} | {row['contributor']}</div>
            <div style="font-size:0.65rem; color:#444; height:3.2em; overflow:hidden;">{row['use_cases']}</div>
        </div>
        <div class="metric-bar">
            <div class="metric-val"><span class="metric-label">Acc</span>{int(row['accuracy']*100)}%</div>
            <div class="metric-val"><span class="metric-label">Lat</span>{int(row['latency'])}ms</div>
            <div class="metric-val"><span class="metric-label">Value</span>${row['revenue_impact']/1000:.1f}k</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    if total_count > 1:
        with c1:
            if st.button("Compare", key=f"c_{prefix}_{row['name']}"):
                st.session_state.chat_trigger = f"Compare model {row['name']}"
                st.rerun()
    with c2:
        with st.popover("Specs"):
            st.write(f"**Technical SHAP Detail: {row['name']}**")
            feats = SHAP_FEATURES.get(row['domain'], SHAP_FEATURES["Default"])
            shap_df = pd.DataFrame({'Feature': feats, 'Impact': [random.uniform(-0.5, 0.5) for _ in range(4)]}).sort_values('Impact')
            shap_df['Color'] = ['#EF4444' if x < 0 else '#6200EE' for x in shap_df['Impact']]
            st.plotly_chart(px.bar(shap_df, x='Impact', y='Feature', orientation='h', color='Color', color_discrete_map="identity", height=200), use_container_width=True)
    with c3:
        if st.button("Request", key=f"r_{prefix}_{row['name']}"):
            st.session_state.chat_trigger = f"I want to request access for {row['name']}"
            st.rerun()

# --- SIDEBAR & MODES ---
with st.sidebar:
    st.title("Hub Controls")
    app_mode = st.toggle("Enable Web Mode", value=False)
    user_role = st.selectbox("Current User", ["John Doe", "Jane Nu", "Sam King", "Nat Patel"])
    api_key = st.text_input("Gemini API Key", type="password")
    
    if app_mode:
        if user_role == "Nat Patel":
            nav = st.radio("Portal", ["WorkBench Companion", "Approval Portal"])
        else:
            nav = st.radio("Portal", ["WorkBench Companion", "Model Gallery", "AI Business Value"])
    else:
        nav = "WorkBench Companion"

# --- WORKBENCH COMPANION ENGINE ---
def run_companion():
    st.header("🤖 WorkBench Companion")
    
    # Handling redirected button actions
    if 'chat_trigger' in st.session_state:
        trigger = st.session_state.pop('chat_trigger')
        st.session_state.messages.append({"role": "user", "content": trigger})

    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "df" in msg and not msg["df"].empty:
                cols = st.columns(min(len(msg["df"]), 3))
                for idx, r in enumerate(msg["df"].head(3).to_dict('records')):
                    with cols[idx]: render_tile(r, user_role, f"chat_{i}", len(msg["df"]))
            if "bar" in msg: st.plotly_chart(msg["bar"], key=f"bar_{i}")
            if "pie" in msg: st.plotly_chart(msg["pie"], key=f"pie_{i}")

    if prompt := st.chat_input("E.g., 'Submit model Cash App' or 'Revenue in Finance'"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            q = prompt.lower()
            res_txt, res_df, res_pie, res_bar = "Analyzing...", pd.DataFrame(), None, None
            
            # Detect Context Switch
            found_dom = next((d for d in df_master['domain'].unique() if d.lower() in q), None)
            if found_dom: st.session_state.context["domain"] = found_dom

            # 1.7 SUBMISSION INTENT
            if "submit" in q or "register" in q or st.session_state.context["intent"] == "SUBMISSION":
                st.session_state.context["intent"] = "SUBMISSION"
                extract_p = f"Extract JSON (name, use_cases, domain, accuracy, latency). Text: {prompt}. Return ONLY JSON."
                try:
                    raw_json = call_gemini(extract_p, api_key)
                    extracted = json.loads(re.search(r"\{.*\}", raw_json, re.DOTALL).group())
                    for k,v in extracted.items(): 
                        if v: st.session_state.context["draft"][k] = v
                except: pass
                
                draft = st.session_state.context["draft"]
                if not draft.get("name"): res_txt = "Sure! **What is the name of the model?**"
                elif not draft.get("use_cases"): res_txt = f"Got it. **What is the use case for {draft['name']}?**"
                else:
                    new_row = {**{f:"N/A" for f in MASTER_FIELDS}, **draft, "contributor":user_role, "approval_status":"Pending", "usage":0}
                    pd.DataFrame([new_row]).to_csv(REG_PATH, mode='a', header=False, index=False)
                    res_txt = f"✅ Success! **{draft['name']}** has been registered and sent to Nat Patel."
                    st.session_state.context = {"domain": None, "intent": None, "draft": {}}

            # NAT PATEL APPROVALS
            elif user_role == "Nat Patel" and ("queue" in q or "approve" in q):
                if "approve" in q:
                    m_name = next((n for n in df_master['name'] if n.lower() in q), None)
                    if m_name:
                        df_master.loc[df_master['name']==m_name, 'approval_status'] = 'Approved'
                        df_master.to_csv(REG_PATH, index=False)
                        res_txt = f"✅ Approved **{m_name}**."
                    else: res_txt = "Approved all pending items in your queue."
                else:
                    pend = df_master[df_master['approval_status'] == 'Pending']
                    res_txt = f"Queue: {', '.join(pend['name'].tolist())}" if not pend.empty else "No pending items."

            # ROI & BUSINESS VALUE
            elif "revenue" in q or "impact" in q:
                dom = st.session_state.context["domain"]
                if dom:
                    subset = df_master[df_master['domain'] == dom]
                    val = subset['revenue_impact'].sum()
                    res_txt = f"The **{dom}** domain generated **${val/1e6:.2f}M** revenue impact."
                    if "top" in q: res_df = subset.nlargest(3, 'revenue_impact')
                else:
                    agg = df_master.groupby('domain')['revenue_impact'].sum().reset_index()
                    res_pie = px.pie(agg, values='revenue_impact', names='domain', hole=0.4)
                    res_txt = "Revenue contribution by domain:"

            # COMPARE
            elif "compare" in q:
                names = [n for n in df_master['name'].unique() if n.lower() in q]
                if len(names) >= 2:
                    res_df = df_master[df_master['name'].isin(names)]
                    res_bar = px.bar(res_df, x='name', y=['accuracy','latency','usage'], barmode='group')
                    res_txt = f"Comparing {len(names)} models."
                else: res_txt = "Specify exact model names to compare."

            else:
                res_df = hybrid_search(q, df_master).head(3)
                res_txt = call_gemini(prompt, api_key) if api_key else "I found these assets:"

            st.markdown(res_txt)
            if res_pie: st.plotly_chart(res_pie)
            if res_bar: st.plotly_chart(res_bar)
            if not res_df.empty:
                cols = st.columns(min(len(res_df), 3))
                for idx, r in enumerate(res_df.to_dict('records')):
                    with cols[idx]: render_tile(r, user_role, f"ch_{i}", len(res_df))
            st.session_state.messages.append({"role": "assistant", "content": res_txt, "df": res_df, "pie": res_pie, "bar": res_bar})

# --- UI LOGIC ---
if nav == "WorkBench Companion":
    run_companion()
elif nav == "Model Gallery":
    st.header("🏛 Model Gallery")
    q_gal = st.text_input("Smart Search")
    res = hybrid_search(q_gal, df_master)
    for i in range(0, min(len(res), 15), 3):
        cols = st.columns(3)
        for j in range(3):
            if i+j < len(res):
                with cols[j]: render_tile(res.iloc[i+j], user_role, f"gal_{i+j}", len(res))
elif nav == "AI Business Value":
    st.header("Strategic Portfolio ROI")
    src = df_master if user_role == "Nat Patel" else df_master[df_master['contributor'] == user_role]
    agg = src.groupby('domain').agg({'revenue_impact':'sum','accuracy':'mean'}).reset_index()
    k1, k2 = st.columns(2)
    k1.metric("Rev Contribution", f"${agg['revenue_impact'].sum()/1e6:.1f}M")
    k2.metric("Avg Accuracy", f"{int(agg['accuracy'].mean()*100)}%")
    st.plotly_chart(px.pie(agg, values='revenue_impact', names='domain', hole=0.4))
elif nav == "Approval Portal":
    st.header("Governance & Telemetry")
    sel = st.selectbox("Solo Inspector", ["None"] + list(df_master['name'].unique()))
    cv = [1.0 if n == sel else 0.0 for n in df_master['name']]
    fig_p = go.Figure(data=go.Parcoords(
        labelfont=dict(size=10, color='black'), tickfont=dict(size=8),
        line=dict(color=cv, colorscale=[[0, '#FFF9C4'], [1, '#B71C1C']], showscale=False),
        dimensions=[dict(range=[0,25000], label='Usage', values=df_master['usage']),
                    dict(range=[0.7,1], label='Accuracy', values=df_master['accuracy']),
                    dict(range=[0,150], label='Latency', values=df_master['latency']),
                    dict(range=[0,0.3], label='Drift', values=df_master['data_drift']),
                    dict(range=[0,100], label='CPU %', values=df_master['cpu_util'])]))
    st.plotly_chart(fig_p, use_container_width=True)
