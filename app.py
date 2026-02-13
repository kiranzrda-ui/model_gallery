import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import os, datetime, random, re, csv, json

# --- CONFIG & STYLING ---
st.set_page_config(page_title="Enterprise Model Hub 12.0", layout="wide")

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
        border-radius: 4px; font-size: 0.55rem; height: 18px; padding: 0 2px; width: 100%;
    }
    .stButton>button:hover { background-color: var(--deep-p); color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- CONSTANTS & SCHEMA ---
REG_PATH = "model_registry_v3.csv"
REQ_PATH = "requests_v3.csv"
MASTER_FIELDS = [
    "name", "model_version", "domain", "type", "accuracy", "latency", 
    "clients", "use_cases", "contributor", "usage", "data_drift", "pred_drift", 
    "cpu_util", "mem_util", "throughput", "error_rate", "model_owner_team", 
    "last_retrained_date", "model_stage", "training_data_source", "approval_status", 
    "monitoring_status", "sla_tier", "feature_store_dependency", "inference_endpoint_id",
    "revenue_impact", "risk_exposure"
]
SHAP_FEATURES = {
    "Finance": ["Credit", "Income", "Debt", "Trans_Vol"],
    "Healthcare": ["Age", "BMI", "BP", "Glucose"],
    "Risk": ["Exposure", "Volatility", "Compliance", "Market_Gap"],
    "Supply Chain": ["Lead_Time", "Inventory", "Distance", "Demand"],
    "Default": ["Feature_A", "Feature_B", "Feature_C", "Feature_D"]
}

# --- DATA ENGINE ---
def load_data():
    if not os.path.exists(REG_PATH): return pd.DataFrame(columns=MASTER_FIELDS)
    df = pd.read_csv(REG_PATH)
    for c in ["usage", "accuracy", "latency", "revenue_impact", "data_drift"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce').fillna(0.0)
    if 'approval_status' not in df.columns: df['approval_status'] = 'Approved'
    return df

df_master = load_data()

# --- INITIALIZE SESSION STATE ---
if 'messages' not in st.session_state: st.session_state.messages = []
if 'basket' not in st.session_state: st.session_state.basket = []
if 'active_context' not in st.session_state: st.session_state.active_context = {"domain": None, "intent": None, "draft": {}}

# --- LOGIC ENGINES ---
def call_gemini(prompt, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        return model.generate_content(prompt).text
    except: return "System Offline: Resulting based on local search."

def hybrid_search(query, df):
    if not query or df.empty: return df
    q = query.lower()
    df['blob'] = df.fillna('').astype(str).apply(' '.join, axis=1)
    vec = TfidfVectorizer(stop_words='english')
    mtx = vec.fit_transform(df['blob'].tolist() + [query])
    df['score'] = cosine_similarity(mtx[-1], mtx[:-1])[0]
    return df[df['score'] > 0.01].sort_values('score', ascending=False)

# --- UI COMPONENTS ---
def render_tile(row, user, prefix, result_count):
    st.markdown(f"""
    <div class="model-card">
        <div>
            <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                <span style="font-size:0.5rem; color:#666;">v{row.get('model_version','1.0')}</span>
                <span style="color:#2E7D32; font-size:0.6rem;">● {row['model_stage']}</span>
            </div>
            <div class="model-title">{row['name'][:22]}</div>
            <div style="font-size:0.6rem; color:gray;">{row['domain']} | {row['contributor']}</div>
            <div style="font-size:0.65rem; color:#444; height:3.2em; overflow:hidden;">{row['use_cases']}</div>
        </div>
        <div class="metric-bar">
            <div class="metric-val"><span class="metric-label">Acc</span>{int(row['accuracy']*100)}%</div>
            <div class="metric-val"><span class="metric-label">Lat</span>{int(row['latency'])}ms</div>
            <div class="metric-val"><span class="metric-label">Drift</span>{row['data_drift']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    if result_count > 1: # Only show compare if more than 1
        with c1:
            if st.button("Compare", key=f"c_{prefix}_{row['name']}"):
                if len(st.session_state.basket) < 5:
                    st.session_state.basket.append(row['name']); st.rerun()
    with c2:
        with st.popover("Specs"):
            feats = SHAP_FEATURES.get(row['domain'], SHAP_FEATURES["Default"])
            shap_df = pd.DataFrame({'Feature': feats, 'Impact': [random.uniform(-0.5, 0.5) for _ in range(4)]}).sort_values('Impact')
            shap_df['Color'] = ['#EF4444' if x < 0 else '#6200EE' for x in shap_df['Impact']]
            st.plotly_chart(px.bar(shap_df, x='Impact', y='Feature', orientation='h', color='Color', color_discrete_map="identity", height=140), use_container_width=True)
    with c3:
        if st.button("Request", key=f"r_{prefix}_{row['name']}"):
            st.session_state.request_trigger = f"I want to request access to {row['name']}"
            st.rerun()

# --- SIDEBAR & MODE SELECTION ---
with st.sidebar:
    st.title("Hub Configuration")
    user_role = st.selectbox("Current User", ["John Doe", "Jane Nu", "Sam King", "Nat Patel"])
    app_mode = st.radio("Experience Mode", ["Companion Mode", "Web Mode"])
    api_key = st.text_input("Gemini API Key", type="password")
    st.divider()
    if st.session_state.basket:
        st.success(f"Basket: {len(st.session_state.basket)}/5 items")
        if st.button("Clear Basket"): st.session_state.basket = []; st.rerun()

# --- MAIN LOGIC ---

def run_workbench_companion():
    st.header("🤖 WorkBench Companion")
    
    # Process Tile Triggers
    if 'request_trigger' in st.session_state:
        req_msg = st.session_state.pop('request_trigger')
        m_name = req_msg.split("access to ")[-1]
        new_row = pd.DataFrame([{"model_name": m_name, "requester": user_role, "status": "Pending", "timestamp": str(datetime.datetime.now())}])
        new_row.to_csv(REQ_PATH, mode='a', header=not os.path.exists(REQ_PATH), index=False)
        st.session_state.messages.append({"role": "user", "content": req_msg})
        st.session_state.messages.append({"role": "assistant", "content": f"✅ logged request for **{m_name}**. Nat Patel has been notified."})

    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "df" in msg and not msg["df"].empty:
                cols = st.columns(min(len(msg["df"]), 3))
                for idx, r in enumerate(msg["df"].head(3).to_dict('records')):
                    with cols[idx]: render_tile(r, user_role, f"ch_{i}", len(msg["df"]))
            if "chart" in msg: st.plotly_chart(msg["chart"], key=f"chat_c_{i}")

    if prompt := st.chat_input("Submit a model, check ROI, or search..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            q = prompt.lower()
            res_txt, res_df, res_chart, is_trending = "Thinking...", pd.DataFrame(), None, False

            # INTENT: 1.7 SUBMISSION WORKFLOW
            if "submit" in q or "register" in q or st.session_state.active_context["intent"] == "SUBMISSION":
                st.session_state.active_context["intent"] = "SUBMISSION"
                # Call Gemini for extraction
                extraction_prompt = f"Extract Model JSON (name, use_cases, domain, accuracy, latency). User text: {prompt}. Return ONLY JSON."
                raw_json = call_gemini(extraction_prompt, api_key)
                try:
                    extracted = json.loads(re.search(r"\{.*\}", raw_json, re.DOTALL).group())
                    # Update local draft
                    for k, v in extracted.items():
                        if v: st.session_state.active_context["draft"][k] = v
                except: pass

                draft = st.session_state.active_context["draft"]
                if not draft.get("name"):
                    res_txt = "Sure! I can help you register that. First, **what is the name of the model?**"
                elif not draft.get("use_cases"):
                    res_txt = f"Got it, name is {draft['name']}. Now, **what is the primary use case?**"
                else:
                    # Finalize and Save
                    new_row = {f: "N/A" for f in MASTER_FIELDS}
                    new_row.update(draft)
                    new_row.update({"contributor": user_role, "approval_status": "Pending", "model_stage": "Experimental", "usage": 0})
                    pd.DataFrame([new_row]).to_csv(REG_PATH, mode='a', header=False, index=False)
                    res_txt = f"✅ Success! **{draft['name']}** has been registered. I've extracted all relevant metadata and notified Nat Patel for approval."
                    st.session_state.active_context = {"domain": None, "intent": None, "draft": {}}
            
            # INTENT: CONTEXTUAL ROI
            elif "revenue" in q or "impact" in q:
                dom = next((d for d in df_master['domain'].unique() if d.lower() in q), st.session_state.active_context["domain"])
                if dom:
                    st.session_state.active_context["domain"] = dom
                    val = df_master[df_master['domain']==dom]['revenue_impact'].sum()
                    res_txt = f"The **{dom}** domain impact is **${val/1e6:.2f}M**."
                    if "top" in q or "which models" in q:
                        res_df = df_master[df_master['domain']==dom].nlargest(3, 'revenue_impact')
                else:
                    agg = df_master.groupby('domain')['revenue_impact'].sum().reset_index()
                    res_chart = px.pie(agg, values='revenue_impact', names='domain', hole=0.4)
                    res_txt = "Here is the revenue impact contribution breakdown by domain:"

            # INTENT: COMPARISON
            elif "compare" in q or "difference" in q:
                names = [n for n in df_master['name'].unique() if n.lower() in q]
                if len(names) >= 2:
                    res_df = df_master[df_master['name'].isin(names)]
                    res_chart = px.bar(res_df, x='name', y=['accuracy','latency','usage'], barmode='group')
                    res_txt = f"Comparing {len(names)} models side-by-side:"
                else: res_txt = "Please specify exactly which models you wish to compare."

            # DEFAULT: SEARCH
            else:
                res_df = hybrid_search(q, df_master).head(3)
                if res_df.empty: res_txt = "No models with that criteria found. Try different keywords."
                else: res_txt = call_gemini(prompt, api_key) if api_key else "I found these relevant assets:"

            st.markdown(res_txt)
            if res_chart: st.plotly_chart(res_chart)
            if not res_df.empty:
                cols = st.columns(min(len(res_df), 3))
                for idx, r in enumerate(res_df.to_dict('records')):
                    with cols[idx]: render_tile(r, user_role, f"chat_n_{idx}", len(res_df))
            
            st.session_state.messages.append({"role": "assistant", "content": res_txt, "df": res_df, "chart": res_chart})

# --- UI LOGIC BY MODE ---
if app_mode == "Companion Mode":
    run_workbench_companion()

else:
    # WEB MODE
    tabs = ["WorkBench Companion", "Model Gallery", "AI Business Value"]
    if user_role == "Nat Patel": tabs.append("Approval Portal")
    
    t = st.tabs(tabs)
    
    with t[0]:
        run_workbench_companion()
        
    with t[1]:
        # MODEL GALLERY
        q_gal = st.text_input("Search (Domain, Use Case, Performance...)")
        res_gal = hybrid_search(q_gal, df_master)
        for i in range(0, min(len(res_gal), 12), 3):
            cols = st.columns(3)
            for j in range(3):
                if i+j < len(res_gal):
                    with cols[j]: render_tile(res_gal.iloc[i+j], user_role, f"gal_{i+j}", len(res_gal))

    with t[2]:
        # BUSINESS VALUE
        st.header("Strategic Portfolio ROI")
        src = df_master if user_role == "Nat Patel" else df_master[df_master['contributor'] == user_role]
        agg = src.groupby('domain').agg({'revenue_impact':'sum','usage':'sum','accuracy':'mean'}).reset_index()
        k1, k2 = st.columns(2)
        k1.metric("Revenue Contribution", f"${agg['revenue_impact'].sum()/1e6:.1f}M")
        k2.metric("Avg Accuracy", f"{int(agg['accuracy'].mean()*100)}%")
        
        # Interactive Selector
        sel_dom = st.selectbox("Drill-down by Domain", agg['domain'].unique())
        st.table(src[src['domain']==sel_dom].nlargest(5, 'revenue_impact')[['name','usage','revenue_impact']])
        c_p, c_b = st.columns(2)
        with c_p: st.plotly_chart(px.pie(agg, values='revenue_impact', names='domain', hole=0.4))
        with c_b: st.plotly_chart(px.bar(agg, x='domain', y='revenue_impact'))

    if user_role == "Nat Patel":
        with t[3]:
            # APPROVAL PORTAL (Telemetry included here)
            st.header("Governance Telemetry")
            sel_tele = st.selectbox("Highlight Asset", ["None"] + list(df_master['name'].unique()))
            cv = [1.0 if n == sel_tele else 0.0 for n in df_master['name']]
            fig_p = go.Figure(data=go.Parcoords(
                labelfont=dict(size=10, color='black'), tickfont=dict(size=8),
                line=dict(color=cv, colorscale=[[0, '#FFF9C4'], [1, '#B71C1C']], showscale=False),
                dimensions=[dict(range=[0,25000], label='Usage', values=df_master['usage']),
                            dict(range=[0.7,1], label='Accuracy', values=df_master['accuracy']),
                            dict(range=[0,150], label='Latency', values=df_master['latency']),
                            dict(range=[0,0.3], label='Drift', values=df_master['data_drift']),
                            dict(range=[0,100], label='CPU %', values=df_master['cpu_util'])]))
            fig_p.update_layout(margin=dict(t=80, b=40, l=80, r=80), height=500)
            st.plotly_chart(fig_p, use_container_width=True)
            
            st.divider()
            st.subheader("Pending Approvals")
            df_req_curr = pd.read_csv(REQ_PATH)
            pend = df_req_curr[df_req_curr['status'] == 'Pending']
            if not pend.empty:
                for idx, r in pend.iterrows():
                    ca, cb = st.columns([4, 1])
                    ca.write(f"💼 **{r['requester']}** requested **{r['model_name']}**")
                    if cb.button("Approve", key=f"ap_{idx}"):
                        df_req_curr.at[idx, 'status'] = 'Approved'
                        df_req_curr.to_csv(REQ_PATH, index=False); st.rerun()
            else: st.success("Queue clear.")
