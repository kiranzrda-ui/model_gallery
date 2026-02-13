import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import os, datetime, random, re, csv, json

# --- CONFIG & STYLING ---
st.set_page_config(page_title="Enterprise Model Hub", layout="wide")

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

# --- SCHEMA ---
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
    "Finance": ["Credit", "Income", "Debt", "Trans_Vol"],
    "Healthcare": ["Age", "BMI", "BP", "Glucose"],
    "Risk": ["Exposure", "Volatility", "Compliance", "Liquidity"],
    "Supply Chain": ["Lead_Time", "Inventory", "Route", "Demand"],
    "Retail": ["Footfall", "Seasonality", "Margin", "Promo"],
    "Default": ["Feature_1", "Feature_2", "Feature_3", "Feature_4"]
}

# --- DATA ENGINE ---
def load_sanitized_data():
    if not os.path.exists(REG_PATH): return pd.DataFrame(columns=MASTER_FIELDS)
    df = pd.read_csv(REG_PATH)
    for col in MASTER_FIELDS:
        if col not in df.columns:
            df[col] = 0.0 if col in ["accuracy", "usage", "revenue_impact", "risk_exposure", "latency", "data_drift"] else "N/A"
    
    nums = ["accuracy", "usage", "revenue_impact", "risk_exposure", "latency", "data_drift", "cpu_util"]
    for n in nums:
        df[n] = pd.to_numeric(df[n].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce').fillna(0.0)
    
    if df['revenue_impact'].sum() == 0:
        df['revenue_impact'] = df['usage'] * 125.0
        
    if 'approval_status' in df.columns:
        df['approval_status'] = df['approval_status'].str.strip().str.lower().replace({'pending_review': 'Pending', 'pending': 'Pending', 'approved': 'Approved'})
    return df

df_master = load_sanitized_data()

# --- STATE ---
if 'messages' not in st.session_state: st.session_state.messages = []
if 'basket' not in st.session_state: st.session_state.basket = []
if 'chat_context' not in st.session_state: st.session_state.chat_context = {"domain": None, "intent": None, "draft": {}}

# --- LOGIC ENGINES ---
def call_gemini(prompt, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        return model.generate_content(prompt).text
    except: return "Assistant: Processing your request using internal logic."

def hybrid_search(query, df):
    if not query or df.empty: return df
    q = query.lower()
    df_res = df.copy().fillna('N/A')
    
    # Textual Search
    df_res['blob'] = df_res.astype(str).apply(' '.join, axis=1)
    vec = TfidfVectorizer(stop_words='english')
    mtx = vec.fit_transform(df_res['blob'].tolist() + [query])
    df_res['relevance'] = cosine_similarity(mtx[-1], mtx[:-1])[0]
    return df_res[df_res['relevance'] > 0.01].sort_values('relevance', ascending=False)

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
                st.session_state.trigger_msg = f"Compare model {row['name']}"
                st.rerun()
    with c2:
        with st.popover("Specs"):
            st.write(f"**Specs: {row['name']}**")
            feats = SHAP_FEATURES.get(row['domain'], SHAP_FEATURES["Default"])
            shap_df = pd.DataFrame({'Feature': feats, 'Impact': [random.uniform(-0.5, 0.5) for _ in range(4)]}).sort_values('Impact')
            shap_df['Color'] = ['#EF4444' if x < 0 else '#6200EE' for x in shap_df['Impact']]
            fig = px.bar(shap_df, x='Impact', y='Feature', orientation='h', color='Color', color_discrete_map="identity", height=200)
            fig.update_layout(margin=dict(l=60,r=10,t=10,b=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    with c3:
        btn_label = "Approve" if user == "Nat Patel" and row['approval_status'] == 'Pending' else "Request"
        if st.button(btn_label, key=f"r_{prefix}_{row['name']}_{random.randint(0,999)}"):
            st.session_state.trigger_msg = f"{btn_label} model {row['name']}"
            st.rerun()

# --- COMPANION ENGINE ---
def run_companion():
    st.header("🤖 WorkBench Companion")
    
    if 'trigger_msg' in st.session_state:
        st.session_state.messages.append({"role": "user", "content": st.session_state.pop('trigger_msg')})

    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "df_data" in msg:
                rdf = pd.DataFrame(msg["df_data"])
                if not rdf.empty:
                    cols = st.columns(min(len(rdf), 3))
                    for idx, r in enumerate(rdf.head(3).to_dict('records')):
                        with cols[idx]: render_tile(r, user_role, f"ch_{i}", len(rdf))
            if "chart_data" in msg:
                st.plotly_chart(px.bar(pd.DataFrame(msg["chart_data"]), x='name', y=['accuracy','latency'], barmode='group'))
            if "pie_data" in msg:
                st.plotly_chart(px.pie(pd.DataFrame(msg["pie_data"]), values='revenue_impact', names='domain', hole=0.4))

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        with st.chat_message("assistant"):
            q = prompt.lower()
            res_txt, res_df, res_pie, res_bar = "Processing...", pd.DataFrame(), None, None
            
            # Context Logic
            found_dom = next((d for d in df_master['domain'].unique() if d.lower() in q), None)
            if found_dom: st.session_state.chat_context["domain"] = found_dom

            # Intent: Submission
            if "submit" in q or "register" in q or st.session_state.chat_context["intent"] == "SUB":
                st.session_state.chat_context["intent"] = "SUB"
                raw = call_gemini(f"Extract JSON (name, use_cases, domain, accuracy, latency, usage). Text: {prompt}", api_key)
                try:
                    ext = json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group())
                    for k,v in ext.items(): 
                        if v: st.session_state.chat_context["draft"][k] = v
                except: pass
                draft = st.session_state.chat_context["draft"]
                if not draft.get("name"): res_txt = "What is the **name** of the model?"
                elif not draft.get("use_cases"): res_txt = f"What is the **use case** for {draft['name']}?"
                else:
                    new_row = {**{f:"N/A" for f in MASTER_FIELDS}, **draft, "contributor":user_role, "approval_status":"Pending", "model_stage":"Prod"}
                    pd.DataFrame([new_row]).to_csv(REG_PATH, mode='a', header=False, index=False)
                    res_txt = f"✅ Model **{draft['name']}** registered and sent to Nat Patel."
                    st.session_state.chat_context = {"domain": None, "intent": None, "draft": {}}

            # NAT PATEL APPROVALS IN CHAT
            elif user_role == "Nat Patel" and ("approve" in q or "queue" in q or "need to approve" in q):
                pend = df_master[df_master['approval_status'] == 'Pending']
                if "approve" in q and "model" in q:
                    # Specific model approval
                    m_name = next((n for n in pend['name'] if n.lower() in q), None)
                    if m_name:
                        df_master.loc[df_master['name']==m_name, 'approval_status'] = 'Approved'
                        df_master.to_csv(REG_PATH, index=False)
                        res_txt = f"✅ Approved **{m_name}**."
                    else: res_txt = "Which model should I approve? Please specify the name."
                else:
                    # Show the queue
                    if not pend.empty:
                        res_txt = f"I found **{len(pend)}** models awaiting your approval:"
                        res_df = pend
                    else: res_txt = "Approval queue is currently empty."

            # Intent: ROI
            elif "revenue" in q or "impact" in q:
                agg = df_master.groupby('domain')['revenue_impact'].sum().reset_index()
                res_pie = agg.to_dict('records')
                res_txt = "Revenue contribution breakdown by domain:"
            
            # Intent: Compare
            elif "compare" in q:
                names = [n for n in df_master['name'].unique() if n.lower() in q or n.split('-')[-1] in q]
                if len(names) >= 2:
                    res_df = df_master[df_master['name'].isin(names)]
                    res_bar = res_df.to_dict('records')
                    res_txt = f"Comparing {len(names)} models."
                else: res_txt = "Specify exact model names to compare."

            else:
                res_df = hybrid_search(q, df_master).head(3)
                res_txt = call_gemini(prompt, api_key) if api_key else "I found these relevant assets:"

            st.markdown(res_txt)
            if res_pie: st.plotly_chart(px.pie(pd.DataFrame(res_pie), values='revenue_impact', names='domain', hole=0.4))
            if res_bar: st.plotly_chart(px.bar(pd.DataFrame(res_bar), x='name', y=['accuracy','latency'], barmode='group'))
            if not res_df.empty:
                cols = st.columns(min(len(res_df), 3))
                for i_idx, r in enumerate(res_df.head(3).to_dict('records')):
                    with cols[i_idx]: render_tile(r, user_role, f"at_{i_idx}", len(res_df), is_companion=True)
            st.session_state.messages.append({"role": "assistant", "content": res_txt, "df_data": res_df.to_dict('records'), "pie_data": res_pie, "chart_data": res_bar})

# --- NAVIGATION ---
with st.sidebar:
    st.title("Hub Controls")
    app_mode = st.toggle("Enable Web Mode", value=False)
    user_role = st.selectbox("Current User", ["John Doe", "Jane Nu", "Sam King", "Nat Patel"])
    api_key = st.text_input("Gemini API Key", type="password")

# --- ROUTING ---
if not app_mode:
    run_companion()
else:
    tabs = ["Model Gallery", "AI Business Value"]
    if user_role == "Nat Patel": tabs.append("Approval Portal")
    t = st.tabs(tabs)
    
    with t[0]:
        q = st.text_input("Search Models")
        res = hybrid_search(q, df_master)
        for i in range(0, min(len(res), 15), 3):
            cols = st.columns(3)
            for j in range(3):
                if i+j < len(res):
                    with cols[j]: render_tile(res.iloc[i+j], user_role, f"gal_{i+j}", len(res), is_companion=False)

    with t[1]:
        st.header("Strategic Portfolio ROI")
        src = df_master if user_role == "Nat Patel" else df_master[df_master['contributor'] == user_role]
        agg = src.groupby('domain').agg({'revenue_impact':'sum','accuracy':'mean'}).reset_index()
        k1, k2 = st.columns(2)
        k1.metric("Rev Impact", f"${agg['revenue_impact'].sum()/1e6:.1f}M")
        k2.metric("Avg Accuracy", f"{int(agg['accuracy'].mean()*100)}%")
        st.plotly_chart(px.pie(agg, values='revenue_impact', names='domain', hole=0.4))

    if user_role == "Nat Patel":
        with t[2]:
            st.header("Approval Portal")
            sel = st.selectbox("Solo Asset Focus", ["None"] + list(df_master['name'].unique()))
            cv = [1.0 if n == sel else 0.0 for n in df_master['name']]
            fig_p = go.Figure(data=go.Parcoords(
                labelfont=dict(size=10, color='black'), tickfont=dict(size=8),
                line=dict(color=cv, colorscale=[[0, '#FFF9C4'], [1, '#B71C1C']], showscale=False),
                dimensions=[dict(range=[0,25000], label='Usage', values=df_master['usage']),
                            dict(range=[0.7,1], label='Accuracy', values=df_master['accuracy']),
                            dict(range=[0,150], label='Latency', values=df_master['latency']),
                            dict(range=[0,0.3], label='Drift', values=df_master['data_drift'])]))
            st.plotly_chart(fig_p, use_container_width=True)
            st.divider()
            pend = df_master[df_master['approval_status'] == 'Pending']
            if not pend.empty:
                for idx, r in pend.iterrows():
                    ca, cb = st.columns([4, 1])
                    ca.write(f"💼 **{r['contributor']}** submitted **{r['name']}**")
                    if cb.button("Approve", key=f"ap_{idx}"):
                        df_master.at[idx, 'approval_status'] = 'Approved'
                        df_master.to_csv(REG_PATH, index=False); st.rerun()
            else: st.success("Queue clear.")
