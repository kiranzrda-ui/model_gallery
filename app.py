import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import os, datetime, random, re, csv, json

# --- CONFIG & STYLING ---
st.set_page_config(page_title="Model Hub 17.0", layout="wide")

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
        border-radius: 4px; font-size: 0.58rem; height: 18px; padding: 0 2px; width: 100%;
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
    "Retail": ["Footfall", "Seasonality", "Stock_Level", "Promo_Impact"],
    "Supply Chain": ["Lead_Time", "Inventory", "Route", "Demand"],
    "Default": ["Feature_1", "Feature_2", "Feature_3", "Feature_4"]
}

# --- DATA ENGINE ---
def load_and_fix_data():
    if not os.path.exists(REG_PATH): return pd.DataFrame(columns=MASTER_FIELDS)
    df = pd.read_csv(REG_PATH)
    
    # Shield logic: Add missing columns before math
    for col in MASTER_FIELDS:
        if col not in df.columns:
            df[col] = 0.0 if col in ["usage", "accuracy", "latency", "revenue_impact", "data_drift", "risk_exposure"] else "N/A"
    
    # Numeric sanitization
    nums = ["usage", "accuracy", "latency", "revenue_impact", "data_drift", "risk_exposure"]
    for c in nums:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce').fillna(0.0)
    
    if df['revenue_impact'].sum() == 0: 
        df['revenue_impact'] = df['usage'] * 125.0
    if df['accuracy'].sum() == 0:
        df['accuracy'] = [random.uniform(0.75, 0.98) for _ in range(len(df))]
    
    return df

def load_reqs():
    if not os.path.exists(REQ_PATH): return pd.DataFrame(columns=["model_name", "requester", "status", "timestamp"])
    return pd.read_csv(REQ_PATH)

df_master = load_and_fix_data()
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
    except: return "AI Engine Offline. Processing based on local hub rules."

def hybrid_search(query, df):
    if not query or df.empty: return df
    q = query.lower()
    df_res = df.copy().fillna('N/A')
    if "official" in q: df_res = df_res[df_res['type'].str.lower() == 'official']
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
            <div style="font-size:0.65rem; color:#444; height:2.4em; overflow:hidden; margin:3px 0;">{row['use_cases']}</div>
        </div>
        <div class="metric-bar">
            <div class="metric-val"><span class="metric-label">Acc</span>{int(row['accuracy']*100)}%</div>
            <div class="metric-val"><span class="metric-label">Lat</span>{int(row['latency'])}ms</div>
            <div class="metric-val"><span class="metric-label">Impact</span>${row['revenue_impact']/1000:.1f}k</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        if total_results > 1:
            if st.button("Compare", key=f"c_{prefix}_{row['name']}"):
                if is_companion: st.session_state.chat_trigger = f"Compare model {row['name']}"
                else: 
                    if len(st.session_state.basket) < 5: st.session_state.basket.append(row['name'])
                    st.toast(f"Added {row['name']} to basket.")
    with c2:
        with st.popover("Specs"):
            st.write(f"**Technical Detail: {row['name']}**")
            feats = SHAP_FEATURES.get(row['domain'], SHAP_FEATURES["Default"])
            shap_df = pd.DataFrame({'Feature': feats, 'Impact': [random.uniform(-0.5, 0.5) for _ in range(4)]}).sort_values('Impact')
            shap_df['Color'] = ['#EF4444' if x < 0 else '#6200EE' for x in shap_df['Impact']]
            fig = px.bar(shap_df, x='Impact', y='Feature', orientation='h', color='Color', color_discrete_map="identity", height=200)
            fig.update_layout(margin=dict(l=60,r=10,t=10,b=10), showlegend=False, xaxis_title="SHAP Impact")
            st.plotly_chart(fig, use_container_width=True)
    with c3:
        if st.button("Request", key=f"r_{prefix}_{row['name']}"):
            if is_companion: st.session_state.chat_trigger = f"I want to request access for {row['name']}"
            else:
                new_r = pd.DataFrame([{"model_name":row['name'], "requester":user, "status":"Pending", "timestamp":str(datetime.datetime.now())}])
                new_r.to_csv(REQ_PATH, mode='a', header=not os.path.exists(REQ_PATH), index=False)
                st.toast("Request sent to Nat Patel")

# --- NAVIGATION ---
with st.sidebar:
    st.title("Hub Controls")
    app_mode = st.toggle("Enable Web Mode", value=False)
    user_role = st.selectbox("Current User", ["John Doe", "Jane Nu", "Sam King", "Nat Patel"])
    api_key = st.text_input("Gemini API Key", type="password")
    if st.session_state.basket:
        st.info(f"Basket: {len(st.session_state.basket)}/5 items")
        if st.button("Clear Basket"): st.session_state.basket = []; st.rerun()

# --- 1. COMPANION MODE ---
if not app_mode:
    st.header("🤖 WorkBench Companion")
    if 'chat_trigger' in st.session_state:
        st.session_state.messages.append({"role": "user", "content": st.session_state.pop('chat_trigger')})

    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "df" in msg and not msg["df"].empty:
                num_cols = min(len(msg["df"]), 3)
                if num_cols > 0:
                    cols = st.columns(num_cols)
                    for idx, r in enumerate(msg["df"].head(3).to_dict('records')):
                        with cols[idx]: render_tile(r, user_role, f"ch_{i}", len(msg["df"]), is_companion=True)
            if "pie" in msg: st.plotly_chart(msg["pie"], key=f"chat_pie_{i}")

    if prompt := st.chat_input("Prompt: 'Show revenue impact', 'Compare models', 'Register model'"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            q = prompt.lower()
            res_txt, res_df, res_pie = "Analyzing...", pd.DataFrame(), None
            
            # Contextual Memory
            found_dom = next((d for d in df_master['domain'].unique() if d.lower() in q), st.session_state.context.get("domain"))
            if found_dom: st.session_state.context["domain"] = found_dom

            # LOGIC: SUBMISSION 1.7
            if "submit" in q or "register" in q or st.session_state.context["intent"] == "SUB":
                st.session_state.context["intent"] = "SUB"
                raw = call_gemini(f"Extract Model JSON (name, use_cases). Text: {prompt}", api_key)
                try:
                    ext = json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group())
                    for k,v in ext.items(): 
                        if v: st.session_state.context["draft"][k] = v
                except: pass
                
                draft = st.session_state.context["draft"]
                if not draft.get("name"): res_txt = "Sure! **What is the name of the model?**"
                elif not draft.get("use_cases"): res_txt = "Noted. **What is the use case for this asset?**"
                else:
                    new_row = {**{f:"N/A" for f in MASTER_FIELDS}, **draft, "contributor":user_role, "approval_status":"Pending", "accuracy":0.85, "revenue_impact": 5000}
                    pd.DataFrame([new_row]).to_csv(REG_PATH, mode='a', header=False, index=False)
                    res_txt = f"✅ Model **{draft['name']}** registered and sent to Nat Patel."
                    st.session_state.context = {"domain": None, "intent": None, "draft": {}}
            
            # LOGIC: REVENUE IMPACT
            elif "revenue" in q or "impact" in q:
                if st.session_state.context["domain"]:
                    d = st.session_state.context["domain"]
                    val = df_master[df_master['domain'] == d]['revenue_impact'].sum()
                    res_txt = f"The **{d}** domain impact is **${val/1e6:.2f}M**."
                    if "top" in q: res_df = df_master[df_master['domain']==d].nlargest(3, 'revenue_impact')
                else:
                    agg = df_master.groupby('domain')['revenue_impact'].sum().reset_index()
                    res_pie = px.pie(agg, values='revenue_impact', names='domain', hole=0.4)
                    res_txt = "Here is the revenue contribution breakdown by domain:"

            # LOGIC: COMPARE
            elif "compare" in q or "difference" in q:
                names = [n for n in df_master['name'].unique() if n.lower() in q]
                if len(names) >= 2:
                    res_df = df_master[df_master['name'].isin(names)]
                    res_txt = f"Comparing performance for {', '.join(names)}."
                else: res_txt = "Please specify exact model names to compare."

            else:
                res_df = hybrid_search(q, df_master).head(3)
                res_txt = call_gemini(prompt, api_key) if api_key else "Relevant models:"
            
            st.markdown(res_txt)
            if res_pie: st.plotly_chart(res_pie)
            if not res_df.empty:
                num_cols = min(len(res_df), 3)
                cols = st.columns(num_cols)
                for idx, r in enumerate(res_df.to_dict('records')):
                    with cols[idx]: render_tile(r, user_role, f"at_{idx}", len(res_df), is_companion=True)
            st.session_state.messages.append({"role": "assistant", "content": res_txt, "df": res_df, "pie": res_pie})

# --- 2. WEB MODE ---
else:
    tabs = ["Model Gallery", "AI Business Value"]
    if st.session_state.basket: tabs.insert(1, "Benchmark Basket")
    if user_role == "Nat Patel": tabs.append("Approval Portal")
    t = st.tabs(tabs)
    
    with t[0]:
        st.subheader("🏛 Model Inventory")
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
            st.header("⚖️ Benchmark Analysis")
            c_df = df_master[df_master['name'].isin(st.session_state.basket)]
            st.table(c_df[['name','accuracy','latency','usage','data_drift']])
            st.plotly_chart(px.bar(c_df, x='name', y=['accuracy','data_drift'], barmode='group'))
        current_idx += 1

    with t[current_idx]:
        st.header("Strategic Portfolio ROI")
        src = df_master if user_role == "Nat Patel" else df_master[df_master['contributor'] == user_role]
        agg = src.groupby('domain').agg({'revenue_impact':'sum','accuracy':'mean','data_drift':'mean'}).reset_index()
        
        k1, k2, k3 = st.columns(3)
        k1.metric("Revenue Contribution", f"${agg['revenue_impact'].sum()/1e6:.1f}M")
        k2.metric("Avg Portfolio Accuracy", f"{int(agg['accuracy'].mean()*100)}%")
        k3.metric("Avg Data Drift", f"{agg['data_drift'].mean():.3f}")
        
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(px.pie(agg, values='revenue_impact', names='domain', hole=0.4))
        with c2: st.plotly_chart(px.bar(agg, x='domain', y='revenue_impact', title="Impact by Domain"))

    if user_role == "Nat Patel":
        with t[-1]:
            st.header("Governance Telemetry")
            sel = st.selectbox("Highlight Line", ["None"] + list(df_master['name'].unique()))
            cv = [1.0 if n == sel else 0.0 for n in df_master['name']]
            fig = go.Figure(data=go.Parcoords(
                labelfont=dict(size=10, color='black'), tickfont=dict(size=8),
                line=dict(color=cv, colorscale=[[0, '#FFF9C4'], [1, '#B71C1C']], showscale=False),
                dimensions=[dict(range=[0,25000], label='Usage', values=df_master['usage']),
                            dict(range=[0.7,1], label='Accuracy', values=df_master['accuracy']),
                            dict(range=[0,150], label='Latency', values=df_master['latency']),
                            dict(range=[0,0.3], label='Drift', values=df_master['data_drift']),
                            dict(range=[0,100], label='CPU %', values=df_master['cpu_util'])]))
            st.plotly_chart(fig, use_container_width=True)
            
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
