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
        padding: 8px 10px; border-radius: 6px; min-height: 250px; 
        display: flex; flex-direction: column; justify-content: space-between;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    .model-title { font-size: 0.82rem; font-weight: 700; color: #1e293b; margin-bottom: 1px; }
    .tag-row { display: flex; justify-content: space-between; margin-bottom: 2px; }
    .owner-tag { font-size: 0.55rem; font-weight: bold; color: var(--deep-p); text-transform: uppercase; }
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

# --- CONSTANTS ---
REG_PATH = "model_registry_v3.csv"
REQ_PATH = "requests_v3.csv"
SHAP_FEATURES = {
    "Finance": ["Credit_Score", "Income", "Debt", "Trans_Vol"],
    "Healthcare": ["Age", "BMI", "BP", "Glucose"],
    "Risk": ["Exposure", "Volatility", "Compliance", "Market_Gap"],
    "Supply Chain": ["Lead_Time", "Inventory", "Distance", "Demand"],
    "Default": ["Feature_A", "Feature_B", "Feature_C", "Feature_D"]
}

# --- DATA ENGINE ---
def load_data():
    if not os.path.exists(REG_PATH): return pd.DataFrame()
    df = pd.read_csv(REG_PATH)
    nums = ['usage', 'accuracy', 'latency', 'data_drift', 'revenue_impact', 'risk_exposure', 'cpu_util']
    for c in nums:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce').fillna(0.0)
    if 'revenue_impact' not in df.columns or df['revenue_impact'].sum() == 0:
        df['revenue_impact'] = df['usage'] * 65.0
    return df

def load_reqs():
    if not os.path.exists(REQ_PATH): return pd.DataFrame(columns=["model_name", "requester", "status", "timestamp"])
    return pd.read_csv(REQ_PATH)

df_master = load_data()
df_reqs = load_reqs()

# --- INITIALIZE STATE ---
if 'messages' not in st.session_state: st.session_state.messages = []
if 'basket' not in st.session_state: st.session_state.basket = []
if 'active_context' not in st.session_state: st.session_state.active_context = {"domain": None, "models": []}

# --- LOGIC ENGINES ---
def call_gemini(prompt, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        return model.generate_content(prompt).text
    except Exception as e: return f"Error: {str(e)}"

def hybrid_search(query, df):
    if not query or df.empty: return df
    q = query.lower()
    # Math logic
    pats = {'latency': r'latency\s*([<>]=?)\s*(\d+)', 'accuracy': r'accuracy\s*([<>]=?)\s*(\d+)'}
    for k, p in pats.items():
        match = re.search(p, q)
        if match:
            op, val = match.groups()
            val = float(val)/100 if (k=='accuracy' and float(val)>1) else float(val)
            if '>' in op: df = df[df[k] >= val]
            else: df = df[df[k] <= val]
    if df.empty: return df
    # Textual
    df['blob'] = df.fillna('').astype(str).apply(' '.join, axis=1)
    vec = TfidfVectorizer(stop_words='english')
    mtx = vec.fit_transform(df['blob'].tolist() + [query])
    df['score'] = cosine_similarity(mtx[-1], mtx[:-1])[0]
    return df[df['score'] > 0.01].sort_values('score', ascending=False)

# --- UI COMPONENTS ---
def render_tile(row, user, prefix):
    # Determine Label
    label = ""
    if row['contributor'] == user: label = "Owned"
    elif not df_reqs[(df_reqs['model_name']==row['name']) & (df_reqs['requester']==user) & (df_reqs['status']=='Approved')].empty: label = "Licensed"

    st.markdown(f"""
    <div class="model-card">
        <div>
            <div class="tag-row"><span class="owner-tag">{label}</span><span style="font-size:0.5rem; color:gray;">v{row.get('model_version','1.0')}</span></div>
            <div class="model-title">{row['name'][:22]}</div>
            <div style="font-size:0.6rem; color:gray;"><b>{row['domain']}</b> | {row['contributor']}</div>
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
    if c1.button("Compare", key=f"c_{prefix}_{row['name']}"):
        if len(st.session_state.basket) < 5:
            st.session_state.basket.append(row['name']); st.rerun()
        else: st.error("Basket full (Max 5)")
    with c2:
        with st.popover("Specs"):
            feats = SHAP_FEATURES.get(row['domain'], SHAP_FEATURES["Default"])
            shap_df = pd.DataFrame({'Feature': feats, 'Impact': [random.uniform(-0.5, 0.5) for _ in range(4)]}).sort_values('Impact')
            shap_df['Color'] = ['#EF4444' if x < 0 else '#6200EE' for x in shap_df['Impact']]
            st.plotly_chart(px.bar(shap_df, x='Impact', y='Feature', orientation='h', color='Color', color_discrete_map="identity", height=150), use_container_width=True)
    if c3.button("Request", key=f"r_{prefix}_{row['name']}"):
        new_r = pd.DataFrame([{"model_name":row['name'], "requester":user, "status":"Pending", "timestamp":str(datetime.datetime.now())}])
        new_r.to_csv(REQ_PATH, mode='a', header=not os.path.exists(REQ_PATH), index=False); st.toast("Sent to Nat Patel")

# --- NAVIGATION ---
with st.sidebar:
    st.title("Hub Controls")
    api_key = st.text_input("Gemini API Key", type="password")
    role = st.selectbox("Current User", ["John Doe", "Jane Nu", "Sam King", "Nat Patel"])
    nav = ["WorkBench Companion", "Model Gallery", "AI Business Value"]
    if role == "Nat Patel": nav.append("Approval")
    view = st.radio("Go to", nav)

# --- 1. WORKBENCH COMPANION ---
if view == "WorkBench Companion":
    st.header("🤖 WorkBench Companion")
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "df" in msg and not msg["df"].empty:
                cols = st.columns(min(len(msg["df"]), 3))
                for idx, r in enumerate(msg["df"].head(3).to_dict('records')):
                    with cols[idx]: render_tile(r, role, f"chat_{i}_{idx}")
            if "pie" in msg: st.plotly_chart(msg["pie"], key=f"pie_{i}")
            if "bar" in msg: st.plotly_chart(msg["bar"], key=f"bar_{i}")

    if prompt := st.chat_input("E.g. 'Compare Fin-Model-016 and 023'"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            q = prompt.lower()
            res_txt, res_df, res_pie, res_bar = "", pd.DataFrame(), None, None
            
            # Detect Context Change
            found_dom = next((d for d in df_master['domain'].unique() if d.lower() in q), None)
            if found_dom: st.session_state.active_context["domain"] = found_dom

            # LOGIC 1: LEADER APPROVALS
            if role == "Nat Patel" and "queue" in q:
                p_reqs = df_reqs[df_reqs['status'] == "Pending"]
                res_txt = f"You have {len(p_reqs)} pending requests."
                res_df = p_reqs # In chat, we show the names
            elif role == "Nat Patel" and "approve" in q:
                df_reqs['status'] = 'Approved'
                df_reqs.to_csv(REQ_PATH, index=False)
                res_txt = "✅ All pending requests have been approved."

            # LOGIC 2: ROI / TRENDING
            elif "revenue" in q or "impact" in q:
                if st.session_state.active_context["domain"]:
                    d = st.session_state.active_context["domain"]
                    val = df_master[df_master['domain']==d]['revenue_impact'].sum()
                    res_txt = f"The **{d}** domain has contributed **${val/1e6:.2f}M** impact."
                    if "top" in q: res_df = df_master[df_master['domain']==d].nlargest(3, 'revenue_impact')
                else:
                    agg = df_master.groupby('domain')['revenue_impact'].sum().reset_index()
                    res_pie = px.pie(agg, values='revenue_impact', names='domain', hole=0.4)
                    res_txt = "Here is the revenue contribution by domain:"

            # LOGIC 3: COMPARE
            elif "compare" in q or "difference" in q:
                names = [n for n in df_master['name'].unique() if n.lower() in q or n.split('-')[-1] in q]
                if len(names) >= 2:
                    res_df = df_master[df_master['name'].isin(names)]
                    st.table(res_df[['name','usage','accuracy','latency','data_drift']])
                    res_bar = px.bar(res_df, x='name', y=['accuracy','latency','usage'], barmode='group')
                    res_txt = f"Comparing {', '.join(names)}."
                else: res_txt = "No specific models found for comparison."

            # LOGIC 4: TRENDING / GEMS
            elif "trending" in q:
                d = st.session_state.active_context["domain"]
                res_df = df_master[df_master['domain']==d].nlargest(3, 'usage') if d else df_master.nlargest(3, 'usage')
                res_txt = f"Trending models {'in ' + d if d else 'globally'}:"

            else:
                res_df = hybrid_search(q, df_master).head(3)
                if res_df.empty: res_txt = "No models with that criteria. Search for some other model."
                else: res_txt = call_gemini(prompt, api_key) if api_key else "I found these relevant assets:"

            st.markdown(res_txt)
            if res_pie: st.plotly_chart(res_pie)
            if res_bar: st.plotly_chart(res_bar)
            if not res_df.empty and "queue" not in q:
                cols = st.columns(min(len(res_df), 3))
                for i_idx, r in enumerate(res_df.head(3).to_dict('records')):
                    with cols[i_idx]: render_tile(r, role, f"chat_n_{i_idx}")
            
            st.session_state.messages.append({"role":"assistant","content":res_txt,"df":res_df,"pie":res_pie,"bar":res_bar})

# --- 2. MODEL GALLERY ---
elif view == "Model Gallery":
    t1, t2 = st.tabs(["🏛 Inventory", "⚖️ Benchmark Basket"])
    with t1:
        if st.session_state.basket: st.info(f"Basket ({len(st.session_state.basket)}/5): {', '.join(st.session_state.basket)}")
        q_gal = st.text_input("Search (Domain, Accuracy, Latency, etc.)")
        res = hybrid_search(q_gal, df_master)
        for i in range(0, min(len(res), 15), 3):
            cols = st.columns(3)
            for j in range(3):
                if i+j < len(res):
                    with cols[j]: render_tile(res.iloc[i+j], role, f"gal_{i+j}")
    with t2:
        if not st.session_state.basket: st.info("Basket empty.")
        else:
            c_df = df_master[df_master['name'].isin(st.session_state.basket)]
            st.table(c_df[['name','accuracy','latency','usage','data_drift']])
            st.plotly_chart(px.bar(c_df, x='name', y=['accuracy','usage'], barmode='group'))
            if st.button("Clear Basket"): st.session_state.basket = []; st.rerun()

# --- 3. AI BUSINESS VALUE ---
elif view == "AI Business Value":
    st.header("Domain Strategy ROI")
    source_df = df_master if role == "Nat Patel" else df_master[df_master['contributor'] == role]
    agg = source_df.groupby('domain').agg({'revenue_impact':'sum','usage':'sum','accuracy':'mean'}).reset_index()
    
    if agg.empty: st.info("No data available for your profile contributions.")
    else:
        k1, k2 = st.columns(2)
        k1.metric("Contribution", f"${agg['revenue_impact'].sum()/1e6:.2f}M")
        k2.metric("Avg Accuracy", f"{int(agg['accuracy'].mean()*100)}%")
        
        sel_dom = st.selectbox("Drill-down (Domain Selection)", agg['domain'].unique())
        st.table(source_df[source_df['domain']==sel_dom].nlargest(5,'revenue_impact')[['name','usage','revenue_impact']])
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(px.pie(agg, values='revenue_impact', names='domain', hole=0.4))
        with c2: st.plotly_chart(px.bar(agg, x='domain', y='revenue_impact'))

# --- 4. APPROVAL ---
elif view == "Approval":
    st.header("Governance Telemetry")
    sel = st.selectbox("Highlight Solo Asset", ["None"] + list(df_master['name'].unique()))
    cv = [1.0 if n == sel else 0.0 for n in df_master['name']]
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
    st.subheader("Leader Approval Queue")
    pend = df_reqs[df_reqs['status'] == 'Pending']
    if not pend.empty:
        for idx, r in pend.iterrows():
            c_a, c_b = st.columns([4, 1])
            c_a.write(f"💼 **{r['requester']}** requested **{r['model_name']}**")
            if c_b.button("Approve", key=f"ap_{idx}"):
                df_reqs.at[idx, 'status'] = 'Approved'
                df_reqs.to_csv(REQ_PATH, index=False); st.rerun()
    else: st.success("Queue Clear.")
