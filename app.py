import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import os, datetime, random, re, csv, json

# --- CONFIG & STYLING ---
st.set_page_config(page_title="Model Hub 7.0", layout="wide")

st.markdown("""
    <style>
    :root { --accent: #6200EE; --lite-purple: #F3E5F5; --deep-purple: #7B1FA2; --pale-yellow: #FFF9C4; }
    .stApp { background-color: #F8FAFC; font-size: 0.82rem; }
    
    .model-card {
        background: white; border: 1px solid #e2e8f0; border-top: 3px solid var(--accent);
        padding: 8px; border-radius: 6px; min-height: 310px; 
        display: flex; flex-direction: column; justify-content: space-between;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .model-title { font-size: 0.85rem; font-weight: 700; color: #1e293b; margin-bottom: 2px; }
    .metric-bar { display: flex; justify-content: space-between; background: #F1F5F9; padding: 4px; border-radius: 4px; margin-bottom: 5px; }
    .metric-val { font-size: 0.72rem; font-weight: 700; color: var(--accent); text-align: center; flex: 1; }
    .metric-label { font-size: 0.5rem; color: #94a3b8; display: block; text-transform: uppercase; }

    /* Micro Lite Purple Buttons */
    .stButton>button { 
        background-color: var(--lite-purple); color: var(--deep-purple); 
        border: 1px solid var(--deep-purple); border-radius: 4px; 
        font-size: 0.6rem; height: 20px; padding: 0 2px; width: 100%;
        line-height: 1.2;
    }
    .stButton>button:hover { background-color: var(--deep-purple); color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- SCHEMA ---
MASTER_COLUMNS = [
    "name", "domain", "type", "accuracy", "latency", "clients", "use_cases", 
    "contributor", "usage", "data_drift", "revenue_impact", "risk_exposure", 
    "approval_status", "model_stage", "model_version", "model_owner_team",
    "cpu_util", "error_rate", "throughput"
]
REG_PATH = "model_registry_v3.csv"
SHAP_FEATURES = {
    "Finance": ["Credit_Score", "Income", "Debt", "History"],
    "Healthcare": ["Age", "BMI", "Blood_Pressure", "Glucose"],
    "Risk": ["Exposure", "Volatility", "Compliance", "Liquidity"],
    "Supply Chain": ["Lead_Time", "Inventory", "Distance", "Demand"],
    "Default": ["F1", "F2", "F3", "F4"]
}

# --- DATA ENGINE ---
def load_sanitized_data():
    if not os.path.exists(REG_PATH): return pd.DataFrame(columns=MASTER_COLUMNS)
    df = pd.read_csv(REG_PATH)
    # Check for missing columns
    for col in MASTER_COLUMNS:
        if col not in df.columns: df[col] = 0.0 if col in ["accuracy","usage","revenue_impact"] else "N/A"
    # Ensure numeric types
    nums = ["accuracy", "latency", "usage", "data_drift", "revenue_impact", "risk_exposure"]
    for c in nums:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce').fillna(0.0)
    if df['revenue_impact'].sum() == 0:
        df['revenue_impact'] = df['usage'] * 75.0
    return df

df_master = load_sanitized_data()

# --- SEARCH ENGINE ---
def hybrid_search(query, df):
    if not query or df.empty: return df
    q = query.lower()
    df_res = df.copy().fillna('N/A')
    pats = {'latency': r'latency\s*([<>]=?)\s*(\d+)', 'accuracy': r'accuracy\s*([<>]=?)\s*(\d+)'}
    for key, pat in pats.items():
        match = re.search(pat, q)
        if match:
            op, val = match.groups()
            val = float(val)/100 if (key=='accuracy' and float(val)>1) else float(val)
            if '>' in op: df_res = df_res[df_res[key] >= val]
            else: df_res = df_res[df_res[key] <= val]
    if df_res.empty: return df_res
    df_res['blob'] = df_res.astype(str).apply(' '.join, axis=1)
    vec = TfidfVectorizer(stop_words='english')
    mtx = vec.fit_transform(df_res['blob'].tolist() + [query])
    df_res['relevance'] = cosine_similarity(mtx[-1], mtx[:-1])[0]
    return df_res[df_res['relevance'] > 0.01].sort_values('relevance', ascending=False)

# --- GEMINI INTEGRATION ---
def call_gemini(prompt, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        return model.generate_content(prompt).text
    except Exception as e: return f"AI Logic Active (Offline). Details: {str(e)}"

# --- UI COMPONENTS ---
def render_tile(row, key_prefix, show_usage=False):
    val_label = "Usage" if show_usage else "Value"
    val_display = f"{int(row['usage'])}" if show_usage else f"${row['revenue_impact']/1000:.1f}k"
    
    st.markdown(f"""
    <div class="model-card">
        <div>
            <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                <span style="font-size:0.55rem; color:#666;">v{row.get('model_version','1.0')}</span>
                <span style="color:#2E7D32; font-size:0.6rem;">● {row['model_stage']}</span>
            </div>
            <div class="model-title">{row['name']}</div>
            <div style="font-size:0.65rem; color:gray;"><b>{row['domain']}</b> | {row['contributor']}</div>
            <div class="use-case-text">{row['use_cases']}</div>
        </div>
        <div class="metric-bar">
            <div class="metric-val"><span class="metric-label">Acc</span>{int(row['accuracy']*100)}%</div>
            <div class="metric-val"><span class="metric-label">Lat</span>{int(row['latency'])}ms</div>
            <div class="metric-val"><span class="metric-label">{val_label}</span>{val_display}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Compare", key=f"c_{key_prefix}_{row['name']}_{random.randint(0,999)}"):
            if row['name'] not in st.session_state.basket: st.session_state.basket.append(row['name'])
    with c2:
        with st.popover("Specs"):
            st.write(f"**SHAP Summary: {row['name']}**")
            feats = SHAP_FEATURES.get(row['domain'], SHAP_FEATURES["Default"])
            shap_df = pd.DataFrame({'Feature': feats, 'Impact': [random.uniform(-0.5, 0.5) for _ in range(4)]}).sort_values('Impact')
            shap_df['Color'] = ['#EF4444' if x < 0 else '#6200EE' for x in shap_df['Impact']]
            fig = px.bar(shap_df, x='Impact', y='Feature', orientation='h', color='Color', color_discrete_map="identity", height=150)
            fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True, key=f"sh_{key_prefix}_{row['name']}_{random.randint(0,999)}")
    with c3:
        if st.button("Request", key=f"r_{key_prefix}_{row['name']}_{random.randint(0,999)}"):
            st.toast("Access Request Logged")

# --- NAVIGATION ---
with st.sidebar:
    st.title("Hub Control")
    api_key = st.text_input("Gemini API Key", type="password")
    nav = st.radio("Go to", ["WorkBench Companion", "Model Gallery", "AI Business Value", "Approval"])
    role = st.selectbox("Role", ["John Doe", "Jane Nu", "Sam King", "Nat Patel", "Admin"])
    if 'basket' not in st.session_state: st.session_state.basket = []

# --- 1. WORKBENCH COMPANION ---
if nav == "WorkBench Companion":
    st.header("🤖 WorkBench Companion")
    if "messages" not in st.session_state: st.session_state.messages = []
    if "context_domain" not in st.session_state: st.session_state.context_domain = None

    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "df" in msg:
                cols = st.columns(min(len(msg["df"]), 3))
                for idx, r in enumerate(msg["df"].head(3).to_dict('records')):
                    with cols[idx]: render_tile(r, f"chat_h_{i}_{idx}", show_usage=msg.get("is_trending", False))
            if "chart_data" in msg:
                c_df = pd.DataFrame(msg["chart_data"])
                st.table(c_df[['name','accuracy','latency','data_drift','usage']])
                st.plotly_chart(px.bar(c_df, x='name', y=['accuracy','latency'], barmode='group'), key=f"hist_chart_{i}")

    if prompt := st.chat_input("How can I assist?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            q = prompt.lower()
            res_txt, res_df, res_data, is_trending = "", pd.DataFrame(), None, False

            # DETECT DOMAIN SWITCH
            found_domain = next((d for d in df_master['domain'].unique() if d.lower() in q), None)
            if found_domain: st.session_state.context_domain = found_domain

            # INTENT: COMPARE
            if "compare" in q:
                names = [n for n in df_master['name'].unique() if n.lower() in q]
                if len(names) >= 2:
                    res_df = df_master[df_master['name'].isin(names)]
                    res_data = res_df.to_dict('records')
                    res_txt = f"Bechmarking **{', '.join(names)}**..."
                    st.table(res_df[['name','accuracy','latency','usage','data_drift']])
                    st.plotly_chart(px.bar(res_df, x='name', y=['accuracy','latency'], barmode='group'))
                else: res_txt = "Please provide exact model names to compare (e.g. Fin-Model-091)."

            # INTENT: SUBMISSION
            elif any(x in q for x in ["submit", "register"]):
                raw_json = call_gemini(f"Extract JSON (name, domain, accuracy(0-1), latency(int)) from: {prompt}. Return ONLY raw JSON.", api_key)
                try:
                    data = json.loads(re.search(r"\{.*\}", raw_json, re.DOTALL).group())
                    new_row = pd.DataFrame([{**{c:"N/A" for c in MASTER_COLUMNS}, **data, "contributor":role, "approval_status":"Pending", "model_stage":"Prod", "revenue_impact": 5000}])
                    pd.concat([df_master, new_row]).to_csv(REG_PATH, index=False)
                    res_txt = f"✅ Registered **{data.get('name')}** and sent to **Nat Patel** for approval."
                except: res_txt = "Parsing failed. Provide Name, Domain, Accuracy, and Latency."

            # INTENT: REVENUE / TOP CONTRIBUTORS
            elif "revenue" in q or "impact" in q or "contributor" in q:
                dom = st.session_state.context_domain
                if dom:
                    subset = df_master[df_master['domain'] == dom]
                    val = subset['revenue_impact'].sum()
                    if "top" in q or "contributor" in q:
                        res_df = subset.nlargest(3, 'revenue_impact')
                        res_txt = f"Here are the top model contributors for the **{dom}** domain:"
                    else:
                        res_txt = f"The **{dom}** domain has generated an impact of **${val/1e6:.2f}M**."
                else: res_txt = "Which domain are you asking about (e.g. Finance, HR)?"

            # INTENT: TRENDING / GEMS
            elif "trending" in q or "popular" in q:
                res_df = df_master.nlargest(3, 'usage')
                res_txt = "Most popular assets right now:"
                is_trending = True
            elif "gems" in q or "hidden" in q:
                res_df = df_master[(df_master['accuracy'] > 0.96) & (df_master['usage'] < 1000)].head(3)
                res_txt = "High performance, low adoption assets:"

            # DEFAULT: SEARCH
            else:
                res_df = hybrid_search(q, df_master).head(3)
                res_txt = call_gemini(prompt, api_key) if api_key else "I found these relevant assets:"

            st.markdown(res_txt)
            if not res_df.empty and len(res_df) < len(df_master):
                cols = st.columns(min(len(res_df), 3))
                for i, r in enumerate(res_df.head(3).to_dict('records')):
                    with cols[i]: render_tile(r, f"chat_new_{i}", show_usage=is_trending)
            
            st.session_state.messages.append({"role": "assistant", "content": res_txt, "df": res_df, "chart_data": res_data, "is_trending": is_trending})

# --- 2. MODEL GALLERY ---
elif nav == "Model Gallery":
    t1, t2 = st.tabs(["🏛 Inventory", "⚖️ Comparison Basket"])
    with t1:
        q_gal = st.text_input("Smart Search")
        res = hybrid_search(q_gal, df_master)
        for i in range(0, min(len(res), 15), 3):
            cols = st.columns(3)
            for j in range(3):
                if i+j < len(res):
                    with cols[j]: render_tile(res.iloc[i+j], f"gal_{i+j}")
    with t2:
        if not st.session_state.basket: st.info("Basket empty.")
        else:
            comp_df = df_master[df_master['name'].isin(st.session_state.basket)]
            st.dataframe(comp_df[['name','accuracy','latency','usage','data_drift']])
            st.plotly_chart(px.bar(comp_df, x='name', y=['accuracy','latency','data_drift'], barmode='group'))

# --- 3. AI BUSINESS VALUE ---
elif nav == "AI Business Value":
    st.header("Executive Strategic ROI")
    agg = df_master.groupby('domain').agg({'revenue_impact':'sum','usage':'sum','accuracy':'mean'}).reset_index()
    k1, k2 = st.columns(2)
    k1.metric("Revenue Contribution", f"${agg['revenue_impact'].sum()/1e6:.1f}M")
    k2.metric("Avg Portfolio Accuracy", f"{int(agg['accuracy'].mean()*100)}%")
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(px.pie(agg, values='revenue_impact', names='domain', hole=0.4))
    with c2: st.plotly_chart(px.bar(agg, x='domain', y='revenue_impact', color='accuracy'))

# --- 4. APPROVAL / ADMIN ---
elif nav == "Approval":
    if role == "Nat Patel" or role == "Admin":
        st.subheader("Leader Approval Queue")
        pend = df_master[df_master['approval_status'] == 'Pending']
        if not pend.empty:
            st.table(pend[['name','domain','contributor']])
            if st.button("Approve All"):
                df_master.loc[df_master['approval_status']=='Pending','approval_status']='Approved'
                df_master.to_csv(REG_PATH, index=False); st.rerun()
        else: st.success("Queue Clear.")
        st.divider()
        sel = st.selectbox("Solo Focus Line", ["None"] + list(df_master['name'].unique()))
        cv = [1.0 if n == sel else 0.0 for n in df_master['name']]
        fig_p = go.Figure(data=go.Parcoords(
            labelfont=dict(size=10, color='black'), tickfont=dict(size=8),
            line=dict(color=cv, colorscale=[[0, '#FFF9C4'], [1, '#B71C1C']], showscale=False),
            dimensions=[dict(range=[0.7,1], label='Accuracy', values=df_master['accuracy']),
                        dict(range=[0,150], label='Latency', values=df_master['latency']),
                        dict(range=[0,25000], label='Usage', values=df_master['usage'])]))
        fig_p.update_layout(margin=dict(t=80, b=40, l=80, r=80), height=500)
        st.plotly_chart(fig_p, use_container_width=True)
