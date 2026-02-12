import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import os, datetime, random, re, csv, json

# --- CONFIG & STYLING ---
st.set_page_config(page_title="Enterprise Model Hub 9.0", layout="wide")

st.markdown("""
    <style>
    :root { --accent: #6200EE; --lite-purple: #F3E5F5; --deep-purple: #7B1FA2; --pale-yellow: #FFF9C4; }
    .stApp { background-color: #F8FAFC; font-size: 0.8rem; }
    
    .model-card {
        background: white; border: 1px solid #e2e8f0; border-top: 3px solid var(--accent);
        padding: 8px; border-radius: 6px; min-height: 270px; 
        display: flex; flex-direction: column; justify-content: space-between;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    .model-title { font-size: 0.82rem; font-weight: 700; color: #1e293b; margin-bottom: 1px; }
    .metric-bar { display: flex; justify-content: space-between; background: #F1F5F9; padding: 3px; border-radius: 4px; margin-bottom: 4px; }
    .metric-val { font-size: 0.7rem; font-weight: 700; color: var(--accent); text-align: center; flex: 1; }
    .metric-label { font-size: 0.45rem; color: #94a3b8; display: block; text-transform: uppercase; }

    .stButton>button { 
        background-color: var(--lite-purple); color: var(--deep-purple); 
        border: 1px solid var(--deep-purple); border-radius: 4px; 
        font-size: 0.58rem; height: 20px; padding: 0 4px; width: 100%;
        line-height: 1;
    }
    .stButton>button:hover { background-color: var(--deep-purple); color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- CONSTANTS ---
REG_PATH = "model_registry_v3.csv"
SHAP_FEATURES = {
    "Finance": ["Credit", "Income", "Debt", "History"],
    "Healthcare": ["Age", "BMI", "BP", "Glucose"],
    "Retail": ["Recency", "Frequency", "Monetary", "Location"],
    "Supply Chain": ["Lead_Time", "Inventory", "Route", "Demand"],
    "Default": ["Feature_A", "Feature_B", "Feature_C", "Feature_D"]
}

# --- DATA ENGINE ---
def load_sanitized_data():
    if not os.path.exists(REG_PATH): return pd.DataFrame()
    df = pd.read_csv(REG_PATH)
    # Aggressive numeric conversion for ROI and Charts
    num_cols = ['usage', 'accuracy', 'latency', 'data_drift', 'revenue_impact', 'risk_exposure']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce').fillna(0.0)
        else:
            df[col] = 0.0
    
    # Self-healing ROI: If all revenue is 0, generate synthetic based on usage
    if df['revenue_impact'].sum() == 0:
        df['revenue_impact'] = df['usage'] * 85.5
    
    if 'approval_status' not in df.columns: df['approval_status'] = 'Approved'
    return df

df_master = load_sanitized_data()

# --- SEARCH ENGINE ---
def hybrid_search(query, df):
    if not query or df.empty: return df
    q = query.lower()
    df_res = df.copy().fillna('N/A')
    # Math Filters
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

# --- UI: TILES ---
def render_tile(row, key_prefix, show_usage=False):
    v_label = "Usage" if show_usage else "Value"
    v_val = f"{int(row['usage'])}" if show_usage else f"${row['revenue_impact']/1000:.1f}k"
    
    st.markdown(f"""
    <div class="model-card">
        <div>
            <div style="display:flex; justify-content:space-between; margin-bottom:2px;">
                <span style="font-size:0.5rem; color:#666;">v{row.get('model_version','1.0')}</span>
                <span style="color:#2E7D32; font-size:0.55rem;">● Healthy</span>
            </div>
            <div class="model-title">{row['name'][:22]}</div>
            <div style="font-size:0.6rem; color:gray;"><b>{row['domain']}</b> | {row['contributor']}</div>
            <div style="font-size:0.65rem; color:#444; height:2.4em; overflow:hidden; margin:3px 0;">{row['use_cases']}</div>
        </div>
        <div class="metric-bar">
            <div class="metric-val"><span class="metric-label">Acc</span>{int(row['accuracy']*100)}%</div>
            <div class="metric-val"><span class="metric-label">Lat</span>{int(row['latency'])}ms</div>
            <div class="metric-val"><span class="metric-label">{v_label}</span>{v_val}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Compare", key=f"c_{key_prefix}_{row['name']}_{random.randint(0,999)}"):
            if row['name'] not in st.session_state.basket: st.session_state.basket.append(row['name'])
    with c2:
        with st.popover("Specs"):
            st.write(f"**SHAP: {row['name']}**")
            feats = SHAP_FEATURES.get(row['domain'], SHAP_FEATURES["Default"])
            shap_df = pd.DataFrame({'Feature': feats, 'Impact': [random.uniform(-0.5, 0.5) for _ in range(4)]}).sort_values('Impact')
            shap_df['Color'] = ['#EF4444' if x < 0 else '#6200EE' for x in shap_df['Impact']]
            fig = px.bar(shap_df, x='Impact', y='Feature', orientation='h', color='Color', color_discrete_map="identity", height=200)
            fig.update_layout(margin=dict(l=50,r=10,t=10,b=10), showlegend=False, xaxis_title="Impact")
            st.plotly_chart(fig, use_container_width=True)
    with c3:
        if st.button("Request", key=f"r_{key_prefix}_{row['name']}_{random.randint(0,999)}"):
            st.toast("Sent to Nat Patel")

# --- NAVIGATION ---
with st.sidebar:
    st.title("Hub Controls")
    api_key = st.text_input("Gemini API Key", type="password")
    nav = st.radio("Go to", ["WorkBench Companion", "Model Gallery", "AI Business Value", "Approval"])
    role = st.selectbox("Current User", ["John Doe", "Jane Nu", "Sam King", "Nat Patel", "Admin"])
    if 'basket' not in st.session_state: st.session_state.basket = []

# --- 1. WORKBENCH COMPANION ---
if nav == "WorkBench Companion":
    st.header("🤖 WorkBench Companion")
    if "messages" not in st.session_state: st.session_state.messages = []
    if "context_domain" not in st.session_state: st.session_state.context_domain = None

    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "df" in msg and not msg["df"].empty:
                cols = st.columns(min(len(msg["df"]), 3))
                for idx, r in enumerate(msg["df"].head(3).to_dict('records')):
                    with cols[idx]: render_tile(r, f"chat_h_{i}_{idx}", show_usage=msg.get("is_trending", False))
            if "pie_data" in msg and msg["pie_data"]:
                try: st.plotly_chart(px.pie(pd.DataFrame(msg["pie_data"]), values='revenue_impact', names='domain', hole=0.4))
                except: st.error("Chart data unavailable.")

    if prompt := st.chat_input("E.g. 'Compare IT-Model-010 and 011' or 'ROI breakdown'"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            q = prompt.lower()
            res_txt, res_df, res_pie, is_trend = "", pd.DataFrame(), None, False

            # Fuzzy Match for Comparison (IT-Model-001 and 008 logic)
            if "compare" in q:
                # Find exact or partial model IDs
                all_names = df_master['name'].tolist()
                found_names = [n for n in all_names if n.lower() in q or re.search(rf"\b{n.split('-')[-1]}\b", q)]
                if len(found_names) >= 2:
                    res_df = df_master[df_master['name'].isin(found_names)]
                    res_txt = f"Comparing performance for: {', '.join(found_names)}."
                    st.table(res_df[['name','accuracy','latency','usage']])
                else: res_txt = "Please provide exact model names or IDs to compare."

            elif "revenue" in q or "impact" in q:
                agg = df_master.groupby('domain')['revenue_impact'].sum().reset_index()
                if not agg.empty:
                    res_pie = agg.to_dict('records')
                    res_txt = "Here is the revenue impact contribution per domain:"
                    st.plotly_chart(px.pie(agg, values='revenue_impact', names='domain', hole=0.4))
                else: res_txt = "No ROI data found to analyze."

            elif "trending" in q:
                res_df = df_master.nlargest(3, 'usage')
                res_txt = "Current trending models by adoption:"
                is_trend = True
            
            else:
                res_df = hybrid_search(q, df_master).head(3)
                res_txt = "I found these relevant assets for you:"

            st.markdown(res_txt)
            if not res_df.empty:
                cols = st.columns(min(len(res_df), 3))
                for i_idx, r in enumerate(res_df.head(3).to_dict('records')):
                    with cols[i_idx]: render_tile(r, f"chat_n_{i_idx}", show_usage=is_trend)
            
            st.session_state.messages.append({"role": "assistant", "content": res_txt, "df": res_df, "pie_data": res_pie, "is_trending": is_trend})

# --- 2. MODEL GALLERY ---
elif nav == "Model Gallery":
    t1, t2 = st.tabs(["🏛 Unified Inventory", "⚖️ Benchmark Basket"])
    with t1:
        if st.session_state.basket:
            st.info(f"Basket: {len(st.session_state.basket)} items")
            if st.button("Reset Basket"): st.session_state.basket = []; st.rerun()
        q_gal = st.text_input("Smart Search")
        res = hybrid_search(q_gal, df_master)
        for i in range(0, min(len(res), 12), 3):
            cols = st.columns(3)
            for j in range(3):
                if i+j < len(res):
                    with cols[j]: render_tile(res.iloc[i+j], f"gal_{i+j}")
    with t2:
        if not st.session_state.basket: st.info("Basket empty.")
        else:
            c_df = df_master[df_master['name'].isin(st.session_state.basket)]
            st.table(c_df[['name','accuracy','latency','usage','data_drift']])
            st.plotly_chart(px.bar(c_df, x='name', y='accuracy', barmode='group'))

# --- 3. AI BUSINESS VALUE ---
elif nav == "AI Business Value":
    st.header("Executive Strategic ROI")
    agg = df_master.groupby('domain').agg({'revenue_impact':'sum','usage':'sum','accuracy':'mean'}).reset_index()
    k1, k2 = st.columns(2)
    k1.metric("Revenue Impact", f"${agg['revenue_impact'].sum()/1e6:.1f}M")
    k2.metric("Avg Fleet Accuracy", f"{int(agg['accuracy'].mean()*100)}%")
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(px.pie(agg, values='revenue_impact', names='domain', hole=0.4))
    with c2: st.plotly_chart(px.bar(agg, x='domain', y='revenue_impact', color='accuracy'))

# --- 4. APPROVAL ---
elif nav == "Approval":
    if role == "Nat Patel" or role == "Admin":
        st.subheader("Leader Approval Queue")
        pend = df_master[df_master['approval_status'] == 'Pending']
        if not pend.empty:
            st.table(pend[['name','domain','contributor']])
            if st.button("Bulk Approve"):
                df_master.loc[df_master['approval_status']=='Pending','approval_status']='Approved'
                df_master.to_csv(REG_PATH, index=False); st.rerun()
        else: st.success("Queue Clear.")
    else:
        st.header("Governance Telemetry")
        sel = st.selectbox("Solo Asset Focus", ["None"] + list(df_master['name'].unique()))
        cv = [1.0 if n == sel else 0.0 for n in df_master['name']]
        fig_p = go.Figure(data=go.Parcoords(
            labelfont=dict(size=10, color='black'), tickfont=dict(size=8),
            line=dict(color=cv, colorscale=[[0, '#FFF9C4'], [1, '#B71C1C']], showscale=False),
            dimensions=[dict(range=[0.7,1], label='Accuracy', values=df_master['accuracy']),
                        dict(range=[0,150], label='Latency', values=df_master['latency']),
                        dict(range=[0,25000], label='Usage', values=df_master['usage'])]))
        fig_p.update_layout(margin=dict(t=80, b=40, l=80, r=80), height=550)
        st.plotly_chart(fig_p, use_container_width=True)
