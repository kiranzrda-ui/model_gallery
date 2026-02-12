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
    
    /* Compact Card UI */
    .model-card {
        background: white; border: 1px solid #e2e8f0; border-top: 3px solid var(--accent);
        padding: 10px; border-radius: 6px; min-height: 330px; 
        display: flex; flex-direction: column; justify-content: space-between;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .model-title { font-size: 0.85rem; font-weight: 700; color: #1e293b; margin-bottom: 2px; }
    .meta-line { font-size: 0.65rem; color: #64748b; margin: 0; }
    .use-case-text { font-size: 0.68rem; color: #334155; height: 2.8em; overflow: hidden; margin: 4px 0; line-height: 1.2; }
    
    .metric-bar { display: flex; justify-content: space-between; background: #F1F5F9; padding: 4px; border-radius: 4px; margin-bottom: 8px; }
    .metric-val { font-size: 0.72rem; font-weight: 700; color: var(--accent); text-align: center; flex: 1; }
    .metric-label { font-size: 0.5rem; color: #94a3b8; display: block; text-transform: uppercase; }

    /* Small Lite Purple Buttons */
    .stButton>button { 
        background-color: var(--lite-purple); color: var(--deep-purple); 
        border: 1px solid var(--deep-purple); border-radius: 4px; 
        font-size: 0.62rem; height: 22px; padding: 0 4px; width: 100%;
    }
    .stButton>button:hover { background-color: var(--deep-purple); color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- DOMAIN SHAP MAPS ---
SHAP_FEATURES = {
    "Finance": ["Credit_Score", "Annual_Income", "Debt_Ratio", "Trans_Freq", "Collateral"],
    "Healthcare": ["Age", "BMI", "Blood_Pressure", "Glucose", "Genetic_Score"],
    "Retail": ["Recency", "Frequency", "Monetary_Val", "Seasonality", "Customer_Age"],
    "HR": ["Tenure", "Engagement_Score", "Salary_Delta", "Education_Lvl", "Travel_Freq"],
    "Supply Chain": ["Lead_Time", "Inventory_Lvl", "Distance", "Supplier_Rating", "Demand_Vol"],
    "IT Ops": ["CPU_Load", "Latency_P99", "Disk_Usage", "Packet_Loss", "Memory_Pressure"]
}

# --- DATA LAYER ---
REG_PATH = "model_registry_v3.csv"
REQ_PATH = "requests_v3.csv"

def get_clean_data():
    if not os.path.exists(REG_PATH): return pd.DataFrame()
    df = pd.read_csv(REG_PATH)
    # Ensure numeric types for ROI & Admin charts
    nums = ['usage', 'accuracy', 'latency', 'data_drift', 'revenue_impact', 'risk_exposure', 'cpu_util', 'error_rate', 'throughput']
    for c in nums:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce').fillna(0)
    if 'approval_status' not in df.columns: df['approval_status'] = 'Approved'
    return df

df_master = get_clean_data()

# --- GEMINI AI CONFIG ---
def get_gemini_response(prompt, api_key):
    try:
        genai.configure(api_key=api_key)
        # Using the updated model identifier
        model = genai.GenerativeModel('gemini-1.5-flash')
        context = f"You are the Enterprise Model Hub Assistant. You help users navigate a database of {len(df_master)} models. You provide professional insights on model drift, ROI, and performance. Keep responses concise."
        response = model.generate_content(context + "\nUser: " + prompt)
        return response.text
    except Exception as e:
        return f"Gemini Error: {str(e)}"

# --- SEARCH ENGINE ---
def hybrid_search(query, df):
    if not query: return df
    q = query.lower()
    df_res = df.copy().fillna('N/A')
    
    # 1. Regex logic (latency < 50, drift < 0.1)
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
    
    # 2. Textual Search
    df_res['blob'] = df_res.astype(str).apply(' '.join, axis=1)
    vec = TfidfVectorizer(stop_words='english')
    mtx = vec.fit_transform(df_res['blob'].tolist() + [query])
    df_res['relevance'] = cosine_similarity(mtx[-1], mtx[:-1])[0]
    return df_res[df_res['relevance'] > 0.01].sort_values('relevance', ascending=False)

# --- UI COMPONENTS ---
def render_tile(row, idx):
    st.markdown(f"""
    <div class="model-card">
        <div>
            <div class="tag-row" style="display:flex; justify-content:space-between; margin-bottom:5px;">
                <span class="registry-badge">{row.get('registry_provider','Vertex AI')}</span>
                <span style="font-size:0.6rem; color:#2E7D32;">● Healthy</span>
            </div>
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
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Request", key=f"req_{idx}"): st.toast("Sent to Leader")
    with c2:
        with st.popover("SHAP"):
            feats = SHAP_FEATURES.get(row['domain'], ["F1","F2","F3","F4","F5"])
            shap_df = pd.DataFrame({'Feature': feats, 'Impact': [random.uniform(-0.5, 0.5) for _ in range(5)]}).sort_values('Impact')
            shap_df['Color'] = ['#EF4444' if x < 0 else '#6200EE' for x in shap_df['Impact']]
            fig = px.bar(shap_df, x='Impact', y='Feature', orientation='h', color='Color', color_discrete_map="identity", height=180)
            fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True, key=f"sh_plot_{idx}")

# --- NAVIGATION ---
with st.sidebar:
    st.title("Model Hub 4.0")
    api_key = st.text_input("Gemini API Key", type="password", help="Get key from Google AI Studio")
    view = st.radio("Navigation", ["AI Copilot", "Marketplace", "Strategic ROI", "My Portfolio", "Admin Ops"])
    st.divider()
    role = st.selectbox("Current User", ["John Doe", "Jane Nu", "Sam King", "Nat Patel", "Admin"])

# --- 1. AI COPILOT ---
if view == "AI Copilot":
    st.header("🤖 Hub Assistant")
    if "messages" not in st.session_state: st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "df" in msg:
                cols = st.columns(min(len(msg["df"]), 3))
                for i, r in enumerate(msg["df"].head(3).to_dict('records')):
                    with cols[i]: render_tile(r, f"chat_res_{i}_{random.randint(0,1000)}")

    if prompt := st.chat_input("Ask about models, ROI, or drift..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            response_text = get_gemini_response(prompt, api_key) if api_key else "Providing search results (Add API key for AI chat)..."
            st.markdown(response_text)
            
            # Logic integration: Pull models related to prompt
            results = hybrid_search(prompt, df_master)
            if len(results) < len(df_master):
                st.write(f"I found {len(results)} relevant assets:")
                cols = st.columns(min(len(results), 3))
                for i, r in enumerate(results.head(3).to_dict('records')):
                    with cols[i]: render_tile(r, f"new_chat_{i}")
                st.session_state.messages.append({"role": "assistant", "content": response_text, "df": results.head(3)})
            else:
                st.session_state.messages.append({"role": "assistant", "content": response_text})

# --- 2. MARKETPLACE ---
elif view == "Marketplace":
    t1, t2, t3 = st.tabs(["🏛 Gallery", "🚀 Ingest", "🔥 Insights"])
    with t1:
        q = st.text_input("💬 Smart Search (Keywords or Logic like 'latency < 40')")
        results = hybrid_search(q, df_master)
        for i in range(0, min(len(results), 21), 3):
            cols = st.columns(3)
            for j in range(3):
                if i+j < len(results):
                    with cols[j]: render_tile(results.iloc[i+j], f"gal_{i+j}")
    with t2:
        with st.form("ingest"):
            st.subheader("Add Asset")
            fn = st.text_input("Model Name*"); fe = st.text_area("Description*")
            if st.form_submit_button("Publish"):
                st.success("Persisted to CSV!")
    with t3:
        st.subheader("🔥 Trending Assets")
        st.table(df_master.nlargest(5, 'usage')[['name','usage','domain']])
        st.subheader("💎 Hidden Gems")
        st.table(df_master[(df_master['accuracy'] > 0.95) & (df_master['usage'] < 1000)].head(5)[['name','accuracy']])

# --- 3. ROI DASHBOARD ---
elif view == "Strategic ROI":
    st.header("Strategic Portfolio ROI")
    agg = df_master.groupby('domain').agg({'revenue_impact':'sum','risk_exposure':'sum','usage':'sum','accuracy':'mean'}).reset_index()
    k1, k2 = st.columns(2)
    k1.metric("Revenue Impact", f"${agg['revenue_impact'].sum()/1e6:.1f}M")
    k2.metric("Fleet Quality", f"{int(agg['accuracy'].mean()*100)}%")
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(px.pie(agg, values='revenue_impact', names='domain', hole=0.4, title="Revenue Attribution"))
    with c2: st.plotly_chart(px.bar(agg, x='domain', y='revenue_impact', color='accuracy', title="ROI vs Performance"))

# --- 4. MY PORTFOLIO ---
elif view == "My Portfolio":
    my_m = df_master[df_master['contributor'] == role]
    if not my_m.empty:
        st.header(f"Contributor Hub: {role}")
        sel = st.selectbox("Inspect Asset", my_m['name'].unique())
        m = my_m[my_m['name'] == sel].iloc[0]
        fig = go.Figure(go.Scatterpolar(r=[m['accuracy']*100, 100-(m['data_drift']*100), m.get('cpu_util',50), 100-m.get('error_rate',0)*10], theta=['Accuracy','Stability','Efficiency','Reliability'], fill='toself'))
        st.plotly_chart(fig, use_container_width=True)
    else: st.info("No assets found for this user.")

# --- 5. ADMIN OPS ---
elif view == "Admin Ops":
    st.header("Fleet Telemetry")
    sel = st.selectbox("Highlight Solo Asset", ["None"] + list(df_master['name'].unique()))
    cv = [1.0 if n == sel else 0.0 for n in df_master['name']]
    fig_p = go.Figure(data=go.Parcoords(
        labelfont=dict(size=10, color='black'), tickfont=dict(size=8),
        line=dict(color=cv, colorscale=[[0, '#FFF9C4'], [1, '#B71C1C']], showscale=False),
        dimensions=[dict(range=[0.7,1], label='Accuracy', values=df_master['accuracy']),
                    dict(range=[0,150], label='Latency', values=df_master['latency']),
                    dict(range=[0,25000], label='Usage', values=df_master['usage']),
                    dict(range=[0,0.3], label='Drift', values=df_master['data_drift'])]))
    fig_p.update_layout(margin=dict(t=80, b=40, l=80, r=80), height=550)
    st.plotly_chart(fig_p, use_container_width=True)
