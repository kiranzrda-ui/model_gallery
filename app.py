import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import os, datetime, random, re, csv, json

# --- CONFIG & STYLING ---
st.set_page_config(page_title="Enterprise AI Services Workbench 1.0", layout="wide")

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
    nums = ['usage', 'accuracy', 'latency', 'data_drift', 'revenue_impact', 'risk_exposure']
    for c in nums:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce').fillna(0)
    if 'approval_status' not in df.columns: df['approval_status'] = 'Approved'
    return df

df_master = load_sanitized_data()

# --- GEMINI INTEGRATION (Fixed 404) ---
def call_gemini(prompt, api_key):
    try:
        genai.configure(api_key=api_key)
        # Using the most current and stable model ID
        model = genai.GenerativeModel('gemini-1.5-flash')
        return model.generate_content(prompt).text
    except Exception as e:
        return f"Gemini Error: {str(e)}"

# --- INTENT 1: SUBMISSION EXTRACTION ---
def extract_model_entities(user_input, api_key):
    prompt = f"""
    Act as a high-precision data extractor. Convert the following model request into a JSON object.
    Required Fields: 
    - name (String)
    - domain (String)
    - accuracy (Float between 0 and 1. If user says 99%, result is 0.99)
    - latency (Integer ms)
    - use_cases (String)
    
    Text: "{user_input}"
    Return ONLY the raw JSON block. No markdown.
    """
    response = call_gemini(prompt, api_key)
    try:
        # Robust JSON cleaning (removes ```json etc)
        clean_json = re.search(r"\{.*\}", response, re.DOTALL).group()
        return json.loads(clean_json)
    except:
        # Fallback to Regex if LLM fails
        name = re.search(r"called ([\w\-\s]+)", user_input, re.I)
        acc = re.search(r"(\d+)%", user_input)
        lat = re.search(r"(\d+)ms", user_input)
        return {
            "name": name.group(1).strip() if name else "New-Asset",
            "domain": "Finance" if "finance" in user_input.lower() else "General",
            "accuracy": float(acc.group(1))/100 if acc else 0.85,
            "latency": int(lat.group(1)) if lat else 40,
            "use_cases": user_input
        }

# --- SEARCH ENGINE ---
def hybrid_search(query, df):
    if not query or df.empty: return df
    df_res = df.copy().fillna('N/A')
    df_res['blob'] = df_res.astype(str).apply(' '.join, axis=1)
    vec = TfidfVectorizer(stop_words='english')
    mtx = vec.fit_transform(df_res['blob'].tolist() + [query])
    df_res['relevance'] = cosine_similarity(mtx[-1], mtx[:-1])[0]
    return df_res[df_res['relevance'] > 0.01].sort_values('relevance', ascending=False)

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
            <div class="metric-val"><span class="metric-label">Drift</span>{row.get('data_drift',0)}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Request Access", key=f"btn_{key_prefix}_{random.randint(0,10**6)}"):
        st.toast("Access Request Logged")

# --- NAVIGATION ---
with st.sidebar:
    st.title("AI Services Workbench")
    api_key = st.text_input("Gemini API Key", type="password")
    nav = st.radio("Navigation", ["Companion Mode", "Model Gallery", "AI Business Value", "Admin"])
    st.divider()
    role = st.selectbox("Current User", ["John Doe", "Jane Nu", "Sam King", "Nat Patel", "Admin"])
    st.session_state.role = role

# --- 1. AI COPILOT ---
if nav == "AI Copilot":
    st.header("🤖 AI Workbench Companion")
    if "messages" not in st.session_state: st.session_state.messages = []
    if "context" not in st.session_state: st.session_state.context = None

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "df" in msg:
                cols = st.columns(min(len(msg["df"]), 3))
                for i, r in enumerate(msg["df"].head(3).to_dict('records')):
                    with cols[i]: render_tile(r, f"chat_h_{i}")
            if "chart" in msg: st.plotly_chart(msg["chart"], use_container_width=True)

    if prompt := st.chat_input("E.g. 'Compare Cash App and Ris-Model-001' or 'Submit GlobalRisk-v2'"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            # A. INTENT: SUBMISSION
            if "submit" in prompt.lower() or "register" in prompt.lower():
                entities = extract_model_entities(prompt, api_key)
                new_row = pd.DataFrame([{
                    "name": entities.get('name'), "domain": entities.get('domain'),
                    "accuracy": entities.get('accuracy'), "latency": entities.get('latency'),
                    "use_cases": entities.get('use_cases'), "contributor": role,
                    "approval_status": "Pending", "model_stage": "Prod", "usage": 0,
                    "revenue_impact": 0, "risk_exposure": 0
                }])
                df_master = pd.concat([df_master, new_row], ignore_index=True)
                df_master.to_csv(REG_PATH, index=False)
                res = f"✅ **Extracted & Registered:** Name: `{entities.get('name')}`, Accuracy: `{entities.get('accuracy')*100}%`. Sent to Nat Patel for approval."
                st.markdown(res)
                st.session_state.messages.append({"role": "assistant", "content": res})

            # B. INTENT: COMPARE
            elif "compare" in prompt.lower():
                # Extract model names present in prompt
                names = [n for n in df_master['name'] if n.lower() in prompt.lower()]
                if len(names) >= 2:
                    cdf = df_master[df_master['name'].isin(names)]
                    fig = px.bar(cdf, x='name', y=['accuracy','latency','data_drift','usage'], barmode='group', title="Side-by-Side Analysis")
                    st.plotly_chart(fig, use_container_width=True)
                    res = f"I have compared {', '.join(names)} across all key metrics."
                    st.session_state.messages.append({"role": "assistant", "content": res, "chart": fig})
                else:
                    st.markdown("Please mention at least two exact model names to compare (e.g., Ris-Model-001).")

            # C. INTENT: ROI & ANALYTICS
            elif any(x in prompt.lower() for x in ["revenue", "impact", "top contributor"]):
                domain = next((d for d in df_master['domain'].unique() if d.lower() in prompt.lower()), st.session_state.context)
                if domain:
                    st.session_state.context = domain
                    subset = df_master[df_master['domain'] == domain]
                    if "top" in prompt.lower() or "contributor" in prompt.lower():
                        top_m = subset.nlargest(3, 'revenue_impact')
                        res = f"The top contributors for **{domain}** are: " + ", ".join(top_m['name'].tolist())
                        st.session_state.messages.append({"role": "assistant", "content": res, "df": top_m})
                    else:
                        impact = subset['revenue_impact'].sum()
                        res = f"The **{domain}** domain has generated an impact of **${impact/1e6:.2f}M**."
                        st.session_state.messages.append({"role": "assistant", "content": res})
                    st.markdown(res)
                else:
                    st.markdown("Which domain (e.g., Finance, Risk, HR) should I analyze?")

            # D. INTENT: PORTFOLIO
            elif "my contributions" in prompt.lower() or "my models" in prompt.lower():
                user_df = df_master[df_master['contributor'] == role]
                res = f"Found **{len(user_df)}** models contributed by you, {role}."
                st.markdown(res)
                cols = st.columns(min(len(user_df), 3))
                for i, r in enumerate(user_df.head(3).to_dict('records')):
                    with cols[i]: render_tile(r, f"chat_p_{i}")
                st.session_state.messages.append({"role": "assistant", "content": res, "df": user_df.head(3)})

            # E. DEFAULT SEARCH
            else:
                res = call_gemini(prompt, api_key) if api_key else "Providing search results..."
                st.markdown(res)
                search_res = hybrid_search(prompt, df_master)
                if not search_res.empty and len(search_res) < len(df_master):
                    cols = st.columns(min(len(search_res), 3))
                    for i, r in enumerate(search_res.head(3).to_dict('records')):
                        with cols[i]: render_tile(r, f"chat_s_{i}")
                    st.session_state.messages.append({"role": "assistant", "content": res, "df": search_res.head(3)})
                else:
                    st.session_state.messages.append({"role": "assistant", "content": res})

# --- OTHER TABS (STABLE) ---
elif nav == "Model Gallery":
    q = st.text_input("Search Registry")
    res = hybrid_search(q, df_master)
    for i in range(0, min(len(res), 12), 3):
        cols = st.columns(3)
        for j in range(3):
            if i+j < len(res):
                with cols[j]: render_tile(res.iloc[i+j], f"gal_{i+j}")

elif nav == "Strategy ROI":
    agg = df_master.groupby('domain').agg({'revenue_impact':'sum','usage':'sum','accuracy':'mean'}).reset_index()
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.pie(agg, values='revenue_impact', names='domain', hole=0.4, title="Revenue Share"))
    c2.plotly_chart(px.bar(agg, x='domain', y='revenue_impact', color='accuracy', title="ROI vs Performance"))

elif nav == "Admin Ops":
    if role == "Nat Patel":
        st.subheader("Leader Approvals")
        pend = df_master[df_master['approval_status'] == 'Pending']
        if not pend.empty:
            st.table(pend[['name','domain','contributor','accuracy']])
            if st.button("Bulk Approve"):
                df_master.loc[df_master['approval_status'] == 'Pending', 'approval_status'] = 'Approved'
                df_master.to_csv(REG_PATH, index=False); st.rerun()
        else: st.success("No requests pending.")
    else:
        sel = st.selectbox("Highlight Asset", ["None"] + list(df_master['name'].unique()))
        cv = [1.0 if n == sel else 0.0 for n in df_master['name']]
        fig = go.Figure(data=go.Parcoords(
            labelfont=dict(size=10, color='black'),
            line=dict(color=cv, colorscale=[[0, '#FFF9C4'], [1, '#B71C1C']], showscale=False),
            dimensions=[dict(range=[0.7,1], label='Accuracy', values=df_master['accuracy']),
                        dict(range=[0,150], label='Latency', values=df_master['latency']),
                        dict(range=[0,25000], label='Usage', values=df_master['usage'])]))
        fig.update_layout(margin=dict(t=80, b=40, l=80, r=80), height=550)
        st.plotly_chart(fig, use_container_width=True)
