import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import datetime

# --- ACCENTURE-INSPIRED STYLING ---
st.set_page_config(page_title="AI Model Marketplace | Enterprise Portal", layout="wide")

st.markdown("""
    <style>
    /* Main colors */
    :root {
        --accent-color: #A100FF; /* Accenture Purple */
        --bg-color: #ffffff;
    }
    .stApp { background-color: white; }
    
    /* Headers */
    h1, h2, h3 { font-family: 'Graphik', 'Arial'; color: #000000; font-weight: 700; }
    
    /* Buttons */
    .stButton>button {
        background-color: #A100FF;
        color: white;
        border-radius: 0px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stButton>button:hover { background-color: #000000; color: white; }
    
    /* Custom Cards */
    .model-card {
        border-left: 5px solid #A100FF;
        padding: 20px;
        background-color: #f9f9f9;
        margin-bottom: 20px;
        border-radius: 4px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
    }
    .status-tag {
        font-size: 0.8rem;
        padding: 2px 8px;
        background: #000;
        color: #fff;
        border-radius: 12px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- DATA INITIALIZATION ---
if 'registry' not in st.session_state:
    st.session_state.registry = pd.DataFrame([
        # HR Models
        {"id": 1, "name": "TalentRetain-AI", "domain": "HR", "type": "Official", "accuracy": 0.92, "latency": "15ms", "clients": "Global Tech Corp", "use_cases": "Employee Churn Prediction", "description": "Predicts flight risk of high-performers using engagement data.", "usage": 1500},
        {"id": 2, "name": "Resume-Screener-Pro", "domain": "HR", "type": "Official", "accuracy": 0.89, "latency": "120ms", "clients": "Retail Giants", "use_cases": "Automated Candidate Scoring", "description": "NLP-based extraction of skills from resumes vs JD.", "usage": 3200},
        
        # Finance Models
        {"id": 3, "name": "CashFlow-Forecaster", "domain": "Finance", "type": "Official", "accuracy": 0.96, "latency": "30ms", "clients": "Major EU Bank", "use_cases": "Quarterly Liquidity Planning", "description": "Time-series forecasting for enterprise cash reserves.", "usage": 800},
        {"id": 4, "name": "Anomaly-Finance-V4", "domain": "Finance", "type": "Official", "accuracy": 0.98, "latency": "5ms", "clients": "Public Sector", "use_cases": "Fraud & Audit Detection", "description": "Detects suspicious ledger entries in real-time.", "usage": 5400},
        
        # Supply Chain & Procurement
        {"id": 5, "name": "Vendor-Risk-Scorer", "domain": "Procurement", "type": "Official", "accuracy": 0.85, "latency": "45ms", "clients": "Auto Manufacturer", "use_cases": "Supplier Reliability Ranking", "description": "Scores suppliers based on delivery history and geopolitical risk.", "usage": 1200},
        {"id": 6, "name": "Demand-Sense-Inventory", "domain": "Supply Chain", "type": "Official", "accuracy": 0.91, "latency": "50ms", "clients": "FMCG Leader", "use_cases": "Inventory Optimization", "description": "Optimizes stock levels across 500+ warehouses.", "usage": 4100},
    ])

if 'search_logs' not in st.session_state:
    st.session_state.search_logs = []

# --- FUNCTIONS ---
def add_contribution(name, domain, description, use_cases):
    new_id = len(st.session_state.registry) + 1
    new_row = {
        "id": new_id, "name": name, "domain": domain, "type": "Community", 
        "accuracy": 0.0, "latency": "N/A", "clients": "Internal Testing",
        "use_cases": use_cases, "description": description, "usage": 0
    }
    st.session_state.registry = pd.concat([st.session_state.registry, pd.DataFrame([new_row])], ignore_index=True)

# --- HEADER ---
col1, col2 = st.columns([1, 4])
with col1:
    st.markdown("<h1 style='border-bottom: 8px solid #A100FF;'>A.</h1>", unsafe_allow_html=True)
with col2:
    st.title("Enterprise AI Model Marketplace")
    st.caption("Strategy | Performance | Governance")

# --- TABS LAYOUT ---
tab1, tab2, tab3, tab4 = st.tabs(["🏛 Official Gallery", "🤝 Community Hub", "🚀 Contribute", "📊 Admin Insights"])

# --- TAB 1: OFFICIAL GALLERY ---
with tab1:
    st.subheader("Vetted Enterprise Models")
    domain_filter = st.multiselect("Filter by Domain", options=["HR", "Finance", "Procurement", "Supply Chain"], default=["HR", "Finance", "Procurement", "Supply Chain"])
    
    search_query = st.text_input("💬 AI Search: Describe your business problem (e.g., 'I need to find fraud in invoices')", key="main_search")
    
    df_official = st.session_state.registry[(st.session_state.registry['type'] == 'Official') & (st.session_state.registry['domain'].isin(domain_filter))]
    
    if search_query:
        # Simple semantic search simulation
        docs = df_official['description'].tolist()
        vectorizer = TfidfVectorizer().fit_transform(docs + [search_query])
        sim = cosine_similarity(vectorizer[-1], vectorizer[:-1])[0]
        df_official = df_official.copy()
        df_official['score'] = sim
        df_official = df_official[df_official['score'] > 0].sort_values(by='score', ascending=False)
        st.session_state.search_logs.append(search_query)

    for _, row in df_official.iterrows():
        st.markdown(f"""
        <div class="model-card">
            <span class="status-tag">{row['domain']}</span>
            <h3>{row['name']}</h3>
            <p><b>Description:</b> {row['description']}</p>
            <p style='color: #666;'><b>Clients:</b> {row['clients']} | <b>Use Case:</b> {row['use_cases']}</p>
            <hr>
            <small>Performance: {row['accuracy']*100}% Accuracy | Latency: {row['latency']}</small>
        </div>
        """, unsafe_allow_html=True)
        st.button(f"Request Integration: {row['name']}", key=f"req_{row['id']}")

# --- TAB 2: COMMUNITY HUB ---
with tab2:
    st.subheader("Data Scientist Contributions")
    st.info("These models are contributed by our internal community and are awaiting formal verification.")
    
    df_comm = st.session_state.registry[st.session_state.registry['type'] == 'Community']
    if df_comm.empty:
        st.write("No community models yet. Be the first to contribute!")
    else:
        st.table(df_comm[['name', 'domain', 'use_cases', 'description']])

# --- TAB 3: CONTRIBUTE ---
with tab3:
    st.subheader("Register a New Model")
    with st.form("contribute_form"):
        st.write("Provide details about your model. Our AI will index it for the community.")
        c_name = st.text_input("Model Name")
        c_domain = st.selectbox("Business Domain", ["HR", "Finance", "Procurement", "Supply Chain", "Other"])
        c_desc = st.text_area("What does this model do? (Free text description)")
        c_use = st.text_input("Specific Use Case (e.g., Invoice Parsing)")
        
        submit = st.form_submit_button("Submit to Registry")
        if submit:
            if c_name and c_desc:
                add_contribution(c_name, c_domain, c_desc, c_use)
                st.success(f"Model '{c_name}' has been successfully added to the Community Hub!")
            else:
                st.error("Please fill in Name and Description.")

# --- TAB 4: ADMIN DASHBOARD ---
with tab4:
    st.subheader("Marketplace Efficacy Metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Model Inventory", len(st.session_state.registry))
    m2.metric("Official vs. Community", f"{len(st.session_state.registry[st.session_state.registry['type']=='Official'])} / {len(st.session_state.registry[st.session_state.registry['type']=='Community'])}")
    m3.metric("Top Domain", st.session_state.registry['domain'].value_counts().idxmax())
    
    st.divider()
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        fig = px.pie(st.session_state.registry, names='domain', title="Inventory by Domain", hole=0.4, color_discrete_sequence=['#A100FF', '#000000', '#444444', '#cccccc'])
        st.plotly_chart(fig)
    with col_chart2:
        fig2 = px.bar(st.session_state.registry, x='name', y='usage', title="Model Consumption Growth", color_discrete_sequence=['#A100FF'])
        st.plotly_chart(fig2)
