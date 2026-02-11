import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import datetime
import random

# --- ACCENTURE-INSPIRED STYLING ---
st.set_page_config(page_title="AI Model Marketplace", layout="wide")

st.markdown("""
    <style>
    :root { --accent-color: #A100FF; }
    .stApp { background-color: #ffffff; }
    
    .model-card {
        border: 1px solid #e0e0e0;
        border-top: 4px solid #A100FF;
        padding: 15px;
        background-color: #ffffff;
        margin-bottom: 20px;
        height: 420px; 
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: 0.3s;
    }
    .model-card:hover { 
        box-shadow: 0px 4px 15px rgba(161, 0, 255, 0.2); 
        transform: translateY(-5px);
    }
    
    .domain-tag { font-size: 0.7rem; font-weight: bold; color: #A100FF; text-transform: uppercase; margin-bottom: 5px; }
    .type-badge { font-size: 0.65rem; padding: 2px 6px; border-radius: 4px; font-weight: bold; float: right; }
    .badge-official { background-color: #e8dbff; color: #A100FF; }
    .badge-community { background-color: #f0f0f0; color: #666; }
    
    .model-title { font-size: 1.1rem; font-weight: 700; color: #000; margin: 5px 0; min-height: 50px; }
    .model-desc { font-size: 0.85rem; color: #444; overflow: hidden; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; }
    .client-text { font-size: 0.8rem; color: #888; font-style: italic; margin-top: 10px; }
    .metric-row { display: flex; justify-content: space-between; font-size: 0.75rem; color: #000; border-top: 1px solid #eee; padding-top: 10px; }
    
    .stButton>button { background-color: #000; color: white; border-radius: 0px; border: none; width: 100%; font-size: 0.8rem; }
    .stButton>button:hover { background-color: #A100FF; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- DATA INITIALIZATION ---
if 'registry' not in st.session_state:
    initial_data = [
        ["Fin-Audit-GPT", "Finance", "Official", 0.98, "200ms", "JP Morgan, Goldman Sachs", "Internal Audit Automation", "High-precision LLM for detecting non-compliant transactions."],
        ["Tax-Compliance-Pro", "Finance", "Official", 0.94, "45ms", "HSBC, Global Banking Group", "Tax Categorization", "Automates tax code mapping for cross-border trade."],
        ["Talent-Match-AI", "HR", "Official", 0.95, "30ms", "Google, Microsoft", "Hiring Bias Removal", "Matches resumes while masking demographic data."],
        ["Eco-Vendor-Scorer", "Procurement", "Official", 0.87, "50ms", "Unilever, Shell", "ESG Scoring", "Evaluates supplier sustainability using news analytics."],
        ["Logi-Route-Optimizer", "Supply Chain", "Official", 0.96, "300ms", "FedEx, DHL", "Last-Mile Delivery", "Reduces carbon footprint via route optimization."],
        ["Contract-Review-Bot", "Legal", "Official", 0.91, "800ms", "BMW, Toyota", "Legal Compliance", "Extracts liability clauses from vendor contracts."]
    ]
    
    domains = ["Finance", "HR", "Procurement", "Supply Chain", "IT", "Marketing", "Legal"]
    clients_list = ["Accenture", "Apple", "Coca-Cola", "NASA", "Amazon", "Samsung", "Meta"]
    
    # Generate 50 models
    for i in range(len(initial_data), 50):
        dom = random.choice(domains)
        initial_data.append([
            f"{dom}-Intelligence-{i+100}", dom, "Official",
            round(random.uniform(0.80, 0.98), 2), f"{random.randint(10, 500)}ms",
            f"{random.choice(clients_list)}", f"Standard {dom} Analysis",
            f"An enterprise {dom} model designed for high-throughput processing and predictive accuracy."
        ])

    st.session_state.registry = pd.DataFrame(initial_data, columns=["name", "domain", "type", "accuracy", "latency", "clients", "use_cases", "description"])
    st.session_state.registry['usage'] = [random.randint(100, 5000) for _ in range(len(st.session_state.registry))]

# --- SEARCH ENGINE (Updated to handle empty strings/nulls) ---
def get_recommendations(query, df):
    if not query: return df
    df = df.copy()
    df['search_blob'] = (df['name'] + " " + df['domain'] + " " + df['use_cases'] + " " + df['description'] + " " + df['clients']).fillna('')
    vectorizer = TfidfVectorizer(stop_words='english')
    matrix = vectorizer.fit_transform(df['search_blob'].tolist() + [query])
    scores = cosine_similarity(matrix[-1], matrix[:-1])[0]
    df['score'] = scores
    return df[df['score'] > 0.01].sort_values(by='score', ascending=False)

# --- HEADER ---
st.title("A. Model Marketplace")
st.caption("Centralized Repository for Enterprise AI & Community Contributions")
st.markdown("---")

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/Accenture.svg/2560px-Accenture.svg.png", width=120)
    role = st.radio("Switch Dashboard", ["Marketplace (Consumer)", "Governance (Admin)"])
    st.divider()
    st.write(f"**Total Assets:** {len(st.session_state.registry)}")

# --- MARKETPLACE VIEW ---
if role == "Marketplace (Consumer)":
    tab_gallery, tab_contribute = st.tabs(["🏛 Unified Gallery", "🚀 Contribute New Model"])
    
    with tab_gallery:
        c_search, c_filter = st.columns([3, 1])
        with c_search:
            query = st.text_input("🔍 Search everything (Tasks, Clients, Names...)", placeholder="e.g. 'Sustainability' or 'JP Morgan'")
        with c_filter:
            type_filter = st.multiselect("Source", ["Official", "Community"], default=["Official", "Community"])

        # Filter and Search
        filtered_df = st.session_state.registry[st.session_state.registry['type'].isin(type_filter)]
        results = get_recommendations(query, filtered_df)

        st.write(f"Results: {len(results)} Models")
        
        # Grid Display (3 tiles per row)
        for i in range(0, len(results), 3):
            cols = st.columns(3)
            for j in range(3):
                if i + j < len(results):
                    row = results.iloc[i + j]
                    badge_class = "badge-official" if row['type'] == "Official" else "badge-community"
                    with cols[j]:
                        st.markdown(f"""
                        <div class="model-card">
                            <div>
                                <span class="type-badge {badge_class}">{row['type']}</span>
                                <div class="domain-tag">{row['domain']}</div>
                                <div class="model-title">{row['name']}</div>
                                <div class="model-desc">{row['description']}</div>
                                <div class="client-text">Use Case: {row['use_cases']}</div>
                                <div class="client-text" style="color:#A100FF;">Clients: {row['clients']}</div>
                            </div>
                            <div>
                                <div class="metric-row">
                                    <span><b>ACCURACY:</b> {int(row['accuracy']*100)}%</span>
                                    <span><b>LATENCY:</b> {row['latency']}</span>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        if st.button("Access Model", key=f"btn_{row['name']}_{i+j}"):
                            st.toast(f"Generating API keys for {row['name']}...")

    with tab_contribute:
        st.subheader("Contribute to the Enterprise Intelligence")
        st.write("Newly contributed models appear instantly in the Unified Gallery for search and discovery.")
        
        with st.form("contribution_form", clear_on_submit=True):
            col_a, col_b = st.columns(2)
            with col_a:
                c_name = st.text_input("Model Name (e.g. Supply-Chain-Optim-V1)")
                c_domain = st.selectbox("Business Domain", domains)
            with col_b:
