import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import datetime
import random

# --- ACCENTURE-INSPIRED STYLING ---
st.set_page_config(page_title="Enterprise AI Model Marketplace", layout="wide")

st.markdown("""
    <style>
    :root { --accent-color: #A100FF; }
    .stApp { background-color: #ffffff; }
    h1, h2, h3 { font-family: 'Graphik', 'Arial'; color: #000000; }
    .stButton>button { background-color: #A100FF; color: white; border-radius: 0px; border: none; width: 100%; }
    .stButton>button:hover { background-color: #000000; color: white; }
    .model-card {
        border: 1px solid #e0e0e0;
        border-left: 5px solid #A100FF;
        padding: 20px;
        background-color: #ffffff;
        margin-bottom: 15px;
        transition: 0.3s;
    }
    .model-card:hover { box-shadow: 5px 5px 15px rgba(0,0,0,0.1); }
    .status-tag { font-size: 0.7rem; padding: 3px 10px; background: #000; color: #fff; border-radius: 0px; text-transform: uppercase; }
    .metric-text { font-size: 0.85rem; color: #666; }
    </style>
    """, unsafe_allow_html=True)

# --- DATA GENERATOR (50+ MODELS) ---
if 'registry' not in st.session_state:
    raw_data = [
        # FINANCE
        ["Fin-Audit-GPT", "Finance", "Official", 0.98, "200ms", "JP Morgan, HSBC", "Internal Audit Automation", "LLM for detecting non-compliant transactions in ledgers."],
        ["Tax-Genie-V2", "Finance", "Official", 0.94, "45ms", "PwC, Deloitte", "VAT Compliance", "Automates tax categorization for cross-border trade."],
        ["Risk-Pulse-Forex", "Finance", "Official", 0.88, "10ms", "Barclays", "Currency Risk", "Real-time hedging recommendations for FX volatility."],
        ["Debt-Collector-Optimizer", "Finance", "Community", 0.82, "100ms", "Internal Tech", "Accounts Receivable", "Predicts best time to contact debtors for recovery."],
        ["Equity-Analyst-Pro", "Finance", "Official", 0.91, "1.2s", "BlackRock", "Portfolio Sentiment", "Scans news and earnings calls for buy/sell signals."],
        
        # HR
        ["Talent-Match-AI", "HR", "Official", 0.95, "30ms", "Google, Accenture", "Hiring Bias Removal", "Matches resumes to JDs while masking gender/ethnic data."],
        ["Culture-Pulse", "HR", "Official", 0.85, "15ms", "Salesforce", "Retention Analysis", "Sentiment analysis on internal Slack/Email to predict attrition."],
        ["Pay-Equity-Scanner", "HR", "Official", 0.99, "5ms", "Microsoft", "Salary Governance", "Identifies pay gaps across demographics automatically."],
        ["Workforce-Planner-2025", "HR", "Community", 0.78, "2s", "Internal Tech", "Capacity Planning", "Simulates hiring needs based on sales pipeline growth."],
        ["Skill-Graph-Builder", "HR", "Official", 0.92, "400ms", "LinkedIn", "L&D Pathing", "Maps current employee skills to future required roles."],

        # PROCUREMENT & SUPPLY CHAIN
        ["Eco-Vendor-Scorer", "Procurement", "Official", 0.87, "50ms", "Unilever", "ESG Scoring", "Evaluates supplier sustainability using satellite and news data."],
        ["Logi-Route-Optimizer", "Supply Chain", "Official", 0.96, "300ms", "FedEx, DHL", "Last-Mile Delivery", "Reduces fuel consumption by optimizing multi-stop routes."],
        ["Inventory-Ghost-Detector", "Supply Chain", "Official", 0.93, "20ms", "Walmart", "Stock Accuracy", "Predicts phantom inventory where system says 'in stock' but shelf is empty."],
        ["Contract-Review-Bot", "Procurement", "Official", 0.91, "800ms", "BMW", "Legal Compliance", "Extracts termination clauses and liability from PDF contracts."],
        ["Price-Negotiator-Agent", "Procurement", "Community", 0.76, "1.5s", "Internal Tech", "Auto-Bidding", "Suggests counter-offer prices based on historical raw material costs."],

        # IT & OPERATIONS
        ["Cloud-Cost-Slasher", "IT", "Official", 0.97, "10ms", "AWS, Netflix", "Cloud FinOps", "Identifies underutilized EC2/S3 instances and auto-scales down."],
        ["Log-Anomaly-Sentinel", "IT", "Official", 0.99, "2ms", "Cisco", "Cybersecurity", "Zero-day threat detection via server log pattern shifts."],
        ["Helpdesk-Triage-V4", "IT", "Official", 0.89, "15ms", "ServiceNow", "Auto-Ticketing", "Categorizes and assigns IT support tickets using NLP."],
        ["SQL-Query-Optimizer", "IT", "Community", 0.84, "500ms", "Internal Tech", "DB Performance", "Suggests missing indexes for slow-running analytical queries."],
        ["Network-Twin-Sim", "IT", "Official", 0.92, "4s", "Verizon", "Infrastructure", "Digital twin for simulating 5G signal coverage in new cities."]
    ]
    
    # Expand to 50+ models by programmatically generating variations
    domains = ["Finance", "HR", "Procurement", "Supply Chain", "IT", "Marketing", "Legal"]
    clients_list = ["Accenture", "Apple", "Shell", "Coca-Cola", "Toyota", "NASA", "Goldman Sachs"]
    
    for i in range(len(raw_data), 55):
        domain = random.choice(domains)
        model_name = f"{domain}-Analyzer-{i}"
        raw_data.append([
            model_name, domain, "Official" if i % 2 == 0 else "Community",
            round(random.uniform(0.75, 0.99), 2), 
            f"{random.randint(5, 500)}ms",
            f"{random.choice(clients_list)}, {random.choice(clients_list)}",
            f"Automated {domain} Support",
            f"This is a high-performance {domain} model designed for enterprise scalability and {random.choice(['risk reduction', 'cost savings', 'efficiency', 'innovation'])}."
        ])

    st.session_state.registry = pd.DataFrame(raw_data, columns=["name", "domain", "type", "accuracy", "latency", "clients", "use_cases", "description"])
    st.session_state.registry['usage'] = [random.randint(100, 10000) for _ in range(len(st.session_state.registry))]

# --- SEARCH LOGIC ---
def get_recommendations(query, df):
    # Combine all text fields for "Rich Search"
    df = df.copy()
    df['combined_text'] = df['name'] + " " + df['domain'] + " " + df['use_cases'] + " " + df['description'] + " " + df['clients']
    
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['combined_text'].tolist() + [query])
    
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
    df['search_score'] = cosine_sim
    return df[df['search_score'] > 0.05].sort_values(by='search_score', ascending=False)

# --- UI LAYOUT ---
st.title("A. Model Marketplace")
st.markdown("---")

sidebar = st.sidebar
sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/Accenture.svg/2560px-Accenture.svg.png", width=150)
role = sidebar.selectbox("User Role", ["Data Scientist", "Administrator"])

# --- DATA SCIENTIST VIEW ---
if role == "Data Scientist":
    tab1, tab2, tab3 = st.tabs(["🔍 Browse Gallery", "🤝 Community Contributions", "🚀 Contribute Model"])
    
    with tab1:
        col_s1, col_s2 = st.columns([3, 1])
        with col_s1:
            q = st.text_input("💬 Search by task, client, or domain (e.g., 'Walmart' or 'Fraud detection')", placeholder="Try 'PwC' or 'NLP'...")
        with col_s2:
            d_filter = st.selectbox("Industry Domain", ["All"] + list(st.session_state.registry['domain'].unique()))

        # Filter and Search
        display_df = st.session_state.registry[st.session_state.registry['type'] == 'Official']
        if d_filter != "All":
            display_df = display_df[display_df['domain'] == d_filter]
        
        if q:
            results = get_recommendations(q, display_df)
        else:
            results = display_df

        st.subheader(f"Available Models ({len(results)})")
        for _, row in results.iterrows():
            with st.container():
                st.markdown(f"""
                <div class="model-card">
                    <div style="display: flex; justify-content: space-between;">
                        <span class="status-tag">{row['domain']}</span>
                        <span style="color: #A100FF; font-weight: bold;">{int(row.get('search_score', 0)*100) if q else ""}</span>
                    </div>
                    <h3>{row['name']}</h3>
                    <p>{row['description']}</p>
                    <div class="metric-text">
                        <b>Use Case:</b> {row['use_cases']} <br>
                        <b>Clients:</b> {row['clients']} <br>
                        <b>Accuracy:</b> {row['accuracy']*100}% | <b>Latency:</b> {row['latency']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"Deploy {row['name']}", key=f"btn_{row['name']}"):
                    st.success(f"Integration endpoint for {row['name']} generated!")

    with tab2:
        st.subheader("Experimental & Community Models")
        comm_df = st.session_state.registry[st.session_state.registry['type'] == 'Community']
        st
