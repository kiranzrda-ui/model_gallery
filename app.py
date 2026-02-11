import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import datetime

# --- CONFIGURATION & STYLING ---
st.set_page_config(page_title="Enterprise Model Gallery", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stCard { border: 1px solid #e6e9ef; padding: 20px; border-radius: 10px; background: white; }
    </style>
    """, unsafe_allow_html=True)

# --- MOCK DATA: THE MODEL REGISTRY ---
if 'model_data' not in st.session_state:
    st.session_state.model_data = pd.DataFrame([
        {"id": 1, "name": "Llama-3-70B", "provider": "Open Source", "task": "Text Generation", "accuracy": 0.88, "latency": "450ms", "license": "Llama 3", "usage_count": 1240, "description": "High performance LLM for complex reasoning and coding tasks."},
        {"id": 2, "name": "Enterprise-Churn-V2", "provider": "Internal", "task": "Classification", "accuracy": 0.94, "latency": "12ms", "license": "Proprietary", "usage_count": 850, "description": "Predicts customer churn based on historical CRM and billing data."},
        {"id": 3, "name": "BERT-Sentiment-Prod", "provider": "Internal", "task": "NLP", "accuracy": 0.91, "latency": "25ms", "license": "Proprietary", "usage_count": 2100, "description": "Optimized BERT for real-time sentiment analysis of support tickets."},
        {"id": 4, "name": "ResNet-50-ImageNet", "provider": "Open Source", "task": "Computer Vision", "accuracy": 0.76, "latency": "40ms", "license": "Apache 2.0", "usage_count": 430, "description": "Standard image classification model trained on ImageNet datasets."},
        {"id": 5, "name": "Fraud-Detection-LightGBM", "provider": "Internal", "task": "Tabular", "accuracy": 0.98, "latency": "5ms", "license": "Proprietary", "usage_count": 3200, "description": "Ultra-low latency fraud detection model for transactional banking."},
    ])

if 'logs' not in st.session_state:
    st.session_state.logs = []

# --- HELPER: SEARCH LOGIC (SIMULATED AI CHAT) ---
def search_models(query):
    docs = st.session_state.model_data['description'].tolist()
    vectorizer = TfidfVectorizer().fit_transform(docs + [query])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity([vectors[-1]], vectors[:-1])[0]
    st.session_state.model_data['score'] = cosine_sim
    # Log the search for admin
    st.session_state.logs.append({"query": query, "timestamp": datetime.datetime.now(), "success": any(cosine_sim > 0.1)})
    return st.session_state.model_data[st.session_state.model_data['score'] > 0.1].sort_values(by='score', ascending=False)

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🛠 Model Gallery Hub")
role = st.sidebar.radio("Switch View", ["Data Scientist (Consumer)", "Administrator (Governance)"])

# --- VIEW 1: DATA SCIENTIST (CONSUMER/CONTRIBUTOR) ---
if role == "Data Scientist (Consumer)":
    st.title("🔍 Model Discovery")
    
    # Chat Interface
    chat_query = st.text_input("💬 Ask the Gallery (e.g., 'I need a fast model for customer sentiment')", "")
    
    if chat_query:
        results = search_models(chat_query)
        st.subheader(f"Recommended Models for: '{chat_query}'")
        
        if not results.empty:
            for _, row in results.iterrows():
                with st.expander(f"📦 {row['name']} ({row['task']}) - Match: {int(row['score']*100)}%"):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Accuracy", f"{row['accuracy']*100}%")
                    col2.metric("Latency", row['latency'])
                    col3.metric("License", row['license'])
                    st.write(f"**Description:** {row['description']}")
                    st.button(f"Consume {row['name']} API", key=row['id'])
        else:
            st.warning("No models matched your specific query. Try broad terms like 'NLP' or 'Fast'.")

    # Full Gallery
    st.divider()
    st.subheader("All Available Models")
    st.dataframe(st.session_state.model_data.drop(columns=['score'] if 'score' in st.session_state.model_data else []))

# --- VIEW 2: ADMINISTRATOR DASHBOARD ---
else:
    st.title("📈 Governance & Efficacy Dashboard")
    
    # Top Row Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Models", len(st.session_state.model_data))
    m2.metric("Avg. Search Efficacy", "84%", "2%")
    m3.metric("Active API Consumptions", "7.2k")

    # Charts
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Model Popularity (Consumption)")
        fig = px.bar(st.session_state.model_data, x='name', y='usage_count', color='provider', template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        st.subheader("Search Queries (Drill-down)")
        if st.session_state.logs:
            log_df = pd.DataFrame(st.session_state.logs)
            st.table(log_df.tail(5))
        else:
            st.info("No search logs recorded yet.")

    # Efficacy Breakdown
    st.divider()
    st.subheader("Model Inventory Health")
    health_data = pd.DataFrame({
        "Model": st.session_state.model_data['name'],
        "Drift Score": [0.02, 0.05, 0.01, 0.08, 0.02],
        "Last Retrained": ["2 days ago", "1 month ago", "5 days ago", "1 year ago", "Yesterday"]
    })
    st.table(health_data)
