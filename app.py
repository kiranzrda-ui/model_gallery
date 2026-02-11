import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import datetime
import random

# --- CONFIGURATION ---
st.set_page_config(page_title="Accenture AI Marketplace", layout="wide")

# CUSTOM CSS: Compact Cards, Responsive Grid, and Light Purple Multiselect Tags
st.markdown("""
    <style>
    :root { --accent-purple: #A100FF; --light-purple: #F3E5F5; }
    
    /* 1. Light Purple Multiselect Tags */
    span[data-baseweb="tag"] {
        background-color: var(--light-purple) !important;
        color: var(--accent-purple) !important;
        border: 1px solid #D1C4E9 !important;
    }
    
    /* 2. Compact Model Tiles */
    .model-card {
        border: 1px solid #e0e0e0;
        border-top: 3px solid var(--accent-purple);
        padding: 12px;
        background-color: #ffffff;
        margin-bottom: 10px;
        min-height: 280px; /* Reduced from 450px */
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        border-radius: 4px;
        transition: transform 0.2s;
    }
    .model-card:hover { transform: translateY(-3px); box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
    
    .model-header { display: flex; justify-content: space-between; align-items: flex-start; }
    .domain-tag { font-size: 0.65rem; font-weight: bold; color: var(--accent-purple); text-transform: uppercase; }
    .model-title { font-size: 0.95rem; font-weight: 700; color: #000; margin-top: 4px; line-height: 1.2; }
    .model-desc { font-size: 0.8rem; color: #555; margin: 8px 0; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
    
    /* Metrics Row */
    .compact-metrics { 
        display: flex; 
        justify-content: space-between; 
        background: #f8f9fa; 
        padding: 6px; 
        border-radius: 4px;
        font-size: 0.75rem;
    }
    
    /* Buttons */
    .stButton>button { background-color: #000; color: white; border-radius: 0px; height: 32px; font-size: 0.75rem; width: 100%; }
    .stButton>button:hover { background-color: var(--accent-purple); }
    </style>
    """, unsafe_allow_html=True)

# --- PERSISTENCE LAYER ---
REGISTRY_PATH = "model_registry.csv"
REQUESTS_PATH = "requests_log.csv"

def init_storage():
    if not os.path.exists(REGISTRY_PATH):
        initial_data = []
        domains = ["Finance", "HR", "Procurement", "Supply Chain", "IT", "Marketing", "Legal"]
        users = ["John Doe", "Jane Nu", "Sam King"]
        for i in range(40):
            acc = random.choice([0.99, 0.92, 0.75]) # High, Med, Low samples
            lat = random.choice([30, 50, 80])       # Low, Med, High samples
            initial_data.append({
                "name": f"{random.choice(domains)}-{i+100}", "domain": random.choice(domains),
                "type": "Official" if i < 10 else "Community", "accuracy": acc, "latency": lat,
                "clients": "Fortune 500", "use_cases": "Enterprise Automation",
                "description": "High-density model for specific business unit optimization.",
                "contributor": "System" if i < 10 else random.choice(users), "usage": random.randint(100, 5000)
            })
        pd.DataFrame(initial_data).to_csv(REGISTRY_PATH, index=False)
    
    if not os.path.exists(REQUESTS_PATH):
        pd.DataFrame(columns=["model_name", "requester", "status", "timestamp"]).to_csv(REQUESTS_PATH, index=False)

init_storage()
registry_df = pd.read_csv(REGISTRY_PATH)
requests_df = pd.read_csv(REQUESTS_PATH)

# --- AUTH & NAVIGATION ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/Accenture.svg/2560px-Accenture.svg.png", width=120)
    user_role = st.selectbox("Login Profile", ["John Doe", "Jane Nu", "Sam King", "Nat Patel (Leader)", "Admin"])
    st.divider()
    
    st.subheader("Advanced Filtering")
    # Filters mapping to requirements
    acc_labels = st.multiselect("Accuracy Level", ["High (>98%)", "Medium (80-97%)", "Low (<80%)"], default=["High (>98%)", "Medium (80-97%)", "Low (<80%)"])
    lat_labels = st.multiselect("Latency Level", ["Low (<40ms)", "Med (41-60ms)", "High (>60ms)"], default=["Low (<40ms)", "Med (41-60ms)", "High (>60ms)"])

# --- HELPERS ---
def filter_df(df):
    # Map selection to actual data ranges
    if not acc_labels: return pd.DataFrame()
    acc_query = []
    if "High (>98%)" in acc_labels: acc_query.append(df['accuracy'] >= 0.98)
    if "Medium (80-97%)" in acc_labels: acc_query.append((df['accuracy'] >= 0.80) & (df['accuracy'] < 0.98))
    if "Low (<80%)" in acc_labels: acc_query.append(df['accuracy'] < 0.80)
    df = df[pd.concat(acc_query, axis=1).any(axis=1)]
    
    lat_query = []
    if "Low (<40ms)" in lat_labels: lat_query.append(df['latency'] < 40)
    if "Med (41-60ms)" in lat_labels: lat_query.append((df['latency'] >= 41) & (df['latency'] <= 60))
    if "High (>60ms)" in lat_labels: lat_query.append(df['latency'] > 60)
    df = df[pd.concat(lat_query, axis=1).any(axis=1)]
    return df

# --- VIEW: CONSUMER (JOHN, JANE, SAM) ---
if user_role in ["John Doe", "Jane Nu", "Sam King"]:
    st.title(f"Consumer Hub: {user_role}")
    t_gal, t_con, t_my = st.tabs(["🏛 Model Gallery", "🚀 Contribute Model", "👤 MyModels"])
    
    with t_gal:
        q = st.text_input("Search models by keyword, client or task...")
        # Search & Filter
        display_df = filter_df(registry_df)
        if q:
            display_df['blob'] = display_df.astype(str).apply(' '.join, axis=1)
            vectorizer = TfidfVectorizer(stop_words='english')
            matrix = vectorizer.fit_transform(display_df['blob'].tolist() + [q])
            display_df['score'] = cosine_similarity(matrix[-1], matrix[:-1])[0]
            display_df = display_df[display_df['score'] > 0].sort_values('score', ascending=False)
        
        # Grid Display
        for i in range(0, len(display_df), 3):
            cols = st.columns(3)
            for j in range(3):
                if i+j < len(display_df):
                    row = display_df.iloc[i+j]
                    with cols[j]:
                        st.markdown(f"""
                        <div class="model-card">
                            <div>
                                <div class="model-header">
                                    <span class="domain-tag">{row['domain']}</span>
                                    <span class="type-badge {'badge-official' if row['type']=='Official' else 'badge-community'}">{row['type']}</span>
                                </div>
                                <div class="model-title">{row['name']}</div>
                                <div class="model-desc">{row['description']}</div>
                            </div>
                            <div>
                                <div class="compact-metrics">
                                    <span><b>ACC:</b> {int(row['accuracy']*100)}%</span>
                                    <span><b>LAT:</b> {row['latency']}ms</span>
                                </div>
                                <div style="font-size: 0.65rem; color: #888; margin: 4px 0;">By: {row['contributor']}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        if st.button("Request Access", key=f"req_{row['name']}"):
                            new_req = pd.DataFrame([{"model_name": row['name'], "requester": user_role, "status": "Pending", "timestamp": str(datetime.datetime.now())}])
                            pd.concat([requests_df, new_req]).to_csv(REQUESTS_PATH, index=False)
                            st.toast("Request sent for approval.")

    with t_con:
        with st.form("con_form", clear_on_submit=True):
            st.subheader("New Intelligence Asset")
            fn, fd = st.columns(2)
            f_name = fn.text_input("Name")
            f_dom = fd.selectbox("Domain", ["Finance", "HR", "Procurement", "Supply Chain", "IT", "Marketing", "Legal"])
            f_desc = st.text_area("Functionality Description")
            f_acc = st.slider("Accuracy", 0.5, 1.0, 0.95)
            f_lat = st.number_input("Latency (ms)", 5, 500, 35)
            if st.form_submit_button("Publish Model"):
                new_mod = pd.DataFrame([{"name": f_name, "domain": f_dom, "type": "Community", "accuracy": f_acc, "latency": f_lat, "clients": "Internal", "use_cases": "New Entry", "description": f_desc, "contributor": user_role, "usage": 0}])
                pd.concat([registry_df, new_mod]).to_csv(REGISTRY_PATH, index=False)
                st.success("Model persisted and searchable.")

    with t_my:
        st.subheader("Your Impact Metrics")
        user_mods = registry_df[registry_df['contributor'] == user_role]
        if not user_mods.empty:
            c1, c2, c3 = st.columns(3)
            c1.metric("Assets", len(user_mods))
            c2.metric("Total Usage", user_mods['usage'].sum())
            c3.metric("Avg Quality", f"{int(user_mods['accuracy'].mean()*100)}%")
            st.dataframe(user_mods[['name', 'domain', 'accuracy', 'usage']], use_container_width=True)
        else:
            st.info("No contributions yet.")

# --- VIEW: LEADER (NAT PATEL) ---
elif user_role == "Nat Patel (Leader)":
    st.title("Approval Portal: Nat Patel")
    pending = requests_df[requests_df['status'] == "Pending"]
    if not pending.empty:
        for idx, row in pending.iterrows():
            col_a, col_b, col_c = st.columns([3, 2, 1])
            col_a.write(f"**{row['requester']}** wants access to **{row['model_name']}**")
            if col_c.button("Approve", key=f"ap_{idx}"):
                requests_df.at[idx, 'status'] = "Approved"
                requests_df.to_csv(REQUESTS_PATH, index=False)
                st.rerun()
    else:
        st.success("No pending approvals.")

# --- VIEW: ADMIN ---
else:
    st.title("Enterprise Governance")
    tabs = st.tabs(["Global Analytics", "Audit Logs"])
    with tabs[0]:
        col1, col2 = st.columns(2)
        fig = px.bar(registry_df, x='domain', y='usage', color='type', barmode='group', title="Usage by Business Unit")
        col1.plotly_chart(fig, use_container_width=True)
        fig2 = px.scatter(registry_df, x='accuracy', y='latency', color='domain', size='usage', title="Asset Portfolio Distribution")
        col2.plotly_chart(fig2, use_container_width=True)
    with tabs[1]:
        st.subheader("Access History")
        st.dataframe(requests_df, use_container_width=True)
