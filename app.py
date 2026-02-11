import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# --- CONFIG & STYLING ---
st.set_page_config(page_title="Accenture AI Marketplace", layout="wide")

# Accenture CSS
st.markdown("""
    <style>
    :root { --accent-color: #A100FF; }
    .stApp { background-color: #ffffff; }
    .model-card {
        border: 1px solid #e0e0e0; border-top: 4px solid #A100FF;
        padding: 15px; background-color: #ffffff; margin-bottom: 20px;
        height: 450px; display: flex; flex-direction: column; justify-content: space-between;
    }
    .type-badge { font-size: 0.65rem; padding: 2px 6px; border-radius: 4px; font-weight: bold; float: right; }
    .badge-official { background-color: #e8dbff; color: #A100FF; }
    .badge-community { background-color: #f0f0f0; color: #666; }
    .model-title { font-size: 1.1rem; font-weight: 700; color: #000; min-height: 50px; }
    .metric-row { display: flex; justify-content: space-between; font-size: 0.75rem; border-top: 1px solid #eee; padding-top: 10px; }
    div[data-testid="stForm"] button { background-color: #A100FF !important; color: white !important; }
    .stButton>button { background-color: #000; color: white; border-radius: 0px; }
    </style>
    """, unsafe_allow_html=True)

# --- DATABASE PERSISTENCE LOGIC ---
REGISTRY_FILE = "model_registry.csv"
REQUESTS_FILE = "requests_log.csv"

def load_data():
    if os.path.exists(REGISTRY_FILE):
        return pd.read_csv(REGISTRY_FILE)
    else:
        # Initial Seed Data
        data = []
        domains = ["Finance", "HR", "Procurement", "Supply Chain", "IT", "Marketing", "Legal"]
        users = ["John Doe", "Jane Nu", "Sam King"]
        for i in range(50):
            dom = domains[i % len(domains)]
            acc = round(0.75 + (i * 0.004), 2) # Mix of high/med/low
            lat = 20 + (i * 2)
            data.append({
                "name": f"{dom}-Model-{i+100}", "domain": dom, "type": "Official" if i < 10 else "Community",
                "accuracy": acc, "latency": lat, "clients": "Global Enterprise",
                "use_cases": "Automated Processing", "description": f"Scalable {dom} intelligence asset.",
                "contributor": "System" if i < 10 else users[i % 3], "usage": 100 + (i * 5)
            })
        df = pd.DataFrame(data)
        df.to_csv(REGISTRY_FILE, index=False)
        return df

def load_requests():
    if os.path.exists(REQUESTS_FILE):
        return pd.read_csv(REQUESTS_FILE)
    return pd.DataFrame(columns=["model_name", "requester", "status", "timestamp"])

def save_data(df, file):
    df.to_csv(file, index=False)

# Load data into memory
if 'df' not in st.session_state:
    st.session_state.df = load_data()
if 'reqs' not in st.session_state:
    st.session_state.reqs = load_requests()

# --- SEARCH ENGINE ---
def search_engine(query, df):
    if not query: return df
    df = df.copy()
    df['blob'] = df.astype(str).apply(" ".join, axis=1)
    vectorizer = TfidfVectorizer(stop_words='english')
    matrix = vectorizer.fit_transform(df['blob'].tolist() + [query])
    scores = cosine_similarity(matrix[-1], matrix[:-1])[0]
    df['score'] = scores
    return df[df['score'] > 0.01].sort_values('score', ascending=False)

# --- SIDEBAR & AUTH ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/Accenture.svg/2560px-Accenture.svg.png", width=120)
    st.title("Gatekeeper")
    current_user = st.selectbox("Login As:", ["John Doe", "Jane Nu", "Sam King", "Nat Patel (Leader)", "Admin"])
    st.divider()
    
    # ACCURACY & LATENCY FILTERS
    st.subheader("Advanced Filters")
    acc_filter = st.multiselect("Accuracy Class", ["Highly Accurate (>98%)", "Medium (80-97%)", "Low (<80%)"], default=["Highly Accurate (>98%)", "Medium (80-97%)", "Low (<80%)"])
    lat_filter = st.multiselect("Latency Class", ["Low (<40ms)", "Medium (41-60ms)", "High (>60ms)"], default=["Low (<40ms)", "Medium (41-60ms)", "High (>60ms)"])

# --- FILTER LOGIC ---
def apply_filters(df):
    # Accuracy logic
    acc_conditions = []
    if "Highly Accurate (>98%)" in acc_filter: acc_conditions.append(df['accuracy'] >= 0.98)
    if "Medium (80-97%)" in acc_filter: acc_conditions.append((df['accuracy'] >= 0.80) & (df['accuracy'] < 0.98))
    if "Low (<80%)" in acc_filter: acc_conditions.append(df['accuracy'] < 0.80)
    if acc_conditions: df = df[pd.concat(acc_conditions, axis=1).any(axis=1)]

    # Latency logic
    lat_conditions = []
    if "Low (<40ms)" in lat_filter: lat_conditions.append(df['latency'] <= 40)
    if "Medium (41-60ms)" in lat_filter: lat_conditions.append((df['latency'] > 40) & (df['latency'] <= 60))
    if "High (>60ms)" in lat_filter: lat_conditions.append(df['latency'] > 60)
    if lat_conditions: df = df[pd.concat(lat_conditions, axis=1).any(axis=1)]
    return df

# --- MAIN INTERFACE ---
if current_user in ["John Doe", "Jane Nu", "Sam King"]:
    st.title(f"Welcome, {current_user}")
    t1, t2, t3 = st.tabs(["🏛 Model Gallery", "🚀 Contribute", "👤 My Dashboard"])

    with t1:
        query = st.text_input("Search Models by keyword, client or task...")
        results = search_engine(query, apply_filters(st.session_state.df))
        
        for i in range(0, len(results), 3):
            cols = st.columns(3)
            for j in range(3):
                if i+j < len(results):
                    row = results.iloc[i+j]
                    with cols[j]:
                        st.markdown(f"""<div class="model-card">
                            <div>
                                <span class="type-badge {'badge-official' if row['type']=='Official' else 'badge-community'}">{row['type']}</span>
                                <div style="color:#A100FF; font-size:0.7rem; font-weight:bold;">{row['domain']}</div>
                                <div class="model-title">{row['name']}</div>
                                <div style="font-size:0.8rem;">{row['description']}</div>
                                <div style="font-size:0.7rem; color:gray; margin-top:5px;">Contributor: {row['contributor']}</div>
                            </div>
                            <div class="metric-row">
                                <span><b>ACC:</b> {int(row['accuracy']*100)}%</span>
                                <span><b>LAT:</b> {row['latency']}ms</span>
                            </div>
                        </div>""", unsafe_allow_html=True)
                        if st.button("Request Access", key=f"req_{row['name']}"):
                            new_req = pd.DataFrame([{"model_name": row['name'], "requester": current_user, "status": "Pending", "timestamp": str(datetime.datetime.now())}])
                            st.session_state.reqs = pd.concat([st.session_state.reqs, new_req], ignore_index=True)
                            save_data(st.session_state.reqs, REQUESTS_FILE)
                            st.success("Request sent to Nat Patel.")

    with t2:
        with st.form("contribute", clear_on_submit=True):
            n = st.text_input("Model Name")
            d = st.selectbox("Domain", ["Finance", "HR", "Procurement", "Supply Chain", "IT", "Marketing", "Legal"])
            desc = st.text_area("Description")
            acc_val = st.slider("Accuracy", 0.5, 1.0, 0.85)
            lat_val = st.number_input("Latency (ms)", 5, 500, 50)
            if st.form_submit_button("Submit Model"):
                new_m = pd.DataFrame([{"name": n, "domain": d, "type": "Community", "accuracy": acc_val, "latency": lat_val, "clients": "Internal", "use_cases": "New Contribution", "description": desc, "contributor": current_user, "usage": 0}])
                st.session_state.df = pd.concat([st.session_state.df, new_m], ignore_index=True)
                save_data(st.session_state.df, REGISTRY_FILE)
                st.success("Model Published and Persisted!")

    with t3:
        st.subheader("My Submissions & Performance")
        my_mods = st.session_state.df[st.session_state.df['contributor'] == current_user]
        if not my_mods.empty:
            col_a, col_b = st.columns(2)
            col_a.metric("Models Contributed", len(my_mods))
            col_b.metric("Total Views/Usage", my_mods['usage'].sum())
            st.dataframe(my_mods[['name', 'domain', 'accuracy', 'latency', 'usage']], use_container_width=True)
        else:
            st.info("You haven't contributed any models yet.")

elif current_user == "Nat Patel (Leader)":
    st.title("Leader Approval Queue")
    st.subheader(f"Pending Requests for {current_user}")
    pending = st.session_state.reqs[st.session_state.reqs['status'] == "Pending"]
    if not pending.empty:
        for idx, row in pending.iterrows():
            c1, c2, c3 = st.columns([2, 2, 1])
            c1.write(f"**Model:** {row['model_name']}")
            c2.write(f"**User:** {row['requester']}")
            if c3.button("Approve", key=f"app_{idx}"):
                st.session_state.reqs.at[idx, 'status'] = "Approved"
                save_data(st.session_state.reqs, REQUESTS_FILE)
                st.rerun()
    else:
        st.success("All requests cleared.")

elif current_user == "Admin":
    st.title("Marketplace Administration")
    tabs = st.tabs(["Marketplace Metrics", "Request Logs", "All Models"])
    with tabs[0]:
        c1, c2 = st.columns(2)
        fig1 = px.sunburst(st.session_state.df, path=['domain', 'type'], values='usage', title="Usage by Domain")
        c1.plotly_chart(fig1)
        fig2 = px.scatter(st.session_state.df, x='accuracy', y='latency', color='type', size='usage', hover_name='name', title="Accuracy vs Latency")
        c2.plotly_chart(fig2)
    with tabs[1]:
        st.dataframe(st.session_state.reqs, use_container_width=True)
    with tabs[2]:
        st.dataframe(st.session_state.df, use_container_width=True)
