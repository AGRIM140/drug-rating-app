import streamlit as st
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
import base64

# ------------------------
# Load model and data
# ------------------------
model = pickle.load(open("model.pkl", "rb"))
df = pd.read_csv("cleaned_drugs_dataset.csv")

X = df[['generic_name', 'medical_condition', 'no_of_reviews', 'side_effects',
        'csa', 'pregnancy_category', 'rx_otc', 'alcohol']]

# Drop duplicates from label maps
label_maps = {}
for col in ['generic_name','medical_condition','side_effects','pregnancy_category','rx_otc','csa']:
    label_maps[col] = {v:k for k,v in enumerate(sorted(df[col].unique()))}

# SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# ------------------------
# Page Config
# ------------------------
st.set_page_config(page_title="Drug Rating Predictor", page_icon="💊", layout="wide")

# Theme toggle
if "theme" not in st.session_state:
    st.session_state.theme = "light"

if st.sidebar.button("🌙 Toggle Theme"):
    st.session_state.theme = "dark" if st.session_state.theme=="light" else "light"

if st.session_state.theme == "dark":
    bg = "#0e1117"; text = "white"; card = "#161b22"
else:
    bg = "#f7f9fc"; text = "black"; card = "white"

st.markdown(f"""
<style>
body {{background-color: {bg}; color: {text};}}
.metric-card {{background: {card}; padding:20px; border-radius:15px; box-shadow:0px 4px 10px rgba(0,0,0,0.2);}}
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    selected = option_menu(
        "📊 Drug Rating System",
        ["Home", "Predict Rating", "Explain Prediction", "Feature Importance", "Dataset Preview"],
        icons=["house", "activity", "search", "bar-chart", "table"],
        default_index=0,
    )

# Home
if selected == "Home":
    st.title("💊 Drug Rating Prediction Dashboard")
    st.write("Smart drug effectiveness prediction & explainability.")
    c1,c2,c3 = st.columns(3)
    for c,l in zip([c1,c2,c3],["✅ ML Model: Random Forest","✅ Explainability: SHAP","✅ Real‑time Rating Predictor"]):
        with c: st.markdown(f"<div class='metric-card'><h3>{l}</h3></div>",unsafe_allow_html=True)

# Predict
if selected == "Predict Rating":
    st.title("🔮 Predict Drug Rating")
    col1,col2 = st.columns(2)
    with col1:
        generic = st.selectbox("Generic Name", options=list(label_maps['generic_name'].keys()))
        med = st.selectbox("Medical Condition", options=list(label_maps['medical_condition'].keys()))
        reviews = st.number_input("Number of Reviews", min_value=0)
        se = st.selectbox("Side Effects", options=list(label_maps['side_effects'].keys()))
    with col2:
        csa = st.selectbox("CSA Schedule", options=list(label_maps['csa'].keys()))
        preg = st.selectbox("Pregnancy Category", options=list(label_maps['pregnancy_category'].keys()))
        rx = st.selectbox("RX/OTC", options=list(label_maps['rx_otc'].keys()))
        alcohol = st.selectbox("Alcohol Warning", [0,1])

    if st.button("Predict Rating ⭐"):
        X_input = np.array([[ label_maps['generic_name'][generic],
                              label_maps['medical_condition'][med], reviews,
                              label_maps['side_effects'][se],
                              label_maps['csa'][csa],
                              label_maps['pregnancy_category'][preg],
                              label_maps['rx_otc'][rx], alcohol]])
        pred = model.predict(X_input)[0]
        st.success(f"⭐ Predicted Rating: {pred:.2f}/10")
        st.session_state.last_input = X_input

# Explain Prediction
if selected == "Explain Prediction":
    st.title("🔍 SHAP Force Plot - Explanation")
    if "last_input" not in st.session_state:
        st.info("Run a prediction first!")
    else:
        sv = explainer.shap_values(st.session_state.last_input)
        fig = shap.force_plot(explainer.expected_value, sv, st.session_state.last_input, matplotlib=True)
        st.pyplot(fig)

# Global importance
if selected == "Feature Importance":
    st.title("📈 SHAP Feature Importance")
    fig1, ax1 = plt.subplots(); shap.summary_plot(shap_values, X, show=False); st.pyplot(fig1)
    fig2, ax2 = plt.subplots(); shap.summary_plot(shap_values, X, plot_type="bar", show=False); st.pyplot(fig2)

# Dataset
if selected == "Dataset Preview":
    st.title("📁 Dataset Preview")
    st.dataframe(df.head())
