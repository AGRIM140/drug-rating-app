# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os

st.set_page_config(page_title="Drug Rating Predictor", page_icon="💊", layout="wide")

# -------------------------
# Utility: load model + data
# -------------------------
@st.cache_data
def load_model(path="model.pkl"):
    with open(path, "rb") as f:
        m = pickle.load(f)
    return m

@st.cache_data
def load_data(path="cleaned_drugs_dataset.csv"):
    df = pd.read_csv(path)
    return df

MODEL_PATH = "model.pkl"
DATA_PATH = "cleaned_drugs_dataset.csv"

# If files are missing, show a helpful message
if not os.path.exists(MODEL_PATH):
    st.warning("model.pkl not found in working directory. Place your trained model file named 'model.pkl' here.")
if not os.path.exists(DATA_PATH):
    st.warning("cleaned_drugs_dataset.csv not found in working directory. Place your cleaned dataset here.")

if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    model = None

if os.path.exists(DATA_PATH):
    df = load_data(DATA_PATH)
else:
    df = pd.DataFrame()  # fallback empty

# -------------------------
# Build mapping dictionaries from dataframe
# -------------------------
# The cleaned dataset contains encoded numeric columns and some readable text columns.
# We'll create mappings where text exists (drug_name -> generic_name encoded int,
# medical_condition_description -> medical_condition encoded int).
mappings = {}

if not df.empty:
    # Generic: use 'drug_name' -> 'generic_name' mapping
    if "drug_name" in df.columns and "generic_name" in df.columns:
        gen_map = dict(zip(df["drug_name"].astype(str), df["generic_name"].astype(int)))
        mappings["generic_name"] = gen_map
    # Medical condition: prefer description if present
    if "medical_condition_description" in df.columns and "medical_condition" in df.columns:
        med_map = dict(zip(df["medical_condition_description"].astype(str), df["medical_condition"].astype(int)))
        mappings["medical_condition"] = med_map

    # Side-effects: if only encoded integers exist, show top encoded values and frequencies
    if "side_effects" in df.columns:
        side_counts = df["side_effects"].value_counts().head(50)
        # side_counts index may be ints (encoded) — build mapping label->encoded
        side_map = {f"encoded_{int(k)} (freq={int(v)})": int(k) for k, v in side_counts.items()}
        mappings["side_effects"] = side_map

    # pregnancy_category / rx_otc / csa: if these are textual keep their labels, else show unique codes
    for col in ["pregnancy_category", "rx_otc", "csa"]:
        if col in df.columns:
            # if textual
            if df[col].dtype == "object" or df[col].dtype == "string":
                unique_vals = df[col].fillna("Unknown").astype(str).unique().tolist()
                mappings[col] = {str(v): v for v in unique_vals}
            else:
                unique_vals = np.sort(df[col].dropna().unique()).tolist()
                mappings[col] = {f"encoded_{int(v)}": int(v) for v in unique_vals}

# Optionally save mappings for later use
# joblib.dump(mappings, "mappings.pkl")

# -------------------------
# CUSTOM CSS and Theme Toggle
# -------------------------
dark_css = """
<style>
body { background: #0e1117; color: #e6eef8; }
.stButton>button { background: linear-gradient(90deg,#4FACFE,#00F2FE); color: white; }
.card { background: rgba(255,255,255,0.04); border-radius:12px; padding:16px; box-shadow: none; }
</style>
"""

light_css = """
<style>
body { background: #f7f9fc; color: #0b1726; }
.stButton>button { background: linear-gradient(90deg,#4FACFE,#00F2FE); color: white; }
.card { background: #ffffff; border-radius:12px; padding:16px; box-shadow: 0 6px 18px rgba(12,20,40,0.06); }
</style>
"""

# Theme selector in sidebar
theme = st.sidebar.radio("Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown(dark_css, unsafe_allow_html=True)
else:
    st.markdown(light_css, unsafe_allow_html=True)

# -------------------------
# App Layout: Sidebar Navigation
# -------------------------
st.sidebar.markdown("## Navigation")
page = st.sidebar.selectbox("Go to", ["Home", "Predict", "Explain", "Analytics", "Bulk Predict", "About"])

# -------------------------
# HOME
# -------------------------
if page == "Home":
    st.title("💊 Drug Rating Predictor")
    st.markdown(
        """
        Welcome — this app predicts drug ratings using a trained Random Forest model.
        Use **Predict** to get a single prediction, **Bulk Predict** for CSVs, and **Explain** to inspect model behavior.
        """
    )
    st.markdown("### Quick stats from dataset")
    if not df.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("Drugs (rows)", f"{len(df)}")
        if "rating" in df.columns:
            c2.metric("Average rating", f"{df['rating'].replace(0, np.nan).dropna().mean():.2f}")
        else:
            c2.metric("Average rating", "N/A")
        c3.metric("Unique drugs", f"{df['drug_name'].nunique() if 'drug_name' in df.columns else 'N/A'}")
    st.info("Tip: If the dropdowns don't show expected labels, make sure `cleaned_drugs_dataset.csv` is the cleaned version used to train your model.")

# -------------------------
# PREDICT (Single)
# -------------------------
if page == "Predict":
    st.title("🔮 Predict Rating (single)")

    st.markdown("Fill inputs below. Where possible choose friendly labels from dropdowns; otherwise enter encoded values.")
    # Input cards
    with st.container():
        left, right = st.columns([2, 3])

        with left:
            st.subheader("Identifiers")
            # Generic name dropdown if mapping exists
            if "generic_name" in mappings:
                gen_names = list(mappings["generic_name"].keys())
                selected_gen = st.selectbox("Drug name", gen_names)
                gen_encoded = mappings["generic_name"][selected_gen]
            else:
                gen_encoded = st.number_input("Generic Name (encoded)", min_value=0, value=0, help="Encoded integer for generic_name")

            # Medical condition dropdown if mapping exists
            if "medical_condition" in mappings:
                med_labels = list(mappings["medical_condition"].keys())
                selected_med = st.selectbox("Medical condition", med_labels)
                med_encoded = mappings["medical_condition"][selected_med]
            else:
                med_encoded = st.number_input("Medical condition (encoded)", min_value=0, value=0)

            # Side effects: show friendly list of top encodings
            st.markdown("---")
            st.subheader("Side effects")
            if "side_effects" in mappings:
                side_labels = list(mappings["side_effects"].keys())
                selected_side = st.selectbox("Choose common side-effect encoding", side_labels)
                side_encoded = mappings["side_effects"][selected_side]
                side_manual = st.checkbox("Or enter encoded value manually")
                if side_manual:
                    side_encoded = st.number_input("Side effects (encoded)", min_value=0, value=int(side_encoded))
            else:
                side_encoded = st.number_input("Side effects (encoded)", min_value=0, value=0)

            st.markdown("---")
            st.subheader("Other flags")
            # pregnancy_category / rx_otc / csa
            if "pregnancy_category" in mappings:
                preg_label = st.selectbox("Pregnancy category", list(mappings["pregnancy_category"].keys()))
                preg_encoded = mappings["pregnancy_category"][preg_label]
            else:
                preg_encoded = st.number_input("Pregnancy category (encoded)", min_value=0, value=0)

            if "rx_otc" in mappings:
                rx_label = st.selectbox("RX/OTC", list(mappings["rx_otc"].keys()))
                rx_encoded = mappings["rx_otc"][rx_label]
            else:
                rx_encoded = st.number_input("RX / OTC (encoded)", min_value=0, value=0)

            if "csa" in mappings:
                csa_label = st.selectbox("CSA schedule", list(mappings["csa"].keys()))
                csa_encoded = mappings["csa"][csa_label]
            else:
                csa_encoded = st.number_input("csa (encoded)", min_value=0, value=0)

            alcohol = st.selectbox("Alcohol warning", [0, 1], help="1 indicates alcohol warning present")

        with right:
            st.subheader("Extras & preview")
            st.markdown("Preview the input vector your model will receive (encoded form).")
            preview_df = pd.DataFrame(
                {
                    "feature": ["generic_name", "medical_condition", "no_of_reviews", "side_effects", "csa", "pregnancy_category", "rx_otc", "alcohol"],
                    "value": [gen_encoded, med_encoded, 0, side_encoded, csa_encoded, preg_encoded, rx_encoded, alcohol]
                }
            )
            st.table(preview_df)

            # allow reviews input
            reviews = st.number_input("Number of reviews", min_value=0, value=0)

    st.markdown("---")
    predict_btn = st.button("🔍 Predict Rating")

    if predict_btn:
        if model is None:
            st.error("Model not loaded. Place model.pkl in working directory.")
        else:
            X = np.array([[gen_encoded, med_encoded, reviews, side_encoded, csa_encoded, preg_encoded, rx_encoded, alcohol]])
            try:
                pred = model.predict(X)[0]
                st.metric("Predicted Rating", f"{pred:.2f} / 10")
                st.success("Prediction complete")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# -------------------------
# EXPLAIN: Feature importance and SHAP (if available)
# -------------------------
if page == "Explain":
    st.title("🧾 Model explainability")

    if model is None:
        st.warning("Model not loaded — can't show explainability.")
    else:
        st.subheader("Feature importances (from model)")
        try:
            importances = model.feature_importances_
            feat_names = ["generic_name", "medical_condition", "no_of_reviews", "side_effects",
                          "csa", "pregnancy_category", "rx_otc", "alcohol"]
            fi = pd.Series(importances, index=feat_names).sort_values(ascending=True)

            fig, ax = plt.subplots(figsize=(8, 4))
            fi.plot(kind="barh", ax=ax)
            ax.set_title("Feature importances")
            st.pyplot(fig)
            fig.savefig("plot.png")
        except Exception as e:
            st.error(f"Could not get feature importances: {e}")

        st.markdown("---")
        st.subheader("SHAP explanation (optional)")
        st.write("If `shap` is installed and your model supports it, a SHAP summary will be generated.")
        try:
            import shap
            # If model is a tree model (RandomForest), TreeExplainer works
            explainer = shap.Explainer(model)
            # sample features from df to construct shap values (we need numeric array)
            if not df.empty:
                # construct X sample using columns in expected order
                X_sample = df[["generic_name","medical_condition","no_of_reviews","side_effects","csa","pregnancy_category","rx_otc","alcohol"]].sample(min(200, len(df)), random_state=42)
                st.write("Computing SHAP values (may take a moment)...")
                shap_values = explainer(X_sample)
                st.pyplot(shap.plots.beeswarm(shap_values, show=False))
            else:
                st.info("No dataset available to compute SHAP sample.")
        except Exception as e:
            st.info("SHAP unavailable or failed to run. Feature importances above are shown instead.")
            st.error(str(e))

# -------------------------
# ANALYTICS: quick charts from cleaned dataset
# -------------------------
if page == "Analytics":
    st.title("📊 Analytics Dashboard")
    if df.empty:
        st.info("Upload `cleaned_drugs_dataset.csv` to view analytics.")
    else:
        tabs = st.tabs(["Ratings", "Side Effects", "Top Drugs"])
        with tabs[0]:
            st.subheader("Rating distribution")
            fig, ax = plt.subplots(figsize=(8, 3.5))
            sns.histplot(df['rating'].replace(0, np.nan).dropna(), kde=True, ax=ax)
            ax.set_xlabel("Rating")
            st.pyplot(fig)

            st.subheader("Rating vs #reviews")
            fig2, ax2 = plt.subplots(figsize=(8, 3.5))
            sns.scatterplot(data=df, x='no_of_reviews', y='rating', alpha=0.6, ax=ax2)
            ax2.set_xscale('symlog')  # handle large outliers
            st.pyplot(fig2)

        with tabs[1]:
            st.subheader("Top reported side-effect encodings")
            if "side_effects" in df.columns:
                top_side = df['side_effects'].value_counts().head(20)
                fig3, ax3 = plt.subplots(figsize=(8, 3.5))
                sns.barplot(x=top_side.values, y=[str(x) for x in top_side.index], ax=ax3)
                ax3.set_xlabel("Frequency")
                ax3.set_ylabel("Side_effects (encoded)")
                st.pyplot(fig3)
            else:
                st.info("No side_effects column in dataset.")

        with tabs[2]:
            st.subheader("Top drugs by count")
            if "drug_name" in df.columns:
                top_drugs = df['drug_name'].value_counts().head(20)
                fig4, ax4 = plt.subplots(figsize=(8, 4))
                sns.barplot(x=top_drugs.values, y=top_drugs.index, ax=ax4)
                st.pyplot(fig4)
            else:
                st.info("No drug_name column in dataset.")

# -------------------------
# BULK PREDICTION (CSV)
# -------------------------
if page == "Bulk Predict":
    st.title("📁 Bulk CSV Prediction")
    st.markdown("Upload a CSV with the same encoded columns your model expects (generic_name, medical_condition, no_of_reviews, side_effects, csa, pregnancy_category, rx_otc, alcohol).")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            st.write("Preview uploaded file:")
            st.dataframe(uploaded_df.head())

            if model is not None:
                # check required columns
                req_cols = ["generic_name","medical_condition","no_of_reviews","side_effects","csa","pregnancy_category","rx_otc","alcohol"]
                missing = [c for c in req_cols if c not in uploaded_df.columns]
                if missing:
                    st.error(f"Uploaded CSV is missing required columns: {missing}")
                else:
                    preds = model.predict(uploaded_df[req_cols].values)
                    uploaded_df["predicted_rating"] = preds
                    st.success("Predictions done — preview below:")
                    st.dataframe(uploaded_df.head())
                    csv_result = uploaded_df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download predictions CSV", csv_result, "predictions.csv", "text/csv")
            else:
                st.error("Model not available in server.")
        except Exception as e:
            st.error(f"Failed to process uploaded file: {e}")

# -------------------------
# ABOUT
# -------------------------
if page == "About":
    st.title("ℹ️ About this app")
    st.markdown("""
    **What this does:** Predicts drug rating using encoded features and a RandomForest model.
    \n**How to get best results:** Use the cleaned dataset that was used during training (matching encodings).
    \n**Files expected in the same folder:** model.pkl, cleaned_drugs_dataset.csv
    """)
    st.markdown("### Deployment")
    st.code("streamlit run app.py")

# -------------------------
# End
# -------------------------




