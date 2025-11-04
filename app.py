import streamlit as st
from streamlit_option_menu import option_menu
import base64
from pathlib import Path
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

# Try optional imports
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except Exception:
    STREAMLIT_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    from streamlit_option_menu import option_menu
    OPTIONMENU_AVAILABLE = True
except Exception:
    OPTIONMENU_AVAILABLE = False

# Paths
MODEL_PATH = Path("model.pkl")
DATA_PATH = Path("cleaned_drugs_dataset.csv")

# Utility loaders
def load_model(path=MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

def load_data(path=DATA_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at {path}")
    return pd.read_csv(path)

# Safe label map builder (maps human labels -> encoded ints or vice versa depending on column types)
def build_label_maps(df):
    maps = {}
    cols = ['generic_name','medical_condition','side_effects','pregnancy_category','rx_otc','csa']
    for col in cols:
        if col in df.columns:
            unique = df[col].dropna().unique()
            # If values are strings, map label->value; else map stringified value->value
            if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
                maps[col] = {str(v): v for v in unique}
            else:
                # numeric encoded values — provide string keys for dropdowns
                maps[col] = {f"encoded_{int(v)}": int(v) for v in sorted(unique)}
    return maps

# SHAP helper
def compute_shap_and_save(model, X, out_dir=Path("shap_outputs")):
    out_dir.mkdir(parents=True, exist_ok=True)
    if not SHAP_AVAILABLE:
        print("SHAP is not installed; skipping SHAP computation.")
        return None
    try:
        explainer = shap.Explainer(model, X)
        shap_vals = explainer(X)
        # Summary bar
        plt.figure(figsize=(8,4))
        shap.plots.bar(shap_vals, max_display=10, show=False)
        plt.tight_layout()
        bar_path = out_dir / "shap_bar.png"
        plt.savefig(bar_path)
        plt.close()
        # Beeswarm
        plt.figure(figsize=(8,4))
        shap.plots.beeswarm(shap_vals, max_display=12, show=False)
        plt.tight_layout()
        bee_path = out_dir / "shap_beeswarm.png"
        plt.savefig(bee_path)
        plt.close()
        print(f"SHAP plots saved to: {bar_path} and {bee_path}")
        return shap_vals
    except Exception as e:
        warnings.warn(f"Failed to compute SHAP: {e}")
        return None

# CLI mode: summary + optional SHAP
def run_cli():
    print("Running in CLI fallback mode (Streamlit not available).\n")
    try:
        df = load_data()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    try:
        model = load_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

    print("Dataset columns:", df.columns.tolist())
    print(f"Rows: {len(df):,}")
    if 'rating' in df.columns:
        print(f"Rating mean: {df['rating'].replace(0, np.nan).dropna().mean():.3f}")

    maps = build_label_maps(df)
    print("Built label maps for columns:", list(maps.keys()))

    # Try a quick prediction if model exists and dataset has required columns
    required = ['generic_name','medical_condition','no_of_reviews','side_effects','csa','pregnancy_category','rx_otc','alcohol']
    if model is not None and all(c in df.columns for c in required):
        sample = df[required].iloc[:5]
        try:
            preds = model.predict(sample.values)
            print("Sample predictions:", preds)
        except Exception as e:
            print(f"Model prediction failed on sample: {e}")

    # SHAP
    if model is not None and all(c in df.columns for c in required):
        X = df[required].dropna().astype(float)
        compute_shap_and_save(model, X)
    else:
        print("Skipping SHAP (missing model or required columns)")

# Streamlit app — only built if streamlit is available
def run_streamlit_app():
    # Local import guard
    import streamlit as st
    from streamlit_option_menu import option_menu

    st.set_page_config(page_title="Drug Rating Predictor", page_icon="💊", layout="wide")

    # Load data and model (with user-friendly error messages)
    try:
        df = load_data()
    except Exception as e:
        st.error(f"Dataset not found: {e}")
        st.stop()
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Model not found: {e}")
        st.stop()

    X = df[['generic_name','medical_condition','no_of_reviews','side_effects','csa','pregnancy_category','rx_otc','alcohol']]
    label_maps = build_label_maps(df)

    # Build UI
    with st.sidebar:
        page = option_menu("Menu", ["Home","Predict","Explain","Analytics","Data"], icons=["house","star","search","bar-chart","table"], menu_icon="cast")
        theme = st.radio("Theme", ["Light","Dark"], index=0)

    if page == "Home":
        st.title("💊 Drug Rating Predictor")
        st.write("Use the Predict tab to make predictions, Explain for SHAP, Analytics for charts.")
        st.write(f"Dataset rows: {len(df):,}")

    if page == "Predict":
        st.header("🔮 Predict a Drug Rating")
        # friendly dropdowns where available
        gen_val = None
        med_val = None
        if 'generic_name' in label_maps:
            gen_label = st.selectbox('Drug', list(label_maps['generic_name'].keys()))
            gen_val = label_maps['generic_name'][gen_label]
        else:
            gen_val = st.number_input('generic_name (encoded)', min_value=0, value=0)
        if 'medical_condition' in label_maps:
            med_label = st.selectbox('Medical condition', list(label_maps['medical_condition'].keys()))
            med_val = label_maps['medical_condition'][med_label]
        else:
            med_val = st.number_input('medical_condition (encoded)', min_value=0, value=0)

        reviews = st.number_input('Number of reviews', min_value=0, value=0)
        if 'side_effects' in label_maps:
            se_label = st.selectbox('Side effects (choose)', list(label_maps['side_effects'].keys()))
            se_val = label_maps['side_effects'][se_label]
        else:
            se_val = st.number_input('side_effects (encoded)', min_value=0, value=0)

        preg = st.selectbox('Pregnancy category', list(label_maps.get('pregnancy_category', {0:0}).keys()))
        preg_val = label_maps.get('pregnancy_category', {str(0):0}).get(preg, 0)
        rx = st.selectbox('RX/OTC', list(label_maps.get('rx_otc', {0:0}).keys()))
        rx_val = label_maps.get('rx_otc', {str(0):0}).get(rx, 0)
        csa_val = st.selectbox('CSA', list(label_maps.get('csa', {0:0}).keys()))
        csa_val = label_maps.get('csa', {str(0):0}).get(csa_val, 0)
        alcohol = st.selectbox('Alcohol warning', [0,1])

        if st.button('Predict'):
            X_input = np.array([[gen_val, med_val, reviews, se_val, csa_val, preg_val, rx_val, alcohol]])
            try:
                pred = model.predict(X_input)[0]
                st.success(f"Predicted rating: {pred:.2f} / 10")
                st.session_state.last_input = X_input
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    if page == "Explain":
        st.header("🔍 Explain with SHAP")
        if not SHAP_AVAILABLE:
            st.warning("SHAP not installed in this environment. Install shap to enable explanations.")
        else:
            try:
                explainer = shap.Explainer(model, X)
                sample = X.sample(min(200, len(X)), random_state=42)
                with st.spinner('Computing SHAP...'):
                    shap_vals = explainer(sample)
                st.subheader('Global feature importance')
                st.pyplot(shap.plots.bar(shap_vals, max_display=10, show=False))
                st.subheader('Beeswarm')
                st.pyplot(shap.plots.beeswarm(shap_vals, max_display=12, show=False))
                # Individual explanation
                if 'last_input' in st.session_state:
                    st.subheader('Explain last prediction')
                    try:
                        inst = st.session_state.last_input
                        inst_vals = np.asarray(inst, dtype=float)
                        shap_inst = explainer(inst_vals)
                        st.pyplot(shap.plots.waterfall(shap_inst[0], show=False))
                    except Exception as e:
                        st.error(f"Failed to explain last input: {e}")
            except Exception as e:
                st.error(f"SHAP failed: {e}")

    if page == "Analytics":
        st.header('📈 Analytics')
        st.subheader('Rating distribution')
        fig, ax = plt.subplots()
        sns = __import__('seaborn')
        sns.histplot(df['rating'].replace(0, np.nan).dropna(), kde=True, ax=ax)
        st.pyplot(fig)

    if page == "Data":
        st.header('📁 Data preview')
        st.dataframe(df.head())

# Entrypoint
def main():
    parser = argparse.ArgumentParser(description='Drug Rating App')
    parser.add_argument('--cli', action='store_true', help='Run CLI fallback (no Streamlit)')
    args = parser.parse_args()

    if args.cli or not STREAMLIT_AVAILABLE:
        run_cli()
    else:
        run_streamlit_app()

if __name__ == '__main__':
    main()

