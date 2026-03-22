"""
AnalyticGAN -- Interactive Streamlit Dashboard
Run: cd analyticgan && C:\\ProgramData\\anaconda3\\python.exe -m streamlit run app/streamlit_app.py
"""

import os, sys, pickle, joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import LabelEncoder

# ---- Paths ----
_app_dir = os.path.dirname(os.path.abspath(__file__))
BASE     = os.path.dirname(_app_dir)
CKPT_DIR = os.path.join(BASE, "checkpoints")
sys.path.insert(0, BASE)

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]

st.set_page_config(page_title="AnalyticGAN", page_icon="🔬", layout="wide")

# ---- Model Classes (inline) ----
class VGMEncoder:
    def __init__(self, n_components=10, eps=0.005):
        self.n_components = n_components
        self.eps = eps
        self.bgm = BayesianGaussianMixture(
            n_components=n_components,
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=0.001,
            max_iter=100, random_state=42, n_init=1)
        self.valid_components = None
        self.n_valid = None
    def fit(self, data):
        self.bgm.fit(np.asarray(data).reshape(-1, 1))
        self.valid_components = np.where(self.bgm.weights_ > self.eps)[0]
        self.n_valid = len(self.valid_components)
        return self
    def transform(self, data):
        data = np.asarray(data).reshape(-1, 1)
        means = self.bgm.means_[self.valid_components].flatten()
        stds = np.sqrt(self.bgm.covariances_[self.valid_components]).flatten()
        probs = self.bgm.predict_proba(data)[:, self.valid_components]
        mode_idx = []
        for p in probs:
            s = p.sum()
            p_norm = (p / s).astype(np.float64) if (s > 0 and np.isfinite(s)) \
                     else np.ones(self.n_valid) / self.n_valid
            mode_idx.append(np.random.choice(self.n_valid, p=p_norm))
        mode_idx = np.array(mode_idx)
        normalized = np.clip(
            (data.flatten() - means[mode_idx]) / (4 * stds[mode_idx] + 1e-8),
            -0.99, 0.99)
        one_hot = np.zeros((len(data), self.n_valid), dtype=np.float32)
        one_hot[np.arange(len(data)), mode_idx] = 1
        return np.column_stack([normalized, one_hot]).astype(np.float32)
    def inverse_transform(self, encoded):
        encoded = np.asarray(encoded)
        means = self.bgm.means_[self.valid_components].flatten()
        stds = np.sqrt(self.bgm.covariances_[self.valid_components]).flatten()
        mode_idx = np.argmax(encoded[:, 1:], axis=1)
        return encoded[:, 0] * 4 * stds[mode_idx] + means[mode_idx]

class TabularPreprocessor:
    def __init__(self, max_gmm_components=10, eps=0.005):
        self.max_gmm_components = max_gmm_components
        self.eps = eps
        self.continuous_cols = []
        self.categorical_cols = []
        self.target_col = None
        self.vgm_encoders = {}
        self.label_encoders = {}
        self.cat_dims = {}
        self.output_info = []
        self.output_dim = 0
    def inverse_transform(self, tensor):
        data = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else tensor
        result, idx = {}, 0
        for kind, col, size in self.output_info:
            if kind == "continuous":
                w = 1 + self.vgm_encoders[col].n_valid
                result[col] = self.vgm_encoders[col].inverse_transform(data[:, idx:idx+w])
                idx += w
            else:
                n_cat = self.cat_dims[col]
                result[col] = self.label_encoders[col].inverse_transform(
                    np.argmax(data[:, idx:idx+n_cat], axis=1))
                idx += n_cat
        return pd.DataFrame(result)
    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim))
    def forward(self, x): return F.relu(x + self.block(x))

class SelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        ad = max(dim // 8, 1)
        self.query = nn.Linear(dim, ad, bias=False)
        self.key = nn.Linear(dim, ad, bias=False)
        self.value = nn.Linear(dim, ad, bias=False)
        self.out_proj = nn.Linear(ad, dim, bias=False)
        self.scale = ad ** -0.5
    def forward(self, x):
        Q, K, V = self.query(x), self.key(x), self.value(x)
        return x + self.out_proj(F.softmax(Q @ K.T * self.scale, dim=-1) @ V)

class Generator(nn.Module):
    def __init__(self, latent_dim, cond_dim, output_dim, output_info, hidden_dims=None):
        super().__init__()
        if hidden_dims is None: hidden_dims = [256, 256]
        self.output_info = output_info
        self.input_layer = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]), nn.ReLU())
        self.res_blocks = nn.ModuleList([ResidualBlock(d) for d in hidden_dims])
        self.self_attn = SelfAttention(hidden_dims[-1])
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
    def forward(self, z, cond):
        x = self.input_layer(torch.cat([z, cond], dim=1))
        for b in self.res_blocks: x = b(x)
        x = self.self_attn(x)
        return self._apply_activations(self.output_layer(x))
    def _apply_activations(self, x):
        out, idx = [], 0
        for kind, _, size in self.output_info:
            if kind == "continuous":
                out.append(torch.tanh(x[:, idx:idx+1]))
                out.append(F.softmax(x[:, idx+1:idx+1+size], dim=1))
                idx += 1 + size
            else:
                out.append(F.softmax(x[:, idx:idx+size], dim=1))
                idx += size
        return torch.cat(out, dim=1)


# ---- Load Models ----
@st.cache_resource
def load_models():
    prep = TabularPreprocessor.load(os.path.join(CKPT_DIR, "preprocessor.pkl"))
    cond_vec = np.load(os.path.join(CKPT_DIR, "cond_vec.npy"))
    G = Generator(latent_dim=128, cond_dim=cond_vec.shape[1],
                  output_dim=prep.output_dim, output_info=prep.output_info)
    _sd = torch.load(os.path.join(CKPT_DIR, "generator_final.pt"), map_location="cpu")
    _sd = {k.replace("_orig_mod.", ""): v for k, v in _sd.items()}
    G.load_state_dict(_sd)
    G.eval()
    clf_path = os.path.join(CKPT_DIR, "fraud_classifier.pkl")
    clf = joblib.load(clf_path) if os.path.isfile(clf_path) else None
    history = None
    hist_path = os.path.join(CKPT_DIR, "training_history.pkl")
    if os.path.isfile(hist_path):
        with open(hist_path, "rb") as f:
            history = pickle.load(f)
    return prep, cond_vec, G, clf, history


prep, cond_vec, G, clf, history = load_models()


# ---- Sidebar ----
st.sidebar.title("AnalyticGAN")
st.sidebar.markdown("Privacy-Preserving Synthetic Data for Fraud Detection")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "🏠 Home",
    "⚡ Generate Synthetic Data",
    "🔍 Fraud Detection",
    "📊 Live Distribution Comparison",
    "📈 Training Dashboard",
    "📋 Evaluation Results",
])
st.sidebar.markdown("---")
st.sidebar.markdown("**EAI 6020** | Northeastern University")
st.sidebar.markdown("Built by Bigyan Khadka")


# ======== PAGE: HOME ========
if page == "🏠 Home":
    st.title("AnalyticGAN")
    st.markdown("### Privacy-Preserving Synthetic Data Generation for Fraud Detection")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Dataset Size", "284,807")
    col2.metric("Fraud Rate", "0.17%")
    col3.metric("Features", "29")
    col4.metric("Training Epochs", f"{len(history['d_loss'])}" if history else "N/A")

    st.markdown("---")
    st.markdown("""
    **AnalyticGAN** is a CTGAN-based synthetic tabular data generation system
    designed for privacy-preserving fraud detection. It extends the original
    CTGAN architecture with:

    - **WGAN-GP loss** for stable adversarial training
    - **Self-Attention** in the generator for inter-feature dependencies
    - **Spectral Normalization + PacGAN** in the discriminator
    - **Conditional Sampling** with fraud oversampled to 20%

    This dashboard lets you generate synthetic data, detect fraud, compare
    distributions interactively, and explore all evaluation results.
    """)

    st.markdown("---")
    st.markdown("#### Project Architecture")
    st.code("""
    Raw CSV --> VGM Encoder --> [WGAN-GP: Generator <-> Discriminator] --> VGM Decoder --> Synthetic CSV
                                        |
                                  Self-Attention
                                  Residual Blocks
                                  Spectral Norm
    """, language="text")


# ======== PAGE: GENERATE ========
elif page == "⚡ Generate Synthetic Data":
    st.title("Generate Synthetic Credit Card Transactions")

    col1, col2 = st.columns(2)
    with col1:
        n_samples = st.slider("Number of samples", 100, 10000, 1000, 100)
    with col2:
        fraud_pct = st.slider("Fraud percentage (%)", 1, 50, 20, 1) / 100.0

    if st.button("Generate Synthetic Data", type="primary"):
        with st.spinner("Generating..."):
            n_fraud = int(n_samples * fraud_pct)
            n_legit = n_samples - n_fraud
            labels = np.concatenate([np.ones(n_fraud), np.zeros(n_legit)])
            np.random.shuffle(labels)
            eye = np.eye(2)
            cond = torch.tensor(eye[labels.astype(int)], dtype=torch.float32)
            with torch.no_grad():
                z = torch.randn(n_samples, 128)
                out = G(z, cond)
            df_gen = prep.inverse_transform(out)
            df_gen["Class"] = labels.astype(int)

        st.success(f"Generated {n_samples} synthetic transactions!")

        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Rows", n_samples)
        m2.metric("Fraud", n_fraud)
        m3.metric("Legitimate", n_legit)
        m4.metric("Fraud %", f"{fraud_pct*100:.0f}%")

        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["Class Distribution", "Data Preview", "Statistics"])

        with tab1:
            fig, ax = plt.subplots(figsize=(6, 3))
            counts = df_gen["Class"].value_counts().sort_index()
            ax.bar(["Legitimate (0)", "Fraud (1)"], counts.values,
                   color=["#378ADD", "#D85A30"])
            ax.set_ylabel("Count")
            ax.set_title("Generated Class Distribution")
            st.pyplot(fig)
            plt.close()

        with tab2:
            st.dataframe(df_gen.head(20), use_container_width=True)

        with tab3:
            st.dataframe(df_gen.describe().round(4), use_container_width=True)

        # Download
        csv = df_gen.to_csv(index=False).encode()
        st.download_button("Download CSV", csv, "synthetic_data.csv", "text/csv",
                          type="primary")


# ======== PAGE: FRAUD DETECTION ========
elif page == "🔍 Fraud Detection":
    st.title("Fraud Detection Classifier")

    if clf is None:
        st.error("fraud_classifier.pkl not found. Run Notebook 6 first.")
    else:
        st.markdown("Upload a CSV with V1-V28 + Amount columns, or generate synthetic data first and upload it here.")

        uploaded = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded is not None:
            df_up = pd.read_csv(uploaded)

            missing = [c for c in FEATURES if c not in df_up.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                with st.spinner("Running fraud classifier..."):
                    proba = clf.predict_proba(df_up[FEATURES].values)[:, 1]
                    preds = (proba >= 0.5).astype(int)
                    df_up["Fraud_Probability"] = proba.round(4)
                    df_up["Prediction"] = ["Fraud" if p else "Legit" for p in preds]

                n_fraud = preds.sum()

                # Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Rows", len(df_up))
                m2.metric("Fraud Detected", int(n_fraud))
                m3.metric("Fraud Rate", f"{n_fraud/len(df_up)*100:.2f}%")

                # Threshold slider
                threshold = st.slider("Fraud Probability Threshold", 0.0, 1.0, 0.5, 0.05)
                preds_custom = (proba >= threshold).astype(int)
                n_fraud_custom = preds_custom.sum()
                st.info(f"At threshold {threshold:.2f}: {n_fraud_custom} fraud cases detected ({n_fraud_custom/len(df_up)*100:.2f}%)")

                # Tabs
                tab1, tab2 = st.tabs(["Predictions Table", "Probability Distribution"])

                with tab1:
                    st.dataframe(
                        df_up[["Fraud_Probability", "Prediction"] + FEATURES[:5]].head(50),
                        use_container_width=True)

                with tab2:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.hist(proba[preds == 0], bins=50, alpha=0.6,
                            color="#378ADD", label="Predicted Legit")
                    ax.hist(proba[preds == 1], bins=50, alpha=0.6,
                            color="#D85A30", label="Predicted Fraud")
                    ax.axvline(threshold, color="red", ls="--", label=f"Threshold={threshold}")
                    ax.set_xlabel("Fraud Probability")
                    ax.set_ylabel("Count")
                    ax.set_title("Distribution of Fraud Probabilities")
                    ax.legend()
                    st.pyplot(fig)
                    plt.close()

                csv = df_up.to_csv(index=False).encode()
                st.download_button("Download Predictions", csv,
                                   "predictions.csv", "text/csv")


# ======== PAGE: LIVE DISTRIBUTION COMPARISON ========
elif page == "📊 Live Distribution Comparison":
    st.title("Interactive Distribution Comparison")
    st.markdown("Generate synthetic data and compare distributions against real data in real-time.")

    col1, col2 = st.columns(2)
    with col1:
        n_gen = st.slider("Synthetic samples", 500, 5000, 2000, 500, key="dist_n")
    with col2:
        feature = st.selectbox("Select feature", FEATURES, index=FEATURES.index("Amount"))

    # Generate synthetic on the fly
    with st.spinner("Generating..."):
        n_fraud = int(n_gen * 0.20)
        n_legit = n_gen - n_fraud
        labels = np.concatenate([np.ones(n_fraud), np.zeros(n_legit)])
        np.random.shuffle(labels)
        eye = np.eye(2)
        cond = torch.tensor(eye[labels.astype(int)], dtype=torch.float32)
        with torch.no_grad():
            z = torch.randn(n_gen, 128)
            out = G(z, cond)
        df_gen = prep.inverse_transform(out)
        df_gen["Class"] = labels.astype(int)

    # Load real data for comparison
    import kagglehub
    _kaggle = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    df_real = pd.read_csv(os.path.join(_kaggle, "creditcard.csv"))

    # Load CTGAN synthetic if available
    ctgan_path = os.path.join(CKPT_DIR, "synthetic_sample.csv")
    df_ctgan = pd.read_csv(ctgan_path) if os.path.isfile(ctgan_path) else None

    # Distribution overlay
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df_real[feature], bins=80, density=True,
            alpha=0.4, color="#378ADD", label="Real Data")
    ax.hist(df_gen[feature], bins=80, density=True,
            alpha=0.4, color="#1D9E75", label=f"Generated (n={n_gen})")
    if df_ctgan is not None and feature in df_ctgan.columns:
        ax.hist(df_ctgan[feature], bins=80, density=True,
                alpha=0.3, color="#D85A30", label="CTGAN Sample")
    ax.set_title(f"Distribution: {feature}", fontsize=14, fontweight="bold")
    ax.set_xlabel(feature)
    ax.set_ylabel("Density")
    ax.legend()
    st.pyplot(fig)
    plt.close()

    # Side-by-side stats
    st.markdown("### Statistics Comparison")
    stats_data = {
        "Metric": ["Mean", "Std", "Min", "25%", "50%", "75%", "Max"],
        "Real": [
            f"{df_real[feature].mean():.4f}",
            f"{df_real[feature].std():.4f}",
            f"{df_real[feature].min():.4f}",
            f"{df_real[feature].quantile(0.25):.4f}",
            f"{df_real[feature].quantile(0.50):.4f}",
            f"{df_real[feature].quantile(0.75):.4f}",
            f"{df_real[feature].max():.4f}",
        ],
        "Synthetic": [
            f"{df_gen[feature].mean():.4f}",
            f"{df_gen[feature].std():.4f}",
            f"{df_gen[feature].min():.4f}",
            f"{df_gen[feature].quantile(0.25):.4f}",
            f"{df_gen[feature].quantile(0.50):.4f}",
            f"{df_gen[feature].quantile(0.75):.4f}",
            f"{df_gen[feature].max():.4f}",
        ],
    }
    st.dataframe(pd.DataFrame(stats_data), use_container_width=True)

    # Multi-feature comparison
    st.markdown("### Multi-Feature Overview")
    check_cols = st.multiselect("Select features to compare",
                                FEATURES, default=["V1", "V2", "V3", "V4", "Amount"])
    if check_cols:
        n_cols = min(len(check_cols), 5)
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
        if n_cols == 1:
            axes = [axes]
        for i, col in enumerate(check_cols[:5]):
            axes[i].hist(df_real[col], bins=50, density=True,
                        alpha=0.4, color="#378ADD", label="Real")
            axes[i].hist(df_gen[col], bins=50, density=True,
                        alpha=0.4, color="#1D9E75", label="Synthetic")
            axes[i].set_title(col, fontweight="bold")
            axes[i].legend(fontsize=7)
            axes[i].set_yticks([])
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ======== PAGE: TRAINING DASHBOARD ========
elif page == "📈 Training Dashboard":
    st.title("WGAN-GP Training Dashboard")

    if history is None:
        st.error("training_history.pkl not found.")
    else:
        epochs = len(history["d_loss"])
        st.markdown(f"**{epochs} epochs** of WGAN-GP training")

        # Key metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Final W-Distance", f"{history['w_dist'][-1]:.4f}")
        m2.metric("Best W-Distance", f"{min(history['w_dist'], key=abs):.4f}")
        m3.metric("Final D Loss", f"{history['d_loss'][-1]:.4f}")
        m4.metric("Final G Loss", f"{history['g_loss'][-1]:.4f}")

        # Epoch range slider
        ep_range = st.slider("Epoch range to display", 1, epochs, (1, epochs))

        # Training curves
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        axes = axes.flatten()
        for ax, key, label, color in zip(
            axes,
            ["d_loss", "g_loss", "gp", "w_dist"],
            ["Discriminator Loss", "Generator Loss",
             "Gradient Penalty", "Wasserstein Distance"],
            ["#D85A30", "#378ADD", "#7F77DD", "#1D9E75"],
        ):
            data = history[key][ep_range[0]-1:ep_range[1]]
            ax.plot(range(ep_range[0], ep_range[0]+len(data)),
                    data, color=color, linewidth=1.5)
            ax.set_title(label, fontweight="bold")
            ax.set_xlabel("Epoch")
            ax.grid(alpha=0.3)
        plt.suptitle(f"WGAN-GP Training Curves (Epochs {ep_range[0]}-{ep_range[1]})",
                     fontsize=13, fontweight="bold", y=1.01)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Training analysis
        st.markdown("### Training Analysis")
        w_improvement = ((history["w_dist"][0] - history["w_dist"][-1]) / history["w_dist"][0]) * 100
        st.markdown(f"""
        - **W-Distance reduced by {w_improvement:.1f}%** from {history['w_dist'][0]:.4f} to {history['w_dist'][-1]:.4f}
        - **Generator loss trend:** {'Increasing (D overpowering G)' if history['g_loss'][-1] > history['g_loss'][0] else 'Decreasing (healthy)'}
        - **Gradient penalty converged to:** {history['gp'][-1]:.4f}
        """)


# ======== PAGE: EVALUATION RESULTS ========
elif page == "📋 Evaluation Results":
    st.title("Evaluation Results")

    # Tabs for different evaluation sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Statistical Fidelity", "Correlation", "ML Efficacy", "Privacy", "CTGAN vs Flow Matching"
    ])

    with tab1:
        st.subheader("Jensen-Shannon Divergence per Column")
        p = os.path.join(CKPT_DIR, "figA_jsd.png")
        if os.path.isfile(p):
            st.image(p, use_container_width=True)
        p2 = os.path.join(CKPT_DIR, "stats_comparison.csv")
        if os.path.isfile(p2):
            st.subheader("Per-Column Statistics")
            st.dataframe(pd.read_csv(p2), use_container_width=True)

    with tab2:
        st.subheader("Correlation Structure: Real vs Synthetic")
        p = os.path.join(CKPT_DIR, "figB_correlation.png")
        if os.path.isfile(p):
            st.image(p, use_container_width=True)

    with tab3:
        st.subheader("ML Efficacy: Train on Synthetic, Test on Real")
        for csv_name, title in [
            ("ml_efficacy.csv", "TSTR vs TRTR"),
            ("classifier_results.csv", "3-Way Classifier Comparison")]:
            p = os.path.join(CKPT_DIR, csv_name)
            if os.path.isfile(p):
                st.markdown(f"**{title}**")
                st.dataframe(pd.read_csv(p), use_container_width=True)
        p = os.path.join(CKPT_DIR, "figF_roc.png")
        if os.path.isfile(p):
            st.subheader("ROC Curves")
            st.image(p, use_container_width=True)
        p = os.path.join(CKPT_DIR, "figG_feature_importance.png")
        if os.path.isfile(p):
            st.subheader("Feature Importances")
            st.image(p, use_container_width=True)

    with tab4:
        st.subheader("Privacy: Nearest-Neighbour Distance Ratio")
        p = os.path.join(CKPT_DIR, "figD_nndr.png")
        if os.path.isfile(p):
            st.image(p, use_container_width=True)

    with tab5:
        st.subheader("CTGAN vs Flow Matching Comparison")
        p = os.path.join(CKPT_DIR, "figH_jsd_comparison.png")
        if os.path.isfile(p):
            st.markdown("**JSD Comparison**")
            st.image(p, use_container_width=True)
        p = os.path.join(CKPT_DIR, "figI_nndr_comparison.png")
        if os.path.isfile(p):
            st.markdown("**NNDR Comparison**")
            st.image(p, use_container_width=True)
        p = os.path.join(CKPT_DIR, "figJ_three_way.png")
        if os.path.isfile(p):
            st.markdown("**Distribution Comparison**")
            st.image(p, use_container_width=True)
        p = os.path.join(CKPT_DIR, "flow_matching_comparison.csv")
        if os.path.isfile(p):
            st.markdown("**Metrics Comparison**")
            st.dataframe(pd.read_csv(p), use_container_width=True)
