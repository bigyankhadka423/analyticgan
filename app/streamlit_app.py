# AnalyticGAN -- Streamlit demo (generated for Notebook 6). ASCII only in UI strings.
import os
import pickle

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils import spectral_norm

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(_APP_DIR)
CKPT_DIR = os.path.join(BASE, "checkpoints")
FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]

st.set_page_config(page_title="AnalyticGAN", layout="wide")


class VGMEncoder:
    def __init__(self, n_components=10, eps=0.005):
        self.n_components = n_components
        self.eps = eps
        self.bgm = BayesianGaussianMixture(
            n_components=n_components,
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=0.001,
            max_iter=100,
            random_state=42,
            n_init=1,
        )
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
            p_norm = (p / s).astype(np.float64) if (s > 0 and np.isfinite(s)) else np.ones(self.n_valid) / self.n_valid
            mode_idx.append(np.random.choice(self.n_valid, p=p_norm))
        mode_idx = np.array(mode_idx)
        normalized = np.clip(
            (data.flatten() - means[mode_idx]) / (4 * stds[mode_idx] + 1e-8),
            -0.99,
            0.99,
        )
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
                result[col] = self.vgm_encoders[col].inverse_transform(data[:, idx : idx + w])
                idx += w
            else:
                n_cat = self.cat_dims[col]
                result[col] = self.label_encoders[col].inverse_transform(
                    np.argmax(data[:, idx : idx + n_cat], axis=1)
                )
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
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )

    def forward(self, x):
        return F.relu(x + self.block(x))


class SelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        ad = max(dim // 8, 1)
        self.query = nn.Linear(dim, ad, bias=False)
        self.key = nn.Linear(dim, ad, bias=False)
        self.value = nn.Linear(dim, ad, bias=False)
        self.out_proj = nn.Linear(ad, dim, bias=False)
        self.scale = ad**-0.5

    def forward(self, x):
        Q, K, V = self.query(x), self.key(x), self.value(x)
        return x + self.out_proj(F.softmax(Q @ K.T * self.scale, dim=-1) @ V)


class Generator(nn.Module):
    def __init__(self, latent_dim, cond_dim, output_dim, output_info, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]
        self.output_info = output_info
        self.input_layer = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
        )
        self.res_blocks = nn.ModuleList([ResidualBlock(d) for d in hidden_dims])
        self.self_attn = SelfAttention(hidden_dims[-1])
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, z, cond):
        x = self.input_layer(torch.cat([z, cond], dim=1))
        for b in self.res_blocks:
            x = b(x)
        x = self.self_attn(x)
        return self._apply_activations(self.output_layer(x))

    def _apply_activations(self, x):
        out, idx = [], 0
        for kind, _, size in self.output_info:
            if kind == "continuous":
                out.append(torch.tanh(x[:, idx : idx + 1]))
                out.append(F.softmax(x[:, idx + 1 : idx + 1 + size], dim=1))
                idx += 1 + size
            else:
                out.append(F.softmax(x[:, idx : idx + size], dim=1))
                idx += size
        return torch.cat(out, dim=1)


@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prep = TabularPreprocessor.load(os.path.join(CKPT_DIR, "preprocessor.pkl"))
    cond_vec = np.load(os.path.join(CKPT_DIR, "cond_vec.npy"))
    G = Generator(
        latent_dim=128,
        cond_dim=cond_vec.shape[1],
        output_dim=prep.output_dim,
        output_info=prep.output_info,
    ).to(device)
    _sd = torch.load(os.path.join(CKPT_DIR, "generator_final.pt"), map_location=device)
    _sd = {k.replace("_orig_mod.", ""): v for k, v in _sd.items()}
    G.load_state_dict(_sd)
    G.eval()
    clf_path = os.path.join(CKPT_DIR, "fraud_classifier.pkl")
    clf = joblib.load(clf_path) if os.path.isfile(clf_path) else None
    return prep, cond_vec, G, clf


prep, cond_vec, G, clf = load_models()
_dev = next(G.parameters()).device

st.title("AnalyticGAN -- Synthetic Fraud Data Platform")
page = st.sidebar.radio("Navigate", ["Generate Synthetic Data", "Fraud Detection", "Project Summary"])

if page == "Generate Synthetic Data":
    st.header("Generate Synthetic Credit Card Transactions")
    n_samples = st.slider("Number of samples", 100, 5000, 1000, 100)
    fraud_pct = st.slider("Fraud percentage (%)", 1, 50, 20, 1) / 100.0
    if st.button("Generate", type="primary"):
        with st.spinner("Generating..."):
            n_fraud = int(n_samples * fraud_pct)
            n_legit = n_samples - n_fraud
            labels = np.concatenate([np.ones(n_fraud), np.zeros(n_legit)])
            np.random.shuffle(labels)
            eye = np.eye(2)
            cond = torch.tensor(eye[labels.astype(int)], dtype=torch.float32, device=_dev)
            with torch.no_grad():
                z = torch.randn(n_samples, 128, device=_dev)
                out = G(z, cond)
            df_gen = prep.inverse_transform(out)
            df_gen["Class"] = labels.astype(int)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total rows", n_samples)
        c2.metric("Fraud", n_fraud)
        c3.metric("Legitimate", n_legit)
        st.subheader("Class Distribution")
        st.bar_chart(df_gen["Class"].value_counts().sort_index())
        st.subheader("Preview (first 10 rows)")
        st.dataframe(df_gen.head(10))
        csv = df_gen.to_csv(index=False).encode()
        st.download_button("Download CSV", csv, "synthetic_data.csv", "text/csv")

elif page == "Fraud Detection":
    st.header("Fraud Detection Classifier")
    if clf is None:
        st.error("fraud_classifier.pkl not found. Run Notebook 6 first.")
    else:
        uploaded = st.file_uploader("Upload CSV with V1-V28 + Amount", type=["csv"])
        if uploaded is not None:
            df_up = pd.read_csv(uploaded)
            missing = [c for c in FEATURES if c not in df_up.columns]
            if missing:
                st.error("Missing columns: " + str(missing))
            else:
                proba = clf.predict_proba(df_up[FEATURES].values)[:, 1]
                preds = (proba >= 0.5).astype(int)
                df_up["Fraud_Probability"] = np.round(proba, 4)
                df_up["Prediction"] = np.where(preds, "Fraud", "Legit")
                n_fraud = int(preds.sum())
                c1, c2, c3 = st.columns(3)
                c1.metric("Total rows", len(df_up))
                c2.metric("Fraud detected", n_fraud)
                c3.metric("Fraud %", f"{n_fraud / len(df_up) * 100:.2f}%")
                st.dataframe(df_up[["Fraud_Probability", "Prediction"] + FEATURES].head(50))
                csv = df_up.to_csv(index=False).encode()
                st.download_button("Download predictions", csv, "predictions.csv", "text/csv")

elif page == "Project Summary":
    st.header("AnalyticGAN -- Evaluation Summary")
    figs = [
        ("figA_jsd.png", "Fig A -- Jensen-Shannon Divergence"),
        ("figB_correlation.png", "Fig B -- Correlation Structure"),
        ("figD_nndr.png", "Fig D -- Privacy NNDR"),
        ("figF_roc.png", "Fig F -- ROC Curves"),
        ("figG_feature_importance.png", "Fig G -- Feature Importances"),
        ("figE_training_recap.png", "Fig E -- Training Curves"),
    ]
    for fname, caption in figs:
        p = os.path.join(CKPT_DIR, fname)
        if os.path.isfile(p):
            st.subheader(caption)
            st.image(p, use_container_width=True)
    for csv_name, title in [
        ("ml_efficacy.csv", "ML Efficacy (TSTR vs TRTR)"),
        ("classifier_results.csv", "Classifier Comparison"),
        ("stats_comparison.csv", "Statistics Comparison"),
    ]:
        p = os.path.join(CKPT_DIR, csv_name)
        if os.path.isfile(p):
            st.subheader(title)
            st.dataframe(pd.read_csv(p), use_container_width=True)
