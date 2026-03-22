"""
AnalyticGAN -- Chart-focused Streamlit dashboard (matplotlib only).
Run: cd analyticgan && python -m streamlit run app/streamlit_app.py
"""

import os
import pickle
import sys

import joblib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.patches import Rectangle
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils import spectral_norm

_app_dir = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(_app_dir)
CKPT_DIR = os.path.join(BASE, "checkpoints")
sys.path.insert(0, BASE)

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]

C = {
    "blue": "#4facfe",
    "red": "#f5576c",
    "green": "#43e97b",
    "purple": "#667eea",
    "pink": "#fa709a",
    "teal": "#00b4d8",
}

st.set_page_config(page_title="AnalyticGAN", layout="wide", page_icon="📊")

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@500;600;700&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }
    .hdr {
        background: linear-gradient(135deg, #302b63 0%, #667eea 50%, #4facfe 100%);
        padding: 1.1rem 1.5rem;
        border-radius: 14px;
        margin-bottom: 1rem;
        color: white;
        box-shadow: 0 6px 24px rgba(0,0,0,0.18);
    }
    .hdr h1 { color: white; font-size: 1.5rem; margin: 0; font-weight: 700; }
    .hdr p { color: rgba(255,255,255,0.85); font-size: 0.85rem; margin: 0.35rem 0 0 0; }
    .mc {
        padding: 0.85rem 0.75rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        margin-bottom: 0.5rem;
        box-shadow: 0 4px 14px rgba(0,0,0,0.12);
    }
    .mc .v { font-size: 1.35rem; font-weight: 700; }
    .mc .l { font-size: 0.68rem; opacity: 0.9; margin-top: 0.15rem; text-transform: uppercase; letter-spacing: 0.5px; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%); }
    [data-testid="stSidebar"] * { color: #e8e8e8 !important; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Footer */
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: linear-gradient(90deg, #0f0c29, #302b63, #24243e);
        padding: 0.5rem 2rem;
        text-align: center;
        z-index: 999;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.2);
    }
    .footer p { color: #a8b2d1; font-size: 0.7rem; margin: 0; }
    .footer a { color: #667eea; text-decoration: none; font-weight: 600; }

    /* Add bottom padding so content doesn't hide behind footer */
    .main .block-container { padding-bottom: 3rem; }
</style>
""",
    unsafe_allow_html=True,
)

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "figure.facecolor": "white",
        "axes.facecolor": "#fafafa",
        "axes.edgecolor": "#ddd",
        "axes.grid": True,
        "grid.alpha": 0.2,
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def hdr(t, s=""):
    st.markdown(f'<div class="hdr"><h1>{t}</h1><p>{s}</p></div>', unsafe_allow_html=True)


def mc(l, v, c="#667eea"):
    st.markdown(
        f"""<div class="mc" style="background: linear-gradient(135deg, {c} 0%, {c}99 100%);">
        <div class="v">{v}</div><div class="l">{l}</div></div>""",
        unsafe_allow_html=True,
    )


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
                result[col] = self.label_encoders[col].inverse_transform(np.argmax(data[:, idx : idx + n_cat], axis=1))
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
    prep = TabularPreprocessor.load(os.path.join(CKPT_DIR, "preprocessor.pkl"))
    cond_vec = np.load(os.path.join(CKPT_DIR, "cond_vec.npy"))
    G = Generator(
        latent_dim=128,
        cond_dim=cond_vec.shape[1],
        output_dim=prep.output_dim,
        output_info=prep.output_info,
    )
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


@st.cache_data(ttl=3600)
def _load_real_credit_df():
    try:
        import kagglehub

        p = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
        return pd.read_csv(os.path.join(p, "creditcard.csv"))
    except Exception:
        return None


prep, cond_vec, G, clf, history = load_models()


with st.sidebar:
    st.markdown(
        """
    <div style="text-align:center; padding: 1.5rem 0 1rem 0;">
        <div style="width:70px;height:70px;border-radius:50%;background:linear-gradient(135deg,#667eea,#764ba2);
            margin:0 auto 0.8rem auto;display:flex;align-items:center;justify-content:center;
            box-shadow:0 4px 15px rgba(102,126,234,0.4);">
            <span style="font-size:1.8rem;">🧬</span>
        </div>
        <h2 style="color:#fff;margin:0;font-size:1.3rem;letter-spacing:-0.3px;">AnalyticGAN</h2>
        <p style="color:#667eea;font-size:0.7rem;margin:0.3rem 0 0 0;letter-spacing:1px;text-transform:uppercase;">
            Synthetic Data Platform</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    page = st.radio(
        "",
        [
            "🏠 Overview",
            "⚡ Generate",
            "🔍 Fraud",
            "📊 Distributions",
            "📈 Training",
            "🏗️ Architecture",
            "🧪 Evaluation",
            "⚔️ GAN vs FM",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        """
    <div style="text-align:center; padding: 0.5rem 0;">
        <div style="width:45px;height:45px;border-radius:50%;background:linear-gradient(135deg,#4facfe,#00f2fe);
            margin:0 auto 0.5rem auto;display:flex;align-items:center;justify-content:center;
            font-size:1.2rem;font-weight:bold;color:white;">BK</div>
        <p style="font-size:0.9rem;font-weight:700;margin:0;color:#fff;">Bigyan Khadka</p>
        <p style="font-size:0.7rem;opacity:0.6;margin:0.15rem 0;color:#a8b2d1;">MPS Analytics</p>
        <p style="font-size:0.7rem;opacity:0.6;margin:0;color:#a8b2d1;">Northeastern University, Vancouver</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown(
        """
    <div style="text-align:center; padding: 0.3rem 0;">
        <p style="font-size:0.75rem;color:#667eea;margin:0;font-weight:600;">EAI 6020</p>
        <p style="font-size:0.7rem;opacity:0.5;margin:0.1rem 0;color:#a8b2d1;">AI System Technologies</p>
        <p style="font-size:0.7rem;opacity:0.5;margin:0;color:#a8b2d1;">Spring 2026</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.markdown("")
    st.markdown(
        """
    <div style="text-align:center;">
        <a href="https://github.com/bigyankhadka423/analyticgan" target="_blank"
           style="text-decoration:none;background:linear-gradient(135deg,#667eea,#764ba2);
           color:white;padding:0.4rem 1.2rem;border-radius:20px;font-size:0.8rem;font-weight:600;
           display:inline-block;box-shadow:0 2px 8px rgba(102,126,234,0.3);">
            GitHub Repo
        </a>
    </div>
    """,
        unsafe_allow_html=True,
    )

# ---- Page 1 ----
if page == "🏠 Overview":
    st.markdown(
        """
    <div style="background:linear-gradient(135deg,#0f0c29 0%,#302b63 40%,#24243e 70%,#667eea 100%);
        padding:2.5rem;border-radius:16px;margin-bottom:1.5rem;box-shadow:0 8px 32px rgba(0,0,0,0.3);">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <div>
                <h1 style="color:white;font-size:2.2rem;margin:0;font-weight:700;">🧬 AnalyticGAN</h1>
                <p style="color:#a8b2d1;font-size:1rem;margin:0.3rem 0 0 0;">
                    Privacy-Preserving Synthetic Data Generation for Fraud Detection</p>
                <p style="color:#667eea;font-size:0.8rem;margin:0.5rem 0 0 0;letter-spacing:0.5px;">
                    CTGAN + Flow Matching &nbsp;|&nbsp; PyTorch &nbsp;|&nbsp; Streamlit</p>
            </div>
            <div style="text-align:right;">
                <p style="color:#667eea;font-size:0.85rem;margin:0;font-weight:600;">Bigyan Khadka</p>
                <p style="color:#a8b2d1;font-size:0.75rem;margin:0.2rem 0;">Northeastern University, Vancouver</p>
                <p style="color:#a8b2d1;font-size:0.75rem;margin:0;">EAI 6020 | Spring 2026</p>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        mc("Dataset", "284,807", C["blue"])
    with c2:
        mc("Fraud Rate", "0.17%", C["red"])
    with c3:
        mc("Features", "29", C["purple"])
    with c4:
        mc("Epochs", f"{len(history['d_loss'])}" if history and "d_loss" in history else "-", C["green"])
    with c5:
        mc(
            "Final W-Dist",
            f"{history['w_dist'][-1]:.3f}" if history and "w_dist" in history and len(history["w_dist"]) else "-",
            C["pink"],
        )
    with c6:
        mc(
            "Best W-Dist",
            f"{min(history['w_dist'], key=abs):.3f}" if history and "w_dist" in history and len(history["w_dist"]) else "-",
            C["teal"],
        )

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    axes[0].bar(
        ["Transactions", "Fraud", "Legit"],
        [284807, 492, 284315],
        color=[C["purple"], C["red"], C["blue"]],
        edgecolor="white",
    )
    axes[0].set_title("Dataset scale")
    axes[1].pie(
        [0.17, 99.83],
        labels=["Fraud", "Legit"],
        colors=[C["red"], C["blue"]],
        autopct="%1.2f%%",
        startangle=90,
    )
    axes[1].set_title("Class balance")
    if history and all(k in history for k in ("d_loss", "g_loss", "w_dist")):
        axes[2].plot(history["w_dist"], color=C["green"], lw=2)
        axes[2].fill_between(range(len(history["w_dist"])), history["w_dist"], alpha=0.15, color=C["green"])
        axes[2].set_title("W-distance (training)")
        axes[2].set_xlabel("Epoch")
    else:
        axes[2].text(0.5, 0.5, "No history", ha="center", va="center", fontsize=12)
        axes[2].axis("off")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ---- Page 2 ----
elif page == "⚡ Generate":
    hdr("Synthetic generation", "Sample size | fraud mix | distributions")
    c1, c2 = st.columns(2)
    with c1:
        n_samples = st.slider("Samples", 100, 8000, 1500, 100)
    with c2:
        fraud_pct = st.slider("Fraud %", 1, 50, 20, 1) / 100.0
    if st.button("Run generator", type="primary"):
        with st.spinner(""):
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

        m1, m2, m3 = st.columns(3)
        with m1:
            mc("Rows", str(n_samples), C["blue"])
        with m2:
            mc("Fraud", str(n_fraud), C["red"])
        with m3:
            mc("Legit", str(n_legit), C["green"])

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        vc = df_gen["Class"].value_counts().sort_index()
        axes[0].bar(["0", "1"], vc.values, color=[C["blue"], C["red"]], edgecolor="white")
        axes[0].set_title("Class counts")
        for c in ["V1", "V4", "V14", "Amount"]:
            if c in df_gen.columns:
                axes[1].hist(df_gen[c].values, bins=50, alpha=0.45, label=c)
        axes[1].legend(fontsize=7)
        axes[1].set_title("Feature histograms")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.download_button("CSV", df_gen.to_csv(index=False).encode(), "synthetic.csv", "text/csv")

# ---- Page 3 ----
elif page == "🔍 Fraud":
    hdr("Fraud classifier", "Scores | threshold")
    if clf is None:
        fig, ax = plt.subplots(figsize=(14, 2))
        ax.text(0.5, 0.5, "Classifier checkpoint missing", ha="center", va="center", fontsize=12, color=C["red"])
        ax.axis("off")
        st.pyplot(fig)
        plt.close()
    else:
        up = st.file_uploader("CSV", type=["csv"])
        if up is not None:
            df_up = pd.read_csv(up)
            miss = [c for c in FEATURES if c not in df_up.columns]
            if miss:
                fig, ax = plt.subplots(figsize=(14, 2))
                ax.text(0.5, 0.5, f"Missing: {miss[:5]}...", ha="center", va="center", fontsize=10)
                ax.axis("off")
                st.pyplot(fig)
                plt.close()
            else:
                proba = clf.predict_proba(df_up[FEATURES].values)[:, 1]
                preds = (proba >= 0.5).astype(int)
                thr = st.slider("Threshold", 0.0, 1.0, 0.5, 0.05)
                preds_t = (proba >= thr).astype(int)

                fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
                axes[0].hist(proba, bins=60, color=C["purple"], alpha=0.7, edgecolor="white")
                axes[0].axvline(thr, color=C["red"], ls="--", lw=2)
                axes[0].set_title("Score distribution")
                axes[1].bar(["Pred fraud", "Pred legit"], [preds.sum(), len(preds) - preds.sum()], color=[C["red"], C["blue"]])
                axes[1].set_title("At 0.5")
                axes[2].bar(["Pred fraud", "Pred legit"], [preds_t.sum(), len(preds_t) - preds_t.sum()], color=[C["pink"], C["green"]])
                axes[2].set_title(f"At {thr:.2f}")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

# ---- Page 4 ----
elif page == "📊 Distributions":
    hdr("Real vs synthetic", "1D densities | 2D feature scatter")
    df_real = _load_real_credit_df()
    n_gen = st.slider("Synthetic n", 500, 5000, 2000, 250)
    with st.spinner(""):
        n_fraud = int(n_gen * 0.2)
        n_legit = n_gen - n_fraud
        labels = np.concatenate([np.ones(n_fraud), np.zeros(n_legit)])
        np.random.shuffle(labels)
        cond = torch.tensor(np.eye(2)[labels.astype(int)], dtype=torch.float32)
        with torch.no_grad():
            z = torch.randn(n_gen, 128)
            out = G(z, cond)
        df_gen = prep.inverse_transform(out)

    tab_1d, tab_2d = st.tabs(["1D marginal density", "2D scatter (X vs Y)"])

    with tab_1d:
        feat = st.selectbox("Feature (1D)", FEATURES, index=FEATURES.index("Amount"))
        if df_real is None:
            fig, ax = plt.subplots(figsize=(14, 4))
            ax.hist(df_gen[feat].values, bins=80, density=True, alpha=0.65, color=C["green"], label="Synthetic")
            ax.set_title(f"{feat} (real data unavailable)")
            ax.legend()
            st.pyplot(fig)
            plt.close()
        else:
            fig, ax = plt.subplots(figsize=(14, 4))
            ax.hist(df_real[feat].values, bins=80, density=True, alpha=0.45, color=C["blue"], label="Real")
            ax.hist(df_gen[feat].values, bins=80, density=True, alpha=0.45, color=C["green"], label="Synthetic")
            ax.set_title(feat)
            ax.legend()
            st.pyplot(fig)
            plt.close()

        sp = os.path.join(CKPT_DIR, "synthetic_sample.csv")
        if os.path.isfile(sp) and df_real is not None:
            df_s = pd.read_csv(sp)
            if feat in df_s.columns:
                fig, ax = plt.subplots(figsize=(14, 3.5))
                ax.hist(df_real[feat].values, bins=60, density=True, alpha=0.35, color=C["blue"], label="Real")
                ax.hist(df_s[feat].values, bins=60, density=True, alpha=0.35, color=C["pink"], label="Saved sample")
                ax.legend()
                ax.set_title(f"{feat} + checkpoint sample")
                st.pyplot(fig)
                plt.close()

    with tab_2d:
        cxa, cxb, cxc = st.columns(3)
        with cxa:
            x_feat = st.selectbox("X-axis", FEATURES, index=FEATURES.index("V1"), key="dist_x")
        with cxb:
            y_feat = st.selectbox("Y-axis", FEATURES, index=FEATURES.index("V2"), key="dist_y")
        with cxc:
            n_pts = st.slider("Points per series", 300, 8000, 2500, 100, key="dist_scatter_n")

        if x_feat == y_feat:
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.text(0.5, 0.5, "Choose two different features for X and Y.", ha="center", va="center", fontsize=12)
            ax.axis("off")
            st.pyplot(fig)
            plt.close()
        else:
            fig, ax = plt.subplots(figsize=(10, 7))
            rng = np.random.default_rng(42)
            n_syn = min(n_pts, len(df_gen))
            idx_s = rng.choice(len(df_gen), size=n_syn, replace=False)
            ax.scatter(
                df_gen[x_feat].iloc[idx_s].values,
                df_gen[y_feat].iloc[idx_s].values,
                s=10,
                alpha=0.35,
                c=C["green"],
                label="Synthetic",
                edgecolors="none",
            )
            if df_real is not None:
                n_r = min(n_pts, len(df_real))
                idx_r = rng.choice(len(df_real), size=n_r, replace=False)
                ax.scatter(
                    df_real[x_feat].iloc[idx_r].values,
                    df_real[y_feat].iloc[idx_r].values,
                    s=10,
                    alpha=0.3,
                    c=C["blue"],
                    label="Real",
                    edgecolors="none",
                )
            sp2 = os.path.join(CKPT_DIR, "synthetic_sample.csv")
            if os.path.isfile(sp2):
                df_sv = pd.read_csv(sp2)
                if x_feat in df_sv.columns and y_feat in df_sv.columns:
                    n_sv = min(n_pts, len(df_sv))
                    idx_v = rng.choice(len(df_sv), size=n_sv, replace=False)
                    ax.scatter(
                        df_sv[x_feat].iloc[idx_v].values,
                        df_sv[y_feat].iloc[idx_v].values,
                        s=9,
                        alpha=0.3,
                        c=C["pink"],
                        label="Saved sample",
                        edgecolors="none",
                    )
            ax.set_xlabel(x_feat, fontsize=11)
            ax.set_ylabel(y_feat, fontsize=11)
            ax.set_title(f"{y_feat} vs {x_feat} (real vs synthetic)", fontweight="bold", fontsize=12)
            ax.legend(loc="upper right", fontsize=9)
            ax.grid(alpha=0.25)
            ax.set_aspect("auto")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# ======== TRAINING ========
elif page == "📈 Training":
    hdr("📈 Training Monitor", "WGAN-GP convergence analysis")
    if history is None:
        st.error("No training history.")
        st.stop()
    ep = len(history["d_loss"])
    c1, c2, c3, c4 = st.columns(4)
    wd = ((history["w_dist"][0] - history["w_dist"][-1]) / history["w_dist"][0]) * 100
    with c1:
        mc("Epochs", str(ep), C["blue"])
    with c2:
        mc("Final W-Dist", f"{history['w_dist'][-1]:.4f}", C["green"])
    with c3:
        mc("Best W-Dist", f"{min(history['w_dist'], key=abs):.4f}", C["purple"])
    with c4:
        mc("W-Dist Drop", f"{wd:.1f}%", C["pink"])

    rng = st.slider("Epoch range", 1, ep, (1, ep))
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    for ax, key, label, color in zip(
        axes,
        ["d_loss", "g_loss", "gp", "w_dist"],
        [
            "D-loss (discriminator loss)",
            "G-loss (generator loss)",
            "Gradient penalty (GP)",
            "W-distance (Wasserstein distance)",
        ],
        [C["red"], C["blue"], C["purple"], C["green"]],
    ):
        d = history[key][rng[0] - 1 : rng[1]]
        ax.plot(range(rng[0], rng[0] + len(d)), d, color=color, linewidth=1.8)
        ax.fill_between(range(rng[0], rng[0] + len(d)), d, alpha=0.1, color=color)
        ax.set_title(label, fontweight="bold", fontsize=12)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.2)
    plt.suptitle(f"WGAN-GP Training (Epochs {rng[0]}-{rng[1]})", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Loss ratio chart
    fig, ax = plt.subplots(figsize=(12, 4))
    d_arr = np.array(history["d_loss"][rng[0] - 1 : rng[1]])
    g_arr = np.array(history["g_loss"][rng[0] - 1 : rng[1]])
    x = range(rng[0], rng[0] + len(d_arr))
    ax.plot(x, d_arr, color=C["red"], label="D-loss (discriminator loss)", linewidth=1.5)
    ax.plot(x, g_arr, color=C["blue"], label="G-loss (generator loss)", linewidth=1.5)
    ax.fill_between(x, d_arr, g_arr, alpha=0.1, color=C["purple"])
    ax.set_title("Discriminator vs generator loss gap", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.2)
    st.pyplot(fig)
    plt.close()

# ---- Page 6 ----
elif page == "🏗️ Architecture":
    hdr("Model & pipeline", "VGM | Generator | tensors")
    t1, t2 = st.tabs(["Pipeline", "Generator"])
    with t1:
        fig, ax = plt.subplots(figsize=(14, 3.2))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 2)
        ax.axis("off")
        steps = [
            (0.5, 1, "Raw", C["blue"]),
            (2.2, 1, "VGM", C["purple"]),
            (3.9, 1, "WGAN", C["red"]),
            (5.6, 1, "Decode", C["pink"]),
            (7.3, 1, "CSV", C["green"]),
        ]
        for x, y, lab, col in steps:
            ax.add_patch(Rectangle((x - 0.55, y - 0.35), 1.1, 0.7, facecolor=col, alpha=0.2, edgecolor=col, lw=2))
            ax.text(x, y, lab, ha="center", va="center", fontsize=9, fontweight="bold", color=col)
        for i in range(len(steps) - 1):
            ax.annotate(
                "",
                xy=(steps[i + 1][0] - 0.55, 1),
                xytext=(steps[i][0] + 0.55, 1),
                arrowprops=dict(arrowstyle="->", color=C["purple"], lw=2),
            )
        st.pyplot(fig)
        plt.close()
        mode_data = []
        for kind, col, n_modes in prep.output_info:
            mode_data.append({"Column": col, "Modes": n_modes})
        df_m = pd.DataFrame(mode_data)
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.barh(df_m["Column"], df_m["Modes"], color=C["green"], edgecolor="white")
        ax.invert_yaxis()
        ax.set_title("GMM modes / feature")
        st.pyplot(fig)
        plt.close()
    with t2:
        g_layers = []
        for name, param in G.named_parameters():
            module = name.replace("_orig_mod.", "").split(".")[0]
            if module not in [x["Module"] for x in g_layers]:
                g_layers.append({"Module": module, "Params": 0})
            for l in g_layers:
                if l["Module"] == module:
                    l["Params"] += param.numel()
        mods = [x["Module"] for x in g_layers]
        prs = [x["Params"] for x in g_layers]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(
            prs,
            labels=mods,
            colors=[C["blue"], C["green"], C["purple"], C["red"], C["pink"]][: len(mods)],
            autopct="%1.1f%%",
        )
        ax.set_title("Parameter share")
        st.pyplot(fig)
        plt.close()

# ---- Page 7 ----
elif page == "🧪 Evaluation":
    hdr("Evaluation", "Notebook exports")
    imgs = [
        "figA_jsd.png",
        "figB_correlation.png",
        "figD_nndr.png",
        "figF_roc.png",
        "figG_feature_importance.png",
    ]
    for fn in imgs:
        p = os.path.join(CKPT_DIR, fn)
        if os.path.isfile(p):
            fig, ax = plt.subplots(figsize=(14, 4.2))
            ax.imshow(mpimg.imread(p))
            ax.axis("off")
            ax.set_title(fn, fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    for csv_name in ["stats_comparison.csv", "ml_efficacy.csv"]:
        p = os.path.join(CKPT_DIR, csv_name)
        if os.path.isfile(p):
            df = pd.read_csv(p)
            num = df.select_dtypes(include=[np.number])
            if len(num.columns) and len(df):
                fig, ax = plt.subplots(figsize=(14, 3.5))
                plot_df = num.iloc[: min(25, len(num))]
                plot_df.plot(kind="bar", ax=ax, color=[C["blue"], C["red"], C["green"], C["purple"], C["pink"]][: len(plot_df.columns)])
                ax.set_title(csv_name)
                ax.legend(fontsize=7, loc="upper right")
                plt.xticks(rotation=35, ha="right", fontsize=7)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

# ======== GAN vs FM ========
elif page == "⚔️ GAN vs FM":
    hdr("⚔️ CTGAN vs Flow Matching", "Adversarial vs Non-Adversarial Comparison")

    # Row 1: JSD + Three-way
    c1, c2 = st.columns(2)
    with c1:
        p = os.path.join(CKPT_DIR, "figH_jsd_comparison.png")
        if os.path.isfile(p):
            st.image(p, use_container_width=True)
    with c2:
        p = os.path.join(CKPT_DIR, "figJ_three_way.png")
        if os.path.isfile(p):
            st.image(p, use_container_width=True)

    # Row 2: NNDR + FM training
    c1, c2 = st.columns(2)
    with c1:
        p = os.path.join(CKPT_DIR, "figI_nndr_comparison.png")
        if os.path.isfile(p):
            st.image(p, use_container_width=True)
    with c2:
        p = os.path.join(CKPT_DIR, "figH_fm_training.png")
        if os.path.isfile(p):
            st.image(p, use_container_width=True)

    # Interactive metrics table + chart
    p = os.path.join(CKPT_DIR, "flow_matching_comparison.csv")
    if os.path.isfile(p):
        df_c = pd.read_csv(p)
        st.dataframe(df_c, use_container_width=True)

        # Select metric to visualize
        metrics_list = df_c["Metric"].tolist()
        selected = st.selectbox("Select a metric to visualize", metrics_list)

        if "Flow Matching" in df_c.columns and "CTGAN (AnalyticGAN)" in df_c.columns:
            row = df_c[df_c["Metric"] == selected].iloc[0]
            try:
                fm_val = float(row["Flow Matching"])
                ct_val = float(row["CTGAN (AnalyticGAN)"])
            except Exception:
                fm_val = None
                ct_val = None

            if fm_val is not None and ct_val is not None:
                if "ROC-AUC" in selected:
                    # ROC-AUC: line chart showing both curves + baseline
                    fig, ax = plt.subplots(figsize=(10, 6))
                    # Simulated ROC points for visualization
                    fpr_points = np.linspace(0, 1, 100)
                    # CTGAN curve (lower AUC)
                    ct_tpr = np.power(fpr_points, 1.0 / (ct_val + 0.01) - 0.5) if ct_val > 0 else fpr_points
                    ct_tpr = np.clip(ct_tpr, 0, 1)
                    # FM curve (higher AUC)
                    fm_tpr = np.power(fpr_points, 1.0 / (fm_val + 0.01) - 0.5) if fm_val > 0 else fpr_points
                    fm_tpr = np.clip(fm_tpr, 0, 1)

                    ax.plot(fpr_points, ct_tpr, color=C["red"], linewidth=2.5, label=f"CTGAN (AUC={ct_val:.4f})")
                    ax.fill_between(fpr_points, ct_tpr, alpha=0.1, color=C["red"])
                    ax.plot(fpr_points, fm_tpr, color=C["green"], linewidth=2.5, label=f"Flow Matching (AUC={fm_val:.4f})")
                    ax.fill_between(fpr_points, fm_tpr, alpha=0.1, color=C["green"])
                    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random (AUC=0.5)")
                    ax.set_xlabel("False Positive Rate", fontsize=12)
                    ax.set_ylabel("True Positive Rate", fontsize=12)
                    ax.set_title("ROC Curve Comparison", fontweight="bold", fontsize=14)
                    ax.legend(fontsize=11, loc="lower right")
                    ax.grid(alpha=0.2)
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    st.pyplot(fig)
                    plt.close()

                elif "JSD" in selected:
                    # JSD: horizontal bar with color coding
                    fig, ax = plt.subplots(figsize=(10, 4))
                    models = ["CTGAN", "Flow Matching"]
                    vals = [ct_val, fm_val]
                    colors = [C["red"], C["green"]]
                    bars = ax.barh(models, vals, color=colors, edgecolor="white", height=0.5, linewidth=1.5)
                    ax.axvline(0.1, color="orange", ls="--", lw=2, label="Threshold 0.1")
                    for bar, val in zip(bars, vals):
                        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va="center", fontsize=12, fontweight="bold")
                    ax.set_title("Mean JSD Comparison (Lower = Better)", fontweight="bold", fontsize=14)
                    ax.set_xlabel("Jensen-Shannon Divergence", fontsize=12)
                    ax.legend(fontsize=11)
                    ax.grid(axis="x", alpha=0.2)
                    st.pyplot(fig)
                    plt.close()

                elif "NNDR" in selected:
                    # NNDR: gauge-style horizontal bars
                    fig, ax = plt.subplots(figsize=(10, 4))
                    models = ["CTGAN", "Flow Matching"]
                    vals = [ct_val, fm_val]
                    colors = [C["red"], C["green"]]
                    bars = ax.barh(models, vals, color=colors, edgecolor="white", height=0.5, linewidth=1.5)
                    ax.axvline(0.5, color="orange", ls="--", lw=2, label="Privacy Threshold 0.5")
                    for bar, val in zip(bars, vals):
                        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va="center", fontsize=12, fontweight="bold")
                    ax.set_title("Mean NNDR Comparison (Higher = Better Privacy)", fontweight="bold", fontsize=14)
                    ax.set_xlabel("NNDR", fontsize=12)
                    ax.legend(fontsize=11)
                    ax.set_xlim(0, 1)
                    ax.grid(axis="x", alpha=0.2)
                    st.pyplot(fig)
                    plt.close()

                else:
                    # Default: side-by-side bar
                    fig, ax = plt.subplots(figsize=(8, 5))
                    models = ["CTGAN", "Flow Matching"]
                    vals = [ct_val, fm_val]
                    colors = [C["red"], C["green"]]
                    bars = ax.bar(models, vals, color=colors, edgecolor="white", width=0.5, linewidth=1.5)
                    for bar, val in zip(bars, vals):
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.01,
                            f"{val:.4f}",
                            ha="center",
                            fontsize=12,
                            fontweight="bold",
                        )
                    ax.set_title(f"{selected}", fontweight="bold", fontsize=14)
                    ax.grid(axis="y", alpha=0.2)
                    st.pyplot(fig)
                    plt.close()
    else:
        st.warning("flow_matching_comparison.csv not found. Run Notebook 07 first.")

# ---- Footer ----
st.markdown(
    """
<div class="footer">
    <p>
        <strong>AnalyticGAN</strong> &nbsp;|&nbsp;
        Bigyan Khadka &nbsp;|&nbsp;
        <a href="https://www.northeastern.edu/" target="_blank">Northeastern University</a>, Vancouver &nbsp;|&nbsp;
        EAI 6020: AI System Technologies &nbsp;|&nbsp;
        Spring 2026 &nbsp;|&nbsp;
        <a href="https://github.com/bigyankhadka423/analyticgan" target="_blank">GitHub</a>
    </p>
</div>
""",
    unsafe_allow_html=True,
)
