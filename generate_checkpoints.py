"""
Rebuild preprocessor.pkl, data_tensor.pt, and cond_vec.npy locally without GAN training.
Run from project root: python generate_checkpoints.py
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import kagglehub
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import LabelEncoder


def detect_column_types(df, target_col, cat_threshold=10):
    continuous, categorical = [], []
    for col in df.columns:
        if col == target_col:
            continue
        if df[col].nunique() <= cat_threshold or df[col].dtype == object:
            categorical.append(col)
        else:
            continuous.append(col)
    return continuous, categorical


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
        data = np.asarray(data).reshape(-1, 1)
        self.bgm.fit(data)
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
        sel_means = means[mode_idx]
        sel_stds = stds[mode_idx]
        normalized = np.clip((data.flatten() - sel_means) / (4 * sel_stds + 1e-8), -0.99, 0.99)
        one_hot = np.zeros((len(data), self.n_valid), dtype=np.float32)
        one_hot[np.arange(len(data)), mode_idx] = 1
        return np.column_stack([normalized, one_hot]).astype(np.float32)

    def inverse_transform(self, encoded):
        encoded = np.asarray(encoded)
        means = self.bgm.means_[self.valid_components].flatten()
        stds = np.sqrt(self.bgm.covariances_[self.valid_components]).flatten()
        normalized = encoded[:, 0]
        one_hot = encoded[:, 1:]
        mode_idx = np.argmax(one_hot, axis=1)
        sel_means = means[mode_idx]
        sel_stds = stds[mode_idx]
        return normalized * 4 * sel_stds + sel_means


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

    def fit(self, df, continuous_cols, categorical_cols, target_col):
        self.continuous_cols = continuous_cols
        self.categorical_cols = categorical_cols
        self.target_col = target_col
        self.output_info = []
        self.output_dim = 0
        for col in continuous_cols:
            enc = VGMEncoder(n_components=self.max_gmm_components, eps=self.eps)
            enc.fit(df[col].values)
            self.vgm_encoders[col] = enc
            self.output_info.append(("continuous", col, enc.n_valid))
            self.output_dim += 1 + enc.n_valid
        for col in categorical_cols:
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            self.label_encoders[col] = le
            n_cat = len(le.classes_)
            self.cat_dims[col] = n_cat
            self.output_info.append(("categorical", col, n_cat))
            self.output_dim += n_cat
        return self

    def transform(self, df):
        parts = []
        for kind, col, _ in self.output_info:
            if kind == "continuous":
                parts.append(self.vgm_encoders[col].transform(df[col].values))
            else:
                lbls = self.label_encoders[col].transform(df[col].astype(str))
                n_cat = self.cat_dims[col]
                oh = np.zeros((len(df), n_cat), dtype=np.float32)
                oh[np.arange(len(df)), lbls] = 1
                parts.append(oh)
        data_arr = np.concatenate(parts, axis=1)
        data_tensor = torch.tensor(data_arr, dtype=torch.float32)
        n_cls = df[self.target_col].nunique()
        tgt = df[self.target_col].values.astype(int)
        cond = np.zeros((len(df), n_cls), dtype=np.float32)
        cond[np.arange(len(df)), tgt] = 1
        return data_tensor, cond

    def inverse_transform(self, tensor):
        data = tensor.detach().cpu().numpy()
        result = {}
        idx = 0
        for kind, col, size in self.output_info:
            if kind == "continuous":
                enc = self.vgm_encoders[col]
                w = 1 + enc.n_valid
                result[col] = enc.inverse_transform(data[:, idx : idx + w])
                idx += w
            else:
                n_cat = self.cat_dims[col]
                chunk = data[:, idx : idx + n_cat]
                lbls = np.argmax(chunk, axis=1)
                result[col] = self.label_encoders[col].inverse_transform(lbls)
                idx += n_cat
        return pd.DataFrame(result)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


def main():
    base = Path(__file__).resolve().parent
    ckpt_dir = base / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading creditcard dataset via kagglehub...")
    dataset_dir = Path(kagglehub.dataset_download("mlg-ulb/creditcardfraud")).resolve()
    csv_candidates = list(dataset_dir.rglob("creditcard.csv"))
    if not csv_candidates:
        raise FileNotFoundError(f"creditcard.csv not found under {dataset_dir}")
    csv_path = csv_candidates[0]
    print(f"Using: {csv_path}")

    df = pd.read_csv(csv_path)
    target_col = "Class"
    continuous_cols, categorical_cols = detect_column_types(df, target_col)

    print("Fitting preprocessor (VGM on continuous columns)...")
    prep = TabularPreprocessor(max_gmm_components=10, eps=0.005)
    prep.fit(df, continuous_cols, categorical_cols, target_col)

    data_tensor, cond = prep.transform(df)

    prep_path = ckpt_dir / "preprocessor.pkl"
    data_path = ckpt_dir / "data_tensor.pt"
    cond_path = ckpt_dir / "cond_vec.npy"

    with open(prep_path, "wb") as f:
        pickle.dump(prep, f)
    torch.save(data_tensor, data_path)
    np.save(cond_path, cond)

    print(f"Saved: {prep_path}")
    print(f"Saved: {data_path}  shape={tuple(data_tensor.shape)}")
    print(f"Saved: {cond_path}  shape={cond.shape}")
    print("Done.")


if __name__ == "__main__":
    main()
