"""
AnalyticGAN -- Professional Interactive Dashboard
Uses Plotly for interactive charts with hover, zoom, click
"""
import os, sys, pickle, joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import jensenshannon

# ---- Paths ----
_app_dir = os.path.dirname(os.path.abspath(__file__))
BASE     = os.path.dirname(_app_dir)
CKPT_DIR = os.path.join(BASE, "checkpoints")
sys.path.insert(0, BASE)
FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]

st.set_page_config(page_title="AnalyticGAN", page_icon="🧬", layout="wide")

# ---- Model Classes ----
class VGMEncoder:
    def __init__(self, n_components=10, eps=0.005):
        self.n_components=n_components; self.eps=eps
        self.bgm=BayesianGaussianMixture(n_components=n_components,weight_concentration_prior_type="dirichlet_process",weight_concentration_prior=0.001,max_iter=100,random_state=42,n_init=1)
        self.valid_components=None; self.n_valid=None
    def fit(self,data):
        self.bgm.fit(np.asarray(data).reshape(-1,1))
        self.valid_components=np.where(self.bgm.weights_>self.eps)[0]; self.n_valid=len(self.valid_components); return self
    def transform(self,data):
        data=np.asarray(data).reshape(-1,1); means=self.bgm.means_[self.valid_components].flatten(); stds=np.sqrt(self.bgm.covariances_[self.valid_components]).flatten()
        probs=self.bgm.predict_proba(data)[:,self.valid_components]; mode_idx=[]
        for p in probs:
            s=p.sum(); p_norm=(p/s).astype(np.float64) if(s>0 and np.isfinite(s)) else np.ones(self.n_valid)/self.n_valid
            mode_idx.append(np.random.choice(self.n_valid,p=p_norm))
        mode_idx=np.array(mode_idx); normalized=np.clip((data.flatten()-means[mode_idx])/(4*stds[mode_idx]+1e-8),-0.99,0.99)
        one_hot=np.zeros((len(data),self.n_valid),dtype=np.float32); one_hot[np.arange(len(data)),mode_idx]=1
        return np.column_stack([normalized,one_hot]).astype(np.float32)
    def inverse_transform(self,encoded):
        encoded=np.asarray(encoded); means=self.bgm.means_[self.valid_components].flatten(); stds=np.sqrt(self.bgm.covariances_[self.valid_components]).flatten()
        mode_idx=np.argmax(encoded[:,1:],axis=1); return encoded[:,0]*4*stds[mode_idx]+means[mode_idx]

class TabularPreprocessor:
    def __init__(self,max_gmm_components=10,eps=0.005):
        self.max_gmm_components=max_gmm_components;self.eps=eps;self.continuous_cols=[];self.categorical_cols=[];self.target_col=None
        self.vgm_encoders={};self.label_encoders={};self.cat_dims={};self.output_info=[];self.output_dim=0
    def inverse_transform(self,tensor):
        data=tensor.detach().cpu().numpy() if hasattr(tensor,"detach") else tensor; result={};idx=0
        for kind,col,size in self.output_info:
            if kind=="continuous":
                w=1+self.vgm_encoders[col].n_valid; result[col]=self.vgm_encoders[col].inverse_transform(data[:,idx:idx+w]); idx+=w
            else:
                n_cat=self.cat_dims[col]; result[col]=self.label_encoders[col].inverse_transform(np.argmax(data[:,idx:idx+n_cat],axis=1)); idx+=n_cat
        return pd.DataFrame(result)
    @staticmethod
    def load(path):
        with open(path,"rb") as f: return pickle.load(f)

class ResidualBlock(nn.Module):
    def __init__(self,dim):
        super().__init__(); self.block=nn.Sequential(nn.Linear(dim,dim),nn.BatchNorm1d(dim),nn.ReLU(),nn.Linear(dim,dim),nn.BatchNorm1d(dim))
    def forward(self,x): return F.relu(x+self.block(x))

class SelfAttention(nn.Module):
    def __init__(self,dim):
        super().__init__(); ad=max(dim//8,1)
        self.query=nn.Linear(dim,ad,bias=False);self.key=nn.Linear(dim,ad,bias=False);self.value=nn.Linear(dim,ad,bias=False)
        self.out_proj=nn.Linear(ad,dim,bias=False);self.scale=ad**-0.5
    def forward(self,x):
        Q,K,V=self.query(x),self.key(x),self.value(x); return x+self.out_proj(F.softmax(Q@K.T*self.scale,dim=-1)@V)

class Generator(nn.Module):
    def __init__(self,latent_dim,cond_dim,output_dim,output_info,hidden_dims=None):
        super().__init__()
        if hidden_dims is None: hidden_dims=[256,256]
        self.output_info=output_info
        self.input_layer=nn.Sequential(nn.Linear(latent_dim+cond_dim,hidden_dims[0]),nn.BatchNorm1d(hidden_dims[0]),nn.ReLU())
        self.res_blocks=nn.ModuleList([ResidualBlock(d) for d in hidden_dims])
        self.self_attn=SelfAttention(hidden_dims[-1]); self.output_layer=nn.Linear(hidden_dims[-1],output_dim)
    def forward(self,z,cond):
        x=self.input_layer(torch.cat([z,cond],dim=1))
        for b in self.res_blocks: x=b(x)
        x=self.self_attn(x); return self._apply_activations(self.output_layer(x))
    def _apply_activations(self,x):
        out=[];idx=0
        for kind,_,size in self.output_info:
            if kind=="continuous":
                out.append(torch.tanh(x[:,idx:idx+1]));out.append(F.softmax(x[:,idx+1:idx+1+size],dim=1));idx+=1+size
            else: out.append(F.softmax(x[:,idx:idx+size],dim=1));idx+=size
        return torch.cat(out,dim=1)

# ---- Load Models ----
@st.cache_resource
def load_models():
    prep=TabularPreprocessor.load(os.path.join(CKPT_DIR,"preprocessor.pkl"))
    cond_vec=np.load(os.path.join(CKPT_DIR,"cond_vec.npy"))
    G=Generator(latent_dim=128,cond_dim=cond_vec.shape[1],output_dim=prep.output_dim,output_info=prep.output_info)
    _sd=torch.load(os.path.join(CKPT_DIR,"generator_final.pt"),map_location="cpu")
    _sd={k.replace("_orig_mod.",""):v for k,v in _sd.items()}
    G.load_state_dict(_sd);G.eval()
    clf_path=os.path.join(CKPT_DIR,"fraud_classifier.pkl")
    clf=joblib.load(clf_path) if os.path.isfile(clf_path) else None
    history=None
    hp=os.path.join(CKPT_DIR,"training_history.pkl")
    if os.path.isfile(hp):
        with open(hp,"rb") as f: history=pickle.load(f)
    return prep,cond_vec,G,clf,history

@st.cache_data
def load_real_data():
    import kagglehub
    _k=kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    return pd.read_csv(os.path.join(_k,"creditcard.csv"))

def _jsd_column(a, b, bins=50):
    """Jensen-Shannon on shared-bin histograms (density-normalized)."""
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    lo = float(min(a.min(), b.min()))
    hi = float(max(a.max(), b.max()))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    p, _ = np.histogram(a, bins=edges, density=True)
    q, _ = np.histogram(b, bins=edges, density=True)
    p = np.asarray(p, dtype=np.float64) + 1e-10
    q = np.asarray(q, dtype=np.float64) + 1e-10
    return float(jensenshannon(p / p.sum(), q / q.sum()))

prep,cond_vec,G,clf,history = load_models()

# ---- Custom CSS ----
st.markdown("""
<style>
    .main-header {font-size:2.5rem; font-weight:700; background:linear-gradient(90deg,#378ADD,#1D9E75); -webkit-background-clip:text; -webkit-text-fill-color:transparent;}
    .metric-card {background:linear-gradient(135deg,#1e1e2e,#2d2d44); padding:20px; border-radius:12px; border-left:4px solid; margin-bottom:10px;}
    .metric-card h3 {color:#999; font-size:0.85rem; margin:0;}
    .metric-card p {color:#fff; font-size:1.8rem; font-weight:700; margin:5px 0 0 0;}
    div[data-testid="stSidebar"] {background:linear-gradient(180deg,#0e1117,#1a1a2e);}
    .stTabs [data-baseweb="tab"] {font-weight:600;}
</style>
""", unsafe_allow_html=True)

# ---- Sidebar ----
st.sidebar.markdown("## 🧬 AnalyticGAN")
st.sidebar.markdown("*Synthetic Fraud Data Platform*")
st.sidebar.markdown("---")
page = st.sidebar.radio("", [
    "🏠 Dashboard",
    "⚡ Generate Data",
    "🔍 Fraud Detector",
    "📊 Distribution Explorer",
    "📈 Training Monitor",
    "🏆 Model Comparison",
])
st.sidebar.markdown("---")
st.sidebar.markdown("**EAI 6020** | Northeastern University")
st.sidebar.markdown("Bigyan Khadka | Spring 2026")


# ======== DASHBOARD ========
if page == "🏠 Dashboard":
    st.markdown('<p class="main-header">AnalyticGAN Dashboard</p>', unsafe_allow_html=True)
    st.markdown("Privacy-Preserving Synthetic Data Generation for Fraud Detection")

    # KPI Row
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card" style="border-color:#378ADD;"><h3>DATASET SIZE</h3><p>284,807</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card" style="border-color:#D85A30;"><h3>FRAUD RATE</h3><p>0.17%</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card" style="border-color:#1D9E75;"><h3>FEATURES</h3><p>29</p></div>', unsafe_allow_html=True)
    with c4:
        n_ep = len(history["d_loss"]) if history else 0
        st.markdown(f'<div class="metric-card" style="border-color:#7F77DD;"><h3>EPOCHS TRAINED</h3><p>{n_ep}</p></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Architecture + W-distance gauge
    col1,col2 = st.columns([2,1])
    with col1:
        st.markdown("### Architecture")
        st.markdown("""
        | Component | Design |
        |---|---|
        | **Generator** | Residual Blocks + Self-Attention |
        | **Discriminator** | Spectral Norm + PacGAN (pac=2) |
        | **Loss** | WGAN-GP (lambda=10) |
        | **Encoding** | Variational Gaussian Mixture |
        | **Sampling** | Fraud oversampled to 20% |
        | **Baseline** | Flow Matching (OT-CFM) |
        """)

    with col2:
        if history:
            final_w = history["w_dist"][-1]
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=final_w,
                title={"text":"W-Distance","font":{"size":16}},
                delta={"reference":history["w_dist"][0],"decreasing":{"color":"#1D9E75"}},
                gauge={
                    "axis":{"range":[0,5],"tickwidth":1},
                    "bar":{"color":"#378ADD"},
                    "steps":[
                        {"range":[0,1],"color":"#1D9E75"},
                        {"range":[1,2],"color":"#FFC107"},
                        {"range":[2,5],"color":"#D85A30"}],
                    "threshold":{"line":{"color":"red","width":2},"thickness":0.75,"value":final_w}
                }))
            fig_gauge.update_layout(height=250,margin=dict(t=50,b=0,l=20,r=20))
            st.plotly_chart(fig_gauge,use_container_width=True)

    # Quick metrics from CSVs
    st.markdown("### Evaluation Snapshot")
    mc1,mc2,mc3 = st.columns(3)
    ml_path = os.path.join(CKPT_DIR,"ml_efficacy.csv")
    if os.path.isfile(ml_path):
        df_ml = pd.read_csv(ml_path)
        with mc1:
            trtr_row = df_ml[df_ml["Setup"].str.contains("TRTR|baseline",case=False,na=False)]
            if not trtr_row.empty:
                st.metric("TRTR ROC-AUC",f"{trtr_row.iloc[0]['ROC-AUC']:.4f}")
        with mc2:
            tstr_row = df_ml[df_ml["Setup"].str.contains("TSTR|synthetic",case=False,na=False)]
            if not tstr_row.empty:
                st.metric("TSTR ROC-AUC",f"{tstr_row.iloc[0]['ROC-AUC']:.4f}")
    fm_path = os.path.join(CKPT_DIR,"flow_matching_comparison.csv")
    if os.path.isfile(fm_path):
        with mc3:
            df_fm = pd.read_csv(fm_path)
            if "Flow Matching" in df_fm.columns:
                jsd_val = df_fm[df_fm["Metric"]=="Mean JSD"]["Flow Matching"].values
                if len(jsd_val)>0:
                    st.metric("FM Mean JSD",f"{jsd_val[0]}")


# ======== GENERATE DATA ========
elif page == "⚡ Generate Data":
    st.markdown('<p class="main-header">Synthetic Data Generator</p>', unsafe_allow_html=True)

    col1,col2,col3 = st.columns(3)
    with col1: n_samples = st.slider("Samples",100,10000,2000,100)
    with col2: fraud_pct = st.slider("Fraud %",1,50,20,1)/100.0
    with col3: st.markdown(f"**{int(n_samples*fraud_pct)}** fraud / **{n_samples-int(n_samples*fraud_pct)}** legit")

    if st.button("Generate",type="primary",use_container_width=True):
        progress = st.progress(0,text="Preparing latent vectors...")
        n_fraud=int(n_samples*fraud_pct);n_legit=n_samples-n_fraud
        labels=np.concatenate([np.ones(n_fraud),np.zeros(n_legit)]);np.random.shuffle(labels)
        eye=np.eye(2);cond=torch.tensor(eye[labels.astype(int)],dtype=torch.float32)

        progress.progress(30,text="Running generator...")
        with torch.no_grad():
            z=torch.randn(n_samples,128);out=G(z,cond)
        progress.progress(70,text="Decoding to tabular format...")
        df_gen=prep.inverse_transform(out);df_gen["Class"]=labels.astype(int)
        progress.progress(100,text="Done!")

        st.session_state["generated_data"]=df_gen

        # Metrics
        m1,m2,m3 = st.columns(3)
        m1.metric("Total",n_samples);m2.metric("Fraud",n_fraud);m3.metric("Legit",n_legit)

        # Tabs
        tab1,tab2,tab3 = st.tabs(["Distribution","Data Table","Feature Explorer"])
        with tab1:
            fig=px.histogram(df_gen,x="Class",color="Class",color_discrete_map={0:"#378ADD",1:"#D85A30"},
                            labels={"Class":"Transaction Type"},title="Generated Class Distribution")
            fig.update_layout(bargap=0.3,showlegend=False)
            st.plotly_chart(fig,use_container_width=True)

        with tab2:
            st.dataframe(df_gen.head(30),use_container_width=True)

        with tab3:
            feat=st.selectbox("Select feature",FEATURES,index=len(FEATURES)-1)
            fig=px.histogram(df_gen,x=feat,color="Class",nbins=60,opacity=0.7,barmode="overlay",
                            color_discrete_map={0:"#378ADD",1:"#D85A30"},
                            title=f"{feat} Distribution by Class")
            st.plotly_chart(fig,use_container_width=True)

        csv=df_gen.to_csv(index=False).encode()
        st.download_button("Download CSV",csv,"synthetic_data.csv","text/csv",type="primary",use_container_width=True)


# ======== FRAUD DETECTOR ========
elif page == "🔍 Fraud Detector":
    st.markdown('<p class="main-header">Fraud Detection</p>',unsafe_allow_html=True)

    if clf is None:
        st.error("fraud_classifier.pkl not found. Run Notebook 6 first.")
    else:
        source = st.radio("Data source",["Upload CSV","Use generated data"],horizontal=True)
        df_input = None

        if source == "Upload CSV":
            uploaded = st.file_uploader("Upload CSV with V1-V28 + Amount",type=["csv"])
            if uploaded: df_input = pd.read_csv(uploaded)
        else:
            if "generated_data" in st.session_state:
                df_input = st.session_state["generated_data"]
            else:
                st.info("Generate data first on the Generate Data page.")

        if df_input is not None:
            missing=[c for c in FEATURES if c not in df_input.columns]
            if missing:
                st.error(f"Missing: {missing}")
            else:
                threshold = st.slider("Detection Threshold",0.0,1.0,0.5,0.01)
                proba=clf.predict_proba(df_input[FEATURES].values)[:,1]
                preds=(proba>=threshold).astype(int)
                df_input["Probability"]=proba.round(4)
                df_input["Prediction"]=["Fraud" if p else "Legit" for p in preds]
                n_fraud=preds.sum()

                # KPIs
                k1,k2,k3 = st.columns(3)
                k1.metric("Transactions",len(df_input))
                k2.metric("Fraud Detected",int(n_fraud))
                k3.metric("Fraud Rate",f"{n_fraud/len(df_input)*100:.2f}%")

                tab1,tab2,tab3 = st.tabs(["Probability Distribution","Risk Scatter","Predictions"])
                with tab1:
                    fig=go.Figure()
                    fig.add_trace(go.Histogram(x=proba[preds==0],name="Legit",marker_color="#378ADD",opacity=0.7))
                    fig.add_trace(go.Histogram(x=proba[preds==1],name="Fraud",marker_color="#D85A30",opacity=0.7))
                    fig.add_vline(x=threshold,line_dash="dash",line_color="red",annotation_text=f"Threshold={threshold}")
                    fig.update_layout(title="Fraud Probability Distribution",barmode="overlay",
                                     xaxis_title="Probability",yaxis_title="Count")
                    st.plotly_chart(fig,use_container_width=True)

                with tab2:
                    sample_df=df_input.head(2000).copy()
                    fig=px.scatter(sample_df,x="V1",y="V2",color="Prediction",
                                  size="Probability",hover_data=["Amount","Probability"],
                                  color_discrete_map={"Legit":"#378ADD","Fraud":"#D85A30"},
                                  title="Transaction Risk Map (V1 vs V2)",opacity=0.6)
                    st.plotly_chart(fig,use_container_width=True)

                with tab3:
                    st.dataframe(df_input[["Probability","Prediction"]+FEATURES[:5]].head(50),use_container_width=True)

                csv=df_input.to_csv(index=False).encode()
                st.download_button("Download Predictions",csv,"predictions.csv","text/csv")


# ======== DISTRIBUTION EXPLORER ========
elif page == "📊 Distribution Explorer":
    st.markdown('<p class="main-header">Distribution Explorer</p>',unsafe_allow_html=True)
    st.markdown("Compare real vs synthetic distributions interactively")

    df_real = load_real_data()

    col1,col2 = st.columns(2)
    with col1: n_gen=st.slider("Synthetic samples",500,5000,2000,500)
    with col2: selected_features=st.multiselect("Features",FEATURES,default=["Amount","V1","V2","V3","V4"])

    # Generate
    n_f=int(n_gen*0.2);n_l=n_gen-n_f;labels=np.concatenate([np.ones(n_f),np.zeros(n_l)]);np.random.shuffle(labels)
    eye=np.eye(2);cond=torch.tensor(eye[labels.astype(int)],dtype=torch.float32)
    with torch.no_grad(): out=G(torch.randn(n_gen,128),cond)
    df_gen=prep.inverse_transform(out);df_gen["Class"]=labels.astype(int)

    # Load CTGAN sample
    ctgan_path=os.path.join(CKPT_DIR,"synthetic_sample.csv")
    df_ctgan=pd.read_csv(ctgan_path) if os.path.isfile(ctgan_path) else None

    for feat in selected_features:
        fig=go.Figure()
        fig.add_trace(go.Histogram(x=df_real[feat],name="Real",marker_color="#378ADD",opacity=0.5,nbinsx=80))
        fig.add_trace(go.Histogram(x=df_gen[feat],name="Generated (Live)",marker_color="#1D9E75",opacity=0.5,nbinsx=80))
        if df_ctgan is not None and feat in df_ctgan.columns:
            fig.add_trace(go.Histogram(x=df_ctgan[feat],name="CTGAN Sample",marker_color="#D85A30",opacity=0.3,nbinsx=80))
        fig.update_layout(title=f"{feat} Distribution",barmode="overlay",
                         xaxis_title=feat,yaxis_title="Count",height=350)
        st.plotly_chart(fig,use_container_width=True)

    # Stats comparison
    st.markdown("### Statistics Comparison")
    stats_rows=[]
    for feat in selected_features:
        stats_rows.append({"Feature":feat,
            "Real Mean":f"{df_real[feat].mean():.4f}","Synth Mean":f"{df_gen[feat].mean():.4f}",
            "Real Std":f"{df_real[feat].std():.4f}","Synth Std":f"{df_gen[feat].std():.4f}",
            "JSD":f"{_jsd_column(df_real[feat].values, df_gen[feat].values):.4f}"})
    st.dataframe(pd.DataFrame(stats_rows),use_container_width=True)


# ======== TRAINING MONITOR ========
elif page == "📈 Training Monitor":
    st.markdown('<p class="main-header">Training Monitor</p>',unsafe_allow_html=True)

    if history is None:
        st.error("training_history.pkl not found.")
    else:
        epochs=len(history["d_loss"])

        # KPIs
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Epochs",epochs)
        k2.metric("Final W-Dist",f"{history['w_dist'][-1]:.4f}",
                  delta=f"{history['w_dist'][-1]-history['w_dist'][0]:.2f}")
        k3.metric("Final D Loss",f"{history['d_loss'][-1]:.4f}")
        k4.metric("Final G Loss",f"{history['g_loss'][-1]:.4f}")

        # Epoch range
        ep_range=st.slider("Epoch range",1,epochs,(1,epochs))

        # Interactive Plotly training curves
        fig=make_subplots(rows=2,cols=2,subplot_titles=["Discriminator Loss","Generator Loss","Gradient Penalty","Wasserstein Distance"])
        ep=list(range(ep_range[0],ep_range[1]+1))
        for row,col,key,color in [(1,1,"d_loss","#D85A30"),(1,2,"g_loss","#378ADD"),(2,1,"gp","#7F77DD"),(2,2,"w_dist","#1D9E75")]:
            data=history[key][ep_range[0]-1:ep_range[1]]
            fig.add_trace(go.Scatter(x=ep,y=data,mode="lines",line=dict(color=color,width=2),name=key,
                                     hovertemplate="Epoch %{x}<br>Value: %{y:.4f}"),row=row,col=col)
        fig.update_layout(height=600,title_text=f"WGAN-GP Training Curves (Epochs {ep_range[0]}-{ep_range[1]})",
                         showlegend=False)
        st.plotly_chart(fig,use_container_width=True)

        # Training analysis
        w_imp=((history["w_dist"][0]-history["w_dist"][-1])/history["w_dist"][0])*100
        st.markdown(f"""
        ### Analysis
        - W-Distance reduced by **{w_imp:.1f}%** ({history['w_dist'][0]:.4f} -> {history['w_dist'][-1]:.4f})
        - Generator loss trend: **{'Increasing (D overpowering G)' if history['g_loss'][-1]>history['g_loss'][0] else 'Decreasing (healthy)'}**
        - Gradient penalty converged to **{history['gp'][-1]:.4f}**
        """)


# ======== MODEL COMPARISON ========
elif page == "🏆 Model Comparison":
    st.markdown('<p class="main-header">CTGAN vs Flow Matching</p>',unsafe_allow_html=True)

    tab1,tab2,tab3,tab4,tab5 = st.tabs(["JSD Comparison","NNDR Privacy","Distributions","ML Efficacy","All Figures"])

    with tab1:
        p=os.path.join(CKPT_DIR,"figH_jsd_comparison.png")
        if os.path.isfile(p): st.image(p,use_container_width=True)

        fm_path=os.path.join(CKPT_DIR,"flow_matching_comparison.csv")
        if os.path.isfile(fm_path):
            df_fm=pd.read_csv(fm_path)
            st.dataframe(df_fm,use_container_width=True)

    with tab2:
        p=os.path.join(CKPT_DIR,"figI_nndr_comparison.png")
        if os.path.isfile(p): st.image(p,use_container_width=True)

    with tab3:
        p=os.path.join(CKPT_DIR,"figJ_three_way.png")
        if os.path.isfile(p): st.image(p,use_container_width=True)

    with tab4:
        for csv_name,title in [("ml_efficacy.csv","TSTR vs TRTR"),("classifier_results.csv","3-Way Classifier")]:
            p=os.path.join(CKPT_DIR,csv_name)
            if os.path.isfile(p):
                st.markdown(f"**{title}**")
                df=pd.read_csv(p)
                st.dataframe(df,use_container_width=True)

                if "ROC-AUC" in df.columns:
                    fig=px.bar(df,x="Setup",y="ROC-AUC",color="Setup",
                              color_discrete_sequence=["#378ADD","#D85A30","#1D9E75"],
                              title=f"{title} - ROC-AUC Comparison")
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig,use_container_width=True)

        p=os.path.join(CKPT_DIR,"figF_roc.png")
        if os.path.isfile(p):
            st.markdown("**ROC Curves**")
            st.image(p,use_container_width=True)

    with tab5:
        st.markdown("### All Evaluation Figures")
        figs=[
            ("figA_jsd.png","JSD per Column"),
            ("figB_correlation.png","Correlation Structure"),
            ("figD_nndr.png","Privacy NNDR"),
            ("figE_training_recap.png","Training Curves"),
            ("figF_roc.png","ROC Curves"),
            ("figG_feature_importance.png","Feature Importances"),
            ("figH_jsd_comparison.png","JSD: CTGAN vs FM"),
            ("figI_nndr_comparison.png","NNDR Comparison"),
            ("figJ_three_way.png","3-Way Distribution"),
        ]
        for fname,caption in figs:
            p=os.path.join(CKPT_DIR,fname)
            if os.path.isfile(p):
                with st.expander(caption,expanded=False):
                    st.image(p,use_container_width=True)
