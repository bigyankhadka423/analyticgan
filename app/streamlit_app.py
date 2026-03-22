"""
AnalyticGAN -- Charts-Only Interactive Dashboard
"""
import os, sys, pickle, joblib, time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import jensenshannon
from scipy.spatial import cKDTree

_app_dir = os.path.dirname(os.path.abspath(__file__))
BASE     = os.path.dirname(_app_dir)
CKPT_DIR = os.path.join(BASE, "checkpoints")
sys.path.insert(0, BASE)
FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]

st.set_page_config(page_title="AnalyticGAN", page_icon="🧬", layout="wide")

# ---- Classes ----
class VGMEncoder:
    def __init__(self, n_components=10, eps=0.005):
        self.n_components=n_components; self.eps=eps
        self.bgm=BayesianGaussianMixture(n_components=n_components,weight_concentration_prior_type="dirichlet_process",weight_concentration_prior=0.001,max_iter=100,random_state=42,n_init=1)
        self.valid_components=None; self.n_valid=None
    def fit(self,data):
        self.bgm.fit(np.asarray(data).reshape(-1,1)); self.valid_components=np.where(self.bgm.weights_>self.eps)[0]; self.n_valid=len(self.valid_components); return self
    def transform(self,data):
        data=np.asarray(data).reshape(-1,1); means=self.bgm.means_[self.valid_components].flatten(); stds=np.sqrt(self.bgm.covariances_[self.valid_components]).flatten()
        probs=self.bgm.predict_proba(data)[:,self.valid_components]; mode_idx=[]
        for p in probs:
            s=p.sum(); p_norm=(p/s).astype(np.float64) if(s>0 and np.isfinite(s)) else np.ones(self.n_valid)/self.n_valid; mode_idx.append(np.random.choice(self.n_valid,p=p_norm))
        mode_idx=np.array(mode_idx); normalized=np.clip((data.flatten()-means[mode_idx])/(4*stds[mode_idx]+1e-8),-0.99,0.99)
        one_hot=np.zeros((len(data),self.n_valid),dtype=np.float32); one_hot[np.arange(len(data)),mode_idx]=1
        return np.column_stack([normalized,one_hot]).astype(np.float32)
    def inverse_transform(self,encoded):
        encoded=np.asarray(encoded); means=self.bgm.means_[self.valid_components].flatten(); stds=np.sqrt(self.bgm.covariances_[self.valid_components]).flatten()
        mode_idx=np.argmax(encoded[:,1:],axis=1); return encoded[:,0]*4*stds[mode_idx]+means[mode_idx]

class TabularPreprocessor:
    def __init__(self,max_gmm_components=10,eps=0.005):
        self.max_gmm_components=max_gmm_components;self.eps=eps;self.continuous_cols=[];self.categorical_cols=[];self.target_col=None;self.vgm_encoders={};self.label_encoders={};self.cat_dims={};self.output_info=[];self.output_dim=0
    def inverse_transform(self,tensor):
        data=tensor.detach().cpu().numpy() if hasattr(tensor,"detach") else tensor; result={};idx=0
        for kind,col,size in self.output_info:
            if kind=="continuous": w=1+self.vgm_encoders[col].n_valid; result[col]=self.vgm_encoders[col].inverse_transform(data[:,idx:idx+w]); idx+=w
            else: n_cat=self.cat_dims[col]; result[col]=self.label_encoders[col].inverse_transform(np.argmax(data[:,idx:idx+n_cat],axis=1)); idx+=n_cat
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
        super().__init__(); ad=max(dim//8,1); self.query=nn.Linear(dim,ad,bias=False); self.key=nn.Linear(dim,ad,bias=False); self.value=nn.Linear(dim,ad,bias=False); self.out_proj=nn.Linear(ad,dim,bias=False); self.scale=ad**-0.5
    def forward(self,x):
        Q,K,V=self.query(x),self.key(x),self.value(x); return x+self.out_proj(F.softmax(Q@K.T*self.scale,dim=-1)@V)

class Generator(nn.Module):
    def __init__(self,latent_dim,cond_dim,output_dim,output_info,hidden_dims=None):
        super().__init__()
        if hidden_dims is None: hidden_dims=[256,256]
        self.output_info=output_info; self.input_layer=nn.Sequential(nn.Linear(latent_dim+cond_dim,hidden_dims[0]),nn.BatchNorm1d(hidden_dims[0]),nn.ReLU())
        self.res_blocks=nn.ModuleList([ResidualBlock(d) for d in hidden_dims]); self.self_attn=SelfAttention(hidden_dims[-1]); self.output_layer=nn.Linear(hidden_dims[-1],output_dim)
    def forward(self,z,cond):
        x=self.input_layer(torch.cat([z,cond],dim=1))
        for b in self.res_blocks: x=b(x)
        x=self.self_attn(x); return self._apply_activations(self.output_layer(x))
    def _apply_activations(self,x):
        out=[];idx=0
        for kind,_,size in self.output_info:
            if kind=="continuous": out.append(torch.tanh(x[:,idx:idx+1])); out.append(F.softmax(x[:,idx+1:idx+1+size],dim=1)); idx+=1+size
            else: out.append(F.softmax(x[:,idx:idx+size],dim=1)); idx+=size
        return torch.cat(out,dim=1)

# ---- Load ----
@st.cache_resource
def load_models():
    prep=TabularPreprocessor.load(os.path.join(CKPT_DIR,"preprocessor.pkl"))
    cv=np.load(os.path.join(CKPT_DIR,"cond_vec.npy"))
    G=Generator(128,cv.shape[1],prep.output_dim,prep.output_info)
    sd=torch.load(os.path.join(CKPT_DIR,"generator_final.pt"),map_location="cpu")
    sd={k.replace("_orig_mod.",""):v for k,v in sd.items()}
    G.load_state_dict(sd); G.eval()
    clf=joblib.load(os.path.join(CKPT_DIR,"fraud_classifier.pkl")) if os.path.isfile(os.path.join(CKPT_DIR,"fraud_classifier.pkl")) else None
    h=None
    hp=os.path.join(CKPT_DIR,"training_history.pkl")
    if os.path.isfile(hp):
        with open(hp,"rb") as f: h=pickle.load(f)
    return prep,cv,G,clf,h

@st.cache_data
def load_real():
    import kagglehub; k=kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    return pd.read_csv(os.path.join(k,"creditcard.csv"))

prep,cond_vec,G,clf,history=load_models()
df_real=load_real()
ctgan_path=os.path.join(CKPT_DIR,"synthetic_sample.csv")
df_ctgan=pd.read_csv(ctgan_path) if os.path.isfile(ctgan_path) else None

# ---- Helpers ----
def gen_synth(n,fp=0.20):
    nf=int(n*fp); nl=n-nf; lb=np.concatenate([np.ones(nf),np.zeros(nl)]); np.random.shuffle(lb)
    c=torch.tensor(np.eye(2)[lb.astype(int)],dtype=torch.float32)
    with torch.no_grad(): z=torch.randn(n,128); o=G(z,c)
    df=prep.inverse_transform(o); df["Class"]=lb.astype(int); return df

def jsd(r,s,bins=50):
    lo=min(r.min(),s.min()); hi=max(r.max(),s.max()); e=np.linspace(lo,hi,bins+1)
    p,_=np.histogram(r,bins=e,density=True); q,_=np.histogram(s,bins=e,density=True)
    p=p+1e-10; q=q+1e-10; return jensenshannon(p/p.sum(),q/q.sum())

C = {"blue":"#4facfe","red":"#f5576c","green":"#43e97b","purple":"#667eea","pink":"#fa709a","orange":"#f093fb","gray":"#a8b2d1"}

# ---- CSS ----
st.markdown("""<style>
.hdr{background:linear-gradient(135deg,#1a1a2e,#16213e,#0f3460);padding:1.5rem 2rem;border-radius:12px;margin-bottom:1.2rem;color:white}
.hdr h1{color:white;font-size:2rem;margin:0}.hdr p{color:#a8b2d1;margin:0}
.mc{padding:1rem;border-radius:10px;text-align:center;color:white;margin-bottom:0.8rem}
.mc .v{font-size:1.6rem;font-weight:bold}.mc .l{font-size:0.8rem;opacity:0.85}
#MainMenu{visibility:hidden}footer{visibility:hidden}.stDeployButton{display:none}
</style>""",unsafe_allow_html=True)

def hdr(t,s=""): st.markdown(f'<div class="hdr"><h1>{t}</h1><p>{s}</p></div>',unsafe_allow_html=True)
def mc(l,v,c="#667eea"): st.markdown(f'<div class="mc" style="background:linear-gradient(135deg,{c},{c}aa)"><div class="v">{v}</div><div class="l">{l}</div></div>',unsafe_allow_html=True)

# ---- Sidebar ----
with st.sidebar:
    st.markdown("## 🧬 AnalyticGAN")
    st.caption("EAI 6020 | Northeastern University")
    st.markdown("---")
    page=st.radio("",["🏠 Overview","⚡ Generator","🔍 Fraud Detector","📊 Distribution Lab","📈 Training","🧪 Evaluation","⚔️ GAN vs FM","📉 Metrics"],label_visibility="collapsed")
    st.markdown("---")
    st.caption("Bigyan Khadka | Prof. Siddharth Rout")
    st.markdown("[GitHub](https://github.com/bigyankhadka423/analyticgan)")

# ======== OVERVIEW ========
if page=="🏠 Overview":
    hdr("🧬 AnalyticGAN","Privacy-Preserving Synthetic Data for Fraud Detection")
    c1,c2,c3,c4,c5=st.columns(5)
    with c1: mc("Dataset","284,807",C["blue"])
    with c2: mc("Fraud Rate","0.17%",C["red"])
    with c3: mc("Features","29",C["purple"])
    with c4: mc("Epochs",f"{len(history['d_loss'])}" if history else "-",C["green"])
    with c5: mc("W-Distance",f"{history['w_dist'][-1]:.3f}" if history else "-",C["pink"])

    c1,c2,c3=st.columns(3)
    with c1:
        fig,ax=plt.subplots(figsize=(5,4))
        cts=df_real["Class"].value_counts().sort_index()
        bars=ax.bar(["Legit","Fraud"],cts.values,color=[C["blue"],C["red"]],edgecolor="white",linewidth=1.5)
        for b,v in zip(bars,cts.values): ax.text(b.get_x()+b.get_width()/2,b.get_height()+1000,f"{v:,}",ha="center",fontsize=10,fontweight="bold")
        ax.set_title("Class Imbalance",fontweight="bold"); ax.set_ylabel("Count"); st.pyplot(fig); plt.close()
    with c2:
        fig,ax=plt.subplots(figsize=(5,4))
        ax.hist(df_real["Amount"],bins=100,color=C["purple"],alpha=0.8,edgecolor="white")
        ax.set_title("Transaction Amount",fontweight="bold"); ax.set_xlabel("$"); ax.set_xlim(0,500); st.pyplot(fig); plt.close()
    with c3:
        fig,ax=plt.subplots(figsize=(5,4))
        fraud=df_real[df_real["Class"]==1]; legit=df_real[df_real["Class"]==0].sample(500,random_state=42)
        ax.scatter(legit["V1"],legit["V2"],alpha=0.3,s=5,c=C["blue"],label="Legit")
        ax.scatter(fraud["V1"],fraud["V2"],alpha=0.7,s=12,c=C["red"],label="Fraud")
        ax.set_title("V1 vs V2",fontweight="bold"); ax.legend(fontsize=8); st.pyplot(fig); plt.close()

    # Correlation heatmap
    fig,ax=plt.subplots(figsize=(12,5))
    corr=df_real[FEATURES[:15]].corr()
    im=ax.imshow(corr.values,cmap="coolwarm",vmin=-1,vmax=1)
    ax.set_xticks(range(15)); ax.set_xticklabels(FEATURES[:15],rotation=45,ha="right",fontsize=8)
    ax.set_yticks(range(15)); ax.set_yticklabels(FEATURES[:15],fontsize=8)
    ax.set_title("Feature Correlation Matrix (Top 15)",fontweight="bold")
    plt.colorbar(im,ax=ax,fraction=0.046); plt.tight_layout(); st.pyplot(fig); plt.close()


# ======== GENERATOR ========
elif page=="⚡ Generator":
    hdr("⚡ Synthetic Data Generator","Generate and analyze synthetic transactions")
    c1,c2=st.columns(2)
    with c1: n=st.slider("Samples",100,10000,2000,100)
    with c2: fp=st.slider("Fraud %",1,50,20,1)/100.0

    if st.button("Generate",type="primary",use_container_width=True):
        bar=st.progress(0); bar.progress(30); df_g=gen_synth(n,fp); bar.progress(100); time.sleep(0.2); bar.empty()
        st.session_state["gen"]=df_g
        nf=int(df_g["Class"].sum())
        c1,c2,c3,c4=st.columns(4)
        with c1: mc("Total",f"{n:,}",C["blue"])
        with c2: mc("Fraud",f"{nf:,}",C["red"])
        with c3: mc("Legit",f"{n-nf:,}",C["green"])
        with c4: mc("Fraud %",f"{fp*100:.0f}%",C["pink"])

    if "gen" in st.session_state:
        df_g=st.session_state["gen"]
        t1,t2,t3=st.tabs(["Charts","Data","Quality"])
        with t1:
            c1,c2=st.columns(2)
            with c1:
                fig,ax=plt.subplots(figsize=(6,4))
                cts=df_g["Class"].value_counts().sort_index()
                ax.bar(["Legit","Fraud"],cts.values,color=[C["blue"],C["red"]],edgecolor="white")
                ax.set_title("Generated Class Split",fontweight="bold"); st.pyplot(fig); plt.close()
            with c2:
                fig,ax=plt.subplots(figsize=(6,4))
                ax.hist(df_real["Amount"],bins=80,density=True,alpha=0.5,color=C["blue"],label="Real")
                ax.hist(df_g["Amount"],bins=80,density=True,alpha=0.5,color=C["green"],label="Synthetic")
                ax.set_title("Amount: Real vs Synthetic",fontweight="bold"); ax.legend(); st.pyplot(fig); plt.close()
            # 6-feature grid
            fig,axes=plt.subplots(2,3,figsize=(14,7))
            for i,col in enumerate(["V1","V2","V3","V4","V14","Amount"]):
                ax=axes.flatten()[i]
                ax.hist(df_real[col],bins=60,density=True,alpha=0.4,color=C["blue"],label="Real")
                ax.hist(df_g[col],bins=60,density=True,alpha=0.4,color=C["green"],label="Synth")
                j=jsd(df_real[col].values,df_g[col].values)
                ax.set_title(f"{col}  (JSD={j:.3f})",fontweight="bold",fontsize=10); ax.legend(fontsize=7); ax.set_yticks([])
            plt.tight_layout(); st.pyplot(fig); plt.close()
        with t2:
            st.dataframe(df_g.head(20),use_container_width=True)
        with t3:
            jsd_scores={col:jsd(df_real[col].values,df_g[col].values) for col in FEATURES}
            fig,ax=plt.subplots(figsize=(14,5))
            colors=[C["red"] if v>0.1 else C["green"] for v in jsd_scores.values()]
            ax.bar(jsd_scores.keys(),jsd_scores.values(),color=colors,edgecolor="white")
            ax.axhline(0.1,color="red",ls="--",alpha=0.5)
            ax.set_title("JSD per Feature (Generated vs Real)",fontweight="bold")
            ax.set_ylabel("JSD"); plt.xticks(rotation=45,ha="right"); plt.tight_layout(); st.pyplot(fig); plt.close()
            avg=np.mean(list(jsd_scores.values())); passed=sum(1 for v in jsd_scores.values() if v<0.1)
            c1,c2=st.columns(2)
            with c1: mc("Mean JSD",f"{avg:.4f}",C["green"] if avg<0.1 else C["red"])
            with c2: mc("Columns < 0.1",f"{passed}/{len(FEATURES)}",C["green"] if passed>20 else C["red"])

        st.download_button("Download CSV",df_g.to_csv(index=False).encode(),"synthetic.csv","text/csv",use_container_width=True)


# ======== FRAUD DETECTOR ========
elif page=="🔍 Fraud Detector":
    hdr("🔍 Fraud Detection","Upload or generate data for real-time classification")
    if clf is None: st.error("Classifier not loaded."); st.stop()
    src=st.radio("Source",["Upload CSV","Use Generated"],horizontal=True)
    df_in=None
    if src=="Upload CSV":
        f=st.file_uploader("CSV",type=["csv"])
        if f: df_in=pd.read_csv(f)
    elif "gen" in st.session_state: df_in=st.session_state["gen"].copy()
    else: st.warning("Generate data first on Generator page.")

    if df_in is not None:
        missing=[c for c in FEATURES if c not in df_in.columns]
        if missing: st.error(f"Missing: {missing}"); st.stop()
        proba=clf.predict_proba(df_in[FEATURES].values)[:,1]
        th=st.slider("Threshold",0.0,1.0,0.5,0.01)
        preds=(proba>=th).astype(int); nf=preds.sum()
        c1,c2,c3=st.columns(3)
        with c1: mc("Transactions",f"{len(df_in):,}",C["blue"])
        with c2: mc("Fraud",f"{nf:,}",C["red"])
        with c3: mc("Fraud %",f"{nf/len(df_in)*100:.2f}%",C["pink"])

        c1,c2,c3=st.columns(3)
        with c1:
            fig,ax=plt.subplots(figsize=(5,4))
            ax.hist(proba[preds==0],bins=50,alpha=0.6,color=C["blue"],label="Legit",edgecolor="white")
            ax.hist(proba[preds==1],bins=50,alpha=0.6,color=C["red"],label="Fraud",edgecolor="white")
            ax.axvline(th,color="red",ls="--",lw=2)
            ax.set_title("Probability Distribution",fontweight="bold"); ax.legend(); st.pyplot(fig); plt.close()
        with c2:
            fig,ax=plt.subplots(figsize=(5,4))
            ax.pie([len(df_in)-nf,nf],labels=["Legit","Fraud"],colors=[C["blue"],C["red"]],autopct="%1.1f%%",textprops={"fontweight":"bold"})
            ax.set_title("Detection Split",fontweight="bold"); st.pyplot(fig); plt.close()
        with c3:
            fig,ax=plt.subplots(figsize=(5,4))
            bins=np.linspace(0,1,21)
            ax.hist(proba,bins=bins,color=C["purple"],edgecolor="white")
            ax.set_title("Score Distribution",fontweight="bold"); ax.set_xlabel("P(Fraud)"); st.pyplot(fig); plt.close()

        df_in["Fraud_Prob"]=proba.round(4); df_in["Pred"]=["FRAUD" if p else "LEGIT" for p in preds]
        st.dataframe(df_in[["Pred","Fraud_Prob"]+FEATURES[:5]].head(30),use_container_width=True)


# ======== DISTRIBUTION LAB ========
elif page=="📊 Distribution Lab":
    hdr("📊 Distribution Lab","Interactive feature comparison")
    c1,c2,c3=st.columns(3)
    with c1: feat=st.selectbox("Feature",FEATURES,index=FEATURES.index("Amount"))
    with c2: ns=st.slider("Samples",500,10000,3000,500,key="lab")
    with c3: bi=st.slider("Bins",20,200,80,10)

    df_g=gen_synth(ns)
    fig,ax=plt.subplots(figsize=(12,5))
    ax.hist(df_real[feat],bins=bi,density=True,alpha=0.5,color=C["blue"],label="Real",edgecolor="white",linewidth=0.5)
    ax.hist(df_g[feat],bins=bi,density=True,alpha=0.5,color=C["green"],label="Synthetic",edgecolor="white",linewidth=0.5)
    if df_ctgan is not None and feat in df_ctgan.columns:
        ax.hist(df_ctgan[feat],bins=bi,density=True,alpha=0.3,color=C["red"],label="CTGAN(saved)",edgecolor="white")
    j=jsd(df_real[feat].values,df_g[feat].values)
    ax.set_title(f"{feat}  |  JSD = {j:.4f}",fontsize=14,fontweight="bold"); ax.legend(); st.pyplot(fig); plt.close()

    # Multi-feature
    sel=st.multiselect("Compare features",FEATURES,default=["V1","V2","V3","V4","V14","Amount"])
    if sel:
        nc=min(len(sel),6); nr=(nc+2)//3
        fig,axes=plt.subplots(nr,3,figsize=(14,4*nr))
        axes=np.array(axes).flatten()
        for i,col in enumerate(sel[:6]):
            axes[i].hist(df_real[col],bins=50,density=True,alpha=0.5,color=C["blue"],label="Real")
            axes[i].hist(df_g[col],bins=50,density=True,alpha=0.5,color=C["green"],label="Synth")
            j=jsd(df_real[col].values,df_g[col].values)
            axes[i].set_title(f"{col} (JSD={j:.3f})",fontweight="bold",fontsize=10); axes[i].legend(fontsize=7); axes[i].set_yticks([])
        for j in range(len(sel[:6]),len(axes)): axes[j].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()


# ======== TRAINING ========
elif page=="📈 Training":
    hdr("📈 Training Monitor","WGAN-GP convergence analysis")
    if history is None: st.error("No training history."); st.stop()
    ep=len(history["d_loss"])
    c1,c2,c3,c4=st.columns(4)
    wd=((history['w_dist'][0]-history['w_dist'][-1])/history['w_dist'][0])*100
    with c1: mc("Epochs",str(ep),C["blue"])
    with c2: mc("Final W-Dist",f"{history['w_dist'][-1]:.4f}",C["green"])
    with c3: mc("Best W-Dist",f"{min(history['w_dist'],key=abs):.4f}",C["purple"])
    with c4: mc("W-Dist Drop",f"{wd:.1f}%",C["pink"])

    rng=st.slider("Epoch range",1,ep,(1,ep))
    fig,axes=plt.subplots(2,2,figsize=(14,9))
    axes=axes.flatten()
    for ax,key,label,color in zip(axes,["d_loss","g_loss","gp","w_dist"],["D Loss","G Loss","Gradient Penalty","W-Distance"],[C["red"],C["blue"],C["purple"],C["green"]]):
        d=history[key][rng[0]-1:rng[1]]
        ax.plot(range(rng[0],rng[0]+len(d)),d,color=color,linewidth=1.8)
        ax.fill_between(range(rng[0],rng[0]+len(d)),d,alpha=0.1,color=color)
        ax.set_title(label,fontweight="bold",fontsize=12); ax.set_xlabel("Epoch"); ax.grid(alpha=0.2)
    plt.suptitle(f"WGAN-GP Training (Epochs {rng[0]}-{rng[1]})",fontsize=14,fontweight="bold",y=1.01)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # Loss ratio chart
    fig,ax=plt.subplots(figsize=(12,4))
    d_arr=np.array(history["d_loss"][rng[0]-1:rng[1]])
    g_arr=np.array(history["g_loss"][rng[0]-1:rng[1]])
    x=range(rng[0],rng[0]+len(d_arr))
    ax.plot(x,d_arr,color=C["red"],label="D Loss",linewidth=1.5)
    ax.plot(x,g_arr,color=C["blue"],label="G Loss",linewidth=1.5)
    ax.fill_between(x,d_arr,g_arr,alpha=0.1,color=C["purple"])
    ax.set_title("D vs G Loss Gap",fontweight="bold"); ax.legend(); ax.grid(alpha=0.2)
    st.pyplot(fig); plt.close()


# ======== EVALUATION ========
elif page=="🧪 Evaluation":
    hdr("🧪 Evaluation Suite","Statistical Fidelity | ML Efficacy | Privacy")
    t1,t2,t3=st.tabs(["Statistical Fidelity","ML Efficacy","Privacy"])
    with t1:
        c1,c2=st.columns(2)
        with c1:
            p=os.path.join(CKPT_DIR,"figA_jsd.png")
            if os.path.isfile(p): st.image(p,use_container_width=True)
        with c2:
            p=os.path.join(CKPT_DIR,"figB_correlation.png")
            if os.path.isfile(p): st.image(p,use_container_width=True)
    with t2:
        c1,c2=st.columns(2)
        with c1:
            p=os.path.join(CKPT_DIR,"figF_roc.png")
            if os.path.isfile(p): st.image(p,use_container_width=True)
        with c2:
            p=os.path.join(CKPT_DIR,"figG_feature_importance.png")
            if os.path.isfile(p): st.image(p,use_container_width=True)
        for csv_name in ["ml_efficacy.csv","classifier_results.csv"]:
            p=os.path.join(CKPT_DIR,csv_name)
            if os.path.isfile(p): st.dataframe(pd.read_csv(p),use_container_width=True)
    with t3:
        c1,c2=st.columns(2)
        with c1:
            p=os.path.join(CKPT_DIR,"figD_nndr.png")
            if os.path.isfile(p): st.image(p,use_container_width=True)
        with c2:
            p=os.path.join(CKPT_DIR,"figI_nndr_comparison.png")
            if os.path.isfile(p): st.image(p,use_container_width=True)


# ======== GAN vs FM ========
elif page=="⚔️ GAN vs FM":
    hdr("⚔️ CTGAN vs Flow Matching","Adversarial vs Non-Adversarial Comparison")

    c1,c2=st.columns(2)
    with c1:
        p=os.path.join(CKPT_DIR,"figH_jsd_comparison.png")
        if os.path.isfile(p): st.image(p,use_container_width=True)
    with c2:
        p=os.path.join(CKPT_DIR,"figJ_three_way.png")
        if os.path.isfile(p): st.image(p,use_container_width=True)

    c1,c2=st.columns(2)
    with c1:
        p=os.path.join(CKPT_DIR,"figI_nndr_comparison.png")
        if os.path.isfile(p): st.image(p,use_container_width=True)
    with c2:
        p=os.path.join(CKPT_DIR,"flow_matching_comparison.csv")
        if os.path.isfile(p):
            df_comp=pd.read_csv(p)
            st.dataframe(df_comp,use_container_width=True)

    # Side-by-side bar chart
    p=os.path.join(CKPT_DIR,"flow_matching_comparison.csv")
    if os.path.isfile(p):
        df_c=pd.read_csv(p)
        if "Flow Matching" in df_c.columns and "CTGAN (AnalyticGAN)" in df_c.columns:
            fig,ax=plt.subplots(figsize=(10,5))
            metrics=df_c["Metric"].tolist()
            fm_vals=[]; ct_vals=[]
            for _,row in df_c.iterrows():
                try: fm_vals.append(float(row["Flow Matching"]))
                except: fm_vals.append(0)
                try: ct_vals.append(float(row["CTGAN (AnalyticGAN)"]))
                except: ct_vals.append(0)
            x=np.arange(len(metrics)); w=0.35
            ax.bar(x-w/2,ct_vals,w,color=C["red"],label="CTGAN",edgecolor="white")
            ax.bar(x+w/2,fm_vals,w,color=C["green"],label="Flow Matching",edgecolor="white")
            ax.set_xticks(x); ax.set_xticklabels(metrics,rotation=15)
            ax.set_title("CTGAN vs Flow Matching — All Metrics",fontweight="bold",fontsize=13)
            ax.legend(fontsize=11); ax.grid(axis="y",alpha=0.2)
            plt.tight_layout(); st.pyplot(fig); plt.close()


# ======== METRICS (chart-only) ========
elif page=="📉 Metrics":
    hdr("📉 Key Metrics","ROC-AUC and scores from saved runs")
    r1,r2=st.columns(2)
    with r1:
        p_ml=os.path.join(CKPT_DIR,"ml_efficacy.csv")
        if os.path.isfile(p_ml):
            dfm=pd.read_csv(p_ml)
            if "Setup" in dfm.columns and "ROC-AUC" in dfm.columns:
                fig,ax=plt.subplots(figsize=(10,5))
                setups=dfm["Setup"].astype(str).tolist()
                aucs=[float(x) if pd.notna(x) else 0.0 for x in dfm["ROC-AUC"]]
                cols=[C["blue"],C["red"],C["green"],C["purple"],C["pink"]]
                bar_c=[cols[i % len(cols)] for i in range(len(setups))]
                ax.barh(setups,aucs,color=bar_c,edgecolor="white")
                ax.set_xlabel("ROC-AUC"); ax.set_title("ML Efficacy (ml_efficacy.csv)",fontweight="bold")
                ax.set_xlim(0,1); ax.grid(axis="x",alpha=0.2)
                plt.tight_layout(); st.pyplot(fig); plt.close()
    with r2:
        p_cl=os.path.join(CKPT_DIR,"classifier_results.csv")
        if os.path.isfile(p_cl):
            dfc=pd.read_csv(p_cl)
            if "Setup" in dfc.columns and "ROC-AUC" in dfc.columns:
                fig,ax=plt.subplots(figsize=(10,5))
                setups=dfc["Setup"].astype(str).tolist()
                aucs=[float(x) if pd.notna(x) else 0.0 for x in dfc["ROC-AUC"]]
                ax.bar(range(len(setups)),aucs,color=C["purple"],edgecolor="white")
                ax.set_xticks(range(len(setups))); ax.set_xticklabels(setups,rotation=25,ha="right")
                ax.set_ylabel("ROC-AUC"); ax.set_title("Classifier comparison",fontweight="bold")
                ax.set_ylim(0,1); ax.grid(axis="y",alpha=0.2)
                plt.tight_layout(); st.pyplot(fig); plt.close()
    p_fm=os.path.join(CKPT_DIR,"flow_matching_comparison.csv")
    if os.path.isfile(p_fm):
        df_fm=pd.read_csv(p_fm)
        cols=[c for c in df_fm.columns if c!="Metric"]
        if cols:
            fig,ax=plt.subplots(figsize=(12,5))
            metrics=df_fm["Metric"].astype(str).tolist()
            x=np.arange(len(metrics)); w=0.8/len(cols)
            for i,cname in enumerate(cols):
                vals=[]
                for _,row in df_fm.iterrows():
                    try: vals.append(float(row[cname]))
                    except: vals.append(0.0)
                ax.bar(x+(i-len(cols)/2)*w+w/2,vals,w,label=cname,edgecolor="white")
            ax.set_xticks(x); ax.set_xticklabels(metrics,rotation=20,ha="right")
            ax.set_title("Flow matching comparison table (numeric columns)",fontweight="bold")
            ax.legend(fontsize=9); ax.grid(axis="y",alpha=0.2)
            plt.tight_layout(); st.pyplot(fig); plt.close()
