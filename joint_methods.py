"""
Joint Novel Species Detection + Clustering
==========================================
All methods work on combined A+B UMAP space.
Detection and clustering happen simultaneously — no two-step pipeline.

Methods
───────
  1a. K-Means(60)  + centroid matching to A
  1b. DEC(60)      + A-fraction matching        ← DEC version of Option 1
  2.  HDBSCAN      + A-fraction labelling
  3.  GMM(60)      + A-fraction matching

Reference: Gaussian Strategy 2 (our best previous result)

Labels used ONLY in evaluate() — never during any fitting step.
"""

import numpy as np, umap, warnings, time
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score
from scipy.optimize import linear_sum_assignment
warnings.filterwarnings("ignore")

SEED=42; K_OLD=50; K_NEW=60; K_NOV=10; N_PER_CLS=100
rng=np.random.default_rng(SEED)

# ── Data ──────────────────────────────────────────────────────────────────────
print("Loading data …")
embeddings=np.load("plantnet_dinov2_raw.npy").astype(np.float32)
labels=np.load("plantnet_labels.npy")
all_classes=np.unique(labels)
counts=np.array([(labels==c).sum() for c in all_classes])
eligible=all_classes[counts>=2*N_PER_CLS]
chosen_60=np.sort(rng.choice(eligible,size=K_NEW,replace=False))
base_cls=chosen_60[:K_OLD]; novel_cls=chosen_60[K_OLD:]

XA,XB,yBl=[],[],[]
for c in base_cls:
    idx=rng.choice(np.where(labels==c)[0],size=2*N_PER_CLS,replace=False)
    XA.append(embeddings[idx[:N_PER_CLS]]); XB.append(embeddings[idx[N_PER_CLS:]]); yBl.extend([c]*N_PER_CLS)
for c in novel_cls:
    idx=rng.choice(np.where(labels==c)[0],size=N_PER_CLS,replace=False)
    XB.append(embeddings[idx]); yBl.extend([c]*N_PER_CLS)

X_A=np.vstack(XA); X_B=np.vstack(XB); y_B=np.array(yBl)
id2idx={c:i for i,c in enumerate(chosen_60)}
y_B_eval=np.array([id2idx[c] for c in y_B])
is_novel_B=y_B_eval>=K_OLD
N_A, N_B = len(X_A), len(X_B)
print(f"  A: {N_A} samples | B: {N_B} samples ({is_novel_B.sum()} novel)")

# ── Combined UMAP on A+B (shared across all methods) ─────────────────────────
print("\nFitting combined UMAP on A+B (768-d → 10-d) …")
t0=time.time()
reducer=umap.UMAP(n_components=10, n_neighbors=20, min_dist=0.05,
                  metric="cosine", random_state=SEED, verbose=False)
X_AB=normalize(reducer.fit_transform(np.vstack([X_A,X_B])),norm="l2").astype(np.float32)
X_A2=X_AB[:N_A]; X_B2=X_AB[N_A:]
print(f"  Done in {time.time()-t0:.1f}s")

# ── Evaluation helpers ────────────────────────────────────────────────────────
def evaluate(novel_mask, novelty_score, tag):
    """novel_mask: boolean over B samples. novelty_score: float per B sample."""
    tp   = is_novel_B[novel_mask].sum()
    prec = tp/novel_mask.sum() if novel_mask.sum()>0 else 0
    rec  = tp/is_novel_B.sum()
    f1   = 2*prec*rec/(prec+rec+1e-9)
    fpr  = (~is_novel_B[novel_mask]).sum()/(~is_novel_B).sum()
    auc  = roc_auc_score(is_novel_B, novelty_score)
    print(f"\n  [{tag}]")
    print(f"    AUC={auc:.4f}  Recall={rec:.1%}  Precision={prec:.1%}  "
          f"F1={f1:.3f}  FPR={fpr:.1%}  Flagged={novel_mask.sum()}")
    return dict(auc=auc, rec=rec, prec=prec, f1=f1)

def cluster_acc_novel(B_cluster_labels, novel_cluster_ids, tag):
    """
    B_cluster_labels : cluster id per B sample (integer array)
    novel_cluster_ids: set of cluster ids deemed novel
    """
    novel_mask = np.isin(B_cluster_labels, list(novel_cluster_ids))
    truly      = is_novel_B[novel_mask]
    y_true     = y_B_eval[novel_mask][truly] - K_OLD   # 0-9
    y_pred     = B_cluster_labels[novel_mask][truly]

    if len(y_true) < K_NOV:
        print(f"    Cluster ACC = N/A  (only {len(y_true)} novel samples recovered)")
        return 0.0

    # remap pred labels to 0..K_NOV-1
    uniq = np.unique(y_pred)
    remap = {v:i for i,v in enumerate(uniq)}
    y_pred_r = np.array([remap[v] for v in y_pred])

    n = max(y_true.max(), y_pred_r.max())+1
    mat = np.zeros((n,n),dtype=np.int64)
    for t,p in zip(y_true, y_pred_r): mat[t,p]+=1
    r,c = linear_sum_assignment(-mat)
    acc = mat[r,c].sum()/len(y_true)
    print(f"    Cluster ACC = {acc:.1%}  ({len(y_true)} novel samples in flagged set)")
    return acc

def a_fraction_novel_ids(cluster_labels_AB, K):
    """
    For each of K clusters, compute fraction of members that come from A.
    Return the K_NOV cluster ids with lowest A-fraction (= most novel).
    cluster_labels_AB: labels over all A+B samples concatenated.
    """
    labels_A = cluster_labels_AB[:N_A]
    labels_B = cluster_labels_AB[N_A:]
    a_frac = np.zeros(K)
    for k in range(K):
        n_a = (labels_A==k).sum()
        n_b = (labels_B==k).sum()
        a_frac[k] = n_a/(n_a+n_b+1e-9)
    novel_ids = set(np.argsort(a_frac)[:K_NOV])   # lowest A-fraction
    return novel_ids, a_frac

all_results = {}

# ══════════════════════════════════════════════════════════════════════════════
# REFERENCE — Gaussian Strategy 2 (best previous method)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*68)
print("REFERENCE — Gaussian on combined UMAP (Strategy 2)")
print("="*68)
km_ref = KMeans(n_clusters=K_OLD, n_init=10, random_state=SEED).fit(X_A2)
cent   = km_ref.cluster_centers_
dists  = np.linalg.norm(X_A2[:,None,:]-cent[None,:,:],axis=-1)
thresh = np.array([dists[km_ref.labels_==k,k].mean() +
                   2*dists[km_ref.labels_==k,k].std() for k in range(K_OLD)])
dists_B= np.linalg.norm(X_B2[:,None,:]-cent[None,:,:],axis=-1)
near_B = dists_B.argmin(1); score_ref=dists_B.min(1)
mask_ref = score_ref > thresh[near_B]
res_ref  = evaluate(mask_ref, score_ref, "Gaussian Strategy 2 (reference)")
# cluster acc: use K-Means(10) on flagged
from sklearn.cluster import KMeans as KM
km_nov = KM(n_clusters=K_NOV, n_init=10, random_state=SEED)
B_tmp  = X_B2[mask_ref]
if len(B_tmp)>=K_NOV:
    tmp_labels = km_nov.fit_predict(B_tmp)
    y_true_r   = y_B_eval[mask_ref][is_novel_B[mask_ref]]-K_OLD
    y_pred_r   = tmp_labels[is_novel_B[mask_ref]]
    n=max(y_true_r.max(),y_pred_r.max())+1
    mat=np.zeros((n,n),dtype=np.int64)
    for t,p in zip(y_true_r,y_pred_r): mat[t,p]+=1
    r,c=linear_sum_assignment(-mat)
    ref_cacc=mat[r,c].sum()/len(y_true_r)
    print(f"    Cluster ACC = {ref_cacc:.1%}  ({y_true_r.sum()>0 and len(y_true_r)} novel samples)")
all_results["Gaussian (ref)"] = (res_ref, ref_cacc)

# ══════════════════════════════════════════════════════════════════════════════
# METHOD 1a — K-Means(60) on combined A+B + centroid matching
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*68)
print("METHOD 1a — K-Means(60) on A+B, match clusters to A centroids")
print("="*68)

km60 = KMeans(n_clusters=K_NEW, n_init=15, random_state=SEED)
labels_AB_km = km60.fit_predict(X_AB)       # over all N_A+N_B samples
cent60 = km60.cluster_centers_              # (60, 10)

# K-Means(50) on A to get old-species centroids
km50a = KMeans(n_clusters=K_OLD, n_init=10, random_state=SEED).fit(X_A2)
cent50 = km50a.cluster_centers_             # (50, 10)

# For each of the 60 clusters: min distance to any old centroid
dist_to_old = np.linalg.norm(
    cent60[:,None,:] - cent50[None,:,:], axis=-1).min(1)   # (60,)
novel_ids_km = set(np.argsort(dist_to_old)[-K_NOV:])      # 10 most distant

# Novelty score per B sample = distance of its assigned cluster to nearest old centroid
B_cluster_labels_km = labels_AB_km[N_A:]
novelty_score_km    = dist_to_old[B_cluster_labels_km]
novel_mask_km       = np.isin(B_cluster_labels_km, list(novel_ids_km))

res_1a = evaluate(novel_mask_km, novelty_score_km, "K-Means(60) centroid match")
cacc_1a = cluster_acc_novel(B_cluster_labels_km, novel_ids_km, "1a")
all_results["K-Means(60) match"] = (res_1a, cacc_1a)

# ══════════════════════════════════════════════════════════════════════════════
# METHOD 1b — DEC(60) on combined A+B + A-fraction matching
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*68)
print("METHOD 1b — DEC(60) on A+B, identify novel clusters by A-fraction")
print("="*68)

# Import torch AFTER all UMAP work
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
torch.manual_seed(SEED)
DEVICE=torch.device("mps" if torch.backends.mps.is_available() else
                    "cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {DEVICE}")

class AE(nn.Module):
    def __init__(self, in_dim=10, latent=6):
        super().__init__()
        self.encoder=nn.Sequential(
            nn.Linear(in_dim,64),nn.ReLU(),
            nn.Linear(64,32),nn.ReLU(),
            nn.Linear(32,latent))
        self.decoder=nn.Sequential(
            nn.Linear(latent,32),nn.ReLU(),
            nn.Linear(32,64),nn.ReLU(),
            nn.Linear(64,in_dim))
    def forward(self,x):
        z=self.encoder(x); return z,self.decoder(z)

def soft_assign(z,c,alpha=1.0):
    d=((z[:,None,:]-c[None,:,:])**2).sum(-1)
    q=(1+d/alpha)**(-(alpha+1)/2)
    return q/q.sum(1,keepdim=True)

def target_dist(q):
    p=(q**2)/q.sum(0,keepdim=True)
    return p/p.sum(1,keepdim=True)

Xt_AB=torch.from_numpy(X_AB)
dl_AB=DataLoader(TensorDataset(Xt_AB),batch_size=256,shuffle=True)

# Phase A: autoencoder on combined A+B
ae=AE().to(DEVICE); opt_ae=torch.optim.Adam(ae.parameters(),lr=1e-3)
print("  Autoencoder pre-training (40 epochs) …")
for ep in range(40):
    ae.train()
    for (xb,) in dl_AB:
        xb=xb.to(DEVICE); _,xh=ae(xb)
        loss=F.mse_loss(xh,xb)
        opt_ae.zero_grad(); loss.backward(); opt_ae.step()
    if (ep+1)%10==0: print(f"    ep {ep+1}")

# Initialise K_NEW=60 cluster centres
ae.eval()
with torch.no_grad():
    Z_AB=ae.encoder(Xt_AB.to(DEVICE)).cpu().numpy()
km_init=KMeans(n_clusters=K_NEW,n_init=10,random_state=SEED).fit(Z_AB)
centres=nn.Parameter(
    torch.tensor(km_init.cluster_centers_,dtype=torch.float32).to(DEVICE))
opt_dec=torch.optim.Adam(list(ae.encoder.parameters())+[centres],lr=1e-4)

# Phase B: joint DEC on combined A+B with K=60
print("  DEC joint training (30 epochs) …")
for ep in range(30):
    ae.train(); ep_loss=0.0
    for (xb,) in dl_AB:
        xb=xb.to(DEVICE); z=ae.encoder(xb)
        q=soft_assign(z,centres)
        with torch.no_grad(): p=target_dist(q)
        loss=F.kl_div(q.log(),p,reduction="batchmean")
        opt_dec.zero_grad(); loss.backward(); opt_dec.step()
        ep_loss+=loss.item()
    if (ep+1)%10==0: print(f"    ep {ep+1}  KL={ep_loss/len(dl_AB):.5f}")

# Get soft assignments for all A+B samples
ae.eval()
with torch.no_grad():
    Z_AB_dec=ae.encoder(Xt_AB.to(DEVICE)).cpu().numpy()
    Q_AB=soft_assign(
        torch.from_numpy(Z_AB_dec).to(DEVICE),centres).cpu().numpy()  # (N_A+N_B, 60)

# Hard assignments
hard_AB=Q_AB.argmax(1)
hard_B =hard_AB[N_A:]

# A-fraction per cluster → 10 lowest = novel
novel_ids_dec,a_frac_dec=a_fraction_novel_ids(hard_AB, K_NEW)

# Novelty score per B sample: 1 - A-fraction of its assigned cluster
novelty_score_dec = 1 - a_frac_dec[hard_B]
novel_mask_dec    = np.isin(hard_B, list(novel_ids_dec))

res_1b  = evaluate(novel_mask_dec, novelty_score_dec, "DEC(60) A-fraction")
cacc_1b = cluster_acc_novel(hard_B, novel_ids_dec, "1b")
all_results["DEC(60) A-fraction"] = (res_1b, cacc_1b)

# ══════════════════════════════════════════════════════════════════════════════
# METHOD 2 — HDBSCAN
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*68)
print("METHOD 2 — HDBSCAN on combined A+B")
print("="*68)
try:
    import hdbscan
    HAS_HDBSCAN=True
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable,"-m","pip","install","hdbscan",
                    "--break-system-packages","-q"])
    import hdbscan; HAS_HDBSCAN=True

clusterer=hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5,
                           metric="euclidean", prediction_data=True)
clusterer.fit(X_AB)
all_labels_hdb=clusterer.labels_          # -1 = noise
n_clusters_hdb=len(set(all_labels_hdb))-(-1 in all_labels_hdb)
print(f"  HDBSCAN found {n_clusters_hdb} clusters  "
      f"(noise: {(all_labels_hdb==-1).sum()} samples)")

B_labels_hdb = all_labels_hdb[N_A:]

# A-fraction per cluster
valid_clusters=[k for k in set(all_labels_hdb) if k!=-1]
a_frac_hdb={}
for k in valid_clusters:
    n_a=(all_labels_hdb[:N_A]==k).sum()
    n_b=(all_labels_hdb[N_A:]==k).sum()
    a_frac_hdb[k]=n_a/(n_a+n_b+1e-9)

# Novel clusters = valid clusters with lowest A-fraction (take up to K_NOV)
sorted_by_afrac=sorted(valid_clusters,key=lambda k:a_frac_hdb[k])
novel_ids_hdb=set(sorted_by_afrac[:K_NOV])

# Noise gets novelty_score=0.5; cluster members get 1-a_fraction
def hdb_novelty(b_label):
    if b_label==-1: return 0.5
    return 1-a_frac_hdb.get(b_label,0.5)
novelty_score_hdb=np.array([hdb_novelty(l) for l in B_labels_hdb])
novel_mask_hdb=np.array([l in novel_ids_hdb for l in B_labels_hdb])

res_2  = evaluate(novel_mask_hdb, novelty_score_hdb, "HDBSCAN A-fraction")
cacc_2 = cluster_acc_novel(B_labels_hdb, novel_ids_hdb, "2")
all_results["HDBSCAN"] = (res_2, cacc_2)

# ══════════════════════════════════════════════════════════════════════════════
# METHOD 3 — GMM(60) on combined A+B
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*68)
print("METHOD 3 — GMM(60) on combined A+B")
print("="*68)
print("  Fitting GMM(60) …")
gmm=GaussianMixture(n_components=K_NEW, covariance_type="diag",
                    max_iter=200, random_state=SEED, n_init=3)
gmm.fit(X_AB)

hard_AB_gmm = gmm.predict(X_AB)
hard_B_gmm  = hard_AB_gmm[N_A:]

novel_ids_gmm,a_frac_gmm=a_fraction_novel_ids(hard_AB_gmm, K_NEW)

# Novelty score: negative log-likelihood (lower = less likely under model)
log_lik_B     = gmm.score_samples(X_B2)
novelty_score_gmm = -log_lik_B                        # higher = more novel
novel_mask_gmm    = np.isin(hard_B_gmm, list(novel_ids_gmm))

res_3  = evaluate(novel_mask_gmm, novelty_score_gmm, "GMM(60) A-fraction")
cacc_3 = cluster_acc_novel(hard_B_gmm, novel_ids_gmm, "3")
all_results["GMM(60)"] = (res_3, cacc_3)

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*76)
print("FINAL SUMMARY  —  Joint Detection + Clustering on combined A+B UMAP")
print("="*76)
print(f"  {'Method':<28} {'AUC':>6} {'Recall':>8} {'Precision':>10} "
      f"{'F1':>6} {'Clust ACC':>10}")
print("  " + "-"*72)
order=["Gaussian (ref)","K-Means(60) match","DEC(60) A-fraction",
       "HDBSCAN","GMM(60)"]
for name in order:
    if name not in all_results: continue
    res,cacc=all_results[name]
    print(f"  {name:<28} {res['auc']:>6.4f} {res['rec']:>8.1%} "
          f"{res['prec']:>10.1%} {res['f1']:>6.3f} {cacc:>10.1%}")
