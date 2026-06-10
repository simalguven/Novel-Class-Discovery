"""
Novel Species Discovery Pipeline  —  with DEC backbone
=======================================================
Knowledge story
───────────────
  Dataset A  — old environment: 50 species · 100 imgs each · unlabelled
  Dataset B  — new environment: same 50 species (new photos) +
               10 brand-new species · 100 imgs each · unlabelled
  Known only: K_OLD=50, K_NEW=60

Pipeline
────────
  1. UMAP   fit on A only       768-d → 10-d
  2. DEC    trained on A only   10-d  →  6-d latent  (tightens old clusters)
  3. Encode B with frozen DEC encoder
  4. K-Means(50) on A's DEC features  →  50 old-species centroids
  5. Novelty detection on B  (three methods, run in both UMAP and DEC space)
       a) Per-cluster Gaussian  (mean + 2σ per centroid)
       b) Local Outlier Factor
       c) Isolation Forest
  6. K-Means(10) on flagged candidates  →  novel species
  7. Evaluate  (labels used ONLY here)
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from scipy.optimize import linear_sum_assignment
import umap
import warnings, time
warnings.filterwarnings("ignore")

SEED      = 42
K_OLD     = 50
K_NEW     = 60
K_NOV     = 10
N_PER_CLS = 100
rng       = np.random.default_rng(SEED)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Build balanced datasets
# ══════════════════════════════════════════════════════════════════════════════
print("Loading PlantNet embeddings …")
embeddings = np.load("plantnet_dinov2_raw.npy").astype(np.float32)
labels     = np.load("plantnet_labels.npy")

all_classes = np.unique(labels)
counts      = np.array([(labels == c).sum() for c in all_classes])
eligible    = all_classes[counts >= 2 * N_PER_CLS]
chosen_60   = np.sort(rng.choice(eligible, size=K_NEW, replace=False))
base_cls    = chosen_60[:K_OLD]
novel_cls   = chosen_60[K_OLD:]

print(f"\nBuilding balanced datasets ({N_PER_CLS} images/class) …")
X_A_list, X_B_list, y_A_list, y_B_list = [], [], [], []
for cls in base_cls:
    idx = rng.choice(np.where(labels == cls)[0], size=2*N_PER_CLS, replace=False)
    X_A_list.append(embeddings[idx[:N_PER_CLS]]);  y_A_list.extend([cls]*N_PER_CLS)
    X_B_list.append(embeddings[idx[N_PER_CLS:]]); y_B_list.extend([cls]*N_PER_CLS)
for cls in novel_cls:
    idx = rng.choice(np.where(labels == cls)[0], size=N_PER_CLS, replace=False)
    X_B_list.append(embeddings[idx]); y_B_list.extend([cls]*N_PER_CLS)

X_A  = np.vstack(X_A_list)   # (5000, 768)
X_B  = np.vstack(X_B_list)   # (6000, 768)
y_B  = np.array(y_B_list)    # eval only

id2idx     = {c: i for i, c in enumerate(chosen_60)}
y_B_eval   = np.array([id2idx[c] for c in y_B])
is_novel_B = y_B_eval >= K_OLD

print(f"  Dataset A : {len(X_A)} samples | {K_OLD} species")
print(f"  Dataset B : {len(X_B)} samples | {K_OLD} old + {K_NOV} novel  "
      f"({is_novel_B.sum()} novel samples)")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — UMAP: fit on A, project B
# ══════════════════════════════════════════════════════════════════════════════
print("\nUMAP: fit on A, project B (768-d → 10-d) …")
t0 = time.time()
reducer  = umap.UMAP(n_components=10, n_neighbors=20, min_dist=0.05,
                     metric="cosine", random_state=SEED, verbose=False)
X_A_umap = reducer.fit_transform(X_A)
X_B_umap = reducer.transform(X_B)
X_A_umap = normalize(X_A_umap, norm="l2").astype(np.float32)
X_B_umap = normalize(X_B_umap, norm="l2").astype(np.float32)
print(f"  Done in {time.time()-t0:.1f}s")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — DEC: trained on A only  →  6-d latent
# ══════════════════════════════════════════════════════════════════════════════
# Import torch here — after UMAP — to avoid MPS / numba conflict on macOS
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
torch.manual_seed(SEED)
DEVICE = torch.device("mps"  if torch.backends.mps.is_available() else
                       "cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {DEVICE}")

class Autoencoder(nn.Module):
    def __init__(self, in_dim=10, latent=6):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 32),     nn.ReLU(),
            nn.Linear(32, latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, 32), nn.ReLU(),
            nn.Linear(32, 64),     nn.ReLU(),
            nn.Linear(64, in_dim),
        )
    def forward(self, x):
        z = self.encoder(x); return z, self.decoder(z)

def soft_assign(z, centres, alpha=1.0):
    d = ((z.unsqueeze(1) - centres.unsqueeze(0))**2).sum(-1)
    q = (1 + d / alpha) ** (-(alpha+1)/2)
    return q / q.sum(1, keepdim=True)

def target_dist(q):
    p = (q**2) / q.sum(0, keepdim=True)
    return p / p.sum(1, keepdim=True)

Xt_A = torch.from_numpy(X_A_umap)
dl_A = DataLoader(TensorDataset(Xt_A), batch_size=256, shuffle=True)

# Phase A — autoencoder pre-training on A
print("\nDEC Phase 1 — autoencoder pre-training on A (40 epochs) …")
ae     = Autoencoder().to(DEVICE)
opt_ae = torch.optim.Adam(ae.parameters(), lr=1e-3)
for ep in range(40):
    ae.train()
    for (xb,) in dl_A:
        xb = xb.to(DEVICE); _, xh = ae(xb)
        loss = F.mse_loss(xh, xb)
        opt_ae.zero_grad(); loss.backward(); opt_ae.step()
    if (ep+1) % 10 == 0:
        print(f"  ep {ep+1:>3}")

# Initialise cluster centres from K-Means on A latent codes
ae.eval()
with torch.no_grad():
    Z_A_init = ae.encoder(Xt_A.to(DEVICE)).cpu().numpy()
km_init  = KMeans(n_clusters=K_OLD, n_init=10, random_state=SEED).fit(Z_A_init)
centres  = nn.Parameter(
    torch.tensor(km_init.cluster_centers_, dtype=torch.float32).to(DEVICE))
opt_dec  = torch.optim.Adam(
    list(ae.encoder.parameters()) + [centres], lr=1e-4)

# Phase B — joint DEC training on A
print("DEC Phase 2 — joint encoder + cluster optimisation on A (30 epochs) …")
for ep in range(30):
    ae.train(); ep_loss = 0.0
    for (xb,) in dl_A:
        xb = xb.to(DEVICE); z = ae.encoder(xb)
        q  = soft_assign(z, centres)
        with torch.no_grad(): p = target_dist(q)
        loss = F.kl_div(q.log(), p, reduction="batchmean")
        opt_dec.zero_grad(); loss.backward(); opt_dec.step()
        ep_loss += loss.item()
    if (ep+1) % 10 == 0:
        print(f"  ep {ep+1:>3}  KL={ep_loss/len(dl_A):.5f}")

# Encode A and B with the frozen DEC encoder
ae.eval()
with torch.no_grad():
    X_A_dec = ae.encoder(Xt_A.to(DEVICE)).cpu().numpy()
    X_B_dec = ae.encoder(
        torch.from_numpy(X_B_umap).to(DEVICE)).cpu().numpy()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — K-Means(50) on A in DEC space  →  old-species centroids
# ══════════════════════════════════════════════════════════════════════════════
print("\nClustering A (DEC space) into 50 groups …")
km_old_dec  = KMeans(n_clusters=K_OLD, n_init=10, random_state=SEED)
A_labels_dec = km_old_dec.fit_predict(X_A_dec)
cent_dec     = km_old_dec.cluster_centers_              # (50, 6)

# Also keep UMAP-space K-Means for baseline comparison
print("Clustering A (UMAP space) into 50 groups …")
km_old_umap   = KMeans(n_clusters=K_OLD, n_init=10, random_state=SEED)
A_labels_umap = km_old_umap.fit_predict(X_A_umap)
cent_umap     = km_old_umap.cluster_centers_            # (50, 10)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Novelty detection
# ══════════════════════════════════════════════════════════════════════════════

def min_centroid_dist(X_query, X_ref, centroids, ref_labels):
    dists = np.linalg.norm(
        X_query[:, None, :] - centroids[None, :, :], axis=-1)
    return dists.min(1), dists.argmin(1)

def gaussian_threshold(X_ref, ref_labels, centroids):
    thresh = np.zeros(len(centroids))
    for k in range(len(centroids)):
        d = np.linalg.norm(X_ref[ref_labels==k] - centroids[k], axis=1)
        thresh[k] = d.mean() + 2 * d.std()
    return thresh

def evaluate(novel_mask, label):
    tp   = is_novel_B[novel_mask].sum()
    fp   = (~is_novel_B[novel_mask]).sum()
    rec  = tp / is_novel_B.sum()
    prec = tp / novel_mask.sum() if novel_mask.sum() > 0 else 0
    f1   = 2*prec*rec / (prec+rec+1e-9)
    fpr  = fp / (~is_novel_B).sum()
    print(f"  {label:<35} flagged={novel_mask.sum():>4}  "
          f"recall={rec:.1%}  prec={prec:.1%}  F1={f1:.3f}  FPR={fpr:.1%}")
    return {"recall": rec, "prec": prec, "f1": f1, "fpr": fpr,
            "flagged": novel_mask.sum()}

def cluster_novel_acc(novel_mask):
    X_cand   = X_B_dec[novel_mask]
    truly    = is_novel_B[novel_mask]
    y_true   = y_B_eval[novel_mask][truly] - K_OLD
    if len(y_true) < K_NOV:
        return 0.0, len(y_true)
    km  = KMeans(n_clusters=K_NOV, n_init=10, random_state=SEED)
    pred = km.fit_predict(X_cand)[truly]
    n   = max(y_true.max(), pred.max()) + 1
    mat = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, pred): mat[t, p] += 1
    r, c = linear_sum_assignment(-mat)
    return mat[r, c].sum() / len(y_true), len(y_true)

print("\n" + "="*72)
print("NOVELTY DETECTION RESULTS")
print("="*72)
all_results = {}

for space_name, X_A_sp, X_B_sp, cent_sp, A_lbl_sp in [
    ("UMAP-10",  X_A_umap, X_B_umap, cent_umap, A_labels_umap),
    ("DEC-6",    X_A_dec,  X_B_dec,  cent_dec,  A_labels_dec),
]:
    print(f"\n── Space: {space_name} " + "─"*50)
    min_d_B, near_B = min_centroid_dist(X_B_sp, X_A_sp, cent_sp, A_lbl_sp)
    min_d_A, near_A = min_centroid_dist(X_A_sp, X_A_sp, cent_sp, A_lbl_sp)

    # a) Gaussian
    thresh = gaussian_threshold(X_A_sp, A_lbl_sp, cent_sp)
    mask_g = min_d_B > thresh[near_B]
    res_g  = evaluate(mask_g, f"Gaussian (mean+2σ)")
    acc_g, n_g = cluster_novel_acc(mask_g)
    print(f"    → novel cluster ACC = {acc_g:.1%}  ({n_g} novel samples in candidates)")

    # b) LOF
    lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
    lof.fit(X_A_sp)
    sc_B  = -lof.score_samples(X_B_sp)
    sc_A  = -lof.score_samples(X_A_sp)
    mask_l = sc_B > np.percentile(sc_A, 95)
    res_l  = evaluate(mask_l, f"LOF (95th pct of A scores)")
    acc_l, n_l = cluster_novel_acc(mask_l)
    print(f"    → novel cluster ACC = {acc_l:.1%}  ({n_l} novel samples in candidates)")

    # c) Isolation Forest
    iso = IsolationForest(n_estimators=200, contamination=0.05,
                          random_state=SEED, n_jobs=-1)
    iso.fit(X_A_sp)
    iso_B  = -iso.score_samples(X_B_sp)
    iso_A  = -iso.score_samples(X_A_sp)
    mask_i = iso_B > np.percentile(iso_A, 95)
    res_i  = evaluate(mask_i, f"Isolation Forest (95th pct)")
    acc_i, n_i = cluster_novel_acc(mask_i)
    print(f"    → novel cluster ACC = {acc_i:.1%}  ({n_i} novel samples in candidates)")

    # AUC
    auc_g = roc_auc_score(is_novel_B, min_d_B)
    auc_l = roc_auc_score(is_novel_B, sc_B)
    auc_i = roc_auc_score(is_novel_B, iso_B)
    print(f"\n  AUC  →  Gaussian={auc_g:.4f}  LOF={auc_l:.4f}  IsoForest={auc_i:.4f}")

    all_results[space_name] = {
        "Gaussian":  (res_g, acc_g, auc_g),
        "LOF":       (res_l, acc_l, auc_l),
        "IsoForest": (res_i, acc_i, auc_i),
    }

# ── Baseline
print("\n── Baseline: naive K-Means(60) on Dataset B (DEC space) ──────────────")
km_naive   = KMeans(n_clusters=K_NEW, n_init=10, random_state=SEED)
naive_pred = km_naive.fit_predict(X_B_dec)
n2  = max(y_B_eval.max(), naive_pred.max()) + 1
mat = np.zeros((n2, n2), dtype=np.int64)
for t, p in zip(y_B_eval, naive_pred): mat[t, p] += 1
r, c = linear_sum_assignment(-mat)
naive_overall = mat[r, c].sum() / len(y_B_eval)
nmat = np.zeros((n2, n2), dtype=np.int64)
for t, p in zip(y_B_eval[is_novel_B], naive_pred[is_novel_B]): nmat[t, p] += 1
rn, cn = linear_sum_assignment(-nmat)
naive_novel = nmat[rn, cn].sum() / is_novel_B.sum()
print(f"  Overall ACC={naive_overall:.1%}  Novel ACC={naive_novel:.1%}")

# ── Final summary table
print("\n" + "="*72)
print("FINAL SUMMARY  (UMAP-10 vs DEC-6 space)")
print("="*72)
print(f"  {'Method':<40} {'Space':<8} {'AUC':>6} {'Recall':>7} {'F1':>6} {'Clust ACC':>10}")
print("  " + "-"*70)
for space in ["UMAP-10", "DEC-6"]:
    for mname in ["Gaussian", "LOF", "IsoForest"]:
        res, acc, auc = all_results[space][mname]
        print(f"  {mname:<40} {space:<8} {auc:>6.4f} "
              f"{res['recall']:>7.1%} {res['f1']:>6.3f} {acc:>10.1%}")
print(f"\n  {'Baseline K-Means(60) on B (DEC space)':<40} {'':8} {'':>6} "
      f"{'':>7} {'':>6} {naive_novel:>10.1%}")
