"""
Confidence-Weighted Supervised Contrastive Learning
====================================================

The DINOv2 embeddings are never modified.
A small projection head is trained on top of them using pseudo-labels
from K-Means(50) on Dataset A.

Robustness mechanisms
─────────────────────
  1. Frozen backbone   — raw 768-d embeddings are read-only numpy arrays;
                         only the projection head has gradients
  2. Three-tier system — C_low samples never enter positive pairs;
                         wrong pseudo-labels on uncertain samples cannot
                         corrupt the representation
  3. Pair weighting    — w(i,j) = conf_i × conf_j; noisy pairs contribute
                         quadratically less than clean ones
  4. Empty-P guard     — anchors with no valid positives in the batch are
                         skipped entirely rather than producing NaN loss
  5. Numerical safety  — log-sum-exp with max subtraction; no raw exp()
  6. Gradient clipping — prevents explosion in early epochs
  7. No augmentation   — pseudo-labels define positives; embedding-space
                         noise would add no semantic meaning

After training, the projection head maps A and B to a 128-d unit sphere
where old species are more separated and novel species land in larger gaps.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score
from scipy.optimize import linear_sum_assignment
import umap
import warnings, time
warnings.filterwarnings("ignore")

SEED = 42; K_OLD = 50; K_NEW = 60; K_NOV = 10; N_PER_CLS = 100
rng  = np.random.default_rng(SEED)
torch.manual_seed(SEED)

# ── Data ──────────────────────────────────────────────────────────────────────
print("Loading data …")
embeddings = np.load("plantnet_dinov2_raw.npy").astype(np.float32)
labels     = np.load("plantnet_labels.npy")

all_classes = np.unique(labels)
counts      = np.array([(labels == c).sum() for c in all_classes])
eligible    = all_classes[counts >= 2 * N_PER_CLS]
chosen_60   = np.sort(rng.choice(eligible, size=K_NEW, replace=False))
base_cls    = chosen_60[:K_OLD]
novel_cls   = chosen_60[K_OLD:]

XA, XB, yBl = [], [], []
for c in base_cls:
    idx = rng.choice(np.where(labels == c)[0], size=2*N_PER_CLS, replace=False)
    XA.append(embeddings[idx[:N_PER_CLS]])
    XB.append(embeddings[idx[N_PER_CLS:]]); yBl.extend([c]*N_PER_CLS)
for c in novel_cls:
    idx = rng.choice(np.where(labels == c)[0], size=N_PER_CLS, replace=False)
    XB.append(embeddings[idx]); yBl.extend([c]*N_PER_CLS)

X_A = np.vstack(XA)                          # (5000, 768) raw DINOv2 — never modified
X_B = np.vstack(XB)                          # (6000, 768)
y_B = np.array(yBl)
id2idx     = {c: i for i, c in enumerate(chosen_60)}
y_B_eval   = np.array([id2idx[c] for c in y_B])
is_novel_B = y_B_eval >= K_OLD
N_A, N_B   = len(X_A), len(X_B)
print(f"  A:{N_A}  B:{N_B}  novel:{is_novel_B.sum()}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Pseudo-labels and confidence scores from UMAP-10 of A
#          (labels are for pseudo-label generation only, not for training)
# ══════════════════════════════════════════════════════════════════════════════
print("\nUMAP on A (for pseudo-label generation) …")
reducer_A = umap.UMAP(n_components=10, n_neighbors=20, min_dist=0.05,
                      metric="cosine", random_state=SEED, verbose=False)
X_A_umap  = normalize(reducer_A.fit_transform(X_A), norm="l2").astype(np.float32)

print("K-Means(50) on A → pseudo-labels …")
km50      = KMeans(n_clusters=K_OLD, n_init=15, random_state=SEED).fit(X_A_umap)
pseudo_y  = km50.labels_                           # (5000,) pseudo-labels 0-49
cent50    = km50.cluster_centers_                  # (50, 10)

# Distance-based confidence: smaller distance to centroid = more confident
dist_A = np.linalg.norm(X_A_umap[:,None,:] - cent50[None,:,:], axis=-1)  # (5000,50)
own_dist   = dist_A[np.arange(N_A), pseudo_y]      # distance to own centroid

# Soft assignment margin: max_prob - 2nd_max_prob
T = 0.1
soft_A = np.exp(-dist_A / T); soft_A /= soft_A.sum(1, keepdims=True)
sorted_soft = np.sort(soft_A, axis=1)
margin  = sorted_soft[:, -1] - sorted_soft[:, -2]

# Normalise both to [0,1]: higher = more confident
conf_density = 1 - (own_dist  - own_dist.min())  / (own_dist.max()  - own_dist.min()  + 1e-9)
conf_margin  =     (margin     - margin.min())    / (margin.max()    - margin.min()    + 1e-9)
confidence   = (conf_density + conf_margin) / 2   # ensemble of both signals

# Three-tier split by confidence percentile
p30 = np.percentile(confidence, 30)
p70 = np.percentile(confidence, 70)
tier = np.where(confidence >= p70, 2,             # C_high
        np.where(confidence >= p30, 1, 0))         # C_mid / C_low

print(f"  C_high: {(tier==2).sum()}  C_mid: {(tier==1).sum()}  C_low: {(tier==0).sum()}")
print(f"  Mean confidence — C_high:{confidence[tier==2].mean():.3f}  "
      f"C_mid:{confidence[tier==1].mean():.3f}  "
      f"C_low:{confidence[tier==0].mean():.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Projection head  (the only trainable component)
# ══════════════════════════════════════════════════════════════════════════════
# Import torch after all UMAP work to avoid MPS/numba conflict
DEVICE = torch.device("mps"  if torch.backends.mps.is_available() else
                       "cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {DEVICE}")

class ProjectionHead(nn.Module):
    """
    Maps frozen 768-d DINOv2 features to a 128-d unit sphere.
    Two-layer MLP with BatchNorm — standard SupCon architecture.
    The input embeddings are NEVER part of this module's parameters.
    """
    def __init__(self, in_dim=768, hidden=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)   # unit sphere output

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Confidence-weighted SupCon loss
# ══════════════════════════════════════════════════════════════════════════════

def supcon_loss(z, pseudo_labels, confidence, tier, temperature=0.1):
    """
    z            : (N, D) L2-normalised projections  — all samples in the batch
    pseudo_labels: (N,)   pseudo-class per sample
    confidence   : (N,)   confidence score in [0,1]
    tier         : (N,)   0=C_low, 1=C_mid, 2=C_high
    temperature  : scalar

    Only C_high and C_mid samples serve as anchors.
    Only C_high and C_mid samples can be in the positive set P(i).
    C_low samples appear only in the denominator as negatives.

    Returns the mean loss over all valid anchors.
    Empty-P anchors are skipped — they contribute 0 to the mean.
    """
    N = z.size(0)

    # Pairwise cosine similarities (z is already L2-normalised)
    sim = (z @ z.T) / temperature                 # (N, N)

    # Numerical stability: subtract row-wise max before exp
    sim_max, _ = sim.max(dim=1, keepdim=True)
    exp_sim    = torch.exp(sim - sim_max)          # (N, N)

    # Self-similarity mask: exclude diagonal
    self_mask  = torch.eye(N, dtype=torch.bool, device=z.device)
    exp_sim    = exp_sim.masked_fill(self_mask, 0.0)

    # Denominator: sum over all non-self samples (including C_low negatives)
    denom      = exp_sim.sum(dim=1, keepdim=True) + 1e-8   # (N, 1)

    # Build positive mask: same pseudo-label, both in C_high or C_mid
    pl         = pseudo_labels.unsqueeze(1)                 # (N, 1)
    same_class = (pl == pl.T)                               # (N, N)
    valid_pair = (tier >= 1).unsqueeze(1) & (tier >= 1).unsqueeze(0)
    pos_mask   = same_class & valid_pair & ~self_mask       # (N, N)

    # Pair weights: conf_i × conf_j
    w          = confidence.unsqueeze(1) * confidence.unsqueeze(0)  # (N, N)
    w          = w * pos_mask.float()                       # zero out non-positives

    # Weighted log-probabilities for each positive pair
    log_prob   = (sim - sim_max) - torch.log(denom)        # (N, N) log P(j|i)
    # Sum weighted log-probs per anchor
    weighted_logp = (w * log_prob).sum(dim=1)              # (N,)
    w_sum         = w.sum(dim=1).clamp(min=1e-8)           # (N,)

    # Only compute loss for C_high and C_mid anchors that have ≥1 positive
    anchor_mask = (tier >= 1) & (pos_mask.sum(dim=1) > 0)

    if anchor_mask.sum() == 0:
        return torch.tensor(0.0, device=z.device, requires_grad=True)

    per_anchor_loss = -(weighted_logp / w_sum)[anchor_mask]
    return per_anchor_loss.mean()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Training loop
# ══════════════════════════════════════════════════════════════════════════════
EPOCHS      = 150
BATCH_SIZE  = 512
LR          = 3e-4
TEMPERATURE = 0.1
CLIP_GRAD   = 1.0

head      = ProjectionHead().to(DEVICE)
optimiser = torch.optim.Adam(head.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)

# Tensors for all A samples (frozen — no grad)
Xt_A    = torch.from_numpy(X_A)                        # raw 768-d
conf_t  = torch.from_numpy(confidence.astype(np.float32))
tier_t  = torch.from_numpy(tier.astype(np.int64))
label_t = torch.from_numpy(pseudo_y.astype(np.int64))

dataset = TensorDataset(Xt_A, label_t, conf_t, tier_t)

def balanced_batch_sampler(dataset_size, pseudo_labels, tier,
                            n_per_class, n_low_neg, rng):
    """
    Sample n_per_class samples from each C_high/C_mid pseudo-class
    plus n_low_neg random C_low samples for hard negatives.
    Returns indices for one batch.
    """
    idx = []
    for k in range(K_OLD):
        pool = np.where((pseudo_labels == k) & (tier >= 1))[0]
        if len(pool) == 0: continue
        chosen = rng.choice(pool, size=min(n_per_class, len(pool)), replace=False)
        idx.extend(chosen.tolist())
    low_pool = np.where(tier == 0)[0]
    if len(low_pool) > 0:
        idx.extend(rng.choice(low_pool,
                              size=min(n_low_neg, len(low_pool)),
                              replace=False).tolist())
    return np.array(idx)

N_PER_CLASS_BATCH = 6    # ~6 samples per class → 300 from C_high/C_mid
N_LOW_NEG_BATCH   = 100  # 100 C_low hard negatives

print(f"\nTraining projection head ({EPOCHS} epochs) …")
print(f"  Batch: ~{N_PER_CLASS_BATCH*K_OLD} C_high/C_mid  +  {N_LOW_NEG_BATCH} C_low negatives")

train_losses = []
for ep in range(EPOCHS):
    head.train()
    # Fresh balanced batch each epoch
    batch_idx = balanced_batch_sampler(
        N_A, pseudo_y, tier, N_PER_CLASS_BATCH, N_LOW_NEG_BATCH, rng)
    rng.shuffle(batch_idx)

    ep_loss = 0.0; n_batches = 0
    for start in range(0, len(batch_idx), BATCH_SIZE):
        b_idx = batch_idx[start:start + BATCH_SIZE]
        if len(b_idx) < 4: continue

        x_b   = Xt_A[b_idx].to(DEVICE)
        pl_b  = label_t[b_idx].to(DEVICE)
        cf_b  = conf_t[b_idx].to(DEVICE)
        ti_b  = tier_t[b_idx].to(DEVICE)

        z     = head(x_b)
        loss  = supcon_loss(z, pl_b, cf_b, ti_b, TEMPERATURE)

        optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), CLIP_GRAD)
        optimiser.step()

        ep_loss += loss.item(); n_batches += 1

    scheduler.step()
    avg = ep_loss / max(n_batches, 1)
    train_losses.append(avg)
    if (ep + 1) % 30 == 0:
        print(f"  ep {ep+1:>3}  loss={avg:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Extract projected features for A and B
#          The original embeddings are untouched — only the head was trained
# ══════════════════════════════════════════════════════════════════════════════
print("\nExtracting projected features …")
head.eval()
with torch.no_grad():
    Z_A = head(torch.from_numpy(X_A).to(DEVICE)).cpu().numpy()   # (5000, 128)
    Z_B = head(torch.from_numpy(X_B).to(DEVICE)).cpu().numpy()   # (6000, 128)

# Verify separation improved: within-class vs between-class distance
print("\nRepresentation quality check:")
for tag, Z, pseudo in [("Raw 768-d", normalize(X_A, norm="l2"), pseudo_y),
                        ("Projected 128-d", Z_A, pseudo_y)]:
    intra, inter = [], []
    for k in range(0, K_OLD, 5):          # sample 10 classes for speed
        mem = Z[pseudo == k]
        if len(mem) < 2: continue
        # intra: mean pairwise distance within cluster
        d = np.linalg.norm(mem[:,None,:] - mem[None,:,:], axis=-1)
        intra.append(d[np.triu_indices(len(mem), k=1)].mean())
        # inter: distance from centroid to other centroids
        c_k = mem.mean(0)
        others = Z[pseudo != k].reshape(-1, Z.shape[1])
        inter.append(np.linalg.norm(others - c_k, axis=1).mean())
    print(f"  {tag:<20}  intra={np.mean(intra):.4f}  inter={np.mean(inter):.4f}  "
          f"ratio={np.mean(inter)/np.mean(intra):.2f}x")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Run the full novel discovery pipeline on projected features
# ══════════════════════════════════════════════════════════════════════════════
print("\nCombined UMAP on projected A+B (128-d → 10-d) …")
reducer_AB = umap.UMAP(n_components=10, n_neighbors=20, min_dist=0.05,
                        metric="cosine", random_state=SEED, verbose=False)
X_AB_proj  = normalize(
    reducer_AB.fit_transform(np.vstack([Z_A, Z_B])), norm="l2").astype(np.float32)
X_A_proj   = X_AB_proj[:N_A]
X_B_proj   = X_AB_proj[N_A:]

# K-Means(50) on A for Hungarian reference centroids
km50_proj  = KMeans(n_clusters=K_OLD, n_init=15, random_state=SEED).fit(X_A_proj)
cent50_proj= km50_proj.cluster_centers_

def run_pipeline(X_AB, X_A_sp, X_B_sp, cent_A, name):
    """Soft K-Means(60) + Hungarian match + evaluate."""
    from sklearn.cluster import KMeans as KM

    # Shared K-Means++ init
    km_init = KM(n_clusters=K_NEW, init="k-means++", n_init=1,
                 random_state=SEED, max_iter=1).fit(X_AB)
    centroids = km_init.cluster_centers_.copy()

    # Soft must-link on A (λ=0.1, confidence from pseudo-labels)
    scale = np.linalg.norm(
        X_A_sp[:,None,:] - cent_A[None,:,:], axis=-1).min(1).mean()

    km_pseudo   = KM(n_clusters=K_OLD, n_init=15, random_state=SEED).fit(X_A_sp)
    A_labels_sp = km_pseudo.labels_
    dist_sp     = np.linalg.norm(X_A_sp[:,None,:] - cent_A[None,:,:], axis=-1)
    own_d       = dist_sp[np.arange(N_A), A_labels_sp]
    conf_sp     = 1 - (own_d - own_d.min()) / (own_d.max() - own_d.min() + 1e-9)

    LAM = 0.1
    penalty_lookup = {
        i: (int(A_labels_sp[i]), float(conf_sp[i]) * scale * LAM)
        for i in range(N_A)
    }

    assign = np.zeros(len(X_AB), dtype=np.int32)
    for _ in range(150):
        dists = np.linalg.norm(X_AB[:,None,:] - centroids[None,:,:], axis=-1).copy()
        for pi, (pl, pen) in penalty_lookup.items():
            dists[pi, :] += pen; dists[pi, pl] -= pen
        assign = dists.argmin(1)
        new_c  = np.array([
            X_AB[assign==k].mean(0) if (assign==k).any() else centroids[k]
            for k in range(K_NEW)])
        if np.linalg.norm(new_c - centroids) < 1e-5: break
        centroids = new_c

    # Hungarian match
    cost   = np.linalg.norm(centroids[:,None,:] - cent_A[None,:,:], axis=-1)
    INF    = cost.max() * 10
    cost_sq= np.full((K_NEW, K_NEW), INF); cost_sq[:K_OLD,:] = cost.T
    _, col = linear_sum_assignment(cost_sq)
    matched= set(col[:K_OLD]); novel_ids = set(range(K_NEW)) - matched
    match_cost = np.full(K_NEW, cost.max()*2)
    for ai, ci in zip(range(K_OLD), col[:K_OLD]):
        match_cost[ci] = cost_sq[ai, ci]

    B_labels      = assign[N_A:]
    novelty_score = match_cost[B_labels]
    novel_mask    = np.isin(B_labels, list(novel_ids))

    tp   = is_novel_B[novel_mask].sum()
    prec = tp / novel_mask.sum() if novel_mask.sum() > 0 else 0
    rec  = tp / is_novel_B.sum()
    f1   = 2*prec*rec/(prec+rec+1e-9)
    auc  = roc_auc_score(is_novel_B, novelty_score)
    fpr  = (~is_novel_B[novel_mask]).sum() / (~is_novel_B).sum()

    # Cluster accuracy
    cand_truly = is_novel_B[novel_mask]
    y_true_c   = y_B_eval[novel_mask][cand_truly] - K_OLD
    y_pred_c   = B_labels[novel_mask][cand_truly]
    cacc = 0.0
    if len(y_true_c) >= K_NOV:
        uniq = {v:i for i,v in enumerate(np.unique(y_pred_c))}
        yp   = np.array([uniq[v] for v in y_pred_c])
        n    = max(y_true_c.max(), yp.max())+1
        mat  = np.zeros((n,n),dtype=np.int64)
        for t,p in zip(y_true_c,yp): mat[t,p]+=1
        r,c  = linear_sum_assignment(-mat)
        cacc = mat[r,c].sum()/len(y_true_c)

    print(f"\n  [{name}]")
    print(f"    AUC={auc:.4f}  Recall={rec:.1%}  Precision={prec:.1%}  "
          f"F1={f1:.3f}  FPR={fpr:.1%}  Cluster ACC={cacc:.1%}")
    return dict(auc=auc, rec=rec, prec=prec, f1=f1, cacc=cacc)

# Compare baseline (raw UMAP) vs SupCon pipeline
print("\n" + "="*68)
print("RESULTS COMPARISON")
print("="*68)

# Baseline: combined UMAP on raw features (replicate from semisup_kmeans)
print("\nBaseline combined UMAP on raw 768-d …")
reducer_raw  = umap.UMAP(n_components=10, n_neighbors=20, min_dist=0.05,
                          metric="cosine", random_state=SEED, verbose=False)
X_AB_raw     = normalize(
    reducer_raw.fit_transform(np.vstack([X_A, X_B])), norm="l2").astype(np.float32)
X_A_raw      = X_AB_raw[:N_A]; X_B_raw = X_AB_raw[N_A:]
km_raw       = KMeans(n_clusters=K_OLD, n_init=15, random_state=SEED).fit(X_A_raw)
res_base     = run_pipeline(X_AB_raw, X_A_raw, X_B_raw, km_raw.cluster_centers_,
                             "Baseline  — raw 768-d → UMAP-10")

# SupCon pipeline
res_supcon   = run_pipeline(X_AB_proj, X_A_proj, X_B_proj, cent50_proj,
                             "SupCon    — 768-d → 128-d → UMAP-10")

print("\n" + "="*68)
print("SUMMARY")
print("="*68)
print(f"  {'Method':<35} {'AUC':>6} {'Recall':>8} {'Prec':>8} "
      f"{'F1':>6} {'ClustACC':>10}")
print("  " + "-"*66)
for name, r in [("Baseline raw→UMAP", res_base),
                 ("SupCon→UMAP",       res_supcon)]:
    mark = " ◄" if r["cacc"] == max(res_base["cacc"], res_supcon["cacc"]) else ""
    print(f"  {name:<35} {r['auc']:>6.4f} {r['rec']:>8.1%} {r['prec']:>8.1%} "
          f"{r['f1']:>6.3f} {r['cacc']:>10.1%}{mark}")
