"""
Fix 1 — Semi-supervised SupCon: pseudo-labels for A, k-NN positives for B
==========================================================================

Problem with SupCon trained on A only:
  Novel species in B visually resemble old ones. The projection head maps
  them toward the nearest old pseudo-class, hurting novel ACC.

Fix:
  Include B samples in the same training batch.
  · A samples  → confidence-weighted SupCon with pseudo-labels  (as before)
  · B samples  → k-NN positives from the raw DINOv2 space       (self-supervised)
  The denominator of every anchor's loss includes ALL samples in the batch
  (A and B), so A and B repel each other unless explicitly paired as positives.
  Novel B samples form clusters via their mutual k-NN links without being
  forced toward any old pseudo-class centroid.

Batch construction for B:
  Pair-sampling: for each B seed, add one of its k-NN to the batch.
  This guarantees every B anchor has at least one positive in the batch.

Loss:
  Unified loss over the whole batch. Positive mask is:
    A-A pairs : same pseudo-class, both in C_high ∪ C_mid, weight = conf_i × conf_j
    B-B pairs : j is a pre-computed k-NN of i,               weight = 1.0
    A-B pairs : no positives                                  weight = 0.0

GCD evaluation (All / Old / Novel ACC) at the end.
"""

import numpy as np, umap, warnings, time
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from scipy.optimize import linear_sum_assignment
warnings.filterwarnings("ignore")

SEED = 42; K_OLD = 50; K_NEW = 60; K_NOV = 10; N_PER_CLS = 100
rng  = np.random.default_rng(SEED); torch.manual_seed(SEED)

# ── Data ──────────────────────────────────────────────────────────────────────
print("Loading data …")
embeddings  = np.load("plantnet_dinov2_raw.npy").astype(np.float32)
labels      = np.load("plantnet_labels.npy")
all_classes = np.unique(labels)
counts      = np.array([(labels == c).sum() for c in all_classes])
eligible    = all_classes[counts >= 2 * N_PER_CLS]
chosen_60   = np.sort(rng.choice(eligible, size=K_NEW, replace=False))
base_cls    = chosen_60[:K_OLD]; novel_cls = chosen_60[K_OLD:]

XA, XB, yBl = [], [], []
for c in base_cls:
    idx = rng.choice(np.where(labels == c)[0], size=2*N_PER_CLS, replace=False)
    XA.append(embeddings[idx[:N_PER_CLS]])
    XB.append(embeddings[idx[N_PER_CLS:]]); yBl.extend([c]*N_PER_CLS)
for c in novel_cls:
    idx = rng.choice(np.where(labels == c)[0], size=N_PER_CLS, replace=False)
    XB.append(embeddings[idx]); yBl.extend([c]*N_PER_CLS)

X_A = np.vstack(XA); X_B = np.vstack(XB); y_B = np.array(yBl)
id2idx     = {c: i for i, c in enumerate(chosen_60)}
y_B_eval   = np.array([id2idx[c] for c in y_B])
is_novel_B = y_B_eval >= K_OLD
N_A, N_B   = len(X_A), len(X_B)
print(f"  A:{N_A}  B:{N_B}  novel:{is_novel_B.sum()}")

# ── GCD evaluation (All / Old / Novel ACC) ────────────────────────────────────
def gcd_acc(features_B, tag, n_init=20):
    km    = KMeans(n_clusters=K_NEW, n_init=n_init, random_state=SEED)
    preds = km.fit_predict(normalize(features_B, norm="l2"))
    mat   = np.zeros((K_NEW, K_NEW), dtype=np.int64)
    for t, p in zip(y_B_eval, preds): mat[t, p] += 1
    row, col = linear_sum_assignment(-mat)
    p2t      = {c: r for r, c in zip(row, col)}
    pm       = np.array([p2t.get(p, -1) for p in preds])
    all_a    = (pm == y_B_eval).mean()
    old_a    = (pm[~is_novel_B] == y_B_eval[~is_novel_B]).mean()
    nov_a    = (pm[is_novel_B]  == y_B_eval[is_novel_B]).mean()
    print(f"  {tag:<45}  All={all_a:.1%}  Old={old_a:.1%}  Novel={nov_a:.1%}")
    return all_a, old_a, nov_a

# ── Shared UMAP-10 for pseudo-label generation ────────────────────────────────
print("\nCombined UMAP-10 for pseudo-labels and baseline …")
t0 = time.time()
r_umap   = umap.UMAP(n_components=10, n_neighbors=20, min_dist=0.05,
                      metric="cosine", random_state=SEED, verbose=False)
X_AB_umap = normalize(
    r_umap.fit_transform(np.vstack([X_A, X_B])), norm="l2").astype(np.float32)
X_A_umap  = X_AB_umap[:N_A]; X_B_umap = X_AB_umap[N_A:]
print(f"  Done in {time.time()-t0:.1f}s")

# Baseline result
print("\nBaseline:")
res_baseline = gcd_acc(X_B_umap, "Combined UMAP-10 (raw, baseline)")

# Pseudo-labels + confidence for A
km50      = KMeans(n_clusters=K_OLD, n_init=15, random_state=SEED).fit(X_A_umap)
pseudo_y  = km50.labels_; cent50 = km50.cluster_centers_
dist_A    = np.linalg.norm(X_A_umap[:,None,:] - cent50[None,:,:], axis=-1)
own_dist  = dist_A[np.arange(N_A), pseudo_y]
T = 0.1
soft_A    = np.exp(-dist_A/T); soft_A /= soft_A.sum(1, keepdims=True)
ss        = np.sort(soft_A, axis=1)
margin    = ss[:,-1] - ss[:,-2]
conf_d    = 1-(own_dist-own_dist.min())/(own_dist.max()-own_dist.min()+1e-9)
conf_m    = (margin-margin.min())/(margin.max()-margin.min()+1e-9)
conf      = (conf_d + conf_m) / 2
p30, p70  = np.percentile(conf, 30), np.percentile(conf, 70)
tier      = np.where(conf>=p70, 2, np.where(conf>=p30, 1, 0))
print(f"\n  Pseudo-labels ready. "
      f"C_high:{(tier==2).sum()}  C_mid:{(tier==1).sum()}  C_low:{(tier==0).sum()}")

# ── k-NN graph for B (fixed, from raw DINOv2 features) ───────────────────────
K_POS = 10    # number of nearest neighbours used as positives for each B sample
print(f"\nPre-computing {K_POS}-NN graph for B samples (raw DINOv2) …")
t0   = time.time()
nbrs = NearestNeighbors(n_neighbors=K_POS+1, metric="cosine", n_jobs=-1)
nbrs.fit(X_B)
knn_B = nbrs.kneighbors(X_B, return_distance=False)[:, 1:]  # (N_B, K_POS)
print(f"  Done in {time.time()-t0:.1f}s")

# ── Model ─────────────────────────────────────────────────────────────────────
DEVICE = torch.device("mps"  if torch.backends.mps.is_available() else
                       "cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {DEVICE}")

class ProjectionHead(nn.Module):
    def __init__(self, in_dim=768, hidden=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim))
    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)

# ── Unified semi-supervised SupCon loss ───────────────────────────────────────
def semi_supcon_loss(
        z,                   # (N, D) all projections in batch
        is_A,                # (N,)  bool — which samples come from A
        pseudo_lbl,          # (N,)  pseudo-label (valid only where is_A)
        conf,                # (N,)  confidence (valid only where is_A)
        tier,                # (N,)  0/1/2 (valid only where is_A)
        b_pos_mask,          # (N, N) bool — k-NN positive pairs among B samples
        tau=0.1):
    """
    Positive mask:
      A-A: same pseudo-class, both tier>=1, weight = conf_i * conf_j
      B-B: k-NN neighbours (b_pos_mask),   weight = 1.0
      A-B: never positive
    """
    N = z.size(0)
    eye = torch.eye(N, dtype=torch.bool, device=z.device)

    sim      = (z @ z.T) / tau
    mx, _    = sim.max(1, keepdim=True)
    exp_sim  = torch.exp(sim - mx).masked_fill(eye, 0.0)
    denom    = exp_sim.sum(1, keepdim=True) + 1e-8
    log_prob = (sim - mx) - torch.log(denom)        # (N, N)

    # ── A-A positive mask (pseudo-label) ─────────────────────────────────────
    is_A_t  = is_A.to(z.device)
    pl_t    = pseudo_lbl.to(z.device)
    cf_t    = conf.to(z.device)
    ti_t    = tier.to(z.device)

    aa_same  = (pl_t.unsqueeze(1) == pl_t.unsqueeze(0))
    aa_valid = (ti_t>=1).unsqueeze(1) & (ti_t>=1).unsqueeze(0)
    aa_both  = is_A_t.unsqueeze(1) & is_A_t.unsqueeze(0)
    aa_mask  = aa_same & aa_valid & aa_both & ~eye     # (N,N)

    aa_w     = cf_t.unsqueeze(1) * cf_t.unsqueeze(0)   # (N,N)
    aa_w     = aa_w * aa_mask.float()

    # ── B-B positive mask (k-NN) ──────────────────────────────────────────────
    bb_mask  = b_pos_mask.to(z.device)                  # (N,N), already bool
    bb_w     = bb_mask.float()                           # uniform weight

    # ── Combined ─────────────────────────────────────────────────────────────
    pos_mask = aa_mask | bb_mask                         # union
    w        = aa_w + bb_w                               # sum weights (no overlap)

    wlp      = (w * log_prob).sum(1)                    # (N,)
    ws       = w.sum(1).clamp(1e-8)

    # Valid anchors: A (tier>=1, has A-A positives) OR B (has B-B positives)
    has_pos  = pos_mask.sum(1) > 0
    a_anchor = is_A_t & (ti_t >= 1)
    b_anchor = ~is_A_t
    anchor   = has_pos & (a_anchor | b_anchor)

    if not anchor.any():
        return torch.tensor(0., device=z.device, requires_grad=True)

    return -(wlp / ws)[anchor].mean()

# ── Batch sampling ────────────────────────────────────────────────────────────
N_A_PER_CLASS = 6    # A samples per pseudo-class
N_A_LOW       = 50   # C_low A samples as hard negatives
N_B_SEEDS     = 150  # B seeds; each brings one k-NN partner → ~300 B total

def sample_batch():
    # A samples
    a_idx = []
    for k in range(K_OLD):
        pool = np.where((pseudo_y==k) & (tier>=1))[0]
        if len(pool):
            a_idx.extend(rng.choice(pool, min(N_A_PER_CLASS,len(pool)),
                                    replace=False).tolist())
    low = np.where(tier==0)[0]
    if len(low):
        a_idx.extend(rng.choice(low, min(N_A_LOW,len(low)), replace=False).tolist())
    a_idx = np.array(a_idx)

    # B samples — pair sampling ensures positives in batch
    seeds  = rng.choice(N_B, size=N_B_SEEDS, replace=False)
    partners = knn_B[seeds, rng.integers(0, K_POS, size=N_B_SEEDS)]
    b_idx  = np.unique(np.concatenate([seeds, partners]))

    return a_idx, b_idx

def build_b_pos_mask(b_idx, N_batch):
    """
    Build boolean (N_batch, N_batch) positive mask for B samples in the batch.
    Position in the batch = position in [a_idx concat b_idx].
    b_idx gives indices into the global B array; offset in batch = N_a + position.
    """
    N_a  = N_batch - len(b_idx)
    mask = torch.zeros(N_batch, N_batch, dtype=torch.bool)
    b_set = {int(bi): pos+N_a for pos, bi in enumerate(b_idx)}
    for pos_i, bi in enumerate(b_idx):
        batch_i = pos_i + N_a
        for knn_j in knn_B[bi]:
            if int(knn_j) in b_set:
                batch_j = b_set[int(knn_j)]
                mask[batch_i, batch_j] = True
                mask[batch_j, batch_i] = True
    return mask

# ── Training ──────────────────────────────────────────────────────────────────
EPOCHS = 150
head   = ProjectionHead().to(DEVICE)
opt    = torch.optim.Adam(head.parameters(), lr=3e-4, weight_decay=1e-4)
sch    = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

Xt_A = torch.from_numpy(X_A)
Xt_B = torch.from_numpy(X_B)
conf_t_all  = torch.from_numpy(conf.astype(np.float32))
tier_t_all  = torch.from_numpy(tier.astype(np.int64))
label_t_all = torch.from_numpy(pseudo_y.astype(np.int64))

print(f"\nTraining semi-supervised SupCon ({EPOCHS} epochs) …")
print(f"  A branch : pseudo-label SupCon (weighted by confidence)")
print(f"  B branch : {K_POS}-NN self-supervised positives (no pseudo-labels)")

for ep in range(EPOCHS):
    head.train()
    a_idx, b_idx = sample_batch()
    batch_idx    = np.concatenate([a_idx, b_idx])
    N_a, N_b_    = len(a_idx), len(b_idx)
    N_batch      = N_a + N_b_

    # Features
    x_a = Xt_A[a_idx].to(DEVICE)
    x_b = Xt_B[b_idx].to(DEVICE)
    x   = torch.cat([x_a, x_b], dim=0)

    # Labels / conf / tier (only meaningful for A portion)
    pl   = torch.cat([label_t_all[a_idx], torch.zeros(N_b_, dtype=torch.long)])
    cf   = torch.cat([conf_t_all[a_idx],  torch.zeros(N_b_)])
    ti   = torch.cat([tier_t_all[a_idx],  torch.zeros(N_b_, dtype=torch.long)])
    is_A = torch.cat([torch.ones(N_a, dtype=torch.bool),
                       torch.zeros(N_b_, dtype=torch.bool)])

    # B-B positive mask (built on CPU, passed to loss)
    bb_mask = build_b_pos_mask(b_idx, N_batch)

    z    = head(x)
    loss = semi_supcon_loss(z, is_A, pl, cf, ti, bb_mask)
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step()
    sch.step()

    if (ep+1) % 50 == 0:
        print(f"  ep {ep+1:>3}/{EPOCHS}  loss={loss.item():.4f}")

# ── Extract features ──────────────────────────────────────────────────────────
print("\nExtracting projected features …")
head.eval()
with torch.no_grad():
    Z_A = head(Xt_A.to(DEVICE)).cpu().numpy()
    Z_B = head(Xt_B.to(DEVICE)).cpu().numpy()

# ── Representation quality ────────────────────────────────────────────────────
def repr_quality(Z, pseudo, tag):
    intra, inter = [], []
    for k in range(0, K_OLD, 5):
        mem = Z[pseudo==k]
        if len(mem)<2: continue
        d = np.linalg.norm(mem[:,None,:]-mem[None,:,:], axis=-1)
        intra.append(d[np.triu_indices(len(mem),k=1)].mean())
        c_k = mem.mean(0)
        inter.append(np.linalg.norm(Z[pseudo!=k]-c_k, axis=1).mean())
    print(f"  {tag:<30}  intra={np.mean(intra):.4f}  "
          f"inter={np.mean(inter):.4f}  ratio={np.mean(inter)/np.mean(intra):.2f}x")

print("\nRepresentation quality (old species only — for reference):")
repr_quality(normalize(X_A,norm="l2"), pseudo_y, "Raw DINOv2 (A)")
repr_quality(Z_A, pseudo_y, "Semi-sup SupCon (A)")

# ── GCD evaluation ────────────────────────────────────────────────────────────
print("\nGCD evaluation …")

# Direct: K-Means on 128-d projected B
res_direct = gcd_acc(Z_B, "Semi-sup SupCon-128  (no UMAP)")

# With UMAP on projected features
print("  Computing UMAP on projected A+B …")
t0 = time.time()
r_sup = umap.UMAP(n_components=10, n_neighbors=20, min_dist=0.05,
                   metric="cosine", random_state=SEED, verbose=False)
X_AB_sup = normalize(
    r_sup.fit_transform(np.vstack([Z_A, Z_B])), norm="l2").astype(np.float32)
X_B_sup  = X_AB_sup[N_A:]
print(f"  Done in {time.time()-t0:.1f}s")
res_umap = gcd_acc(X_B_sup, "Semi-sup SupCon-128 → UMAP-10")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*72)
print("GCD EVALUATION SUMMARY  (K-Means(60) + Hungarian, all B samples)")
print("="*72)
print(f"  {'Feature space':<47} {'All':>6} {'Old':>6} {'Novel':>8}")
print("  " + "-"*70)

rows = [
    ("Baseline UMAP-10 (raw, no SupCon)",        res_baseline),
    ("Semi-sup SupCon-128 (no UMAP)",             res_direct),
    ("Semi-sup SupCon-128 → UMAP-10",             res_umap),
]
best_nov = max(r[2] for _, r in rows)
for name, (all_a, old_a, nov_a) in rows:
    mark = " ◄" if nov_a == best_nov else ""
    print(f"  {name:<47} {all_a:>6.1%} {old_a:>6.1%} {nov_a:>8.1%}{mark}")

print()
print("  Improvement of best SupCon over baseline:")
best = max(rows[1:], key=lambda x: x[1][2])
delta_all = best[1][0] - res_baseline[0]
delta_old = best[1][1] - res_baseline[1]
delta_nov = best[1][2] - res_baseline[2]
print(f"    All:   {res_baseline[0]:.1%} → {best[1][0]:.1%}  ({delta_all:+.1%})")
print(f"    Old:   {res_baseline[1]:.1%} → {best[1][1]:.1%}  ({delta_old:+.1%})")
print(f"    Novel: {res_baseline[2]:.1%} → {best[1][2]:.1%}  ({delta_nov:+.1%})")
