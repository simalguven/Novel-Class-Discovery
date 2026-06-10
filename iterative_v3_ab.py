"""
Iterative Pseudo-label Refinement — V3 + A→B Pseudo-label Propagation  (v2)
============================================================================

Why the first version degraded
-------------------------------
Round 0 peaked at 94.3% Novel then every subsequent round got worse.
Two root causes:

  1. GMM in 128-d is unreliable.
     ~100 samples per component, 128 dimensions, diagonal covariance →
     severely underdetermined. GMM fit on UMAP-10 (round 0) works because
     the space is compact and well-separated. In teacher's 128-d space it
     produces noisy covariance estimates that mislead the clustering.

  2. Anchoring threshold `median(conf_A)` becomes too permissive over rounds.
     As the teacher's space drifts, conf_A inflates (well-separated clusters
     look very confident). The rising median passes novel B samples through
     the threshold, creating corrupt A–B_novel positives that poison the loss.

Fixes applied
-------------
  1. Replace GMM → K-Means(50)
       K-Means is equivalent to spherical GMM — the correct geometric prior
       for L2-normalised embeddings. It has no covariance parameters to
       overfit and is stable in any dimension.
       Confidence comes from the same density + margin formula proven in V1.

  2. Raise anchoring threshold: median(conf_A)  →  75th-percentile(conf_A)
       Only the most confidently old-species B samples are anchored.
       Novel species get lower confidence (distances spread across all 50
       centroids), so the stricter threshold keeps them free.

Everything else — EMA teacher, SupCon loss, batch structure, evaluation —
is unchanged from the first version.
"""

import numpy as np, umap, warnings, time
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from scipy.optimize import linear_sum_assignment
warnings.filterwarnings("ignore")

SEED   = 42; K_OLD = 50; K_NEW = 60; K_NOV = 10; N_PER_CLS = 100; K_POS = 10
rng    = np.random.default_rng(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("mps"  if torch.backends.mps.is_available() else
                       "cuda" if torch.cuda.is_available() else "cpu")

# ── Data ──────────────────────────────────────────────────────────────────────
print("Loading data …")
embeddings  = np.load("plantnet_dinov2_raw.npy").astype(np.float32)
labels      = np.load("plantnet_labels.npy")
all_classes = np.unique(labels)
counts      = np.array([(labels == c).sum() for c in all_classes])
eligible    = all_classes[counts >= 2 * N_PER_CLS]
chosen_60   = np.sort(rng.choice(eligible, size=K_NEW, replace=False))
base_cls    = chosen_60[:K_OLD]; novel_cls = chosen_60[K_OLD:]

XA, XB, yAl, yBl = [], [], [], []
for c in base_cls:
    idx = rng.choice(np.where(labels == c)[0], size=2*N_PER_CLS, replace=False)
    XA.append(embeddings[idx[:N_PER_CLS]]); yAl.extend([c]*N_PER_CLS)
    XB.append(embeddings[idx[N_PER_CLS:]]); yBl.extend([c]*N_PER_CLS)
for c in novel_cls:
    idx = rng.choice(np.where(labels == c)[0], size=N_PER_CLS, replace=False)
    XB.append(embeddings[idx]); yBl.extend([c]*N_PER_CLS)

X_A = np.vstack(XA); X_B = np.vstack(XB)
y_A = np.array(yAl); y_B = np.array(yBl)
id2idx     = {c: i for i, c in enumerate(chosen_60)}
y_A_eval   = np.array([id2idx[c] for c in y_A])   # 0–49   (eval only)
y_B_eval   = np.array([id2idx[c] for c in y_B])   # 0–59   (eval only)
is_novel_B = y_B_eval >= K_OLD
N_A, N_B   = len(X_A), len(X_B)
print(f"  A:{N_A}  B:{N_B}  novel:{is_novel_B.sum()}")

# ── Baseline UMAP (A+B combined) ──────────────────────────────────────────────
print("\nBaseline: combined UMAP-10 on raw features …")
t0 = time.time()
r_base    = umap.UMAP(n_components=10, n_neighbors=20, min_dist=0.05,
                       metric="cosine", random_state=SEED, verbose=False)
X_AB_base = normalize(
    r_base.fit_transform(np.vstack([X_A, X_B])), norm="l2").astype(np.float32)
X_A_umap  = X_AB_base[:N_A]          # A portion — used in round 0
X_B_umap  = X_AB_base[N_A:]          # B portion — used for propagation in round 0
print(f"  Done in {time.time()-t0:.1f}s")

# ── Evaluation helpers ─────────────────────────────────────────────────────────
def gcd_acc(feat_B, tag="", n_init=20, verbose=True):
    km    = KMeans(n_clusters=K_NEW, n_init=n_init, random_state=SEED)
    preds = km.fit_predict(normalize(feat_B, norm="l2"))
    mat   = np.zeros((K_NEW, K_NEW), dtype=np.int64)
    for t, p in zip(y_B_eval, preds): mat[t, p] += 1
    row, col = linear_sum_assignment(-mat)
    p2t  = {c: r for r, c in zip(row, col)}
    pm   = np.array([p2t.get(p, -1) for p in preds])
    all_a = (pm == y_B_eval).mean()
    old_a = (pm[~is_novel_B] == y_B_eval[~is_novel_B]).mean()
    nov_a = (pm[is_novel_B]  == y_B_eval[is_novel_B]).mean()
    if verbose:
        print(f"  {tag:<50}  All={all_a:.1%}  Old={old_a:.1%}  Novel={nov_a:.1%}")
    return all_a, old_a, nov_a

def pseudo_label_acc(pseudo_y):
    mat = np.zeros((K_OLD, K_OLD), dtype=np.int64)
    for p, t in zip(pseudo_y, y_A_eval): mat[p % K_OLD, t] += 1
    r, c = linear_sum_assignment(-mat)
    return mat[r, c].sum() / N_A

res_baseline = gcd_acc(X_B_umap, "Baseline UMAP-10 (raw)")

# ── Fixed k-NN graph on B in raw DINOv2 space ─────────────────────────────────
print(f"\nPre-computing {K_POS}-NN for B (raw DINOv2, fixed across rounds) …")
t0    = time.time()
nbrs  = NearestNeighbors(n_neighbors=K_POS+1, metric="cosine", n_jobs=-1).fit(X_B)
knn_B = nbrs.kneighbors(X_B, return_distance=False)[:, 1:]   # (N_B, K_POS)
print(f"  Done in {time.time()-t0:.1f}s")

# ── Projection head ────────────────────────────────────────────────────────────
class ProjectionHead(nn.Module):
    def __init__(self, in_dim=768, hidden=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim))
    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)

# ── A→B pseudo-label propagation ──────────────────────────────────────────────
def cluster_confidence(Z_norm, centroids, pseudo_y):
    """
    Density + margin confidence — the same formula used in V1, proven stable.

    density : inverse normalised distance to own centroid (closer = more confident)
    margin  : gap between top-1 and top-2 soft assignments (exp(-dist/T))
    conf    : mean of both, in [0, 1]
    """
    T       = 0.1
    dist    = np.linalg.norm(Z_norm[:, None, :] - centroids[None, :, :], axis=-1)
    own_d   = dist[np.arange(len(Z_norm)), pseudo_y]
    soft    = np.exp(-dist / T); soft /= soft.sum(1, keepdims=True)
    ss      = np.sort(soft, axis=1)
    margin  = ss[:, -1] - ss[:, -2]
    c_dens  = 1 - (own_d  - own_d.min())  / (own_d.max()  - own_d.min()  + 1e-9)
    c_marg  = (margin - margin.min()) / (margin.max() - margin.min() + 1e-9)
    return ((c_dens + c_marg) / 2).astype(np.float32)


def propagate_to_B(centroids, Z_B_norm, conf_A):
    """
    Assign each B sample to its nearest K-Means centroid (computed from A).
    Anchor B samples whose confidence > 75th-percentile(conf_A).

    Stricter threshold (75th pct vs old median) prevents novel species —
    which spread distance across all 50 centroids and get lower confidence —
    from being incorrectly anchored as old.

    Returns
    -------
    pseudo_y_B   : (N_B,) nearest-centroid assignments
    conf_B       : (N_B,) density+margin confidence in [0, 1]
    tier_B       : (N_B,) 0 = free, 1 = anchored
    anchored_mask: (N_B,) bool
    """
    pseudo_y_B = np.linalg.norm(
        Z_B_norm[:, None, :] - centroids[None, :, :], axis=-1
    ).argmin(axis=1).astype(np.int64)

    conf_B = cluster_confidence(Z_B_norm, centroids, pseudo_y_B)

    threshold     = float(np.percentile(conf_A, 75))   # stricter than median
    anchored_mask = conf_B > threshold

    tier_B                = np.zeros(N_B, dtype=np.int64)
    tier_B[anchored_mask] = 1

    return pseudo_y_B, conf_B, tier_B, anchored_mask

# ── Extended SupCon loss (A–A, B_old–B_old, A–B_old + k-NN B) ─────────────────
def semi_supcon_loss_v2(z, is_labeled, pseudo_lbl, conf, tier, bb_mask, tau=0.1):
    """
    Replaces the is_A flag with is_labeled which is True for both A samples
    and anchored B samples.

    Labeled positive pair (i,j):
        both is_labeled, both tier >= 1, same pseudo_lbl, weight = conf_i * conf_j
        → covers A–A, B_old–B_old, A–B_old cross-domain pairs

    k-NN positive pair (i,j):
        in bb_mask (fixed raw-DINOv2 graph, all B),  weight = 1
        → covers structural neighbourhood for free B
    """
    N   = z.size(0)
    eye = torch.eye(N, dtype=torch.bool, device=z.device)

    sim = (z @ z.T) / tau
    mx, _ = sim.max(1, keepdim=True)
    exs   = torch.exp(sim - mx).masked_fill(eye, 0.0)
    denom = exs.sum(1, keepdim=True) + 1e-8
    lp    = (sim - mx) - torch.log(denom)

    il = is_labeled.to(z.device)
    pl = pseudo_lbl.to(z.device)
    cf = conf.to(z.device)
    ti = tier.to(z.device)

    same_ps     = (pl.unsqueeze(1) == pl.unsqueeze(0))
    both_lab    = il.unsqueeze(1) & il.unsqueeze(0)
    valid_tier  = (ti >= 1).unsqueeze(1) & (ti >= 1).unsqueeze(0)
    lab_mask    = same_ps & both_lab & valid_tier & ~eye
    lab_weight  = (cf.unsqueeze(1) * cf.unsqueeze(0)) * lab_mask.float()

    bb_m = bb_mask.to(z.device)
    w    = lab_weight + bb_m.float()
    wlp  = (w * lp).sum(1)
    ws   = w.sum(1).clamp(1e-8)

    anch = w.sum(1).gt(0)           # any sample with at least one positive
    if not anch.any():
        return torch.tensor(0., device=z.device, requires_grad=True)
    return -(wlp / ws)[anch].mean()

# ── Batch sampling ─────────────────────────────────────────────────────────────
N_A_PER_CLASS    = 6
N_A_LOW          = 50
N_B_ANCH_PER_CLS = 4     # anchored B per pseudo-class — fewer than A to keep balance
N_B_FREE_SEEDS   = 100   # seeds from free (unanchored) B

def sample_batch(pseudo_y_A, tier_A, anchored_mask, pseudo_y_B, tier_B):
    # ── A: balanced per class + low-confidence pool ───────────────────────────
    a_idx = []
    for k in range(K_OLD):
        pool = np.where((pseudo_y_A == k) & (tier_A >= 1))[0]
        if len(pool):
            a_idx.extend(rng.choice(pool, min(N_A_PER_CLASS, len(pool)),
                                    replace=False).tolist())
    low = np.where(tier_A == 0)[0]
    if len(low):
        a_idx.extend(rng.choice(low, min(N_A_LOW, len(low)), replace=False).tolist())

    # ── Anchored B: balanced per pseudo-class ────────────────────────────────
    b_anch_idx = []
    for k in range(K_OLD):
        pool = np.where(anchored_mask & (pseudo_y_B == k) & (tier_B >= 1))[0]
        if len(pool):
            b_anch_idx.extend(rng.choice(pool, min(N_B_ANCH_PER_CLS, len(pool)),
                                         replace=False).tolist())
    b_anch_idx = np.array(b_anch_idx, dtype=np.int64)

    # ── Free B: seeds from non-anchored pool + kNN partners ──────────────────
    free_pool = np.where(~anchored_mask)[0]
    n_seeds   = min(N_B_FREE_SEEDS, len(free_pool))
    if n_seeds > 0:
        seeds    = rng.choice(free_pool, size=n_seeds, replace=False)
        partners = knn_B[seeds, rng.integers(0, K_POS, size=n_seeds)]
        b_free_idx = np.unique(np.concatenate([seeds, partners]))
    else:
        b_free_idx = np.array([], dtype=np.int64)

    # Remove any free index that is already in anchored (kNN partner may be anchored)
    anch_set   = set(b_anch_idx.tolist())
    b_free_idx = np.array([i for i in b_free_idx if i not in anch_set], dtype=np.int64)

    return np.array(a_idx, dtype=np.int64), b_anch_idx, b_free_idx

def build_bb_mask(b_idx, N_batch_total, N_a):
    """k-NN positive mask for all B samples in the batch (anchored + free)."""
    mask  = torch.zeros(N_batch_total, N_batch_total, dtype=torch.bool)
    b_set = {int(bi): pos + N_a for pos, bi in enumerate(b_idx)}
    for pos_i, bi in enumerate(b_idx):
        pi = pos_i + N_a
        for kj in knn_B[bi]:
            if int(kj) in b_set:
                pj = b_set[int(kj)]
                mask[pi, pj] = True; mask[pj, pi] = True
    return mask

# ── Iterative training ─────────────────────────────────────────────────────────
ROUNDS   = 6
EPOCHS_0 = 100
EPOCHS_R = 50
LR_0     = 3e-4
LR_R     = 1e-4

Xt_A = torch.from_numpy(X_A)
Xt_B = torch.from_numpy(X_B)

head = None; teacher = None
all_results = [("Baseline (raw UMAP-10)", *res_baseline)]

print("\n" + "="*72)
print("ITERATIVE PSEUDO-LABEL REFINEMENT  (K-Means + EMA + A→B PROPAGATION)")
print("="*72)

for rnd in range(ROUNDS):
    print(f"\n{'─'*72}")
    print(f"ROUND {rnd}  {'(fresh head)' if rnd == 0 else '(fine-tuning)'}")
    print(f"{'─'*72}")

    # ── Step 1a: features for GMM ─────────────────────────────────────────────
    if rnd == 0:
        # UMAP space: A and B are already co-embedded, propagation is natural
        Z_A_for_gmm = X_A_umap
        Z_B_for_gmm = X_B_umap
    else:
        teacher.eval()
        with torch.no_grad():
            Z_A_for_gmm = teacher(Xt_A.to(DEVICE)).cpu().numpy()
            Z_B_for_gmm = teacher(Xt_B.to(DEVICE)).cpu().numpy()

    Z_A_norm = normalize(Z_A_for_gmm, norm="l2").astype(np.float32)
    Z_B_norm = normalize(Z_B_for_gmm, norm="l2").astype(np.float32)

    # ── Step 1b: K-Means(50) on A ────────────────────────────────────────────
    print(f"  [1] K-Means(50) on A …")
    km50       = KMeans(n_clusters=K_OLD, n_init=15, random_state=SEED).fit(Z_A_norm)
    pseudo_y_A = km50.labels_
    centroids  = normalize(km50.cluster_centers_, norm="l2").astype(np.float32)

    conf_A   = cluster_confidence(Z_A_norm, centroids, pseudo_y_A)
    p30, p70 = np.percentile(conf_A, 30), np.percentile(conf_A, 70)
    tier_A   = np.where(conf_A >= p70, 2, np.where(conf_A >= p30, 1, 0)).astype(np.int64)

    pl_acc = pseudo_label_acc(pseudo_y_A)
    print(f"       A pseudo-label acc: {pl_acc:.1%}  |  "
          f"tier2:{(tier_A==2).sum()}  tier1:{(tier_A==1).sum()}  tier0:{(tier_A==0).sum()}")

    # ── Step 1c: Propagate K-Means labels to B ───────────────────────────────
    pseudo_y_B, conf_B, tier_B, anchored_mask = propagate_to_B(centroids, Z_B_norm, conf_A)

    # Diagnostic — uses ground truth for reporting only, never in training
    n_anch = int(anchored_mask.sum())
    if n_anch > 0:
        true_old_anch  = int((~is_novel_B[anchored_mask]).sum())
        anch_precision = true_old_anch / n_anch
        anch_recall    = true_old_anch / int((~is_novel_B).sum())
        print(f"       B anchored: {n_anch}/{N_B}  "
              f"precision={anch_precision:.1%}  recall={anch_recall:.1%}  "
              f"threshold={np.percentile(conf_A, 75):.3f}")
    else:
        print(f"       B anchored: 0/{N_B}  (threshold={np.percentile(conf_A, 75):.3f})")

    # Torch tensors
    conf_A_t  = torch.from_numpy(conf_A)
    tier_A_t  = torch.from_numpy(tier_A)
    label_A_t = torch.from_numpy(pseudo_y_A.astype(np.int64))
    conf_B_t  = torch.from_numpy(conf_B)
    tier_B_t  = torch.from_numpy(tier_B)
    label_B_t = torch.from_numpy(pseudo_y_B)

    # ── Step 2: train / fine-tune ─────────────────────────────────────────────
    epochs = EPOCHS_0 if rnd == 0 else EPOCHS_R
    lr     = LR_0     if rnd == 0 else LR_R

    if rnd == 0:
        head    = ProjectionHead().to(DEVICE)
        teacher = ProjectionHead().to(DEVICE)
        teacher.load_state_dict(head.state_dict())
        for p in teacher.parameters(): p.requires_grad = False

    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    print(f"  [2] Training {epochs} epochs (lr={lr:.0e}) …")
    t0 = time.time()

    for ep in range(epochs):
        head.train()
        a_idx, b_anch_idx, b_free_idx = sample_batch(
            pseudo_y_A, tier_A, anchored_mask, pseudo_y_B, tier_B)

        n_a     = len(a_idx)
        n_banch = len(b_anch_idx)
        n_bfree = len(b_free_idx)

        # Batch order: [A | B_anchored | B_free]
        b_all_idx = np.concatenate([b_anch_idx, b_free_idx])
        N_bt      = n_a + len(b_all_idx)

        x = torch.cat([Xt_A[a_idx], Xt_B[b_all_idx]], 0).to(DEVICE)

        # Pseudo-labels: A and anchored-B have labels; free-B gets dummy 0
        pl = torch.cat([label_A_t[a_idx],
                        label_B_t[b_anch_idx],
                        torch.zeros(n_bfree, dtype=torch.long)])

        # Confidence: A and anchored-B use their scores; free-B = 0
        cf = torch.cat([conf_A_t[a_idx],
                        conf_B_t[b_anch_idx],
                        torch.zeros(n_bfree)])

        # Tier: same split
        ti = torch.cat([tier_A_t[a_idx],
                        tier_B_t[b_anch_idx],
                        torch.zeros(n_bfree, dtype=torch.long)])

        # is_labeled: True for A + anchored-B, False for free-B
        is_labeled = torch.cat([torch.ones(n_a + n_banch, dtype=torch.bool),
                                 torch.zeros(n_bfree, dtype=torch.bool)])

        bbm = build_bb_mask(b_all_idx, N_bt, n_a)

        z    = head(x)
        loss = semi_supcon_loss_v2(z, is_labeled, pl, cf, ti, bbm)

        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step()
        sch.step()

        # EMA teacher update
        with torch.no_grad():
            for pq, pk in zip(head.parameters(), teacher.parameters()):
                pk.data.mul_(0.99).add_(0.01 * pq.data)

        if (ep + 1) % (epochs // 2) == 0:
            print(f"       ep {ep+1}/{epochs}  loss={loss.item():.4f}  "
                  f"batch=[A:{n_a} B_anch:{n_banch} B_free:{n_bfree}]")

    print(f"       Training time: {time.time()-t0:.1f}s")

    # ── Step 3: Evaluate with teacher ────────────────────────────────────────
    teacher.eval()
    with torch.no_grad():
        Z_B_eval = teacher(Xt_B.to(DEVICE)).cpu().numpy()

    res = gcd_acc(Z_B_eval, f"Round {rnd} — GMM+EMA+Prop", verbose=False)
    print(f"  [3] All={res[0]:.1%}  Old={res[1]:.1%}  Novel={res[2]:.1%}")
    all_results.append((f"Round {rnd}", *res))

# ── Final summary ──────────────────────────────────────────────────────────────
print("\n" + "="*72)
print("SUMMARY — GCD metrics across rounds")
print("="*72)
print(f"  {'Method':<50} {'All':>7} {'Old':>7} {'Novel':>9}")
print("  " + "-"*74)

best_nov = max(r[3] for r in all_results)
best_all = max(r[1] for r in all_results)
for tag, all_a, old_a, nov_a in all_results:
    marks = []
    if nov_a == best_nov: marks.append("Novel◄")
    if all_a == best_all: marks.append("All◄")
    mark = "  " + "/".join(marks) if marks else ""
    print(f"  {tag:<50} {all_a:>7.1%} {old_a:>7.1%} {nov_a:>9.1%}{mark}")

print()
print("  Δ vs baseline (best round):")
best = max(all_results[1:], key=lambda x: x[1])
for metric, idx in [("All", 1), ("Old", 2), ("Novel", 3)]:
    delta = best[idx] - all_results[0][idx]
    print(f"    {metric}: {all_results[0][idx]:.1%} → {best[idx]:.1%}  ({delta:+.1%})")
