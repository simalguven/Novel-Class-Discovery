"""
Iterative Pseudo-label Refinement with Semi-supervised SupCon
+ Soft-constrained Semi-supervised K-Means Evaluation
====================================================

Each round:
  1. Compute pseudo-labels from CURRENT projected A features
     Round 0: raw UMAP-10 of A
     Round t>0: previous round's projected A

  2. Recompute confidence scores from new pseudo-labels

  3. Train / fine-tune projection head with semi-supervised SupCon
       A branch: confidence-weighted pseudo-label SupCon
       B branch: k-NN positives in raw DINOv2 space

  4. Extract new projected features Z_A, Z_B

  5. Evaluate:
       a) Plain GCD K-Means(60) on Z_B
       b) Soft-constrained K-Means(60) on Z_A + Z_B
          - A samples are softly pulled toward their pseudo-label centroids
          - B samples are free
          - novel clusters are identified as unmatched clusters

  6. Report pseudo-label accuracy on A

Important fix:
  Soft-constrained KMeans runs in the CURRENT representation space.
  Therefore, the 50 old anchor centroids must be recomputed in Z_A space
  every round. Do not reuse cent50 from UMAP-10 when Z_A is 128-d.
"""

import numpy as np
import umap
import warnings
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

SEED = 42

K_OLD = 50
K_NEW = 60
K_NOV = 10

N_PER_CLS = 100
K_POS = 10

ROUNDS = 6
EPOCHS_0 = 100
EPOCHS_R = 50

LR_0 = 3e-4
LR_R = 1e-4

TAU = 0.1

N_A_PER_CLASS = 6
N_A_LOW = 50
N_B_SEEDS = 150

SOFT_KMEANS_LAMBDAS = [0.1, 0.5, 1.0, 2.0, 5.0]

rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print(f"Using device: {DEVICE}")


# ──────────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────────

print("Loading data …")

embeddings = np.load("plantnet_dinov2_raw.npy").astype(np.float32)
labels = np.load("plantnet_labels.npy")

all_classes = np.unique(labels)
counts = np.array([(labels == c).sum() for c in all_classes])
eligible = all_classes[counts >= 2 * N_PER_CLS]

chosen_60 = np.sort(rng.choice(eligible, size=K_NEW, replace=False))
base_cls = chosen_60[:K_OLD]
novel_cls = chosen_60[K_OLD:]

XA, XB, yAl, yBl = [], [], [], []

for c in base_cls:
    idx = rng.choice(
        np.where(labels == c)[0],
        size=2 * N_PER_CLS,
        replace=False,
    )

    XA.append(embeddings[idx[:N_PER_CLS]])
    yAl.extend([c] * N_PER_CLS)

    XB.append(embeddings[idx[N_PER_CLS:]])
    yBl.extend([c] * N_PER_CLS)

for c in novel_cls:
    idx = rng.choice(
        np.where(labels == c)[0],
        size=N_PER_CLS,
        replace=False,
    )

    XB.append(embeddings[idx])
    yBl.extend([c] * N_PER_CLS)

X_A = np.vstack(XA).astype(np.float32)
X_B = np.vstack(XB).astype(np.float32)

y_A = np.array(yAl)
y_B = np.array(yBl)

id2idx = {c: i for i, c in enumerate(chosen_60)}

y_A_eval = np.array([id2idx[c] for c in y_A])  # 0–49
y_B_eval = np.array([id2idx[c] for c in y_B])  # 0–59

is_novel_B = y_B_eval >= K_OLD

N_A = len(X_A)
N_B = len(X_B)

print(f"  A: {N_A}")
print(f"  B: {N_B}")
print(f"  Novel samples in B: {is_novel_B.sum()}")


# ──────────────────────────────────────────────────────────────────────────────
# Baseline UMAP
# ──────────────────────────────────────────────────────────────────────────────

print("\nBaseline: combined UMAP-10 on raw features …")

t0 = time.time()

r_base = umap.UMAP(
    n_components=10,
    n_neighbors=20,
    min_dist=0.05,
    metric="cosine",
    random_state=SEED,
    verbose=False,
)

X_AB_base = r_base.fit_transform(np.vstack([X_A, X_B]))
X_AB_base = normalize(X_AB_base, norm="l2").astype(np.float32)

X_A_umap = X_AB_base[:N_A]
X_B_base = X_AB_base[N_A:]

print(f"  Done in {time.time() - t0:.1f}s")


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ──────────────────────────────────────────────────────────────────────────────

def gcd_acc(feat_B, tag="", n_init=20, verbose=True):
    """
    Standard GCD evaluation:
    KMeans(60) on B features only, then Hungarian matching to true 60 classes.
    """

    feat_B = normalize(feat_B, norm="l2")

    km = KMeans(
        n_clusters=K_NEW,
        n_init=n_init,
        random_state=SEED,
    )

    preds = km.fit_predict(feat_B)

    mat = np.zeros((K_NEW, K_NEW), dtype=np.int64)

    for t, p in zip(y_B_eval, preds):
        mat[t, p] += 1

    row, col = linear_sum_assignment(-mat)

    p2t = {cluster_id: true_id for true_id, cluster_id in zip(row, col)}
    mapped_preds = np.array([p2t.get(p, -1) for p in preds])

    all_a = (mapped_preds == y_B_eval).mean()
    old_a = (mapped_preds[~is_novel_B] == y_B_eval[~is_novel_B]).mean()
    nov_a = (mapped_preds[is_novel_B] == y_B_eval[is_novel_B]).mean()

    if verbose:
        print(
            f"  {tag:<45} "
            f"All={all_a:.1%}  Old={old_a:.1%}  Novel={nov_a:.1%}"
        )

    return all_a, old_a, nov_a


def pseudo_label_acc(pseudo_y):
    """
    Evaluation-only pseudo-label accuracy on A.
    Uses Hungarian matching between pseudo clusters and true old classes.
    """

    mat = np.zeros((K_OLD, K_OLD), dtype=np.int64)

    for p, t in zip(pseudo_y, y_A_eval):
        mat[p % K_OLD, t] += 1

    row, col = linear_sum_assignment(-mat)

    return mat[row, col].sum() / N_A


def cluster_acc_from_assignments(assign_B):
    """
    Accuracy of 60-cluster assignments on B after Hungarian matching.
    Used for soft-constrained KMeans, where assignments come from A+B clustering.
    """

    mat = np.zeros((K_NEW, K_NEW), dtype=np.int64)

    for t, p in zip(y_B_eval, assign_B):
        mat[t, p] += 1

    row, col = linear_sum_assignment(-mat)

    p2t = {cluster_id: true_id for true_id, cluster_id in zip(row, col)}
    mapped_preds = np.array([p2t.get(p, -1) for p in assign_B])

    all_a = (mapped_preds == y_B_eval).mean()
    old_a = (mapped_preds[~is_novel_B] == y_B_eval[~is_novel_B]).mean()
    nov_a = (mapped_preds[is_novel_B] == y_B_eval[is_novel_B]).mean()

    return all_a, old_a, nov_a


def novelty_detection_metrics(assign_B, novel_ids, novelty_score):
    """
    Novelty detection metrics based on which clusters are identified as novel.
    """

    novel_mask = np.isin(assign_B, list(novel_ids))

    tp = is_novel_B[novel_mask].sum()
    flagged = novel_mask.sum()

    precision = tp / flagged if flagged > 0 else 0.0
    recall = tp / is_novel_B.sum()
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    fpr = (~is_novel_B[novel_mask]).sum() / (~is_novel_B).sum()

    try:
        auc = roc_auc_score(is_novel_B, novelty_score)
    except ValueError:
        auc = 0.0

    return {
        "auc": auc,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "fpr": fpr,
        "flagged": int(flagged),
    }


def novel_cluster_acc(assign_B, novel_ids):
    """
    Cluster accuracy only on truly novel samples that were assigned to clusters
    identified as novel.
    """

    novel_mask = np.isin(assign_B, list(novel_ids))
    truly_novel_in_flagged = is_novel_B[novel_mask]

    y_true = y_B_eval[novel_mask][truly_novel_in_flagged] - K_OLD
    y_pred = assign_B[novel_mask][truly_novel_in_flagged]

    if len(y_true) < K_NOV:
        return 0.0, int(truly_novel_in_flagged.sum())

    uniq = {v: i for i, v in enumerate(np.unique(y_pred))}
    yp = np.array([uniq[v] for v in y_pred])

    n = max(y_true.max(), yp.max()) + 1
    mat = np.zeros((n, n), dtype=np.int64)

    for t, p in zip(y_true, yp):
        mat[t, p] += 1

    row, col = linear_sum_assignment(-mat)

    return mat[row, col].sum() / len(y_true), int(truly_novel_in_flagged.sum())


res_baseline = gcd_acc(X_B_base, "Baseline UMAP-10 raw")


# ──────────────────────────────────────────────────────────────────────────────
# Fixed k-NN graph for B in raw DINOv2 space
# ──────────────────────────────────────────────────────────────────────────────

print(f"\nPre-computing {K_POS}-NN for B in raw DINOv2 space …")

t0 = time.time()

nbrs = NearestNeighbors(
    n_neighbors=K_POS + 1,
    metric="cosine",
    n_jobs=-1,
).fit(X_B)

knn_B = nbrs.kneighbors(X_B, return_distance=False)[:, 1:]

print(f"  Done in {time.time() - t0:.1f}s")


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    def __init__(self, in_dim=768, hidden=256, out_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


# ──────────────────────────────────────────────────────────────────────────────
# Semi-supervised SupCon loss
# ──────────────────────────────────────────────────────────────────────────────

def semi_supcon_loss(z, is_A, pseudo_lbl, conf, tier, bb_mask, tau=0.1):
    """
    Semi-supervised SupCon:

    A-A positives:
      same pseudo-label, both confidence tier >= 1

    B-B positives:
      fixed raw-DINO kNN graph

    Positive weights:
      A-A uses conf_i * conf_j
      B-B uses 1
    """

    N = z.size(0)
    device = z.device

    eye = torch.eye(N, dtype=torch.bool, device=device)

    sim = (z @ z.T) / tau
    mx, _ = sim.max(dim=1, keepdim=True)

    exp_sim = torch.exp(sim - mx).masked_fill(eye, 0.0)
    denom = exp_sim.sum(dim=1, keepdim=True) + 1e-8

    log_prob = (sim - mx) - torch.log(denom)

    is_A = is_A.to(device)

    same_pseudo = pseudo_lbl.unsqueeze(1) == pseudo_lbl.unsqueeze(0)
    valid_conf = (tier >= 1).unsqueeze(1) & (tier >= 1).unsqueeze(0)
    both_A = is_A.unsqueeze(1) & is_A.unsqueeze(0)

    aa_mask = same_pseudo & valid_conf & both_A & ~eye
    aa_weight = conf.unsqueeze(1) * conf.unsqueeze(0) * aa_mask.float()

    bb_mask = bb_mask.to(device)

    weights = aa_weight + bb_mask.float()

    weighted_log_prob = (weights * log_prob).sum(dim=1)
    weight_sum = weights.sum(dim=1).clamp_min(1e-8)

    a_anchor = is_A & (tier >= 1) & aa_mask.sum(dim=1).gt(0)
    b_anchor = (~is_A) & bb_mask.sum(dim=1).gt(0)

    anchors = a_anchor | b_anchor

    if not anchors.any():
        return torch.tensor(0.0, device=device, requires_grad=True)

    return -(weighted_log_prob / weight_sum)[anchors].mean()


# ──────────────────────────────────────────────────────────────────────────────
# Batch sampling
# ──────────────────────────────────────────────────────────────────────────────

def sample_batch(pseudo_y, tier):
    """
    A:
      balanced by pseudo-class, mostly mid/high confidence,
      plus some low-confidence samples.

    B:
      random seed samples plus one raw-DINO kNN positive per seed.
    """

    a_idx = []

    for k in range(K_OLD):
        pool = np.where((pseudo_y == k) & (tier >= 1))[0]

        if len(pool):
            picked = rng.choice(
                pool,
                size=min(N_A_PER_CLASS, len(pool)),
                replace=False,
            )

            a_idx.extend(picked.tolist())

    low = np.where(tier == 0)[0]

    if len(low):
        picked_low = rng.choice(
            low,
            size=min(N_A_LOW, len(low)),
            replace=False,
        )

        a_idx.extend(picked_low.tolist())

    seeds = rng.choice(N_B, size=N_B_SEEDS, replace=False)
    partners = knn_B[seeds, rng.integers(0, K_POS, size=N_B_SEEDS)]

    b_idx = np.unique(np.concatenate([seeds, partners]))

    return np.array(a_idx), b_idx


def build_bb_mask(b_idx, n_batch_total, n_a):
    """
    Builds a B-B positive mask inside the current mini-batch.
    """

    mask = torch.zeros(n_batch_total, n_batch_total, dtype=torch.bool)

    b_set = {int(bi): pos + n_a for pos, bi in enumerate(b_idx)}

    for pos_i, bi in enumerate(b_idx):
        pi = pos_i + n_a

        for kj in knn_B[bi]:
            kj = int(kj)

            if kj in b_set:
                pj = b_set[kj]

                mask[pi, pj] = True
                mask[pj, pi] = True

    return mask


# ──────────────────────────────────────────────────────────────────────────────
# Soft-constrained KMeans
# ──────────────────────────────────────────────────────────────────────────────

def kmeans_pp_init(X_all, k=K_NEW):
    """
    KMeans++ initialization.
    Used so plain, hard, and soft KMeans can share the same type of initialization.
    """

    km = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=1,
        random_state=SEED,
        max_iter=1,
    )

    km.fit(X_all)

    return km.cluster_centers_.copy()


def soft_seeded_kmeans(
    X_all,
    pinned_idx,
    pinned_labels,
    confidence,
    lam,
    scale,
    max_iter=150,
    tol=1e-5,
):
    """
    Soft must-link KMeans.

    For pinned A samples only:

        cost(i → k) =
            distance(i, centroid_k) + λ * confidence_i * scale, if k != pseudo_label_i
            distance(i, centroid_k),                             if k == pseudo_label_i

    λ = 0 gives plain KMeans behavior.
    Larger λ makes pseudo-label constraints stronger.

    B samples are never pinned.
    """

    N = len(X_all)

    centroids = kmeans_pp_init(X_all, K_NEW)

    penalty_lookup = {}

    for pi, pl, cf in zip(pinned_idx, pinned_labels, confidence):
        penalty_lookup[int(pi)] = (int(pl), float(cf) * float(scale) * float(lam))

    assign = np.zeros(N, dtype=np.int32)

    for _ in range(max_iter):
        dists = np.linalg.norm(
            X_all[:, None, :] - centroids[None, :, :],
            axis=-1,
        ).copy()

        for pi, (pl, penalty) in penalty_lookup.items():
            dists[pi, :] += penalty
            dists[pi, pl] -= penalty

        assign = dists.argmin(axis=1)

        new_centroids = np.array([
            X_all[assign == k].mean(axis=0) if (assign == k).any()
            else centroids[k]
            for k in range(K_NEW)
        ])

        shift = np.linalg.norm(new_centroids - centroids)

        centroids = new_centroids

        if shift < tol:
            break

    return assign, centroids


def hard_seeded_kmeans(
    X_all,
    pinned_idx,
    pinned_labels,
    max_iter=150,
    tol=1e-5,
):
    """
    Hard must-link KMeans.
    Pinned A samples are forced to their pseudo-label cluster.
    """

    centroids = kmeans_pp_init(X_all, K_NEW)

    assign = np.zeros(len(X_all), dtype=np.int32)

    for _ in range(max_iter):
        dists = np.linalg.norm(
            X_all[:, None, :] - centroids[None, :, :],
            axis=-1,
        )

        assign = dists.argmin(axis=1)

        assign[pinned_idx] = pinned_labels

        new_centroids = np.array([
            X_all[assign == k].mean(axis=0) if (assign == k).any()
            else centroids[k]
            for k in range(K_NEW)
        ])

        shift = np.linalg.norm(new_centroids - centroids)

        centroids = new_centroids

        if shift < tol:
            break

    return assign, centroids


def identify_novel_clusters_hungarian(
    assign_all,
    centroids60,
    cent50_current,
):
    """
    Identifies novel clusters using Hungarian matching.

    We have:
      - 50 A-only centroids in CURRENT representation space
      - 60 centroids from KMeans(60) on A+B in CURRENT representation space

    The 50 best-matched combined centroids are treated as old.
    The remaining 10 are treated as novel.
    """

    cost = np.linalg.norm(
        centroids60[:, None, :] - cent50_current[None, :, :],
        axis=-1,
    )  # shape: (60, 50)

    INF = cost.max() * 10.0 + 1e-9

    cost_sq = np.full((K_NEW, K_NEW), INF, dtype=np.float32)
    cost_sq[:K_OLD, :] = cost.T  # shape: (50, 60)

    row, col = linear_sum_assignment(cost_sq)

    matched_old_clusters = set(col[:K_OLD])
    novel_ids = set(range(K_NEW)) - matched_old_clusters

    match_cost = np.full(K_NEW, cost.max() * 2.0 + 1e-9)

    for old_id, combined_cluster_id in zip(range(K_OLD), col[:K_OLD]):
        match_cost[combined_cluster_id] = cost_sq[old_id, combined_cluster_id]

    assign_B = assign_all[N_A:]
    novelty_score = match_cost[assign_B]

    return novel_ids, novelty_score


def compute_cent50_current_space(Z_A, pseudo_y):
    """
    Recompute the 50 old anchor centroids in the SAME feature space as Z_A.

    This is required because pseudo_y may have been generated from UMAP-10,
    while Z_A after the projection head is 128-dimensional.
    """

    cent50_current = np.zeros((K_OLD, Z_A.shape[1]), dtype=np.float32)

    for k in range(K_OLD):
        members = Z_A[pseudo_y == k]

        if len(members) == 0:
            cent50_current[k] = Z_A[rng.integers(0, N_A)]
        else:
            cent50_current[k] = members.mean(axis=0)

    cent50_current = normalize(cent50_current, norm="l2").astype(np.float32)

    return cent50_current


def soft_constraint_evaluation(
    Z_A,
    Z_B,
    pseudo_y,
    conf,
    tag="",
    lambdas=SOFT_KMEANS_LAMBDAS,
):
    """
    Runs soft-constrained semi-supervised KMeans on current representation.

    Inputs:
      Z_A: current projected A
      Z_B: current projected B
      pseudo_y: current KMeans(50) pseudo-labels for A
      conf: confidence score for each A sample

    Important:
      The old anchor centroids are recomputed in Z_A space.
    """

    print(f"\n  [Soft-constrained KMeans evaluation] {tag}")

    Z_A = normalize(Z_A, norm="l2").astype(np.float32)
    Z_B = normalize(Z_B, norm="l2").astype(np.float32)

    X_all = np.vstack([Z_A, Z_B]).astype(np.float32)

    pinned_idx = np.arange(N_A, dtype=np.int32)
    pinned_labels = pseudo_y.astype(np.int32)

    conf_norm = conf.astype(np.float32)
    conf_norm = (conf_norm - conf_norm.min()) / (
        conf_norm.max() - conf_norm.min() + 1e-9
    )

    # Critical fix:
    # recompute 50 old centroids in the current representation space.
    cent50_current = compute_cent50_current_space(Z_A, pseudo_y)

    dists_A = np.linalg.norm(
        Z_A[:, None, :] - cent50_current[None, :, :],
        axis=-1,
    )

    scale = dists_A.min(axis=1).mean()

    results = {}

    # Plain KMeans(60) on A+B
    km_plain = KMeans(
        n_clusters=K_NEW,
        n_init=15,
        random_state=SEED,
    )

    assign_plain = km_plain.fit_predict(X_all)

    novel_ids_plain, novelty_score_plain = identify_novel_clusters_hungarian(
        assign_plain,
        km_plain.cluster_centers_,
        cent50_current,
    )

    assign_B_plain = assign_plain[N_A:]

    plain_all, plain_old, plain_nov = cluster_acc_from_assignments(assign_B_plain)

    plain_det = novelty_detection_metrics(
        assign_B_plain,
        novel_ids_plain,
        novelty_score_plain,
    )

    plain_novel_cacc, plain_novel_count = novel_cluster_acc(
        assign_B_plain,
        novel_ids_plain,
    )

    results["Plain A+B KMeans"] = {
        "all": plain_all,
        "old": plain_old,
        "novel": plain_nov,
        "det": plain_det,
        "novel_cluster_acc": plain_novel_cacc,
        "novel_count": plain_novel_count,
    }

    print(
        f"    Plain A+B KMeans      "
        f"All={plain_all:.1%} Old={plain_old:.1%} Novel={plain_nov:.1%} | "
        f"AUC={plain_det['auc']:.4f} F1={plain_det['f1']:.3f}"
    )

    # Hard all-A KMeans
    assign_hard, cent_hard = hard_seeded_kmeans(
        X_all,
        pinned_idx,
        pinned_labels,
    )

    novel_ids_hard, novelty_score_hard = identify_novel_clusters_hungarian(
        assign_hard,
        cent_hard,
        cent50_current,
    )

    assign_B_hard = assign_hard[N_A:]

    hard_all, hard_old, hard_nov = cluster_acc_from_assignments(assign_B_hard)

    hard_det = novelty_detection_metrics(
        assign_B_hard,
        novel_ids_hard,
        novelty_score_hard,
    )

    hard_novel_cacc, hard_novel_count = novel_cluster_acc(
        assign_B_hard,
        novel_ids_hard,
    )

    results["Hard all-A"] = {
        "all": hard_all,
        "old": hard_old,
        "novel": hard_nov,
        "det": hard_det,
        "novel_cluster_acc": hard_novel_cacc,
        "novel_count": hard_novel_count,
    }

    print(
        f"    Hard all-A            "
        f"All={hard_all:.1%} Old={hard_old:.1%} Novel={hard_nov:.1%} | "
        f"AUC={hard_det['auc']:.4f} F1={hard_det['f1']:.3f}"
    )

    # Soft all-A KMeans
    for lam in lambdas:
        assign_soft, cent_soft = soft_seeded_kmeans(
            X_all=X_all,
            pinned_idx=pinned_idx,
            pinned_labels=pinned_labels,
            confidence=conf_norm,
            lam=lam,
            scale=scale,
        )

        novel_ids_soft, novelty_score_soft = identify_novel_clusters_hungarian(
            assign_soft,
            cent_soft,
            cent50_current,
        )

        assign_B_soft = assign_soft[N_A:]

        soft_all, soft_old, soft_nov = cluster_acc_from_assignments(assign_B_soft)

        soft_det = novelty_detection_metrics(
            assign_B_soft,
            novel_ids_soft,
            novelty_score_soft,
        )

        soft_novel_cacc, soft_novel_count = novel_cluster_acc(
            assign_B_soft,
            novel_ids_soft,
        )

        key = f"Soft λ={lam}"

        results[key] = {
            "all": soft_all,
            "old": soft_old,
            "novel": soft_nov,
            "det": soft_det,
            "novel_cluster_acc": soft_novel_cacc,
            "novel_count": soft_novel_count,
        }

        print(
            f"    Soft λ={lam:<4}           "
            f"All={soft_all:.1%} Old={soft_old:.1%} Novel={soft_nov:.1%} | "
            f"AUC={soft_det['auc']:.4f} F1={soft_det['f1']:.3f}"
        )

    best_key = max(results.keys(), key=lambda k: results[k]["all"])
    best = results[best_key]

    print(
        f"    Best soft-constrained by All: {best_key} "
        f"All={best['all']:.1%} Old={best['old']:.1%} Novel={best['novel']:.1%}"
    )

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Iterative training
# ──────────────────────────────────────────────────────────────────────────────

Xt_A = torch.from_numpy(X_A)
Xt_B = torch.from_numpy(X_B)

head = None

# Round 0 starts from UMAP A pseudo-labels.
Z_A_current = X_A_umap

all_results = [("Baseline raw UMAP-10", *res_baseline)]
soft_results_by_round = {}

print("\n" + "=" * 72)
print("ITERATIVE PSEUDO-LABEL REFINEMENT")
print("=" * 72)

for rnd in range(ROUNDS):
    print(f"\n{'─' * 72}")
    print(f"ROUND {rnd} {'fresh head' if rnd == 0 else 'fine-tuning'}")
    print(f"{'─' * 72}")

    # ──────────────────────────────────────────────────────────────────────────
    # Step 1: Pseudo-labels from current A features
    # ──────────────────────────────────────────────────────────────────────────

    if rnd == 0:
        src = "baseline combined UMAP-10 A portion"
    else:
        src = f"projected A from round {rnd - 1}"

    print(f"  [1] KMeans(50) on {src} …")

    Z_A_current_norm = normalize(Z_A_current, norm="l2").astype(np.float32)

    km50 = KMeans(
        n_clusters=K_OLD,
        n_init=15,
        random_state=SEED,
    ).fit(Z_A_current_norm)

    pseudo_y = km50.labels_

    pl_acc = pseudo_label_acc(pseudo_y)

    print(
        f"      Pseudo-label accuracy on A: {pl_acc:.1%} "
        f"(eval only)"
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Step 2: Confidence scores in pseudo-label source space
    # ──────────────────────────────────────────────────────────────────────────

    cent50_source = km50.cluster_centers_.astype(np.float32)

    dist_A = np.linalg.norm(
        Z_A_current_norm[:, None, :] - cent50_source[None, :, :],
        axis=-1,
    )

    own_dist = dist_A[np.arange(N_A), pseudo_y]

    T = 0.1
    soft_A = np.exp(-dist_A / T)
    soft_A /= soft_A.sum(axis=1, keepdims=True)

    sorted_soft = np.sort(soft_A, axis=1)
    margin = sorted_soft[:, -1] - sorted_soft[:, -2]

    conf_density = 1.0 - (
        (own_dist - own_dist.min()) /
        (own_dist.max() - own_dist.min() + 1e-9)
    )

    conf_margin = (
        (margin - margin.min()) /
        (margin.max() - margin.min() + 1e-9)
    )

    conf = ((conf_density + conf_margin) / 2.0).astype(np.float32)

    p30, p70 = np.percentile(conf, 30), np.percentile(conf, 70)

    tier = np.where(
        conf >= p70,
        2,
        np.where(conf >= p30, 1, 0),
    ).astype(np.int64)

    print(
        f"  [2] Confidence "
        f"C_high={(tier == 2).sum()} "
        f"C_mid={(tier == 1).sum()} "
        f"C_low={(tier == 0).sum()}"
    )

    conf_t = torch.from_numpy(conf.astype(np.float32))
    tier_t = torch.from_numpy(tier.astype(np.int64))
    label_t = torch.from_numpy(pseudo_y.astype(np.int64))

    # ──────────────────────────────────────────────────────────────────────────
    # Step 3: Train / fine-tune projection head
    # ──────────────────────────────────────────────────────────────────────────

    epochs = EPOCHS_0 if rnd == 0 else EPOCHS_R
    lr = LR_0 if rnd == 0 else LR_R

    if rnd == 0:
        head = ProjectionHead(in_dim=X_A.shape[1]).to(DEVICE)

    opt = torch.optim.Adam(
        head.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=epochs,
    )

    print(f"  [3] Training {epochs} epochs lr={lr:.0e} …")

    t0 = time.time()

    for ep in range(epochs):
        head.train()

        a_idx, b_idx = sample_batch(pseudo_y, tier)

        n_a = len(a_idx)
        n_b = len(b_idx)
        n_total = n_a + n_b

        x = torch.cat(
            [
                Xt_A[a_idx],
                Xt_B[b_idx],
            ],
            dim=0,
        ).to(DEVICE)

        pl = torch.cat(
            [
                label_t[a_idx],
                torch.zeros(n_b, dtype=torch.long),
            ],
            dim=0,
        ).to(DEVICE)

        cf = torch.cat(
            [
                conf_t[a_idx],
                torch.zeros(n_b),
            ],
            dim=0,
        ).to(DEVICE)

        ti = torch.cat(
            [
                tier_t[a_idx],
                torch.zeros(n_b, dtype=torch.long),
            ],
            dim=0,
        ).to(DEVICE)

        isA = torch.cat(
            [
                torch.ones(n_a, dtype=torch.bool),
                torch.zeros(n_b, dtype=torch.bool),
            ],
            dim=0,
        )

        bbm = build_bb_mask(b_idx, n_total, n_a)

        z = head(x)

        loss = semi_supcon_loss(
            z=z,
            is_A=isA,
            pseudo_lbl=pl,
            conf=cf,
            tier=ti,
            bb_mask=bbm,
            tau=TAU,
        )

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        sch.step()

        if (ep + 1) % max(1, epochs // 2) == 0:
            print(f"      ep {ep + 1}/{epochs} loss={loss.item():.4f}")

    print(f"      Training time: {time.time() - t0:.1f}s")

    # ──────────────────────────────────────────────────────────────────────────
    # Step 4: Extract projected features
    # ──────────────────────────────────────────────────────────────────────────

    head.eval()

    with torch.no_grad():
        Z_A = head(Xt_A.to(DEVICE)).cpu().numpy().astype(np.float32)
        Z_B = head(Xt_B.to(DEVICE)).cpu().numpy().astype(np.float32)

    Z_A = normalize(Z_A, norm="l2").astype(np.float32)
    Z_B = normalize(Z_B, norm="l2").astype(np.float32)

    # Representation diagnostic on A
    intra, inter = [], []

    for k in range(0, K_OLD, 5):
        mem = Z_A[pseudo_y == k]

        if len(mem) < 2:
            continue

        d = np.linalg.norm(
            mem[:, None, :] - mem[None, :, :],
            axis=-1,
        )

        intra.append(d[np.triu_indices(len(mem), k=1)].mean())

        other = Z_A[pseudo_y != k]

        if len(other):
            inter.append(
                np.linalg.norm(other - mem.mean(axis=0), axis=1).mean()
            )

    if intra and inter:
        ratio = np.mean(inter) / (np.mean(intra) + 1e-9)

        print(
            f"      Representation "
            f"intra={np.mean(intra):.4f} "
            f"inter={np.mean(inter):.4f} "
            f"ratio={ratio:.2f}x"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Step 5a: Plain GCD evaluation on Z_B
    # ──────────────────────────────────────────────────────────────────────────

    print(f"  [5a] Plain GCD evaluation on Z_B …")

    res = gcd_acc(Z_B, f"Round {rnd} SupCon-128")

    all_results.append((f"Round {rnd} plain", *res))

    # ──────────────────────────────────────────────────────────────────────────
    # Step 5b: Soft-constrained KMeans evaluation on Z_A + Z_B
    # ──────────────────────────────────────────────────────────────────────────

    print(f"  [5b] Soft-constrained KMeans on Z_A + Z_B …")

    soft_res = soft_constraint_evaluation(
        Z_A=Z_A,
        Z_B=Z_B,
        pseudo_y=pseudo_y,
        conf=conf,
        tag=f"Round {rnd}",
        lambdas=SOFT_KMEANS_LAMBDAS,
    )

    soft_results_by_round[rnd] = soft_res

    # Update features for next round's pseudo-label generation
    Z_A_current = Z_A


# ──────────────────────────────────────────────────────────────────────────────
# Final summary
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 72)
print("SUMMARY — Plain GCD metrics across rounds")
print("=" * 72)

print(f"  {'Method':<45} {'All':>7} {'Old':>7} {'Novel':>9}")
print("  " + "-" * 70)

best_nov = max(r[3] for r in all_results)
best_all = max(r[1] for r in all_results)

for tag, all_a, old_a, nov_a in all_results:
    marks = []

    if nov_a == best_nov:
        marks.append("Novel◄")

    if all_a == best_all:
        marks.append("All◄")

    mark = "  " + "/".join(marks) if marks else ""

    print(
        f"  {tag:<45} "
        f"{all_a:>7.1%} {old_a:>7.1%} {nov_a:>9.1%}{mark}"
    )

print()
print("  Δ vs baseline, best plain round by All:")

best_plain = max(all_results[1:], key=lambda x: x[1])

for metric, idx in [("All", 1), ("Old", 2), ("Novel", 3)]:
    delta = best_plain[idx] - all_results[0][idx]

    print(
        f"    {metric}: "
        f"{all_results[0][idx]:.1%} → {best_plain[idx]:.1%} "
        f"({delta:+.1%})"
    )


print("\n" + "=" * 90)
print("SUMMARY — Soft-constrained KMeans best results per round")
print("=" * 90)

print(
    f"  {'Round':<8} {'Best method':<22} "
    f"{'All':>7} {'Old':>7} {'Novel':>9} "
    f"{'AUC':>7} {'F1':>7} {'NovelClustACC':>15}"
)

print("  " + "-" * 88)

global_best = None

for rnd, resdict in soft_results_by_round.items():
    best_key = max(resdict.keys(), key=lambda k: resdict[k]["all"])
    best = resdict[best_key]

    if global_best is None or best["all"] > global_best[2]["all"]:
        global_best = (rnd, best_key, best)

    print(
        f"  {rnd:<8} {best_key:<22} "
        f"{best['all']:>7.1%} "
        f"{best['old']:>7.1%} "
        f"{best['novel']:>9.1%} "
        f"{best['det']['auc']:>7.4f} "
        f"{best['det']['f1']:>7.3f} "
        f"{best['novel_cluster_acc']:>15.1%}"
    )

if global_best is not None:
    rnd, key, best = global_best

    print("\n  Global best soft-constrained result by All:")
    print(
        f"    Round {rnd}, {key}: "
        f"All={best['all']:.1%}, "
        f"Old={best['old']:.1%}, "
        f"Novel={best['novel']:.1%}, "
        f"AUC={best['det']['auc']:.4f}, "
        f"F1={best['det']['f1']:.3f}, "
        f"NovelClusterACC={best['novel_cluster_acc']:.1%}"
    )