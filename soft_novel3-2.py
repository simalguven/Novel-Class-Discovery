"""
Hybrid Soft-Novelty Iterative Refinement
=========================================

Goal
----
Reliable behavior when K_NOV is small or large.

Key design
----------
1. Use a soft novelty score q_novel for binary old-vs-novel evidence.
2. Use a hard high-margin confidence gate for class-level positives.
   - q_novel says: "is this sample novel?"
   - margin gate says: "is its assigned novel class trustworthy?"
3. Novel-class SupCon uses only confident novel assignments, but with soft q weights.
4. Novel-vs-A repulsion uses confident novel assignments and linear q, not q^2.
5. Old-B-to-A attraction is weak or disabled in the rare-novel regime.
6. Novel prototypes are found with guided KMeans++ and B-only novel EM.
7. B kNN graph is refreshed after every round.

Ground-truth labels are used only for evaluation and diagnostics.
"""

import warnings
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import umap

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
SEED = 42

K_OLD = 50
K_NOV = 10
K_NEW = K_OLD + K_NOV

# For balanced novelty, use:
# K_OLD = 50
# K_NOV = 50
# K_NEW = K_OLD + K_NOV

N_PER_CLS = 100
K_POS = 10

ROUNDS = 2
EPOCHS_0 = 100
EPOCHS_R = 50
ITERS_PER_EPOCH = 20
LR_0, LR_R = 3e-4, 1e-4

SM_EMA = 0.9
SM_WARMUP = 5
DA_CLAMP_MIN = 0.5
DA_CLAMP_MAX = 2.0

W_AA = 1.0
W_DISTILL = 1.0
TAU_SCHEDULE = [0.15, 0.12, 0.10, 0.09, 0.08, 0.07]

RHO_NOV = K_NOV / K_NEW

W_BB_SCHEDULE = [0.30, 0.30, 0.30, 0.30]

if RHO_NOV < 0.25:
    # Rare-novel regime. Hard version works because it uses high-purity novel positives.
    # Keep the soft score, but do not over-train uncertain novel labels.
    W_NN_SCHEDULE = [0.00, 0.30, 0.30, 0.25]
    W_NA_SCHEDULE = [0.00, 0.15, 0.10, 0.05]
    W_OA_SCHEDULE = [0.00, 0.00, 0.00, 0.00]
else:
    # Balanced/high-novel regime. More soft positives are useful.
    W_NN_SCHEDULE = [0.00, 0.30, 0.35, 0.35]
    W_NA_SCHEDULE = [0.00, 0.20, 0.10, 0.05]
    W_OA_SCHEDULE = [0.00, 0.05, 0.05, 0.03]

TAU_Q_MIN = 0.04
TAU_Q_MAX = 0.10

# Drop the lowest-margin fraction of B assignments from class-level supervision.
# This mimics the hard pipeline's purity filter.
CONF_MARGIN_DROP_Q = 0.20

# Usually keep this disabled for stability. Set to 25 to refresh once inside Round 1.
PSEUDO_REFRESH_EVERY = None

rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
EPS = 1e-8


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────
def l2_normalize_np(x):
    return normalize(x, norm="l2").astype(np.float32)


def softmax_np(sim, temperature):
    s = sim / max(float(temperature), EPS)
    s = s - s.max(axis=1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(axis=1, keepdims=True) + EPS
    return p.astype(np.float32)


def build_knn(feats, k=K_POS):
    feats = l2_normalize_np(feats)
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="cosine", n_jobs=-1).fit(feats)
    return nbrs.kneighbors(feats, return_distance=False)[:, 1:]


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────
print("Loading data …")
embeddings = np.load("plantnet_dinov2_raw.npy").astype(np.float32)
labels = np.load("plantnet_labels.npy")

all_classes = np.unique(labels)
counts = np.array([(labels == c).sum() for c in all_classes])
eligible = all_classes[counts >= 2 * N_PER_CLS]
chosen_classes = np.sort(rng.choice(eligible, size=K_NEW, replace=False))

base_cls = chosen_classes[:K_OLD]
novel_cls = chosen_classes[K_OLD:]

XA, XB, yAl, yBl = [], [], [], []

for c in base_cls:
    idx = rng.choice(np.where(labels == c)[0], size=2 * N_PER_CLS, replace=False)
    XA.append(embeddings[idx[:N_PER_CLS]])
    yAl.extend([c] * N_PER_CLS)
    XB.append(embeddings[idx[N_PER_CLS:]])
    yBl.extend([c] * N_PER_CLS)

for c in novel_cls:
    idx = rng.choice(np.where(labels == c)[0], size=N_PER_CLS, replace=False)
    XB.append(embeddings[idx])
    yBl.extend([c] * N_PER_CLS)

X_A, X_B = np.vstack(XA), np.vstack(XB)
y_A, y_B = np.array(yAl), np.array(yBl)

id2idx = {c: i for i, c in enumerate(chosen_classes)}
y_A_eval = np.array([id2idx[c] for c in y_A])
y_B_eval = np.array([id2idx[c] for c in y_B])
is_novel_B = y_B_eval >= K_OLD

N_A, N_B = len(X_A), len(X_B)
print(f"A: {N_A:,} old-only samples")
print(f"B: {N_B:,} mixed samples | true novel fraction={is_novel_B.mean():.1%}")
print(f"K_OLD={K_OLD}, K_NOV={K_NOV}, K_NEW={K_NEW}, rho_novel={RHO_NOV:.1%}")
print(f"Device: {DEVICE}")


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────
def gcd_acc(feat_B, tag="", n_init=20, verbose=True):
    km = KMeans(n_clusters=K_NEW, n_init=n_init, random_state=SEED)
    preds = km.fit_predict(l2_normalize_np(feat_B))

    mat = np.zeros((K_NEW, K_NEW), dtype=np.int64)
    for t, p in zip(y_B_eval, preds):
        mat[t, p] += 1

    row, col = linear_sum_assignment(-mat)
    p2t = {c: r for r, c in zip(row, col)}
    pm = np.array([p2t.get(p, -1) for p in preds])

    all_a = (pm == y_B_eval).mean()
    old_a = (pm[~is_novel_B] == y_B_eval[~is_novel_B]).mean()
    nov_a = (pm[is_novel_B] == y_B_eval[is_novel_B]).mean()

    if verbose:
        print(f"  {tag:<45}  All={all_a:.1%}  Old={old_a:.1%}  Novel={nov_a:.1%}")
    return all_a, old_a, nov_a


def pseudo_novel_cluster_acc(hard_B, mask, label=""):
    """Diagnostic only. Uses ground truth to measure novel-class pseudo-label purity."""
    if mask.sum() == 0:
        print(f"  {label} pseudo novel-cluster acc: empty mask")
        return None

    true_nov = y_B_eval[mask] - K_OLD
    pred_nov = hard_B[mask] - K_OLD

    valid = (true_nov >= 0) & (true_nov < K_NOV) & (pred_nov >= 0) & (pred_nov < K_NOV)
    if valid.sum() == 0:
        print(f"  {label} pseudo novel-cluster acc: no valid novel pairs")
        return None

    mat = np.zeros((K_NOV, K_NOV), dtype=np.int64)
    for t, p in zip(true_nov[valid], pred_nov[valid]):
        mat[t, p] += 1

    row, col = linear_sum_assignment(-mat)
    acc = mat[row, col].sum() / max(int(valid.sum()), 1)
    print(f"  {label} pseudo novel-cluster acc={acc:.1%} on {int(valid.sum())} samples")
    return acc


# ──────────────────────────────────────────────────────────────────────────────
# Baseline UMAP-10
# ──────────────────────────────────────────────────────────────────────────────
print("Fitting baseline UMAP …")
r_base = umap.UMAP(
    n_components=10,
    n_neighbors=20,
    min_dist=0.05,
    metric="cosine",
    random_state=SEED,
    verbose=False,
)
X_AB_base = l2_normalize_np(r_base.fit_transform(np.vstack([X_A, X_B])))
X_A_umap, X_B_base = X_AB_base[:N_A], X_AB_base[N_A:]

res_baseline = gcd_acc(X_B_base, "Baseline UMAP-10 (raw)")
knn_B_curr = build_knn(X_B, k=K_POS)


# ──────────────────────────────────────────────────────────────────────────────
# Model, EMA teacher, memory bank, SoftMatch
# ──────────────────────────────────────────────────────────────────────────────
class ProjectionHead(nn.Module):
    def __init__(self, in_dim=768, hidden=512, out_dim=128):
        super().__init__()
        self.skip = nn.Linear(in_dim, out_dim, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        z = self.skip(x) + self.alpha * self.mlp(x)
        return F.normalize(z, dim=-1)


@torch.no_grad()
def ema_update(student, teacher, m=0.999):
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(m).add_(ps.data, alpha=1 - m)
    for bs, bt in zip(student.buffers(), teacher.buffers()):
        bt.data.copy_(bs.data)


class MemoryBank:
    def __init__(self, size, dim, num_classes, device):
        self.size = size
        self.dim = dim
        self.num_classes = num_classes
        self.features = torch.zeros(size, dim, device=device)
        self.labels = torch.zeros(size, dtype=torch.long, device=device)
        self.weights = torch.zeros(size, device=device)
        self.ptr = 0
        self.is_full = False

    @torch.no_grad()
    def enqueue(self, feats, labels, weights):
        if feats is None or feats.numel() == 0:
            return

        b_size = feats.size(0)
        if b_size >= self.size:
            feats = feats[-self.size:]
            labels = labels[-self.size:]
            weights = weights[-self.size:]
            b_size = self.size

        ptr = int(self.ptr)
        if ptr + b_size > self.size:
            rem = self.size - ptr
            self.features[ptr:] = feats[:rem]
            self.labels[ptr:] = labels[:rem]
            self.weights[ptr:] = weights[:rem]

            wrap = b_size - rem
            self.features[:wrap] = feats[rem:]
            self.labels[:wrap] = labels[rem:]
            self.weights[:wrap] = weights[rem:]

            self.ptr = wrap
            self.is_full = True
        else:
            self.features[ptr:ptr + b_size] = feats
            self.labels[ptr:ptr + b_size] = labels
            self.weights[ptr:ptr + b_size] = weights
            self.ptr = ptr + b_size
            if self.ptr == self.size:
                self.is_full = True
                self.ptr = 0

    def get_all(self):
        if self.is_full:
            return self.features, self.labels, self.weights
        if self.ptr > 0:
            return self.features[:self.ptr], self.labels[:self.ptr], self.weights[:self.ptr]
        return None, None, None


class PrototypeClassifier:
    def __init__(self, n_classes, tau_proto=0.1):
        self.K = n_classes
        self.tau = tau_proto
        self.prototypes = None

    @torch.no_grad()
    def update_prototypes(self, encoder, X_all_t, hard_y, device):
        encoder.eval()
        z_all = encoder(X_all_t.to(device))
        protos = torch.zeros(self.K, z_all.size(1), device=device)

        for k in range(self.K):
            mask = hard_y == k
            if mask.any():
                protos[k] = z_all[mask].mean(0)

        norms = protos.norm(dim=1, keepdim=True).clamp_min(1e-8)
        self.prototypes = protos / norms

    @torch.no_grad()
    def predict(self, z):
        if self.prototypes is None:
            return torch.full((z.size(0), self.K), 1.0 / self.K, device=z.device)

        sim = z @ self.prototypes.T / self.tau
        sim = sim - sim.max(dim=1, keepdim=True).values
        p = torch.exp(sim)
        return p / p.sum(dim=1, keepdim=True).clamp_min(1e-8)


class SoftMatch:
    def __init__(self, n_classes, ema=SM_EMA, lam_dist=2.0):
        self.K = n_classes
        self.ema = ema
        self.lam = lam_dist
        self.mu = torch.tensor(0.5)
        self.sigma2 = torch.tensor(0.1)
        self.p_model = torch.full((n_classes,), 1.0 / n_classes)
        self.p_targ = torch.full((n_classes,), 1.0 / n_classes)

    @torch.no_grad()
    def warm_start(self, probs):
        max_p = probs.max(dim=1).values
        self.mu = max_p.mean().cpu()
        self.sigma2 = (max_p.var(unbiased=False) + 1e-4).cpu()
        self.p_model = probs.mean(dim=0).cpu()

    @torch.no_grad()
    def update(self, probs):
        max_p = probs.max(dim=1).values
        m = max_p.mean()
        v = max_p.var(unbiased=False) + 1e-8
        self.mu = self.ema * self.mu + (1 - self.ema) * m.cpu()
        self.sigma2 = self.ema * self.sigma2 + (1 - self.ema) * v.cpu()
        self.p_model = self.ema * self.p_model + (1 - self.ema) * probs.mean(dim=0).cpu()

    @torch.no_grad()
    def align(self, probs):
        ratio = (self.p_targ / (self.p_model + 1e-8)).to(probs.device)
        ratio = ratio.clamp(DA_CLAMP_MIN, DA_CLAMP_MAX)
        aligned = probs * ratio.unsqueeze(0)
        return aligned / aligned.sum(dim=1, keepdim=True).clamp_min(1e-8)

    @torch.no_grad()
    def weight(self, probs):
        max_p = probs.max(dim=1).values.cpu()
        diff = (self.mu - max_p).clamp(min=0)
        w = torch.exp(-(diff ** 2) / (self.lam * self.sigma2 + 1e-8))
        return w.to(probs.device)

    @torch.no_grad()
    def get_bias_correction(self, labels):
        labels_cpu = labels.cpu()
        ratio = self.p_targ / (self.p_model + 1e-8)
        ratio = ratio / ratio.max()
        return ratio[labels_cpu].to(labels.device)


# ──────────────────────────────────────────────────────────────────────────────
# Pseudo-label functions
# ──────────────────────────────────────────────────────────────────────────────
def make_pseudo_labels(Z_A, target_max_p=0.7):
    Z_n = l2_normalize_np(Z_A)
    km = KMeans(n_clusters=K_OLD, n_init=15, random_state=SEED).fit(Z_n)
    centers = l2_normalize_np(km.cluster_centers_)

    sim = (Z_n @ centers.T).astype(np.float32)
    hard = sim.argmax(axis=1).astype(np.int64)

    best_T, best_ap, best_p = None, None, None
    for T in np.geomspace(1e-3, 0.5, 60):
        p = softmax_np(sim, T)
        ap = p.max(axis=1).mean()
        if best_T is None or abs(ap - target_max_p) < abs(best_ap - target_max_p):
            best_T, best_ap, best_p = T, ap, p

    return hard, best_p.astype(np.float32), centers, float(best_T), float(best_ap)


def make_constrained_joint_labels(Z_anchor_A, Z_anchor_B, hard_y_A, target_max_p=0.7):
    """
    Hybrid output:
      hard_AB: joint hard labels for A+B
      soft_p: calibrated soft labels for SoftMatch
      q_novel_B: soft binary novelty evidence
      q_old_B: conservative old evidence
      confident_novel_B: high-purity novel class assignments
      confident_old_B: high-purity old class assignments
      T, ap: calibration diagnostics
    """
    K_NOV_LOC = K_NEW - K_OLD
    Z_A = l2_normalize_np(Z_anchor_A)
    Z_B = l2_normalize_np(Z_anchor_B)

    # 1. Old prototypes from A.
    old_protos = np.zeros((K_OLD, Z_A.shape[1]), dtype=np.float32)
    for k in range(K_OLD):
        mem = Z_A[hard_y_A == k]
        if len(mem):
            old_protos[k] = mem.mean(axis=0)
        else:
            old_protos[k] = Z_A[rng.integers(0, len(Z_A))]
    old_protos = l2_normalize_np(old_protos)

    # 2. Guided KMeans++ initialization on B.
    seeds = [old_protos[k].copy() for k in range(K_OLD)]

    sim_to_old = Z_B @ old_protos.T
    dist_sq = 2.0 - 2.0 * sim_to_old.max(axis=1)
    dist_sq = np.clip(dist_sq, 0.0, None)

    for _ in range(K_NOV_LOC):
        probs = dist_sq / (dist_sq.sum() + EPS)
        next_idx = rng.choice(len(Z_B), p=probs)
        next_seed = Z_B[next_idx].copy()
        seeds.append(next_seed)

        dist_to_new = 2.0 - 2.0 * (Z_B @ next_seed)
        dist_sq = np.minimum(dist_sq, np.clip(dist_to_new, 0.0, None))

    init_centers = np.asarray(seeds, dtype=np.float32)

    km_B = KMeans(n_clusters=K_NEW, init=init_centers, n_init=1, random_state=SEED).fit(Z_B)
    centers_B = l2_normalize_np(km_B.cluster_centers_)

    # 3. Hungarian matching to identify old-like B centroids.
    sim_matrix = centers_B @ old_protos.T
    row_ind, _ = linear_sum_assignment(-sim_matrix)
    unmatched_indices = np.setdiff1d(np.arange(K_NEW), row_ind)
    novel_protos = centers_B[unmatched_indices]

    if len(novel_protos) != K_NOV_LOC:
        far_idx = np.argsort(sim_to_old.max(axis=1))[:K_NOV_LOC]
        novel_protos = Z_B[far_idx].copy()

    centers = np.vstack([old_protos, novel_protos]).astype(np.float32)
    centers = l2_normalize_np(centers)

    Z_AB = np.vstack([Z_A, Z_B]).astype(np.float32)

    # 4. B-only novel EM. A never updates novel centers.
    for _ in range(20):
        sim_AB = Z_AB @ centers.T
        labels_AB = sim_AB.argmax(axis=1)
        labels_B = labels_AB[len(Z_A):]

        for k in range(K_OLD, K_NEW):
            mem = Z_B[labels_B == k]
            if len(mem):
                c = mem.mean(axis=0)
                centers[k] = c / (np.linalg.norm(c) + EPS)

    sim_AB = Z_AB @ centers.T
    hard_AB = sim_AB.argmax(axis=1).astype(np.int64)

    # 5. Soft novelty score from relative old-vs-novel evidence.
    sim_B = Z_B @ centers.T
    max_old = sim_B[:, :K_OLD].max(axis=1)
    max_nov = sim_B[:, K_OLD:].max(axis=1)
    diff = max_nov - max_old

    tau_q = float(np.clip(np.std(diff) / 2.0, TAU_Q_MIN, TAU_Q_MAX))
    q_raw = 1.0 / (1.0 + np.exp(-diff / tau_q))
    q_novel_B = q_raw.astype(np.float32)
    q_old_B = ((1.0 - q_raw) ** 2).astype(np.float32)

    # 6. High-margin confidence gates for class-level supervision.
    sim_sorted = np.sort(sim_AB, axis=1)
    margin_all = sim_sorted[:, -1] - sim_sorted[:, -2]
    margin_B = margin_all[len(Z_A):]
    margin_thresh = np.quantile(margin_B, CONF_MARGIN_DROP_Q)
    is_confident_B = margin_B > margin_thresh

    hard_B = hard_AB[len(Z_A):]
    confident_novel_B = (hard_B >= K_OLD) & (q_novel_B >= 0.5) & is_confident_B
    confident_old_B = (hard_B < K_OLD) & (q_old_B >= 0.5) & is_confident_B

    hard_q = q_novel_B >= 0.5
    print(
        f"  q_novel: mean={q_novel_B.mean():.1%} | tau_q={tau_q:.4f} | "
        f"hard novel@0.5={hard_q.mean():.1%}"
    )
    if "is_novel_B" in globals():
        fp = int((hard_q & (~is_novel_B)).sum())
        fn = int(((~hard_q) & is_novel_B).sum())
        print(
            f"  q diagnostic: FP@0.5={fp} FN@0.5={fn} | "
            f"E[q|old]={q_novel_B[~is_novel_B].mean():.3f} | "
            f"E[q|novel]={q_novel_B[is_novel_B].mean():.3f}"
        )
        print(
            f"  conf gates: conf_novel={confident_novel_B.mean():.1%} | "
            f"conf_old={confident_old_B.mean():.1%} | margin_thr={margin_thresh:.4f}"
        )
        pseudo_novel_cluster_acc(hard_B, is_novel_B & (hard_B >= K_OLD), label="all predicted-novel true-novel")
        pseudo_novel_cluster_acc(hard_B, is_novel_B & confident_novel_B, label="confident true-novel")

    # 7. Soft probabilities for SoftMatch warm_start.
    best_T, best_ap, best_p = None, None, None
    for T in np.geomspace(1e-3, 0.5, 60):
        p = softmax_np(sim_AB, T)
        ap = p.max(axis=1).mean()
        if best_T is None or abs(ap - target_max_p) < abs(best_ap - target_max_p):
            best_T, best_ap, best_p = T, ap, p

    return (
        hard_AB,
        best_p.astype(np.float32),
        q_novel_B,
        q_old_B,
        confident_novel_B.astype(bool),
        confident_old_B.astype(bool),
        float(best_T),
        float(best_ap),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Batch sampling and masks
# ──────────────────────────────────────────────────────────────────────────────
def sample_batch(pseudo_y_A, knn_B_current):
    a_idx = []
    for k in range(K_OLD):
        pool = np.where(pseudo_y_A == k)[0]
        if len(pool):
            n = min(6, len(pool))
            a_idx.extend(rng.choice(pool, n, replace=False).tolist())

    seeds = rng.choice(N_B, size=min(150, N_B), replace=False)
    partners = knn_B_current[seeds, rng.integers(0, knn_B_current.shape[1], size=len(seeds))]
    b_idx = np.unique(np.concatenate([seeds, partners]))
    return np.array(a_idx, dtype=int), np.array(b_idx, dtype=int)


def build_bb_mask(b_idx, N_total, N_a, knn_B_current):
    mask = torch.zeros(N_total, N_total, dtype=torch.bool)
    b_set = {int(bi): pos + N_a for pos, bi in enumerate(b_idx)}

    for pos_i, bi in enumerate(b_idx):
        pi = pos_i + N_a
        for kj in knn_B_current[int(bi)]:
            if int(kj) in b_set:
                pj = b_set[int(kj)]
                mask[pi, pj] = True
                mask[pj, pi] = True
    return mask


# ──────────────────────────────────────────────────────────────────────────────
# Hybrid soft contrastive loss
# ──────────────────────────────────────────────────────────────────────────────
def _weighted_logprob_loss(lp, pair_w):
    pair_w = pair_w.clamp_min(0.0)
    pair_sum = pair_w.sum(dim=1)
    has = pair_sum > 0
    if not has.any():
        return None
    loss = -((pair_w * lp).sum(dim=1)[has] / pair_sum[has].clamp_min(1e-8))
    return loss.mean()


def supcon_loss(
    z,
    y_old_batch,
    w_old_batch,
    hard_y_kn_AB,
    q_novel_b_batch,
    q_old_b_batch,
    conf_novel_batch,
    conf_old_batch,
    is_A,
    bb_mask,
    mem_bank_z=None,
    mem_bank_y=None,
    mem_bank_w=None,
    tau=0.1,
    w_bb=1.0,
    w_nn=0.0,
    w_na=0.0,
    w_oa=0.0,
):
    N = z.size(0)
    eye = torch.eye(N, dtype=torch.bool, device=z.device)

    sim = (z @ z.T) / tau
    mx = sim.max(dim=1, keepdim=True).values
    exs = torch.exp(sim - mx).masked_fill(eye, 0.0)
    denom = exs.sum(dim=1, keepdim=True) + 1e-8
    lp = (sim - mx) - torch.log(denom)

    is_At = is_A.to(z.device)
    a_pos = is_At.nonzero(as_tuple=True)[0]
    b_pos = (~is_At).nonzero(as_tuple=True)[0]
    n_a = a_pos.numel()

    q_novel_b_batch = q_novel_b_batch.to(z.device).float().clamp(0.0, 1.0)
    q_old_b_batch = q_old_b_batch.to(z.device).float().clamp(0.0, 1.0)
    conf_novel_batch = conf_novel_batch.to(z.device).bool()
    conf_old_batch = conf_old_batch.to(z.device).bool()

    losses = []

    # AA: A-only old SupCon + high-purity memory bank.
    if n_a > 0:
        z_AA = z[a_pos]
        y_AA = y_old_batch[a_pos].to(z.device)
        w_AA = w_old_batch[a_pos].to(z.device)

        if mem_bank_z is not None and mem_bank_z.size(0) > 0:
            z_ALL = torch.cat([z_AA, mem_bank_z], dim=0)
            y_ALL = torch.cat([y_AA, mem_bank_y], dim=0)
            w_ALL = torch.cat([w_AA, mem_bank_w], dim=0)
        else:
            z_ALL, y_ALL, w_ALL = z_AA, y_AA, w_AA

        N_AA = z_AA.size(0)
        N_ALL = z_ALL.size(0)

        if N_AA > 0 and N_ALL > 1:
            sim_aa = (z_AA @ z_ALL.T) / tau
            mx_aa = sim_aa.max(dim=1, keepdim=True).values
            exs_aa = torch.exp(sim_aa - mx_aa)

            eye_mask = torch.zeros(N_AA, N_ALL, dtype=torch.bool, device=z.device)
            eye_mask[:N_AA, :N_AA].fill_diagonal_(True)
            exs_aa = exs_aa.masked_fill(eye_mask, 0.0)

            denom_aa = exs_aa.sum(dim=1, keepdim=True) + 1e-8
            lp_aa = (sim_aa - mx_aa) - torch.log(denom_aa)

            same_aa = (y_AA.unsqueeze(1) == y_ALL.unsqueeze(0)).float()
            pair_w = w_AA.unsqueeze(1) * w_ALL.unsqueeze(0) * same_aa
            pair_w = pair_w.masked_fill(eye_mask, 0.0)

            l_aa = _weighted_logprob_loss(lp_aa, pair_w)
            if l_aa is not None:
                losses.append(W_AA * l_aa)

    # BB: B-B kNN positives, downweight old/novel boundary pairs.
    if w_bb > 0:
        bb_w = bb_mask.to(z.device).float().masked_fill(eye, 0.0)
        q_i = q_novel_b_batch.unsqueeze(0)
        q_j = q_novel_b_batch.unsqueeze(1)
        same_regime = q_i * q_j + (1.0 - q_i) * (1.0 - q_j)
        bb_w = bb_w * same_regime

        l_bb = _weighted_logprob_loss(lp, bb_w)
        if l_bb is not None:
            losses.append(w_bb * l_bb)

    # NN: class-level novel positives only from confident novel assignments.
    if w_nn > 0 and b_pos.numel() > 1:
        y_kn = hard_y_kn_AB.to(z.device)
        same_kn = (y_kn.unsqueeze(0) == y_kn.unsqueeze(1)).float()
        nov_assigned = (y_kn >= K_OLD).float()

        cn = conf_novel_batch.float()
        q_i = q_novel_b_batch.unsqueeze(0)
        q_j = q_novel_b_batch.unsqueeze(1)
        q_outer = torch.sqrt((q_i * q_j).clamp_min(1e-8))

        b_mask_row = (~is_At).float().unsqueeze(1)
        b_mask_col = (~is_At).float().unsqueeze(0)

        nn_w = (
            same_kn
            * nov_assigned.unsqueeze(0)
            * nov_assigned.unsqueeze(1)
            * cn.unsqueeze(0)
            * cn.unsqueeze(1)
            * q_outer
            * b_mask_row
            * b_mask_col
        )
        nn_w = nn_w.masked_fill(eye, 0.0)

        l_nn = _weighted_logprob_loss(lp, nn_w)
        if l_nn is not None:
            losses.append(w_nn * l_nn)

    # NA: repel only confident novel samples from A, with linear q.
    if w_na > 0 and n_a > 0 and b_pos.numel() > 0:
        q_b = q_novel_b_batch[b_pos]
        cn_b = conf_novel_batch[b_pos]
        active = cn_b & (q_b > 1e-3)
        if active.any():
            z_b = z[b_pos][active]
            z_a = z[a_pos]
            q_eff = q_b[active]

            sim_ba = (z_b @ z_a.T) / tau
            l_na = torch.logsumexp(sim_ba, dim=1) - np.log(max(int(n_a), 1))
            losses.append(w_na * (q_eff * l_na).sum() / q_eff.sum().clamp_min(1e-8))

    # OA: optional weak old-B-to-A attraction from confident old assignments.
    if w_oa > 0 and n_a > 0 and b_pos.numel() > 0:
        q_old_b = q_old_b_batch[b_pos]
        co_b = conf_old_batch[b_pos]
        active = co_b & (q_old_b > 1e-3)
        if active.any():
            z_b = z[b_pos][active]
            z_a = z[a_pos]
            y_a = y_old_batch[a_pos].long()
            w_a = w_old_batch[a_pos].to(z.device).float()
            y_kn = hard_y_kn_AB.to(z.device)
            y_b = y_kn[b_pos][active]
            q_eff = q_old_b[active]

            old_assigned = y_b < K_OLD
            if old_assigned.any():
                z_b_old = z_b[old_assigned]
                y_b_old = y_b[old_assigned]
                q_b_old = q_eff[old_assigned]

                sim_ba = (z_b_old @ z_a.T) / tau
                mx_ba = sim_ba.max(dim=1, keepdim=True).values
                exs_ba = torch.exp(sim_ba - mx_ba)
                denom_ba = exs_ba.sum(dim=1, keepdim=True) + 1e-8
                lp_ba = (sim_ba - mx_ba) - torch.log(denom_ba)

                same = (y_b_old.unsqueeze(1) == y_a.unsqueeze(0)).float()
                pair_w = same * w_a.unsqueeze(0) * q_b_old.unsqueeze(1)

                l_oa = _weighted_logprob_loss(lp_ba, pair_w)
                if l_oa is not None:
                    losses.append(w_oa * l_oa)

    if not losses:
        return torch.tensor(0.0, device=z.device, requires_grad=True)
    return sum(losses)


# ──────────────────────────────────────────────────────────────────────────────
# Iterative training loop
# ──────────────────────────────────────────────────────────────────────────────
Xt_A = torch.from_numpy(X_A)
Xt_B = torch.from_numpy(X_B)

head = None
teacher = None
initial_teacher = None
mem_bank = None

Z_A_current = X_A_umap
Z_A_proj_prev = None
Z_B_proj_prev = None

all_results = [("Baseline (raw UMAP-10)", *res_baseline)]

hard_kn_t_current = None
soft_kn_t_current = None
sm_kn_current = None
q_novel_B_np = np.zeros(N_B, dtype=np.float32)
q_old_B_np = np.zeros(N_B, dtype=np.float32)
conf_novel_B_np = np.zeros(N_B, dtype=bool)
conf_old_B_np = np.zeros(N_B, dtype=bool)

print("\n" + "=" * 80)
print("HYBRID SOFT NOVELTY + HIGH-PURITY CLASS GATES")
print("=" * 80)

for rnd in range(ROUNDS):
    print(f"\n{'─' * 80}")
    print(f"ROUND {rnd}  {'(fresh head)' if rnd == 0 else '(fine-tuning)'}")

    hard_y, soft_p, centers_A, used_T, ach_max = make_pseudo_labels(Z_A_current, target_max_p=0.7)
    print(f"  A pseudo-labels: T={used_T:.4f}, achieved mean max p={ach_max:.3f}")

    soft_p_t = torch.from_numpy(soft_p)
    hard_y_t = torch.from_numpy(hard_y.astype(np.int64))

    sm = SoftMatch(n_classes=K_OLD, ema=SM_EMA, lam_dist=2.0)
    sm.warm_start(soft_p_t)
    proto_clf = PrototypeClassifier(n_classes=K_OLD, tau_proto=0.07)

    w_bb = W_BB_SCHEDULE[min(rnd, len(W_BB_SCHEDULE) - 1)]
    w_nn = W_NN_SCHEDULE[min(rnd, len(W_NN_SCHEDULE) - 1)]
    w_na = W_NA_SCHEDULE[min(rnd, len(W_NA_SCHEDULE) - 1)]
    w_oa = W_OA_SCHEDULE[min(rnd, len(W_OA_SCHEDULE) - 1)]

    use_joint = (rnd > 0) and ((w_nn > 0) or (w_na > 0) or (w_oa > 0))

    if use_joint:
        assert Z_A_proj_prev is not None and Z_B_proj_prev is not None
        (
            hard_kn,
            soft_kn,
            q_novel_B_np,
            q_old_B_np,
            conf_novel_B_np,
            conf_old_B_np,
            T_kn,
            ap_kn,
        ) = make_constrained_joint_labels(
            Z_A_proj_prev,
            Z_B_proj_prev,
            hard_y,
            target_max_p=0.7,
        )

        hard_kn_t_current = torch.from_numpy(hard_kn.astype(np.int64))
        soft_kn_t_current = torch.from_numpy(soft_kn.astype(np.float32))
        sm_kn_current = SoftMatch(n_classes=K_NEW, ema=SM_EMA, lam_dist=2.0)
        sm_kn_current.warm_start(soft_kn_t_current)
        print(f"  Joint labels: T={T_kn:.4f}, achieved mean max p={ap_kn:.3f}")
    else:
        hard_kn_t_current = None
        soft_kn_t_current = None
        sm_kn_current = None
        q_novel_B_np = np.zeros(N_B, dtype=np.float32)
        q_old_B_np = np.zeros(N_B, dtype=np.float32)
        conf_novel_B_np = np.zeros(N_B, dtype=bool)
        conf_old_B_np = np.zeros(N_B, dtype=bool)
        print("  Joint B targets disabled this round.")

    q_novel_B_t = torch.from_numpy(q_novel_B_np.astype(np.float32))
    q_old_B_t = torch.from_numpy(q_old_B_np.astype(np.float32))
    conf_novel_B_t = torch.from_numpy(conf_novel_B_np.astype(bool))
    conf_old_B_t = torch.from_numpy(conf_old_B_np.astype(bool))

    epochs = EPOCHS_0 if rnd == 0 else EPOCHS_R
    lr = LR_0 if rnd == 0 else LR_R
    tau = TAU_SCHEDULE[min(rnd, len(TAU_SCHEDULE) - 1)]

    if rnd == 0:
        head = ProjectionHead().to(DEVICE)
        teacher = ProjectionHead().to(DEVICE)
        teacher.load_state_dict(head.state_dict())
        for p in teacher.parameters():
            p.requires_grad_(False)

        initial_teacher = ProjectionHead().to(DEVICE)
        initial_teacher.load_state_dict(head.state_dict())
        for p in initial_teacher.parameters():
            p.requires_grad_(False)
        initial_teacher.eval()

        mem_bank = MemoryBank(size=1024, dim=128, num_classes=K_OLD, device=DEVICE)

    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = epochs * ITERS_PER_EPOCH
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    t0 = time.time()

    for ep in range(epochs):
        if (
            use_joint
            and PSEUDO_REFRESH_EVERY is not None
            and ep > 0
            and ep % PSEUDO_REFRESH_EVERY == 0
        ):
            teacher.eval()
            with torch.no_grad():
                Z_A_refresh = teacher(Xt_A.to(DEVICE)).cpu().numpy()
                Z_B_refresh = teacher(Xt_B.to(DEVICE)).cpu().numpy()

            (
                hard_kn,
                soft_kn,
                q_novel_B_np,
                q_old_B_np,
                conf_novel_B_np,
                conf_old_B_np,
                T_kn,
                ap_kn,
            ) = make_constrained_joint_labels(Z_A_refresh, Z_B_refresh, hard_y, target_max_p=0.7)

            hard_kn_t_current = torch.from_numpy(hard_kn.astype(np.int64))
            soft_kn_t_current = torch.from_numpy(soft_kn.astype(np.float32))
            q_novel_B_t = torch.from_numpy(q_novel_B_np.astype(np.float32))
            q_old_B_t = torch.from_numpy(q_old_B_np.astype(np.float32))
            conf_novel_B_t = torch.from_numpy(conf_novel_B_np.astype(bool))
            conf_old_B_t = torch.from_numpy(conf_old_B_np.astype(bool))
            knn_B_curr = build_knn(Z_B_refresh, k=K_POS)

        hard_y_tensor = torch.from_numpy(hard_y.astype(np.int64))
        proto_clf.update_prototypes(teacher, Xt_A, hard_y_tensor, DEVICE)

        teacher.eval()
        with torch.no_grad():
            z_A_all = teacher(Xt_A.to(DEVICE))
            p_live_all = proto_clf.predict(z_A_all)
            p_aligned_all = sm.align(p_live_all)
        sm.update(p_aligned_all)

        head.train()
        ep_loss = 0.0

        for _ in range(ITERS_PER_EPOCH):
            a_idx, b_idx = sample_batch(hard_y, knn_B_curr)
            n_a, n_b = len(a_idx), len(b_idx)
            n_tot = n_a + n_b

            x_batch = torch.cat([Xt_A[a_idx], Xt_B[b_idx]], dim=0).to(DEVICE)
            is_A = torch.cat([torch.ones(n_a, dtype=torch.bool), torch.zeros(n_b, dtype=torch.bool)])
            bbm = build_bb_mask(b_idx, n_tot, n_a, knn_B_curr)

            hy_batch = hard_y_t[a_idx].to(DEVICE)

            if ep < SM_WARMUP:
                conf_weight = torch.ones(n_a, device=DEVICE)
                w_A_batch = torch.ones(n_a, device=DEVICE)
            else:
                conf_weight = sm.weight(p_aligned_all[a_idx])
                bias_weight = sm.get_bias_correction(hy_batch).to(DEVICE)
                w_A_batch = conf_weight * bias_weight

            y_old_batch = torch.zeros(n_tot, dtype=torch.long, device=DEVICE)
            y_old_batch[:n_a] = hy_batch
            w_old_batch = torch.zeros(n_tot, device=DEVICE)
            w_old_batch[:n_a] = w_A_batch

            if use_joint and hard_kn_t_current is not None:
                kn_idx_AB = np.concatenate([a_idx, np.array(b_idx) + N_A]).astype(np.int64)
                hy_kn_batch = hard_kn_t_current[kn_idx_AB].to(DEVICE)

                q_b_batch = torch.cat(
                    [torch.zeros(n_a, dtype=torch.float32), q_novel_B_t[b_idx].float()], dim=0
                ).to(DEVICE)
                q_old_b_batch = torch.cat(
                    [torch.zeros(n_a, dtype=torch.float32), q_old_B_t[b_idx].float()], dim=0
                ).to(DEVICE)
                conf_novel_batch = torch.cat(
                    [torch.zeros(n_a, dtype=torch.bool), conf_novel_B_t[b_idx].bool()], dim=0
                ).to(DEVICE)
                conf_old_batch = torch.cat(
                    [torch.zeros(n_a, dtype=torch.bool), conf_old_B_t[b_idx].bool()], dim=0
                ).to(DEVICE)

                y_old_batch[n_a:] = hy_kn_batch[n_a:]
            else:
                hy_kn_batch = torch.zeros(n_tot, dtype=torch.long, device=DEVICE)
                q_b_batch = torch.zeros(n_tot, dtype=torch.float32, device=DEVICE)
                q_old_b_batch = torch.zeros(n_tot, dtype=torch.float32, device=DEVICE)
                conf_novel_batch = torch.zeros(n_tot, dtype=torch.bool, device=DEVICE)
                conf_old_batch = torch.zeros(n_tot, dtype=torch.bool, device=DEVICE)

            z = head(x_batch)
            mb_z, mb_y, mb_w = mem_bank.get_all()

            loss = supcon_loss(
                z=z,
                y_old_batch=y_old_batch,
                w_old_batch=w_old_batch,
                hard_y_kn_AB=hy_kn_batch,
                q_novel_b_batch=q_b_batch,
                q_old_b_batch=q_old_b_batch,
                conf_novel_batch=conf_novel_batch,
                conf_old_batch=conf_old_batch,
                is_A=is_A,
                bb_mask=bbm,
                mem_bank_z=mb_z,
                mem_bank_y=mb_y,
                mem_bank_w=mb_w,
                tau=tau,
                w_bb=w_bb,
                w_nn=w_nn,
                w_na=w_na,
                w_oa=w_oa,
            )

            if n_a > 1:
                with torch.no_grad():
                    z_orig_A = initial_teacher(x_batch[:n_a])
                sim_orig = z_orig_A @ z_orig_A.T
                sim_new = z[:n_a] @ z[:n_a].T
                l_distill = F.mse_loss(sim_new, sim_orig)
            else:
                l_distill = torch.tensor(0.0, device=DEVICE)

            total_loss = loss + W_DISTILL * l_distill

            opt.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            sch.step()
            ema_update(head, teacher, m=0.999)

            ep_loss += float(total_loss.detach().cpu())

            with torch.no_grad():
                z_teacher = teacher(x_batch[:n_a])
                valid_mask = conf_weight >= 0.85
                if valid_mask.any():
                    mem_bank.enqueue(
                        z_teacher[valid_mask],
                        hy_batch[valid_mask],
                        conf_weight[valid_mask],
                    )

        if (ep + 1) % max(1, epochs // 5) == 0 or ep == 0:
            print(
                f"  ep {ep + 1:>3}/{epochs} | "
                f"loss={ep_loss / ITERS_PER_EPOCH:.4f} | "
                f"w_bb={w_bb:.2f} w_nn={w_nn:.2f} w_na={w_na:.2f} w_oa={w_oa:.2f}"
            )

    teacher.eval()
    with torch.no_grad():
        Z_A = teacher(Xt_A.to(DEVICE)).cpu().numpy()
        Z_B = teacher(Xt_B.to(DEVICE)).cpu().numpy()

    res_proj = gcd_acc(Z_B, f"Round {rnd} — projected (128-d)", verbose=True)
    all_results.append((f"Round {rnd}  proj", *res_proj))

    Z_B_norm = l2_normalize_np(Z_B)
    res_pure = gcd_acc(Z_B_norm, f"Round {rnd} — pure normalized", verbose=True)
    all_results.append((f"Round {rnd}  pure", *res_pure))

    Z_A_proj_prev = Z_A.copy()
    Z_B_proj_prev = Z_B.copy()
    Z_A_current = l2_normalize_np(Z_A)

    knn_B_curr = build_knn(Z_B_norm, k=K_POS)

    print(f"  Round {rnd} completed in {(time.time() - t0) / 60.0:.1f} min")


# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SUMMARY — GCD metrics across rounds")
print("=" * 80)
print(f"  {'Method':<45} {'All':>7} {'Old':>7} {'Novel':>9}")
print("  " + "-" * 72)

best_nov = max(r[3] for r in all_results)
best_all = max(r[1] for r in all_results)

for tag, a, o, n in all_results:
    marks = []
    if n == best_nov:
        marks.append("Novel◄")
    if a == best_all:
        marks.append("All◄")
    mark = "  " + "/".join(marks) if marks else ""
    print(f"  {tag:<45} {a:>7.1%} {o:>7.1%} {n:>9.1%}{mark}")

if q_novel_B_np is not None and q_novel_B_np.size:
    hard_q = q_novel_B_np >= 0.5
    print("\nFinal soft novelty diagnostics")
    print(f"  E[q_novel_B]        = {q_novel_B_np.mean():.1%}")
    print(f"  hard novel@0.5      = {hard_q.mean():.1%}")
    print(f"  true novel fraction = {is_novel_B.mean():.1%}")
    print(f"  hard@0.5 FP         = {int((hard_q & (~is_novel_B)).sum())}")
    print(f"  hard@0.5 FN         = {int(((~hard_q) & is_novel_B).sum())}")
    print(f"  conf_novel fraction = {conf_novel_B_np.mean():.1%}")
    print(f"  conf_old fraction   = {conf_old_B_np.mean():.1%}")

print("\nDone.")

