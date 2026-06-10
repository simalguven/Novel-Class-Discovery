"""
Iterative Pseudo-label Refinement with SoftMatch + Semi-supervised SupCon for GCD
==================================================================================
Includes:
  - SoftMatch confidence weighting on A
  - B-disagreement weighting on A (Option B from analysis)
  - Feature distillation anchor to original DINOv2 geometry
  - Memory bank with confidence-gated enqueue
  - Decaying NA schedule to prevent late-round cluster shattering
"""

import numpy as np, umap, warnings, time
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from scipy.optimize import linear_sum_assignment
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
SEED = 42
K_OLD, K_NEW, K_NOV = 50, 60, 10
N_PER_CLS = 100
K_POS = 10

ORACLE_MODE = "off"

ROUNDS    = 6
EPOCHS_0  = 100
EPOCHS_R  = 50
ITERS_PER_EPOCH = 20
LR_0, LR_R = 3e-4, 1e-4

SM_EMA       = 0.9
SM_WARMUP    = 5
DA_CLAMP_MIN = 0.5
DA_CLAMP_MAX = 2.0

SOFT_LABEL_T = 0.02

W_AA = 1.0
W_BB_SCHEDULE  = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
W_NN_SCHEDULE  = [0.0, 0.3, 0.4, 0.4, 0.4, 0.4]
W_NA_SCHEDULE  = [0.0, 0.2, 0.1, 0.05, 0.0, 0.0]
W_DISTILL      = 2.0

# B-disagreement hyper-parameters
BDIS_K_NEIGHBORS = 15
BDIS_W_FLOOR     = 0.3
BDIS_CONF_QUANT  = 0.5

TAU_SCHEDULE = [0.15, 0.12, 0.10, 0.09, 0.08, 0.07]

rng  = np.random.default_rng(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("mps"  if torch.backends.mps.is_available() else
                       "cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────
print("Loading data …")
embeddings  = np.load("plantnet_dinov2_raw.npy").astype(np.float32)
labels      = np.load("plantnet_labels.npy")
all_classes = np.unique(labels)
counts      = np.array([(labels == c).sum() for c in all_classes])
eligible    = all_classes[counts >= 2 * N_PER_CLS]
chosen_60   = np.sort(rng.choice(eligible, size=K_NEW, replace=False))
base_cls    = chosen_60[:K_OLD]
novel_cls   = chosen_60[K_OLD:]

XA, XB, yAl, yBl = [], [], [], []
for c in base_cls:
    idx = rng.choice(np.where(labels == c)[0], size=2*N_PER_CLS, replace=False)
    XA.append(embeddings[idx[:N_PER_CLS]]); yAl.extend([c]*N_PER_CLS)
    XB.append(embeddings[idx[N_PER_CLS:]]); yBl.extend([c]*N_PER_CLS)
for c in novel_cls:
    idx = rng.choice(np.where(labels == c)[0], size=N_PER_CLS, replace=False)
    XB.append(embeddings[idx]); yBl.extend([c]*N_PER_CLS)

X_A, X_B = np.vstack(XA), np.vstack(XB)
y_A, y_B = np.array(yAl), np.array(yBl)
id2idx     = {c: i for i, c in enumerate(chosen_60)}
y_A_eval   = np.array([id2idx[c] for c in y_A])
y_B_eval   = np.array([id2idx[c] for c in y_B])
is_novel_B = y_B_eval >= K_OLD
N_A, N_B   = len(X_A), len(X_B)
print(f"  A:{N_A}  B:{N_B}  novel(B):{is_novel_B.sum()}")

# ──────────────────────────────────────────────────────────────────────────────
# Baseline UMAP-10 on combined raw features
# ──────────────────────────────────────────────────────────────────────────────
print("\nBaseline: combined UMAP-10 on raw features …")
t0 = time.time()
r_base    = umap.UMAP(n_components=10, n_neighbors=20, min_dist=0.05,
                      metric="cosine", random_state=SEED, verbose=False)
X_AB_base = normalize(
    r_base.fit_transform(np.vstack([X_A, X_B])), norm="l2").astype(np.float32)
X_A_umap, X_B_base = X_AB_base[:N_A], X_AB_base[N_A:]
print(f"  Done in {time.time()-t0:.1f}s")

# ──────────────────────────────────────────────────────────────────────────────
# Eval helpers
# ──────────────────────────────────────────────────────────────────────────────
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
        print(f"  {tag:<45}  All={all_a:.1%}  Old={old_a:.1%}  Novel={nov_a:.1%}")
    return all_a, old_a, nov_a

def pseudo_label_acc(pseudo_y, n_clusters=K_OLD, y_eval=None):
    if y_eval is None: y_eval = y_A_eval
    n_true = len(np.unique(y_eval))
    K = max(n_clusters, n_true)
    mat = np.zeros((K, K), dtype=np.int64)
    for p, t in zip(pseudo_y, y_eval):
        mat[p % K, t] += 1
    r, c = linear_sum_assignment(-mat)
    return mat[r, c].sum() / len(pseudo_y)

res_baseline = gcd_acc(X_B_base, "Baseline UMAP-10 (raw)")

# ──────────────────────────────────────────────────────────────────────────────
# Fixed raw-DINOv2 kNN graph for B
# ──────────────────────────────────────────────────────────────────────────────
print(f"\nPre-computing {K_POS}-NN for B (raw DINOv2) …")
t0   = time.time()
nbrs_B = NearestNeighbors(n_neighbors=K_POS+1, metric="cosine", n_jobs=-1).fit(X_B)
knn_B_raw = nbrs_B.kneighbors(X_B, return_distance=False)[:, 1:]
print(f"  Done in {time.time()-t0:.1f}s")

def build_oracle_knn(y_eval, N):
    knn = np.zeros((N, K_POS), dtype=np.int64)
    for i in range(N):
        same = np.where((y_eval == y_eval[i]) & (np.arange(N) != i))[0]
        knn[i] = rng.choice(same, size=K_POS, replace=(len(same) < K_POS))
    return knn

knn_B_oracle = build_oracle_knn(y_B_eval, N_B) if ORACLE_MODE == "AB" else None

# ──────────────────────────────────────────────────────────────────────────────
# Model — residual projection head + EMA teacher
# ──────────────────────────────────────────────────────────────────────────────
class ProjectionHead(nn.Module):
    def __init__(self, in_dim=768, hidden=512, out_dim=128):
        super().__init__()
        self.skip = nn.Linear(in_dim, out_dim, bias=False)
        self.mlp  = nn.Sequential(
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
        pt.data.mul_(m).add_(ps.data, alpha=1-m)
    for bs, bt in zip(student.buffers(), teacher.buffers()):
        bt.data.copy_(bs.data)

# ──────────────────────────────────────────────────────────────────────────────
# Memory Bank
# ──────────────────────────────────────────────────────────────────────────────
class MemoryBank:
    def __init__(self, size, dim, num_classes, device):
        self.size = size
        self.dim = dim
        self.features = torch.zeros(size, dim, device=device)
        self.labels = torch.zeros(size, dtype=torch.long, device=device)
        self.weights = torch.zeros(size, device=device)
        self.ptr = 0
        self.is_full = False

    @torch.no_grad()
    def enqueue(self, feats, labels, weights):
        b_size = feats.size(0)
        if b_size == 0:
            return
        ptr = int(self.ptr)
        if ptr + b_size > self.size:
            rem = self.size - ptr
            self.features[ptr:] = feats[:rem]
            self.labels[ptr:] = labels[:rem]
            self.weights[ptr:] = weights[:rem]
            self.features[:b_size - rem] = feats[rem:]
            self.labels[:b_size - rem] = labels[rem:]
            self.weights[:b_size - rem] = weights[rem:]
            self.ptr = (ptr + b_size) % self.size
            self.is_full = True
        else:
            self.features[ptr:ptr+b_size] = feats
            self.labels[ptr:ptr+b_size] = labels
            self.weights[ptr:ptr+b_size] = weights
            self.ptr = ptr + b_size
            if self.ptr == self.size:
                self.is_full = True
                self.ptr = 0

    def get_all(self):
        if self.is_full:
            return self.features, self.labels, self.weights
        elif self.ptr > 0:
            return self.features[:self.ptr], self.labels[:self.ptr], self.weights[:self.ptr]
        else:
            return None, None, None

# ──────────────────────────────────────────────────────────────────────────────
# SoftMatch & Prototypes
# ──────────────────────────────────────────────────────────────────────────────
class PrototypeClassifier:
    def __init__(self, n_classes, tau_proto=0.1):
        self.K          = n_classes
        self.tau        = tau_proto
        self.prototypes = None

    @torch.no_grad()
    def update_prototypes(self, encoder, X_all_t, hard_y, device):
        encoder.eval()
        z_all = encoder(X_all_t.to(device))
        protos = torch.zeros(self.K, z_all.size(1), device=device)
        for k in range(self.K):
            mask = (hard_y == k)
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
        p   = torch.exp(sim)
        return p / p.sum(dim=1, keepdim=True).clamp_min(1e-8)

class SoftMatch:
    def __init__(self, n_classes, ema=SM_EMA, lam_dist=2.0):
        self.K       = n_classes
        self.ema     = ema
        self.lam     = lam_dist
        self.mu      = torch.tensor(0.5)
        self.sigma2  = torch.tensor(0.1)
        self.p_model = torch.full((n_classes,), 1.0 / n_classes)
        self.p_targ  = torch.full((n_classes,), 1.0 / n_classes)

    @torch.no_grad()
    def warm_start(self, probs):
        max_p = probs.max(dim=1).values
        self.mu      = max_p.mean().cpu()
        self.sigma2  = (max_p.var(unbiased=False) + 1e-4).cpu()
        self.p_model = probs.mean(dim=0).cpu()

    @torch.no_grad()
    def update(self, probs):
        max_p = probs.max(dim=1).values
        m     = max_p.mean()
        v     = max_p.var(unbiased=False) + 1e-8
        self.mu     = self.ema * self.mu     + (1 - self.ema) * m.cpu()
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
        max_p  = probs.max(dim=1).values.cpu()
        diff   = (self.mu - max_p).clamp(min=0)
        w      = torch.exp(-(diff ** 2) / (self.lam * self.sigma2 + 1e-8))
        return w.to(probs.device)

# ──────────────────────────────────────────────────────────────────────────────
# Multi-term SupCon loss
# ──────────────────────────────────────────────────────────────────────────────
def supcon_loss(z, y_old_batch, w_old_batch,
                hard_y_kn_AB, is_conf_novel, is_conf_old,
                is_A, bb_mask,
                mem_bank_z=None, mem_bank_y=None, mem_bank_w=None,
                tau=0.1, w_bb=1.0, w_nn=0.0, w_na=0.0):
    N = z.size(0)
    eye = torch.eye(N, dtype=torch.bool, device=z.device)
    sim = (z @ z.T) / tau
    mx, _ = sim.max(dim=1, keepdim=True)
    exs   = torch.exp(sim - mx).masked_fill(eye, 0.0)
    denom = exs.sum(dim=1, keepdim=True) + 1e-8
    lp    = (sim - mx) - torch.log(denom)

    is_At    = is_A.to(z.device)
    is_cn    = is_conf_novel.to(z.device)
    is_co    = is_conf_old.to(z.device)

    a_and_old_pos = (is_At | is_co).nonzero(as_tuple=True)[0]
    cn_pos   = is_cn.nonzero(as_tuple=True)[0]
    n_a_o    = a_and_old_pos.numel()
    n_cn     = cn_pos.numel()
    n_a      = is_At.sum().item()

    losses = []

    if n_a_o > 0:
        z_AA = z[a_and_old_pos]
        y_AA = y_old_batch[a_and_old_pos].to(z.device)
        w_AA = w_old_batch[a_and_old_pos].to(z.device)

        if mem_bank_z is not None and mem_bank_z.size(0) > 0:
            z_ALL = torch.cat([z_AA, mem_bank_z], dim=0)
            y_ALL = torch.cat([y_AA, mem_bank_y], dim=0)
            w_ALL = torch.cat([w_AA, mem_bank_w], dim=0)
        else:
            z_ALL = z_AA
            y_ALL = y_AA
            w_ALL = w_AA

        N_AA = z_AA.size(0)
        N_ALL = z_ALL.size(0)

        if N_AA > 0 and N_ALL > 1:
            sim_aa = (z_AA @ z_ALL.T) / tau
            mx_aa, _ = sim_aa.max(dim=1, keepdim=True)
            exs_aa = torch.exp(sim_aa - mx_aa)

            eye_mask = torch.zeros(N_AA, N_ALL, dtype=torch.bool, device=z.device)
            eye_mask[:N_AA, :N_AA].fill_diagonal_(True)
            exs_aa = exs_aa.masked_fill(eye_mask, 0.0)

            denom_aa = exs_aa.sum(dim=1, keepdim=True) + 1e-8
            lp_aa = (sim_aa - mx_aa) - torch.log(denom_aa)

            same_aa = (y_AA.unsqueeze(1) == y_ALL.unsqueeze(0)).float()
            pair_w = w_AA.unsqueeze(1) * w_ALL.unsqueeze(0) * same_aa
            pair_w = pair_w.masked_fill(eye_mask, 0.0)

            aa_sum = pair_w.sum(dim=1)
            has_aa = aa_sum > 0
            if has_aa.any():
                l_aa = -((pair_w * lp_aa).sum(dim=1)[has_aa] / aa_sum[has_aa].clamp_min(1e-8))
                losses.append(W_AA * l_aa.mean())

    bb_w = bb_mask.to(z.device).float()
    bb_w = bb_w.masked_fill(eye, 0.0)
    bb_sum = bb_w.sum(dim=1)
    if (bb_sum > 0).any() and w_bb > 0:
        has = bb_sum > 0
        l_bb = -((bb_w * lp).sum(dim=1)[has] / bb_sum[has].clamp_min(1e-8))
        losses.append(w_bb * l_bb.mean())

    if w_nn > 0 and n_cn > 1:
        y_kn = hard_y_kn_AB.to(z.device)
        nn_w = torch.zeros(N, N, device=z.device)
        same_kn = (y_kn.unsqueeze(0) == y_kn.unsqueeze(1)).float()
        ii, jj  = torch.meshgrid(cn_pos, cn_pos, indexing="ij")
        block = same_kn[ii, jj]
        nn_w[ii, jj] = block
        nn_w = nn_w.masked_fill(eye, 0.0)
        nn_sum = nn_w.sum(dim=1)
        if (nn_sum > 0).any():
            has = nn_sum > 0
            l_nn = -((nn_w * lp).sum(dim=1)[has] / nn_sum[has].clamp_min(1e-8))
            losses.append(w_nn * l_nn.mean())

    if w_na > 0 and n_cn > 0 and n_a > 0:
        z_cn  = z[cn_pos]
        a_pos_only = is_At.nonzero(as_tuple=True)[0]
        z_a   = z[a_pos_only]
        sim_cn_a = (z_cn @ z_a.T) / tau
        l_na = torch.logsumexp(sim_cn_a, dim=1) - np.log(max(n_a, 1))
        losses.append(w_na * l_na.mean())

    if not losses:
        return torch.tensor(0., device=z.device, requires_grad=True)
    return sum(losses)

# ──────────────────────────────────────────────────────────────────────────────
# Batch sampling & clustering tools
# ──────────────────────────────────────────────────────────────────────────────
N_A_PER_CLASS = 6
N_B_SEEDS     = 150

def sample_batch(pseudo_y_A, knn_B_current):
    a_idx = []
    for k in range(K_OLD):
        pool = np.where(pseudo_y_A == k)[0]
        if len(pool):
            n = min(N_A_PER_CLASS, len(pool))
            a_idx.extend(rng.choice(pool, n, replace=False).tolist())

    seeds    = rng.choice(N_B, size=N_B_SEEDS, replace=False)
    partners = knn_B_current[seeds, rng.integers(0, knn_B_current.shape[1], size=N_B_SEEDS)]
    b_idx    = np.unique(np.concatenate([seeds, partners]))

    return np.array(a_idx, dtype=int), np.array(b_idx, dtype=int)

def build_bb_mask(b_idx, N_total, N_a, knn_B_current):
    mask  = torch.zeros(N_total, N_total, dtype=torch.bool)
    b_set = {int(bi): pos + N_a for pos, bi in enumerate(b_idx)}
    for pos_i, bi in enumerate(b_idx):
        pi = pos_i + N_a
        for kj in knn_B_current[int(bi)]:
            if int(kj) in b_set:
                pj = b_set[int(kj)]
                mask[pi, pj] = True
                mask[pj, pi] = True
    return mask

def make_pseudo_labels(Z_A, target_max_p=0.7):
    Z_n = normalize(Z_A, norm="l2").astype(np.float32)
    km  = KMeans(n_clusters=K_OLD, n_init=15, random_state=SEED).fit(Z_n)
    centers = normalize(km.cluster_centers_, norm="l2").astype(np.float32)
    sim     = (Z_n @ centers.T).astype(np.float32)
    hard    = sim.argmax(axis=1)

    sim_sorted = np.sort(sim, axis=1)
    margin = sim_sorted[:, -1] - sim_sorted[:, -2]
    lo = np.quantile(margin, 0.05)
    hi = np.quantile(margin, 0.95)
    margin_w = np.clip((margin - lo) / max(hi - lo, 1e-8), 0.0, 1.0).astype(np.float32)

    best_T, best_ap, best_p = None, None, None
    for T in np.geomspace(1e-3, 0.5, 60):
        s = sim / T
        s = s - s.max(axis=1, keepdims=True)
        p = np.exp(s); p /= p.sum(axis=1, keepdims=True) + 1e-8
        ap = p.max(axis=1).mean()
        if best_T is None or abs(ap - target_max_p) < abs(best_ap - target_max_p):
            best_T, best_ap, best_p = T, ap, p
    return hard, best_p.astype(np.float32), centers, float(best_T), float(best_ap), margin_w

def make_constrained_joint_labels(Z_anchor_A, Z_anchor_B, hard_y_A,
                                  novel_quantile=0.75, target_max_p=0.7,
                                  n_em_iters=20, verbose=True):
    K_NOV_LOC = K_NEW - K_OLD
    Z_A = normalize(Z_anchor_A, norm="l2").astype(np.float32)
    Z_B = normalize(Z_anchor_B, norm="l2").astype(np.float32)

    old_protos = np.zeros((K_OLD, Z_A.shape[1]), dtype=np.float32)
    for k in range(K_OLD):
        mem = Z_A[hard_y_A == k]
        if len(mem):
            old_protos[k] = mem.mean(axis=0)
    old_protos = normalize(old_protos, norm="l2").astype(np.float32)

    sim_B_to_old = Z_B @ old_protos.T
    max_sim_B    = sim_B_to_old.max(axis=1)
    threshold    = np.quantile(max_sim_B, 1 - novel_quantile)
    novel_mask_B = max_sim_B < threshold

    if n_novel_cand := int(novel_mask_B.sum()):
        pass

    if n_novel_cand >= K_NOV_LOC:
        km_nov = KMeans(n_clusters=K_NOV_LOC, n_init=15, random_state=SEED).fit(Z_B[novel_mask_B])
        novel_protos = normalize(km_nov.cluster_centers_, norm="l2").astype(np.float32)
    else:
        idx = rng.choice(N_B, size=K_NOV_LOC, replace=False)
        novel_protos = Z_B[idx].copy()

    Z_AB = np.vstack([Z_A, Z_B]).astype(np.float32)
    centers = np.vstack([old_protos, novel_protos]).astype(np.float32)

    for it in range(n_em_iters):
        sim    = Z_AB @ centers.T
        labels = sim.argmax(axis=1)
        for k in range(K_OLD, K_NEW):
            mem = Z_AB[labels == k]
            if len(mem):
                centers[k] = mem.mean(axis=0)
                centers[k] /= np.linalg.norm(centers[k]) + 1e-8

    sim     = Z_AB @ centers.T
    hard_AB = sim.argmax(axis=1).astype(np.int64)

    hard_B          = hard_AB[N_A:]
    crit1_novel_B   = hard_B >= K_OLD
    confident_novel_B = crit1_novel_B & novel_mask_B

    crit1_old_B = hard_B < K_OLD
    crit2_old_B = max_sim_B > np.quantile(max_sim_B, 0.50)
    confident_old_B = crit1_old_B & crit2_old_B

    best_T, best_ap, best_p = None, None, None
    for T in np.geomspace(1e-3, 0.5, 60):
        s = sim / T
        s = s - s.max(axis=1, keepdims=True)
        p = np.exp(s); p /= p.sum(axis=1, keepdims=True) + 1e-8
        ap = p.max(axis=1).mean()
        if best_T is None or abs(ap - target_max_p) < abs(best_ap - target_max_p):
            best_T, best_ap, best_p = T, ap, p

    return (hard_AB, best_p.astype(np.float32), confident_novel_B, confident_old_B,
            float(best_T), float(best_ap))

# ──────────────────────────────────────────────────────────────────────────────
# B-disagreement weighting helpers (Option B)
# ──────────────────────────────────────────────────────────────────────────────
def b_disagreement_weights(Z_A_space, Z_B_space, hard_y_A, hard_kn_B,
                           confident_old_B_mask, k_neighbors=15, w_floor=0.3):
    """For each A sample, find k-NN among confident-old B in shared embedding space.
    Compute fraction of neighbors whose pseudo-label agrees with A's KMeans label.
    Return per-A weight in [w_floor, 1.0]."""
    n_conf = int(confident_old_B_mask.sum())
    if n_conf < k_neighbors:
        return np.ones(len(Z_A_space), dtype=np.float32)

    Z_A_n = normalize(Z_A_space, norm="l2").astype(np.float32)
    Z_B_conf = normalize(Z_B_space[confident_old_B_mask], norm="l2").astype(np.float32)
    y_B_conf = hard_kn_B[confident_old_B_mask]

    nbrs = NearestNeighbors(n_neighbors=k_neighbors, metric="cosine",
                            n_jobs=-1).fit(Z_B_conf)
    nbr_idx = nbrs.kneighbors(Z_A_n, return_distance=False)

    nbr_labels = y_B_conf[nbr_idx]
    agree = (nbr_labels == hard_y_A[:, None]).mean(axis=1)
    w = np.maximum(agree, w_floor).astype(np.float32)
    return w

def bootstrap_confident_old_B_round0(X_B_raw, hard_y_A, X_A_raw, conf_quantile=0.5):
    """Round 0 only: assign each B sample to nearest A-cluster-center in raw DINOv2.
    Mark as confident-old if similarity exceeds conf_quantile of all B max-sims."""
    X_B_n = normalize(X_B_raw, norm="l2").astype(np.float32)
    X_A_n = normalize(X_A_raw, norm="l2").astype(np.float32)
    centers_A_raw = np.zeros((K_OLD, X_B_raw.shape[1]), dtype=np.float32)
    for k in range(K_OLD):
        mem = X_A_n[hard_y_A == k]
        if len(mem):
            centers_A_raw[k] = mem.mean(axis=0)
    centers_A_raw = normalize(centers_A_raw, norm="l2")

    sim_B = X_B_n @ centers_A_raw.T
    max_sim_B = sim_B.max(axis=1)
    hard_B_pred = sim_B.argmax(axis=1)
    threshold = np.quantile(max_sim_B, conf_quantile)
    conf_old_mask = max_sim_B > threshold
    return hard_B_pred.astype(np.int64), conf_old_mask

# ──────────────────────────────────────────────────────────────────────────────
# Iterative training loop
# ──────────────────────────────────────────────────────────────────────────────
Xt_A = torch.from_numpy(X_A)
Xt_B = torch.from_numpy(X_B)

head            = None
teacher         = None
initial_teacher = None
Z_A_current     = X_A_umap
knn_B_curr      = knn_B_oracle if ORACLE_MODE == "AB" else knn_B_raw
all_results     = [("Baseline (raw UMAP-10)", *res_baseline)]

hard_kn_t_frozen     = None
soft_kn_t_frozen     = None
sm_kn_frozen         = None
conf_novel_B_frozen  = None
conf_old_B_frozen    = None

# Freeze A-side pseudo-labels after Round 1 to prevent label-space rotation
hard_y_frozen   = None
bdis_w_A_frozen = None
FREEZE_AFTER_ROUND = 1

# Persistent memory bank across rounds (provides continuity for AA term)
mem_bank = None

print("\n" + "="*72)
print("ITERATIVE PSEUDO-LABEL REFINEMENT (SoftMatch + B-disagreement + Distill)")
print("="*72)


for rnd in range(ROUNDS):
    print(f"\n{'─'*72}")
    print(f"ROUND {rnd}  {'(fresh head)' if rnd==0 else '(fine-tuning)'}")

    src = "baseline UMAP-10 (A-portion)" if rnd == 0 else f"projected A from round {rnd-1}"

    # ── A-side pseudo-labels (frozen after Round FREEZE_AFTER_ROUND) ─────────
    # We always recompute soft_p / centers / margin_w_A from current Z_A
    # because SoftMatch warm_start needs them in current geometry.
    # Only hard_y and bdis_w_A get frozen.
    hard_y_new, soft_p, centers, used_T, ach_max, margin_w_A = make_pseudo_labels(
        Z_A_current, target_max_p=0.7)

    if ORACLE_MODE in ("A", "AB"):
        hard_y = y_A_eval.copy().astype(np.int64)
        soft_p = np.zeros((N_A, K_OLD), dtype=np.float32)
        soft_p[np.arange(N_A), hard_y] = 0.95
        soft_p += 0.05 / K_OLD
        soft_p /= soft_p.sum(axis=1, keepdims=True)
        margin_w_A = np.ones(N_A, dtype=np.float32)
    elif hard_y_frozen is not None:
        hard_y = hard_y_frozen
        print(f"  Using frozen hard_y from round {FREEZE_AFTER_ROUND}")
    else:
        hard_y = hard_y_new

    # ── B-disagreement weighting on A ────────────────────────────────────────
    if ORACLE_MODE in ("A", "AB"):
        bdis_w_A = np.ones(N_A, dtype=np.float32)
    elif bdis_w_A_frozen is not None:
        bdis_w_A = bdis_w_A_frozen
    elif rnd == 0:
        hard_kn_B_r0, conf_old_B_r0 = bootstrap_confident_old_B_round0(
            X_B, hard_y, X_A, conf_quantile=BDIS_CONF_QUANT)
        bdis_w_A = b_disagreement_weights(
            X_A, X_B, hard_y, hard_kn_B_r0, conf_old_B_r0,
            k_neighbors=BDIS_K_NEIGHBORS, w_floor=BDIS_W_FLOOR)
    else:
        teacher.eval()
        with torch.no_grad():
            Z_A_teacher = teacher(Xt_A.to(DEVICE)).cpu().numpy()
            Z_B_teacher = teacher(Xt_B.to(DEVICE)).cpu().numpy()
        if hard_kn_t_frozen is not None and conf_old_B_frozen is not None:
            hard_kn_B_full = hard_kn_t_frozen[N_A:].numpy().astype(np.int64)
            bdis_w_A = b_disagreement_weights(
                Z_A_teacher, Z_B_teacher, hard_y, hard_kn_B_full,
                conf_old_B_frozen,
                k_neighbors=BDIS_K_NEIGHBORS, w_floor=BDIS_W_FLOOR)
        else:
            hard_kn_B_r0, conf_old_B_r0 = bootstrap_confident_old_B_round0(
                X_B, hard_y, X_A, conf_quantile=BDIS_CONF_QUANT)
            bdis_w_A = b_disagreement_weights(
                Z_A_teacher, Z_B_teacher, hard_y, hard_kn_B_r0, conf_old_B_r0,
                k_neighbors=BDIS_K_NEIGHBORS, w_floor=BDIS_W_FLOOR)

    n_low = int((bdis_w_A < 0.5).sum())
    print(f"  B-disagreement: mean_w={bdis_w_A.mean():.3f}  "
          f"N_low(<0.5)={n_low}/{N_A}")

    soft_p_t = torch.from_numpy(soft_p)
    hard_y_t = torch.from_numpy(hard_y.astype(np.int64))

    sm = SoftMatch(n_classes=K_OLD, ema=SM_EMA, lam_dist=2.0)
    sm.warm_start(soft_p_t)
    proto_clf = PrototypeClassifier(n_classes=K_OLD, tau_proto=0.07)

    w_nn = W_NN_SCHEDULE[min(rnd, len(W_NN_SCHEDULE)-1)]
    w_na = W_NA_SCHEDULE[min(rnd, len(W_NA_SCHEDULE)-1)]
    use_kn = (w_nn > 0) or (w_na > 0)

    if ORACLE_MODE == "AB":
        w_nn, w_na, use_kn = max(w_nn, 0.4), max(w_na, 0.3), True
        hard_kn = np.concatenate([y_A_eval, y_B_eval]).astype(np.int64)
        soft_kn = np.zeros((N_A + N_B, K_NEW), dtype=np.float32)
        soft_kn[np.arange(N_A + N_B), hard_kn] = 0.95
        soft_kn += 0.05 / K_NEW
        soft_kn /= soft_kn.sum(axis=1, keepdims=True)
        soft_kn_t = torch.from_numpy(soft_kn)
        hard_kn_t = torch.from_numpy(hard_kn)
        confident_novel_B = is_novel_B.copy()
        confident_old_B   = ~is_novel_B.copy()
        sm_kn = SoftMatch(n_classes=K_NEW, ema=SM_EMA, lam_dist=2.0)
        sm_kn.warm_start(soft_kn_t)

    elif use_kn and hard_kn_t_frozen is not None:
        hard_kn_t          = hard_kn_t_frozen
        soft_kn_t          = soft_kn_t_frozen
        sm_kn              = sm_kn_frozen
        confident_novel_B  = conf_novel_B_frozen
        confident_old_B    = conf_old_B_frozen

    elif use_kn:
        anchor_A = np.concatenate([normalize(Z_A_proj_prev, norm="l2"),
                                   normalize(X_A_umap, norm="l2")], axis=1).astype(np.float32)
        anchor_B = np.concatenate([normalize(Z_B_proj_prev, norm="l2"),
                                   normalize(X_B_base, norm="l2")], axis=1).astype(np.float32)

        (hard_kn, soft_kn, confident_novel_B, confident_old_B, T_kn, ap_kn) = \
            make_constrained_joint_labels(anchor_A, anchor_B, hard_y,
                                          novel_quantile=0.83, target_max_p=0.7)

        soft_kn_t  = torch.from_numpy(soft_kn)
        hard_kn_t  = torch.from_numpy(hard_kn.astype(np.int64))
        sm_kn = SoftMatch(n_classes=K_NEW, ema=SM_EMA, lam_dist=2.0)
        sm_kn.warm_start(soft_kn_t)

        hard_kn_t_frozen     = hard_kn_t
        soft_kn_t_frozen     = soft_kn_t
        sm_kn_frozen         = sm_kn
        conf_novel_B_frozen  = confident_novel_B
        conf_old_B_frozen    = confident_old_B
    else:
        soft_kn_t          = None
        hard_kn_t          = None
        sm_kn              = None
        confident_novel_B  = np.zeros(N_B, dtype=bool)
        confident_old_B    = np.zeros(N_B, dtype=bool)

    epochs = EPOCHS_0 if rnd == 0 else EPOCHS_R
    lr     = LR_0     if rnd == 0 else LR_R
    tau    = TAU_SCHEDULE[min(rnd, len(TAU_SCHEDULE)-1)]
    w_bb   = W_BB_SCHEDULE[min(rnd, len(W_BB_SCHEDULE)-1)]

    if rnd == 0:
        head    = ProjectionHead().to(DEVICE)
        teacher = ProjectionHead().to(DEVICE)
        teacher.load_state_dict(head.state_dict())
        for p in teacher.parameters(): p.requires_grad_(False)

        initial_teacher = ProjectionHead().to(DEVICE)
        initial_teacher.load_state_dict(head.state_dict())
        for p in initial_teacher.parameters(): p.requires_grad_(False)
        initial_teacher.eval()

        # Initialize the persistent memory bank once at Round 0
        mem_bank = MemoryBank(size=2048, dim=128, num_classes=K_OLD, device=DEVICE)

    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = epochs * ITERS_PER_EPOCH
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    t0 = time.time()
    for ep in range(epochs):
        hard_y_tensor = torch.from_numpy(hard_y.astype(np.int64))
        proto_clf.update_prototypes(teacher, Xt_A, hard_y_tensor, DEVICE)
        teacher.eval()
        with torch.no_grad():
            z_A_all    = teacher(Xt_A.to(DEVICE))
            p_live_all = proto_clf.predict(z_A_all)
            p_aligned_all = sm.align(p_live_all)
        sm.update(p_aligned_all)

        head.train()
        for _ in range(ITERS_PER_EPOCH):
            a_idx, b_idx = sample_batch(hard_y, knn_B_curr)
            n_a, n_b = len(a_idx), len(b_idx)
            n_tot = n_a + n_b

            x_batch = torch.cat([Xt_A[a_idx], Xt_B[b_idx]], dim=0).to(DEVICE)
            is_A    = torch.cat([torch.ones(n_a, dtype=torch.bool),
                                 torch.zeros(n_b, dtype=torch.bool)])
            bbm     = build_bb_mask(b_idx, n_tot, n_a, knn_B_curr)

            hy_batch = hard_y_t[a_idx].to(DEVICE)

            # ── A-side trust weight: SoftMatch × B-disagreement ──────────────
            w_bd = torch.from_numpy(bdis_w_A[a_idx]).to(DEVICE)
            if ORACLE_MODE != "off":
                w_A_batch = torch.ones(n_a, device=DEVICE)
            elif ep < SM_WARMUP:
                w_A_batch = w_bd
            else:
                w_sm = sm.weight(p_aligned_all[a_idx])
                w_A_batch = w_sm * w_bd

            y_old_batch = torch.zeros(n_tot, dtype=torch.long, device=DEVICE)
            y_old_batch[:n_a] = hy_batch
            w_old_batch = torch.zeros(n_tot, device=DEVICE)
            w_old_batch[:n_a] = w_A_batch

            if (w_nn > 0 or w_na > 0) and hard_kn_t is not None:
                kn_idx_AB   = np.concatenate([a_idx, np.array(b_idx) + N_A]).astype(np.int64)
                hy_kn_batch = hard_kn_t[kn_idx_AB].to(DEVICE)

                is_cn_batch = torch.cat([
                    torch.zeros(n_a, dtype=torch.bool),
                    torch.from_numpy(confident_novel_B[b_idx]),
                ]).to(DEVICE)

                is_co_batch = torch.cat([
                    torch.zeros(n_a, dtype=torch.bool),
                    torch.from_numpy(confident_old_B[b_idx]),
                ]).to(DEVICE)

                y_old_batch[n_a:] = hy_kn_batch[n_a:]
                w_old_batch[n_a:] = 1.0
            else:
                hy_kn_batch = torch.zeros(n_tot, dtype=torch.long, device=DEVICE)
                is_cn_batch = torch.zeros(n_tot, dtype=torch.bool, device=DEVICE)
                is_co_batch = torch.zeros(n_tot, dtype=torch.bool, device=DEVICE)

            z    = head(x_batch)
            mb_z, mb_y, mb_w = mem_bank.get_all()

            loss = supcon_loss(z, y_old_batch, w_old_batch,
                               hy_kn_batch, is_cn_batch, is_co_batch,
                               is_A, bbm,
                               mb_z, mb_y, mb_w,
                               tau=tau, w_bb=w_bb, w_nn=w_nn, w_na=w_na)

            # ── Trust-weighted feature distillation ──────────────────────────
            if n_a > 0:
                with torch.no_grad():
                    z_orig_A = initial_teacher(x_batch[:n_a])
                cos_per_sample = (z[:n_a] * z_orig_A).sum(dim=1)
                # Pull each A sample back to DINOv2 geometry proportional to trust
                w_sum = w_A_batch.sum().clamp_min(1e-8)
                l_distill = - (w_A_batch * cos_per_sample).sum() / w_sum
            else:
                l_distill = torch.tensor(0.0, device=DEVICE)

            total_loss = loss + (W_DISTILL * l_distill)

            opt.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            sch.step()
            ema_update(head, teacher, m=0.999)

            # ── Confidence-gated memory bank enqueue ─────────────────────────
            with torch.no_grad():
                z_teacher = teacher(x_batch[:n_a])
                valid_mask = w_A_batch >= 0.5
                if valid_mask.any():
                    mem_bank.enqueue(z_teacher[valid_mask],
                                     hy_batch[valid_mask],
                                     w_A_batch[valid_mask])

    teacher.eval()
    with torch.no_grad():
        Z_A = teacher(Xt_A.to(DEVICE)).cpu().numpy()
        Z_B = teacher(Xt_B.to(DEVICE)).cpu().numpy()

    res_proj = gcd_acc(Z_B, f"Round {rnd} — projected (128-d)", verbose=False)
    all_results.append((f"Round {rnd}  proj", *res_proj))

    Z_B_fused = np.concatenate([
        normalize(Z_B,      norm="l2"),
        normalize(X_B_base, norm="l2"),
    ], axis=1)
    res_fuse = gcd_acc(Z_B_fused, f"Round {rnd} — fused (proj ⊕ baseline)", verbose=False)
    all_results.append((f"Round {rnd}  fuse", *res_fuse))

    # ── Freeze hard_y and bdis_w_A after Round FREEZE_AFTER_ROUND ────────────
    if rnd == FREEZE_AFTER_ROUND and ORACLE_MODE == "off" and hard_y_frozen is None:
        hard_y_frozen = hard_y.copy()
        bdis_w_A_frozen = bdis_w_A.copy()
        print(f"  Frozen hard_y and bdis_w_A from round {rnd} for subsequent rounds")

    Z_A_proj_prev = Z_A.copy()
    Z_B_proj_prev = Z_B.copy()
    Z_A_current = np.concatenate([normalize(Z_A, norm="l2"),
                                  normalize(X_A_umap, norm="l2")], axis=1).astype(np.float32)
    knn_B_curr = knn_B_oracle if ORACLE_MODE == "AB" else knn_B_raw

# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*72)
print("SUMMARY — GCD metrics across rounds")
print("="*72)
print(f"  {'Method':<45} {'All':>7} {'Old':>7} {'Novel':>9}")
print("  " + "-"*70)

best_nov = max(r[3] for r in all_results)
best_all = max(r[1] for r in all_results)
for tag, a, o, n in all_results:
    marks = []
    if n == best_nov: marks.append("Novel◄")
    if a == best_all: marks.append("All◄")
    mark = "  " + "/".join(marks) if marks else ""
    print(f"  {tag:<45} {a:>7.1%} {o:>7.1%} {n:>9.1%}{mark}")
