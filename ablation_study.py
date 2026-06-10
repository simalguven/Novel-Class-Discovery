"""
ablation_study.py
=================
Ablation over loss components for the Memory-Bank GCD method.

Row order in the final summary table
──────────────────────────────────────────────────────────────────
  Baselines
  ① UMAP-10            raw UMAP-10 on DINOv2 features, no training
  ② GCD                Vaze et al. CVPR 2022 (adapted to pre-extracted embeddings)
  ③ SimGCD             Wen  et al. ICCV 2023 (adapted to pre-extracted embeddings)

  Round-0 ablations  (each trained from scratch for EPOCHS_0 epochs)
  ④ L_AA  only
  ⑤ L_AA + L_BB
  ⑥ L_AA + L_BB + L_Distill   ← full Round 0

  Round-1 ablations  (fine-tuned from ⑥ checkpoint for EPOCHS_R epochs)
  ⑦ + L_NN
  ⑧ + L_NN + L_NA              ← full model

Adaptation notes for GCD / SimGCD
──────────────────────────────────────────────────────────────────
  Both baselines operate on *pre-extracted* DINOv2 embeddings with no
  image augmentation.  They use ORACLE labels for the old (labeled) split A.
  Our ablation variants (④-⑧) use KMeans pseudo-labels — no oracle access.
  All methods are evaluated identically: K-means on projected B features
  followed by Hungarian-matched ACC (All / Old / Novel).
"""

import copy, warnings
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
# Config  (kept identical to memory_bank_general.py)
# ──────────────────────────────────────────────────────────────────────────────
SEED             = 20
K_OLD, K_NEW     = 50, 80
N_PER_CLS        = 100
K_POS            = 10

EPOCHS_0         = 100      # round-0 epochs (R0 ablations)
EPOCHS_R         = 50       # round-1 fine-tune epochs (R1 ablations)
ITERS_PER_EPOCH  = 20
LR_0, LR_R       = 3e-4, 1e-4

SM_EMA           = 0.9
SM_WARMUP        = 5
DA_CLAMP_MIN     = 0.5
DA_CLAMP_MAX     = 2.0

W_AA             = 1.0
W_BB_VAL         = 0.3      # used when BB is active
W_DISTILL_VAL    = 1.0      # used when Distill is active
W_NN_VAL         = 0.3      # used when NN is active
W_NA_VAL         = 0.2      # used when NA is active

TAU_R0           = 0.15
TAU_R1           = 0.12

# SimGCD hyper-params (adapted; no augmentation)
SIMGCD_SUP_WEIGHT         = 0.35
SIMGCD_MEMAX_WEIGHT       = 2.0
SIMGCD_TEACHER_TEMP       = 0.04
SIMGCD_WARMUP_TEMP        = 0.07
SIMGCD_WARMUP_TEMP_EPOCHS = 30
SIMGCD_EPOCHS             = 100

rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("mps"  if torch.backends.mps.is_available() else
                       "cuda" if torch.cuda.is_available() else "cpu")


def reset_torch_seed(seed=SEED):
    """Reset torch-side stochastic state before an independent run."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_rng_state():
    """Copy the NumPy Generator state used by sampling helpers."""
    return copy.deepcopy(rng.bit_generator.state)


def set_rng_state(state):
    """Restore the NumPy Generator state used by sampling helpers."""
    rng.bit_generator.state = copy.deepcopy(state)


def reset_experiment_state(rng_state=None, seed=SEED):
    """Reset all stochastic state before a controlled experimental branch."""
    reset_torch_seed(seed)
    if rng_state is not None:
        set_rng_state(rng_state)

# ──────────────────────────────────────────────────────────────────────────────
# Data loading  (identical to memory_bank_general.py)
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
id2idx    = {c: i for i, c in enumerate(chosen_60)}
y_A_eval  = np.array([id2idx[c] for c in y_A])   # 0-based indices into chosen_60
y_B_eval  = np.array([id2idx[c] for c in y_B])
is_novel_B = y_B_eval >= K_OLD
N_A, N_B  = len(X_A), len(X_B)

Xt_A = torch.from_numpy(X_A)
Xt_B = torch.from_numpy(X_B)

# Oracle one-hot labels for A (used by GCD and SimGCD baselines)
# y_A_eval already maps to 0..(K_OLD-1) since A only contains old classes
y_A_oracle_t = torch.from_numpy(y_A_eval).long()   # shape (N_A,)

# ── UMAP-10 baseline ──────────────────────────────────────────────────────────
print("Fitting UMAP baseline …")
r_base    = umap.UMAP(n_components=10, n_neighbors=20, min_dist=0.05,
                      metric="cosine", random_state=SEED, verbose=False)
X_AB_base = normalize(
    r_base.fit_transform(np.vstack([X_A, X_B])), norm="l2").astype(np.float32)
X_A_umap, X_B_base = X_AB_base[:N_A], X_AB_base[N_A:]

# ── KNN graph on raw B (used for bb_mask in our ablations) ───────────────────
nbrs_B    = NearestNeighbors(n_neighbors=K_POS+1, metric="cosine", n_jobs=-1).fit(X_B)
knn_B_raw = nbrs_B.kneighbors(X_B, return_distance=False)[:, 1:]

# State immediately after data construction.  Restoring this before each R0
# branch prevents baselines or earlier ablation rows from changing mini-batches.
POST_DATA_RNG_STATE = get_rng_state()

# ──────────────────────────────────────────────────────────────────────────────
# Shared model components
# ──────────────────────────────────────────────────────────────────────────────
class ProjectionHead(nn.Module):
    def __init__(self, in_dim=768, hidden=512, out_dim=128):
        super().__init__()
        self.skip = nn.Linear(in_dim, out_dim, bias=False)
        self.mlp  = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        return F.normalize(self.skip(x) + self.alpha * self.mlp(x), dim=-1)


@torch.no_grad()
def ema_update(student, teacher, m=0.999):
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(m).add_(ps.data, alpha=1-m)
    for bs, bt in zip(student.buffers(), teacher.buffers()):
        bt.data.copy_(bs.data)


class MemoryBank:
    def __init__(self, size, dim, device):
        self.size = size; self.dim = dim
        self.features = torch.zeros(size, dim, device=device)
        self.labels   = torch.zeros(size, dtype=torch.long, device=device)
        self.weights  = torch.zeros(size, device=device)
        self.ptr = 0; self.is_full = False

    @torch.no_grad()
    def enqueue(self, feats, labels, weights):
        b = feats.size(0)
        ptr = int(self.ptr)
        if ptr + b > self.size:
            rem = self.size - ptr
            self.features[ptr:]      = feats[:rem]
            self.labels[ptr:]        = labels[:rem]
            self.weights[ptr:]       = weights[:rem]
            self.features[:b - rem]  = feats[rem:]
            self.labels[:b - rem]    = labels[rem:]
            self.weights[:b - rem]   = weights[rem:]
            self.ptr = (ptr + b) % self.size
            self.is_full = True
        else:
            self.features[ptr:ptr+b] = feats
            self.labels[ptr:ptr+b]   = labels
            self.weights[ptr:ptr+b]  = weights
            self.ptr = ptr + b
            if self.ptr == self.size:
                self.is_full = True; self.ptr = 0

    def clone(self):
        other = MemoryBank(self.size, self.dim, self.features.device)
        other.features = self.features.clone()
        other.labels   = self.labels.clone()
        other.weights  = self.weights.clone()
        other.ptr      = int(self.ptr)
        other.is_full  = bool(self.is_full)
        return other

    def get_all(self):
        if self.is_full:
            return self.features, self.labels, self.weights
        elif self.ptr > 0:
            return self.features[:self.ptr], self.labels[:self.ptr], self.weights[:self.ptr]
        return None, None, None


class PrototypeClassifier:
    def __init__(self, n_classes, tau_proto=0.1):
        self.K = n_classes; self.tau = tau_proto; self.prototypes = None

    @torch.no_grad()
    def update_prototypes(self, encoder, X_all_t, hard_y, device):
        encoder.eval()
        z = encoder(X_all_t.to(device))
        protos = torch.zeros(self.K, z.size(1), device=device)
        for k in range(self.K):
            mask = (hard_y == k)
            if mask.any(): protos[k] = z[mask].mean(0)
        norms = protos.norm(dim=1, keepdim=True).clamp_min(1e-8)
        self.prototypes = protos / norms

    @torch.no_grad()
    def predict(self, z):
        if self.prototypes is None:
            return torch.full((z.size(0), self.K), 1.0/self.K, device=z.device)
        sim = z @ self.prototypes.T / self.tau
        sim = sim - sim.max(dim=1, keepdim=True).values
        p   = torch.exp(sim)
        return p / p.sum(dim=1, keepdim=True).clamp_min(1e-8)


class SoftMatch:
    def __init__(self, n_classes, ema=SM_EMA, lam_dist=2.0):
        self.K = n_classes; self.ema = ema; self.lam = lam_dist
        self.mu      = torch.tensor(0.5)
        self.sigma2  = torch.tensor(0.1)
        self.p_model = torch.full((n_classes,), 1.0/n_classes)
        self.p_targ  = torch.full((n_classes,), 1.0/n_classes)

    @torch.no_grad()
    def warm_start(self, probs):
        max_p = probs.max(dim=1).values
        self.mu      = max_p.mean().cpu()
        self.sigma2  = (max_p.var(unbiased=False) + 1e-4).cpu()
        self.p_model = probs.mean(dim=0).cpu()

    @torch.no_grad()
    def update(self, probs):
        max_p = probs.max(dim=1).values
        m, v  = max_p.mean(), max_p.var(unbiased=False) + 1e-8
        self.mu      = self.ema * self.mu      + (1-self.ema)*m.cpu()
        self.sigma2  = self.ema * self.sigma2  + (1-self.ema)*v.cpu()
        self.p_model = self.ema * self.p_model + (1-self.ema)*probs.mean(0).cpu()

    @torch.no_grad()
    def align(self, probs):
        ratio   = (self.p_targ / (self.p_model + 1e-8)).to(probs.device)
        ratio   = ratio.clamp(DA_CLAMP_MIN, DA_CLAMP_MAX)
        aligned = probs * ratio.unsqueeze(0)
        return aligned / aligned.sum(dim=1, keepdim=True).clamp_min(1e-8)

    @torch.no_grad()
    def weight(self, probs):
        max_p = probs.max(dim=1).values.cpu()
        diff  = (self.mu - max_p).clamp(min=0)
        return torch.exp(-(diff**2) / (self.lam*self.sigma2 + 1e-8)).to(probs.device)

    @torch.no_grad()
    def get_bias_correction(self, labels):
        ratio = self.p_targ / (self.p_model + 1e-8)
        ratio = ratio / ratio.max()
        return ratio[labels.cpu()].to(labels.device)

# ──────────────────────────────────────────────────────────────────────────────
# Loss functions
# ──────────────────────────────────────────────────────────────────────────────
def supcon_loss(z, y_old_batch, w_old_batch,
                hard_y_kn_AB, is_conf_novel, is_conf_old,
                is_A, bb_mask,
                mem_bank_z=None, mem_bank_y=None, mem_bank_w=None,
                tau=0.1, w_bb=1.0, w_nn=0.0, w_na=0.0):
    """Full SupCon loss with optional L_AA / L_BB / L_NN / L_NA terms."""
    N   = z.size(0)
    eye = torch.eye(N, dtype=torch.bool, device=z.device)
    sim = (z @ z.T) / tau
    mx, _ = sim.max(dim=1, keepdim=True)
    exs   = torch.exp(sim - mx).masked_fill(eye, 0.0)
    denom = exs.sum(dim=1, keepdim=True) + 1e-8
    lp    = (sim - mx) - torch.log(denom)

    is_At = is_A.to(z.device)
    is_cn = is_conf_novel.to(z.device)
    is_co = is_conf_old.to(z.device)

    a_and_old_pos = (is_At | is_co).nonzero(as_tuple=True)[0]
    cn_pos        = is_cn.nonzero(as_tuple=True)[0]
    n_a_o = a_and_old_pos.numel()
    n_cn  = cn_pos.numel()
    n_a   = is_At.sum().item()

    losses = []

    # ── L_AA: supervised SupCon on A (+ optional memory bank) ────────────────
    if n_a_o > 0:
        z_AA = z[a_and_old_pos]
        y_AA = y_old_batch[a_and_old_pos].to(z.device)
        w_AA = w_old_batch[a_and_old_pos].to(z.device)
        if mem_bank_z is not None and mem_bank_z.size(0) > 0:
            z_ALL = torch.cat([z_AA, mem_bank_z], dim=0)
            y_ALL = torch.cat([y_AA, mem_bank_y], dim=0)
            w_ALL = torch.cat([w_AA, mem_bank_w], dim=0)
        else:
            z_ALL = z_AA; y_ALL = y_AA; w_ALL = w_AA
        N_AA  = z_AA.size(0)
        N_ALL = z_ALL.size(0)
        if N_AA > 0 and N_ALL > 1:
            sim_aa   = (z_AA @ z_ALL.T) / tau
            mx_aa, _ = sim_aa.max(dim=1, keepdim=True)
            exs_aa   = torch.exp(sim_aa - mx_aa)
            eye_mask = torch.zeros(N_AA, N_ALL, dtype=torch.bool, device=z.device)
            eye_mask[:N_AA, :N_AA].fill_diagonal_(True)
            exs_aa   = exs_aa.masked_fill(eye_mask, 0.0)
            denom_aa = exs_aa.sum(dim=1, keepdim=True) + 1e-8
            lp_aa    = (sim_aa - mx_aa) - torch.log(denom_aa)
            same_aa  = (y_AA.unsqueeze(1) == y_ALL.unsqueeze(0)).float()
            pair_w   = w_AA.unsqueeze(1) * w_ALL.unsqueeze(0) * same_aa
            pair_w   = pair_w.masked_fill(eye_mask, 0.0)
            aa_sum   = pair_w.sum(dim=1)
            has_aa   = aa_sum > 0
            if has_aa.any():
                l_aa = -((pair_w * lp_aa).sum(dim=1)[has_aa] / aa_sum[has_aa].clamp_min(1e-8))
                losses.append(W_AA * l_aa.mean())

    # ── L_BB: KNN-based self-supervised contrastive on B ─────────────────────
    bb_w = bb_mask.to(z.device).float().masked_fill(eye, 0.0)
    bb_sum = bb_w.sum(dim=1)
    if (bb_sum > 0).any() and w_bb > 0:
        has = bb_sum > 0
        l_bb = -((bb_w * lp).sum(dim=1)[has] / bb_sum[has].clamp_min(1e-8))
        losses.append(w_bb * l_bb.mean())

    # ── L_NN: supervised SupCon on confident novel B samples ─────────────────
    if w_nn > 0 and n_cn > 1:
        y_kn  = hard_y_kn_AB.to(z.device)
        nn_w  = torch.zeros(N, N, device=z.device)
        same_kn = (y_kn.unsqueeze(0) == y_kn.unsqueeze(1)).float()
        ii, jj  = torch.meshgrid(cn_pos, cn_pos, indexing="ij")
        nn_w[ii, jj] = same_kn[ii, jj]
        nn_w = nn_w.masked_fill(eye, 0.0)
        nn_sum = nn_w.sum(dim=1)
        if (nn_sum > 0).any():
            has   = nn_sum > 0
            l_nn  = -((nn_w * lp).sum(dim=1)[has] / nn_sum[has].clamp_min(1e-8))
            losses.append(w_nn * l_nn.mean())

    # ── L_NA: novel-to-old repulsion ─────────────────────────────────────────
    if w_na > 0 and n_cn > 0 and n_a > 0:
        z_cn  = z[cn_pos]
        a_pos = is_At.nonzero(as_tuple=True)[0]
        z_a   = z[a_pos]
        sim_cn_a = (z_cn @ z_a.T) / tau
        l_na = torch.logsumexp(sim_cn_a, dim=1) - np.log(max(n_a, 1))
        losses.append(w_na * l_na.mean())

    if not losses:
        return torch.tensor(0., device=z.device, requires_grad=True)
    return sum(losses)


# ──────────────────────────────────────────────────────────────────────────────
# Pseudo-label helpers  (identical to memory_bank_general.py)
# ──────────────────────────────────────────────────────────────────────────────
def make_pseudo_labels(Z_A, target_max_p=0.7):
    Z_n  = normalize(Z_A, norm="l2").astype(np.float32)
    km   = KMeans(n_clusters=K_OLD, n_init=15, random_state=SEED).fit(Z_n)
    centers = normalize(km.cluster_centers_, norm="l2").astype(np.float32)
    sim  = (Z_n @ centers.T).astype(np.float32)
    hard = sim.argmax(axis=1)
    best_T, best_ap, best_p = None, None, None
    for T in np.geomspace(1e-3, 0.5, 60):
        s = sim / T; s = s - s.max(axis=1, keepdims=True)
        p = np.exp(s); p /= p.sum(axis=1, keepdims=True) + 1e-8
        ap = p.max(axis=1).mean()
        if best_T is None or abs(ap - target_max_p) < abs(best_ap - target_max_p):
            best_T, best_ap, best_p = T, ap, p
    return hard, best_p.astype(np.float32), centers, float(best_T), float(best_ap)


def make_constrained_joint_labels(Z_anchor_A, Z_anchor_B, hard_y_A, target_max_p=0.7):
    K_NOV_LOC = K_NEW - K_OLD
    Z_A = normalize(Z_anchor_A, norm="l2").astype(np.float32)
    Z_B = normalize(Z_anchor_B, norm="l2").astype(np.float32)
    old_protos = np.zeros((K_OLD, Z_A.shape[1]), dtype=np.float32)
    for k in range(K_OLD):
        mem = Z_A[hard_y_A == k]
        if len(mem): old_protos[k] = mem.mean(axis=0)
    old_protos = normalize(old_protos, norm="l2").astype(np.float32)
    sim_B_old  = Z_B @ old_protos.T
    max_sim_B  = sim_B_old.max(axis=1)
    km_1d = KMeans(n_clusters=2, n_init=10, random_state=SEED).fit(max_sim_B.reshape(-1,1))
    c1, c2 = km_1d.cluster_centers_.flatten()
    low_c, high_c = min(c1,c2), max(c1,c2)
    threshold     = (low_c + high_c) / 2.0
    novel_mask_B  = max_sim_B < threshold
    if int(novel_mask_B.sum()) >= K_NOV_LOC:
        km_nov = KMeans(n_clusters=K_NOV_LOC, n_init=15, random_state=SEED).fit(Z_B[novel_mask_B])
        novel_protos = normalize(km_nov.cluster_centers_, norm="l2").astype(np.float32)
    else:
        idx = rng.choice(len(Z_B), size=K_NOV_LOC, replace=False)
        novel_protos = Z_B[idx].copy()
    Z_AB    = np.vstack([Z_A, Z_B]).astype(np.float32)
    centers = np.vstack([old_protos, novel_protos]).astype(np.float32)
    for _ in range(20):
        sim_it  = Z_AB @ centers.T
        labs_it = sim_it.argmax(axis=1)
        for k in range(K_OLD, K_NEW):
            mem = Z_AB[labs_it == k]
            if len(mem):
                centers[k] = mem.mean(axis=0)
                centers[k] /= np.linalg.norm(centers[k]) + 1e-8
    sim     = Z_AB @ centers.T
    hard_AB = sim.argmax(axis=1).astype(np.int64)
    sim_sorted   = np.sort(sim, axis=1)
    margin_all   = sim_sorted[:,-1] - sim_sorted[:,-2]
    margin_B     = margin_all[len(Z_A):]
    conf_thresh  = np.quantile(margin_B, 0.20)
    is_conf_B    = margin_B > conf_thresh
    hard_B       = hard_AB[len(Z_A):]
    crit1_nov    = hard_B >= K_OLD
    conf_nov_B   = crit1_nov & novel_mask_B & is_conf_B
    safe_old_thr = (threshold + high_c) / 2.0
    crit1_old    = hard_B < K_OLD
    crit2_old    = max_sim_B > safe_old_thr
    conf_old_B   = crit1_old & crit2_old & is_conf_B
    best_T, best_ap, best_p = None, None, None
    for T in np.geomspace(1e-3, 0.5, 60):
        s = sim / T; s = s - s.max(axis=1, keepdims=True)
        p = np.exp(s); p /= p.sum(axis=1, keepdims=True) + 1e-8
        ap = p.max(axis=1).mean()
        if best_T is None or abs(ap - target_max_p) < abs(best_ap - target_max_p):
            best_T, best_ap, best_p = T, ap, p
    return (hard_AB, best_p.astype(np.float32),
            conf_nov_B, conf_old_B, float(best_T), float(best_ap))


# ──────────────────────────────────────────────────────────────────────────────
# Sampling / mask helpers
# ──────────────────────────────────────────────────────────────────────────────
def sample_batch(pseudo_y_A, knn_B_current):
    a_idx = []
    for k in range(K_OLD):
        pool = np.where(pseudo_y_A == k)[0]
        if len(pool):
            n = min(6, len(pool))
            a_idx.extend(rng.choice(pool, n, replace=False).tolist())
    seeds    = rng.choice(N_B, size=150, replace=False)
    partners = knn_B_current[seeds, rng.integers(0, knn_B_current.shape[1], size=150)]
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
                mask[pi, pj] = True; mask[pj, pi] = True
    return mask


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation  (identical to memory_bank_general.py)
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
        print(f"  {tag:<52}  All={all_a:.1%}  Old={old_a:.1%}  Novel={nov_a:.1%}")
    return all_a, old_a, nov_a


# ──────────────────────────────────────────────────────────────────────────────
# ① UMAP baseline
# ──────────────────────────────────────────────────────────────────────────────
def run_umap_baseline():
    return gcd_acc(X_B_base, "① UMAP-10 baseline (raw, no training)")


# ──────────────────────────────────────────────────────────────────────────────
# ② GCD baseline  (Vaze et al. CVPR 2022, adapted)
#
# Adaptation: no image augmentation available → we omit the self-supervised
# view-based contrastive term and keep only the supervised SupCon on A
# using ORACLE labels.  This is the most direct adaptation of GCD's
# labeled-data supervision signal to our pre-extracted embedding setting.
# Evaluation: K-means on projected B (same as original GCD pipeline).
# ──────────────────────────────────────────────────────────────────────────────
def run_gcd():
    """GCD: oracle SupCon on A, no self-sup on B (no augmentation), KMeans on B."""
    print("\n── GCD baseline ──────────────────────────────────────────────────────")
    reset_experiment_state(POST_DATA_RNG_STATE)
    head    = ProjectionHead().to(DEVICE)
    teacher = ProjectionHead().to(DEVICE)
    teacher.load_state_dict(head.state_dict())
    for p in teacher.parameters(): p.requires_grad_(False)

    # Use oracle labels for A (GCD has access to labeled set)
    y_oracle = y_A_oracle_t  # shape (N_A,), values in [0, K_OLD)

    opt = torch.optim.Adam(head.parameters(), lr=LR_0, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_0*ITERS_PER_EPOCH)

    for ep in range(EPOCHS_0):
        head.train()
        for _ in range(ITERS_PER_EPOCH):
            # Sample a mini-batch from A
            batch_idx_a = rng.choice(N_A, size=min(200, N_A), replace=False)
            x_a = Xt_A[batch_idx_a].to(DEVICE)
            y_a = y_oracle[batch_idx_a].to(DEVICE)
            n_a = len(batch_idx_a)

            z_a = head(x_a)    # (n_a, 128)

            # Supervised SupCon on A with oracle labels
            sim  = (z_a @ z_a.T) / TAU_R0
            eye  = torch.eye(n_a, dtype=torch.bool, device=DEVICE)
            sim  = sim.masked_fill(eye, float('-inf'))
            lp   = F.log_softmax(sim, dim=-1)

            same = (y_a.unsqueeze(0) == y_a.unsqueeze(1))
            same.fill_diagonal_(False)
            pos_count = same.sum(dim=1).clamp_min(1).float()
            loss = -(same.float() * lp).sum(dim=1) / pos_count
            loss = loss[same.any(dim=1)].mean()

            if not torch.isfinite(loss):
                continue

            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step(); sch.step()
            ema_update(head, teacher)

    teacher.eval()
    with torch.no_grad():
        Z_B = teacher(Xt_B.to(DEVICE)).cpu().numpy()

    return gcd_acc(normalize(Z_B, norm="l2"), "② GCD (Vaze et al. CVPR 2022, adapted)")


# ──────────────────────────────────────────────────────────────────────────────
# ③ SimGCD baseline  (Wen et al. ICCV 2023, adapted)
#
# Adaptation: no image augmentation → omit InfoNCE view-contrastive.
# We keep the three components that do not require two views:
#   • Supervised CE on A (oracle labels, old classes only)
#   • Supervised SupCon on A (oracle labels)
#   • Teacher-student distillation on ALL samples (EMA teacher)
#   • MeMax entropy regularisation on all samples
# A linear parametric head over K_NEW classes sits on top of the projection.
# Evaluation: K-means on projected 128-d features (fair comparison).
# ──────────────────────────────────────────────────────────────────────────────
def run_simgcd():
    """SimGCD: oracle CE + SupCon on A + teacher-student distill on all + memax."""
    import math
    print("\n── SimGCD baseline ───────────────────────────────────────────────────")
    reset_experiment_state(POST_DATA_RNG_STATE)

    head     = ProjectionHead().to(DEVICE)                     # 768 → 128
    clf      = nn.Linear(128, K_NEW, bias=False).to(DEVICE)    # parametric classifier
    teacher_head = copy.deepcopy(head);  teacher_head.eval()
    teacher_clf  = copy.deepcopy(clf);   teacher_clf.eval()
    for p in list(teacher_head.parameters()) + list(teacher_clf.parameters()):
        p.requires_grad_(False)

    # Temperature schedule for teacher (warmup → final)
    temp_schedule = np.concatenate([
        np.linspace(SIMGCD_WARMUP_TEMP, SIMGCD_TEACHER_TEMP, SIMGCD_WARMUP_TEMP_EPOCHS),
        np.ones(SIMGCD_EPOCHS - SIMGCD_WARMUP_TEMP_EPOCHS) * SIMGCD_TEACHER_TEMP
    ])

    y_oracle = y_A_oracle_t  # oracle labels for A

    params = list(head.parameters()) + list(clf.parameters())
    opt = torch.optim.Adam(params, lr=LR_0, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=SIMGCD_EPOCHS*ITERS_PER_EPOCH)

    for ep in range(SIMGCD_EPOCHS):
        t_temp  = float(temp_schedule[ep])
        s_temp  = 0.1   # fixed student temperature (same as original)

        head.train(); clf.train()
        for _ in range(ITERS_PER_EPOCH):
            # ── labeled mini-batch from A ──────────────────────────────────
            idx_a = rng.choice(N_A, size=min(128, N_A), replace=False)
            x_a   = Xt_A[idx_a].to(DEVICE)
            y_a   = y_oracle[idx_a].to(DEVICE)

            # ── unlabeled mini-batch from B ────────────────────────────────
            idx_b = rng.choice(N_B, size=min(128, N_B), replace=False)
            x_b   = Xt_B[idx_b].to(DEVICE)

            x_all = torch.cat([x_a, x_b], dim=0)
            n_a_b, n_b_b = len(idx_a), len(idx_b)

            # Forward
            z_all    = head(x_all)            # (n_a+n_b, 128)  normalised
            logits   = clf(z_all)             # (n_a+n_b, K_NEW)
            z_a, z_b = z_all[:n_a_b], z_all[n_a_b:]
            lg_a = logits[:n_a_b]

            # ── teacher forward (no grad) ──────────────────────────────────
            with torch.no_grad():
                t_z   = teacher_head(x_all)
                t_lg  = teacher_clf(t_z)
                t_prb = F.softmax(t_lg / t_temp, dim=-1)

            # (i) Supervised CE on A (old K_OLD classes only)
            cls_loss = F.cross_entropy(lg_a[:, :K_OLD] / s_temp, y_a)

            # (ii) Supervised SupCon on A (oracle labels)
            n_a_loc = z_a.size(0)
            sim_a   = (z_a @ z_a.T) / TAU_R0
            eye_a   = torch.eye(n_a_loc, dtype=torch.bool, device=DEVICE)
            sim_a   = sim_a.masked_fill(eye_a, float('-inf'))
            lp_a    = F.log_softmax(sim_a, dim=-1)
            same_a  = (y_a.unsqueeze(0) == y_a.unsqueeze(1))
            same_a.fill_diagonal_(False)
            pc_a    = same_a.sum(dim=1).clamp_min(1).float()
            sup_con = -(same_a.float() * lp_a).sum(dim=1) / pc_a
            sup_con = sup_con[same_a.any(dim=1)].mean() if same_a.any() else torch.tensor(0., device=DEVICE)

            # (iii) Teacher-student distillation on ALL samples
            s_log_prb = F.log_softmax(logits / s_temp, dim=-1)
            cluster_loss = -torch.mean(torch.sum(t_prb * s_log_prb, dim=-1))

            # (iv) MeMax entropy regularisation (encourage balanced predictions)
            avg_prbs  = F.softmax(logits / s_temp, dim=-1).mean(dim=0)
            me_max    = -torch.sum(torch.log(avg_prbs ** (-avg_prbs))) + math.log(float(K_NEW))
            cluster_loss = cluster_loss + SIMGCD_MEMAX_WEIGHT * me_max

            loss = (SIMGCD_SUP_WEIGHT * cls_loss
                    + SIMGCD_SUP_WEIGHT * sup_con
                    + (1 - SIMGCD_SUP_WEIGHT) * cluster_loss)

            if not torch.isfinite(loss):
                continue

            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sch.step()
            ema_update(head, teacher_head)
            ema_update(clf,  teacher_clf)

    teacher_head.eval()
    with torch.no_grad():
        Z_B = teacher_head(Xt_B.to(DEVICE)).cpu().numpy()

    return gcd_acc(normalize(Z_B, norm="l2"), "③ SimGCD (Wen et al. ICCV 2023, adapted)")


# ──────────────────────────────────────────────────────────────────────────────
# Round-0 ablation runner
#
# Trains from scratch for EPOCHS_0 epochs.
# Uses KMeans pseudo-labels on A (no oracle) — same as memory_bank_general.py.
# Returns: (all_acc, old_acc, nov_acc), trained head, trained teacher, Z_A, Z_B
# ──────────────────────────────────────────────────────────────────────────────
def run_r0_config(label, w_bb=0.0, w_distill=0.0, rng_state=None):
    print(f"\n── {label} ──────────────────────────────────────────────")
    reset_experiment_state(rng_state if rng_state is not None else POST_DATA_RNG_STATE)

    # Fresh model
    head            = ProjectionHead().to(DEVICE)
    teacher         = ProjectionHead().to(DEVICE)
    teacher.load_state_dict(head.state_dict())
    for p in teacher.parameters(): p.requires_grad_(False)
    initial_teacher = copy.deepcopy(head)
    for p in initial_teacher.parameters(): p.requires_grad_(False)
    initial_teacher.eval()

    mem_bank = MemoryBank(size=1024, dim=128, device=DEVICE)

    # Pseudo-labels from UMAP features (round 0 start)
    hard_y, soft_p, _, _, _ = make_pseudo_labels(X_A_umap, target_max_p=0.7)
    soft_p_t = torch.from_numpy(soft_p)
    hard_y_t = torch.from_numpy(hard_y.astype(np.int64))

    sm       = SoftMatch(n_classes=K_OLD, ema=SM_EMA, lam_dist=2.0)
    sm.warm_start(soft_p_t)
    proto_clf = PrototypeClassifier(n_classes=K_OLD, tau_proto=0.07)

    opt = torch.optim.Adam(head.parameters(), lr=LR_0, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_0*ITERS_PER_EPOCH)

    for ep in range(EPOCHS_0):
        hard_y_tensor = torch.from_numpy(hard_y.astype(np.int64))
        proto_clf.update_prototypes(teacher, Xt_A, hard_y_tensor, DEVICE)
        teacher.eval()
        with torch.no_grad():
            z_A_all    = teacher(Xt_A.to(DEVICE))
            p_live_all = proto_clf.predict(z_A_all)
            p_al_all   = sm.align(p_live_all)
        sm.update(p_al_all)

        head.train()
        for _ in range(ITERS_PER_EPOCH):
            a_idx, b_idx = sample_batch(hard_y, knn_B_raw)
            n_a, n_b   = len(a_idx), len(b_idx)
            n_tot      = n_a + n_b

            x_batch = torch.cat([Xt_A[a_idx], Xt_B[b_idx]], dim=0).to(DEVICE)
            is_A    = torch.cat([torch.ones(n_a, dtype=torch.bool),
                                 torch.zeros(n_b, dtype=torch.bool)])
            bbm     = build_bb_mask(b_idx, n_tot, n_a, knn_B_raw) if w_bb > 0 \
                      else torch.zeros(n_tot, n_tot, dtype=torch.bool)

            hy_batch = hard_y_t[a_idx].to(DEVICE)

            if ep < SM_WARMUP:
                conf_weight = torch.ones(n_a, device=DEVICE)
                w_A_batch   = torch.ones(n_a, device=DEVICE)
            else:
                conf_weight = sm.weight(p_al_all[a_idx])
                bias_weight = sm.get_bias_correction(hy_batch).to(DEVICE)
                w_A_batch   = conf_weight * bias_weight

            y_old_batch = torch.zeros(n_tot, dtype=torch.long, device=DEVICE)
            y_old_batch[:n_a] = hy_batch
            w_old_batch = torch.zeros(n_tot, device=DEVICE)
            w_old_batch[:n_a] = w_A_batch

            z = head(x_batch)
            mb_z, mb_y, mb_w = mem_bank.get_all()

            # No novel/old confident masks in R0
            is_cn = torch.zeros(n_tot, dtype=torch.bool, device=DEVICE)
            is_co = torch.zeros(n_tot, dtype=torch.bool, device=DEVICE)
            hy_kn = torch.zeros(n_tot, dtype=torch.long, device=DEVICE)

            loss = supcon_loss(z, y_old_batch, w_old_batch,
                               hy_kn, is_cn, is_co,
                               is_A, bbm,
                               mb_z, mb_y, mb_w,
                               tau=TAU_R0, w_bb=w_bb, w_nn=0.0, w_na=0.0)

            # Relational Knowledge Distillation (only when w_distill > 0)
            if w_distill > 0 and n_a > 1:
                with torch.no_grad():
                    z_orig_A = initial_teacher(x_batch[:n_a])
                sim_orig  = z_orig_A @ z_orig_A.T
                sim_new   = z[:n_a]  @ z[:n_a].T
                l_distill = F.mse_loss(sim_new, sim_orig)
            else:
                l_distill = torch.tensor(0.0, device=DEVICE)

            total_loss = loss + w_distill * l_distill

            opt.zero_grad(); total_loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step(); sch.step()
            ema_update(head, teacher)

            # High-purity memory bank (only updated when AA loss is active)
            with torch.no_grad():
                z_teacher = teacher(x_batch[:n_a])
                valid_mask = conf_weight >= 0.85
                if valid_mask.any():
                    mem_bank.enqueue(z_teacher[valid_mask],
                                     hy_batch[valid_mask],
                                     conf_weight[valid_mask])

    teacher.eval()
    with torch.no_grad():
        Z_A = teacher(Xt_A.to(DEVICE)).cpu().numpy()
        Z_B = teacher(Xt_B.to(DEVICE)).cpu().numpy()

    res = gcd_acc(normalize(Z_B, norm="l2"), label)
    return res, head, teacher, initial_teacher, mem_bank.clone(), get_rng_state(), Z_A, Z_B


# ──────────────────────────────────────────────────────────────────────────────
# Round-1 ablation runner
#
# Fine-tunes from the full R0 checkpoint (L_AA + L_BB + Distill) adding
# L_NN and/or L_NA.  Uses make_constrained_joint_labels for novel discovery.
# ──────────────────────────────────────────────────────────────────────────────
def run_r1_config(label, w_nn, w_na,
                  r0_head, r0_teacher, r0_initial_teacher, r0_mem_bank,
                  Z_A_prev, Z_B_prev, hard_y_A,
                  knn_B_curr, rng_state=None):
    print(f"\n── {label} ──────────────────────────────────────────────")
    if rng_state is not None:
        set_rng_state(rng_state)

    head    = copy.deepcopy(r0_head)
    teacher = copy.deepcopy(r0_teacher)
    for p in teacher.parameters(): p.requires_grad_(False)

    # Match memory_bank_general.py: R1 RKD still uses the original R0
    # initialization, not the trained R0 checkpoint.
    initial_teacher = copy.deepcopy(r0_initial_teacher)
    for p in initial_teacher.parameters(): p.requires_grad_(False)
    initial_teacher.eval()

    # Match memory_bank_general.py: the high-purity memory bank carries
    # from R0 into R1.  Clone so parallel R1 ablations share the same start.
    mem_bank = r0_mem_bank.clone()

    # Pseudo-labels for A from R0 features
    hard_y, soft_p, _, _, _ = make_pseudo_labels(
        normalize(Z_A_prev, norm="l2"), target_max_p=0.7)
    soft_p_t = torch.from_numpy(soft_p)
    hard_y_t = torch.from_numpy(hard_y.astype(np.int64))

    sm       = SoftMatch(n_classes=K_OLD, ema=SM_EMA, lam_dist=2.0)
    sm.warm_start(soft_p_t)
    proto_clf = PrototypeClassifier(n_classes=K_OLD, tau_proto=0.07)

    # Joint A+B labels for novel supervision (L_NN / L_NA)
    anchor_A = normalize(Z_A_prev, norm="l2").astype(np.float32)
    anchor_B = normalize(Z_B_prev, norm="l2").astype(np.float32)
    (hard_kn, soft_kn,
     conf_nov_B, conf_old_B,
     T_kn, ap_kn) = make_constrained_joint_labels(anchor_A, anchor_B, hard_y_A)

    hard_kn_t = torch.from_numpy(hard_kn.astype(np.int64))
    soft_kn_t = torch.from_numpy(soft_kn)
    sm_kn     = SoftMatch(n_classes=K_NEW, ema=SM_EMA, lam_dist=2.0)
    sm_kn.warm_start(soft_kn_t)

    opt = torch.optim.Adam(head.parameters(), lr=LR_R, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_R*ITERS_PER_EPOCH)

    for ep in range(EPOCHS_R):
        hard_y_tensor = torch.from_numpy(hard_y.astype(np.int64))
        proto_clf.update_prototypes(teacher, Xt_A, hard_y_tensor, DEVICE)
        teacher.eval()
        with torch.no_grad():
            z_A_all    = teacher(Xt_A.to(DEVICE))
            p_live_all = proto_clf.predict(z_A_all)
            p_al_all   = sm.align(p_live_all)
        sm.update(p_al_all)

        head.train()
        for _ in range(ITERS_PER_EPOCH):
            a_idx, b_idx = sample_batch(hard_y, knn_B_curr)
            n_a, n_b   = len(a_idx), len(b_idx)
            n_tot      = n_a + n_b

            x_batch = torch.cat([Xt_A[a_idx], Xt_B[b_idx]], dim=0).to(DEVICE)
            is_A    = torch.cat([torch.ones(n_a, dtype=torch.bool),
                                 torch.zeros(n_b, dtype=torch.bool)])
            bbm     = build_bb_mask(b_idx, n_tot, n_a, knn_B_curr)

            hy_batch = hard_y_t[a_idx].to(DEVICE)

            if ep < SM_WARMUP:
                conf_weight = torch.ones(n_a, device=DEVICE)
                w_A_batch   = torch.ones(n_a, device=DEVICE)
            else:
                conf_weight = sm.weight(p_al_all[a_idx])
                bias_weight = sm.get_bias_correction(hy_batch).to(DEVICE)
                w_A_batch   = conf_weight * bias_weight

            # Joint indices for R1 losses
            kn_idx_AB   = np.concatenate([a_idx, b_idx + N_A]).astype(np.int64)
            hy_kn_batch = hard_kn_t[kn_idx_AB].to(DEVICE)

            is_cn_batch = torch.cat([
                torch.zeros(n_a, dtype=torch.bool),
                torch.from_numpy(conf_nov_B[b_idx]),
            ]).to(DEVICE)
            is_co_batch = torch.cat([
                torch.zeros(n_a, dtype=torch.bool),
                torch.from_numpy(conf_old_B[b_idx]),
            ]).to(DEVICE)

            y_old_batch = torch.zeros(n_tot, dtype=torch.long, device=DEVICE)
            y_old_batch[:n_a] = hy_batch
            y_old_batch[n_a:] = hy_kn_batch[n_a:]

            w_old_batch = torch.zeros(n_tot, device=DEVICE)
            w_old_batch[:n_a] = w_A_batch
            if ep < SM_WARMUP:
                w_old_batch[n_a:] = 1.0
            else:
                bias_w_kn = sm_kn.get_bias_correction(hy_kn_batch[n_a:]).to(DEVICE)
                w_old_batch[n_a:] = bias_w_kn

            z = head(x_batch)
            mb_z, mb_y, mb_w = mem_bank.get_all()

            loss = supcon_loss(z, y_old_batch, w_old_batch,
                               hy_kn_batch, is_cn_batch, is_co_batch,
                               is_A, bbm,
                               mb_z, mb_y, mb_w,
                               tau=TAU_R1, w_bb=W_BB_VAL, w_nn=w_nn, w_na=w_na)

            # Distillation (always on in R1)
            if n_a > 1:
                with torch.no_grad():
                    z_orig_A = initial_teacher(x_batch[:n_a])
                sim_orig  = z_orig_A @ z_orig_A.T
                sim_new   = z[:n_a]  @ z[:n_a].T
                l_distill = F.mse_loss(sim_new, sim_orig)
            else:
                l_distill = torch.tensor(0.0, device=DEVICE)

            total_loss = loss + W_DISTILL_VAL * l_distill

            opt.zero_grad(); total_loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step(); sch.step()
            ema_update(head, teacher)

            with torch.no_grad():
                z_teacher = teacher(x_batch[:n_a])
                valid_mask = conf_weight >= 0.85
                if valid_mask.any():
                    mem_bank.enqueue(z_teacher[valid_mask],
                                     hy_batch[valid_mask],
                                     conf_weight[valid_mask])

    teacher.eval()
    with torch.no_grad():
        Z_B = teacher(Xt_B.to(DEVICE)).cpu().numpy()

    # Update KNN for potential next round (return updated features too)
    Z_B_norm = normalize(Z_B, norm="l2")
    res = gcd_acc(Z_B_norm, label)
    return res


# ──────────────────────────────────────────────────────────────────────────────
# Main execution
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*72)
print("ABLATION STUDY — Loss Component Incremental Analysis")
print("="*72)

all_results = []

# ① UMAP baseline
print("\n── ① UMAP-10 baseline ────────────────────────────────────────────────")
res_umap = run_umap_baseline()
all_results.append(("① UMAP-10 (raw, no training)", *res_umap))

# ② GCD baseline
res_gcd = run_gcd()
all_results.append(("② GCD [Vaze et al., CVPR 2022]", *res_gcd))

# ③ SimGCD baseline
res_simgcd = run_simgcd()
all_results.append(("③ SimGCD [Wen et al., ICCV 2023]", *res_simgcd))

# ── Round-0 ablations ─────────────────────────────────────────────────────────
# ④ L_AA only
res_aa, head_aa, teacher_aa, init_aa, mem_aa, rng_aa_end, _, _ = run_r0_config(
    "④ L_AA  only", w_bb=0.0, w_distill=0.0, rng_state=POST_DATA_RNG_STATE)
all_results.append(("④ L_AA  only", *res_aa))

# ⑤ L_AA + L_BB
res_aabb, head_aabb, teacher_aabb, init_aabb, mem_aabb, rng_aabb_end, _, _ = run_r0_config(
    "⑤ L_AA + L_BB", w_bb=W_BB_VAL, w_distill=0.0, rng_state=POST_DATA_RNG_STATE)
all_results.append(("⑤ L_AA + L_BB", *res_aabb))

# ⑥ L_AA + L_BB + L_Distill  (= full Round 0)
res_r0, r0_head, r0_teacher, r0_initial_teacher, r0_mem_bank, r0_end_rng_state, Z_A_r0, Z_B_r0 = run_r0_config(
    "⑥ L_AA + L_BB + L_Distill  [full R0]",
    w_bb=W_BB_VAL, w_distill=W_DISTILL_VAL, rng_state=POST_DATA_RNG_STATE)
all_results.append(("⑥ L_AA + L_BB + L_Distill (R0)", *res_r0))

#version 1 
# Prepare for R1: recompute KNN on R0-projected B features (matches original code)
# Z_B_r0_norm = normalize(Z_B_r0, norm="l2").astype(np.float32)
# nbrs_r0     = NearestNeighbors(n_neighbors=K_POS+1, metric="cosine", n_jobs=-1).fit(Z_B_r0_norm)
# knn_B_r1    = nbrs_r0.kneighbors(Z_B_r0_norm, return_distance=False)[:, 1:]
# hard_y_r0, _, _, _, _ = make_pseudo_labels(
#     normalize(Z_A_r0, norm="l2"), target_max_p=0.7)

#version 2 
# Prepare for R1.
# memory_bank_general.py does NOT update knn_B_curr after Round 0.
# Round 1 still uses the raw-DINO KNN graph for B sampling and L_BB masks.
# knn_B_r1 = knn_B_raw

# hard_y_r0, _, _, _, _ = make_pseudo_labels(
#     normalize(Z_A_r0, norm="l2"), target_max_p=0.7)


#version 3 
# Prepare for R1: DINO + UMAP KNN variant
# Uses the baseline UMAP-10 representation computed from raw DINO features.
X_B_umap_norm = normalize(X_B_base, norm="l2").astype(np.float32)

nbrs_umap = NearestNeighbors(
    n_neighbors=K_POS + 1,
    metric="cosine",
    n_jobs=-1
).fit(X_B_umap_norm)

knn_B_r1 = nbrs_umap.kneighbors(
    X_B_umap_norm,
    return_distance=False
)[:, 1:]

hard_y_r0, _, _, _, _ = make_pseudo_labels(
    normalize(Z_A_r0, norm="l2"),
    target_max_p=0.7
)
# ── Round-1 ablations ─────────────────────────────────────────────────────────
# ⑦ + L_NN
res_nn = run_r1_config(
    "⑦ + L_NN", w_nn=W_NN_VAL, w_na=0.0,
    r0_head=r0_head, r0_teacher=r0_teacher,
    r0_initial_teacher=r0_initial_teacher, r0_mem_bank=r0_mem_bank,
    Z_A_prev=Z_A_r0, Z_B_prev=Z_B_r0,
    hard_y_A=hard_y_r0, knn_B_curr=knn_B_r1,
    rng_state=r0_end_rng_state)
all_results.append(("⑦ + L_NN", *res_nn))

# ⑧ + L_NN + L_NA  (full model)
res_full = run_r1_config(
    "⑧ + L_NN + L_NA  [full model]", w_nn=W_NN_VAL, w_na=W_NA_VAL,
    r0_head=r0_head, r0_teacher=r0_teacher,
    r0_initial_teacher=r0_initial_teacher, r0_mem_bank=r0_mem_bank,
    Z_A_prev=Z_A_r0, Z_B_prev=Z_B_r0,
    hard_y_A=hard_y_r0, knn_B_curr=knn_B_r1,
    rng_state=r0_end_rng_state)
all_results.append(("⑧ + L_NN + L_NA (full model)", *res_full))

# ──────────────────────────────────────────────────────────────────────────────
# Summary table
# ──────────────────────────────────────────────────────────────────────────────
print("\n\n" + "="*72)
print("ABLATION SUMMARY")
print("="*72)
header = f"  {'Method':<45}  {'All':>7}  {'Old':>7}  {'Novel':>9}"
print(header)
print("  " + "─"*68)

SECTION_BREAKS = {
    "④ L_AA  only":  "  ── Our method (Round-0 ablations) ──────────────────────────────────",
    "⑦ + L_NN":      "  ── Our method (Round-1 ablations) ──────────────────────────────────",
}

best_nov = max(r[3] for r in all_results)
best_all = max(r[1] for r in all_results)

for tag, a, o, n in all_results:
    if tag in SECTION_BREAKS:
        print(SECTION_BREAKS[tag])
    marks = []
    if n == best_nov: marks.append("Novel◄")
    if a == best_all: marks.append("All◄")
    mark = "  " + "/".join(marks) if marks else ""
    print(f"  {tag:<45}  {a:>7.1%}  {o:>7.1%}  {n:>9.1%}{mark}")

print("="*72)
