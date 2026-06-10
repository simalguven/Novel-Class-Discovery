"""
Iterative Pseudo-label Refinement: Soft Novelty + RKD + High-Purity MemBank
==================================================================================

Minimal soft-novelty variant using a Bayesian-Shifted Sigmoid.
Uses 1D KMeans to estimate empirical cluster priors, shifting the sigmoid 
to demand stronger proof of novelty when novel classes are rare.
"""

import numpy as np, umap, warnings, time
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from scipy.optimize import linear_sum_assignment
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
SEED = 20
K_OLD, K_NEW, K_NOV = 50, 100, 50
N_PER_CLS = 100
K_POS = 10

ORACLE_MODE = "off"

ROUNDS    = 2
EPOCHS_0  = 100
EPOCHS_R  = 50
ITERS_PER_EPOCH = 20
LR_0, LR_R = 3e-4, 1e-4

SM_EMA       = 0.9
SM_WARMUP    = 5
DA_CLAMP_MIN = 0.5
DA_CLAMP_MAX = 2.0

W_AA = 1.0
W_BB_SCHEDULE  = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
W_NN_SCHEDULE  = [0.0, 0.3, 0.4, 0.4, 0.4, 0.4]
W_NA_SCHEDULE  = [0.0, 0.2, 0.1, 0.05, 0.0, 0.0]
W_OA_SCHEDULE  = [0.0, 0.2, 0.15, 0.05, 0.0, 0.0]   
W_DISTILL      = 1.0

TAU_SCHEDULE = [0.15, 0.12, 0.10, 0.09, 0.08, 0.07]

rng  = np.random.default_rng(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("mps"  if torch.backends.mps.is_available() else
                       "cuda" if torch.cuda.is_available() else "cpu")

EPS = 1e-8

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

# ──────────────────────────────────────────────────────────────────────────────
# Baseline UMAP-10 on combined raw features
# ──────────────────────────────────────────────────────────────────────────────
r_base    = umap.UMAP(n_components=10, n_neighbors=20, min_dist=0.05,
                      metric="cosine", random_state=SEED, verbose=False)
X_AB_base = normalize(
    r_base.fit_transform(np.vstack([X_A, X_B])), norm="l2").astype(np.float32)
X_A_umap, X_B_base = X_AB_base[:N_A], X_AB_base[N_A:]

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

res_baseline = gcd_acc(X_B_base, "Baseline UMAP-10 (raw)")

nbrs_B = NearestNeighbors(n_neighbors=K_POS+1, metric="cosine", n_jobs=-1).fit(X_B)
knn_B_raw = nbrs_B.kneighbors(X_B, return_distance=False)[:, 1:]

# ──────────────────────────────────────────────────────────────────────────────
# Network, Memory Bank & SoftMatch
# ──────────────────────────────────────────────────────────────────────────────
class ProjectionHead(nn.Module):
    def __init__(self, in_dim=768, hidden=512, out_dim=128):
        super().__init__()
        self.skip = nn.Linear(in_dim, out_dim, bias=False)
        self.mlp  = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden), nn.GELU(),
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
        if feats is None or feats.numel() == 0:
            return
        b_size = feats.size(0)
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
        self.K = n_classes
        self.ema = ema
        self.lam = lam_dist
        self.mu = torch.tensor(0.5)
        self.sigma2 = torch.tensor(0.1)
        self.p_model = torch.full((n_classes,), 1.0 / n_classes)
        self.p_targ  = torch.full((n_classes,), 1.0 / n_classes)

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
        diff  = (self.mu - max_p).clamp(min=0)
        w     = torch.exp(-(diff ** 2) / (self.lam * self.sigma2 + 1e-8))
        return w.to(probs.device)

    @torch.no_grad()
    def get_bias_correction(self, labels):
        labels_cpu = labels.cpu()
        ratio = self.p_targ / (self.p_model + 1e-8)
        ratio = ratio / ratio.max()
        return ratio[labels_cpu].to(labels.device)

# ──────────────────────────────────────────────────────────────────────────────
# Soft Contrastive Loss
# ──────────────────────────────────────────────────────────────────────────────
def supcon_loss(z, y_old_batch, w_old_batch,
                hard_y_kn_AB, q_novel_b_batch,
                is_A, bb_mask,
                mem_bank_z=None, mem_bank_y=None, mem_bank_w=None,
                tau=0.1, w_bb=1.0, w_nn=0.0, w_na=0.0, w_oa=0.0):
    N = z.size(0)
    eye = torch.eye(N, dtype=torch.bool, device=z.device)

    sim = (z @ z.T) / tau
    mx, _ = sim.max(dim=1, keepdim=True)
    exs   = torch.exp(sim - mx).masked_fill(eye, 0.0)
    denom = exs.sum(dim=1, keepdim=True) + 1e-8
    lp    = (sim - mx) - torch.log(denom)

    is_At = is_A.to(z.device)
    a_pos = is_At.nonzero(as_tuple=True)[0]
    b_pos = (~is_At).nonzero(as_tuple=True)[0]
    n_a   = a_pos.numel()

    losses = []

    # ── AA: hard A pseudo-label SupCon + memory bank ─────────────────────────
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

        N_AA  = z_AA.size(0)
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
            pair_w  = w_AA.unsqueeze(1) * w_ALL.unsqueeze(0) * same_aa
            pair_w  = pair_w.masked_fill(eye_mask, 0.0)

            aa_sum = pair_w.sum(dim=1)
            has    = aa_sum > 0
            if has.any():
                l_aa = -((pair_w * lp_aa).sum(dim=1)[has] / aa_sum[has].clamp_min(1e-8))
                losses.append(W_AA * l_aa.mean())

    # ── BB: B-B kNN positives (hard mask) ────────────────────────────────────
    bb_w = bb_mask.to(z.device).float()
    bb_w = bb_w.masked_fill(eye, 0.0)
    bb_sum = bb_w.sum(dim=1)
    if (bb_sum > 0).any() and w_bb > 0:
        has = bb_sum > 0
        l_bb = -((bb_w * lp).sum(dim=1)[has] / bb_sum[has].clamp_min(1e-8))
        losses.append(w_bb * l_bb.mean())

    if q_novel_b_batch is not None:
        q_novel_b_batch = q_novel_b_batch.to(z.device).float().clamp(0.0, 1.0)

    # ── NN: soft novel-novel attraction (B samples) ──────────────────────────
    if w_nn > 0 and q_novel_b_batch is not None and b_pos.numel() > 1:
        y_kn = hard_y_kn_AB.to(z.device)
        b_mask_row = (~is_At).float().unsqueeze(1)
        b_mask_col = (~is_At).float().unsqueeze(0)

        nov_assigned = (y_kn >= K_OLD).float()
        same_kn = (y_kn.unsqueeze(0) == y_kn.unsqueeze(1)).float()
        same_novel = same_kn * nov_assigned.unsqueeze(0) * nov_assigned.unsqueeze(1)

        q_outer = q_novel_b_batch.unsqueeze(0) * q_novel_b_batch.unsqueeze(1)
        nn_w = same_novel * q_outer * b_mask_row * b_mask_col
        nn_w = nn_w.masked_fill(eye, 0.0)

        nn_sum = nn_w.sum(dim=1)
        if (nn_sum > 0).any():
            has = nn_sum > 0
            l_nn = -((nn_w * lp).sum(dim=1)[has] / nn_sum[has].clamp_min(1e-8))
            losses.append(w_nn * l_nn.mean())

    # ── NA: soft novel-B repulsion from A ────────────────────────────────────
    if w_na > 0 and q_novel_b_batch is not None and n_a > 0 and b_pos.numel() > 0:
        q_b = q_novel_b_batch[b_pos]
        active = q_b > 1e-3
        if active.any():
            z_b = z[b_pos][active]
            z_a = z[a_pos]
            q_eff = q_b[active]

            sim_ba = (z_b @ z_a.T) / tau
            l_na = torch.logsumexp(sim_ba, dim=1) - np.log(max(n_a, 1))
            losses.append(w_na * (q_eff * l_na).sum() / q_eff.sum().clamp_min(1e-8))

    # ── OA: soft old-B to A attraction ───────────────────────────────────────
    if w_oa > 0 and q_novel_b_batch is not None and n_a > 0 and b_pos.numel() > 0:
        # CONTINUOUS CONFIDENCE SHARPENING: Penalize uncertainty to prevent sink effect
        q_old_b = (1.0 - q_novel_b_batch[b_pos]) ** 2    # [n_b]
        active  = q_old_b > 1e-3
        if active.any():
            z_b   = z[b_pos][active]                  # [m, d]
            z_a   = z[a_pos]                          # [n_a, d]
            y_a   = y_old_batch[a_pos].long()         # [n_a]
            w_a   = w_old_batch[a_pos].to(z.device).float()  # [n_a]
            y_kn  = hard_y_kn_AB.to(z.device)
            y_b   = y_kn[b_pos][active]               # [m]
            q_eff = q_old_b[active]                   # [m]

            old_assigned = (y_b < K_OLD)
            if old_assigned.any():
                z_b_old = z_b[old_assigned]
                y_b_old = y_b[old_assigned]
                q_b_old = q_eff[old_assigned]

                sim_ba = (z_b_old @ z_a.T) / tau
                mx_ba, _ = sim_ba.max(dim=1, keepdim=True)
                exs_ba = torch.exp(sim_ba - mx_ba)
                denom_ba = exs_ba.sum(dim=1, keepdim=True) + 1e-8
                lp_ba    = (sim_ba - mx_ba) - torch.log(denom_ba)

                same = (y_b_old.unsqueeze(1) == y_a.unsqueeze(0)).float()  
                pair_w = same * w_a.unsqueeze(0) * q_b_old.unsqueeze(1)

                pos_sum = pair_w.sum(dim=1)
                has = pos_sum > 0
                if has.any():
                    l_oa = -((pair_w * lp_ba).sum(dim=1)[has] / pos_sum[has].clamp_min(1e-8))
                    losses.append(w_oa * l_oa.mean())

    if not losses:
        return torch.tensor(0., device=z.device, requires_grad=True)
    return sum(losses)

# ──────────────────────────────────────────────────────────────────────────────
# Batch sampling & masks
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
# Pseudo-label functions
# ──────────────────────────────────────────────────────────────────────────────
def make_pseudo_labels(Z_A, target_max_p=0.7):
    Z_n = normalize(Z_A, norm="l2").astype(np.float32)
    km  = KMeans(n_clusters=K_OLD, n_init=15, random_state=SEED).fit(Z_n)
    centers = normalize(km.cluster_centers_, norm="l2").astype(np.float32)
    sim     = (Z_n @ centers.T).astype(np.float32)
    hard    = sim.argmax(axis=1)

    best_T, best_ap, best_p = None, None, None
    for T in np.geomspace(1e-3, 0.5, 60):
        s = sim / T
        s = s - s.max(axis=1, keepdims=True)
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
        if len(mem):
            old_protos[k] = mem.mean(axis=0)
    old_protos = normalize(old_protos, norm="l2").astype(np.float32)

    sim_B_to_old = Z_B @ old_protos.T
    max_sim_B    = sim_B_to_old.max(axis=1)

    # ── Soft novelty: 1D KMeans + Bayesian Sigmoid ───────────────────────────
    km_1d = KMeans(n_clusters=2, n_init=10, random_state=SEED).fit(max_sim_B.reshape(-1, 1))
    centers = km_1d.cluster_centers_.flatten()
    
    # Identify which cluster is Novel (lower similarity) and Old (higher similarity)
    nov_idx = np.argmin(centers)
    old_idx = np.argmax(centers)
    low_c, high_c = centers[nov_idx], centers[old_idx]
    
    # 1. Calculate empirical priors from KMeans cluster sizes
    p_nov = np.mean(km_1d.labels_ == nov_idx)
    p_old = 1.0 - p_nov
    
    # 2. Base threshold and scale
    threshold = (low_c + high_c) / 2.0
    span  = max(threshold - low_c, EPS)
    scale = span / np.log(19.0)
    
    # 3. Calculate base logits
    base_logits = (threshold - max_sim_B) / scale
    
    # 4. Apply Bayesian Shift: ln(P(nov) / P(old))
    # This natively fixes the imbalance by requiring stronger evidence for the minority class
    prior_shift = np.log((p_nov + 1e-5) / (p_old + 1e-5))
    shifted_logits = base_logits + prior_shift
    
    # 5. Final continuous soft probability
    q_novel_B = 1.0 / (1.0 + np.exp(-shifted_logits))
    q_novel_B = q_novel_B.astype(np.float32)

    # ── Novel prototype init: weight by q ────────────────────────────────────
    seed_cut  = np.quantile(q_novel_B, 0.5)
    seed_mask = q_novel_B >= seed_cut
    if seed_mask.sum() < K_NOV_LOC:
        seed_mask = np.ones(len(Z_B), dtype=bool)

    km_nov = KMeans(n_clusters=K_NOV_LOC, n_init=15,
                    random_state=SEED).fit(Z_B[seed_mask])
    novel_protos = normalize(km_nov.cluster_centers_, norm="l2").astype(np.float32)

    # ── Hard EM refinement ───────────────────────────────────────────────────
    Z_AB    = np.vstack([Z_A, Z_B]).astype(np.float32)
    centers = np.vstack([old_protos, novel_protos]).astype(np.float32)

    for _ in range(20):
        sim    = Z_AB @ centers.T
        labels = sim.argmax(axis=1)
        for k in range(K_OLD, K_NEW):
            mem = Z_AB[labels == k]
            if len(mem):
                centers[k] = mem.mean(axis=0)
                centers[k] /= np.linalg.norm(centers[k]) + EPS

    sim     = Z_AB @ centers.T
    hard_AB = sim.argmax(axis=1).astype(np.int64)

    # ── Soft probabilities (for SoftMatch warm_start) ────────────────────────
    best_T, best_ap, best_p = None, None, None
    for T in np.geomspace(1e-3, 0.5, 60):
        s = sim / T
        s = s - s.max(axis=1, keepdims=True)
        p = np.exp(s); p /= p.sum(axis=1, keepdims=True) + EPS
        ap = p.max(axis=1).mean()
        if best_T is None or abs(ap - target_max_p) < abs(best_ap - target_max_p):
            best_T, best_ap, best_p = T, ap, p

    return (hard_AB, best_p.astype(np.float32), q_novel_B,
            float(best_T), float(best_ap))

# ──────────────────────────────────────────────────────────────────────────────
# Iterative training loop
# ──────────────────────────────────────────────────────────────────────────────
Xt_A = torch.from_numpy(X_A)
Xt_B = torch.from_numpy(X_B)

head            = None
teacher         = None
initial_teacher = None
Z_A_current     = X_A_umap
knn_B_curr      = knn_B_raw
all_results     = [("Baseline (raw UMAP-10)", *res_baseline)]

hard_kn_t_frozen   = None
soft_kn_t_frozen   = None
sm_kn_frozen       = None
q_novel_B_frozen   = None

mem_bank = None

print("\n" + "="*72)
print("SOFT NOVELTY (Bayesian Sigmoid) + RKD + High-Purity MemBank")
print("="*72)

for rnd in range(ROUNDS):
    print(f"\n{'─'*72}")
    print(f"ROUND {rnd}  {'(fresh head)' if rnd==0 else '(fine-tuning)'}")

    hard_y, soft_p, centers, used_T, ach_max = make_pseudo_labels(
        Z_A_current, target_max_p=0.7)

    soft_p_t = torch.from_numpy(soft_p)
    hard_y_t = torch.from_numpy(hard_y.astype(np.int64))

    sm = SoftMatch(n_classes=K_OLD, ema=SM_EMA, lam_dist=2.0)
    sm.warm_start(soft_p_t)
    proto_clf = PrototypeClassifier(n_classes=K_OLD, tau_proto=0.07)

    w_nn = W_NN_SCHEDULE[min(rnd, len(W_NN_SCHEDULE)-1)]
    w_na = W_NA_SCHEDULE[min(rnd, len(W_NA_SCHEDULE)-1)]
    w_oa = W_OA_SCHEDULE[min(rnd, len(W_OA_SCHEDULE)-1)]
    use_kn = (w_nn > 0) or (w_na > 0) or (w_oa > 0)

    if use_kn and hard_kn_t_frozen is not None:
        hard_kn_t   = hard_kn_t_frozen
        soft_kn_t   = soft_kn_t_frozen
        sm_kn       = sm_kn_frozen
        q_novel_B_np = q_novel_B_frozen

    elif use_kn:
        anchor_A = normalize(Z_A_proj_prev, norm="l2").astype(np.float32)
        anchor_B = normalize(Z_B_proj_prev, norm="l2").astype(np.float32)

        hard_kn, soft_kn, q_novel_B_np, T_kn, ap_kn = \
            make_constrained_joint_labels(anchor_A, anchor_B, hard_y)

        soft_kn_t = torch.from_numpy(soft_kn)
        hard_kn_t = torch.from_numpy(hard_kn.astype(np.int64))
        sm_kn = SoftMatch(n_classes=K_NEW, ema=SM_EMA, lam_dist=2.0)
        sm_kn.warm_start(soft_kn_t)

        hard_kn_t_frozen = hard_kn_t
        soft_kn_t_frozen = soft_kn_t
        sm_kn_frozen     = sm_kn
        q_novel_B_frozen = q_novel_B_np
    else:
        soft_kn_t    = None
        hard_kn_t    = None
        sm_kn        = None
        q_novel_B_np = np.zeros(N_B, dtype=np.float32)

    q_novel_B_t = torch.from_numpy(q_novel_B_np.astype(np.float32))

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

        mem_bank = MemoryBank(size=1024, dim=128, num_classes=K_OLD, device=DEVICE)

    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = epochs * ITERS_PER_EPOCH
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    for ep in range(epochs):
        hard_y_tensor = torch.from_numpy(hard_y.astype(np.int64))
        proto_clf.update_prototypes(teacher, Xt_A, hard_y_tensor, DEVICE)
        teacher.eval()
        with torch.no_grad():
            z_A_all       = teacher(Xt_A.to(DEVICE))
            p_live_all    = proto_clf.predict(z_A_all)
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

            if ep < SM_WARMUP:
                conf_weight = torch.ones(n_a, device=DEVICE)
                w_A_batch   = torch.ones(n_a, device=DEVICE)
            else:
                conf_weight = sm.weight(p_aligned_all[a_idx])
                bias_weight = sm.get_bias_correction(hy_batch).to(DEVICE)
                w_A_batch   = conf_weight * bias_weight

            y_old_batch = torch.zeros(n_tot, dtype=torch.long, device=DEVICE)
            y_old_batch[:n_a] = hy_batch
            w_old_batch = torch.zeros(n_tot, device=DEVICE)
            w_old_batch[:n_a] = w_A_batch

            if use_kn and hard_kn_t is not None:
                kn_idx_AB   = np.concatenate([a_idx, np.array(b_idx) + N_A]).astype(np.int64)
                hy_kn_batch = hard_kn_t[kn_idx_AB].to(DEVICE)

                q_b_batch = torch.cat([
                    torch.zeros(n_a, dtype=torch.float32),
                    q_novel_B_t[b_idx].float(),
                ]).to(DEVICE)
                y_old_batch[n_a:] = hy_kn_batch[n_a:]
            else:
                hy_kn_batch = torch.zeros(n_tot, dtype=torch.long, device=DEVICE)
                q_b_batch   = torch.zeros(n_tot, device=DEVICE)

            z = head(x_batch)
            mb_z, mb_y, mb_w = mem_bank.get_all()

            loss = supcon_loss(z, y_old_batch, w_old_batch,
                               hy_kn_batch, q_b_batch,
                               is_A, bbm,
                               mb_z, mb_y, mb_w,
                               tau=tau, w_bb=w_bb,
                               w_nn=w_nn, w_na=w_na, w_oa=w_oa)

            if n_a > 1:
                with torch.no_grad():
                    z_orig_A = initial_teacher(x_batch[:n_a])
                sim_orig = z_orig_A @ z_orig_A.T
                sim_new  = z[:n_a] @ z[:n_a].T
                l_distill = F.mse_loss(sim_new, sim_orig)
            else:
                l_distill = torch.tensor(0.0, device=DEVICE)

            total_loss = loss + (W_DISTILL * l_distill)

            opt.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            sch.step()
            ema_update(head, teacher, m=0.999)

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

    res_proj = gcd_acc(Z_B, f"Round {rnd} — projected (128-d)", verbose=False)
    all_results.append((f"Round {rnd}  proj", *res_proj))

    Z_B_norm = normalize(Z_B, norm="l2")
    res_pure = gcd_acc(Z_B_norm, f"Round {rnd} — pure", verbose=False)
    all_results.append((f"Round {rnd}  pure", *res_pure))

    Z_A_proj_prev = Z_A.copy()
    Z_B_proj_prev = Z_B.copy()
    Z_A_current = normalize(Z_A, norm="l2").astype(np.float32)

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
