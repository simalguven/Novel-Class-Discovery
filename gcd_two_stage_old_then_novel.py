"""
Iterative Pseudo-label Refinement with SoftMatch + Semi-supervised SupCon for GCD
==================================================================================

Generalized Category Discovery (GCD) setup
------------------------------------------
A : unlabeled  — assumed to contain ONLY the K_OLD "old" classes
B : unlabeled  — contains a MIX of old + novel classes (total = K_NEW)

We have NO labels at any point during training.  Ground-truth labels are loaded
ONLY for evaluation (`gcd_acc`, `pseudo_label_acc`).

What's new vs. the original pipeline
------------------------------------
1.  **SoftMatch confidence weighting** (Chen et al., ICLR 2023).
    Instead of the hand-crafted (own_dist + margin)/2 tier scheme, every A sample
    gets a continuous weight λ(x) ∈ (0, 1] derived from the *truncated Gaussian*
    of max-softmax confidence:

        λ(x) = exp( - (max(0, μ_t − p̂(x)))² / (2 σ_t²) )

    where μ_t, σ_t are EMA-tracked statistics of the confidence distribution.
    This keeps **all** A samples in the loss (no hard cut-off) but down-weights
    low-confidence ones smoothly.  No discarded samples → no information loss.

2.  **Distribution Alignment (DA)** — also from SoftMatch.
    Pseudo-label predictions are re-normalised toward a uniform class prior
    using a running estimate of the model's marginal distribution.  This is
    essential in GCD where unbalanced K-Means tends to collapse novel-class
    clusters into majority old-class ones.

3.  **Joint A+B clustering at K_NEW** for B-side pseudo-labels.
    Clustering A alone at K_OLD gives no signal for novel classes in B.
    We additionally run K-Means(K_NEW) on the *combined* projected features and
    use those cluster assignments as soft positives in the B branch (replacing
    the fixed raw-DINOv2 kNN graph in later rounds).  Round 0 still uses raw
    kNN as a bootstrap because the projection head is untrained.

4.  **Soft pseudo-labels via prototype similarity** for the SupCon positive set.
    Hard labels make every same-cluster pair a positive with equal weight.
    We instead weight each pair by the product of soft-assignment probabilities
    p_i[k] · p_j[k], summed over clusters.  This gracefully handles ambiguous
    boundary points without requiring hard tier thresholds.

5.  **Temperature annealing** (τ: 0.15 → 0.07 over rounds) tightens the embedding
    geometry once pseudo-labels stabilise.

6.  **EMA teacher head** for stable confidence estimation across epochs.

7.  **Pseudo-label identity stabilization**.
    K-Means cluster IDs on A are aligned to the previous round with Hungarian
    matching, and soft-prob columns / centroids are reordered to match the
    stabilized IDs.  This is critical once old-B → A alignment is enabled:
    without stable IDs, "cluster 12" can silently change meaning across rounds.

8.  **Anchored A pseudo-label source**.
    A pseudo-labels are always produced from [projection || baseline UMAP]
    after round 0, rather than switching to projection-only.  This prevents
    learned-representation drift from corrupting the A-side old-class anchor.

9.  **Stricter confident-old B selection**.
    Old-B → A alignment uses a precomputed confident-old mask based on:
      • K_NEW assignment is an old ID,
      • high similarity to an A old prototype,
      • high old-prototype top-1/top-2 margin,
      • old-vs-novel centroid margin is positive,
      • sufficiently high K_NEW soft confidence.

10. **Non-uniform K_NEW distribution-alignment prior**.
    For joint A+B labels, old classes have twice the expected mass of novel
    classes in the combined set (old: A+B, novel: B only), so K_NEW SoftMatch
    uses a 2:1 old:novel target prior instead of uniform.
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
K_POS = 10                        # k-NN positives in raw DINOv2 space (round 0)

# ── Oracle / debug modes ──────────────────────────────────────────────────────
# These cheat by using ground-truth labels.  Purpose: isolate whether the
# *contrastive learning machinery* is sound, separately from pseudo-label noise.
#
#   "off"      : production — pseudo-labels everywhere (default).
#   "A"        : ground-truth labels for A's K_OLD SupCon term only.
#                K_NEW joint labels and B-kNN remain pseudo-derived.
#                If this beats "off", A-side label noise is a real bottleneck.
#   "AB"       : ground-truth labels for K_OLD on A and K_NEW on (A⊕B), plus
#                same-class B-kNN graph.  This is the absolute ceiling — what
#                the architecture can do with perfect supervision.
#                If this saturates well below 100%, the head/loss/τ is the
#                bottleneck, not pseudo-labels.
#
# When ORACLE_MODE != "off", SoftMatch λ is forced to 1 (no need to filter
# perfect labels).  Pseudo-label refinement / early-stop logic is skipped.
ORACLE_MODE = "off"               # one of: "off", "A", "AB"

ROUNDS    = 6
EPOCHS_0  = 100
EPOCHS_R  = 50
ITERS_PER_EPOCH = 20              # gradient steps per "epoch" (was 1 implicitly)
LR_0, LR_R = 3e-4, 1e-4

# SoftMatch hyper-parameters
SM_EMA       = 0.9                # EMA momentum for μ_t, σ_t and DA
SM_WARMUP    = 5                  # epochs of full-weight A loss before SoftMatch kicks in
DA_CLAMP_MIN = 0.5
DA_CLAMP_MAX = 2.0

# Soft-pseudo-label sharpness.  Picked so that ⟨max p⟩ ≈ 0.5–0.7 — high enough
# that  Σ_k p_i[k]·p_j[k]  carries real same/different-cluster signal, low enough
# to retain uncertainty information at boundaries.
SOFT_LABEL_T = 0.02

# Loss-branch balancing.  Without this, B-B kNN positives (binary, dense) swamp
# A-A soft positives (sparse, fractional).  These are *relative* scales applied
# to the per-row weight sums in the loss — see supcon_loss.
W_AA = 1.0
W_BB_SCHEDULE = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3]   # B-kNN weight, flat across rounds.
W_NN_SCHEDULE = [0.0, 0.3, 0.4, 0.4, 0.4, 0.4]   # Novel-Novel SupCon (within
                                                  # confident-novel B subset).
                                                  # Round 0 = 0 (no K_NEW labels yet).
W_NA_SCHEDULE = [0.0, 0.2, 0.3, 0.3, 0.3, 0.3]   # Novel-vs-A repulsion. Lower
                                                  # than W_NN because the gradient
                                                  # is denser (every confident-novel
                                                  # B repels against every A sample
                                                  # in the batch ≈ 300 pairs vs
                                                  # ~3 NN pairs).
W_OB_SCHEDULE = [0.0, 0.2, 0.3, 0.3, 0.3, 0.3]   # Old-B → A alignment.
                                                  # Confident-old B anchors are
                                                  # pulled toward A samples with
                                                  # the matching K_OLD pseudo-label.

# Pseudo-label stability controls
ALIGN_A_LABELS_TO_PREV = True       # Hungarian-align KMeans IDs across rounds.
ANCHOR_A_LABEL_SOURCE  = True       # Always use [proj || baseline UMAP] for A pseudo-labels.
FREEZE_KNEW_LABELS     = False      # Recompute K_NEW labels each round after A label alignment.

# Confident-old B selection for OB branch.
# These quantiles are computed among B samples assigned to an old K_NEW label.
CONF_OLD_SIM_Q         = 0.50       # require old-prototype max-sim above this quantile
CONF_OLD_MARGIN_Q      = 0.50       # require old top-1/top-2 margin above this quantile
CONF_OLD_SOFT_MIN      = 0.60       # require K_NEW soft max-prob at least this value
CONF_OLD_NOV_MARGIN_MIN = 0.00      # require max_old_sim - max_novel_sim > this

# K_NEW DA prior for joint A+B clustering:
# old classes appear in A and B (2 units); novel classes appear in B only (1 unit).
KNEW_OLD_PRIOR_MASS = 2.0
KNEW_NOV_PRIOR_MASS = 1.0


# Temperature schedule for SupCon (annealing, round-indexed)
TAU_SCHEDULE = [0.15, 0.12, 0.10, 0.09, 0.08, 0.07]

# Reproducibility
rng  = np.random.default_rng(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("mps"  if torch.backends.mps.is_available() else
                       "cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────────────────────────────────────
# Data loading (identical to original)
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
y_A_eval   = np.array([id2idx[c] for c in y_A])     # eval-only
y_B_eval   = np.array([id2idx[c] for c in y_B])     # eval-only
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
    """Hungarian-matched accuracy of pseudo-labels vs ground truth (eval only)."""
    if y_eval is None: y_eval = y_A_eval
    n_true = len(np.unique(y_eval))
    K = max(n_clusters, n_true)
    mat = np.zeros((K, K), dtype=np.int64)
    for p, t in zip(pseudo_y, y_eval):
        mat[p % K, t] += 1
    r, c = linear_sum_assignment(-mat)
    return mat[r, c].sum() / len(pseudo_y)


# ──────────────────────────────────────────────────────────────────────────────
# Pseudo-label stability / diagnostics helpers
# ──────────────────────────────────────────────────────────────────────────────
def _safe_counts(y, K):
    return np.bincount(np.asarray(y, dtype=np.int64), minlength=K)

def label_count_summary(y, K, name="labels"):
    counts = _safe_counts(y, K)
    zero = int((counts == 0).sum())
    print(f"       {name} sizes: min={counts.min()}  p10={np.percentile(counts, 10):.1f}  "
          f"med={np.median(counts):.1f}  p90={np.percentile(counts, 90):.1f}  "
          f"max={counts.max()}  empty={zero}")
    if zero:
        empty = np.where(counts == 0)[0][:10].tolist()
        print(f"       WARNING: empty {name} IDs, first few: {empty}")

def align_labels_to_previous(cur_y, prev_y, K):
    """
    Hungarian-align current KMeans IDs to previous-round IDs.

    Returns
    -------
    aligned_y : same shape as cur_y, but IDs mapped into previous-round convention.
    cur_to_prev : length-K array where cur_to_prev[current_id] = previous_id.
    agreement : fraction of aligned_y equal to prev_y.
    """
    cur_y = np.asarray(cur_y, dtype=np.int64)
    prev_y = np.asarray(prev_y, dtype=np.int64)
    confusion = np.zeros((K, K), dtype=np.int64)  # rows: previous, cols: current
    for p, c in zip(prev_y, cur_y):
        if 0 <= p < K and 0 <= c < K:
            confusion[p, c] += 1

    row, col = linear_sum_assignment(-confusion)
    cur_to_prev = np.arange(K, dtype=np.int64)
    for prev_id, cur_id in zip(row, col):
        cur_to_prev[cur_id] = prev_id

    aligned_y = cur_to_prev[cur_y]
    agreement = float((aligned_y == prev_y).mean())
    return aligned_y.astype(np.int64), cur_to_prev, agreement

def reorder_by_cur_to_prev(arr, cur_to_prev, axis=0):
    """
    Reorder cluster-indexed arrays from current-ID order into previous-ID order.
    For soft probabilities use axis=1. For centers use axis=0.
    """
    arr = np.asarray(arr)
    K = len(cur_to_prev)
    inv = np.arange(K, dtype=np.int64)
    for cur_id, prev_id in enumerate(cur_to_prev):
        inv[prev_id] = cur_id
    return np.take(arr, inv, axis=axis)

def make_knew_prior():
    prior = np.concatenate([
        np.full(K_OLD, KNEW_OLD_PRIOR_MASS, dtype=np.float32),
        np.full(K_NEW - K_OLD, KNEW_NOV_PRIOR_MASS, dtype=np.float32),
    ])
    prior /= prior.sum()
    return torch.from_numpy(prior)

def diagnostic_label_transition(prev_y, cur_y, K, name):
    aligned, _, agree = align_labels_to_previous(cur_y, prev_y, K)
    changed = 1.0 - agree
    print(f"       {name} transition: aligned agreement={agree:.1%}, changed={changed:.1%}")
    return aligned, agree


res_baseline = gcd_acc(X_B_base, "Baseline UMAP-10 (raw)")

# ──────────────────────────────────────────────────────────────────────────────
# Fixed raw-DINOv2 kNN graph for B (used in round 0 as bootstrap)
# ──────────────────────────────────────────────────────────────────────────────
print(f"\nPre-computing {K_POS}-NN for B (raw DINOv2) …")
t0   = time.time()
nbrs = NearestNeighbors(n_neighbors=K_POS+1, metric="cosine", n_jobs=-1).fit(X_B)
knn_B_raw = nbrs.kneighbors(X_B, return_distance=False)[:, 1:]
print(f"  Done in {time.time()-t0:.1f}s")

# Ground-truth same-class B-kNN.  For each B sample, K_POS neighbours drawn
# uniformly at random from same-class B samples.  Used only in ORACLE_MODE="AB"
# to give SupCon a perfect B-side positive set (every B-B "neighbour" is same
# class, by construction).  This is cheating; the purpose is to upper-bound
# what the architecture can learn given any quality of B-side supervision.
def build_oracle_knn_B():
    knn = np.zeros((N_B, K_POS), dtype=np.int64)
    for i in range(N_B):
        same = np.where((y_B_eval == y_B_eval[i]) & (np.arange(N_B) != i))[0]
        knn[i] = rng.choice(same, size=K_POS, replace=(len(same) < K_POS))
    return knn
knn_B_oracle = build_oracle_knn_B() if ORACLE_MODE == "AB" else None
if knn_B_oracle is not None:
    print(f"  Oracle B-kNN built (every neighbour same-class)")

# ──────────────────────────────────────────────────────────────────────────────
# Model — residual projection head + EMA teacher
# ──────────────────────────────────────────────────────────────────────────────
class ProjectionHead(nn.Module):
    """
    Residual projection head:  z = L2_normalize(x_proj + α · MLP(x))

    The skip connection L (linear projection 768→128) is initialized to a
    PCA-like "preserve" mapping by Xavier init, and the MLP starts near-zero
    via the final BN layer.  Net effect: at initialization the head is close
    to a linear projection of the input — it cannot underperform a random
    linear embedding the way the previous BN-then-ReLU head did.

    The α gate is a learned scalar, initialized to 0.1 so the residual branch
    contributes mildly at first and grows only if it helps.
    """
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
        # Zero-init the MLP's last layer so the residual is exactly 0 at start.
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
# Prototype classifier — live model predictions for SoftMatch
# ──────────────────────────────────────────────────────────────────────────────
class PrototypeClassifier:
    """
    A non-parametric softmax classifier built from the EMA teacher's current
    prototypes.  This is what SoftMatch should actually track: the *model's
    current belief* about which cluster a sample belongs to, updated as the
    projection head evolves.

    Usage per training epoch
    ─────────────────────────
    1. Call update_prototypes(teacher, X_A_all, hard_y) once per epoch to
       recompute class-mean embeddings in the teacher's current embedding space.
    2. Call predict(z) on any batch of embeddings → softmax probs.
    3. Feed those probs into sm.update() and sm.weight().

    This replaces the frozen `soft_p_t` for confidence tracking.  The hard
    pseudo-labels (`hard_y_t`) still come from the pre-epoch K-Means — we
    only use the classifier for *weighting*, not for *label assignment*.
    """
    def __init__(self, n_classes, tau_proto=0.1):
        self.K          = n_classes
        self.tau        = tau_proto
        self.prototypes = None              # (K, d) L2-normalized, on DEVICE

    @torch.no_grad()
    def update_prototypes(self, encoder, X_all_t, hard_y, device):
        """Recompute class-mean embeddings from the current encoder."""
        encoder.eval()
        z_all = encoder(X_all_t.to(device))     # (N, d) L2-normalized
        protos = torch.zeros(self.K, z_all.size(1), device=device)
        counts = torch.zeros(self.K, device=device)
        for k in range(self.K):
            mask = (hard_y == k)
            if mask.any():
                protos[k] = z_all[mask].mean(0)
                counts[k] = mask.sum()
        # L2-normalize prototypes (zero-count classes get zero vector, handled below)
        norms = protos.norm(dim=1, keepdim=True).clamp_min(1e-8)
        self.prototypes = protos / norms

    @torch.no_grad()
    def predict(self, z):
        """
        z : (N, d) L2-normalized embeddings (already on device)
        Returns: (N, K) softmax probs based on cosine sim to prototypes.
        """
        if self.prototypes is None:
            # Fallback: uniform — means λ = 1 for everyone (warm-up behavior)
            return torch.full((z.size(0), self.K), 1.0 / self.K, device=z.device)
        sim = z @ self.prototypes.T / self.tau    # (N, K)
        sim = sim - sim.max(dim=1, keepdim=True).values
        p   = torch.exp(sim)
        return p / p.sum(dim=1, keepdim=True).clamp_min(1e-8)
class SoftMatch:
    """
    Tracks (μ_t, σ_t) of max-softmax confidence with EMA, and a running
    distribution-alignment ratio  p_target / p_model  per class.

    Sample weight λ(x):
        if  p̂(x) ≥ μ_t:  λ = 1
        else            :  λ = exp( − (μ_t − p̂)² / (2 λ_dist · σ_t²) )

    DA: realign predictions so that the model's average prediction matches
        a uniform prior  p_target = 1/K  (standard choice in absence of label info).
    """
    def __init__(self, n_classes, ema=SM_EMA, lam_dist=2.0):
        self.K       = n_classes
        self.ema     = ema
        self.lam     = lam_dist
        # NOTE: μ, σ² are warm-started from the actual soft-label distribution
        # in `warm_start()` below, not from the uniform 1/K prior.  The 1/K prior
        # is wildly off the typical max-prob distribution and made μ crawl up
        # too slowly under EMA, causing λ_A ≈ 1 throughout training.
        self.mu      = torch.tensor(0.5)
        self.sigma2  = torch.tensor(0.1)
        self.p_model = torch.full((n_classes,), 1.0 / n_classes)
        self.p_targ  = torch.full((n_classes,), 1.0 / n_classes)

    @torch.no_grad()
    def warm_start(self, probs):
        """Initialize μ, σ², p_model directly from the full A-set predictions."""
        max_p = probs.max(dim=1).values
        self.mu      = max_p.mean().cpu()
        self.sigma2  = (max_p.var(unbiased=False) + 1e-4).cpu()
        self.p_model = probs.mean(dim=0).cpu()

    @torch.no_grad()
    def update(self, probs):
        """probs: (N, K) softmax over classes for unlabeled samples."""
        max_p = probs.max(dim=1).values          # (N,)
        m     = max_p.mean()
        v     = max_p.var(unbiased=False) + 1e-8
        self.mu     = self.ema * self.mu     + (1 - self.ema) * m.cpu()
        self.sigma2 = self.ema * self.sigma2 + (1 - self.ema) * v.cpu()
        avg_p       = probs.mean(dim=0)
        self.p_model = self.ema * self.p_model + (1 - self.ema) * avg_p.cpu()

    @torch.no_grad()
    def align(self, probs):
        """Distribution alignment: probs *= p_target / p_model, then renormalize."""
        ratio = (self.p_targ / (self.p_model + 1e-8)).to(probs.device)
        ratio = ratio.clamp(DA_CLAMP_MIN, DA_CLAMP_MAX)
        aligned = probs * ratio.unsqueeze(0)
        return aligned / aligned.sum(dim=1, keepdim=True).clamp_min(1e-8)

    @torch.no_grad()
    def weight(self, probs):
        """λ(x) ∈ (0,1] from truncated Gaussian on max-softmax."""
        max_p  = probs.max(dim=1).values.cpu()
        diff   = (self.mu - max_p).clamp(min=0)
        w      = torch.exp(-(diff ** 2) / (self.lam * self.sigma2 + 1e-8))
        return w.to(probs.device)

# ──────────────────────────────────────────────────────────────────────────────
# Multi-term SupCon loss: AA (K_OLD on A) + BB (kNN on B) + NN + NA + OB
#   NN: SupCon among confident-novel B (high-purity novel pool, K_NEW labels)
#   NA: Repulsion of confident-novel B from all A samples (known-old)
#   OB: Alignment of confident-old B anchors to same-label A samples
# ──────────────────────────────────────────────────────────────────────────────
def supcon_loss(z, hard_y_A, w_A,
                hard_y_kn_AB, is_conf_novel, is_conf_old,
                is_A, bb_mask,
                tau=0.1, w_bb=1.0, w_nn=0.0, w_na=0.0, w_ob=0.0):
    """
    Five independent terms, each normalised, summed with separate weights.

    Args
    ----
    z              : (N, d) L2-normalized embeddings.
    hard_y_A       : (n_a,) K_OLD cluster id for batch's A samples.
    w_A            : (n_a,) SoftMatch-K_OLD λ.
    hard_y_kn_AB   : (N,) K_NEW cluster id for ALL batch samples (used by NN).
                      Can be all-zero when no K_NEW labels yet.
    is_conf_novel  : (N,) bool — True only for batch samples that are
                      confident-novel B (intersection criterion).  All A
                      entries are False.  Drives NN + NA.
    is_conf_old    : (N,) bool — True only for confident-old B samples.
                      All A entries are False.  Drives OB.
    is_A           : (N,) bool — A vs B membership.
    bb_mask        : (N, N) bool — kNN positives for B-B pairs.
    tau            : temperature.
    w_bb, w_nn, w_na, w_ob : per-branch weights.  W_AA is module-level constant.

    Term details
    ────────────
    AA (per A anchor): supervised contrastive on K_OLD pseudo-labels,
        SoftMatch-weighted via outer product of λ.
    BB (per B anchor): kNN positives, weight 1.
    NN (per confident-novel B anchor): supervised contrastive on K_NEW labels,
        positives drawn ONLY from other confident-novel B samples in batch.
        This is the "tight" version of the old KN term.
    NA (per confident-novel B anchor): logsumexp over similarities to ALL A
        samples in the batch — pushes novel B away from known-old A samples.
        Asymmetric: A samples don't get gradient from this term.
    OB (per confident-old B anchor): supervised contrastive alignment to A
        samples whose K_OLD pseudo-label matches the B sample's old K_NEW label.
        Asymmetric: only confident-old B samples act as anchors.
    """
    N = z.size(0)
    eye = torch.eye(N, dtype=torch.bool, device=z.device)
    sim = (z @ z.T) / tau
    mx, _ = sim.max(dim=1, keepdim=True)
    exs   = torch.exp(sim - mx).masked_fill(eye, 0.0)
    denom = exs.sum(dim=1, keepdim=True) + 1e-8
    lp    = (sim - mx) - torch.log(denom)             # (N, N) log p(j | i)

    is_At    = is_A.to(z.device)
    is_cn    = is_conf_novel.to(z.device)
    is_co    = is_conf_old.to(z.device)
    a_pos    = is_At.nonzero(as_tuple=True)[0]
    cn_pos   = is_cn.nonzero(as_tuple=True)[0]
    co_pos   = is_co.nonzero(as_tuple=True)[0]
    n_a      = a_pos.numel()
    n_cn     = cn_pos.numel()
    n_co     = co_pos.numel()

    losses = []

    # ── AA term ──────────────────────────────────────────────────────────────
    aa_w = torch.zeros(N, N, device=z.device)
    if n_a > 1:
        y  = hard_y_A.to(z.device)
        wA = w_A.to(z.device)
        same = (y.unsqueeze(0) == y.unsqueeze(1)).float()
        pair = (wA.unsqueeze(0) * wA.unsqueeze(1)) * same
        ii, jj = torch.meshgrid(a_pos, a_pos, indexing="ij")
        aa_w[ii, jj] = pair
    aa_w.masked_fill_(eye, 0.0)
    aa_sum = aa_w.sum(dim=1)
    if (aa_sum > 0).any():
        has = aa_sum > 0
        l_aa = -((aa_w * lp).sum(dim=1)[has] / aa_sum[has].clamp_min(1e-8))
        losses.append(W_AA * l_aa.mean())

    # ── BB term ──────────────────────────────────────────────────────────────
    bb_w = bb_mask.to(z.device).float()
    bb_w.masked_fill_(eye, 0.0)
    bb_sum = bb_w.sum(dim=1)
    if (bb_sum > 0).any() and w_bb > 0:
        has = bb_sum > 0
        l_bb = -((bb_w * lp).sum(dim=1)[has] / bb_sum[has].clamp_min(1e-8))
        losses.append(w_bb * l_bb.mean())

    # ── NN term: SupCon within confident-novel B set ─────────────────────────
    if w_nn > 0 and n_cn > 1:
        y_kn = hard_y_kn_AB.to(z.device)
        nn_w = torch.zeros(N, N, device=z.device)
        same_kn = (y_kn.unsqueeze(0) == y_kn.unsqueeze(1)).float()
        ii, jj  = torch.meshgrid(cn_pos, cn_pos, indexing="ij")
        # Restrict to (cn × cn) pairs only
        block = same_kn[ii, jj]
        nn_w[ii, jj] = block
        nn_w.masked_fill_(eye, 0.0)
        nn_sum = nn_w.sum(dim=1)
        if (nn_sum > 0).any():
            has = nn_sum > 0
            l_nn = -((nn_w * lp).sum(dim=1)[has] / nn_sum[has].clamp_min(1e-8))
            losses.append(w_nn * l_nn.mean())

    # ── NA term: repulsion of confident-novel B from all A in batch ──────────
    if w_na > 0 and n_cn > 0 and n_a > 0:
        z_cn  = z[cn_pos]                                # (n_cn, d)
        z_a   = z[a_pos]                                 # (n_a, d)
        sim_cn_a = (z_cn @ z_a.T) / tau                  # (n_cn, n_a)
        # Loss per confident-novel anchor: log Σ_j exp(sim_ij)
        # Shifted by -log(n_a) to keep magnitude scale-invariant w.r.t. batch size.
        l_na = torch.logsumexp(sim_cn_a, dim=1) - np.log(max(n_a, 1))
        losses.append(w_na * l_na.mean())

    # ── OB term: alignment of confident-old B anchors to matching A samples ──
    if w_ob > 0 and n_co > 0 and n_a > 0:
        y_A_batch = hard_y_A.to(z.device)                # labels for a_pos order
        y_kn      = hard_y_kn_AB.to(z.device)            # labels for full batch
        ob_w = torch.zeros(N, N, device=z.device)

        # Rows: confident-old B anchors.  Cols: A samples in batch.
        # Positive when the B old K_NEW label equals the A K_OLD pseudo-label.
        match = (y_kn[co_pos].unsqueeze(1) == y_A_batch.unsqueeze(0)).float()
        ii, jj = torch.meshgrid(co_pos, a_pos, indexing="ij")
        ob_w[ii, jj] = match
        ob_w.masked_fill_(eye, 0.0)
        ob_sum = ob_w.sum(dim=1)
        if (ob_sum > 0).any():
            has = ob_sum > 0
            l_ob = -((ob_w * lp).sum(dim=1)[has] / ob_sum[has].clamp_min(1e-8))
            losses.append(w_ob * l_ob.mean())

    if not losses:
        return torch.tensor(0., device=z.device, requires_grad=True)
    return sum(losses)

# ──────────────────────────────────────────────────────────────────────────────
# Batch sampling
# ──────────────────────────────────────────────────────────────────────────────
N_A_PER_CLASS = 6
N_B_SEEDS     = 150

def sample_batch(pseudo_y_A, knn_B_current):
    """Sample a batch with class-balanced A and kNN-coupled B samples."""
    a_idx = []
    for k in range(K_OLD):
        pool = np.where(pseudo_y_A == k)[0]
        if len(pool):
            n = min(N_A_PER_CLASS, len(pool))
            a_idx.extend(rng.choice(pool, n, replace=False).tolist())
    seeds    = rng.choice(N_B, size=N_B_SEEDS, replace=False)
    partners = knn_B_current[seeds, rng.integers(0, knn_B_current.shape[1],
                                                 size=N_B_SEEDS)]
    b_idx    = np.unique(np.concatenate([seeds, partners]))
    return np.array(a_idx), b_idx

def build_bb_mask(b_idx, N_total, N_a, knn_B_current):
    """Build B-B kNN adjacency for the current batch."""
    mask  = torch.zeros(N_total, N_total, dtype=torch.bool)
    b_set = {int(bi): pos + N_a for pos, bi in enumerate(b_idx)}
    for pos_i, bi in enumerate(b_idx):
        pi = pos_i + N_a
        for kj in knn_B_current[bi]:
            if int(kj) in b_set:
                pj = b_set[int(kj)]
                mask[pi, pj] = True
                mask[pj, pi] = True
    return mask

# ──────────────────────────────────────────────────────────────────────────────
# Pseudo-label & soft-assignment generator (replaces tier scheme)
# ──────────────────────────────────────────────────────────────────────────────
def make_pseudo_labels(Z_A, target_max_p=0.7):
    """
    K-Means(K_OLD) on L2-normalized Z_A.

    Returns
    -------
    hard_y      : (N_A,) hard cluster assignment
    soft_p      : (N_A, K_OLD) softmax probs (sharp; for SoftMatch only)
    centers     : (K_OLD, d) L2-normalized centroids
    chosen_T    : float

    The temperature is selected by closed-form approximation to hit the
    target average max-prob, then verified.  Print is loud so failure is visible.
    """
    Z_n = normalize(Z_A, norm="l2").astype(np.float32)
    km  = KMeans(n_clusters=K_OLD, n_init=15, random_state=SEED).fit(Z_n)
    centers = normalize(km.cluster_centers_, norm="l2").astype(np.float32)
    sim     = (Z_n @ centers.T).astype(np.float32)        # (N_A, K_OLD), in [-1, 1]
    hard    = sim.argmax(axis=1)

    # Search T over a wide log range; pick the one whose ⟨max p⟩ is closest to target.
    best_T, best_ap, best_p = None, None, None
    for T in np.geomspace(1e-3, 0.5, 60):
        s = sim / T
        s = s - s.max(axis=1, keepdims=True)
        p = np.exp(s); p /= p.sum(axis=1, keepdims=True) + 1e-8
        ap = p.max(axis=1).mean()
        if best_T is None or abs(ap - target_max_p) < abs(best_ap - target_max_p):
            best_T, best_ap, best_p = T, ap, p
    return hard, best_p.astype(np.float32), centers, float(best_T), float(best_ap)

def make_joint_pseudo_labels(Z_anchor_A, Z_anchor_B, target_max_p=0.7):
    """
    K-Means(K_NEW=60) on L2-normalized concat of A and B anchor features.

    "Anchor features" means features built so as not to drift across rounds:
        round 0:  baseline UMAP only          (X_A_umap, X_B_base)
        round 1+: [proj || baseline UMAP]     (concat, both L2-normalized)

    Returns
    -------
    hard_AB     : (N_A + N_B,) hard cluster id ∈ [0, K_NEW)
    soft_AB     : (N_A + N_B, K_NEW) sharp softmax
    chosen_T    : float
    achieved_ap : float
    """
    Z_n = normalize(np.vstack([Z_anchor_A, Z_anchor_B]), norm="l2").astype(np.float32)
    km  = KMeans(n_clusters=K_NEW, n_init=15, random_state=SEED).fit(Z_n)
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
    return hard, best_p.astype(np.float32), float(best_T), float(best_ap)

def make_constrained_joint_labels(Z_anchor_A, Z_anchor_B, hard_y_A,
                                  novel_quantile=0.75, target_max_p=0.7,
                                  n_em_iters=20, verbose=True):
    """
    Two-stage K_NEW=60 clustering that explicitly separates "old-class
    classification" from "novel-class discovery".

    Additions for stability:
      • returns both confident-novel and confident-old B masks,
      • confident-old B is selected with independent geometric checks,
      • diagnostics expose old/novel assignment balance and mask quality.

    Procedure
    ─────────
    1. Build K_OLD=50 OLD prototypes as L2-normalized class means of A's
       anchor features, grouped by hard_y_A (current K_OLD pseudo-labels).

    2. For each B sample, compute its max cosine similarity to ANY old
       prototype.  Low max-sim ⇒ "outlier" wrt all known old classes
       ⇒ novel-class candidate.

    3. Take the bottom (1 − novel_quantile) fraction of B by max-sim — these
       are the strongest novel candidates.  We cluster ONLY this subset at
       K_NOV = K_NEW − K_OLD = 10 to seed novel prototypes.

    4. Run constrained K-Means(K_NEW): the first K_OLD centroids are PINNED
       to the old prototypes; the last K_NOV centroids are initialized from
       novel candidates and updated.

    5. Build confident masks:
       confident-novel B = novel assignment ∩ low old-prototype similarity.
       confident-old B   = old assignment ∩ high old-prototype similarity
                           ∩ high old top1/top2 margin
                           ∩ old beats novel centroids
                           ∩ K_NEW soft confidence above threshold.
    """
    K_NOV_LOC = K_NEW - K_OLD
    Z_A = normalize(Z_anchor_A, norm="l2").astype(np.float32)
    Z_B = normalize(Z_anchor_B, norm="l2").astype(np.float32)

    # ── Stage 1: build old prototypes from A ──────────────────────────────────
    old_protos = np.zeros((K_OLD, Z_A.shape[1]), dtype=np.float32)
    for k in range(K_OLD):
        mem = Z_A[hard_y_A == k]
        if len(mem):
            old_protos[k] = mem.mean(axis=0)
        else:
            # Rare but important for diagnostics.  Use a deterministic fallback
            # so the pipeline does not crash; empty clusters are printed upstream.
            old_protos[k] = Z_A[rng.integers(0, len(Z_A))]
    old_protos = normalize(old_protos, norm="l2").astype(np.float32)

    # ── Stage 2: max-sim of each B sample to old prototypes ───────────────────
    sim_B_to_old = Z_B @ old_protos.T              # (N_B, K_OLD)
    max_sim_B    = sim_B_to_old.max(axis=1)        # (N_B,)
    threshold    = np.quantile(max_sim_B, 1 - novel_quantile)
    novel_mask_B = max_sim_B < threshold
    n_novel_cand = int(novel_mask_B.sum())
    if verbose:
        print(f"       Novel candidates in B: {n_novel_cand} / {N_B} "
              f"(threshold max-sim < {threshold:.3f})")

    # ── Stage 3: seed K_NOV novel prototypes from candidate subset ────────────
    if n_novel_cand >= K_NOV_LOC:
        km_nov = KMeans(n_clusters=K_NOV_LOC, n_init=15,
                        random_state=SEED).fit(Z_B[novel_mask_B])
        novel_protos = normalize(km_nov.cluster_centers_, norm="l2").astype(np.float32)
    else:
        idx = rng.choice(N_B, size=K_NOV_LOC, replace=False)
        novel_protos = Z_B[idx].copy()

    # ── Stage 4: constrained K-Means with old protos pinned ───────────────────
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

    # ── Soft probs via temperature search ─────────────────────────────────────
    best_T, best_ap, best_p = None, None, None
    for T in np.geomspace(1e-3, 0.5, 60):
        s = sim / T
        s = s - s.max(axis=1, keepdims=True)
        p = np.exp(s); p /= p.sum(axis=1, keepdims=True) + 1e-8
        ap = p.max(axis=1).mean()
        if best_T is None or abs(ap - target_max_p) < abs(best_ap - target_max_p):
            best_T, best_ap, best_p = T, ap, p

    hard_B = hard_AB[N_A:]
    soft_B_max = best_p[N_A:].max(axis=1)

    # ── Confident-novel mask on B ─────────────────────────────────────────────
    crit1_novel_B = hard_B >= K_OLD
    crit2_novel_B = novel_mask_B
    confident_novel_B = crit1_novel_B & crit2_novel_B

    # ── Confident-old mask on B ───────────────────────────────────────────────
    final_sim_B_old = Z_B @ centers[:K_OLD].T
    final_sim_B_nov = Z_B @ centers[K_OLD:].T if K_NOV_LOC > 0 else np.full((N_B, 1), -np.inf)

    old_top2 = np.partition(final_sim_B_old, -2, axis=1)[:, -2:]
    second_old = old_top2[:, 0]
    max_old = old_top2[:, 1]
    old_margin = max_old - second_old
    max_novel = final_sim_B_nov.max(axis=1)
    old_vs_novel = max_old - max_novel

    old_label_B = hard_B < K_OLD
    if old_label_B.any():
        sim_thr = float(np.quantile(max_old[old_label_B], CONF_OLD_SIM_Q))
        margin_thr = float(np.quantile(old_margin[old_label_B], CONF_OLD_MARGIN_Q))
    else:
        sim_thr, margin_thr = np.inf, np.inf

    confident_old_B = (
        old_label_B
        & (max_old >= sim_thr)
        & (old_margin >= margin_thr)
        & (old_vs_novel > CONF_OLD_NOV_MARGIN_MIN)
        & (soft_B_max >= CONF_OLD_SOFT_MIN)
    )

    diagnostics = {
        "novel_candidate_threshold": float(threshold),
        "novel_candidate_count": int(n_novel_cand),
        "conf_old_sim_threshold": float(sim_thr),
        "conf_old_margin_threshold": float(margin_thr),
        "old_label_count_B": int(old_label_B.sum()),
        "novel_label_count_B": int(crit1_novel_B.sum()),
        "confident_old_count_B": int(confident_old_B.sum()),
        "confident_novel_count_B": int(confident_novel_B.sum()),
        "soft_B_max_mean": float(soft_B_max.mean()),
        "soft_B_max_old_label_mean": float(soft_B_max[old_label_B].mean()) if old_label_B.any() else float("nan"),
        "soft_B_max_novel_label_mean": float(soft_B_max[crit1_novel_B].mean()) if crit1_novel_B.any() else float("nan"),
        "max_old_mean": float(max_old.mean()),
        "old_margin_mean": float(old_margin.mean()),
        "old_vs_novel_mean": float(old_vs_novel.mean()),
    }

    return (hard_AB, best_p.astype(np.float32),
            confident_novel_B, confident_old_B,
            float(best_T), float(best_ap), diagnostics)


def joint_knn_from_proj(Z_A, Z_B, k=K_POS):
    """Build B-side kNN graph in the *projected* space (used from round 1+).

    We restrict neighbours to be drawn from B itself (so the graph has the
    same shape as the round-0 raw kNN graph), but the metric is now the
    learned representation.
    """
    Z_Bn = normalize(Z_B, norm="l2")
    nb   = NearestNeighbors(n_neighbors=k+1, metric="cosine",
                            n_jobs=-1).fit(Z_Bn)
    return nb.kneighbors(Z_Bn, return_distance=False)[:, 1:]


# ──────────────────────────────────────────────────────────────────────────────
# TWO-STAGE CURRICULUM TRAINING LOOP
#   Stage 1: A-only old-category discovery/refinement.
#   Stage 2: novel discovery on B using Stage-1 old model as anchor.
#   Stage 3: optional B-old adaptation with frozen novel geometry.
# ──────────────────────────────────────────────────────────────────────────────
import copy

# ── Curriculum hyper-parameters ───────────────────────────────────────────────
STAGE1_OLD_ROUNDS       = 4     # A-only old refinement rounds
STAGE1_EPOCHS_BOOT      = 100   # first A-only bootstrap round
STAGE1_EPOCHS_REFINE    = 50    # subsequent A-only prototype-refinement rounds
STAGE2_DISCOVERY_ROUNDS = 1     # normally 1; discovery tends to peak early
STAGE2_EPOCHS           = 50
STAGE3_ADAPT_ROUNDS     = 2     # optional old-B adaptation rounds
STAGE3_EPOCHS           = 25

STAGE1_LR_BOOT          = 3e-4
STAGE1_LR_REFINE        = 1e-4
STAGE2_LR               = 1e-4
STAGE3_LR               = 3e-5

# Stage-specific loss weights.
# Stage 1 uses AA only.
STAGE2_W_BB = 0.30
STAGE2_W_NN = 0.30
STAGE2_W_NA = 0.20
STAGE2_W_OB = 0.20

# Stage 3 freezes novel discovery and focuses on old B -> A transfer.
STAGE3_W_BB = 0.10
STAGE3_W_NN = 0.00
STAGE3_W_NA = 0.00
STAGE3_W_OB = 0.70
STAGE3_W_DISTILL_NOVEL = 1.00
STAGE3_W_DISTILL_ALL_B = 0.10

# Prototype reassignment / pruning.
PROTO_REASSIGN_TAU = 0.07
PRUNE_MIN_PER_CLUSTER = 30
PRUNE_KEEP_GOOD = 0.95
PRUNE_KEEP_MID  = 0.80
PRUNE_KEEP_LOW  = 0.60
PRUNE_KEEP_BAD  = 0.45

# Discovery anchor.  Higher = fewer B points selected as novel candidates.
DISCOVERY_NOVEL_QUANTILE = 0.83

# A-only stage does not use B, so use a larger class-balanced A batch.
N_A_PER_CLASS_STAGE1 = 8

Xt_A = torch.from_numpy(X_A)
Xt_B = torch.from_numpy(X_B)
all_results = [("Baseline (raw UMAP-10)", *res_baseline)]
knn_B_curr  = knn_B_oracle if ORACLE_MODE == "AB" else knn_B_raw


def one_hot_soft(labels, K, high=0.95):
    p = np.zeros((len(labels), K), dtype=np.float32)
    p[np.arange(len(labels)), labels.astype(np.int64)] = high
    p += (1.0 - high) / K
    p /= p.sum(axis=1, keepdims=True)
    return p.astype(np.float32)


def top2_margin_from_probs(p):
    top2 = np.partition(p, -2, axis=1)[:, -2:]
    return (top2[:, 1] - top2[:, 0]).astype(np.float32)


@torch.no_grad()
def extract_features(encoder):
    encoder.eval()
    Z_A = encoder(Xt_A.to(DEVICE)).cpu().numpy().astype(np.float32)
    Z_B = encoder(Xt_B.to(DEVICE)).cpu().numpy().astype(np.float32)
    return Z_A, Z_B


def fused_anchor_A(Z_A):
    return np.concatenate([
        normalize(Z_A, norm="l2"),
        normalize(X_A_umap, norm="l2"),
    ], axis=1).astype(np.float32)


def fused_anchor_B(Z_B):
    return np.concatenate([
        normalize(Z_B, norm="l2"),
        normalize(X_B_base, norm="l2"),
    ], axis=1).astype(np.float32)


def evaluate_B(tag, Z_B):
    print(f"  [eval] {tag}")
    res_proj = gcd_acc(Z_B, f"{tag} — projected")
    all_results.append((f"{tag}  proj", *res_proj))
    Z_B_fused = np.concatenate([
        normalize(Z_B,      norm="l2"),
        normalize(X_B_base, norm="l2"),
    ], axis=1).astype(np.float32)
    res_fuse = gcd_acc(Z_B_fused, f"{tag} — fused")
    all_results.append((f"{tag}  fuse", *res_fuse))


def compute_old_prototypes_np(Z_A, hard_y, weights=None):
    Z = normalize(Z_A, norm="l2").astype(np.float32)
    D = Z.shape[1]
    protos = np.zeros((K_OLD, D), dtype=np.float32)
    if weights is None:
        weights = np.ones(len(Z), dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    for k in range(K_OLD):
        idx = np.where(hard_y == k)[0]
        if len(idx) == 0:
            protos[k] = Z[rng.integers(0, len(Z))]
            continue
        w = weights[idx].clip(min=0.0)
        if w.sum() <= 1e-8:
            w = np.ones_like(w)
        protos[k] = (Z[idx] * w[:, None]).sum(0) / (w.sum() + 1e-8)
    return normalize(protos, norm="l2").astype(np.float32)


def prototype_reassign(Z_A, prototypes, tau=PROTO_REASSIGN_TAU):
    Z = normalize(Z_A, norm="l2").astype(np.float32)
    P = normalize(prototypes, norm="l2").astype(np.float32)
    sim = (Z @ P.T).astype(np.float32)
    hard = sim.argmax(axis=1).astype(np.int64)
    top2 = np.partition(sim, -2, axis=1)[:, -2:]
    margin = (top2[:, 1] - top2[:, 0]).astype(np.float32)

    logits = sim / tau
    logits = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p = p / (p.sum(axis=1, keepdims=True) + 1e-8)
    return hard, p.astype(np.float32), margin, sim


def rank01(x):
    x = np.asarray(x, dtype=np.float32)
    if len(x) <= 1:
        return np.ones_like(x, dtype=np.float32)
    lo, hi = np.percentile(x, 5), np.percentile(x, 95)
    return np.clip((x - lo) / (hi - lo + 1e-8), 0.0, 1.0).astype(np.float32)


def cluster_prune_mask(hard_y, soft_p, margin, prev_y=None, agreement_mask=None,
                       min_per_cluster=PRUNE_MIN_PER_CLUSTER, verbose=True):
    """
    Per-cluster pruning for A pseudo-label cleanup.

    A sample is scored by confidence, margin, low entropy, temporal stability,
    and optional cross-view agreement.  We keep a larger fraction of reliable
    clusters and a smaller fraction of unreliable clusters, but never fewer than
    min_per_cluster when a cluster has enough members.
    """
    hard_y = np.asarray(hard_y, dtype=np.int64)
    max_p = soft_p.max(axis=1).astype(np.float32)
    entropy = (-(soft_p * np.log(soft_p + 1e-8)).sum(axis=1) / np.log(K_OLD)).astype(np.float32)
    margin = np.asarray(margin, dtype=np.float32)
    if prev_y is None:
        temporal = np.ones(len(hard_y), dtype=np.float32)
    else:
        temporal = (np.asarray(prev_y, dtype=np.int64) == hard_y).astype(np.float32)
    if agreement_mask is None:
        agreement = np.ones(len(hard_y), dtype=np.float32)
    else:
        agreement = agreement_mask.astype(np.float32)

    keep = np.zeros(len(hard_y), dtype=bool)
    reliability = np.zeros(K_OLD, dtype=np.float32)
    keep_ratios = np.zeros(K_OLD, dtype=np.float32)

    for k in range(K_OLD):
        idx = np.where(hard_y == k)[0]
        if len(idx) == 0:
            reliability[k] = 0.0
            keep_ratios[k] = 0.0
            continue

        score = (
            0.35 * rank01(max_p[idx]) +
            0.25 * rank01(margin[idx]) +
            0.15 * (1.0 - rank01(entropy[idx])) +
            0.15 * temporal[idx] +
            0.10 * agreement[idx]
        ).astype(np.float32)

        med_score = float(np.median(score))
        reliability[k] = med_score
        if med_score >= 0.75:
            keep_ratio = PRUNE_KEEP_GOOD
        elif med_score >= 0.55:
            keep_ratio = PRUNE_KEEP_MID
        elif med_score >= 0.35:
            keep_ratio = PRUNE_KEEP_LOW
        else:
            keep_ratio = PRUNE_KEEP_BAD
        keep_ratios[k] = keep_ratio

        n_keep = int(np.ceil(len(idx) * keep_ratio))
        if len(idx) >= min_per_cluster:
            n_keep = max(min_per_cluster, n_keep)
        n_keep = min(len(idx), n_keep)
        chosen = idx[np.argsort(score)[-n_keep:]]
        keep[chosen] = True

    if verbose:
        counts = _safe_counts(hard_y, K_OLD)
        print(f"       A prune: kept={keep.sum()} / {len(keep)} ({keep.mean():.1%})")
        print(f"       A reliability: min={reliability.min():.3f}  med={np.median(reliability):.3f}  "
              f"max={reliability.max():.3f}")
        print(f"       A keep ratio:  min={keep_ratios.min():.2f}  med={np.median(keep_ratios):.2f}  "
              f"max={keep_ratios.max():.2f}")
        worst = np.argsort(reliability)[:5]
        print("       Worst A clusters: " + ", ".join(
            [f"k={k} rel={reliability[k]:.3f} size={counts[k]} keep={keep_ratios[k]:.2f}" for k in worst]
        ))
    return keep, reliability, keep_ratios


def sample_A_batch_stage1(pseudo_y_A, keep_mask):
    """Class-balanced A-only batch.  Falls back to all samples in a class if pruning removed too many."""
    a_idx = []
    for k in range(K_OLD):
        pool = np.where((pseudo_y_A == k) & keep_mask)[0]
        if len(pool) == 0:
            pool = np.where(pseudo_y_A == k)[0]
        if len(pool):
            n = min(N_A_PER_CLASS_STAGE1, len(pool))
            a_idx.extend(rng.choice(pool, n, replace=False).tolist())
    return np.array(a_idx, dtype=np.int64)


def sample_joint_batch_with_keep(pseudo_y_A, keep_mask, knn_B_current):
    a_idx = []
    for k in range(K_OLD):
        pool = np.where((pseudo_y_A == k) & keep_mask)[0]
        if len(pool) == 0:
            pool = np.where(pseudo_y_A == k)[0]
        if len(pool):
            n = min(N_A_PER_CLASS, len(pool))
            a_idx.extend(rng.choice(pool, n, replace=False).tolist())
    seeds    = rng.choice(N_B, size=N_B_SEEDS, replace=False)
    partners = knn_B_current[seeds, rng.integers(0, knn_B_current.shape[1], size=N_B_SEEDS)]
    b_idx    = np.unique(np.concatenate([seeds, partners])).astype(np.int64)
    return np.array(a_idx, dtype=np.int64), b_idx


def print_A_diagnostics(tag, hard_y, soft_p, keep_mask=None, prev_y=None):
    print(f"  [A-labels] {tag}")
    acc = pseudo_label_acc(hard_y, K_OLD)
    print(f"       A pseudo-label ACC: {acc:.1%}  (eval only)")
    label_count_summary(hard_y, K_OLD, "A pseudo-label")
    print(f"       soft max-p: mean={soft_p.max(1).mean():.3f}  "
          f"p10={np.percentile(soft_p.max(1), 10):.3f}  "
          f"p90={np.percentile(soft_p.max(1), 90):.3f}")
    if prev_y is not None:
        agree = (hard_y == prev_y).mean()
        print(f"       agreement vs previous A labels: {agree:.1%}")
    if keep_mask is not None:
        print(f"       active A samples: {keep_mask.sum()} / {len(keep_mask)} ({keep_mask.mean():.1%})")


def print_B_mask_diagnostics(tag, hard_kn, conf_novel_B, conf_old_B):
    kn_B = hard_kn[N_A:]
    print(f"  [B masks] {tag}")
    label_count_summary(kn_B, K_NEW, "B K_NEW label")
    mat = np.zeros((K_NEW, K_NEW), dtype=np.int64)
    for t, p in zip(y_B_eval, kn_B):
        mat[t, p] += 1
    r, c = linear_sum_assignment(-mat)
    kn_acc_B = mat[r, c].sum() / N_B
    novel_recall = (kn_B[is_novel_B] >= K_OLD).mean()
    old_purity = (kn_B[~is_novel_B] < K_OLD).mean()
    print(f"       K_NEW label ACC on B: {kn_acc_B:.1%}  (eval only)")
    print(f"       Novel B → novel-cluster recall: {novel_recall:.1%}  (eval only)")
    print(f"       Old   B → old-cluster purity:   {old_purity:.1%}  (eval only)")
    n_cn = int(conf_novel_B.sum())
    n_co = int(conf_old_B.sum())
    if n_cn:
        print(f"       confident-novel B: n={n_cn}  true-novel purity={is_novel_B[conf_novel_B].mean():.1%}  "
              f"recall={(conf_novel_B & is_novel_B).sum()/max(1,is_novel_B.sum()):.1%}  (eval only)")
    else:
        print("       confident-novel B: n=0")
    if n_co:
        print(f"       confident-old B:   n={n_co}  true-old purity={(~is_novel_B[conf_old_B]).mean():.1%}  "
              f"recall={(conf_old_B & ~is_novel_B).sum()/max(1,(~is_novel_B).sum()):.1%}  (eval only)")
    else:
        print("       confident-old B:   n=0")


def representation_probe_A(Z_A, hard_y, tag):
    Za_n = normalize(Z_A, norm="l2")
    intra, inter = [], []
    for k in range(0, K_OLD, 5):
        mem = Za_n[hard_y == k]
        if len(mem) < 2:
            continue
        d = np.linalg.norm(mem[:, None, :] - mem[None, :, :], axis=-1)
        intra.append(d[np.triu_indices(len(mem), k=1)].mean())
        inter.append(np.linalg.norm(Za_n[hard_y != k] - mem.mean(0), axis=1).mean())
    if intra and inter:
        print(f"       {tag} A repr.: intra={np.mean(intra):.4f}  "
              f"inter={np.mean(inter):.4f}  ratio={np.mean(inter)/np.mean(intra):.2f}x")


def train_phase(phase_name, epochs, lr, tau, hard_y, soft_p, keep_mask,
                mode="A_ONLY", hard_kn=None, conf_novel_B=None, conf_old_B=None,
                w_bb=0.0, w_nn=0.0, w_na=0.0, w_ob=0.0,
                distill_ref_Z_B=None, w_distill_novel=0.0, w_distill_all_b=0.0):
    """
    Generic phase trainer.
      mode="A_ONLY": AA-only training on A.
      mode="JOINT":  A+B training with BB/NN/NA/OB branches as requested.

    Distillation is applied to B projected embeddings against a frozen reference
    feature matrix, usually the Stage-2 discovery checkpoint, to prevent novel
    forgetting during Stage-3 old adaptation.
    """
    global head, teacher

    hard_y_t = torch.from_numpy(hard_y.astype(np.int64))
    soft_p_t = torch.from_numpy(soft_p.astype(np.float32))
    keep_mask = np.asarray(keep_mask, dtype=bool)

    sm = SoftMatch(n_classes=K_OLD, ema=SM_EMA, lam_dist=2.0)
    sm.warm_start(soft_p_t)
    proto_clf = PrototypeClassifier(n_classes=K_OLD, tau_proto=PROTO_REASSIGN_TAU)

    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = epochs * ITERS_PER_EPOCH
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps))

    print(f"  [train] {phase_name}: mode={mode}  epochs={epochs}×{ITERS_PER_EPOCH}  "
          f"lr={lr:.0e}  tau={tau}  w_bb={w_bb}  w_nn={w_nn}  w_na={w_na}  "
          f"w_ob={w_ob}  distill_nov={w_distill_novel}  distill_allB={w_distill_all_b}")
    print(f"       SoftMatch init: μ={sm.mu.item():.3f}  σ={sm.sigma2.sqrt().item():.3f}")
    if mode == "JOINT":
        assert hard_kn is not None and conf_novel_B is not None and conf_old_B is not None
        hard_kn_t = torch.from_numpy(hard_kn.astype(np.int64))
    else:
        hard_kn_t = None

    t0 = time.time()
    step = 0
    for ep in range(epochs):
        # Live prototype confidence update for A.
        proto_clf.update_prototypes(teacher, Xt_A, hard_y_t, DEVICE)
        teacher.eval()
        with torch.no_grad():
            z_A_all = teacher(Xt_A.to(DEVICE))
            p_live_all = proto_clf.predict(z_A_all)
            p_aligned_all = sm.align(p_live_all)
        sm.update(p_aligned_all)

        head.train()
        ep_loss, ep_lambda, ep_aa_active, ep_cn, ep_co = 0.0, 0.0, 0.0, 0.0, 0.0
        for _ in range(ITERS_PER_EPOCH):
            if mode == "A_ONLY":
                a_idx = sample_A_batch_stage1(hard_y, keep_mask)
                b_idx = np.array([], dtype=np.int64)
            else:
                a_idx, b_idx = sample_joint_batch_with_keep(hard_y, keep_mask, knn_B_curr)

            n_a, n_b = len(a_idx), len(b_idx)
            n_tot = n_a + n_b
            x_parts = [Xt_A[a_idx]]
            if n_b:
                x_parts.append(Xt_B[b_idx])
            x_batch = torch.cat(x_parts, dim=0).to(DEVICE)
            is_A = torch.cat([
                torch.ones(n_a, dtype=torch.bool),
                torch.zeros(n_b, dtype=torch.bool),
            ])
            bbm = build_bb_mask(b_idx, n_tot, n_a, knn_B_curr) if n_b else torch.zeros(n_tot, n_tot, dtype=torch.bool)

            hy_batch = hard_y_t[a_idx].to(DEVICE)
            if ORACLE_MODE != "off" or ep < SM_WARMUP:
                w_A_batch = torch.ones(n_a, device=DEVICE)
            else:
                w_A_batch = sm.weight(p_aligned_all[a_idx])
            # Hard pruning: removed A samples do not contribute as AA positives.
            w_A_batch = w_A_batch * torch.from_numpy(keep_mask[a_idx].astype(np.float32)).to(DEVICE)

            if mode == "JOINT" and n_b:
                kn_idx_AB = np.concatenate([a_idx, b_idx + N_A]).astype(np.int64)
                hy_kn_batch = hard_kn_t[kn_idx_AB].to(DEVICE)
                cn_b_batch = conf_novel_B[b_idx]
                co_b_batch = conf_old_B[b_idx]
                is_cn_batch = torch.cat([
                    torch.zeros(n_a, dtype=torch.bool),
                    torch.from_numpy(cn_b_batch),
                ]).to(DEVICE)
                is_co_batch = torch.cat([
                    torch.zeros(n_a, dtype=torch.bool),
                    torch.from_numpy(co_b_batch),
                ]).to(DEVICE)
            else:
                hy_kn_batch = torch.zeros(n_tot, dtype=torch.long, device=DEVICE)
                is_cn_batch = torch.zeros(n_tot, dtype=torch.bool, device=DEVICE)
                is_co_batch = torch.zeros(n_tot, dtype=torch.bool, device=DEVICE)
                cn_b_batch = np.zeros(n_b, dtype=bool)

            z = head(x_batch)
            loss = supcon_loss(z,
                               hy_batch, w_A_batch,
                               hy_kn_batch, is_cn_batch, is_co_batch,
                               is_A, bbm,
                               tau=tau, w_bb=w_bb, w_nn=w_nn, w_na=w_na, w_ob=w_ob)

            # Novel-preserving distillation for Stage 3.
            if n_b and distill_ref_Z_B is not None and (w_distill_novel > 0 or w_distill_all_b > 0):
                z_b = z[n_a:]
                z_ref = torch.from_numpy(distill_ref_Z_B[b_idx]).to(DEVICE)
                z_ref = F.normalize(z_ref, dim=-1)
                if w_distill_all_b > 0:
                    l_all = 1.0 - (z_b * z_ref).sum(dim=1).mean()
                    loss = loss + w_distill_all_b * l_all
                if w_distill_novel > 0 and cn_b_batch.any():
                    m = torch.from_numpy(cn_b_batch).to(DEVICE)
                    l_nov = 1.0 - (z_b[m] * z_ref[m]).sum(dim=1).mean()
                    loss = loss + w_distill_novel * l_nov

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            sch.step()
            ema_update(head, teacher, m=0.999)

            ep_loss += float(loss.item())
            ep_lambda += float(w_A_batch.mean().item()) if n_a else 0.0
            ep_aa_active += float((w_A_batch > 0).float().mean().item()) if n_a else 0.0
            ep_cn += float(is_cn_batch.sum().item())
            ep_co += float(is_co_batch.sum().item())
            step += 1

        ep_loss /= ITERS_PER_EPOCH
        ep_lambda /= ITERS_PER_EPOCH
        ep_aa_active /= ITERS_PER_EPOCH
        ep_cn /= ITERS_PER_EPOCH
        ep_co /= ITERS_PER_EPOCH
        if (ep + 1) % max(1, epochs // 5) == 0 or ep < SM_WARMUP:
            tag = "warm-up" if ep < SM_WARMUP else "       "
            print(f"       ep {ep+1:>3}/{epochs} {tag} loss={ep_loss:.4f}  "
                  f"⟨λ_A⟩={ep_lambda:.3f}  active_A={ep_aa_active:.1%}  "
                  f"μ={sm.mu.item():.3f}  σ={sm.sigma2.sqrt().item():.3f}  "
                  f"cn/batch={ep_cn:.1f}  co/batch={ep_co:.1f}")
    print(f"       Training time: {time.time()-t0:.1f}s ({step} grad steps)")


def build_stage2_labels(Z_A, Z_B, hard_y, tag):
    anchor_A = fused_anchor_A(Z_A)
    anchor_B = fused_anchor_B(Z_B)
    print(f"  [K_NEW] {tag}: constrained discovery on anchor dim={anchor_A.shape[1]}")
    if ORACLE_MODE == "AB":
        hard_kn = np.concatenate([y_A_eval, y_B_eval]).astype(np.int64)
        soft_kn = one_hot_soft(hard_kn, K_NEW)
        conf_novel_B = is_novel_B.copy()
        conf_old_B = ~is_novel_B.copy()
        T_kn, ap_kn = 0.0, soft_kn.max(1).mean()
        diag = {}
        print("       ORACLE-AB: using ground-truth K_NEW labels/masks")
    else:
        hard_kn, soft_kn, conf_novel_B, conf_old_B, T_kn, ap_kn, diag = make_constrained_joint_labels(
            anchor_A, anchor_B, hard_y,
            novel_quantile=DISCOVERY_NOVEL_QUANTILE,
            target_max_p=0.7,
            n_em_iters=20,
            verbose=True,
        )
    soft_kn_t = torch.from_numpy(soft_kn.astype(np.float32))
    sm_kn = SoftMatch(n_classes=K_NEW, ema=SM_EMA, lam_dist=2.0)
    sm_kn.warm_start(soft_kn_t)
    sm_kn.p_targ = make_knew_prior()
    print(f"       K_NEW soft T={T_kn:.4f}  ⟨max p⟩={ap_kn:.3f}  "
          f"SoftMatch μ={sm_kn.mu.item():.3f} σ={sm_kn.sigma2.sqrt().item():.3f}")
    if diag:
        print(f"       thresholds: novel_oldsim<{diag.get('novel_candidate_threshold', float('nan')):.3f}  "
              f"conf_old_sim>{diag.get('conf_old_sim_threshold', float('nan')):.3f}  "
              f"conf_old_margin>{diag.get('conf_old_margin_threshold', float('nan')):.3f}")
    print_B_mask_diagnostics(tag, hard_kn, conf_novel_B, conf_old_B)
    return hard_kn, soft_kn, conf_novel_B, conf_old_B, sm_kn


# ──────────────────────────────────────────────────────────────────────────────
# MAIN CURRICULUM
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*82)
print("TWO-STAGE GCD CURRICULUM")
print("  Stage 1: A-only old-category pseudo-label refinement")
print("  Stage 2: Novel discovery on B with Stage-1 old model as anchor")
print("  Stage 3: Old-B adaptation with frozen novel geometry")
print("="*82)

# Initialize model once.  Stage 1 learns the old specialist; Stage 2/3 continue from it.
head = ProjectionHead().to(DEVICE)
teacher = ProjectionHead().to(DEVICE)
teacher.load_state_dict(head.state_dict())
for p in teacher.parameters():
    p.requires_grad_(False)

# ── Stage 1 bootstrap labels from stable baseline UMAP ────────────────────────
print("\n" + "─"*82)
print("STAGE 1 — A-only old-category discovery/refinement")
print("─"*82)
print("  [bootstrap] KMeans(K_OLD) on A baseline UMAP")
hard_y, soft_p, centers_A, used_T, ach_max = make_pseudo_labels(X_A_umap, target_max_p=0.7)
if ORACLE_MODE in ("A", "AB"):
    print(f"  [ORACLE-{ORACLE_MODE}] Replacing A pseudo-labels with ground truth")
    hard_y = y_A_eval.copy().astype(np.int64)
    soft_p = one_hot_soft(hard_y, K_OLD)
margin = top2_margin_from_probs(soft_p)
keep_A = np.ones(N_A, dtype=bool)
prev_hard_y = None
Z_A, Z_B = None, None
print_A_diagnostics("bootstrap", hard_y, soft_p, keep_A)

for r in range(STAGE1_OLD_ROUNDS):
    print("\n" + "─"*82)
    print(f"STAGE 1 / OLD ROUND {r}")
    print("─"*82)

    if r > 0 and ORACLE_MODE == "off":
        # Prototype-based hard reassignment instead of fresh KMeans.
        # Prototypes are computed from the current teacher representation and
        # previous active labels, preserving label identity across rounds.
        protos = compute_old_prototypes_np(Z_A, hard_y, weights=keep_A.astype(np.float32))
        proto_y, proto_soft, proto_margin, _ = prototype_reassign(Z_A, protos, tau=PROTO_REASSIGN_TAU)

        # Cross-check against a fresh KMeans view, but do not let KMeans define labels.
        km_source = fused_anchor_A(Z_A)
        km_y_raw, km_soft_raw, _, _, _ = make_pseudo_labels(km_source, target_max_p=0.7)
        km_y_aligned, _, km_agree_to_proto = align_labels_to_previous(km_y_raw, proto_y, K_OLD)
        agreement_mask = (km_y_aligned == proto_y)

        prev_copy = hard_y.copy()
        hard_y = proto_y
        soft_p = proto_soft
        margin = proto_margin
        print(f"  [reassign] prototype labels adopted; fresh-KMeans cross-check agreement={agreement_mask.mean():.1%}")
        print(f"       agreement with previous A labels={(hard_y == prev_copy).mean():.1%}")
        keep_A, reliability, keep_ratios = cluster_prune_mask(
            hard_y, soft_p, margin, prev_y=prev_copy, agreement_mask=agreement_mask, verbose=True)
        prev_hard_y = prev_copy
    elif r == 0:
        print("  [reassign] bootstrap round: no pruning, no prototype reassignment yet")
    else:
        # Oracle A labels: no pruning needed.
        keep_A = np.ones(N_A, dtype=bool)

    print_A_diagnostics(f"Stage1 round {r} pre-train", hard_y, soft_p, keep_A, prev_hard_y)

    epochs = STAGE1_EPOCHS_BOOT if r == 0 else STAGE1_EPOCHS_REFINE
    lr = STAGE1_LR_BOOT if r == 0 else STAGE1_LR_REFINE
    tau = TAU_SCHEDULE[min(r, len(TAU_SCHEDULE)-1)]
    train_phase(
        phase_name=f"Stage1-old-r{r}",
        epochs=epochs,
        lr=lr,
        tau=tau,
        hard_y=hard_y,
        soft_p=soft_p,
        keep_mask=keep_A,
        mode="A_ONLY",
        w_bb=0.0, w_nn=0.0, w_na=0.0, w_ob=0.0,
    )

    Z_A, Z_B = extract_features(teacher)
    representation_probe_A(Z_A, hard_y, f"Stage1 r{r}")
    evaluate_B(f"Stage1 r{r}", Z_B)

# Final Stage-1 old-label refresh before discovery.
if ORACLE_MODE == "off":
    print("\n  [Stage1 final] final prototype reassignment + pruning before novel discovery")
    protos = compute_old_prototypes_np(Z_A, hard_y, weights=keep_A.astype(np.float32))
    final_y, final_soft, final_margin, _ = prototype_reassign(Z_A, protos, tau=PROTO_REASSIGN_TAU)
    agreement_mask = (final_y == hard_y)
    keep_A, reliability, keep_ratios = cluster_prune_mask(
        final_y, final_soft, final_margin, prev_y=hard_y, agreement_mask=agreement_mask, verbose=True)
    hard_y, soft_p, margin = final_y, final_soft, final_margin
else:
    hard_y = y_A_eval.copy().astype(np.int64)
    soft_p = one_hot_soft(hard_y, K_OLD)
    keep_A = np.ones(N_A, dtype=bool)
print_A_diagnostics("Stage1 final", hard_y, soft_p, keep_A)

# Save the old-specialist checkpoint.
stage1_state = copy.deepcopy(teacher.state_dict())
Z_A_stage1 = Z_A.copy()
Z_B_stage1 = Z_B.copy()
hard_y_stage1 = hard_y.copy()
soft_p_stage1 = soft_p.copy()
keep_A_stage1 = keep_A.copy()

# ── Stage 2: novel discovery using old-specialized anchor ────────────────────
print("\n" + "="*82)
print("STAGE 2 — novel discovery with Stage-1 old model as anchor")
print("="*82)
hard_kn, soft_kn, conf_novel_B, conf_old_B, sm_kn = build_stage2_labels(
    Z_A_stage1, Z_B_stage1, hard_y_stage1, tag="Stage2 pre-train discovery labels")

for r in range(STAGE2_DISCOVERY_ROUNDS):
    print("\n" + "─"*82)
    print(f"STAGE 2 / DISCOVERY ROUND {r}")
    print("─"*82)
    train_phase(
        phase_name=f"Stage2-discovery-r{r}",
        epochs=STAGE2_EPOCHS,
        lr=STAGE2_LR,
        tau=0.10,
        hard_y=hard_y_stage1,
        soft_p=soft_p_stage1,
        keep_mask=keep_A_stage1,
        mode="JOINT",
        hard_kn=hard_kn,
        conf_novel_B=conf_novel_B,
        conf_old_B=conf_old_B,
        w_bb=STAGE2_W_BB,
        w_nn=STAGE2_W_NN,
        w_na=STAGE2_W_NA,
        w_ob=STAGE2_W_OB,
    )
    Z_A, Z_B = extract_features(teacher)
    representation_probe_A(Z_A, hard_y_stage1, f"Stage2 r{r}")
    evaluate_B(f"Stage2 r{r}", Z_B)

# Recompute and freeze discovery labels/masks at the Stage-2 checkpoint.
print("\n  [Stage2 final] recomputing K_NEW labels/masks to freeze novel geometry")
hard_kn_s2, soft_kn_s2, conf_novel_B_s2, conf_old_B_s2, sm_kn_s2 = build_stage2_labels(
    Z_A, Z_B, hard_y_stage1, tag="Stage2 final frozen labels")
stage2_state = copy.deepcopy(teacher.state_dict())
Z_A_stage2 = Z_A.copy()
Z_B_stage2 = Z_B.copy()
stage2_ref_Z_B = Z_B_stage2.copy()

# ── Stage 3: optional old-B adaptation with frozen novel geometry ────────────
print("\n" + "="*82)
print("STAGE 3 — old-B adaptation, frozen novel labels, novel distillation")
print("="*82)
print("  Stage 3 disables NN and NA.  It uses AA + OB + small BB, plus distillation")
print("  toward the Stage-2 B representation for confident-novel B samples.")

hard_y_adapt = hard_y_stage1.copy()
soft_p_adapt = soft_p_stage1.copy()
keep_A_adapt = keep_A_stage1.copy()
prev_adapt_y = hard_y_adapt.copy()

for r in range(STAGE3_ADAPT_ROUNDS):
    print("\n" + "─"*82)
    print(f"STAGE 3 / OLD-B ADAPT ROUND {r}")
    print("─"*82)

    # Refresh A labels by prototype reassignment in the current representation,
    # but do not run fresh KMeans as the source of truth.
    if ORACLE_MODE == "off":
        protos = compute_old_prototypes_np(Z_A, hard_y_adapt, weights=keep_A_adapt.astype(np.float32))
        new_y, new_soft, new_margin, _ = prototype_reassign(Z_A, protos, tau=PROTO_REASSIGN_TAU)
        agreement_mask = (new_y == hard_y_adapt)
        keep_A_adapt, reliability, keep_ratios = cluster_prune_mask(
            new_y, new_soft, new_margin, prev_y=hard_y_adapt, agreement_mask=agreement_mask, verbose=True)
        prev_adapt_y = hard_y_adapt.copy()
        hard_y_adapt, soft_p_adapt = new_y, new_soft
    else:
        hard_y_adapt = y_A_eval.copy().astype(np.int64)
        soft_p_adapt = one_hot_soft(hard_y_adapt, K_OLD)
        keep_A_adapt = np.ones(N_A, dtype=bool)

    print_A_diagnostics(f"Stage3 round {r} pre-train", hard_y_adapt, soft_p_adapt,
                        keep_A_adapt, prev_y=prev_adapt_y)
    print_B_mask_diagnostics(f"Stage3 round {r} frozen B masks", hard_kn_s2, conf_novel_B_s2, conf_old_B_s2)

    # Measure novel drift before training this round.
    if conf_novel_B_s2.any():
        cur = normalize(Z_B[conf_novel_B_s2], norm="l2")
        ref = normalize(stage2_ref_Z_B[conf_novel_B_s2], norm="l2")
        drift = 1.0 - (cur * ref).sum(axis=1).mean()
        print(f"       pre-train novel drift from Stage2: {drift:.4f}  (lower is better)")

    train_phase(
        phase_name=f"Stage3-old-adapt-r{r}",
        epochs=STAGE3_EPOCHS,
        lr=STAGE3_LR,
        tau=0.09,
        hard_y=hard_y_adapt,
        soft_p=soft_p_adapt,
        keep_mask=keep_A_adapt,
        mode="JOINT",
        hard_kn=hard_kn_s2,
        conf_novel_B=conf_novel_B_s2,
        conf_old_B=conf_old_B_s2,
        w_bb=STAGE3_W_BB,
        w_nn=STAGE3_W_NN,
        w_na=STAGE3_W_NA,
        w_ob=STAGE3_W_OB,
        distill_ref_Z_B=stage2_ref_Z_B,
        w_distill_novel=STAGE3_W_DISTILL_NOVEL,
        w_distill_all_b=STAGE3_W_DISTILL_ALL_B,
    )

    Z_A, Z_B = extract_features(teacher)
    representation_probe_A(Z_A, hard_y_adapt, f"Stage3 r{r}")
    if conf_novel_B_s2.any():
        cur = normalize(Z_B[conf_novel_B_s2], norm="l2")
        ref = normalize(stage2_ref_Z_B[conf_novel_B_s2], norm="l2")
        drift = 1.0 - (cur * ref).sum(axis=1).mean()
        print(f"       post-train novel drift from Stage2: {drift:.4f}  (lower is better)")
    evaluate_B(f"Stage3 r{r}", Z_B)

# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*82)
print("SUMMARY — GCD metrics across curriculum")
print("="*82)
print(f"  {'Method':<45} {'All':>7} {'Old':>7} {'Novel':>9}")
print("  " + "-"*70)

best_nov = max(r[3] for r in all_results)
best_all = max(r[1] for r in all_results)
best_old = max(r[2] for r in all_results)
for tag, a, o, n in all_results:
    marks = []
    if n == best_nov: marks.append("Novel◄")
    if o == best_old: marks.append("Old◄")
    if a == best_all: marks.append("All◄")
    mark = "  " + "/".join(marks) if marks else ""
    print(f"  {tag:<45} {a:>7.1%} {o:>7.1%} {n:>9.1%}{mark}")

print()
print("  Δ vs baseline (best by All):")
best = max(all_results[1:], key=lambda x: x[1])
for metric, idx in [("All", 1), ("Old", 2), ("Novel", 3)]:
    delta = best[idx] - all_results[0][idx]
    print(f"    {metric}: {all_results[0][idx]:.1%} → {best[idx]:.1%}  ({delta:+.1%})")

print("\n  Best checkpoints:")
print(f"    Best All:   {max(all_results, key=lambda x: x[1])[0]}")
print(f"    Best Old:   {max(all_results, key=lambda x: x[2])[0]}")
print(f"    Best Novel: {max(all_results, key=lambda x: x[3])[0]}")
