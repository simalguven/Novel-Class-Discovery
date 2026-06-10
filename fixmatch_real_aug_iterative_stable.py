"""
Iterative pseudo-label refinement with FixMatch real image augmentations.

Key changes vs feature-only refinement:
  - Uses weak/strong image augmentations instead of feature MixUp.
  - Runs images through a DINOv2 backbone plus LoRA adapters.
  - Adds a K_OLD pseudo-cluster classifier for FixMatch consistency on A.
  - Keeps representation learning objective through pseudo-label SupCon on A.
  - Keeps B safe for GCD by avoiding old-class FixMatch labels on B; B gets
    augmentation consistency + optional raw-DINO mutual-nearest-neighbor positives.

Labels are used only for evaluation/sampling exactly as in the original scripts.
"""

import os
import time
import math
import warnings
from contextlib import nullcontext
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import umap
from PIL import Image, ImageFile
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from torch.utils.checkpoint import checkpoint

warnings.filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ── User configuration ─────────────────────────────────────────────────────────
IMAGE_INDEX  = "plantnet_image_paths.txt"
DINO_MODEL   = "dinov2_vitb14"

LORA_RANK    = 8
LORA_ALPHA   = 16.0

LR_HEAD_0    = 3e-4
LR_HEAD_R    = 1e-4
LR_BB_0      = 2e-6    # conservative: raw DINOv2 baseline is already strong
LR_BB_R      = 1e-6

# Diagnostic: fully freeze backbone in round 0 to verify the projection head
# alone matches the UMAP+KMeans baseline.
FREEZE_BACKBONE_R0 = True
TRAIN_LORA          = False   # default: train head/classifier only; enable after frozen-head ablation wins

MNN_K              = 3
MNN_WEIGHT         = 0.2
MNN_OLD_ONLY       = True       # implemented as a conservative centroid-distance filter
MNN_PARTNERS_PER_B = 1          # cap forced B-neighbor partners per sampled B

N_A_PER_CLS  = 1
N_B_STEP     = 16
N_MNN_STEP   = 8
EMB_BATCH    = 32

OLD_CENTROID_BLEND = 0.5        # 0 = no stabilisation, 1 = freeze old centroids

USE_AMP        = True
AMP_DTYPE      = torch.bfloat16
USE_CHECKPOINT = True

# FixMatch / representation losses
ROUNDS       = 6
EPOCHS_0     = 100
EPOCHS_R     = 50
FIXMATCH_TAU       = 0.80     # conservative final threshold; do not force bad pseudo-labels
FIXMATCH_TAU_START = 0.50     # warmup threshold while the pseudo-cluster classifier calibrates
FIXMATCH_TAU_WARM  = 20       # epochs to ramp START -> final TAU
FIXMATCH_MIN_KEEP  = 0        # real FixMatch: skip samples below threshold rather than forcing them
USE_CLUSTER_TARGET_FM = True   # use K-Means pseudo-label as the strong-view target
LAMBDA_FM          = 0.10     # weak→strong consistency CE; keep secondary to representation learning
LAMBDA_ANCH        = 0.25     # CE to round-level cluster pseudo-labels; avoid overfitting noisy clusters
LAMBDA_A_SUP       = 1.0      # A pseudo-label SupCon on weak+strong views
LAMBDA_B_AUG       = 0.05     # B same-image weak↔strong contrastive consistency
LAMBDA_B_MNN       = 0.20     # raw-DINO MNN positives; useful but should not dominate novel discovery
SUPCON_TAU         = 0.1
CE_TAU             = 1.0      # temperature before classifier softmax
CLASSIFIER_LOGIT_SCALE = 3.0  # enough confidence without making CE dominate the embedding
REQUIRE_CLUSTER_AGREE = False # True is very strict early; use cluster-confidence instead
ANCHOR_TIER_MIN = 2           # train classifier only on high-confidence pseudo-clusters
SUPCON_TIER_MIN = 1           # SupCon can still use high+mid confidence samples

# ──────────────────────────────────────────────────────────────────────────────
SEED = 42
K_OLD = 50
K_NEW = 60
K_NOV = 10
N_PER_CLS = 100

rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

if DEVICE.type != "cuda":
    USE_AMP = False

# ── Transforms ─────────────────────────────────────────────────────────────────
# FixMatch needs asymmetric augmentation: pseudo-label from weak view, train on strong view.
weak_aug = T.Compose([
    T.RandomResizedCrop(224, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC),
    T.RandomHorizontalFlip(p=0.5),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

strong_aug = T.Compose([
    # Fine-grained plant classes are sensitive to cropping; 0.2 often removes diagnostic parts.
    T.RandomResizedCrop(224, scale=(0.6, 1.0), interpolation=T.InterpolationMode.BICUBIC),
    T.RandomHorizontalFlip(p=0.5),
    T.RandAugment(num_ops=2, magnitude=5),
    T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4,
                                 saturation=0.2, hue=0.1)], p=0.8),
    T.RandomGrayscale(p=0.2),
    T.RandomApply([T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Kept under the original variable name so older code using train_aug still works.
train_aug = strong_aug

eval_tfm = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── Utility ────────────────────────────────────────────────────────────────────
def seed_worker(worker_id: int) -> None:
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def amp_context():
    if USE_AMP and DEVICE.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=AMP_DTYPE)
    return nullcontext()


def load_pil(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_image_batch(paths: Sequence[str], global_indices: np.ndarray, tfm: T.Compose) -> torch.Tensor:
    """Small-batch image loader used by the custom sampler."""
    imgs = []
    for gi in global_indices.tolist():
        try:
            img = load_pil(paths[int(gi)])
        except Exception as exc:
            raise RuntimeError(f"Failed to read image index {int(gi)}: {paths[int(gi)]}") from exc
        imgs.append(tfm(img))
    return torch.stack(imgs, dim=0)


def l2n(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return F.normalize(x, dim=-1, eps=eps)


def zero_loss(device: torch.device) -> torch.Tensor:
    return torch.tensor(0.0, device=device, requires_grad=True)

# ── DINOv2 + LoRA ──────────────────────────────────────────────────────────────
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be > 0")
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_B(self.lora_A(x)) * self.scale


def inject_lora_qkv(model: nn.Module, rank: int, alpha: float) -> int:
    """Inject LoRA into DINOv2 attention qkv layers."""
    replaced = 0
    for module in model.modules():
        for child_name, child in list(module.named_children()):
            if child_name == "qkv" and isinstance(child, nn.Linear):
                setattr(module, child_name, LoRALinear(child, rank=rank, alpha=alpha))
                replaced += 1
    return replaced


def freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)


def set_lora_trainable(model: nn.Module, trainable: bool) -> None:
    for m in model.modules():
        if isinstance(m, LoRALinear):
            for p in m.lora_A.parameters():
                p.requires_grad_(trainable)
            for p in m.lora_B.parameters():
                p.requires_grad_(trainable)


def lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    params: List[nn.Parameter] = []
    for m in model.modules():
        if isinstance(m, LoRALinear):
            params.extend([p for p in m.lora_A.parameters() if p.requires_grad])
            params.extend([p for p in m.lora_B.parameters() if p.requires_grad])
    return params


def dinov2_features(backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Return CLS features for torch.hub DINOv2 models."""
    if hasattr(backbone, "forward_features"):
        out = backbone.forward_features(x)
        if isinstance(out, dict):
            if "x_norm_clstoken" in out:
                return out["x_norm_clstoken"]
            if "x_prenorm" in out:
                return out["x_prenorm"][:, 0]
        if torch.is_tensor(out):
            return out
    out = backbone(x)
    if isinstance(out, dict):
        if "x_norm_clstoken" in out:
            return out["x_norm_clstoken"]
    return out


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int = 768, hidden: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return l2n(self.net(x))


class FixMatchDINO(nn.Module):
    def __init__(self, backbone: nn.Module, feat_dim: int = 768, proj_dim: int = 128, n_cls: int = K_OLD):
        super().__init__()
        self.backbone = backbone
        self.head = ProjectionHead(in_dim=feat_dim, out_dim=proj_dim)
        self.classifier = nn.Linear(proj_dim, n_cls)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if USE_CHECKPOINT and self.training and DEVICE.type == "cuda":
            return checkpoint(lambda y: dinov2_features(self.backbone, y), x, use_reentrant=False)
        return dinov2_features(self.backbone, x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encode(x)
        z = self.head(h)
        logits = (CLASSIFIER_LOGIT_SCALE * self.classifier(z)) / CE_TAU
        return h, z, logits

# ── Losses ────────────────────────────────────────────────────────────────────
def fixmatch_tau_for_epoch(ep: int) -> float:
    if FIXMATCH_TAU_WARM <= 0:
        return float(FIXMATCH_TAU)
    progress = min(1.0, float(ep + 1) / float(FIXMATCH_TAU_WARM))
    return float(FIXMATCH_TAU_START + progress * (FIXMATCH_TAU - FIXMATCH_TAU_START))


def force_min_keep(mask: torch.Tensor, scores: torch.Tensor, candidates: torch.Tensor, min_keep: int) -> torch.Tensor:
    """Ensure FixMatch has a small curriculum signal even before high confidence emerges."""
    if min_keep <= 0:
        return mask
    cand_idx = torch.where(candidates)[0]
    if cand_idx.numel() == 0 or mask.sum().item() >= min_keep:
        return mask
    k = min(int(min_keep), int(cand_idx.numel()))
    top_local = torch.topk(scores[cand_idx], k=k, largest=True).indices
    forced = cand_idx[top_local]
    mask = mask.clone()
    mask[forced] = True
    return mask


def weighted_ce(logits: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    if logits.numel() == 0:
        return zero_loss(logits.device)
    loss = F.cross_entropy(logits, target, reduction="none")
    if weight is not None:
        w = weight.to(logits.device).float().clamp_min(0.0)
        return (loss * w).sum() / w.sum().clamp_min(1e-8)
    return loss.mean()


def contrastive_from_mask(z: torch.Tensor, pos_mask: torch.Tensor, tau: float = SUPCON_TAU) -> torch.Tensor:
    """Multi-positive contrastive loss from a boolean/weighted positive mask."""
    n = z.size(0)
    if n <= 1:
        return zero_loss(z.device)
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    pos = pos_mask.to(z.device).float().masked_fill(eye, 0.0)
    has_pos = pos.sum(1) > 0
    if not has_pos.any():
        return zero_loss(z.device)

    sim = (z @ z.T) / tau
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    exp_sim = torch.exp(sim).masked_fill(eye, 0.0)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True).clamp_min(1e-8))
    loss = -((pos * log_prob).sum(1) / pos.sum(1).clamp_min(1e-8))
    return loss[has_pos].mean()


def a_pseudo_supcon_loss(z_w: torch.Tensor,
                         z_s: torch.Tensor,
                         pseudo_lbl: torch.Tensor,
                         conf: torch.Tensor,
                         tier: torch.Tensor) -> torch.Tensor:
    """SupCon on A using two real augmented views and current cluster pseudo-labels."""
    valid = tier >= SUPCON_TIER_MIN
    if valid.sum() < 2:
        return zero_loss(z_w.device)
    z = torch.cat([z_w, z_s], dim=0)
    labels = torch.cat([pseudo_lbl, pseudo_lbl], dim=0).to(z.device)
    valid2 = torch.cat([valid, valid], dim=0).to(z.device)
    conf2 = torch.cat([conf, conf], dim=0).to(z.device).float()

    same = labels[:, None] == labels[None, :]
    valid_pair = valid2[:, None] & valid2[None, :]
    pos = same & valid_pair
    weight = (conf2[:, None] * conf2[None, :]) * pos.float()
    return contrastive_from_mask(torch.cat([z_w, z_s], dim=0), weight, tau=SUPCON_TAU)


def two_view_instance_contrast(z_w: torch.Tensor, z_s: torch.Tensor) -> torch.Tensor:
    """InfoNCE-style same-image positive loss for B, avoids forcing novel samples into old clusters."""
    n = z_w.size(0)
    if n == 0:
        return zero_loss(z_w.device)
    z = torch.cat([z_w, z_s], dim=0)
    pos = torch.zeros(2 * n, 2 * n, dtype=torch.bool, device=z.device)
    idx = torch.arange(n, device=z.device)
    pos[idx, idx + n] = True
    pos[idx + n, idx] = True
    return contrastive_from_mask(z, pos, tau=SUPCON_TAU)


def expand_b_pos_mask(base_mask: torch.Tensor, include_self_views: bool = True) -> torch.Tensor:
    """Expand an N×N B-neighbor mask to a 2N×2N weak/strong-view mask."""
    n = base_mask.size(0)
    out = torch.zeros(2 * n, 2 * n, dtype=torch.float32)
    if n == 0:
        return out
    # Neighbor positives across all view combinations.
    for r_off in [0, n]:
        for c_off in [0, n]:
            out[r_off:r_off+n, c_off:c_off+n] = base_mask.float()
    if include_self_views:
        idx = torch.arange(n)
        out[idx, idx + n] = 1.0
        out[idx + n, idx] = 1.0
    return out

# ── Evaluation ────────────────────────────────────────────────────────────────
def gcd_acc(feat_B: np.ndarray, y_B_eval: np.ndarray, is_novel_B: np.ndarray,
            tag: str = "", n_init: int = 20, verbose: bool = True) -> Tuple[float, float, float]:
    km = KMeans(n_clusters=K_NEW, n_init=n_init, random_state=SEED)
    preds = km.fit_predict(normalize(feat_B, norm="l2"))
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


def pseudo_label_acc(pseudo_y: np.ndarray, y_A_eval: np.ndarray) -> float:
    mat = np.zeros((K_OLD, K_OLD), dtype=np.int64)
    for p, t in zip(pseudo_y, y_A_eval):
        mat[p % K_OLD, t] += 1
    r, c = linear_sum_assignment(-mat)
    return mat[r, c].sum() / len(y_A_eval)

# ── Clustering / confidence ───────────────────────────────────────────────────
def align_centers_and_labels(new_centers: np.ndarray,
                             new_labels: np.ndarray,
                             prev_centers: Optional[np.ndarray],
                             prev_labels: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Align K-Means ids to the previous round.

    The A set is fixed across rounds, so the safest id alignment is by sample
    overlap, not by centroid geometry. This also handles the UMAP-10 → 128-D
    transition where centroid-space alignment is impossible. If previous labels
    are unavailable, fall back to centroid similarity when dimensions match.
    """
    aligned_centers = np.zeros_like(new_centers)

    if prev_labels is not None and len(prev_labels) == len(new_labels):
        overlap = np.zeros((K_OLD, K_OLD), dtype=np.int64)
        for old_l, new_l in zip(prev_labels.astype(np.int64), new_labels.astype(np.int64)):
            if 0 <= old_l < K_OLD and 0 <= new_l < K_OLD:
                overlap[old_l, new_l] += 1
        old_ids, new_ids = linear_sum_assignment(-overlap)
        remap = {int(new_id): int(old_id) for old_id, new_id in zip(old_ids, new_ids)}
        for new_id, old_id in remap.items():
            aligned_centers[old_id] = new_centers[new_id]
        aligned_labels = np.array([remap[int(l)] for l in new_labels], dtype=np.int64)
        matched = overlap[old_ids, new_ids].sum() / max(1, len(new_labels))
        print(f"       Aligned cluster ids by A-sample overlap: {matched:.1%} assignment agreement")
        return aligned_centers, aligned_labels

    if prev_centers is None:
        return new_centers, new_labels
    if prev_centers.shape[1] != new_centers.shape[1]:
        print(f"       Cannot centroid-align ids: previous dim={prev_centers.shape[1]}, current dim={new_centers.shape[1]}")
        return new_centers, new_labels

    prev_n = normalize(prev_centers, norm="l2")
    new_n = normalize(new_centers, norm="l2")
    sim = prev_n @ new_n.T
    old_ids, new_ids = linear_sum_assignment(-sim)
    remap = {}
    for old_id, new_id in zip(old_ids, new_ids):
        aligned_centers[old_id] = new_centers[new_id]
        remap[int(new_id)] = int(old_id)
    aligned_labels = np.array([remap[int(l)] for l in new_labels], dtype=np.int64)
    return aligned_centers, aligned_labels


def cluster_and_confidence(Z_A_current: np.ndarray,
                           prev_centers: Optional[np.ndarray],
                           prev_labels: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    km50 = KMeans(n_clusters=K_OLD, n_init=15, random_state=SEED).fit(
        normalize(Z_A_current, norm="l2")
    )
    pseudo_y = km50.labels_.astype(np.int64)
    centers = km50.cluster_centers_.astype(np.float32)
    centers, pseudo_y = align_centers_and_labels(centers, pseudo_y, prev_centers, prev_labels)

    if (prev_centers is not None and OLD_CENTROID_BLEND > 0 and
            prev_centers.shape[1] == centers.shape[1]):
        centers = normalize(
            OLD_CENTROID_BLEND * normalize(prev_centers, norm="l2") +
            (1.0 - OLD_CENTROID_BLEND) * normalize(centers, norm="l2"),
            norm="l2",
        ).astype(np.float32)
        Z_norm_tmp = normalize(Z_A_current, norm="l2").astype(np.float32)
        dist_tmp = np.linalg.norm(Z_norm_tmp[:, None, :] - centers[None, :, :], axis=-1)
        pseudo_y = dist_tmp.argmin(axis=1).astype(np.int64)

    Z_norm = normalize(Z_A_current, norm="l2").astype(np.float32)
    centers_n = normalize(centers, norm="l2").astype(np.float32)
    dist_A = np.linalg.norm(Z_norm[:, None, :] - centers_n[None, :, :], axis=-1)
    own_dist = dist_A[np.arange(len(Z_A_current)), pseudo_y]
    temp = 0.1
    soft_A = np.exp(-dist_A / temp)
    soft_A /= soft_A.sum(1, keepdims=True)
    ss = np.sort(soft_A, axis=1)
    margin = ss[:, -1] - ss[:, -2]
    cd = 1 - (own_dist - own_dist.min()) / (own_dist.max() - own_dist.min() + 1e-9)
    cm = (margin - margin.min()) / (margin.max() - margin.min() + 1e-9)
    conf = ((cd + cm) / 2).astype(np.float32)
    p30, p70 = np.percentile(conf, 30), np.percentile(conf, 70)
    tier = np.where(conf >= p70, 2, np.where(conf >= p30, 1, 0)).astype(np.int64)
    return pseudo_y, centers_n, conf, tier

# ── Sampling / MNN ─────────────────────────────────────────────────────────────
def build_mnn_graph(X_B: np.ndarray, k: int) -> List[np.ndarray]:
    print(f"\nPre-computing B mutual-{k}NN graph in raw DINOv2 space …")
    t0 = time.time()
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="cosine", n_jobs=-1).fit(X_B)
    knn = nbrs.kneighbors(X_B, return_distance=False)[:, 1:]
    knn_sets = [set(row.tolist()) for row in knn]
    graph: List[np.ndarray] = []
    for i in range(len(X_B)):
        mutual = [j for j in knn[i].tolist() if i in knn_sets[j]]
        graph.append(np.array(mutual, dtype=np.int64))
    avg_deg = np.mean([len(g) for g in graph])
    print(f"  Done in {time.time() - t0:.1f}s | avg mutual degree={avg_deg:.2f}")
    return graph


def compute_b_old_candidate_mask(X_B: np.ndarray,
                                 centers_A_raw: np.ndarray,
                                 X_A_raw: np.ndarray,
                                 pseudo_y: np.ndarray,
                                 q: float = 0.90) -> np.ndarray:
    """Conservative old-candidate filter for B when MNN_OLD_ONLY=True.

    This filter deliberately operates in raw DINOv2 feature space. The round-level
    K-Means centers may be UMAP-10 in round 0 or 128-D projected centers in later
    rounds, so they must not be used directly as raw 768-D centroids. Instead, raw
    A centroids are recomputed from the current pseudo-label assignments.
    """
    XA = normalize(X_A_raw, norm="l2")
    XB = normalize(X_B, norm="l2")
    cent = np.zeros((K_OLD, XA.shape[1]), dtype=np.float32)
    global_mean = XA.mean(0)
    for k in range(K_OLD):
        members = XA[pseudo_y == k]
        cent[k] = members.mean(0) if len(members) else global_mean
    cent = normalize(cent, norm="l2")
    a_dist = np.linalg.norm(XA - cent[pseudo_y], axis=1)
    cutoff = np.quantile(a_dist, q)
    b_dist = np.linalg.norm(XB[:, None, :] - cent[None, :, :], axis=-1).min(1)
    return b_dist <= cutoff


def sample_batch(pseudo_y: np.ndarray,
                 tier: np.ndarray,
                 mnn_graph: List[np.ndarray],
                 b_old_ok: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    a_idx: List[int] = []
    for k in range(K_OLD):
        # Prefer high-confidence anchors for CE/FixMatch; fall back to mid-confidence
        # samples so SupCon still gets coverage for every pseudo-cluster.
        pool = np.where((pseudo_y == k) & (tier >= ANCHOR_TIER_MIN))[0]
        if len(pool) == 0:
            pool = np.where((pseudo_y == k) & (tier >= SUPCON_TIER_MIN))[0]
        if len(pool):
            a_idx.extend(rng.choice(pool, min(N_A_PER_CLS, len(pool)), replace=False).tolist())

    # B samples for view-consistency and MNN positives. Do not class-label B as old.
    if b_old_ok is not None and MNN_OLD_ONLY:
        seed_pool = np.where(b_old_ok)[0]
        if len(seed_pool) < N_B_STEP:
            seed_pool = np.arange(len(mnn_graph))
    else:
        seed_pool = np.arange(len(mnn_graph))

    n_seed = min(N_B_STEP, len(seed_pool))
    b_seeds = rng.choice(seed_pool, size=n_seed, replace=False).astype(np.int64)

    partners: List[int] = []
    for bi in b_seeds[:N_MNN_STEP]:
        neigh = mnn_graph[int(bi)]
        if len(neigh):
            if b_old_ok is not None and MNN_OLD_ONLY:
                neigh = np.array([n for n in neigh.tolist() if b_old_ok[int(n)]], dtype=np.int64)
            if len(neigh):
                take = min(MNN_PARTNERS_PER_B, len(neigh))
                partners.extend(rng.choice(neigh, size=take, replace=False).tolist())
    b_idx = np.unique(np.concatenate([b_seeds, np.array(partners, dtype=np.int64)]))
    return np.array(a_idx, dtype=np.int64), b_idx.astype(np.int64)


def build_mnn_mask(b_idx: np.ndarray, mnn_graph: List[np.ndarray]) -> torch.Tensor:
    n = len(b_idx)
    mask = torch.zeros(n, n, dtype=torch.float32)
    pos = {int(b): i for i, b in enumerate(b_idx.tolist())}
    for i, b in enumerate(b_idx.tolist()):
        for nb in mnn_graph[int(b)].tolist():
            j = pos.get(int(nb))
            if j is not None:
                mask[i, j] = MNN_WEIGHT
                mask[j, i] = MNN_WEIGHT
    return mask

# ── Feature extraction ─────────────────────────────────────────────────────────
@torch.no_grad()
def extract_projected(model: FixMatchDINO,
                      paths: Sequence[str],
                      subset_global_indices: np.ndarray,
                      batch_size: int = EMB_BATCH) -> np.ndarray:
    model.eval()
    feats: List[np.ndarray] = []
    for s in range(0, len(subset_global_indices), batch_size):
        gi = subset_global_indices[s:s + batch_size]
        x = load_image_batch(paths, gi, eval_tfm).to(DEVICE, non_blocking=True)
        with amp_context():
            _, z, _ = model(x)
        feats.append(z.float().cpu().numpy())
    return np.vstack(feats).astype(np.float32)

# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    # ── Image paths ────────────────────────────────────────────────────────────
    print("Loading image paths …")
    with open(IMAGE_INDEX) as fh:
        all_image_paths = [line.strip() for line in fh]
    print(f"  {len(all_image_paths):,} paths loaded")

    # ── Data ───────────────────────────────────────────────────────────────────
    print("Loading pre-computed DINOv2 embeddings …")
    embeddings = np.load("plantnet_dinov2_raw.npy").astype(np.float32)
    labels = np.load("plantnet_labels.npy")  # evaluation/split construction only

    all_classes = np.unique(labels)
    counts = np.array([(labels == c).sum() for c in all_classes])
    eligible = all_classes[counts >= 2 * N_PER_CLS]
    chosen_60 = np.sort(rng.choice(eligible, K_NEW, replace=False))
    base_cls = chosen_60[:K_OLD]
    novel_cls = chosen_60[K_OLD:]

    XA, XB, yAl, yBl, idxA, idxB = [], [], [], [], [], []
    for c in base_cls:
        idx = rng.choice(np.where(labels == c)[0], 2 * N_PER_CLS, replace=False)
        XA.append(embeddings[idx[:N_PER_CLS]])
        yAl.extend([c] * N_PER_CLS)
        idxA.extend(idx[:N_PER_CLS].tolist())
        XB.append(embeddings[idx[N_PER_CLS:]])
        yBl.extend([c] * N_PER_CLS)
        idxB.extend(idx[N_PER_CLS:].tolist())
    for c in novel_cls:
        idx = rng.choice(np.where(labels == c)[0], N_PER_CLS, replace=False)
        XB.append(embeddings[idx])
        yBl.extend([c] * N_PER_CLS)
        idxB.extend(idx.tolist())

    X_A = np.vstack(XA).astype(np.float32)
    X_B = np.vstack(XB).astype(np.float32)
    y_A = np.array(yAl)
    y_B = np.array(yBl)
    img_idx_A = np.array(idxA, dtype=np.int64)
    img_idx_B = np.array(idxB, dtype=np.int64)

    id2idx = {c: i for i, c in enumerate(chosen_60)}
    y_A_eval = np.array([id2idx[c] for c in y_A])
    y_B_eval = np.array([id2idx[c] for c in y_B])
    is_novel_B = y_B_eval >= K_OLD
    N_A, N_B = len(X_A), len(X_B)
    print(f"  A:{N_A}  B:{N_B}  novel_B:{is_novel_B.sum()}")

    # ── Baseline from existing raw embeddings ─────────────────────────────────
    print("\nBaseline: combined UMAP-10 on raw pre-computed features …")
    t0 = time.time()
    r_base = umap.UMAP(n_components=10, n_neighbors=20, min_dist=0.05,
                       metric="cosine", random_state=SEED, verbose=False)
    X_AB_base = normalize(r_base.fit_transform(np.vstack([X_A, X_B])), norm="l2").astype(np.float32)
    X_A_umap = X_AB_base[:N_A]
    X_B_base = X_AB_base[N_A:]
    print(f"  Done in {time.time() - t0:.1f}s")
    res_baseline = gcd_acc(X_B_base, y_B_eval, is_novel_B, "Baseline UMAP-10 raw")

    mnn_graph = build_mnn_graph(X_B, MNN_K)

    # ── Backbone / model ──────────────────────────────────────────────────────
    print(f"\nLoading DINOv2 backbone: {DINO_MODEL} …")
    backbone = torch.hub.load("facebookresearch/dinov2", DINO_MODEL)
    freeze_all(backbone)
    n_lora = inject_lora_qkv(backbone, rank=LORA_RANK, alpha=LORA_ALPHA)
    print(f"  Injected LoRA into {n_lora} qkv layers")
    set_lora_trainable(backbone, True)

    model = FixMatchDINO(backbone).to(DEVICE)

    # Round 0 pseudo-labels use best available starting signal.
    Z_A_current = X_A_umap
    prev_centers: Optional[np.ndarray] = None
    prev_pseudo_y: Optional[np.ndarray] = None
    all_results = [("Baseline raw UMAP-10", *res_baseline)]

    print("\n" + "=" * 80)
    print("ITERATIVE PSEUDO-LABEL REFINEMENT + FIXMATCH REAL AUGMENTATIONS")
    print("=" * 80)

    for rnd in range(ROUNDS):
        print(f"\n{'─' * 80}")
        print(f"ROUND {rnd}  {'(fresh head/classifier)' if rnd == 0 else '(fine-tuning)'}")
        print(f"{'─' * 80}")

        # ── Step 1/2: pseudo-labels and confidence ───────────────────────────
        src = "baseline combined UMAP-10 A-portion" if rnd == 0 else f"projected A from round {rnd - 1}"
        print(f"  [1] K-Means(50) pseudo-labels on {src} …")
        pseudo_y, centers50, conf, tier = cluster_and_confidence(Z_A_current, prev_centers, prev_pseudo_y)
        prev_centers = centers50.copy()
        prev_pseudo_y = pseudo_y.copy()
        pl_acc = pseudo_label_acc(pseudo_y, y_A_eval)
        print(f"       Pseudo-label accuracy on A: {pl_acc:.1%}  (eval only, not used in training)")
        print(f"  [2] Confidence — C_high:{(tier == 2).sum()}  C_mid:{(tier == 1).sum()}  C_low:{(tier == 0).sum()}")

        # Conservative B filter for MNN-only positives; B is never given old-class CE/FixMatch labels.
        b_old_ok = compute_b_old_candidate_mask(X_B, centers50, X_A, pseudo_y) if MNN_OLD_ONLY else None
        if b_old_ok is not None:
            print(f"       B old-candidate filter for MNN: {b_old_ok.sum()}/{N_B} kept")

        label_t = torch.from_numpy(pseudo_y.astype(np.int64))
        conf_t = torch.from_numpy(conf.astype(np.float32))
        tier_t = torch.from_numpy(tier.astype(np.int64))

        epochs = EPOCHS_0 if rnd == 0 else EPOCHS_R
        lr_head = LR_HEAD_0 if rnd == 0 else LR_HEAD_R
        lr_bb = LR_BB_0 if rnd == 0 else LR_BB_R

        # Diagnostic freeze option applies only to LoRA adapters; base DINO is always frozen.
        set_lora_trainable(model.backbone, TRAIN_LORA and not (FREEZE_BACKBONE_R0 and rnd == 0))
        param_groups = [
            {"params": list(model.head.parameters()) + list(model.classifier.parameters()), "lr": lr_head, "weight_decay": 1e-4},
        ]
        bb_params = [p for p in lora_parameters(model.backbone) if p.requires_grad]
        if bb_params:
            param_groups.append({"params": bb_params, "lr": lr_bb, "weight_decay": 0.0})
        opt = torch.optim.AdamW(param_groups)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        lora_status = f"lr_lora={lr_bb:.0e}" if bb_params else "lora=off"
        print(f"  [3] Training {epochs} epochs  lr_head={lr_head:.0e}  {lora_status}  tau={FIXMATCH_TAU_START:.2f}→{FIXMATCH_TAU:.2f}")
        t0 = time.time()
        for ep in range(epochs):
            model.train()
            a_idx, b_idx = sample_batch(pseudo_y, tier, mnn_graph, b_old_ok)
            if len(a_idx) == 0:
                raise RuntimeError("No A samples selected. Lower confidence threshold/tier logic or increase N_A_PER_CLS.")

            # Real weak/strong augmentations.
            x_aw = load_image_batch(all_image_paths, img_idx_A[a_idx], weak_aug).to(DEVICE, non_blocking=True)
            x_as = load_image_batch(all_image_paths, img_idx_A[a_idx], strong_aug).to(DEVICE, non_blocking=True)

            y_a = label_t[a_idx].to(DEVICE)
            c_a = conf_t[a_idx].to(DEVICE)
            t_a = tier_t[a_idx].to(DEVICE)

            with amp_context():
                _, z_aw, logits_aw = model(x_aw)
                _, z_as, logits_as = model(x_as)

                # Anchor CE: current K-Means pseudo-cluster labels are the labeled set.
                # Train both views so the classifier calibrates before strict FixMatch is expected.
                anchor_mask = t_a >= ANCHOR_TIER_MIN
                loss_anchor_w = weighted_ce(logits_aw[anchor_mask], y_a[anchor_mask], c_a[anchor_mask])
                loss_anchor_s = weighted_ce(logits_as[anchor_mask], y_a[anchor_mask], c_a[anchor_mask])
                loss_anchor = 0.5 * (loss_anchor_w + loss_anchor_s)

                # FixMatch: weak-view pseudo-label supervises strong view.
                # In this cluster-refinement setting the safer default is cluster-anchored:
                #   target = current K-Means pseudo-label, confidence = p_weak(cluster target).
                # This avoids random early classifier argmax labels taking over.
                tau_ep = fixmatch_tau_for_epoch(ep)
                with torch.no_grad():
                    probs_w = F.softmax(logits_aw.detach(), dim=1)
                    pred_conf, pred_y = probs_w.max(dim=1)
                    cluster_conf = probs_w.gather(1, y_a[:, None]).squeeze(1)

                    if USE_CLUSTER_TARGET_FM:
                        fm_y = y_a
                        fm_conf = cluster_conf
                        fm_candidates = anchor_mask
                        if REQUIRE_CLUSTER_AGREE:
                            fm_candidates = fm_candidates & (pred_y == y_a)
                    else:
                        fm_y = pred_y
                        fm_conf = pred_conf
                        fm_candidates = anchor_mask
                        if REQUIRE_CLUSTER_AGREE:
                            fm_candidates = fm_candidates & (pred_y == y_a)

                    fm_mask = fm_candidates & (fm_conf >= tau_ep)
                    fm_mask = force_min_keep(fm_mask, fm_conf, fm_candidates, FIXMATCH_MIN_KEEP)

                loss_fm = weighted_ce(logits_as[fm_mask], fm_y[fm_mask], fm_conf[fm_mask]) if fm_mask.any() else zero_loss(DEVICE)

                loss_a_sup = a_pseudo_supcon_loss(z_aw, z_as, y_a, c_a, t_a)

                loss_b_aug = zero_loss(DEVICE)
                loss_b_mnn = zero_loss(DEVICE)
                if len(b_idx) > 0:
                    x_bw = load_image_batch(all_image_paths, img_idx_B[b_idx], weak_aug).to(DEVICE, non_blocking=True)
                    x_bs = load_image_batch(all_image_paths, img_idx_B[b_idx], strong_aug).to(DEVICE, non_blocking=True)
                    _, z_bw, _ = model(x_bw)
                    _, z_bs, _ = model(x_bs)
                    loss_b_aug = two_view_instance_contrast(z_bw, z_bs)

                    base_mnn = build_mnn_mask(b_idx, mnn_graph)
                    if base_mnn.sum() > 0:
                        b_pos = expand_b_pos_mask(base_mnn, include_self_views=True).to(DEVICE)
                        loss_b_mnn = contrastive_from_mask(torch.cat([z_bw, z_bs], dim=0), b_pos, tau=SUPCON_TAU)

                loss = (LAMBDA_ANCH * loss_anchor +
                        LAMBDA_FM * loss_fm +
                        LAMBDA_A_SUP * loss_a_sup +
                        LAMBDA_B_AUG * loss_b_aug +
                        LAMBDA_B_MNN * loss_b_mnn)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()

            if (ep + 1) % max(1, epochs // 2) == 0 or ep == 0:
                fm_kept = int(fm_mask.sum().item())
                print(
                    f"       ep {ep+1:>3}/{epochs}  "
                    f"loss={loss.item():.4f}  "
                    f"anch={loss_anchor.item():.3f}  fm={loss_fm.item():.3f}({fm_kept}/{len(a_idx)},tau={tau_ep:.2f})  "
                    f"Acon={loss_a_sup.item():.3f}  Baug={loss_b_aug.item():.3f}  Bmnn={loss_b_mnn.item():.3f}"
                )
        print(f"       Training time: {time.time() - t0:.1f}s")

        # ── Step 4: extract projected eval features from real images ──────────
        print("  [4] Extracting projected eval features from real images …")
        Z_A = extract_projected(model, all_image_paths, img_idx_A, batch_size=EMB_BATCH)
        Z_B = extract_projected(model, all_image_paths, img_idx_B, batch_size=EMB_BATCH)

        # Representation diagnostic on A using current pseudo labels.
        Za_n = normalize(Z_A, norm="l2")
        intra, inter = [], []
        for k in range(0, K_OLD, 5):
            mem = Za_n[pseudo_y == k]
            if len(mem) < 2:
                continue
            d = np.linalg.norm(mem[:, None, :] - mem[None, :, :], axis=-1)
            intra.append(d[np.triu_indices(len(mem), k=1)].mean())
            inter.append(np.linalg.norm(Za_n[pseudo_y != k] - mem.mean(0), axis=1).mean())
        if intra and inter:
            ratio = np.mean(inter) / (np.mean(intra) + 1e-9)
            print(f"       Representation: intra={np.mean(intra):.4f}  inter={np.mean(inter):.4f}  ratio={ratio:.2f}x")

        # ── Step 5: GCD evaluation ────────────────────────────────────────────
        print("  [5] GCD evaluation on projected B features …")
        res = gcd_acc(Z_B, y_B_eval, is_novel_B, f"Round {rnd} — FixMatch real-aug")
        all_results.append((f"Round {rnd}", *res))

        # Update features for next round.
        Z_A_current = Z_A

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY — GCD metrics across rounds")
    print("=" * 80)
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
        print(f"  {tag:<45} {all_a:>7.1%} {old_a:>7.1%} {nov_a:>9.1%}{mark}")

    print("\n  Δ vs baseline (best round by All):")
    best = max(all_results[1:], key=lambda x: x[1])
    for metric, idx in [("All", 1), ("Old", 2), ("Novel", 3)]:
        delta = best[idx] - all_results[0][idx]
        print(f"    {metric}: {all_results[0][idx]:.1%} → {best[idx]:.1%}  ({delta:+.1%})")


if __name__ == "__main__":
    main()
