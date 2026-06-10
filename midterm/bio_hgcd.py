"""
BioHGCD: Biological Hierarchical Generalized Category Discovery
================================================================
Run in the openldn conda environment:

    conda activate openldn
    pip install matplotlib umap-learn   # one-time, only missing pkgs
    python bio_hgcd.py

Sections:
  1. Config
  2. DINOv2 feature extraction from Micro_Organism
  3. GCD split  (known labeled | unlabeled = known_unlabeled + novel)
  4. Contrastive fine-tuning — SupCon projection head on known labels
  5. Ward hierarchical clustering + dendrogram
  6. Novel cluster count estimation — gap statistic
  7. Multi-level evaluation — ACC, NMI, ARI, Dendrogram Purity
  8. Anchor-guided novel class detection
  9. UMAP visualization
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")  # safe in terminal / notebook
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import pdist

try:
    import umap as umap_lib
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("umap-learn not installed — skipping UMAP plots. pip install umap-learn")

# ============================================================
# 1. CONFIG
# ============================================================

DATA_DIR   = Path("./Micro_Organism")
FEAT_CACHE = Path("./cache_dinov2_raw.npy")
FTFT_CACHE = Path("./cache_dinov2_finetuned.npy")
LBL_CACHE  = Path("./cache_labels.npy")

# ImageFolder sorts class dirs alphabetically:
ALL_CLASSES = [
    "Amoeba", "Euglena", "Hydra", "Paramecium",
    "Rod_bacteria", "Spherical_bacteria", "Spiral_bacteria", "Yeast",
]
# GCD split — bacteria/yeast as known, protists as novel.
# Scientific story: a lab has protocols to identify bacteria and yeast
# but encounters unknown protist species in an environmental water sample.
KNOWN_CLASSES = ["Rod_bacteria", "Spherical_bacteria", "Spiral_bacteria", "Yeast"]
NOVEL_CLASSES = ["Amoeba", "Euglena", "Hydra", "Paramecium"]

KNOWN_IDS = [ALL_CLASSES.index(c) for c in KNOWN_CLASSES]
NOVEL_IDS = [ALL_CLASSES.index(c) for c in NOVEL_CLASSES]
N_CLASSES  = len(ALL_CLASSES)

DEVICE = (
    "cuda"  if torch.cuda.is_available()              else
    "mps"   if torch.backends.mps.is_available()      else
    "cpu"
)
print(f"Device: {DEVICE}")

SEED             = 42
LABELED_FRACTION = 0.5     # fraction of known-class samples added to D_L
BATCH_SIZE       = 32
SUPCON_EPOCHS    = 60
SUPCON_LR        = 1e-3
PROJ_DIM         = 128     # projection head output dimension
PCA_COMPONENTS   = 50

torch.manual_seed(SEED)
np.random.seed(SEED)

# ============================================================
# 2. DINOV2 FEATURE EXTRACTION
# ============================================================

DINO_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def extract_dinov2_features():
    if FEAT_CACHE.exists() and LBL_CACHE.exists():
        print("Loading cached DINOv2 features …")
        return np.load(FEAT_CACHE), np.load(LBL_CACHE)

    print("Loading DINOv2 ViT-B/14 …")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model.eval().to(DEVICE)

    dataset = datasets.ImageFolder(str(DATA_DIR), transform=DINO_TRANSFORM)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_feats, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            feats = model(imgs.to(DEVICE))   # [B, 768]  —  [CLS] token
            all_feats.append(feats.cpu().numpy())
            all_labels.append(lbls.numpy())

    features = np.concatenate(all_feats,  axis=0)
    labels   = np.concatenate(all_labels, axis=0)

    np.save(FEAT_CACHE, features)
    np.save(LBL_CACHE,  labels)
    print(f"Extracted features: {features.shape}  labels: {labels.shape}")
    return features, labels


features_raw, labels = extract_dinov2_features()
class_names = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
print("Class order :", class_names)
print("Samples/class:", {c: int((labels == i).sum()) for i, c in enumerate(class_names)})

# ============================================================
# 3. GCD SPLIT
# ============================================================
# D_L  — labeled,   known classes only, LABELED_FRACTION of each
# D_U  — unlabeled, rest of known + all novel  (what we cluster)

def build_gcd_split(labels, known_ids, labeled_fraction=0.5, seed=42):
    rng = np.random.default_rng(seed)
    labeled_idx, unlabeled_idx = [], []

    for kid in known_ids:
        idx = np.where(labels == kid)[0]
        rng.shuffle(idx)
        n_l = max(1, int(len(idx) * labeled_fraction))
        labeled_idx.extend(idx[:n_l].tolist())
        unlabeled_idx.extend(idx[n_l:].tolist())

    for nid in range(N_CLASSES):
        if nid not in known_ids:
            unlabeled_idx.extend(np.where(labels == nid)[0].tolist())

    return np.array(labeled_idx), np.array(unlabeled_idx)


labeled_idx, unlabeled_idx = build_gcd_split(labels, KNOWN_IDS, LABELED_FRACTION)
print(f"\nGCD split — labeled: {len(labeled_idx)}  unlabeled: {len(unlabeled_idx)}")

# ============================================================
# 4. CONTRASTIVE FINE-TUNING  (SupCon on known labels)
# ============================================================

class ProjectionHead(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def supcon_loss(z, y, temperature=0.1):
    """Supervised Contrastive Loss — Khosla et al. (2020)."""
    n = z.size(0)
    sim = torch.mm(z, z.T) / temperature          # [N, N]
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    pos = (y.unsqueeze(0) == y.unsqueeze(1)) & ~eye

    # Exclude self from denominator
    sim_no_self = sim.masked_fill(eye, -1e4)
    log_prob    = sim_no_self - torch.logsumexp(sim_no_self, dim=1, keepdim=True)

    # Zero out non-positive slots BEFORE summing to avoid 0 * -inf = NaN
    log_prob_pos = log_prob.masked_fill(~pos, 0.0)
    n_pos        = pos.float().sum(dim=1).clamp(min=1)
    loss         = -log_prob_pos.sum(dim=1) / n_pos
    return loss.mean()


def train_projection_head(features_all, labels_all, labeled_idx):
    if FTFT_CACHE.exists():
        print("Loading cached fine-tuned features …")
        return np.load(FTFT_CACHE)

    head      = ProjectionHead(in_dim=features_all.shape[1], out_dim=PROJ_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(head.parameters(), lr=SUPCON_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SUPCON_EPOCHS)

    X_l = torch.tensor(features_all[labeled_idx], dtype=torch.float32).to(DEVICE)
    y_l = torch.tensor(labels_all[labeled_idx],   dtype=torch.long).to(DEVICE)

    print("Training SupCon projection head …")
    head.train()
    for epoch in range(1, SUPCON_EPOCHS + 1):
        optimizer.zero_grad()
        loss = supcon_loss(head(X_l), y_l)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch % 10 == 0:
            print(f"  epoch {epoch:3d}/{SUPCON_EPOCHS}  loss={loss.item():.4f}")

    # Project all samples through trained head
    head.eval()
    with torch.no_grad():
        finetuned = head(
            torch.tensor(features_all, dtype=torch.float32).to(DEVICE)
        ).cpu().numpy()

    np.save(FTFT_CACHE, finetuned)
    print(f"Fine-tuned features: {finetuned.shape}")
    return finetuned


features_ft = train_projection_head(features_raw, labels, labeled_idx)

# ============================================================
# 5. HIERARCHICAL CLUSTERING + DENDROGRAM
# ============================================================
# Cluster D_U (the unlabeled pool).  Labeled samples serve as anchors
# at evaluation / novel-cluster detection time.

def reduce_pca(features, n_components=50, seed=42):
    n_comp = min(n_components, features.shape[0] - 1, features.shape[1] - 1)
    pca    = PCA(n_components=n_comp, random_state=seed)
    reduced = pca.fit_transform(features)
    var_exp = pca.explained_variance_ratio_.sum()
    print(f"PCA {features.shape[1]}d → {reduced.shape[1]}d  (var explained: {var_exp:.3f})")
    return reduced


def build_ward_linkage(features):
    normed = normalize(features, norm="l2")
    Z      = linkage(normed, method="ward", metric="euclidean")
    c, _   = cophenet(Z, pdist(normed))
    print(f"Cophenetic correlation (Ward): {c:.4f}")
    return Z, normed


print("\n--- Building hierarchical clustering on unlabeled set (D_U) ---")
feats_u    = features_ft[unlabeled_idx]
labels_u   = labels[unlabeled_idx]
feats_u_r  = reduce_pca(feats_u, PCA_COMPONENTS)
Z_u, normed_u = build_ward_linkage(feats_u_r)


def save_dendrogram(Z, title, fname):
    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(
        Z, ax=ax,
        truncate_mode="lastp", p=40,
        leaf_rotation=90, leaf_font_size=7,
        show_contracted=True,
    )
    ax.set_title(title)
    ax.set_ylabel("Ward distance")
    ax.set_xlabel("Cluster  (n samples)")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved: {fname}")


save_dendrogram(Z_u, "BioHGCD — Ward Dendrogram (D_U unlabeled)", "bio_hgcd_dendrogram.png")

# ============================================================
# 6. NOVEL CLUSTER COUNT ESTIMATION — gap statistic
# ============================================================

def gap_statistic(Z, features_normed, k_range=range(2, 16), n_refs=10, seed=42):
    rng = np.random.default_rng(seed)

    def wcss(pred, X):
        s = 0.0
        for k in np.unique(pred):
            c = X[pred == k]
            if len(c) > 1:
                s += ((c - c.mean(0)) ** 2).sum()
        return s

    gaps, sks = [], []
    for k in k_range:
        pred   = fcluster(Z, t=k, criterion="maxclust") - 1
        log_wk = np.log(wcss(pred, features_normed) + 1e-9)

        ref_logwk = []
        for _ in range(n_refs):
            ref     = rng.uniform(features_normed.min(0), features_normed.max(0),
                                  size=features_normed.shape)
            ref_Z   = linkage(ref, method="ward")
            ref_pred = fcluster(ref_Z, t=k, criterion="maxclust") - 1
            ref_logwk.append(np.log(wcss(ref_pred, ref) + 1e-9))

        gaps.append(np.mean(ref_logwk) - log_wk)
        sks.append(np.std(ref_logwk) * np.sqrt(1 + 1 / n_refs))

    gaps, sks  = np.array(gaps), np.array(sks)
    klist      = list(k_range)
    # Tibshirani criterion
    best_k = klist[0]
    for i in range(len(klist) - 1):
        if gaps[i] >= gaps[i + 1] - sks[i + 1]:
            best_k = klist[i]
            break

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(klist, gaps, yerr=sks, fmt="o-", capsize=4, label="Gap(k)")
    ax.axvline(best_k,    color="red",    linestyle="--", label=f"Est. K={best_k}")
    ax.axvline(N_CLASSES, color="green",  linestyle=":",  label=f"True K={N_CLASSES}")
    ax.set_xlabel("K")
    ax.set_ylabel("Gap statistic")
    ax.set_title("Novel Class Count Estimation (Gap Statistic)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("bio_hgcd_gap.png", dpi=150)
    plt.close()
    print(f"Estimated K = {best_k}  (true K = {N_CLASSES})")
    print("Saved: bio_hgcd_gap.png")
    return best_k


print("\n--- Estimating K via gap statistic ---")
estimated_k = gap_statistic(Z_u, normed_u, k_range=range(2, N_CLASSES + 6))

# ============================================================
# 7. EVALUATION METRICS
# ============================================================

def hungarian_acc(pred, true):
    n = max(pred.max(), true.max()) + 1
    C = np.zeros((n, n), dtype=np.int64)
    for p, t in zip(pred, true):
        C[p, t] += 1
    r, c = linear_sum_assignment(-C)
    return C[r, c].sum() / len(pred)


def dendrogram_purity(Z, labels, k):
    """Fraction of samples in majority-pure clusters (>50% one class)."""
    pred  = fcluster(Z, t=k, criterion="maxclust") - 1
    pure  = sum(
        mask.sum()
        for c in np.unique(pred)
        for mask in [(pred == c)]
        if mask.sum() > 0 and
           np.bincount(labels[mask], minlength=N_CLASSES).max() / mask.sum() > 0.5
    )
    return pure / len(labels)


def evaluate(Z, labels, k, tag=""):
    pred = fcluster(Z, t=k, criterion="maxclust") - 1
    acc  = hungarian_acc(pred, labels)
    nmi  = normalized_mutual_info_score(labels, pred)
    ari  = adjusted_rand_score(labels, pred)
    dp   = dendrogram_purity(Z, labels, k)
    print(f"  {tag:30s}  K={k:3d}  ACC={acc:.4f}  NMI={nmi:.4f}  ARI={ari:.4f}  DP={dp:.4f}")
    return dict(k=k, acc=acc, nmi=nmi, ari=ari, dp=dp)


print("\n=== Multi-level evaluation on D_U ===")
results = [evaluate(Z_u, labels_u, k, tag=f"Ward K={k}") for k in range(2, N_CLASSES * 3 + 1)]

print(f"\n--- At true K={N_CLASSES} ---")
evaluate(Z_u, labels_u, N_CLASSES, tag="Ward (true K)")

# ---- Plot metrics vs K ----
ks   = [r["k"] for r in results]
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, metric in zip(axes, ["acc", "nmi", "ari", "dp"]):
    vals = [r[metric] for r in results]
    ax.plot(ks, vals, "o-", markersize=4)
    ax.axvline(N_CLASSES,   color="green", linestyle=":", label=f"True K={N_CLASSES}")
    ax.axvline(estimated_k, color="red",   linestyle="--", label=f"Est. K={estimated_k}")
    ax.set_xlabel("K")
    ax.set_ylabel(metric.upper())
    ax.set_title(metric.upper())
    ax.legend(fontsize=7)
plt.suptitle("BioHGCD — DINOv2 + Ward  (unlabeled set D_U)")
plt.tight_layout()
plt.savefig("bio_hgcd_metrics.png", dpi=150)
plt.close()
print("Saved: bio_hgcd_metrics.png")

# ============================================================
# 8. ANCHOR-GUIDED NOVEL CLUSTER DETECTION
# ============================================================
# A cluster is "known" if it contains at least one labeled known sample.
# All other clusters at cut K are novel candidates.

def anchor_guided_detection(Z, feats_all, labels_all, labeled_idx, known_ids, cut_k):
    # Run linkage on ALL samples so we can propagate label anchors
    print(f"\n--- Anchor-guided detection (K={cut_k}) on full dataset ---")
    feats_all_r = reduce_pca(feats_all, PCA_COMPONENTS)
    Z_all, _    = build_ward_linkage(feats_all_r)
    pred_all    = fcluster(Z_all, t=cut_k, criterion="maxclust") - 1

    labeled_set       = set(labeled_idx.tolist())
    known_cluster_ids = {
        pred_all[i]
        for i in range(len(pred_all))
        if i in labeled_set and labels_all[i] in known_ids
    }
    novel_cluster_ids = set(np.unique(pred_all)) - known_cluster_ids

    print(f"  Known clusters  ({len(known_cluster_ids)}): {sorted(known_cluster_ids)}")
    print(f"  Novel candidates({len(novel_cluster_ids)}): {sorted(novel_cluster_ids)}")
    print()
    print(f"  {'Cluster':>9}  {'Tag':>6}  {'N':>4}  Composition")
    for c in sorted(np.unique(pred_all)):
        mask   = pred_all == c
        comp   = {class_names[k]: int(v)
                  for k, v in enumerate(np.bincount(labels_all[mask], minlength=N_CLASSES))
                  if v > 0}
        tag    = "KNOWN" if c in known_cluster_ids else "NOVEL*"
        print(f"  {c:>9}  {tag:>6}  {mask.sum():>4}  {comp}")

    return Z_all, pred_all, known_cluster_ids, novel_cluster_ids


Z_all, pred_all, known_cids, novel_cids = anchor_guided_detection(
    Z_u, features_ft, labels, labeled_idx, KNOWN_IDS, cut_k=N_CLASSES
)

# ============================================================
# 9. UMAP VISUALIZATION
# ============================================================

def plot_umap(features, labels_gt, pred, known_cids, novel_cids):
    if not HAS_UMAP:
        print("Skipping UMAP (umap-learn not installed).")
        return

    print("\nRunning UMAP …")
    reducer = umap_lib.UMAP(n_components=2, random_state=SEED,
                            n_neighbors=15, min_dist=0.1)
    emb     = reducer.fit_transform(normalize(features, norm="l2"))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    cmap10    = matplotlib.colormaps.get_cmap("tab10").resampled(N_CLASSES)

    # Left — ground truth
    ax = axes[0]
    for i, cname in enumerate(class_names):
        m = labels_gt == i
        ax.scatter(emb[m, 0], emb[m, 1], c=[cmap10(i)],
                   label=cname, s=18, alpha=0.75)
    ax.set_title("Ground Truth")
    ax.legend(fontsize=7, markerscale=1.5)
    ax.set_xticks([]); ax.set_yticks([])

    # Right — predicted (red = novel candidate)
    ax = axes[1]
    cmap20 = matplotlib.colormaps.get_cmap("tab20").resampled(len(np.unique(pred)))
    for j, c in enumerate(sorted(np.unique(pred))):
        m     = pred == c
        color = "red" if c in novel_cids else cmap20(j)
        tag   = "NOVEL*" if c in novel_cids else "known"
        ax.scatter(emb[m, 0], emb[m, 1], c=[color],
                   label=f"C{c} [{tag}]", s=18, alpha=0.75)
    ax.set_title("Predicted Clusters  (red = novel candidate)")
    ax.legend(fontsize=6, markerscale=1.5)
    ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle("BioHGCD — DINOv2 + Ward + Anchor-guided detection", fontsize=13)
    plt.tight_layout()
    plt.savefig("bio_hgcd_umap.png", dpi=150)
    plt.close()
    print("Saved: bio_hgcd_umap.png")


# Use PCA-reduced features of ALL samples for UMAP
feats_all_r_for_umap = reduce_pca(features_ft, PCA_COMPONENTS)
plot_umap(feats_all_r_for_umap, labels, pred_all, known_cids, novel_cids)

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("BioHGCD pipeline complete")
print("=" * 60)
print("Output files:")
for f in [
    "bio_hgcd_dendrogram.png   — Ward linkage dendrogram (D_U)",
    "bio_hgcd_gap.png          — gap statistic for K estimation",
    "bio_hgcd_metrics.png      — ACC / NMI / ARI / DP vs K",
    "bio_hgcd_umap.png         — UMAP: ground truth vs predicted",
]:
    print(f"  {f}")

best = max(results, key=lambda r: r["nmi"])
print(f"\nBest NMI at K={best['k']}: ACC={best['acc']:.4f}  NMI={best['nmi']:.4f}"
      f"  ARI={best['ari']:.4f}  DP={best['dp']:.4f}")
