"""
BioHGCD — DIBaS (Digital Images of Bacteria Species)
=====================================================
33 bacterial species, 20 Gram-stained images each (660 total).
Jagiellonian University, Kraków, Poland.

GCD split rationale:
  Known   — Gram-positive species  (a lab with existing Gram+ protocols)
  Novel   — Gram-negative species  (unknown organisms found in a new sample)

Run:
    conda activate openldn
    python bio_hgcd_dibas.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
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
    print("umap-learn not found — skipping UMAP. pip install umap-learn")

# ============================================================
# 1. CONFIG
# ============================================================

DATA_DIR   = Path("./DIBaS/images")
FEAT_CACHE = Path("./dibas_dinov2_raw.npy")
FTFT_CACHE = Path("./dibas_dinov2_finetuned.npy")
LBL_CACHE  = Path("./dibas_labels.npy")

DEVICE = (
    "cuda"  if torch.cuda.is_available()         else
    "mps"   if torch.backends.mps.is_available() else
    "cpu"
)
print(f"Device: {DEVICE}")

SEED             = 42
BATCH_SIZE       = 16
LABELED_FRACTION = 0.5
SUPCON_EPOCHS    = 80
SUPCON_LR        = 5e-4
PROJ_DIM         = 256
PCA_COMPONENTS   = 64

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Gram stain taxonomy ──────────────────────────────────────
# Source: standard microbiology references.
# Gram-POSITIVE  →  known classes (labeled)
# Gram-NEGATIVE  →  novel classes (to discover)
# Candida albicans is a fungus — treated as its own category (novel)

GRAM_POSITIVE = {
    "Actinomyces.israeli",
    "Bifidobacterium.spp",
    "Candida.albicans",          # fungus — visually distinct, treated as novel-like
    "Clostridium.perfringens",
    "Enterococcus.faecalis",
    "Enterococcus.faecium",
    "Lactobacillus.casei",
    "Lactobacillus.crispatus",
    "Lactobacillus.delbrueckii",
    "Lactobacillus.gasseri",
    "Lactobacillus.jehnsenii",
    "Lactobacillus.johnsonii",
    "Lactobacillus.paracasei",
    "Lactobacillus.plantarum",
    "Lactobacillus.reuteri",
    "Lactobacillus.rhamnosus",
    "Lactobacillus.salivarius",
    "Listeria.monocytogenes",
    "Micrococcus.spp",
    "Propionibacterium.acnes",
    "Staphylococcus.aureus",
    "Staphylococcus.epidermidis",
    "Staphylococcus.saprophiticus",
    "Streptococcus.agalactiae",
}

GRAM_NEGATIVE = {
    "Acinetobacter.baumanii",
    "Bacteroides.fragilis",
    "Escherichia.coli",
    "Fusobacterium",
    "Neisseria.gonorrhoeae",
    "Porfyromonas.gingivalis",
    "Proteus",
    "Pseudomonas.aeruginosa",
    "Veionella",
}

# Genus groupings — used for hierarchical evaluation
# (did the dendrogram recover genus-level clusters?)
GENUS_MAP = {
    "Acinetobacter.baumanii":       "Acinetobacter",
    "Actinomyces.israeli":          "Actinomyces",
    "Bacteroides.fragilis":         "Bacteroides",
    "Bifidobacterium.spp":          "Bifidobacterium",
    "Candida.albicans":             "Candida",
    "Clostridium.perfringens":      "Clostridium",
    "Enterococcus.faecium":         "Enterococcus",
    "Enterococcus.faecalis":        "Enterococcus",
    "Escherichia.coli":             "Escherichia",
    "Fusobacterium":                "Fusobacterium",
    "Lactobacillus.casei":          "Lactobacillus",
    "Lactobacillus.crispatus":      "Lactobacillus",
    "Lactobacillus.delbrueckii":    "Lactobacillus",
    "Lactobacillus.gasseri":        "Lactobacillus",
    "Lactobacillus.jehnsenii":      "Lactobacillus",
    "Lactobacillus.johnsonii":      "Lactobacillus",
    "Lactobacillus.paracasei":      "Lactobacillus",
    "Lactobacillus.plantarum":      "Lactobacillus",
    "Lactobacillus.reuteri":        "Lactobacillus",
    "Lactobacillus.rhamnosus":      "Lactobacillus",
    "Lactobacillus.salivarius":     "Lactobacillus",
    "Listeria.monocytogenes":       "Listeria",
    "Micrococcus.spp":              "Micrococcus",
    "Neisseria.gonorrhoeae":        "Neisseria",
    "Porfyromonas.gingivalis":      "Porphyromonas",
    "Propionibacterium.acnes":      "Propionibacterium",
    "Proteus":                      "Proteus",
    "Pseudomonas.aeruginosa":       "Pseudomonas",
    "Staphylococcus.aureus":        "Staphylococcus",
    "Staphylococcus.epidermidis":   "Staphylococcus",
    "Staphylococcus.saprophiticus": "Staphylococcus",
    "Streptococcus.agalactiae":     "Streptococcus",
    "Veionella":                    "Veionella",
}

# ============================================================
# 2. LOAD DATA & ASSIGN SPLIT
# ============================================================

def get_class_names():
    return sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])

class_names = get_class_names()
N_CLASSES   = len(class_names)
print(f"Found {N_CLASSES} classes: {class_names}")

# Build known/novel index lists
KNOWN_IDS = [i for i, c in enumerate(class_names) if c in GRAM_POSITIVE]
NOVEL_IDS = [i for i, c in enumerate(class_names) if c in GRAM_NEGATIVE]
print(f"Known (Gram+): {len(KNOWN_IDS)} species")
print(f"Novel (Gram-): {len(NOVEL_IDS)} species")

# Genus labels for each sample (for hierarchical eval)
genus_names  = sorted(set(GENUS_MAP.values()))
genus_to_idx = {g: i for i, g in enumerate(genus_names)}

# ============================================================
# 3. DINOV2 FEATURE EXTRACTION
# ============================================================

DINO_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def extract_features():
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
            feats = model(imgs.to(DEVICE))
            all_feats.append(feats.cpu().numpy())
            all_labels.append(lbls.numpy())

    features = np.concatenate(all_feats,  axis=0)
    labels   = np.concatenate(all_labels, axis=0)
    np.save(FEAT_CACHE, features)
    np.save(LBL_CACHE,  labels)
    print(f"Features: {features.shape}  Labels: {labels.shape}")
    return features, labels


features_raw, labels = extract_features()

# Genus label per sample
genus_labels = np.array([
    genus_to_idx[GENUS_MAP[class_names[l]]] for l in labels
])

print("\nSamples per class:")
for i, c in enumerate(class_names):
    gram = "Gram+" if c in GRAM_POSITIVE else "Gram-"
    print(f"  {c:40s} [{gram}]  n={int((labels == i).sum())}")

# ============================================================
# 4. GCD SPLIT
# ============================================================

def build_gcd_split(labels, known_ids, frac=0.5, seed=42):
    rng = np.random.default_rng(seed)
    labeled_idx, unlabeled_idx = [], []
    for kid in known_ids:
        idx = np.where(labels == kid)[0]
        rng.shuffle(idx)
        n_l = max(1, int(len(idx) * frac))
        labeled_idx.extend(idx[:n_l].tolist())
        unlabeled_idx.extend(idx[n_l:].tolist())
    for nid in range(N_CLASSES):
        if nid not in known_ids:
            unlabeled_idx.extend(np.where(labels == nid)[0].tolist())
    return np.array(labeled_idx), np.array(unlabeled_idx)


labeled_idx, unlabeled_idx = build_gcd_split(labels, KNOWN_IDS, LABELED_FRACTION)
print(f"\nGCD split — labeled: {len(labeled_idx)}  unlabeled: {len(unlabeled_idx)}")

# ============================================================
# 5. SUPCON FINE-TUNING
# ============================================================

class ProjectionHead(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=512, out_dim=256):
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
    n   = z.size(0)
    sim = torch.mm(z, z.T) / temperature
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    pos = (y.unsqueeze(0) == y.unsqueeze(1)) & ~eye
    sim_no_self  = sim.masked_fill(eye, -1e4)
    log_prob     = sim_no_self - torch.logsumexp(sim_no_self, dim=1, keepdim=True)
    log_prob_pos = log_prob.masked_fill(~pos, 0.0)
    n_pos        = pos.float().sum(dim=1).clamp(min=1)
    return (-log_prob_pos.sum(dim=1) / n_pos).mean()


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
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if epoch % 10 == 0:
            print(f"  epoch {epoch:3d}/{SUPCON_EPOCHS}  loss={loss.item():.4f}")

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
# 6. HIERARCHICAL CLUSTERING
# ============================================================

def reduce_pca(features, n_components=64, seed=42):
    n_comp  = min(n_components, features.shape[0] - 1, features.shape[1] - 1)
    from sklearn.decomposition import PCA
    pca     = PCA(n_components=n_comp, random_state=seed)
    reduced = pca.fit_transform(features)
    var_exp = pca.explained_variance_ratio_.sum()
    print(f"PCA {features.shape[1]}d → {reduced.shape[1]}d  (var explained: {var_exp:.3f})")
    return reduced


def build_ward(features):
    normed = normalize(features, norm="l2")
    Z      = linkage(normed, method="ward", metric="euclidean")
    c, _   = cophenet(Z, pdist(normed))
    print(f"Cophenetic correlation (Ward): {c:.4f}")
    return Z, normed


print("\n--- Hierarchical clustering on D_U ---")
feats_u   = features_ft[unlabeled_idx]
labels_u  = labels[unlabeled_idx]
genus_u   = genus_labels[unlabeled_idx]
feats_u_r = reduce_pca(feats_u, PCA_COMPONENTS)
Z_u, norm_u = build_ward(feats_u_r)


def save_dendrogram(Z, n_species, title, fname):
    fig, ax = plt.subplots(figsize=(18, 7))
    dendrogram(
        Z, ax=ax,
        truncate_mode="lastp", p=min(60, n_species * 2),
        leaf_rotation=90, leaf_font_size=7,
        show_contracted=True,
    )
    ax.set_title(title, fontsize=13)
    ax.set_ylabel("Ward distance")
    ax.set_xlabel("Cluster  (n samples)")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved: {fname}")


save_dendrogram(Z_u, N_CLASSES, "BioHGCD-DIBaS — Ward Dendrogram (D_U)", "dibas_dendrogram.png")

# ============================================================
# 7. GENUS-LEVEL DENDROGRAM — key hierarchical test
# ============================================================
# Build a separate dendrogram using genus-level mean features.
# If DINOv2 recovers genus structure, the 11 Lactobacillus species
# should all cluster together before merging with other genera.

def genus_level_dendrogram(features_all, labels_all):
    unique_genera = sorted(set(GENUS_MAP.values()))
    genus_feats   = []
    for g in unique_genera:
        species_in_genus = [i for i, c in enumerate(class_names) if GENUS_MAP[c] == g]
        idx = np.isin(labels_all, species_in_genus)
        if idx.sum() > 0:
            genus_feats.append(normalize(features_all[idx], norm="l2").mean(axis=0))
    genus_feats = np.array(genus_feats)
    genus_feats = normalize(genus_feats, norm="l2")

    Z_g = linkage(genus_feats, method="ward")

    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(Z_g, ax=ax, labels=unique_genera, leaf_rotation=45, leaf_font_size=9)
    ax.set_title("BioHGCD-DIBaS — Genus-level Dendrogram (mean DINOv2 features)", fontsize=12)
    ax.set_ylabel("Ward distance")
    plt.tight_layout()
    plt.savefig("dibas_genus_dendrogram.png", dpi=150)
    plt.close()
    print("Saved: dibas_genus_dendrogram.png")


print("\n--- Genus-level dendrogram ---")
feats_all_r = reduce_pca(features_ft, PCA_COMPONENTS)
genus_level_dendrogram(feats_all_r, labels)

# ============================================================
# 8. GAP STATISTIC
# ============================================================

def gap_statistic(Z, features_normed, k_range, n_refs=10, seed=42):
    rng = np.random.default_rng(seed)

    def wcss(pred, X):
        return sum(
            ((X[pred == k] - X[pred == k].mean(0)) ** 2).sum()
            for k in np.unique(pred) if (pred == k).sum() > 1
        )

    gaps, sks = [], []
    for k in k_range:
        pred   = fcluster(Z, t=k, criterion="maxclust") - 1
        log_wk = np.log(wcss(pred, features_normed) + 1e-9)
        ref_logwk = []
        for _ in range(n_refs):
            ref      = rng.uniform(features_normed.min(0), features_normed.max(0),
                                   size=features_normed.shape)
            ref_pred = fcluster(linkage(ref, method="ward"), t=k, criterion="maxclust") - 1
            ref_logwk.append(np.log(wcss(ref_pred, ref) + 1e-9))
        gaps.append(np.mean(ref_logwk) - log_wk)
        sks.append(np.std(ref_logwk) * np.sqrt(1 + 1 / n_refs))

    gaps, sks = np.array(gaps), np.array(sks)
    klist     = list(k_range)
    best_k    = klist[0]
    for i in range(len(klist) - 1):
        if gaps[i] >= gaps[i + 1] - sks[i + 1]:
            best_k = klist[i]
            break

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.errorbar(klist, gaps, yerr=sks, fmt="o-", capsize=4, label="Gap(k)")
    ax.axvline(best_k,    color="red",   linestyle="--", label=f"Est. K={best_k}")
    ax.axvline(N_CLASSES, color="green", linestyle=":",  label=f"True K={N_CLASSES}")
    ax.set_xlabel("K")
    ax.set_ylabel("Gap statistic")
    ax.set_title("DIBaS — Novel Species Count Estimation")
    ax.legend()
    plt.tight_layout()
    plt.savefig("dibas_gap.png", dpi=150)
    plt.close()
    print(f"Estimated K = {best_k}  (true K = {N_CLASSES})")
    print("Saved: dibas_gap.png")
    return best_k


print("\n--- Gap statistic ---")
estimated_k = gap_statistic(Z_u, norm_u, k_range=range(2, N_CLASSES + 12))

# ============================================================
# 9. EVALUATION
# ============================================================

def hungarian_acc(pred, true):
    n = max(pred.max(), true.max()) + 1
    C = np.zeros((n, n), dtype=np.int64)
    for p, t in zip(pred, true):
        C[p, t] += 1
    r, c = linear_sum_assignment(-C)
    return C[r, c].sum() / len(pred)


def dendrogram_purity(Z, labels, k):
    pred = fcluster(Z, t=k, criterion="maxclust") - 1
    n_lbl = labels.max() + 1
    pure  = sum(
        mask.sum()
        for c in np.unique(pred)
        for mask in [(pred == c)]
        if mask.sum() > 0 and
           np.bincount(labels[mask], minlength=n_lbl).max() / mask.sum() > 0.5
    )
    return pure / len(labels)


def evaluate(Z, labels, k, tag=""):
    pred = fcluster(Z, t=k, criterion="maxclust") - 1
    acc  = hungarian_acc(pred, labels)
    nmi  = normalized_mutual_info_score(labels, pred)
    ari  = adjusted_rand_score(labels, pred)
    dp   = dendrogram_purity(Z, labels, k)
    print(f"  {tag:35s}  K={k:3d}  ACC={acc:.4f}  NMI={nmi:.4f}  ARI={ari:.4f}  DP={dp:.4f}")
    return dict(k=k, acc=acc, nmi=nmi, ari=ari, dp=dp)


print("\n=== Species-level evaluation on D_U ===")
results = [evaluate(Z_u, labels_u, k, f"Ward K={k}") for k in range(2, N_CLASSES * 2 + 1)]

print(f"\n--- At true species K={N_CLASSES} ---")
evaluate(Z_u, labels_u, N_CLASSES, "Ward (true K, species)")

print(f"\n--- Genus-level evaluation (K={len(genus_names)}) ---")
evaluate(Z_u, genus_u, len(genus_names), "Ward (genus level)")

# ── Metrics plot ────────────────────────────────────────────
ks  = [r["k"] for r in results]
fig, axes = plt.subplots(1, 4, figsize=(20, 4))
for ax, metric in zip(axes, ["acc", "nmi", "ari", "dp"]):
    vals = [r[metric] for r in results]
    ax.plot(ks, vals, "o-", markersize=3)
    ax.axvline(N_CLASSES,   color="green", linestyle=":", label=f"True K={N_CLASSES}")
    ax.axvline(estimated_k, color="red",   linestyle="--", label=f"Est. K={estimated_k}")
    ax.set_xlabel("K")
    ax.set_ylabel(metric.upper())
    ax.set_title(metric.upper())
    ax.legend(fontsize=7)
plt.suptitle("BioHGCD-DIBaS — DINOv2 + Ward  (unlabeled D_U)")
plt.tight_layout()
plt.savefig("dibas_metrics.png", dpi=150)
plt.close()
print("Saved: dibas_metrics.png")

# ============================================================
# 10. ANCHOR-GUIDED NOVEL DETECTION — two versions
# ============================================================
#
# v1  Single-anchor   : 1 labeled known-class sample → whole cluster = KNOWN
#                       Fast, conservative, but one stray sample can rescue a novel cluster
#
# v2  Majority-vote   : a cluster is KNOWN only if it has >= min_anchors labeled samples
#                       AND they all agree on the same known class (majority >= majority_frac)
#                       More robust — one stray mislabeled sample cannot flip the decision

def _build_full_linkage(features_all):
    feats_r  = reduce_pca(features_all, PCA_COMPONENTS)
    Z_all, _ = build_ward(feats_r)
    return Z_all


def detect_v1_single_anchor(Z_all, labels_all, labeled_idx, known_ids, cut_k):
    """Original rule: any single labeled known-class sample tags the cluster as KNOWN."""
    pred        = fcluster(Z_all, t=cut_k, criterion="maxclust") - 1
    labeled_set = set(labeled_idx.tolist())
    known_cids  = {
        pred[i]
        for i in range(len(pred))
        if i in labeled_set and labels_all[i] in known_ids
    }
    novel_cids = set(np.unique(pred)) - known_cids
    return pred, known_cids, novel_cids


def detect_v2_majority_vote(Z_all, labels_all, labeled_idx, known_ids, cut_k,
                             min_anchors=3, majority_frac=0.6):
    """
    Improved rule: a cluster is KNOWN only if:
      1. It contains at least min_anchors labeled known-class samples, AND
      2. At least majority_frac of those labeled samples agree on the same known species.
    Everything else is NOVEL.
    """
    pred        = fcluster(Z_all, t=cut_k, criterion="maxclust") - 1
    labeled_set = set(labeled_idx.tolist())

    # Collect labeled known-class votes per cluster
    cluster_votes = {}   # cluster_id → Counter of known species ids
    for i in range(len(pred)):
        if i in labeled_set and labels_all[i] in known_ids:
            c = pred[i]
            if c not in cluster_votes:
                cluster_votes[c] = {}
            sp = labels_all[i]
            cluster_votes[c][sp] = cluster_votes[c].get(sp, 0) + 1

    known_cids = set()
    for c, votes in cluster_votes.items():
        total    = sum(votes.values())
        top_vote = max(votes.values())
        if total >= min_anchors and top_vote / total >= majority_frac:
            known_cids.add(c)

    novel_cids = set(np.unique(pred)) - known_cids
    return pred, known_cids, novel_cids


def print_detection_results(pred, known_cids, novel_cids, labels_all, version_name):
    print(f"\n{'='*60}")
    print(f"  {version_name}")
    print(f"{'='*60}")
    print(f"  Known clusters : {len(known_cids)}  → {sorted(known_cids)}")
    print(f"  Novel clusters : {len(novel_cids)}  → {sorted(novel_cids)}")
    print()
    print(f"  {'Cluster':>9}  {'Tag':>10}  {'N':>4}  Top-2 species")
    for c in sorted(np.unique(pred)):
        mask   = pred == c
        counts = np.bincount(labels_all[mask], minlength=N_CLASSES)
        top2   = [(class_names[i], int(counts[i]))
                  for i in counts.argsort()[::-1][:2] if counts[i] > 0]
        tag    = "KNOWN" if c in known_cids else "NOVEL*"
        print(f"  {c:>9}  {tag:>10}  {mask.sum():>4}  {top2}")


def compare_detections_across_k(Z_all, labels_all, labeled_idx, known_ids,
                                  k_values, min_anchors=3, majority_frac=0.6):
    """Side-by-side comparison of v1 vs v2 at multiple K values."""
    print(f"\n{'='*70}")
    print(f"  Comparison: Single-anchor (v1) vs Majority-vote (v2)")
    print(f"  v2 settings: min_anchors={min_anchors}, majority_frac={majority_frac}")
    print(f"{'='*70}")
    print(f"  {'K':>4}  {'v1 known':>9} {'v1 novel':>9}  {'v2 known':>9} {'v2 novel':>9}  {'Disagreements':>14}")
    for k in k_values:
        _, k1, n1 = detect_v1_single_anchor(Z_all, labels_all, labeled_idx, known_ids, k)
        _, k2, n2 = detect_v2_majority_vote(Z_all, labels_all, labeled_idx, known_ids, k,
                                             min_anchors, majority_frac)
        # Clusters v1 calls KNOWN but v2 calls NOVEL (stray-sample rescues)
        rescued = k1 - k2
        print(f"  {k:>4}  {len(k1):>9} {len(n1):>9}  {len(k2):>9} {len(n2):>9}  "
              f"{len(rescued):>14}  {sorted(rescued) if rescued else '—'}")


# ── Build full-dataset linkage once ────────────────────────
print("\n--- Building full-dataset linkage for anchor detection ---")
Z_all = _build_full_linkage(features_ft)

# ── Run both versions at true K ────────────────────────────
pred_v1, known_v1, novel_v1 = detect_v1_single_anchor(
    Z_all, labels, labeled_idx, KNOWN_IDS, N_CLASSES)
pred_v2, known_v2, novel_v2 = detect_v2_majority_vote(
    Z_all, labels, labeled_idx, KNOWN_IDS, N_CLASSES, min_anchors=3, majority_frac=0.6)

print_detection_results(pred_v1, known_v1, novel_v1, labels,
                        "v1 — Single-anchor  (≥1 labeled sample → KNOWN)")
print_detection_results(pred_v2, known_v2, novel_v2, labels,
                        "v2 — Majority-vote  (≥3 samples, ≥60% agree → KNOWN)")

# ── Cross-K comparison ──────────────────────────────────────
compare_detections_across_k(
    Z_all, labels, labeled_idx, KNOWN_IDS,
    k_values=[20, 25, 30, 33, 39, 45, 55],
    min_anchors=3, majority_frac=0.6
)

# Use v2 for downstream (UMAP, summary) — it's the more robust result
pred_all, known_cids, novel_cids = pred_v2, known_v2, novel_v2

# ============================================================
# 11. UMAP
# ============================================================

def plot_umap(features, labels_gt, pred, known_cids, novel_cids):
    if not HAS_UMAP:
        print("Skipping UMAP.")
        return

    print("\nRunning UMAP …")
    reducer = umap_lib.UMAP(n_components=2, random_state=SEED,
                            n_neighbors=10, min_dist=0.1)
    emb     = reducer.fit_transform(normalize(features, norm="l2"))

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Left — ground truth coloured by Gram stain
    ax   = axes[0]
    cmap = matplotlib.colormaps.get_cmap("tab20").resampled(N_CLASSES)
    for i, cname in enumerate(class_names):
        m    = labels_gt == i
        gram = "Gram+" if cname in GRAM_POSITIVE else "Gram-"
        marker = "o" if gram == "Gram+" else "^"
        ax.scatter(emb[m, 0], emb[m, 1], c=[cmap(i)],
                   label=f"{cname.split('.')[0]} [{gram}]",
                   marker=marker, s=25, alpha=0.75)
    ax.set_title("Ground Truth  (● Gram+  ▲ Gram−)")
    ax.legend(fontsize=5, markerscale=1.2, ncol=2, loc="best")
    ax.set_xticks([]); ax.set_yticks([])

    # Right — predicted clusters
    ax    = axes[1]
    cmap2 = matplotlib.colormaps.get_cmap("tab20").resampled(len(np.unique(pred)))
    for j, c in enumerate(sorted(np.unique(pred))):
        m     = pred == c
        color = "red" if c in novel_cids else cmap2(j)
        tag   = "NOVEL*" if c in novel_cids else "known"
        ax.scatter(emb[m, 0], emb[m, 1], c=[color],
                   label=f"C{c}[{tag}]", s=25, alpha=0.75)
    ax.set_title("Predicted Clusters  (red = novel candidate)")
    ax.legend(fontsize=6, markerscale=1.2, loc="best")
    ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle("BioHGCD-DIBaS — DINOv2 + Ward + Anchor-guided", fontsize=13)
    plt.tight_layout()
    plt.savefig("dibas_umap.png", dpi=150)
    plt.close()
    print("Saved: dibas_umap.png")


feats_all_r2 = reduce_pca(features_ft, PCA_COMPONENTS)
plot_umap(feats_all_r2, labels, pred_all, known_cids, novel_cids)

# ============================================================
# SUMMARY
# ============================================================

best_species = max(results, key=lambda r: r["nmi"])
print("\n" + "=" * 65)
print("BioHGCD-DIBaS complete")
print("=" * 65)
print(f"  Best NMI at K={best_species['k']}: "
      f"ACC={best_species['acc']:.4f}  NMI={best_species['nmi']:.4f}  "
      f"ARI={best_species['ari']:.4f}  DP={best_species['dp']:.4f}")
print(f"  Estimated K (gap statistic): {estimated_k}  |  True K: {N_CLASSES}")
print("\nOutput files:")
for f in [
    "dibas_dendrogram.png      — Ward dendrogram on D_U",
    "dibas_genus_dendrogram.png— genus-level mean-feature dendrogram",
    "dibas_gap.png             — K estimation via gap statistic",
    "dibas_metrics.png         — ACC/NMI/ARI/DP vs K",
    "dibas_umap.png            — UMAP: ground truth vs predicted",
]:
    print(f"  {f}")
