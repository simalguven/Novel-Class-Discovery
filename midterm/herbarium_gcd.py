"""
Herbarium GCD — Taxonomy-Guided Hierarchical Generalized Category Discovery
============================================================================
Dataset : PlantNet-300K  (1,081 plant species, 306K images)
Backbone: DINOv2 ViT-B/14  (frozen)
Novel   : 3-level contrastive loss  (species / genus / family)
Cluster : Taxonomy-constrained Ward hierarchical clustering
Detect  : Graded anchor detection  (novel species / novel genus / novel family)

Run:
    conda activate openldn
    pip install requests  # for GBIF taxonomy lookup (one-time)
    python herbarium_gcd.py
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"   # prevent OpenMP conflict between torch and sklearn

import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment
from scipy.cluster.hierarchy import linkage, fcluster, cophenet, dendrogram
from scipy.spatial.distance import pdist

try:
    import umap as umap_lib
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("requests not installed — family lookup disabled. pip install requests")

# ============================================================
# 1. CONFIG
# ============================================================

DATA_DIR      = Path("./PlantNet300K/plantnet_300K/images_train")
META_DIR      = Path("./PlantNet300K/plantnet_300K")
FEAT_CACHE    = Path("./plantnet_dinov2_raw.npy")
FTFT_CACHE    = Path("./plantnet_dinov2_finetuned.npy")
LBL_CACHE     = Path("./plantnet_labels.npy")
GENUS_CACHE   = Path("./plantnet_genus_labels.npy")
FAMILY_CACHE  = Path("./plantnet_family_labels.npy")
TAXON_CACHE   = Path("./plantnet_taxonomy.json")

DEVICE = (
    "cuda"  if torch.cuda.is_available()         else
    "mps"   if torch.backends.mps.is_available() else
    "cpu"
)
print(f"Device: {DEVICE}")

SEED             = 42
BATCH_SIZE       = 32
LABELED_FRACTION = 0.5      # 50% of known species images go into D_L
KNOWN_FRACTION   = 0.5      # 50% of all species are "known"
SUPCON_EPOCHS    = 60
SUPCON_LR        = 5e-4
PROJ_DIM         = 256
PCA_COMPONENTS   = 128
CLUSTER_SAMPLE_U = 6000   # D_U samples for Ward (pdist is O(N²) memory)
CLUSTER_SAMPLE_L = 2000   # D_L anchor samples for graded detection

# Contrastive loss weights for three taxonomy levels
LAMBDA_SPECIES = 1.0
LAMBDA_GENUS   = 0.5
LAMBDA_FAMILY  = 0.25

torch.manual_seed(SEED)
np.random.seed(SEED)

# ============================================================
# 2. TAXONOMY — derive genus from species name, lookup family
# ============================================================

def load_species_id_to_name() -> dict:
    """
    Load PlantNet species name mapping.
    Returns dict: species_id (str) -> scientific name (str, spaces not underscores)
    e.g. '1355868' -> 'Lactuca virosa'
    """
    name_file = META_DIR / "plantnet300K_species_names.json"
    with open(name_file) as f:
        raw = json.load(f)
    # Convert underscore-separated names to space-separated
    return {k: v.replace("_", " ") for k, v in raw.items()}


def derive_genus(species_name: str) -> str:
    """Genus is the first word of the binomial scientific name."""
    return species_name.strip().split()[0]


def lookup_family_gbif(genus: str, cache: dict) -> str:
    """Query GBIF API for plant family given genus name."""
    if genus in cache:
        return cache[genus]
    if not HAS_REQUESTS:
        return "Unknown"
    try:
        url = f"https://api.gbif.org/v1/species/match?name={genus}&rank=GENUS&kingdom=Plantae"
        resp = requests.get(url, timeout=5)
        data = resp.json()
        family = data.get("family", "Unknown")
        cache[genus] = family
        time.sleep(0.05)   # be polite to the API
        return family
    except Exception:
        cache[genus] = "Unknown"
        return "Unknown"


def build_taxonomy(class_names: list) -> dict:
    """
    Build species_folder_id → genus → family mapping.
    class_names are numeric folder names (e.g. '1355868').
    Returns dict with keys 'genus_map', 'family_map',
    'genus_names', 'family_names', 'id_to_sciname'.
    """
    if TAXON_CACHE.exists():
        print("Loading cached taxonomy …")
        with open(TAXON_CACHE) as f:
            return json.load(f)

    print("Building taxonomy …")
    id_to_sciname = load_species_id_to_name()   # '1355868' -> 'Lactuca virosa'

    # Derive genus from species scientific name
    genus_map = {}    # folder_id -> genus string
    for cname in class_names:
        sciname = id_to_sciname.get(cname, cname)
        genus_map[cname] = derive_genus(sciname)

    unique_genera = sorted(set(genus_map.values()))
    print(f"  {len(unique_genera)} genera")

    # Lookup family per genus via GBIF
    print(f"  Looking up families for {len(unique_genera)} genera via GBIF …")
    family_cache = {}
    family_map   = {}
    for i, cname in enumerate(class_names):
        genus  = genus_map[cname]
        family = lookup_family_gbif(genus, family_cache)
        family_map[cname] = family
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(class_names)} done")

    unique_families = sorted(set(family_map.values()))
    print(f"  {len(unique_families)} families")

    taxonomy = {
        "id_to_sciname": id_to_sciname,
        "genus_map":     genus_map,
        "family_map":    family_map,
        "genus_names":   sorted(set(genus_map.values())),
        "family_names":  sorted(set(family_map.values())),
    }
    with open(TAXON_CACHE, "w") as f:
        json.dump(taxonomy, f, indent=2)
    print("Taxonomy saved.")
    return taxonomy


# ============================================================
# 3. DATA LOADING
# ============================================================

DINO_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def extract_dinov2_features():
    """Extract DINOv2 features from PlantNet train images."""
    if FEAT_CACHE.exists() and LBL_CACHE.exists():
        print("Loading cached DINOv2 features …")
        return np.load(FEAT_CACHE), np.load(LBL_CACHE)

    print("Loading DINOv2 ViT-B/14 …")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model.eval().to(DEVICE)

    dataset = datasets.ImageFolder(str(DATA_DIR), transform=DINO_TRANSFORM)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=4)

    all_feats, all_labels = [], []
    n_batches = len(loader)
    with torch.no_grad():
        for i, (imgs, lbls) in enumerate(loader):
            feats = model(imgs.to(DEVICE))
            all_feats.append(feats.cpu().numpy())
            all_labels.append(lbls.numpy())
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{n_batches} batches")

    features = np.concatenate(all_feats,  axis=0)
    labels   = np.concatenate(all_labels, axis=0)
    np.save(FEAT_CACHE, features)
    np.save(LBL_CACHE,  labels)
    print(f"Features: {features.shape}  Labels: {labels.shape}")
    return features, labels


# ============================================================
# 4. GCD SPLIT — known/novel by species, stratified
# ============================================================

def build_gcd_split(labels, n_classes, known_fraction=0.5,
                    labeled_fraction=0.5, seed=42):
    """
    Split species into known (labeled) and novel (to discover).
    Known/novel split is seeded-random — reproducible but not alphabetical.
    """
    rng = np.random.default_rng(seed)

    # Deterministic known/novel species split
    all_class_ids = list(range(n_classes))
    rng.shuffle(all_class_ids)
    n_known   = int(n_classes * known_fraction)
    known_ids = set(all_class_ids[:n_known])
    novel_ids = set(all_class_ids[n_known:])

    labeled_idx, unlabeled_idx = [], []
    for kid in known_ids:
        idx = np.where(labels == kid)[0]
        rng.shuffle(idx)
        n_l = max(1, int(len(idx) * labeled_fraction))
        labeled_idx.extend(idx[:n_l].tolist())
        unlabeled_idx.extend(idx[n_l:].tolist())
    for nid in novel_ids:
        unlabeled_idx.extend(np.where(labels == nid)[0].tolist())

    return (np.array(labeled_idx), np.array(unlabeled_idx),
            sorted(known_ids), sorted(novel_ids))


def subsample_stratified(indices, group_labels, max_n, seed=42):
    """Stratified subsample of indices by group_labels, capped at max_n."""
    indices = np.asarray(indices)
    if len(indices) <= max_n:
        return indices
    rng    = np.random.default_rng(seed)
    groups = defaultdict(list)
    for idx, g in zip(indices, group_labels):
        groups[g].append(idx)
    per_group = max(1, max_n // len(groups))
    sampled   = []
    for g_idxs in groups.values():
        arr = np.array(g_idxs)
        rng.shuffle(arr)
        sampled.extend(arr[:per_group].tolist())
    sampled = np.array(sampled)
    if len(sampled) > max_n:
        rng.shuffle(sampled)
        sampled = sampled[:max_n]
    return sampled


# ============================================================
# 5. THREE-LEVEL SUPCON LOSS
# ============================================================

class ProjectionHead(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=1024, out_dim=256):
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
    """SupCon loss — safe against 0×-inf = NaN."""
    n   = z.size(0)
    sim = torch.mm(z, z.T) / temperature
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    pos = (y.unsqueeze(0) == y.unsqueeze(1)) & ~eye
    if pos.sum() == 0:
        return torch.tensor(0.0, device=z.device, requires_grad=True)
    sim_no_self  = sim.masked_fill(eye, -1e4)
    log_prob     = sim_no_self - torch.logsumexp(sim_no_self, dim=1, keepdim=True)
    log_prob_pos = log_prob.masked_fill(~pos, 0.0)
    n_pos        = pos.float().sum(dim=1).clamp(min=1)
    return (-log_prob_pos.sum(dim=1) / n_pos).mean()


def three_level_supcon(z, species_labels, genus_labels, family_labels,
                        lam_species=1.0, lam_genus=0.5, lam_family=0.25):
    """
    Three-level supervised contrastive loss:
      L = λ_s × L_species + λ_g × L_genus + λ_f × L_family

    Species level: pulls same-species images together most tightly
    Genus level  : pulls same-genus images together (softer)
    Family level : pulls same-family images together (softest)
    """
    L_s = supcon_loss(z, species_labels)
    L_g = supcon_loss(z, genus_labels)
    L_f = supcon_loss(z, family_labels)
    return lam_species * L_s + lam_genus * L_g + lam_family * L_f


def train_projection_head(features_all, labels_all,
                           genus_labels_all, family_labels_all,
                           labeled_idx):
    if FTFT_CACHE.exists():
        print("Loading cached fine-tuned features …")
        return np.load(FTFT_CACHE)

    head      = ProjectionHead(in_dim=features_all.shape[1], out_dim=PROJ_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(head.parameters(), lr=SUPCON_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SUPCON_EPOCHS)

    # Keep tensors on CPU; move per-batch to DEVICE to avoid OOM on MPS
    X_l  = torch.tensor(features_all[labeled_idx],      dtype=torch.float32)
    y_s  = torch.tensor(labels_all[labeled_idx],        dtype=torch.long)
    y_g  = torch.tensor(genus_labels_all[labeled_idx],  dtype=torch.long)
    y_f  = torch.tensor(family_labels_all[labeled_idx], dtype=torch.long)

    dataset = torch.utils.data.TensorDataset(X_l, y_s, y_g, y_f)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=512, shuffle=True, drop_last=True
    )

    print("Training 3-level SupCon projection head …")
    head.train()
    for epoch in range(1, SUPCON_EPOCHS + 1):
        epoch_loss, n_batches = 0.0, 0
        for x_b, ys_b, yg_b, yf_b in loader:
            x_b  = x_b.to(DEVICE);  ys_b = ys_b.to(DEVICE)
            yg_b = yg_b.to(DEVICE); yf_b = yf_b.to(DEVICE)
            optimizer.zero_grad()
            z    = head(x_b)
            loss = three_level_supcon(z, ys_b, yg_b, yf_b,
                                       LAMBDA_SPECIES, LAMBDA_GENUS, LAMBDA_FAMILY)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1
        scheduler.step()
        if epoch % 10 == 0:
            print(f"  epoch {epoch:3d}/{SUPCON_EPOCHS}  loss={epoch_loss/max(n_batches,1):.4f}")

    head.eval()
    with torch.no_grad():
        chunks = []
        for i in range(0, len(features_all), 512):
            chunk = torch.tensor(
                features_all[i:i + 512], dtype=torch.float32
            ).to(DEVICE)
            chunks.append(head(chunk).cpu().numpy())
        finetuned = np.concatenate(chunks, axis=0)

    np.save(FTFT_CACHE, finetuned)
    print(f"Fine-tuned features: {finetuned.shape}")
    return finetuned


# ============================================================
# 6. TAXONOMY-CONSTRAINED WARD CLUSTERING
# ============================================================

def reduce_pca(features, n_components=128, seed=42):
    n_comp  = min(n_components, features.shape[0] - 1, features.shape[1] - 1)
    pca     = PCA(n_components=n_comp, random_state=seed)
    reduced = pca.fit_transform(features)
    var_exp = pca.explained_variance_ratio_.sum()
    print(f"PCA {features.shape[1]}d → {reduced.shape[1]}d  "
          f"(var explained: {var_exp:.3f})")
    return reduced


def build_ward(features):
    normed = normalize(features, norm="l2")
    Z      = linkage(normed, method="ward", metric="euclidean")
    c, _   = cophenet(Z, pdist(normed))
    print(f"Cophenetic correlation (Ward): {c:.4f}")
    return Z, normed


def taxonomy_constrained_ward(features, genus_labels, family_labels,
                               labeled_local_idx=None, species_labels=None,
                               genus_penalty=0.3, family_penalty=0.1):
    """
    Ward linkage with two tiers of constraints:

    Soft (all pairs):
        adjusted_dist = dist × (1 + genus_penalty × cross_genus
                                  + family_penalty × cross_family)

    Hard (labeled pairs only, when labeled_local_idx + species_labels provided):
        Same known species  → dist = 0          (must-link: Ward merges first)
        Diff known species  → dist = LARGE       (cannot-link: Ward never merges)

    Hard constraints take priority and override the soft penalty for labeled pairs.
    """
    print("Building taxonomy-constrained Ward linkage …")
    normed = normalize(features, norm="l2")

    from scipy.spatial.distance import squareform
    base_dists = squareform(pdist(normed, metric="euclidean"))

    # ── Soft penalty (all pairs) ─────────────────────────────
    genus_diff  = (genus_labels[:, None] != genus_labels[None, :]).astype(float)
    family_diff = (family_labels[:, None] != family_labels[None, :]).astype(float)
    penalty     = 1 + genus_penalty * genus_diff + family_penalty * family_diff
    adj_dists   = (base_dists * penalty).astype(np.float64)

    # ── Hard constraints from labeled data ───────────────────
    if labeled_local_idx is not None and species_labels is not None:
        l_idx     = np.asarray(labeled_local_idx)
        l_species = species_labels[l_idx]

        # Boolean masks over the (n_labeled × n_labeled) block
        same = (l_species[:, None] == l_species[None, :])   # must-link
        diff = ~same                                          # cannot-link

        CANNOT_LINK = adj_dists.max() * 1e3   # large but finite — no float overflow

        # Vectorised write into adj_dists
        rows, cols = np.meshgrid(l_idx, l_idx, indexing="ij")
        adj_dists[rows[same], cols[same]] = 0.0
        adj_dists[rows[diff], cols[diff]] = CANNOT_LINK
        np.fill_diagonal(adj_dists, 0.0)      # self-distance must stay 0

        n_ml = int(same.sum() - len(l_idx))   # exclude diagonal
        n_cl = int(diff.sum())
        print(f"  Hard constraints: {n_ml // 2} must-link, "
              f"{n_cl // 2} cannot-link pairs from {len(l_idx)} labeled samples")

    Z = linkage(squareform(adj_dists), method="ward")
    c, _ = cophenet(Z, pdist(normed))
    print(f"Cophenetic correlation (constrained Ward): {c:.4f}")
    return Z, normed


# ============================================================
# 7. EVALUATION
# ============================================================

def hungarian_acc(pred, true):
    n = max(pred.max(), true.max()) + 1
    C = np.zeros((n, n), dtype=np.int64)
    for p, t in zip(pred, true):
        C[p, t] += 1
    r, c = linear_sum_assignment(-C)
    return C[r, c].sum() / len(pred)


def dendrogram_purity(Z, labels, k):
    """Weighted-average per-cluster purity at a flat cut of K clusters."""
    pred   = fcluster(Z, t=k, criterion="maxclust") - 1
    n_lbl  = int(labels.max()) + 1
    total_pure, total_n = 0, 0
    for c in np.unique(pred):
        mask = pred == c
        if not mask.any():
            continue
        counts = np.bincount(labels[mask], minlength=n_lbl)
        total_pure += counts.max()
        total_n    += mask.sum()
    return total_pure / total_n if total_n > 0 else 0.0


def evaluate_level(Z, labels, k, tag=""):
    pred = fcluster(Z, t=k, criterion="maxclust") - 1
    acc  = hungarian_acc(pred, labels)
    nmi  = normalized_mutual_info_score(labels, pred)
    ari  = adjusted_rand_score(labels, pred)
    dp   = dendrogram_purity(Z, labels, k)
    print(f"  {tag:40s}  K={k:4d}  "
          f"ACC={acc:.4f}  NMI={nmi:.4f}  ARI={ari:.4f}  DP={dp:.4f}")
    return dict(k=k, acc=acc, nmi=nmi, ari=ari, dp=dp)


def kmeans_pca_baseline(features, labels, k, n_components=10, tag="UMAP+KMeans"):
    """L2-normalise → UMAP(10D) → KMeans(k). Returns metric dict.
    NUMBA_DISABLE_JIT=1 is set at module top to prevent segfault on macOS M-series.
    Falls back to PCA if umap-learn is not installed.
    """
    normed = normalize(features, norm="l2")
    if HAS_UMAP:
        print(f"    fitting UMAP(10D) on {normed.shape[0]} samples …")
        reducer = umap_lib.UMAP(n_components=10, random_state=SEED,
                                 n_neighbors=15, min_dist=0.1)
        emb = reducer.fit_transform(normed)
    else:
        print(f"    umap-learn not available — falling back to PCA(10D)")
        n_comp = min(n_components, normed.shape[0] - 1, normed.shape[1] - 1)
        emb = PCA(n_components=n_comp, random_state=SEED).fit_transform(normed)
    km      = KMeans(n_clusters=k, random_state=SEED, n_init=3)
    pred    = km.fit_predict(emb)
    n_lbl   = int(labels.max()) + 1
    acc     = hungarian_acc(pred, labels)
    nmi     = normalized_mutual_info_score(labels, pred)
    ari     = adjusted_rand_score(labels, pred)
    total_pure, total_n = 0, 0
    for c in np.unique(pred):
        mask = pred == c
        counts = np.bincount(labels[mask], minlength=n_lbl)
        total_pure += counts.max()
        total_n    += mask.sum()
    cp = total_pure / total_n if total_n > 0 else 0.0
    print(f"  {tag:45s}  K={k:4d}  "
          f"ACC={acc:.4f}  NMI={nmi:.4f}  ARI={ari:.4f}  CP={cp:.4f}")
    return dict(k=k, acc=acc, nmi=nmi, ari=ari, cp=cp)


def semisup_kmeans_fit(emb, labels_all, n_labeled, known_ids, k,
                        n_iters=100, seed=42):
    """
    Semi-supervised KMeans (figure 4 algorithm):
      - Known centroids initialised from labeled sample means
      - Novel centroids initialised with k-means++ on unlabeled points
      - Labeled points hard-constrained to their ground-truth class each iteration
    Returns full assignment array (length = n_labeled + n_unlabeled).
    """
    rng      = np.random.default_rng(seed)
    n, d     = emb.shape
    n_u      = n - n_labeled
    known_list = sorted(known_ids)
    n_known    = len(known_list)
    n_novel    = k - n_known
    cls2cen    = {cid: i for i, cid in enumerate(known_list)}  # known class → centroid idx

    centroids = np.zeros((k, d), dtype=np.float64)

    # ── Init known centroids from labeled means ──────────────
    for cid, ci in cls2cen.items():
        mask = labels_all[:n_labeled] == cid
        if mask.any():
            centroids[ci] = emb[:n_labeled][mask].mean(0)
        else:
            centroids[ci] = emb[n_labeled + rng.integers(0, n_u)]

    # ── k-means++ init for novel centroids ───────────────────
    emb_u = emb[n_labeled:]
    novel_idx = [int(rng.integers(0, n_u))]
    for _ in range(n_novel - 1):
        chosen = emb_u[novel_idx]                           # (m, d)
        D2     = ((emb_u[:, None, :] - chosen[None]) ** 2).sum(2).min(1)
        probs  = D2 / (D2.sum() + 1e-9)
        novel_idx.append(int(rng.choice(n_u, p=probs)))
    for j, ci in enumerate(range(n_known, k)):
        centroids[ci] = emb_u[novel_idx[j]]

    # Hard label assignments for labeled points (fixed throughout)
    assignments = np.zeros(n, dtype=np.int32)
    for i in range(n_labeled):
        assignments[i] = cls2cen.get(int(labels_all[i]), 0)

    # ── EM iterations ────────────────────────────────────────
    chunk = 512
    for it in range(n_iters):
        # Assign unlabeled points to nearest centroid (chunked for memory)
        for s in range(0, n_u, chunk):
            e   = min(s + chunk, n_u)
            D2  = ((emb_u[s:e, None, :] - centroids[None]) ** 2).sum(2)
            assignments[n_labeled + s : n_labeled + e] = D2.argmin(1)

        # Update centroids
        new_cen = np.zeros_like(centroids)
        counts  = np.zeros(k, dtype=np.int32)
        for i in range(n):
            new_cen[assignments[i]] += emb[i]
            counts[assignments[i]]  += 1
        mask_nonempty = counts > 0
        new_cen[mask_nonempty] /= counts[mask_nonempty, None]
        new_cen[~mask_nonempty] = centroids[~mask_nonempty]

        if np.allclose(centroids, new_cen, atol=1e-6):
            print(f"    converged at iteration {it + 1}")
            break
        centroids = new_cen

    return assignments


def raw_dinov2_semisup_kmeans(features_raw_sub, labels_sub, n_labeled,
                               known_ids, k, seed=42):
    """
    Raw DINOv2 → L2-norm → UMAP(10D) → semi-supervised KMeans.
    Evaluated on D_U only (rows n_labeled:) to match Ward / baseline eval.
    """
    if not HAS_UMAP:
        print("umap-learn not installed — skipping.")
        return None

    print(f"  Fitting UMAP(10D) on {len(features_raw_sub)} raw DINOv2 samples …")
    normed  = normalize(features_raw_sub, norm="l2")
    reducer = umap_lib.UMAP(n_components=10, random_state=seed,
                             n_neighbors=15, min_dist=0.1)
    emb     = reducer.fit_transform(normed)

    print(f"  Running semi-supervised KMeans (K={k}) …")
    assignments = semisup_kmeans_fit(emb, labels_sub, n_labeled,
                                      known_ids, k, seed=seed)

    # Evaluate on D_U only
    pred_u   = assignments[n_labeled:]
    labels_u = labels_sub[n_labeled:]
    n_lbl    = int(labels_u.max()) + 1
    acc      = hungarian_acc(pred_u, labels_u)
    nmi      = normalized_mutual_info_score(labels_u, pred_u)
    ari      = adjusted_rand_score(labels_u, pred_u)
    total_pure, total_n = 0, 0
    for c in np.unique(pred_u):
        mask   = pred_u == c
        counts = np.bincount(labels_u[mask], minlength=n_lbl)
        total_pure += counts.max()
        total_n    += mask.sum()
    cp  = total_pure / total_n if total_n > 0 else 0.0
    tag = "Raw DINOv2 → UMAP(10D) → SemiSup KMeans"
    print(f"  {tag:55s}  K={k:4d}  "
          f"ACC={acc:.4f}  NMI={nmi:.4f}  ARI={ari:.4f}  CP={cp:.4f}")
    return dict(tag=tag, acc=acc, nmi=nmi, ari=ari, cp=cp)


def semisup_umap_kmeans(features, labels, genus_labels_arr, family_labels_arr,
                         n_labeled, k,
                         lam_species=1.0, lam_genus=0.5, lam_family=0.25):
    """
    Three-level semi-supervised UMAP + KMeans with five fusion mechanisms.

    features       : all samples (D_L subsample first, then D_U subsample)
    n_labeled      : number of leading rows that are labeled anchors (D_L)
    Evaluated on D_U only (rows n_labeled:) to match the Ward/baseline eval set.

    Five fusion mechanisms:
      1. Direct concat       [E_s | E_g | E_f]          30D → KMeans
      2. Weighted concat     [λ_s·E_s | λ_g·E_g | λ_f·E_f]  30D → KMeans
      3. Weighted average    (λ_s·E_s + λ_g·E_g + λ_f·E_f)/Σλ  10D → KMeans
      4. PCA fusion          concat 30D → PCA(10D)       → KMeans
      5. Kernel sum+KernelPCA  weighted RBF kernels → KernelPCA(30D) → KMeans
    """
    if not HAS_UMAP:
        print("umap-learn not available — skipping semi-supervised UMAP.")
        return {}

    n      = len(features)
    normed = normalize(features, norm="l2")

    # y arrays: labeled rows get their label, unlabeled get -1 (umap-learn convention)
    def make_y(label_arr):
        y = np.full(n, -1, dtype=np.int32)
        y[:n_labeled] = label_arr[:n_labeled].astype(np.int32)
        return y

    y_s = make_y(labels)
    y_g = make_y(genus_labels_arr)
    y_f = make_y(family_labels_arr)

    # n_neighbors ~ sqrt(N) for semi-supervised so labels propagate through the graph.
    # target_weight mirrors lambda hierarchy: species tightest, family softest.
    n_neighbors = max(50, int(np.sqrt(n)))
    print(f"  n={n}  n_neighbors={n_neighbors}")

    def fit_umap(y, target_weight, tag):
        print(f"    fitting UMAP(10D) [{tag}, "
              f"n_neighbors={n_neighbors}, target_weight={target_weight}] …")
        reducer = umap_lib.UMAP(n_components=10, random_state=SEED,
                                 n_neighbors=n_neighbors, min_dist=0.1,
                                 target_weight=target_weight)
        return reducer.fit_transform(normed, y=y)

    E_s = fit_umap(y_s, target_weight=0.5,  tag="species")
    E_g = fit_umap(y_g, target_weight=0.35, tag="genus")
    E_f = fit_umap(y_f, target_weight=0.2,  tag="family")

    # Evaluate only on D_U portion for fair comparison with other baselines
    labels_u = labels[n_labeled:]
    n_lbl    = int(labels_u.max()) + 1

    def _metrics(pred_all):
        pred = pred_all[n_labeled:]
        acc  = hungarian_acc(pred, labels_u)
        nmi  = normalized_mutual_info_score(labels_u, pred)
        ari  = adjusted_rand_score(labels_u, pred)
        total_pure, total_n = 0, 0
        for c in np.unique(pred):
            mask = pred == c
            counts = np.bincount(labels_u[mask], minlength=n_lbl)
            total_pure += counts.max()
            total_n    += mask.sum()
        cp = total_pure / total_n if total_n > 0 else 0.0
        return acc, nmi, ari, cp

    def eval_kmeans(emb, tag):
        pred_all = KMeans(n_clusters=k, random_state=SEED, n_init=3).fit_predict(emb)
        acc, nmi, ari, cp = _metrics(pred_all)
        print(f"  {tag:55s}  K={k:4d}  "
              f"ACC={acc:.4f}  NMI={nmi:.4f}  ARI={ari:.4f}  CP={cp:.4f}")
        return dict(tag=tag, acc=acc, nmi=nmi, ari=ari, cp=cp)

    # L2-normalise each embedding before fusion so scales are comparable
    Es = normalize(E_s, norm="l2")
    Eg = normalize(E_g, norm="l2")
    Ef = normalize(E_f, norm="l2")
    lam_sum = lam_species + lam_genus + lam_family

    results = {}

    print("\n  --- Fusion 1: Direct concatenation [30D] ---")
    results["concat"] = eval_kmeans(
        np.hstack([Es, Eg, Ef]),
        "SemiSup UMAP — Direct concat [30D]"
    )

    print("\n  --- Fusion 2: Weighted concatenation [30D] ---")
    results["weighted_concat"] = eval_kmeans(
        np.hstack([lam_species * Es, lam_genus * Eg, lam_family * Ef]),
        "SemiSup UMAP — Weighted concat [30D]"
    )

    print("\n  --- Fusion 3: Weighted average [10D] ---")
    results["weighted_avg"] = eval_kmeans(
        (lam_species * Es + lam_genus * Eg + lam_family * Ef) / lam_sum,
        "SemiSup UMAP — Weighted average [10D]"
    )

    print("\n  --- Fusion 4: PCA on concatenation [30D→10D] ---")
    E_pca = PCA(n_components=10, random_state=SEED).fit_transform(np.hstack([Es, Eg, Ef]))
    results["pca_fusion"] = eval_kmeans(
        E_pca,
        "SemiSup UMAP — PCA fusion [30D→10D]"
    )

    print("\n  --- Fusion 5: Kernel sum + KernelPCA [30D] ---")
    def rbf_kernel(E):
        from scipy.spatial.distance import squareform, pdist
        D = squareform(pdist(E, metric="euclidean"))
        sigma = np.median(D[D > 0])
        return np.exp(-D ** 2 / (2 * sigma ** 2 + 1e-9))

    K_combined = (lam_species * rbf_kernel(Es) +
                  lam_genus   * rbf_kernel(Eg) +
                  lam_family  * rbf_kernel(Ef)) / lam_sum
    E_kpca = KernelPCA(n_components=30, kernel="precomputed",
                        random_state=SEED).fit_transform(K_combined)
    results["kernel_kpca"] = eval_kmeans(
        normalize(E_kpca, norm="l2"),
        "SemiSup UMAP — Kernel sum + KernelPCA [30D]"
    )

    return results


# ============================================================
# 8. GRADED NOVEL DETECTION (three levels)
# ============================================================

def graded_novel_detection(Z_all, labels_all, genus_labels_all, family_labels_all,
                             labeled_idx, known_ids, class_names,
                             genus_names, family_names, cut_k,
                             min_anchors=3, majority_frac=0.6):
    """
    Tags each cluster at three levels:
      KNOWN         — cluster dominated by a known labeled species
      NOVEL-SPECIES — novel species but genus is known (cluster near known genus)
      NOVEL-GENUS   — novel genus but family is known
      NOVEL-FAMILY  — completely new family
    """
    pred        = fcluster(Z_all, t=cut_k, criterion="maxclust") - 1
    labeled_set = set(labeled_idx.tolist())
    known_set   = set(known_ids)

    # --- species-level anchor votes ---
    cluster_species_votes  = defaultdict(lambda: defaultdict(int))
    cluster_genus_votes    = defaultdict(lambda: defaultdict(int))
    cluster_family_votes   = defaultdict(lambda: defaultdict(int))

    for i in range(len(pred)):
        if i in labeled_set and labels_all[i] in known_set:
            c = pred[i]
            cluster_species_votes[c][int(labels_all[i])]       += 1
            cluster_genus_votes[c][int(genus_labels_all[i])]   += 1
            cluster_family_votes[c][int(family_labels_all[i])] += 1

    # Known genera and families from labeled data
    labeled_genus_ids  = set(genus_labels_all[labeled_idx].tolist())
    labeled_family_ids = set(family_labels_all[labeled_idx].tolist())

    def majority_tag(votes):
        total    = sum(votes.values())
        top_vote = max(votes.values()) if votes else 0
        return total >= min_anchors and top_vote / max(total, 1) >= majority_frac

    results = {}
    for c in sorted(np.unique(pred)):
        mask = pred == c

        if majority_tag(cluster_species_votes[c]):
            tag = "KNOWN"
        else:
            # Check if cluster's genus/family overlaps with known taxa
            # Use majority genus of unlabeled samples in cluster
            cluster_genera   = genus_labels_all[mask]
            cluster_families = family_labels_all[mask]
            dominant_genus   = int(np.bincount(cluster_genera).argmax())
            dominant_family  = int(np.bincount(cluster_families).argmax())

            if dominant_genus in labeled_genus_ids:
                tag = "NOVEL-SPECIES"   # new species in known genus
            elif dominant_family in labeled_family_ids:
                tag = "NOVEL-GENUS"     # new genus in known family
            else:
                tag = "NOVEL-FAMILY"    # completely new family

        results[c] = {
            "tag":    tag,
            "n":      int(mask.sum()),
            "dominant_genus":  genus_names[int(np.bincount(genus_labels_all[mask]).argmax())],
            "dominant_family": family_names[int(np.bincount(family_labels_all[mask]).argmax())],
        }

    # Summary
    tag_counts = defaultdict(int)
    for v in results.values():
        tag_counts[v["tag"]] += 1

    print(f"\n{'='*60}")
    print(f"  Graded Novel Detection  (K={cut_k})")
    print(f"{'='*60}")
    for tag in ["KNOWN", "NOVEL-SPECIES", "NOVEL-GENUS", "NOVEL-FAMILY"]:
        print(f"  {tag:15s}: {tag_counts[tag]:4d} clusters")

    print(f"\n  {'Cluster':>9}  {'Tag':>14}  {'N':>5}  "
          f"{'Dominant Genus':>25}  {'Dominant Family':>25}")
    for c, info in sorted(results.items()):
        print(f"  {c:>9}  {info['tag']:>14}  {info['n']:>5}  "
              f"{info['dominant_genus']:>25}  {info['dominant_family']:>25}")

    return pred, results


# ============================================================
# 9. VISUALIZATION
# ============================================================

def plot_metrics(results, n_true, estimated_k, tag, fname):
    ks  = [r["k"] for r in results]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    for ax, metric in zip(axes, ["acc", "nmi", "ari", "dp"]):
        vals = [r[metric] for r in results]
        ax.plot(ks, vals, "o-", markersize=3)
        ax.axvline(n_true,      color="green", linestyle=":",  label=f"True K={n_true}")
        ax.axvline(estimated_k, color="red",   linestyle="--", label=f"Est. K={estimated_k}")
        ax.set_xlabel("K"); ax.set_ylabel(metric.upper())
        ax.set_title(metric.upper()); ax.legend(fontsize=7)
    plt.suptitle(f"Herbarium GCD — DINOv2 + Taxonomy Ward  ({tag})")
    plt.tight_layout()
    plt.savefig(fname, dpi=150); plt.close()
    print(f"Saved: {fname}")


def plot_umap_herbarium(features, labels, genus_labels, pred_all,
                        detection_results, fname):
    if not HAS_UMAP:
        print("Skipping UMAP (umap-learn not installed).")
        return
    print("Running UMAP …")
    reducer = umap_lib.UMAP(n_components=2, random_state=SEED,
                             n_neighbors=15, min_dist=0.1)
    emb     = reducer.fit_transform(normalize(features, norm="l2"))

    pred = pred_all   # cluster assignment per sample from graded_novel_detection
    tag_to_color = {
        "KNOWN":         "steelblue",
        "NOVEL-SPECIES": "orange",
        "NOVEL-GENUS":   "red",
        "NOVEL-FAMILY":  "darkred",
    }

    fig, ax = plt.subplots(figsize=(12, 10))
    for tag, color in tag_to_color.items():
        cluster_ids = [c for c, info in detection_results.items()
                       if info["tag"] == tag]
        if not cluster_ids:
            continue
        mask = np.isin(pred, cluster_ids)
        ax.scatter(emb[mask, 0], emb[mask, 1], c=color,
                   label=tag, s=8, alpha=0.5)
    ax.legend(markerscale=3, fontsize=10)
    ax.set_title("Herbarium GCD — UMAP coloured by discovery tag")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(fname, dpi=150); plt.close()
    print(f"Saved: {fname}")


# ============================================================
# 10. GAP STATISTIC
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
        ref_lw = []
        for _ in range(n_refs):
            ref      = rng.uniform(features_normed.min(0), features_normed.max(0),
                                   size=features_normed.shape)
            ref_pred = fcluster(linkage(ref, method="ward"),
                                t=k, criterion="maxclust") - 1
            ref_lw.append(np.log(wcss(ref_pred, ref) + 1e-9))
        gaps.append(np.mean(ref_lw) - log_wk)
        sks.append(np.std(ref_lw) * np.sqrt(1 + 1 / n_refs))

    gaps, sks = np.array(gaps), np.array(sks)
    klist     = list(k_range)
    best_k    = klist[0]
    for i in range(len(klist) - 1):
        if gaps[i] >= gaps[i + 1] - sks[i + 1]:
            best_k = klist[i]
            break

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.errorbar(klist, gaps, yerr=sks, fmt="o-", capsize=4)
    ax.axvline(best_k, color="red", linestyle="--", label=f"Est. K={best_k}")
    ax.set_xlabel("K"); ax.set_ylabel("Gap statistic")
    ax.set_title("PlantNet-300K — Novel Species Count Estimation")
    ax.legend(); plt.tight_layout()
    plt.savefig("plantnet_gap.png", dpi=150); plt.close()
    print(f"Estimated K = {best_k}")
    return best_k


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # ── 1. Load data ────────────────────────────────────────
    if not DATA_DIR.exists():
        print(f"ERROR: dataset not found at {DATA_DIR}")
        print("Make sure PlantNet-300K is downloaded and extracted.")
        exit(1)

    # ImageFolder reads numeric species-ID folder names alphabetically
    tmp_ds      = datasets.ImageFolder(str(DATA_DIR))
    class_names = tmp_ds.classes           # e.g. ['1355868', '1355920', ...]
    N_CLASSES   = len(class_names)
    print(f"Found {N_CLASSES} species")

    # ── 2. Build taxonomy ───────────────────────────────────
    taxonomy     = build_taxonomy(class_names)
    genus_names  = taxonomy["genus_names"]
    family_names = taxonomy["family_names"]
    genus_to_idx = {g: i for i, g in enumerate(genus_names)}
    family_to_idx = {f: i for i, f in enumerate(family_names)}

    print(f"Genera: {len(genus_names)}  Families: {len(family_names)}")

    # ── 3. Extract DINOv2 features ──────────────────────────
    features_raw, labels = extract_dinov2_features()

    # Build per-sample genus/family label arrays
    if GENUS_CACHE.exists() and FAMILY_CACHE.exists():
        genus_labels  = np.load(GENUS_CACHE)
        family_labels = np.load(FAMILY_CACHE)
    else:
        genus_labels  = np.array([
            genus_to_idx[taxonomy["genus_map"][class_names[l]]] for l in labels
        ])
        family_labels = np.array([
            family_to_idx[taxonomy["family_map"][class_names[l]]] for l in labels
        ])
        np.save(GENUS_CACHE,  genus_labels)
        np.save(FAMILY_CACHE, family_labels)

    print(f"Samples: {len(labels)}  "
          f"Species: {N_CLASSES}  "
          f"Genera: {len(genus_names)}  "
          f"Families: {len(family_names)}")

    # ── 4. GCD split ────────────────────────────────────────
    labeled_idx, unlabeled_idx, known_ids, novel_ids = build_gcd_split(
        labels, N_CLASSES,
        known_fraction=KNOWN_FRACTION,
        labeled_fraction=LABELED_FRACTION,
    )
    print(f"\nGCD split:")
    print(f"  Known species : {len(known_ids)}")
    print(f"  Novel species : {len(novel_ids)}")
    print(f"  Labeled D_L   : {len(labeled_idx)}")
    print(f"  Unlabeled D_U : {len(unlabeled_idx)}")

    # ── 5. Three-level SupCon fine-tuning ───────────────────
    features_ft = train_projection_head(
        features_raw, labels, genus_labels, family_labels, labeled_idx
    )

    # ── 6. Hierarchical clustering on D_U ───────────────────
    # Ward pdist is O(N²) memory; subsample D_U to a tractable size.
    sub_u_idx = subsample_stratified(
        unlabeled_idx, genus_labels[unlabeled_idx], CLUSTER_SAMPLE_U
    )
    print(f"\n--- Standard Ward on D_U subsample "
          f"({len(sub_u_idx)}/{len(unlabeled_idx)}) ---")
    feats_u   = features_ft[sub_u_idx]
    labels_u  = labels[sub_u_idx]
    genus_u   = genus_labels[sub_u_idx]
    family_u  = family_labels[sub_u_idx]
    feats_u_r = reduce_pca(feats_u, PCA_COMPONENTS)
    Z_std, norm_u = build_ward(feats_u_r)

    print("\n--- Taxonomy-constrained Ward on D_U ---")
    Z_tax, _ = taxonomy_constrained_ward(
        feats_u_r, genus_u, family_u,
        genus_penalty=0.3, family_penalty=0.1
    )

    # Save both dendrograms
    for Z, tag, fname in [
        (Z_std, "Standard Ward",              "plantnet_dendrogram_std.png"),
        (Z_tax, "Taxonomy-Constrained Ward",  "plantnet_dendrogram_tax.png"),
    ]:
        fig, ax = plt.subplots(figsize=(18, 7))
        dendrogram(Z, ax=ax, truncate_mode="lastp", p=60,
                   leaf_rotation=90, leaf_font_size=6, show_contracted=True)
        ax.set_title(f"PlantNet-300K — {tag} Dendrogram (D_U)")
        ax.set_ylabel("Ward distance")
        plt.tight_layout()
        plt.savefig(fname, dpi=150); plt.close()
        print(f"Saved: {fname}")

    # ── 6b. Baseline: L2-norm → UMAP(10D) → KMeans ─────────
    print("\n=== Baseline: L2-norm → UMAP(10D) → KMeans (known K) ===")
    print("  Raw DINOv2 768D features (D_U subsample):")
    bl_raw = kmeans_pca_baseline(
        features_raw[sub_u_idx], labels_u, N_CLASSES,
        tag="Raw DINOv2 768D → UMAP(10D) → KMeans"
    )
    print("  SupCon fine-tuned 256D features (D_U subsample):")
    bl_ft = kmeans_pca_baseline(
        features_ft[sub_u_idx], labels_u, N_CLASSES,
        tag="SupCon 256D → UMAP(10D) → KMeans"
    )

    # ── 7. Genus-level sanity check dendrogram ──────────────
    print("\n--- Genus-level dendrogram ---")
    unique_genera = sorted(set(genus_labels.tolist()))
    genus_feats   = np.array([
        normalize(features_ft[genus_labels == g], norm="l2").mean(0)
        for g in unique_genera if (genus_labels == g).sum() > 0
    ])
    genus_feats   = normalize(genus_feats, norm="l2")
    Z_genus       = linkage(genus_feats, method="ward")
    fig, ax       = plt.subplots(figsize=(max(12, len(unique_genera) // 3), 6))
    dendrogram(Z_genus, ax=ax,
               labels=[genus_names[g] for g in unique_genera],
               leaf_rotation=90, leaf_font_size=7)
    ax.set_title("PlantNet-300K — Genus-level Dendrogram")
    ax.set_ylabel("Ward distance")
    plt.tight_layout()
    plt.savefig("plantnet_genus_dendrogram.png", dpi=150); plt.close()
    print("Saved: plantnet_genus_dendrogram.png")

    # ── 8. Gap statistic ────────────────────────────────────
    # Step ~N/50 keeps the sweep to ~100 K values; full sweep at 1K+ values
    # would require ~10K Ward fits on reference data (hours of runtime).
    print("\n--- Gap statistic ---")
    _step       = max(1, N_CLASSES // 50)
    k_range     = range(max(2, N_CLASSES // 4), N_CLASSES + N_CLASSES // 4, _step)
    estimated_k = gap_statistic(Z_std, norm_u, k_range=k_range)

    # ── 9. Multi-level evaluation ───────────────────────────
    N_GENERA  = len(genus_names)
    N_FAMILIES = len(family_names)

    print("\n=== Species-level evaluation (Standard Ward) ===")
    k_eval     = list(range(
        max(2, N_CLASSES - N_CLASSES // 5),
        N_CLASSES + N_CLASSES // 5 + 1,
        max(1, N_CLASSES // 20)
    ))
    std_results = [evaluate_level(Z_std, labels_u, k, f"Std Ward K={k}")
                   for k in k_eval]

    print("\n=== Species-level evaluation (Taxonomy-Constrained Ward) ===")
    tax_results = [evaluate_level(Z_tax, labels_u, k, f"Tax Ward K={k}")
                   for k in k_eval]

    print(f"\n--- At true K={N_CLASSES} ---")
    evaluate_level(Z_std, labels_u, N_CLASSES, "Std Ward (true K)")
    evaluate_level(Z_tax, labels_u, N_CLASSES, "Tax Ward (true K)")

    print(f"\n--- Genus-level (K={N_GENERA}) ---")
    # Map unlabeled labels to genus labels for genus-level eval
    n_genus_u = len(set(genus_u.tolist()))
    evaluate_level(Z_std, genus_u, n_genus_u, "Std Ward (genus level)")
    evaluate_level(Z_tax, genus_u, n_genus_u, "Tax Ward (genus level)")

    # Plot both on same figure for comparison
    ks = [r["k"] for r in std_results]
    fig, axes = plt.subplots(1, 4, figsize=(22, 4))
    for ax, metric in zip(axes, ["acc", "nmi", "ari", "dp"]):
        ax.plot(ks, [r[metric] for r in std_results], "b-o",
                markersize=3, label="Standard Ward")
        ax.plot(ks, [r[metric] for r in tax_results], "r-^",
                markersize=3, label="Taxonomy Ward")
        ax.axvline(N_CLASSES,   color="green", linestyle=":",  label=f"True K={N_CLASSES}")
        ax.axvline(estimated_k, color="gray",  linestyle="--", label=f"Est. K={estimated_k}")
        ax.set_xlabel("K"); ax.set_ylabel(metric.upper())
        ax.set_title(metric.upper()); ax.legend(fontsize=6)
    plt.suptitle("PlantNet-300K — Standard vs Taxonomy-Constrained Ward")
    plt.tight_layout()
    plt.savefig("plantnet_metrics.png", dpi=150); plt.close()
    print("Saved: plantnet_metrics.png")

    # ── 10. Graded novel detection ──────────────────────────
    # Combine D_L subsample (anchors) + D_U subsample (clusters).
    # pdist on 8K samples = ~256 MB — well within RAM limits.
    print("\n--- Building detection linkage (subsampled D_L + D_U) ---")
    sub_l_idx       = subsample_stratified(
        labeled_idx, genus_labels[labeled_idx], CLUSTER_SAMPLE_L
    )
    sub_all_idx     = np.concatenate([sub_l_idx, sub_u_idx])
    # Positions 0..len(sub_l_idx)-1 in sub_all_idx are labeled anchors
    sub_local_lbl   = np.arange(len(sub_l_idx))

    labels_sub  = labels[sub_all_idx]
    genus_sub   = genus_labels[sub_all_idx]
    family_sub  = family_labels[sub_all_idx]

    feats_sub_r  = reduce_pca(features_ft[sub_all_idx], PCA_COMPONENTS)
    Z_all_tax, _ = taxonomy_constrained_ward(
        feats_sub_r, genus_sub, family_sub,
        labeled_local_idx=sub_local_lbl,
        species_labels=labels_sub,
        genus_penalty=0.3, family_penalty=0.1
    )

    # Evaluate hard-constrained Ward on D_U portion only (fair comparison)
    n_l          = len(sub_l_idx)
    pred_hard    = fcluster(Z_all_tax, t=N_CLASSES, criterion="maxclust") - 1
    pred_hard_u  = pred_hard[n_l:]
    labels_hard_u = labels_sub[n_l:]
    n_lbl_h      = int(labels_hard_u.max()) + 1
    hard_acc     = hungarian_acc(pred_hard_u, labels_hard_u)
    hard_nmi     = normalized_mutual_info_score(labels_hard_u, pred_hard_u)
    hard_ari     = adjusted_rand_score(labels_hard_u, pred_hard_u)
    total_pure, total_n = 0, 0
    for c in np.unique(pred_hard_u):
        mask = pred_hard_u == c
        counts = np.bincount(labels_hard_u[mask], minlength=n_lbl_h)
        total_pure += counts.max(); total_n += mask.sum()
    hard_cp = total_pure / total_n if total_n > 0 else 0.0
    hard_result = dict(acc=hard_acc, nmi=hard_nmi, ari=hard_ari, cp=hard_cp)
    print(f"\n--- Hard-constrained Ward on D_U (K={N_CLASSES}) ---")
    print(f"  ACC={hard_acc:.4f}  NMI={hard_nmi:.4f}  "
          f"ARI={hard_ari:.4f}  CP={hard_cp:.4f}")

    pred_all, detection_results = graded_novel_detection(
        Z_all_tax, labels_sub, genus_sub, family_sub,
        sub_local_lbl, known_ids, class_names,
        genus_names, family_names,
        cut_k=N_CLASSES,
        min_anchors=3, majority_frac=0.6
    )

    # ── 10b. Semi-supervised UMAP + KMeans (5 fusion mechanisms) ─
    # ── 6c. Raw DINOv2 → UMAP(10D) → semi-supervised KMeans ─
    print("\n=== Raw DINOv2 → UMAP(10D) → Semi-supervised KMeans ===")
    bl_semisup = raw_dinov2_semisup_kmeans(
        features_raw[sub_all_idx], labels_sub,
        n_labeled=len(sub_l_idx), known_ids=known_ids, k=N_CLASSES
    )

    print("\n=== Semi-supervised UMAP+KMeans (3-level taxonomy, 5 fusions) ===")
    ss_results = semisup_umap_kmeans(
        features_ft[sub_all_idx], labels_sub, genus_sub, family_sub,
        n_labeled=len(sub_l_idx), k=N_CLASSES,
        lam_species=LAMBDA_SPECIES, lam_genus=LAMBDA_GENUS, lam_family=LAMBDA_FAMILY
    )

    # ── 11. UMAP ────────────────────────────────────────────
    plot_umap_herbarium(feats_sub_r, labels_sub, genus_sub,
                         pred_all, detection_results, "plantnet_umap.png")

    # ── Summary ─────────────────────────────────────────────
    best_std = max(std_results, key=lambda r: r["nmi"])
    best_tax = max(tax_results, key=lambda r: r["nmi"])
    print("\n" + "=" * 75)
    print("Herbarium GCD — Method Comparison (D_U subsample, true K)")
    print("=" * 75)
    header = f"  {'Method':<45}  {'ACC':>6}  {'NMI':>6}  {'ARI':>6}  {'DP/CP':>6}"
    print(header)
    print("  " + "-" * 71)

    def _row(name, r, pk="cp"):
        print(f"  {name:<55}  "
              f"{r['acc']:>6.4f}  {r['nmi']:>6.4f}  {r['ari']:>6.4f}  {r[pk]:>6.4f}")

    if bl_raw:
        _row("Raw DINOv2 768D → UMAP(10D) → KMeans", bl_raw)
    if bl_ft:
        _row("SupCon 256D → UMAP(10D) → KMeans", bl_ft)
    if bl_semisup:
        _row("Raw DINOv2 → UMAP(10D) → SemiSup KMeans", bl_semisup)

    ss_labels = {
        "concat":          "SemiSup UMAP — Direct concat [30D]",
        "weighted_concat": "SemiSup UMAP — Weighted concat [30D]",
        "weighted_avg":    "SemiSup UMAP — Weighted average [10D]",
        "pca_fusion":      "SemiSup UMAP — PCA fusion [30D→10D]",
        "kernel_kpca":     "SemiSup UMAP — Kernel sum + KernelPCA [30D]",
    }
    for key, label in ss_labels.items():
        if key in ss_results:
            _row(label, ss_results[key])

    print("  " + "-" * 71)
    print(f"  {'Standard Ward (best NMI)':<55}  "
          f"{best_std['acc']:>6.4f}  {best_std['nmi']:>6.4f}  "
          f"{best_std['ari']:>6.4f}  {best_std['dp']:>6.4f}")
    print(f"  {'Taxonomy-Constrained Ward (best NMI)':<55}  "
          f"{best_tax['acc']:>6.4f}  {best_tax['nmi']:>6.4f}  "
          f"{best_tax['ari']:>6.4f}  {best_tax['dp']:>6.4f}")
    print(f"  {'Hard-Constrained Ward (true K, D_U eval)':<55}  "
          f"{hard_result['acc']:>6.4f}  {hard_result['nmi']:>6.4f}  "
          f"{hard_result['ari']:>6.4f}  {hard_result['cp']:>6.4f}")
    print(f"\nOutput files:")
    for f in [
        "plantnet_dendrogram_std.png   — standard Ward dendrogram",
        "plantnet_dendrogram_tax.png   — taxonomy-constrained dendrogram",
        "plantnet_genus_dendrogram.png — genus-level sanity check",
        "plantnet_gap.png              — K estimation",
        "plantnet_metrics.png          — metrics comparison std vs tax",
        "plantnet_umap.png             — UMAP with discovery tags",
    ]:
        print(f"  {f}")
