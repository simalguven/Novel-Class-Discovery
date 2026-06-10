"""
Semi-supervised K-Means with pseudo-labels from high-confidence regions of Dataset A
=====================================================================================

Idea
────
1. Combined UMAP on A+B
2. K-Means(50) on A's portion  →  initial cluster structure
3. Score every A sample by confidence (two methods):
     Density     : distance to own centroid (close = core = certain)
     Uncertainty : soft-assignment margin  (max_prob - 2nd_max_prob)
4. Label only the top-N% most confident samples per cluster
5. Semi-supervised K-Means(60) on all of A+B:
     · Labeled A samples are pinned to their pseudo-class centroid
     · All other A samples + all B samples are free
6. Identify novel clusters  →  those with no (or few) pinned A samples
7. Evaluate + compare vs plain K-Means(60) Hungarian match

The pinned samples act as "species anchors" — they force old-species
centroids to stay in the right place, leaving the remaining 10 centroids
free to find the novel species in B.
"""

import numpy as np, umap, warnings, time
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score
from scipy.optimize import linear_sum_assignment
warnings.filterwarnings("ignore")

SEED=42; K_OLD=50; K_NEW=60; K_NOV=10; N_PER_CLS=100
rng=np.random.default_rng(SEED)

# ── Data ──────────────────────────────────────────────────────────────────────
print("Loading data …")
embeddings=np.load("plantnet_dinov2_raw.npy").astype(np.float32)
labels=np.load("plantnet_labels.npy")
all_classes=np.unique(labels)
counts=np.array([(labels==c).sum() for c in all_classes])
eligible=all_classes[counts>=2*N_PER_CLS]
chosen_60=np.sort(rng.choice(eligible,size=K_NEW,replace=False))
base_cls=chosen_60[:K_OLD]; novel_cls=chosen_60[K_OLD:]

XA,XB,yBl=[],[],[]
for c in base_cls:
    idx=rng.choice(np.where(labels==c)[0],size=2*N_PER_CLS,replace=False)
    XA.append(embeddings[idx[:N_PER_CLS]]); XB.append(embeddings[idx[N_PER_CLS:]]); yBl.extend([c]*N_PER_CLS)
for c in novel_cls:
    idx=rng.choice(np.where(labels==c)[0],size=N_PER_CLS,replace=False)
    XB.append(embeddings[idx]); yBl.extend([c]*N_PER_CLS)

X_A=np.vstack(XA); X_B=np.vstack(XB); y_B=np.array(yBl)
id2idx={c:i for i,c in enumerate(chosen_60)}
y_B_eval=np.array([id2idx[c] for c in y_B])
is_novel_B=y_B_eval>=K_OLD
N_A,N_B=len(X_A),len(X_B)
print(f"  A:{N_A}  B:{N_B}  novel in B:{is_novel_B.sum()}")

# ── Combined UMAP on A+B ──────────────────────────────────────────────────────
print("\nCombined UMAP on A+B (768-d → 10-d) …")
t0=time.time()
reducer=umap.UMAP(n_components=10,n_neighbors=20,min_dist=0.05,
                  metric="cosine",random_state=SEED,verbose=False)
X_AB=normalize(reducer.fit_transform(np.vstack([X_A,X_B])),norm="l2").astype(np.float32)
X_A2=X_AB[:N_A]; X_B2=X_AB[N_A:]
print(f"  Done in {time.time()-t0:.1f}s")

import torch  # after UMAP

# ── Evaluation helpers ────────────────────────────────────────────────────────
def evaluate(novel_mask, novelty_score, tag):
    tp   = is_novel_B[novel_mask].sum()
    prec = tp/novel_mask.sum() if novel_mask.sum()>0 else 0
    rec  = tp/is_novel_B.sum()
    f1   = 2*prec*rec/(prec+rec+1e-9)
    auc  = roc_auc_score(is_novel_B, novelty_score)
    fpr  = (~is_novel_B[novel_mask]).sum()/(~is_novel_B).sum()
    return dict(auc=auc,rec=rec,prec=prec,f1=f1,fpr=fpr,flagged=int(novel_mask.sum()))

def cluster_acc_novel(B_labels, novel_ids):
    novel_mask = np.isin(B_labels, list(novel_ids))
    truly = is_novel_B[novel_mask]
    y_true = y_B_eval[novel_mask][truly]-K_OLD
    y_pred = B_labels[novel_mask][truly]
    if len(y_true)<K_NOV: return 0.0, int(truly.sum())
    uniq={v:i for i,v in enumerate(np.unique(y_pred))}
    yp=np.array([uniq[v] for v in y_pred])
    n=max(y_true.max(),yp.max())+1
    mat=np.zeros((n,n),dtype=np.int64)
    for t,p in zip(y_true,yp): mat[t,p]+=1
    r,c=linear_sum_assignment(-mat)
    return mat[r,c].sum()/len(y_true), int(truly.sum())

def print_result(tag, res, cacc, n_novel):
    print(f"\n  {tag}")
    print(f"    AUC={res['auc']:.4f}  Recall={res['rec']:.1%}  "
          f"Precision={res['prec']:.1%}  F1={res['f1']:.3f}  "
          f"FPR={res['fpr']:.1%}  Flagged={res['flagged']}")
    print(f"    Cluster ACC={cacc:.1%}  ({n_novel} novel samples in flagged set)")

# ── Step 1: K-Means(50) on A → cluster structure ──────────────────────────────
print("\nK-Means(50) on A …")
km50=KMeans(n_clusters=K_OLD,n_init=15,random_state=SEED).fit(X_A2)
A_labels=km50.labels_          # pseudo-label per A sample (0-49)
cent50=km50.cluster_centers_   # (50, 10)

# Soft assignment probabilities (temperature-scaled softmax over distances)
dist_A=np.linalg.norm(X_A2[:,None,:]-cent50[None,:,:],axis=-1)  # (N_A,50)
T=0.1
soft_A=np.exp(-dist_A/T); soft_A/=soft_A.sum(1,keepdims=True)   # (N_A,50)

# ── Step 2: Confidence scoring ─────────────────────────────────────────────────
# Method 1 — Density: distance to own centroid (smaller = more certain)
own_dist=dist_A[np.arange(N_A), A_labels]                        # (N_A,)

# Method 2 — Uncertainty margin: max_prob - 2nd_max_prob (larger = more certain)
sorted_soft=np.sort(soft_A,axis=1)
margin=sorted_soft[:,-1]-sorted_soft[:,-2]                        # (N_A,)

def select_confident_labels(scores, ascending, top_fraction):
    """
    Within each pseudo-class cluster, select the top-fraction most confident
    samples. ascending=True means lower score = more confident (density).
    Returns boolean mask over A samples.
    """
    confident=np.zeros(N_A,dtype=bool)
    n_per=max(1,int(N_PER_CLS*top_fraction))   # samples to label per cluster
    for k in range(K_OLD):
        idx_k=np.where(A_labels==k)[0]
        sc=scores[idx_k]
        order=np.argsort(sc) if ascending else np.argsort(-sc)
        confident[idx_k[order[:n_per]]]=True
    return confident

# ── Step 3: Semi-supervised K-Means ──────────────────────────────────────────
def kmeans_pp_init(X_all):
    """Standard K-Means++ initialisation on all samples — shared by both methods."""
    km = KMeans(n_clusters=K_NEW, init="k-means++", n_init=1, random_state=SEED,
                max_iter=1)   # one pass just to get init centroids
    km.fit(X_all)
    return km.cluster_centers_.copy()

def seeded_kmeans(X_all, pinned_idx, pinned_labels, max_iter=150):
    """Hard must-link with K-Means++ init shared with baseline."""
    centroids = kmeans_pp_init(X_all)
    assign = np.zeros(len(X_all), dtype=np.int32)
    for _ in range(max_iter):
        dists = np.linalg.norm(X_all[:,None,:] - centroids[None,:,:], axis=-1)
        assign = dists.argmin(1)
        assign[pinned_idx] = pinned_labels          # hard constraint
        new_cent = np.array([
            X_all[assign==k].mean(0) if (assign==k).any() else centroids[k]
            for k in range(K_NEW)])
        if np.linalg.norm(new_cent - centroids) < 1e-5: break
        centroids = new_cent
    return assign, centroids

def soft_seeded_kmeans(X_all, pinned_idx, pinned_labels, confidence, lam, max_iter=150):
    """
    Soft must-link with K-Means++ init shared with baseline.
    Same starting centroids — only the per-iteration penalty differs.

      cost(sample i → cluster k) = dist(i, centroid_k)
                                   + λ × confidence(i) × scale   if k ≠ pseudo_class(i)
                                   + 0                            if k == pseudo_class(i)

    scale = mean nearest-centroid distance of A samples (unit-normalises λ).
    λ=0   → identical to baseline plain K-Means
    λ→∞   → hard must-link
    """
    N = len(X_all)
    centroids = kmeans_pp_init(X_all)              # same init as baseline

    # Scale penalty to the typical intra-cluster distance
    dists_A = np.linalg.norm(X_A2[:,None,:] - cent50[None,:,:], axis=-1)
    scale   = dists_A.min(1).mean()          # mean nearest-centroid distance in A

    # Build per-sample penalty lookup: penalty_row[i] = (pseudo_label, confidence*scale*λ)
    penalty_lookup = {}                       # idx → (pseudo_label, penalty_magnitude)
    for i, (pi, pl, cf) in enumerate(zip(pinned_idx, pinned_labels, confidence)):
        penalty_lookup[int(pi)] = (int(pl), float(cf) * scale * lam)

    assign = np.zeros(N, dtype=np.int32)

    for _ in range(max_iter):
        dists = np.linalg.norm(X_all[:,None,:] - centroids[None,:,:], axis=-1).copy()

        # Add penalty row by row only for pinned samples
        for pi, (pl, pen) in penalty_lookup.items():
            dists[pi, :] += pen          # penalise all clusters …
            dists[pi, pl] -= pen         # … except the pseudo-class (net zero)

        assign = dists.argmin(1)

        new_cent = np.array([
            X_all[assign==k].mean(0) if (assign==k).any() else centroids[k]
            for k in range(K_NEW)])
        if np.linalg.norm(new_cent - centroids) < 1e-5: break
        centroids = new_cent

    return assign, centroids

def identify_novel_clusters_hungarian(assign_all, centroids60):
    """
    Same identification method as baseline:
    Hungarian match between 60 combined centroids and 50 A-only centroids.
    10 unmatched combined centroids = novel.
    Novelty score = match cost of each B sample's cluster centroid.
    """
    cost = np.linalg.norm(
        centroids60[:,None,:] - cent50[None,:,:], axis=-1)   # (60, 50)
    INF  = cost.max() * 10
    cost_sq = np.full((K_NEW, K_NEW), INF)
    cost_sq[:K_OLD, :] = cost.T                              # (50, 60) in square
    _, col = linear_sum_assignment(cost_sq)
    matched    = set(col[:K_OLD])
    novel_ids  = set(range(K_NEW)) - matched

    # Novelty score: match cost of each combined cluster to its best A-centroid
    match_cost = np.full(K_NEW, cost.max() * 2)
    for ai, ci in zip(range(K_OLD), col[:K_OLD]):
        match_cost[ci] = cost_sq[ai, ci]

    assign_B      = assign_all[N_A:]
    novelty_score = match_cost[assign_B]
    novel_mask    = np.isin(assign_B, list(novel_ids))
    return novel_ids, novel_mask, novelty_score

# ── Baseline: plain K-Means(60) + Hungarian (no pseudo-labels) ───────────────
print("\n" + "="*70)
print("BASELINE — K-Means(60) on A+B, Hungarian match to A centroids")
print("="*70)
km60_base  = KMeans(n_clusters=K_NEW, n_init=15, random_state=SEED).fit(X_AB)
assign_base= np.concatenate([km60_base.labels_[:N_A], km60_base.labels_[N_A:]])
novel_ids_base, novel_mask_base, novelty_score_base = \
    identify_novel_clusters_hungarian(km60_base.labels_, km60_base.cluster_centers_)
assign_B_base = km60_base.labels_[N_A:]
res_base  = evaluate(novel_mask_base, novelty_score_base, "Baseline")
cacc_base, nn_base = cluster_acc_novel(assign_B_base, novel_ids_base)
print_result("Baseline K-Means(60) Hungarian", res_base, cacc_base, nn_base)

results = {"Baseline": (*res_base.values(), cacc_base)}

# ── Experiment: soft must-link on ALL A samples, confidence = penalty weight ──
# All A samples participate — confident ones have a strong pull toward their
# pseudo-class, uncertain boundary samples have a weak pull and can cross freely.
# This is the correct use of soft must-link.

for conf_method, scores, ascending, mname in [
    ("Density",     own_dist, True,  "density"),
    ("Uncertainty", margin,   False, "margin"),
]:
    print(f"\n{'='*70}")
    print(f"SOFT MUST-LINK — ALL A samples, penalty weighted by {conf_method}")
    print(f"{'='*70}")

    # Use ALL A samples — confidence determines penalty strength, not selection
    pinned_idx    = np.arange(N_A, dtype=np.int32)
    pinned_labels = A_labels.astype(np.int32)

    # Normalise confidence scores to [0,1]
    s_min, s_max = scores.min(), scores.max()
    if ascending:   # density: smaller distance = more confident → invert
        conf_norm = 1 - (scores - s_min) / (s_max - s_min + 1e-9)
    else:
        conf_norm = (scores - s_min) / (s_max - s_min + 1e-9)

    print(f"  All {N_A} A samples used  |  "
          f"mean confidence={conf_norm.mean():.3f}  "
          f"std={conf_norm.std():.3f}")
    print(f"  (boundary samples near 0 → almost free to move)")
    print(f"  (core samples near 1 → strongly pulled to pseudo-class)")

    # Hard must-link on ALL A samples (upper bound of constraint)
    assign_hard, cent_hard = seeded_kmeans(X_AB, pinned_idx, pinned_labels)
    novel_ids_h, novel_mask_h, score_h = identify_novel_clusters_hungarian(assign_hard, cent_hard)
    res_h  = evaluate(novel_mask_h, score_h, f"Hard all-A")
    cacc_h, nn_h = cluster_acc_novel(assign_hard[N_A:], novel_ids_h)
    print_result("  Hard (all A, no flexibility)", res_h, cacc_h, nn_h)
    results[f"Hard all-A {conf_method}"] = (*res_h.values(), cacc_h)

    # Soft must-link: λ controls overall constraint strength
    # At λ=0 → plain K-Means; as λ→∞ → hard must-link
    for lam in [0.1, 0.5, 1.0, 2.0, 5.0]:
        assign_soft, cent_soft = soft_seeded_kmeans(
            X_AB, pinned_idx, pinned_labels, conf_norm, lam=lam)
        novel_ids_s, novel_mask_s, score_s = identify_novel_clusters_hungarian(
            assign_soft, cent_soft)
        res_s  = evaluate(novel_mask_s, score_s, f"Soft λ={lam}")
        cacc_s, nn_s = cluster_acc_novel(assign_soft[N_A:], novel_ids_s)
        print_result(f"  Soft λ={lam:<5}", res_s, cacc_s, nn_s)
        results[f"Soft {conf_method} λ={lam}"] = (*res_s.values(), cacc_s)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n"+"="*80)
print("SUMMARY")
print("="*80)
print(f"  {'Method':<30} {'AUC':>6} {'Recall':>8} {'Prec':>8} "
      f"{'F1':>6} {'FPR':>6} {'ClustACC':>10}")
print("  "+"-"*78)
for tag,(auc,rec,prec,f1,fpr,flagged,cacc) in results.items():
    marker=" ◄" if cacc==max(v[-1] for v in results.values()) else ""
    print(f"  {tag:<30} {auc:>6.4f} {rec:>8.1%} {prec:>8.1%} "
          f"{f1:>6.3f} {fpr:>6.1%} {cacc:>10.1%}{marker}")
