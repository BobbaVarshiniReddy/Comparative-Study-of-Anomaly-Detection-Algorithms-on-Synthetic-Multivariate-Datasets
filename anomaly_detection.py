"""
Comparative Study of Anomaly Detection Algorithms

Algorithms:
- KNN
- KDE
- Gaussian Mixture Models

Experiments:
- Monte Carlo Simulation
- Cauchy Perturbation Analysis

Evaluation Metric:
- ROC-AUC
"""
# ============================================================
# ANOMALY DETECTION | KNN, KDE, GMM | 4D Datasets
# Monte Carlo + Cauchy Swap Simulation
# ============================================================

import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ============================================================
# DATASETS
# ============================================================

def make_enclosed(cauchy_cluster=None):
    means = [[-4,4,-2,2],[4,4,2,2],[-4,-4,-2,-2],[4,-4,2,-2]]
    clusters = []
    for i, m in enumerate(means):
        if i == cauchy_cluster:
            c = np.clip(np.random.standard_cauchy((200, 4)), -20, 20) + m
        else:
            c = np.random.multivariate_normal(m, np.eye(4), 200)
        clusters.append(c)
    X_normal  = np.vstack(clusters)
    X_anomaly = np.array([[0,0,0,0],[0.2,0.1,0.1,0.2],[-0.2,0.3,-0.1,0.1],
                           [0.3,-0.2,0.2,-0.1],[-0.3,-0.2,-0.2,-0.3]])
    X = np.vstack([X_normal, X_anomaly])
    y = np.hstack([np.zeros(len(X_normal)), np.ones(len(X_anomaly))])
    return X, y

def make_global(cauchy_cluster=None):
    means = [[1,1,1,1],[7,7,7,7]]
    covs  = [np.eye(4), np.eye(4)*0.08]
    sizes = [300, 100]
    clusters = []
    for i in range(2):
        if i == cauchy_cluster:
            c = np.clip(np.random.standard_cauchy((sizes[i], 4)), -20, 20) + means[i]
        else:
            c = np.random.multivariate_normal(means[i], covs[i], sizes[i])
        clusters.append(c)
    X_normal  = np.vstack(clusters)
    X_anomaly = np.array([[8,1,5,9],[8.5,0.5,5.5,8.5],[7.8,1.2,4.8,9.2],[8.2,0.7,5.2,8.8]])
    X = np.vstack([X_normal, X_anomaly])
    y = np.hstack([np.zeros(len(X_normal)), np.ones(len(X_anomaly))])
    return X, y

# ============================================================
# ANOMALY SCORE FUNCTIONS
# ============================================================

def knn_scores(X, k=6):
    model = NearestNeighbors(n_neighbors=k)

    model.fit(X)

    distances, indices = model.kneighbors(X)

    scores = distances[:, 1:].mean(axis=1)

    return scores

def kde_scores(X, bandwidth=1.5):
    model = KernelDensity(
        kernel='gaussian',
        bandwidth=bandwidth
    )

    model.fit(X)

    log_density = model.score_samples(X)

    scores = -log_density

    return scores

def gmm_scores(X, n_components):

    model = GaussianMixture(
        n_components=n_components,
        random_state=42
    )

    model.fit(X)
    log_probability = model.score_samples(X)
    scores = -log_probability

    return scores

def run_all(X, y, n_components):
    return (roc_auc_score(y, knn_scores(X,k=6)),
            roc_auc_score(y, kde_scores(X, bandwidth=1.5)),
            roc_auc_score(y, gmm_scores(X, n_components)))

# ============================================================
# SUMMARY TABLE  |  Efficiency = Mean AUC / Variance
# ============================================================

def make_summary(knn_list, kde_list, gmm_list):
    rows = []
    for name, lst in [("KNN", knn_list), ("KDE", kde_list), ("GMM", gmm_list)]:
        m, v = np.mean(lst), np.var(lst)
        rows.append({"Algorithm": name, "Mean_AUC": round(m,4),
                     "Variance": round(v,6),
                     "Efficiency(M/V)": round(m/v,2) if v > 1e-9 else "inf"})
    return pd.DataFrame(rows)

# ============================================================
# MONTE CARLO SIMULATION
# ============================================================

def monte_carlo(dataset_fn, n_components, runs=10):
    knn_list, kde_list, gmm_list = [], [], []
    for _ in range(runs):
        X, y = dataset_fn()
        k, d, g = run_all(X, y, n_components)
        knn_list.append(k); kde_list.append(d); gmm_list.append(g)
    return make_summary(knn_list, kde_list, gmm_list)

# ============================================================
# CAUCHY SWAP SIMULATION
# ============================================================

def cauchy_swap(dataset_fn, n_clusters, n_components):
    knn_list, kde_list, gmm_list = [], [], []
    detail_rows = []
    for i in range(n_clusters):
        X, y = dataset_fn(cauchy_cluster=i)
        k, d, g = run_all(X, y, n_components)
        knn_list.append(k); kde_list.append(d); gmm_list.append(g)
        detail_rows.append({"Round": i+1, "Cauchy_Cluster": i+1,
                             "KNN_AUC": round(k,4), "KDE_AUC": round(d,4), "GMM_AUC": round(g,4)})
    return pd.DataFrame(detail_rows), make_summary(knn_list, kde_list, gmm_list)

# ============================================================
# MAIN — each section displayed as its own table
# ============================================================

print("ENCLOSED DATASET — MONTE CARLO (100 runs)")
enc_mc = monte_carlo(make_enclosed, n_components=4, runs=100)
display(enc_mc)

print("ENCLOSED DATASET — CAUCHY SWAP (4 rounds) — Per Round")
enc_detail, enc_cs = cauchy_swap(make_enclosed, n_clusters=4, n_components=4)
display(enc_detail)

print("ENCLOSED DATASET — CAUCHY SWAP — Summary")
display(enc_cs)

print("GLOBAL DATASET — MONTE CARLO (100 runs)")
glob_mc = monte_carlo(make_global, n_components=2, runs=100)
display(glob_mc)

print("GLOBAL DATASET — CAUCHY SWAP (2 rounds) — Per Round")
glob_detail, glob_cs = cauchy_swap(make_global, n_clusters=2, n_components=2)
display(glob_detail)

print("GLOBAL DATASET — CAUCHY SWAP — Summary")
display(glob_cs)

enc_mc.to_csv("enclosed_montecarlo.csv",  index=False)
enc_cs.to_csv("enclosed_cauchyswap.csv",  index=False)
glob_mc.to_csv("global_montecarlo.csv",   index=False)
glob_cs.to_csv("global_cauchyswap.csv",   index=False)

print("Done. Results saved to CSV.")
