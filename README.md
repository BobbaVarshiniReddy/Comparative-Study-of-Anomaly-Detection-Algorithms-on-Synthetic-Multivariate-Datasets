# Comparative-Study-of-Anomaly-Detection-Algorithms-on-Synthetic-Multivariate-Datasets

## Overview

This project presents a comparative study of three unsupervised anomaly detection techniques:

- K-Nearest Neighbors (KNN)
- Kernel Density Estimation (KDE)
- Gaussian Mixture Models (GMM)

The algorithms are evaluated on synthetic 4-dimensional datasets generated under different statistical conditions to analyze their robustness and anomaly detection performance.

## Objectives

- Compare the effectiveness of KNN, KDE, and GMM for anomaly detection.
- Evaluate performance using ROC-AUC.
- Study model robustness through Monte Carlo simulations.
- Analyze the impact of heavy-tailed distributions using Cauchy perturbations.

## Dataset Generation

Two synthetic dataset configurations were created:

### 1. Enclosed Dataset
- Four Gaussian clusters in 4D space.
- Anomalies placed near the center region between clusters.

### 2. Global Dataset
- Two Gaussian clusters with different variances.
- Anomalies placed away from normal cluster distributions.

### Cauchy Perturbation Experiment
Selected clusters are replaced with samples drawn from a Cauchy distribution to simulate heavy-tailed and noisy data conditions.

## Methods

### KNN-Based Detection
Anomaly score is computed as the average distance to the nearest neighbors.

### KDE-Based Detection
Kernel Density Estimation is used to estimate data density.
Lower density points receive higher anomaly scores.

### GMM-Based Detection
Gaussian Mixture Models estimate the probability distribution of normal data.
Points with low likelihood are assigned higher anomaly scores.

## Evaluation Metric

The primary evaluation metric is:

- ROC-AUC (Receiver Operating Characteristic - Area Under Curve)

## Experiments

### Monte Carlo Simulation
- 100 independent runs
- Randomly generated datasets in each run
- Mean ROC-AUC and variance recorded

### Cauchy Swap Simulation
- One cluster replaced with Cauchy-distributed samples
- Performance evaluated under distributional shifts

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn

## Output

The program generates:

- ROC-AUC comparison tables
- Performance summaries
- CSV result files for further analysis

Generated CSV files:
- enclosed_montecarlo.csv
- enclosed_cauchyswap.csv
- global_montecarlo.csv
- global_cauchyswap.csv

## Future Improvements

- Isolation Forest comparison
- One-Class SVM implementation
- Higher-dimensional datasets
- Real-world anomaly detection datasets
- Visualization of anomaly score distributions

## Author

Bobba Varshini Reddy  
B.S. (Honours) in Mathematics and Computing  
Indian Institute of Technology Kharagpur
