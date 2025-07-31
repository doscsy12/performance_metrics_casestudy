import os
import pandas as pd
import numpy as np
import random
from scipy.linalg import sqrtm
from scipy.stats import skew, kurtosis
from dtaidistance import dtw
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d

# calculate cfid
def frechet_distance(mu1, sigma1, mu2, sigma2):
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return np.sum((mu1 - mu2) ** 2) + np.trace(sigma1 + sigma2 - 2 * covmean)

def calculate_cfid(real_embs, synth_embs):
    cfid_scores = []
    for r, s in zip(real_embs, synth_embs):
        mu1 = r
        mu2 = s
        cov1 = np.eye(len(r))
        cov2 = np.eye(len(s))
        
        dist = frechet_distance(mu1, cov1, mu2, cov2)
        cfid_scores.append(dist)
    return cfid_scores

# cfid - worst case
def compute_cfid_single_pair(real_enc, synth_enc):
    mu1 = real_enc
    mu2 = synth_enc
    sigma1 = np.cov(real_enc.T) if real_enc.ndim > 1 else np.eye(len(real_enc))
    sigma2 = np.cov(synth_enc.T) if synth_enc.ndim > 1 else np.eye(len(synth_enc))
    return frechet_distance(mu1, sigma1, mu2, sigma2)

def calculate_cfid_worst_case(data_dir, filetype="original", filefolder="time_series"):
    real_dir = os.path.join(data_dir, filetype, filefolder)
    i = random.randint(0, 99)
    real_path = os.path.join(real_dir, f"series_{i:03d}.csv")

    if not os.path.exists(real_path):
        raise FileNotFoundError(f"File {real_path} not found.")

    real_df = pd.read_csv(real_path)
    real_data = real_df.values

    min_vals = real_data.min(axis=0)
    max_vals = real_data.max(axis=0)
    noise_data = np.random.uniform(low=min_vals, high=max_vals, size=real_data.shape)

    score = compute_cfid_single_pair(real_data, noise_data)
    return score, f"series_{i:03d}.csv"

# skewness
def calculate_skewness(real_embs, synth_embs):
    # compute per pair
    skew_diffs = []

    for i, (r, s) in enumerate(zip(real_embs, synth_embs)):
        r_skew = skew(r, axis=0)
        s_skew = skew(s, axis=0)
        diff = np.linalg.norm(r_skew - s_skew)
        skew_diffs.append(diff)

    return skew_diffs

def calculate_skewness_worst_case(data_dir, filetype="original", filefolder="time_series"):
    real_dir = os.path.join(data_dir, filetype, filefolder)
    i = random.randint(0, 99)
    real_path = os.path.join(real_dir, f"series_{i:03d}.csv")
    
    real_df = pd.read_csv(real_path)
    real_data = real_df.values

    min_vals = real_data.min(axis=0)
    max_vals = real_data.max(axis=0)
    noise_data = np.random.uniform(low=min_vals, high=max_vals, size=real_data.shape)

    r_skew = skew(real_data, axis=0)
    s_skew = skew(noise_data, axis=0)
    diff = np.linalg.norm(r_skew - s_skew)

    return diff, i

# kurtosis
def calculate_kurtosis(real_embs, synth_embs):
    # compute per pair
    kurt_diffs = []

    for r, s in zip(real_embs, synth_embs):
        r_kurt = kurtosis(r, axis=0)
        s_kurt = kurtosis(s, axis=0)
        diff = np.linalg.norm(r_kurt - s_kurt)
        kurt_diffs.append(diff)

    return kurt_diffs

def calculate_kurtosis_worst_case(data_dir, filetype="original", filefolder="time_series"):
    real_dir = os.path.join(data_dir, filetype, filefolder)
    i = random.randint(0, 99)
    real_path = os.path.join(real_dir, f"series_{i:03d}.csv")

    real_df = pd.read_csv(real_path)
    real_data = real_df.values

    min_vals = real_data.min(axis=0)
    max_vals = real_data.max(axis=0)
    noise_data = np.random.uniform(low=min_vals, high=max_vals, size=real_data.shape)

    r_kurt = kurtosis(real_data, axis=0)
    s_kurt = kurtosis(noise_data, axis=0)
    diff = np.linalg.norm(r_kurt - s_kurt)

    return diff, i

# DTW
def load_multivariate_series_pairs(real_dir, synth_dir, n=100, cols=['col1','col2','col3']):
    # assume each col is equally impt 
    real_series_list = []
    synth_series_list = []

    for i in range(n):
        real_path = os.path.join(real_dir, f"series_{i:03d}.csv")
        synth_path = os.path.join(synth_dir, f"sample_{i}.csv")

        if not os.path.exists(real_path) or not os.path.exists(synth_path):
            continue

        df_real = pd.read_csv(real_path)
        df_synth = pd.read_csv(synth_path)

        if df_real.empty or df_synth.empty:
            continue

        real_ts = df_real[cols].values
        synth_ts = df_synth[cols].values

        real_series_list.append(real_ts)
        synth_series_list.append(synth_ts)

    return real_series_list, synth_series_list

def normalize_ts(ts):
    # dtw does not need encoding
    # best to normalise though
    scaler = StandardScaler()
    return scaler.fit_transform(ts)

def calculate_dtw(real_series_list, synth_series_list):
    # compute per pair
    def multivariate_dtw(ts_real, ts_synth):
        dists = []
        for d in range(ts_real.shape[1]):
            dists.append(dtw.distance(ts_real[:, d], ts_synth[:, d]))
        return np.mean(dists)

    dtw_distances = []
    for ts_real, ts_synth in zip(real_series_list, synth_series_list):
        ts_real_norm = normalize_ts(ts_real)
        ts_synth_norm = normalize_ts(ts_synth)
        dist = multivariate_dtw(ts_real_norm, ts_synth_norm)
        dtw_distances.append(dist)

    return dtw_distances

# dtw - worst case 
def multivariate_dtw(ts_real, ts_synth):
    dists = []
    for d in range(ts_real.shape[1]):
        dists.append(dtw.distance(ts_real[:, d], ts_synth[:, d]))
    return np.mean(dists)

def calculate_dtw_worst_case(data_dir, filetype="original", filefolder="time_series"):
    real_dir = os.path.join(data_dir, filetype, filefolder)
    i = random.randint(0, 99)
    real_path = os.path.join(real_dir, f"series_{i:03d}.csv")

    real_df = pd.read_csv(real_path)
    real_data = real_df.values

    min_vals = real_data.min(axis=0)
    max_vals = real_data.max(axis=0)
    noise_data = np.random.uniform(low=min_vals, high=max_vals, size=real_data.shape)

    real_norm = normalize_ts(real_data)
    noise_norm = normalize_ts(noise_data)

    worst_dtw_distance = multivariate_dtw(real_norm, noise_norm)

    return worst_dtw_distance, i

# NNDR
def resample_ts(ts, target_len):
    # 
    T, D = ts.shape
    x_old = np.linspace(0, 1, T)
    x_new = np.linspace(0, 1, target_len)
    ts_resampled = np.zeros((target_len, D))

    for d in range(D):
        interp_func = interp1d(x_old, ts[:, d], kind='linear', fill_value='extrapolate')
        ts_resampled[:, d] = interp_func(x_new)

    return ts_resampled

def calculate_nndr(real_list, synth_list, target_len=100):
    # compute per pair
    stats = []
    for i, (real_ts, synth_ts) in enumerate(zip(real_list, synth_list)):
        # resample
        real_resampled = resample_ts(real_ts, target_len)
        synth_resampled = resample_ts(synth_ts, target_len)

        scaler = StandardScaler()
        real_norm = scaler.fit_transform(real_resampled)
        synth_norm = scaler.transform(synth_resampled)

        nbrs = NearestNeighbors(n_neighbors=2).fit(real_norm)
        distances, _ = nbrs.kneighbors(synth_norm, n_neighbors=2)

        ratios = distances[:, 0] / distances[:, 1]
        mean_ratio = np.mean(ratios)
        pct5_ratio = np.percentile(ratios, 5)

        stats.append((i, mean_ratio, pct5_ratio))

    return stats

def calculate_nndr_best_case(data_dir, filetype="original", filefolder="time_series", target_len=100):
    real_dir = os.path.join(data_dir, filetype, filefolder)
    i = random.randint(0, 99)
    real_path = os.path.join(real_dir, f"series_{i:03d}.csv")

    real_df = pd.read_csv(real_path)
    real_data = real_df.values

    min_vals = real_data.min(axis=0)
    max_vals = real_data.max(axis=0)
    noise_data = np.random.uniform(low=min_vals, high=max_vals, size=real_data.shape)

    real_resampled = resample_ts(real_data, target_len)
    synth_resampled = resample_ts(noise_data, target_len)

    scaler = StandardScaler()
    real_norm = scaler.fit_transform(real_resampled)
    synth_norm = scaler.transform(synth_resampled)

    nbrs = NearestNeighbors(n_neighbors=2).fit(real_norm)
    distances, _ = nbrs.kneighbors(synth_norm, n_neighbors=2)

    ratios = distances[:, 0] / distances[:, 1]
    mean_ratio = np.mean(ratios)
    pct5_ratio = np.percentile(ratios, 5)

    return pct5_ratio, i

