import config
from data_loader import read_static_data, read_real_timeseries_data, read_synth_timeseries_data
from encoder import encode_all_pairs
from analysis import calculate_cfid, calculate_skewness, calculate_kurtosis, calculate_dtw, calculate_nndr
from analysis import load_multivariate_series_pairs
from analysis import calculate_cfid_worst_case, calculate_skewness_worst_case, calculate_kurtosis_worst_case
from analysis import calculate_dtw_worst_case, calculate_nndr_best_case
from report_generator import generate_report

def main():
    static_df = read_static_data(config.data_dir)
    first_id = int(static_df['.id'].iloc[0])  # example use
    real_df = read_real_timeseries_data(config.data_dir, first_id)
    synth_df = read_synth_timeseries_data(config.data_dir, config.synth_type, first_id)

    id_list = [int(i) for i in static_df['.id'].tolist()[:100]]  
    real_embs, synth_embs, valid_ids = encode_all_pairs(config.data_dir, config.synth_type, id_list)
    
    # metrics
    cfid_scores = calculate_cfid(real_embs, synth_embs)
    skew_vals = calculate_skewness(real_embs, synth_embs)
    kurt_vals = calculate_kurtosis(real_embs, synth_embs)

    mean_cfid_score = sum(cfid_scores) / len(cfid_scores)
    mean_skew_diff = sum(skew_vals) / len(skew_vals)
    mean_kurt_diff = sum(kurt_vals) / len(kurt_vals)

    real_dir = f"{config.data_dir}/original/time_series"
    synth_dir = f"{config.data_dir}/{config.synth_type}/time_series"
    cols = ["col1", "col2", "col3"]  

    real_series_list, synth_series_list = load_multivariate_series_pairs(
        real_dir, synth_dir, n=100, cols=cols
    )
    dtw_scores = calculate_dtw(real_series_list, synth_series_list)
    mean_dtw_score = sum(dtw_scores) / len(dtw_scores)

    nndr_stats = calculate_nndr(real_series_list, synth_series_list)
    mean_nndr_score = sum([r[2] for r in nndr_stats]) / len(nndr_stats)
    
    scores = {
        'C-FID': mean_cfid_score,
        'Skewness Difference (SD)': mean_skew_diff,
        'Kurtosis Difference (KD)': mean_kurt_diff,
        'DTW': mean_dtw_score,
        'NNDR (5th pct)': mean_nndr_score,
    }
    
    # worst/best case scenario
    # used to normalise scores
    worst_cfid_score, example_file = calculate_cfid_worst_case(config.data_dir)
    print(f"Worst-case C-FID (real vs random noise in {example_file}): {worst_cfid_score:.4f}")

    worst_skew_diff, idx = calculate_skewness_worst_case(config.data_dir)
    print(f"Skewness Difference between real series_{idx:03d}.csv and column-wise noise: {worst_skew_diff:.4f}")

    worst_kurt_diff, idx_kurt = calculate_kurtosis_worst_case(config.data_dir)
    print(f"Kurtosis Difference between real series_{idx_kurt:03d}.csv and column-wise noise: {worst_kurt_diff:.6f}")

    worst_dtw_distance, idx_dtw = calculate_dtw_worst_case(config.data_dir)
    print(f"DTW distance between real series_{idx_dtw:03d}.csv and noise: {worst_dtw_distance:.6f}")

    best_nndr_5p, idx_nndr = calculate_nndr_best_case(config.data_dir)
    print(f"NNDR (5th percentile) between real series_{idx_nndr:03d}.csv and noise: {best_nndr_5p:.6f}")

    worst_case = {
    'C-FID': worst_cfid_score,
    'Skewness Difference (SD)': worst_skew_diff,
    'Kurtosis Difference (KD)': worst_kurt_diff,
    'DTW': worst_dtw_distance,
    'NNDR (5th pct)': 0.0, 
    }

    best_case = {
        'NNDR (5th pct)': best_nndr_5p,
    }

    generate_report(scores, worst_case, best_case)

if __name__ == "__main__":
    main()
