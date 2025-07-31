import os
import pandas as pd

def read_static_data(data_dir, filename="static.csv"):
    filepath = os.path.join(data_dir, "original", filename)
    return pd.read_csv(filepath)

def read_real_timeseries_data(data_dir, id_val):
    # Pad with zeros to 3 digits: '0' → '000'
    filename = f"series_{int(id_val):03d}.csv"
    ts_path = os.path.join(data_dir, "original", "time_series", filename)
    return pd.read_csv(ts_path)

def read_synth_timeseries_data(data_dir, synth_type, id_val):
    # No padding: '0' → '0' for tsv1
    filename = f"sample_{int(id_val)}.csv"
    ts_path = os.path.join(data_dir, synth_type, "time_series", filename)
    return pd.read_csv(ts_path)
