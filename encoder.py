import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from data_loader import read_real_timeseries_data, read_synth_timeseries_data

class RNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1):
        super(RNNEncoder, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        _, h_n = self.rnn(x)
        return h_n[-1]  # Final hidden state

def encode_series(df, model, device='cpu'):
    model.eval()
    with torch.no_grad():
        ts_tensor = torch.tensor(df.values, dtype=torch.float32).unsqueeze(0).to(device)  # (1, T, D)
        encoding = model(ts_tensor).squeeze(0).cpu().numpy()
    return encoding

def encode_all_pairs(data_dir, synth_type, id_list, device='cpu'):
    real_encs, synth_encs, valid_ids = [], [], []

    # Infer input_dim from the first real time series
    sample_real = read_real_timeseries_data(data_dir, id_list[0])
    input_dim = sample_real.shape[1]
    model = RNNEncoder(input_dim=input_dim).to(device)

    for id_val in id_list:
        try:
            df_real = read_real_timeseries_data(data_dir, id_val)
            df_synth = read_synth_timeseries_data(data_dir, synth_type, id_val)

            if df_real.empty or df_synth.empty:
                continue

            real_enc = encode_series(df_real, model, device)
            synth_enc = encode_series(df_synth, model, device)

            real_encs.append(real_enc)
            synth_encs.append(synth_enc)
            valid_ids.append(id_val)

        except Exception as e:
            print(f"Skipping ID {id_val} due to error: {e}")
            continue

    return np.array(real_encs), np.array(synth_encs), valid_ids

