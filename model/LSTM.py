import torch
import torch.nn as nn
import torch.nn.functional as F


class WoffMan(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=100, fs=30, n_fft=64):
        super(WoffMan, self).__init__()

        self.fs = fs
        self.n_fft = n_fft   # 🔥 FIX: fixed FFT size

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (B, T, input_dim)

        # -------- LSTM -------- #
        lstm_out, _ = self.lstm(x)

        # -------- Time signal -------- #
        time_signal = self.fc(lstm_out).squeeze(-1)  # (B, T)

        # -------- Normalize -------- #
        mean = time_signal.mean(dim=1, keepdim=True)
        std = time_signal.std(dim=1, keepdim=True) + 1e-6
        time_signal = (time_signal - mean) / std

        # -------- FIXED FFT -------- #
        rppg_fft = torch.fft.rfft(time_signal, n=self.n_fft)  # 🔥 FIX
        rppg_spectrum = torch.abs(rppg_fft)

        # -------- Frequency Mask -------- #
        freqs = torch.fft.rfftfreq(self.n_fft, d=1.0 / self.fs).to(x.device)
        mask = (freqs >= 0.7) & (freqs <= 4.0)

        # Always apply mask (no condition)
        rppg_spectrum = rppg_spectrum * mask

        # -------- Normalize safely -------- #
        norm = torch.norm(rppg_spectrum, p=2, dim=-1, keepdim=True)

        # 🔥 Prevent zero division AND zero vector collapse
        rppg_spectrum = rppg_spectrum / (norm + 1e-6)

        return rppg_spectrum