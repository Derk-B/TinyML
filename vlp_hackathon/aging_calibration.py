from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def fit_reference_channel_max(clean_df: pd.DataFrame, led_indices: np.ndarray) -> np.ndarray:
    """Per-channel peak RSS over a fresh (unaged) installation.

    Used as the online AGC's target: the running peak tracker is compared
    against this reference to estimate how much a channel has decayed.
    """
    feature_columns = [f"led_{i}" for i in led_indices.tolist()]
    return clean_df[feature_columns].to_numpy(dtype=np.float32).max(axis=0)


def run_online_aging_calibration(
    x_raw: np.ndarray,
    ref_max: np.ndarray,
    *,
    leak: float = 0.99993,
    gain_clip: tuple[float, float] = (1.0, 6.0),
    eps: float = 1e-6,
) -> np.ndarray:
    """Sequential per-channel AGC: tracks a decaying peak per channel and
    rescales each incoming sample toward the fresh-installation reference peak.

    Stateful and strictly sequential (each sample updates the running peak
    used for the next one) so this must be applied in stream order, matching
    how the Pico firmware processes samples one request at a time.
    """
    ema_peak = ref_max.copy().astype(np.float32)
    corrected = np.empty_like(x_raw)
    for i in range(len(x_raw)):
        row = x_raw[i]
        ema_peak = np.maximum(np.abs(row), ema_peak * leak)
        gain = np.clip(ref_max / np.maximum(ema_peak, eps), gain_clip[0], gain_clip[1])
        corrected[i] = row * gain
    return corrected


def write_aging_header(
    ref_max: np.ndarray,
    leak: float,
    gain_clip: tuple[float, float],
    header_path: Path,
) -> None:
    def c_float(value: float) -> str:
        text = f"{float(value):.9g}"
        if "." not in text and "e" not in text.lower():
            text += ".0"
        return text + "f"

    values = ", ".join(c_float(v) for v in ref_max)
    header_path.write_text(
        "#pragma once\n\n"
        f"constexpr int kAgingChannels = {len(ref_max)};\n"
        f"constexpr float kRefChannelMax[{len(ref_max)}] = {{{values}}};\n"
        f"constexpr float kAgingLeak = {c_float(leak)};\n"
        f"constexpr float kAgingGainMin = {c_float(gain_clip[0])};\n"
        f"constexpr float kAgingGainMax = {c_float(gain_clip[1])};\n",
        encoding="utf-8",
    )
