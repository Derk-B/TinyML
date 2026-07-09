#!/usr/bin/env python3
"""Export the trained Task 3 PyTorch MLP to TFLite and Pico assets."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
import litert_torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlp_hackathon.baseline_model import BaselineMLP
from vlp_hackathon.export import tflite_to_c_array


def c_float(value: float) -> str:
    text = f"{float(value):.9g}"
    if "." not in text and "e" not in text.lower():
        text += ".0"
    return text + "f"


def c_float_array(name: str, values: np.ndarray) -> str:
    body = ", ".join(c_float(v) for v in values.tolist())
    return f"constexpr float {name}[36] = {{ {body} }};\n"


def main() -> None:
    state_path = ROOT / "models" / "task3_model.pt"
    scaling_path = ROOT / "models" / "task3_scaling.npz"
    out_path = ROOT / "models" / "task3.tflite"

    firmware_dir = ROOT / "firmware" / "vlp_serial"
    firmware_path = firmware_dir / "vlp_model.tflite"

    model = BaselineMLP(36)
    model.load_state_dict(torch.load(state_path, map_location="cpu"))
    model.eval()

    example = torch.zeros(1, 36, dtype=torch.float32)

    edge_model = litert_torch.convert(
        model,
        sample_args=(example,),
        strict_export=True,
    )

    tflite_bytes = edge_model.model_content()

    out_path.write_bytes(tflite_bytes)
    firmware_path.write_bytes(tflite_bytes)

    tflite_to_c_array(
        firmware_path,
        firmware_dir / "model_data.cc",
        firmware_dir / "model_data.h",
    )

    scaling = np.load(scaling_path)

    rss_mean = np.asarray(scaling["rss_mean"], dtype=np.float32)
    rss_std = np.asarray(scaling["rss_std"], dtype=np.float32)

    target_min_cm = np.asarray(scaling["target_min_cm"], dtype=np.float32)
    target_range_cm = np.asarray(scaling["target_range_cm"], dtype=np.float32)

    (firmware_dir / "preprocess_data.h").write_text(
        "#pragma once\n\n"
        "constexpr int kInputFeatures = 36;\n\n"
        + c_float_array("kRssMean", rss_mean)
        + c_float_array("kRssStd", rss_std),
        encoding="utf-8",
    )

    (firmware_dir / "target_scale_data.h").write_text(
        "#pragma once\n\n"
        f"constexpr float kTargetXMinCm = {c_float(target_min_cm[0])};\n"
        f"constexpr float kTargetYMinCm = {c_float(target_min_cm[1])};\n"
        f"constexpr float kTargetXRangeCm = {c_float(target_range_cm[0])};\n"
        f"constexpr float kTargetYRangeCm = {c_float(target_range_cm[1])};\n",
        encoding="utf-8",
    )

    print(f"Wrote {out_path} ({len(tflite_bytes)} bytes)")
    print(f"Wrote {firmware_path}")
    print("Updated model_data.cc/.h and preprocessing/target scaling headers")


if __name__ == "__main__":
    main()
