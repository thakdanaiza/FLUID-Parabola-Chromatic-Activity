"""Reproduce V06 chromatic activity and short/long optical-flow metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import yaml

CONDA_DLL_HANDLE = None
if os.name == "nt":
    conda_dll_dir = Path(sys.prefix) / "Library" / "bin"
    if conda_dll_dir.is_dir():
        CONDA_DLL_HANDLE = os.add_dll_directory(str(conda_dll_dir))
        os.environ["PATH"] = str(conda_dll_dir) + os.pathsep + os.environ.get("PATH", "")

try:
    import cupy as cp
except Exception as exc:  # pragma: no cover - depends on the host CUDA setup
    cp = None
    CUPY_IMPORT_ERROR = exc
else:
    CUPY_IMPORT_ERROR = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--profile", action="append", help="Profile to run; may be repeated.")
    return parser.parse_args()


def resolve_path(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def require_cupy(gpu_id: int) -> None:
    if cp is None:
        raise RuntimeError(
            "The V06 flight configuration requires CuPy/CUDA. "
            f"CuPy import failed: {CUPY_IMPORT_ERROR!r}"
        )
    cp.cuda.Device(gpu_id).use()
    test = cp.zeros((2, 2), dtype=cp.float32)
    cp.cuda.Stream.null.synchronize()
    cp.asnumpy(test)


def bgr_to_lab_ab_gpu(frame_bgr: np.ndarray):
    bgr = cp.asarray(frame_bgr, dtype=cp.float32) / cp.float32(255.0)
    red, green, blue = bgr[:, :, 2], bgr[:, :, 1], bgr[:, :, 0]

    def linear(channel):
        return cp.where(
            channel <= 0.04045,
            channel / 12.92,
            cp.power((channel + 0.055) / 1.055, 2.4),
        )

    red, green, blue = linear(red), linear(green), linear(blue)
    x = (red * 0.4124564 + green * 0.3575761 + blue * 0.1804375) / 0.95047
    y = red * 0.2126729 + green * 0.7151522 + blue * 0.0721750
    z = (red * 0.0193339 + green * 0.1191920 + blue * 0.9503041) / 1.08883
    epsilon = cp.float32(216.0 / 24389.0)
    kappa = cp.float32(24389.0 / 27.0)

    def f(value):
        return cp.where(value > epsilon, cp.cbrt(value), (kappa * value + 16.0) / 116.0)

    fx, fy, fz = f(x), f(y), f(z)
    return cp.stack((500.0 * (fx - fy), 200.0 * (fy - fz)), axis=2).astype(cp.float32)


def build_activity_stack(
    video: Path,
    selection: dict[str, Any],
    activity: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict[str, Any]]:
    if activity["backend"] != "cupy":
        raise ValueError("Only the retained V06 flight backend 'cupy' is supported.")
    gpu_id = int(activity["gpu_id"])
    require_cupy(gpu_id)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start, end = int(selection["start_frame"]), int(selection["end_frame"])
    x, y, roi_width, roi_height = map(int, selection["roi_xywh"])
    if start < 0 or end >= frame_count or end < start:
        raise ValueError(f"Frame selection {start}..{end} is outside a {frame_count}-frame video.")
    if x < 0 or y < 0 or x + roi_width > width or y + roi_height > height:
        raise ValueError("Configured ROI is outside the input video.")
    expected_fps = float(selection["expected_fps"])
    if not np.isclose(fps, expected_fps, rtol=0, atol=0.02):
        raise ValueError(f"Input FPS {fps:.9f} does not match V06 expected FPS {expected_fps:.9f}.")

    window = int(activity["source_window_frames"])
    stride = int(activity["stride_frames"])
    lag = int(activity["temporal_lag_frames"])
    if activity["reducer"] != "mean" or lag >= window:
        raise ValueError("V06 requires a mean reducer and temporal lag smaller than the window.")
    starts = np.arange(start, end - window + 2, stride, dtype=np.int64)
    stack = np.empty((roi_height, roi_width, starts.size), dtype=np.float32)
    for map_index, block_start in enumerate(starts):
        labs = []
        for frame_index in range(int(block_start), int(block_start) + window):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Could not decode source frame {frame_index}.")
            labs.append(bgr_to_lab_ab_gpu(frame[y : y + roi_height, x : x + roi_width]))
        pairs = []
        for pair_index in range(window - lag):
            difference = labs[pair_index + lag] - labs[pair_index]
            pairs.append(cp.sqrt(cp.sum(difference * difference, axis=2)).astype(cp.float32))
        stack[:, :, map_index] = cp.asnumpy(cp.mean(cp.stack(pairs, axis=0), axis=0))
        print(f"activity {map_index + 1}/{starts.size}", flush=True)
    cap.release()
    times = (starts.astype(np.float64) + (window - 1) / 2.0) / fps
    counts = np.full(starts.size, window - lag, dtype=np.int32)
    device = cp.cuda.runtime.getDeviceProperties(gpu_id)["name"]
    if isinstance(device, bytes):
        device = device.decode("utf-8")
    source = {"width": width, "height": height, "fps": fps, "frame_count": frame_count, "gpu": str(device)}
    return stack, times, starts, fps, source


def normalize_stack(stack: np.ndarray, low_q: float, high_q: float) -> tuple[np.ndarray, float, float]:
    low, high = np.percentile(stack[np.isfinite(stack)], (low_q, high_q))
    scaled = (stack - low) / (high - low)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)
    return (np.clip(scaled, 0.0, 1.0) * 255).astype(np.uint8), float(low), float(high)


def flow_metrics(
    stack_u8: np.ndarray,
    times: np.ndarray,
    gap: int,
    flow_config: dict[str, Any],
    blur_kernel: int,
) -> list[dict[str, Any]]:
    rows = []
    for start in range(stack_u8.shape[2] - gap):
        first = cv2.GaussianBlur(stack_u8[:, :, start], (blur_kernel, blur_kernel), 0)
        second = cv2.GaussianBlur(stack_u8[:, :, start + gap], (blur_kernel, blur_kernel), 0)
        _, mask_u8 = cv2.threshold(first, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = mask_u8.astype(bool)
        flow = cv2.calcOpticalFlowFarneback(
            first,
            second,
            None,
            pyr_scale=float(flow_config["pyr_scale"]),
            levels=int(flow_config["levels"]),
            winsize=int(flow_config["winsize"]),
            iterations=int(flow_config["iterations"]),
            poly_n=int(flow_config["poly_n"]),
            poly_sigma=float(flow_config["poly_sigma"]),
            flags=int(flow_config["flags"]),
        )
        magnitude = np.hypot(flow[:, :, 0], flow[:, :, 1])
        values = magnitude[mask & np.isfinite(magnitude)]
        dt = float(times[start + gap] - times[start])
        rows.append(
            {
                "PairStartIndex": start,
                "PairEndIndex": start + gap,
                "Time_sec": float(times[start] - times[0]),
                "Dt_sec": dt,
                "MeanFlowMagnitude_px_per_pair": float(np.mean(values)),
                "MedianFlowMagnitude_px_per_pair": float(np.median(values)),
                "P95FlowMagnitude_px_per_pair": float(np.percentile(values, 95)),
                "MeanApparentSpeed_px_per_sec": float(np.mean(values) / dt),
                "N_Interface_Pixels": int(values.size),
            }
        )
    return rows


def run_profile(
    video: Path,
    output_root: Path,
    name: str,
    config: dict[str, Any],
) -> None:
    selection = config["profiles"][name]
    activity = config["activity"]
    stack, times, starts, fps, source = build_activity_stack(video, selection, activity)
    expected_maps = int(selection["expected_map_count"])
    if stack.shape[2] != expected_maps:
        raise AssertionError(f"{name}: expected {expected_maps} maps, generated {stack.shape[2]}.")
    out_dir = output_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    output_config = config["output"]
    with h5py.File(out_dir / "activity_stack.h5", "w") as handle:
        handle.create_dataset(
            "deltaE_stack",
            data=stack,
            compression=output_config["h5_compression"],
            compression_opts=int(output_config["gzip_level"]),
            shuffle=True,
        )
        handle.create_dataset("frameTimes", data=times)
        handle.create_dataset("blockStartFramesZeroBased", data=starts)
        handle.create_dataset("validPairCounts", data=np.full(starts.size, 14, dtype=np.int32))
        handle.attrs["activity_mode"] = "ab-chroma"
        handle.attrs["source_window_frames"] = int(activity["source_window_frames"])
        handle.attrs["stride_frames"] = int(activity["stride_frames"])
        handle.attrs["temporal_lag_frames"] = int(activity["temporal_lag_frames"])
        handle.attrs["roi_xywh"] = json.dumps(selection["roi_xywh"])

    flat = stack.reshape(-1, stack.shape[2])
    thresholds = {
        f"q{int(q)}": float(np.percentile(stack, q))
        for q in config["raw_activity"]["pooled_area_percentiles"]
    }
    raw_rows = []
    for index in range(stack.shape[2]):
        row: dict[str, Any] = {
            "map_index": index,
            "block_start_frame_zero_based": int(starts[index]),
            "time_sec": float(times[index] - times[0]),
            "mean_activity_delta_ab": float(np.mean(flat[:, index])),
            "median_activity_delta_ab": float(np.median(flat[:, index])),
            "p95_activity_delta_ab": float(np.percentile(flat[:, index], 95)),
        }
        for label, threshold in thresholds.items():
            row[f"area_fraction_{label}"] = float(np.mean(stack[:, :, index] >= threshold))
        raw_rows.append(row)
    write_csv(out_dir / "raw_activity_metrics.csv", raw_rows)

    normalization = config["normalization"]
    stack_u8, norm_low, norm_high = normalize_stack(
        stack,
        float(normalization["low_percentile"]),
        float(normalization["high_percentile"]),
    )
    flow_config = config["optical_flow"]
    cv2.setNumThreads(int(flow_config["opencv_threads"]))
    blur_kernel = int(normalization["gaussian_blur_kernel"])
    short_gap = int(flow_config["short_interval_map_gap"])
    long_gap = int(flow_config["long_interval_map_gap"])
    short_rows = flow_metrics(stack_u8, times, short_gap, flow_config, blur_kernel)
    long_rows = flow_metrics(stack_u8, times, long_gap, flow_config, blur_kernel)
    write_csv(out_dir / "short_interval_metrics.csv", short_rows)
    write_csv(out_dir / "long_interval_metrics.csv", long_rows)

    metadata = {
        "profile": name,
        "input_video": str(video),
        "input_video_sha256": sha256(video),
        "selection": selection,
        "activity": activity,
        "normalization": normalization,
        "normalization_observed_p1_p99": [norm_low, norm_high],
        "optical_flow": flow_config,
        "source_video": source,
        "map_count": int(stack.shape[2]),
        "short_pair_count": len(short_rows),
        "long_pair_count": len(long_rows),
        "pooled_activity_thresholds": thresholds,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"completed {name}: {out_dir}", flush=True)


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base = config_path.parent
    video = resolve_path(base, config["input_video"])
    if not video.is_file():
        raise FileNotFoundError(f"Input video not found: {video}")
    output_root = resolve_path(base, config["output_root"])
    profiles = args.profile or list(config["profiles_to_run"])
    unknown = [name for name in profiles if name not in config["profiles"]]
    if unknown:
        raise KeyError(f"Unknown profile(s): {', '.join(unknown)}")
    for name in profiles:
        run_profile(video, output_root, name, config)


if __name__ == "__main__":
    main()
