# audio_analyzer/analysis_core.py
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import librosa
import scipy.signal


BANDS_HZ = {
    "bass": (20.0, 250.0),
    "mid": (250.0, 4000.0),
    "treble": (4000.0, 16000.0),
}

CHANGE_WEIGHTS = {
    "mood": 0.55,
    "pitch": 0.30,
    "loud": 0.15,
}


@dataclass
class AnalysisParams:
    sr: int = 22050
    hop_sec: float = 0.2
    n_fft: int = 2048
    win_length: int = 2048

    # Change-point target spacing
    major_min_gap_sec: float = 8.0
    major_target_gap_sec: float = 12.0
    minor_min_gap_sec: float = 3.0

    # Peak-picking behavior
    major_percentile_start: float = 90.0
    minor_percentile_start: float = 80.0


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def zscore(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    return (x - mu) / (sd + eps)


def robust_diff(x: np.ndarray) -> np.ndarray:
    return np.diff(x, prepend=x[0])


def normalize_01(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    mn = np.nanmin(x)
    mx = np.nanmax(x)
    return (x - mn) / (mx - mn + eps)


def normalize_01_percentile(
    x: np.ndarray,
    p_lo: float = 10.0,
    p_hi: float = 95.0,
    eps: float = 1e-9,
) -> np.ndarray:
    lo = float(np.nanpercentile(x, p_lo))
    hi = float(np.nanpercentile(x, p_hi))
    y = (x - lo) / (hi - lo + eps)
    return np.clip(y, 0.0, 1.0)


def time_axis(num_frames: int, hop_sec: float) -> np.ndarray:
    return np.arange(num_frames) * hop_sec


def hz_to_bin(freqs: np.ndarray, f_lo: float, f_hi: float) -> np.ndarray:
    return np.where((freqs >= f_lo) & (freqs < f_hi))[0]


def smooth_series(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    k = int(k)
    if k % 2 == 0:
        k += 1
    kernel = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(x, kernel, mode="same")


def smooth_seconds(x: np.ndarray, hop_sec: float, win_sec: float) -> np.ndarray:
    k = max(1, int(round(float(win_sec) / max(1e-9, float(hop_sec)))))
    return smooth_series(x, k)


def envelope_follower(
    x: np.ndarray,
    hop_sec: float,
    attack_sec: float = 0.08,
    release_sec: float = 0.35,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    out = np.zeros_like(x, dtype=np.float32)

    a = float(max(1e-6, attack_sec))
    r = float(max(1e-6, release_sec))
    dt = float(max(1e-9, hop_sec))

    # coefficients in [0..1], closer to 1 = slower response
    a_coeff = math.exp(-dt / a)
    r_coeff = math.exp(-dt / r)

    y = 0.0
    for i in range(x.shape[0]):
        xi = float(x[i])
        if xi > y:
            y = a_coeff * y + (1.0 - a_coeff) * xi
        else:
            y = r_coeff * y + (1.0 - r_coeff) * xi
        out[i] = y
    return out


def compute_features(y: np.ndarray, sr: int, params: AnalysisParams) -> Dict[str, np.ndarray]:
    hop_length = max(1, int(round(params.hop_sec * sr)))

    S = librosa.stft(
        y,
        n_fft=params.n_fft,
        hop_length=hop_length,
        win_length=params.win_length,
        center=True,
        window="hann",
    )
    P = np.abs(S) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=params.n_fft)

    rms = librosa.feature.rms(S=np.sqrt(P), frame_length=params.win_length, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    frame_len = params.win_length
    y_pad = np.pad(y, (frame_len // 2, frame_len // 2), mode="reflect")
    frames = librosa.util.frame(y_pad, frame_length=frame_len, hop_length=hop_length)
    peak_env = np.max(np.abs(frames), axis=0)

    band_energies = {}
    for name, (f_lo, f_hi) in BANDS_HZ.items():
        idx = hz_to_bin(freqs, f_lo, min(f_hi, freqs[-1]))
        band_energies[name] = P[idx, :].sum(axis=0) if idx.size else np.zeros(P.shape[1], dtype=np.float32)

    total_energy = P.sum(axis=0) + 1e-12
    bass_ratio = band_energies["bass"] / total_energy
    mid_ratio = band_energies["mid"] / total_energy
    treble_ratio = band_energies["treble"] / total_energy

    chroma = librosa.feature.chroma_stft(S=np.sqrt(P), sr=sr, hop_length=hop_length, n_fft=params.n_fft).T
    centroid = librosa.feature.spectral_centroid(S=np.sqrt(P), sr=sr, hop_length=hop_length)[0]

    spec = np.sqrt(P)
    spec_norm = spec / (np.sum(spec, axis=0, keepdims=True) + 1e-12)
    diff_spec = np.diff(spec_norm, axis=1, prepend=spec_norm[:, :1])
    flux = np.sqrt(np.sum(np.maximum(diff_spec, 0.0) ** 2, axis=0))

    frames_n = P.shape[1]
    return {
        "rms_db": rms_db.astype(np.float32),
        "peak_env": peak_env[:frames_n].astype(np.float32),
        "bass_energy": band_energies["bass"].astype(np.float32),
        "mid_energy": band_energies["mid"].astype(np.float32),
        "treble_energy": band_energies["treble"].astype(np.float32),
        "bass_ratio": bass_ratio.astype(np.float32),
        "mid_ratio": mid_ratio.astype(np.float32),
        "treble_ratio": treble_ratio.astype(np.float32),
        "chroma": chroma.astype(np.float32),
        "centroid_hz": centroid.astype(np.float32),
        "flux": flux.astype(np.float32),
        "hop_length": hop_length,
        "frames": frames_n,
    }


def compute_change_score(feat: Dict[str, np.ndarray]) -> np.ndarray:
    loud = feat["rms_db"]
    loud_d = np.maximum(robust_diff(loud), 0.0)
    loud_term = zscore(loud_d)

    chroma = feat["chroma"]
    a = chroma
    b = np.roll(chroma, 1, axis=0)
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    cos_sim = np.sum(a_norm * b_norm, axis=1)
    chroma_dist = 1.0 - cos_sim
    chroma_dist[0] = 0.0

    centroid = feat["centroid_hz"]
    centroid_d = np.abs(robust_diff(centroid))
    pitch_term = zscore(chroma_dist) * 0.65 + zscore(centroid_d) * 0.35

    flux = feat["flux"]
    bass_r = feat["bass_ratio"]
    mid_r = feat["mid_ratio"]
    treble_r = feat["treble_ratio"]

    ratio_change = np.abs(robust_diff(bass_r)) + np.abs(robust_diff(mid_r)) + np.abs(robust_diff(treble_r))
    mood_term = zscore(flux) * 0.65 + zscore(ratio_change) * 0.35

    score = (
        CHANGE_WEIGHTS["mood"] * mood_term
        + CHANGE_WEIGHTS["pitch"] * pitch_term
        + CHANGE_WEIGHTS["loud"] * loud_term
    )

    if score.shape[0] >= 3:
        score = np.convolve(score, np.ones(3) / 3.0, mode="same")

    return score.astype(np.float32)


def compute_energy_envelope(
    feat: Dict[str, np.ndarray],
    hop_sec: float,
    *,
    rms_smooth_sec: float = 0.9,
    bass_smooth_sec: float = 0.6,
    punch_attack_sec: float = 0.08,
    punch_release_sec: float = 0.35,
    final_smooth_sec: float = 0.25,
    w_rms: float = 0.75,
    w_bass: float = 0.15,
    w_punch: float = 0.10,
    p_lo: float = 10.0,
    p_hi: float = 95.0,
) -> np.ndarray:
    """
    Sustained intensity energy in [0..1], driven mostly by loudness, with bass lift + punch accent.

    - rms_db -> section-level intensity (heavily smoothed)
    - bass_ratio -> gentle lift during bass-heavy moments (smoothed)
    - flux -> transient accent through an attack/release envelope (not flickery)
    """
    rms01 = normalize_01_percentile(feat["rms_db"], p_lo=p_lo, p_hi=p_hi)
    bass01 = normalize_01_percentile(feat["bass_ratio"], p_lo=p_lo, p_hi=p_hi)
    flux01 = normalize_01_percentile(feat["flux"], p_lo=p_lo, p_hi=p_hi)

    e_base = smooth_seconds(rms01, hop_sec, rms_smooth_sec)
    b = smooth_seconds(bass01, hop_sec, bass_smooth_sec)

    punch_env = envelope_follower(flux01, hop_sec, attack_sec=punch_attack_sec, release_sec=punch_release_sec)
    punch_env = normalize_01_percentile(punch_env, p_lo=p_lo, p_hi=p_hi)

    e = (w_rms * e_base) + (w_bass * b) + (w_punch * punch_env)
    e = np.clip(e, 0.0, 1.0)
    e = smooth_seconds(e, hop_sec, final_smooth_sec)
    return np.clip(e, 0.0, 1.0).astype(np.float32)


def pick_peaks_target_rate(
    score: np.ndarray,
    hop_sec: float,
    min_gap_sec: float,
    target_gap_sec: float,
    percentile_start: float,
    percentile_floor: float = 60.0,
) -> np.ndarray:
    T = score.shape[0]
    duration = T * hop_sec
    target_count = max(1, int(round(duration / target_gap_sec)))
    min_distance = max(1, int(round(min_gap_sec / hop_sec)))

    percentiles = list(np.linspace(percentile_start, percentile_floor, num=16))
    peaks_best = np.array([], dtype=int)

    for p in percentiles:
        thr = np.percentile(score, p)
        peaks, props = scipy.signal.find_peaks(score, height=thr, distance=min_distance)
        if peaks.size == 0:
            continue
        if peaks.size > target_count:
            heights = props["peak_heights"]
            order = np.argsort(heights)[::-1][:target_count]
            peaks = np.sort(peaks[order])
        peaks_best = peaks
        if peaks_best.size >= target_count:
            break

    if peaks_best.size == 0:
        peaks, _ = scipy.signal.find_peaks(score, distance=min_distance)
        if peaks.size:
            order = np.argsort(score[peaks])[::-1][:target_count]
            peaks_best = np.sort(peaks[order])

    return peaks_best


def pick_minor_peaks(score: np.ndarray, hop_sec: float, min_gap_sec: float, percentile_start: float) -> np.ndarray:
    min_distance = max(1, int(round(min_gap_sec / hop_sec)))
    thr = np.percentile(score, percentile_start)
    peaks, _ = scipy.signal.find_peaks(score, height=thr, distance=min_distance)
    return peaks.astype(int)


def write_timeline_json(
    out_path: Path,
    t: np.ndarray,
    feat: Dict[str, np.ndarray],
    score: np.ndarray,
    major_idx: np.ndarray,
    minor_idx: np.ndarray,
    params: AnalysisParams,
    source_video: str,
) -> None:
    major_set = set(int(i) for i in major_idx.tolist())
    minor_set = set(int(i) for i in minor_idx.tolist())

    timeline = []
    for i in range(len(t)):
        timeline.append({
            "t": float(t[i]),
            "loudness_rms_db": float(feat["rms_db"][i]),
            "waveform_peak_env": float(feat["peak_env"][i]),
            "bass_energy": float(feat["bass_energy"][i]),
            "mid_energy": float(feat["mid_energy"][i]),
            "treble_energy": float(feat["treble_energy"][i]),
            "bass_ratio": float(feat["bass_ratio"][i]),
            "mid_ratio": float(feat["mid_ratio"][i]),
            "treble_ratio": float(feat["treble_ratio"][i]),
            "centroid_hz": float(feat["centroid_hz"][i]),
            "chroma": feat["chroma"][i].astype(float).tolist(),
            "change_score": float(score[i]),
            "is_minor_change": i in minor_set,
            "is_major_change": i in major_set,
        })

    summary = {
        "source_video": source_video,
        "sr": params.sr,
        "hop_sec": params.hop_sec,
        "bands_hz": BANDS_HZ,
        "change_weights": CHANGE_WEIGHTS,
        "major_change_times_sec": [float(t[i]) for i in major_idx.tolist()],
        "minor_change_times_sec": [float(t[i]) for i in minor_idx.tolist()],
        "major_min_gap_sec": params.major_min_gap_sec,
        "major_target_gap_sec": params.major_target_gap_sec,
        "minor_min_gap_sec": params.minor_min_gap_sec,
    }

    out_path.write_text(json.dumps({"summary": summary, "timeline": timeline}, indent=2), encoding="utf-8")
