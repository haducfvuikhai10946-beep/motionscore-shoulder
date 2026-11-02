"""The fixed thirteen paper descriptors, with original arithmetic preserved."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

NAMES = [
    "angle_rom_mean_deg", "symmetry_rom_diff_deg",
    "upward_mean_abs_acceleration_deg_s2", "downward_mean_abs_acceleration_deg_s2",
    "overall_mean_abs_velocity_deg_s", "overall_mean_abs_acceleration_deg_s2",
    "peakhold_fraction", "bilateral_velocity_asymmetry", "trunk_tilt_rms_deg",
    "trunk_tilt_cycle_sd_deg", "shoulder_center_sway_width", "pelvis_center_sway_width",
    "velocity_peak_mean_ratio",
]
JOINTS = [f"{side}_{joint}" for side in ["LEFT", "RIGHT"]
          for joint in ["SHOULDER", "ELBOW", "WRIST", "HIP"]]
PHASE_NAMES = ["valley_hold", "rising", "peak_hold", "falling"]


def validate_pose(df: pd.DataFrame) -> float:
    required = ["timestamp_sec"] + [f"{j}_{a}" for j in JOINTS for a in "xy"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing pose columns: {sorted(missing)}")
    if len(df) < 10 or not np.isfinite(df[required].to_numpy(float)).all():
        raise ValueError("Pose must contain >=10 finite frames; imputation is not automatic")
    dt = np.diff(df.timestamp_sec.to_numpy(float))
    if (dt <= 0).any() or not np.allclose(dt, .05, atol=2e-6, rtol=0):
        raise ValueError("The paper protocol requires uniformly spaced 20 fps poses")
    return float(1 / np.median(dt))


def angles(df: pd.DataFrame) -> np.ndarray:
    result = []
    for side in ["LEFT", "RIGHT"]:
        anchor = df[[side + "_SHOULDER_x", side + "_SHOULDER_y"]].to_numpy(float)
        a = df[[side + "_ELBOW_x", side + "_ELBOW_y"]].to_numpy(float) - anchor
        b = df[[side + "_HIP_x", side + "_HIP_y"]].to_numpy(float) - anchor
        den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        theta = np.degrees(np.arccos(np.clip((a*b).sum(1)/(den + 1e-9), -1, 1)))
        theta[den <= 1e-9] = np.nan
        result.append(theta)
    return np.array(result)


def frame_features(df: pd.DataFrame) -> np.ndarray:
    """Match the original scalar dot/norm implementation, including float32 casts."""
    validate_pose(df)
    bilateral = []
    for side in ["LEFT", "RIGHT"]:
        values = []
        for _, row in df.iterrows():
            anchor = np.array([row[f"{side}_SHOULDER_x"], row[f"{side}_SHOULDER_y"]])
            a = np.array([row[f"{side}_ELBOW_x"], row[f"{side}_ELBOW_y"]]) - anchor
            b = np.array([row[f"{side}_HIP_x"], row[f"{side}_HIP_y"]]) - anchor
            cosine = np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-9)
            values.append(float(np.degrees(np.arccos(np.clip(cosine, -1., 1.)))))
        bilateral.append(np.array(values).astype(np.float32))
    left, right = bilateral
    mean_angle = ((left + right) / 2.).astype(np.float32)
    velocity = np.gradient(mean_angle).astype(np.float32)
    x = np.stack([mean_angle, left, right, velocity], axis=1)
    return ((x - np.mean(x, axis=0, keepdims=True)) /
            (np.std(x, axis=0, keepdims=True) + 1e-6)).astype(np.float32)


def retained_cycles(labels: np.ndarray, margin: int = 10) -> list[np.ndarray]:
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) == 0 or not np.isin(labels, range(4)).all():
        raise ValueError("Labels must be a nonempty 1D array in {0,1,2,3}")
    starts = np.r_[0, np.flatnonzero(np.diff(labels) != 0)+1]
    ends = np.r_[starts[1:], len(labels)]
    runs = [(int(labels[s]), int(s), int(e)) for s,e in zip(starts, ends)]
    result = []
    for j in range(len(runs)-3):
        block = runs[j:j+4]
        if [b[0] for b in block] == [1,2,3,0]:
            cut = np.array([block[0][1]] + [b[2] for b in block], int)
            if cut[0] >= margin and cut[-1] <= len(labels)-margin:
                result.append(cut)
    return result


def cycle_features(df: pd.DataFrame, cuts: list[np.ndarray]) -> np.ndarray:
    fps = validate_pose(df)
    theta = angles(df)
    if not np.isfinite(theta).all():
        raise ValueError("Degenerate shoulder geometry")
    smooth = savgol_filter(theta, 9, 2, axis=1)
    vel = np.gradient(smooth, axis=1)*fps
    acc = np.gradient(vel, axis=1)*fps
    left = df[["LEFT_SHOULDER_x","LEFT_SHOULDER_y"]].to_numpy()
    right = df[["RIGHT_SHOULDER_x","RIGHT_SHOULDER_y"]].to_numpy()
    shoulder = (left+right)/2
    pelvis = (df[["LEFT_HIP_x","LEFT_HIP_y"]].to_numpy()+
              df[["RIGHT_HIP_x","RIGHT_HIP_y"]].to_numpy())/2
    width = float(np.median(np.linalg.norm(left-right, axis=1)))
    if width <= 1:
        raise ValueError("Median shoulder width must exceed one pixel")
    tilt = np.degrees(np.arctan2((shoulder-pelvis)[:,0], -(shoulder-pelvis)[:,1]))
    vals = []
    for c in cuts:
        c = np.asarray(c, int)
        if c.shape != (5,) or not (np.diff(c)>0).all() or not (0<=c[0]<c[-1]<=len(df)):
            raise ValueError("A cycle needs five strictly increasing, zero-based boundaries")
        st,en = c[0],c[-1]
        rom = np.ptp(theta[:,st:en], axis=1)
        v = vel[:,st:en]; av = abs(v); meanv = av.mean()
        if meanv <= 0:
            raise ValueError("Zero-motion cycle cannot define velocity ratios")
        up = abs(acc[:,c[0]:c[1]]).mean(axis=1).mean()
        down = abs(acc[:,c[2]:c[3]]).mean(axis=1).mean()
        vals.append([rom.mean(), abs(rom[0]-rom[1]), up, down, meanv,
                     abs(acc[:,st:en]).mean(), (c[2]-c[1])/(en-st),
                     abs(v[0]-v[1]).mean()/(av[0]+av[1]).mean(),
                     np.sqrt(np.mean(tilt[st:en]**2)), tilt[st:en].std(),
                     np.ptp(np.quantile(shoulder[st:en,0],[.05,.95]))/width,
                     np.ptp(np.quantile(pelvis[st:en,0],[.05,.95]))/width,
                     np.quantile(av,.95)/meanv])
    return np.array(vals, float).reshape(-1,13)


def aggregate(df: pd.DataFrame, labels: np.ndarray) -> dict:
    if len(labels) != len(df):
        raise ValueError("One phase label is required for each pose frame")
    cuts = retained_cycles(labels)
    mat = cycle_features(df, cuts)
    if len(mat) < 2:
        raise ValueError("At least two retained cycles are needed for odd/even reporting")
    return {"cuts": np.asarray(cuts), "cycles": mat,
            "full": np.median(mat,axis=0), "odd": np.median(mat[::2],axis=0),
            "even": np.median(mat[1::2],axis=0)}
