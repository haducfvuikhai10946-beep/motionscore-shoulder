"""Portable, inspectable scoring; loads JSON/LightGBM text, never pickle."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .features import NAMES


class MotionScorer:
    def __init__(self, model_dir: str | Path, backend: str = "LightGBM"):
        self.root = Path(model_dir)
        self.ref = json.loads((self.root/"reference.json").read_text("utf-8"))
        if self.ref["features"] != NAMES:
            raise ValueError("Model feature order does not match this package")
        self.backend = backend
        self.models = None
        if backend == "LightGBM":
            import lightgbm as lgb
            self.models = [lgb.Booster(model_file=str(self.root/"LightGBM"/f"F{j+1:02}.txt"))
                           for j in range(13)]
        elif backend in ["Ridge", "ElasticNet"]:
            self.linear = json.loads((self.root/f"{backend}.json").read_text("utf-8"))
        elif backend != "Identity":
            raise ValueError("Supported backends: Identity, Ridge, ElasticNet, LightGBM")
        self.ecdf = self.ref.get("reference_scores",{}).get(backend)

    def standardize(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x,float))
        if x.shape[1] != 13:
            raise ValueError("Expected thirteen features in the documented order")
        m,s = np.asarray(self.ref["median"]), np.asarray(self.ref["iqr"])
        return np.clip((np.where(np.isfinite(x),x,m)-m)/s,-6,6)

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = self.standardize(x)
        if self.backend == "Identity":
            return z
        if self.models is not None:
            return np.column_stack([m.predict(z,num_threads=2) for m in self.models])
        return z@np.asarray(self.linear["coef"]).T+np.asarray(self.linear["intercept"])

    def score(self, x: np.ndarray) -> dict:
        mapped = self.predict(x)
        clipped = np.clip(mapped,-6,6)
        median,iqr = np.asarray(self.ref["median"]),np.asarray(self.ref["iqr"])
        reconstructed = median+clipped*iqr
        # Preserve the historical reconstruct -> standardize round trip.
        values = self.standardize(reconstructed)
        raw = values*iqr+median
        penalty = abs(values)
        for j,anchor in enumerate(self.ref["anchors"]):
            if anchor == "zero":
                penalty[:,j] = np.clip(raw[:,j]/iqr[j],0,6)
            elif anchor != "reference":
                raise ValueError(f"Unsupported anchor: {anchor}")
        deviation = np.zeros(len(values)); parts = {}
        for key,block in self.ref["blocks"].items():
            value = penalty[:,block["indices"]]@np.array(block["weights"])
            parts[key] = block["budget"]*value
            deviation += parts[key]
        score = 100/(1+deviation)
        thresholds = self.ref["thresholds"]
        band = np.where(deviation<=thresholds[0],0,np.where(deviation<=thresholds[1],1,2))
        result = {"mapped_z":mapped,"penalties":penalty,"D":deviation,
                  "MotionScore":score,"band":band,"contributions":parts}
        if self.ecdf is not None:
            ref = np.sort(np.asarray(self.ecdf,float))
            result["percentile"] = np.searchsorted(ref,score,side="right")/len(ref)*100
        return result
