import numpy as np


def errors(actual, reference, thresholds=None):
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Output shape mismatch: {a.shape} != {b.shape}")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite output/reference")
    delta = a - b
    rmse = float(np.sqrt(np.mean(delta ** 2)))
    denom = float(np.linalg.norm(a.ravel()) * np.linalg.norm(b.ravel()))
    cosine = float(np.dot(a.ravel(), b.ravel()) / denom) if denom else (1.0 if np.array_equal(a, b) else 0.0)
    result = {"mae": float(np.abs(delta).mean()), "max_abs": float(np.abs(delta).max()),
              "rmse": rmse, "relative_rmse": rmse / max(float(np.sqrt(np.mean(b ** 2))), 1e-12),
              "cosine": cosine, "per_action_dim_mae": np.abs(delta).reshape(-1, a.shape[-1]).mean(0).tolist(),
              "gate": "not_configured"}
    if thresholds:
        allowed = {"max_abs", "mae", "rmse", "relative_rmse", "min_cosine"}
        if set(thresholds) - allowed:
            raise ValueError(f"Unknown gates: {set(thresholds) - allowed}")
        passed = all(result["cosine"] >= v if k == "min_cosine" else result[k] <= v
                     for k, v in thresholds.items())
        result["gate"] = "passed" if passed else "failed"
    return result


def latency_summary(values):
    v = np.asarray(values, dtype=float)
    if not len(v) or not np.isfinite(v).all() or (v < 0).any():
        raise ValueError("Invalid latency samples")
    return {"count": len(v), "mean_ms": float(v.mean()), "min_ms": float(v.min()),
            **{f"p{p}_ms": float(np.percentile(v, p)) for p in (50, 95, 99)}}
