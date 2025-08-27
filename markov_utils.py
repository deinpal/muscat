import os
import numpy as np
import pandas as pd
import rasterio

# Optional: scikit-learn for kappa/accuracy; fall back to numpy if unavailable
try:
    from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score
    _HAS_SK = True
except Exception:
    _HAS_SK = False

def _read_band_as_float_with_nan(path):
    """Read single-band raster as float32 with NoData -> np.nan."""
    with rasterio.open(path) as src:
        arr = src.read(1, masked=True)  # respects nodata
        profile = src.profile
    return np.array(arr, dtype=np.float32), profile

def _safe_unique(values):
    v = values[~np.isnan(values)]
    return np.unique(v)

def _compute_confusion_np(ref, pred, labels):
    """Confusion matrix using numpy (if sklearn not present)."""
    label_to_idx = {v: i for i, v in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for r, p in zip(ref, pred):
        if np.isnan(r) or np.isnan(p):
            continue
        i = label_to_idx[r]
        j = label_to_idx[p]
        cm[i, j] += 1
    return cm

def _kappa_from_cm(cm):
    total = cm.sum()
    if total == 0:
        return 0.0
    po = np.trace(cm) / total
    pe = (cm.sum(axis=0) * cm.sum(axis=1)).sum() / (total ** 2)
    if pe == 1:
        return 0.0
    return (po - pe) / (1 - pe)

def _oa_from_cm(cm):
    total = cm.sum()
    return float(np.trace(cm)) / total if total else 0.0

def run_markov_prediction(
    initial, final, out,
    steps=1, mode="stochastic",
    drivers=None, validation=None,
    export_dir=None, progress_callback=None
):
    """
    Runs a Markov chain prediction from an initial raster to a final raster,
    computes transition probability matrix, area change table, and (optional) validation.

    progress_callback: callable(percent:int) to report progress
    """

    # ---------------- Load rasters (float with NaN nodata) ----------------
    arr0, profile0 = _read_band_as_float_with_nan(initial)
    arr1, profile1 = _read_band_as_float_with_nan(final)

    # Basic dimension/profile check (same size is required)
    if arr0.shape != arr1.shape:
        raise ValueError("Initial and Final rasters do not match in dimensions.")

    # ---------------- Build Transition Matrix ----------------
    # Use union of classes present in both maps
    states = np.unique(np.concatenate((_safe_unique(arr0), _safe_unique(arr1))))
    n = len(states)
    trans_counts = np.zeros((n, n), dtype=np.float64)

    flat0 = arr0.ravel()
    flat1 = arr1.ravel()
    total_pairs = flat0.size
    # Build index lookup once
    state_to_idx = {v: i for i, v in enumerate(states)}

    processed = 0
    for s0, s1 in zip(flat0, flat1):
        if not np.isnan(s0) and not np.isnan(s1):
            i = state_to_idx[s0]
            j = state_to_idx[s1]
            trans_counts[i, j] += 1
        processed += 1
        if progress_callback and (processed % 200000 == 0):  # update every ~200k pixels
            pct = int(30 * processed / total_pairs)
            progress_callback(min(pct, 30))

    # Normalize rows to probabilities (safe)
    row_sums = trans_counts.sum(axis=1, keepdims=True)
    np.maximum(row_sums, 1.0, out=row_sums)  # avoid divide by zero
    transition_matrix = trans_counts / row_sums

    # ---------------- Run Prediction ----------------
    current = arr1.copy()
    if mode.lower() not in ("stochastic", "deterministic"):
        mode = "stochastic"

    for step in range(steps):
        flat = current.ravel()
        new_flat = flat.copy()

        # Vectorized mapping: for each class i, sample/assign next class
        # Build mask per class
        for i, cls in enumerate(states):
            cls_mask = ~np.isnan(flat) & (flat == cls)
            if not np.any(cls_mask):
                continue
            if mode.lower() == "stochastic":
                # Draw new classes according to row probs
                probs = transition_matrix[i]
                # Sample indices of states
                choices = np.random.choice(states, size=int(cls_mask.sum()), p=probs)
                new_flat[cls_mask] = choices
            else:
                # Deterministic: use argmax
                j = int(np.argmax(transition_matrix[i]))
                new_flat[cls_mask] = states[j]

        current = new_flat.reshape(current.shape)

        if progress_callback:
            pct = 30 + int(70 * (step + 1) / max(steps, 1))
            progress_callback(min(pct, 100))

    # ---------------- Save prediction raster ----------------
    out_profile = profile0.copy()
    out_profile.update(dtype=rasterio.float32, count=1, compress='lzw')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with rasterio.open(out, 'w', **out_profile) as dst:
        dst.write(current.astype(np.float32), 1)

    # ---------------- Area Change Table ----------------
    rows = []
    for s in states:
        init_area = int(np.sum(arr0 == s))
        final_area = int(np.sum(arr1 == s))
        pred_area = int(np.sum(current == s))
        rows.append([
            s, init_area, final_area, pred_area,
            final_area - init_area, pred_area - init_area
        ])
    area_df = pd.DataFrame(
        rows,
        columns=["Class", "Initial", "Final", "Predicted",
                 "Change(Final-Initial)", "Change(Pred-Initial)"]
    )

    # ---------------- Validation (Optional) ----------------
    validation_results = None
    if validation:
        val_arr, _ = _read_band_as_float_with_nan(validation)
        if val_arr.shape != current.shape:
            raise ValueError("Validation raster does not match prediction raster dimensions.")

        mask = ~np.isnan(val_arr) & ~np.isnan(current)
        ref = val_arr[mask].astype(float)
        pred = current[mask].astype(float)

        # Map values to nearest states (in case of float encoding)
        # Here we assume exact class values are used, so this is pass-through.

        if _HAS_SK:
            cm = confusion_matrix(ref, pred, labels=states)
            oa = accuracy_score(ref, pred)
            kappa = cohen_kappa_score(ref, pred)
        else:
            cm = _compute_confusion_np(ref, pred, labels=states)
            oa = _oa_from_cm(cm)
            kappa = _kappa_from_cm(cm)

        # Labeled confusion matrix
        cm_df = pd.DataFrame(cm, index=[f"Ref_{s}" for s in states],
                                 columns=[f"Pred_{s}" for s in states])

        validation_results = {
            "Confusion Matrix": cm_df,
            "Overall Accuracy": float(oa),
            "Kappa": float(kappa)
        }

    # ---------------- Exports ----------------
    if export_dir is None:
        export_dir = os.path.dirname(out)
    os.makedirs(export_dir, exist_ok=True)

    # Transition probability CSV
    trans_df = pd.DataFrame(transition_matrix, index=states, columns=states)
    trans_csv = os.path.join(export_dir, "transition_matrix.csv")
    trans_df.to_csv(trans_csv)

    # Area change CSV
    area_csv = os.path.join(export_dir, "area_change_table.csv")
    area_df.to_csv(area_csv, index=False)

    # Validation exports
    if validation_results:
        cm_csv = os.path.join(export_dir, "confusion_matrix.csv")
        validation_results["Confusion Matrix"].to_csv(cm_csv)
        with open(os.path.join(export_dir, "validation_metrics.txt"), "w") as f:
            f.write(f"Overall Accuracy: {validation_results['Overall Accuracy']:.4f}\n")
            f.write(f"Kappa: {validation_results['Kappa']:.4f}\n")

    # Ensure 100%
    if progress_callback:
        progress_callback(100)

    return current, transition_matrix, area_df, validation_results