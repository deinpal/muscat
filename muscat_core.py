# muscat_core.py
"""
Core module for MUSCAT: Markov-based Urban Simulation and Change Analysis Tool
Contains the Markov prediction function and utilities.
"""

import rasterio
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score
import os


def run_markov_prediction(initial, final, out, steps=1, mode="stochastic", drivers=None,
                          validation=None, export_dir=None, progress_callback=None):
    """
    Runs a Markov chain prediction, computes transition matrix, area change,
    and optional validation. Exports tables if export_dir is given.

    Parameters
    ----------
    initial : str
        Path to the initial raster (T0).
    final : str
        Path to the final raster (T1).
    out : str
        Output raster path for the predicted map.
    steps : int, default=1
        Number of simulation steps.
    mode : str, default="stochastic"
        Simulation mode (currently stochastic only).
    drivers : list, optional
        Spatial driver rasters (not used yet).
    validation : str, optional
        Path to validation raster (if provided).
    export_dir : str, optional
        Directory to export CSV and validation reports.
    progress_callback : callable, optional
        Function to report progress (0–100).
    """

    # ----- Load rasters -----
    with rasterio.open(initial) as src0:
        arr0 = src0.read(1)
        profile = src0.profile
    with rasterio.open(final) as src1:
        arr1 = src1.read(1)

    states = np.unique(arr0[~np.isnan(arr0)])
    transition_matrix = np.zeros((len(states), len(states)))

    total = arr0.size
    processed = 0

    # ----- Transition matrix -----
    for s0, s1 in zip(arr0.flatten(), arr1.flatten()):
        if not np.isnan(s0) and not np.isnan(s1):
            i = np.where(states == s0)[0][0]
            j = np.where(states == s1)[0][0]
            transition_matrix[i, j] += 1
        processed += 1
        if progress_callback and processed % 5000 == 0:
            progress_callback((processed / total) * 50)  # first 50%

    transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)

    # ----- Simulation -----
    current = arr1.copy()
    flat = current.flatten()
    for step in range(steps):
        new_flat = []
        for val in flat:
            if np.isnan(val):
                new_flat.append(val)
            else:
                i = np.where(states == val)[0][0]
                probs = transition_matrix[i]
                new_val = np.random.choice(states, p=probs)
                new_flat.append(new_val)
        flat = np.array(new_flat)
        if progress_callback:
            progress_callback(50 + ((step + 1) / steps) * 40)  # next 40%
    current = flat.reshape(current.shape)

    # Save raster
    profile.update(dtype=rasterio.float32, count=1, compress="lzw")
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(current.astype(np.float32), 1)

    # ----- Area change table -----
    area_table = []
    for s in states:
        init_area = np.sum(arr0 == s)
        final_area = np.sum(arr1 == s)
        pred_area = np.sum(current == s)
        area_table.append([s, init_area, final_area, pred_area,
                           final_area - init_area, pred_area - init_area])
    area_df = pd.DataFrame(area_table,
        columns=["Class", "Initial", "Final", "Predicted",
                 "Change(Final-Initial)", "Change(Pred-Initial)"])

    # ----- Validation -----
    validation_results = None
    if validation:
        with rasterio.open(validation) as val_src:
            val_arr = val_src.read(1)

        mask = ~np.isnan(val_arr) & ~np.isnan(current)
        ref = val_arr[mask].astype(int).flatten()
        pred = current[mask].astype(int).flatten()

        cm = confusion_matrix(ref, pred, labels=states)
        oa = accuracy_score(ref, pred)
        kappa = cohen_kappa_score(ref, pred)

        validation_results = {
            "Confusion Matrix": cm,
            "Overall Accuracy": oa,
            "Kappa": kappa
        }

    # ----- Export -----
    if export_dir:
        os.makedirs(export_dir, exist_ok=True)

        pd.DataFrame(transition_matrix, index=states, columns=states).to_csv(
            os.path.join(export_dir, "transition_matrix.csv"))
        area_df.to_csv(os.path.join(export_dir, "area_change_table.csv"), index=False)

        if validation_results:
            np.savetxt(os.path.join(export_dir, "confusion_matrix.csv"),
                       validation_results["Confusion Matrix"], fmt="%d", delimiter=",")
            with open(os.path.join(export_dir, "validation_metrics.txt"), "w") as f:
                f.write("Overall Accuracy: {:.3f}\n".format(validation_results["Overall Accuracy"]))
                f.write("Kappa: {:.3f}\n".format(validation_results["Kappa"]))

    if progress_callback:
        progress_callback(100)

    return current, transition_matrix, area_df, validation_results