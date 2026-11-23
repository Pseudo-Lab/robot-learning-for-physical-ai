"""Aggregate recorded HW4 robot-arm tracking episodes."""

import numpy as np


def metrics(rows) -> dict[str, float | int]:
    """Summarize nonempty episode rows with errors and target positions in metres."""
    mean_errors_m = [row["mean_error_m"] for row in rows]
    mean_errors_cm = np.array([100 * error for error in mean_errors_m])
    distances_m = np.linalg.norm([row["target"] for row in rows], axis=1)
    median_distance_m = np.median(distances_m)
    far = distances_m >= median_distance_m

    return {
        "mean_return": float(np.mean([row["episode_return"] for row in rows])),
        "mean_error_cm": float(100 * np.mean(mean_errors_m)),
        "mean_final_error_cm": float(
            100 * np.mean([row["final_error_m"] for row in rows])
        ),
        "episodes_final_within_5cm": int(
            sum(row["final_error_m"] <= 0.05 for row in rows)
        ),
        "p90_error_cm": float(np.quantile(mean_errors_cm, 0.9)),
        "median_goal_distance_cm": float(100 * median_distance_m),
        "far_mean_cm": float(mean_errors_cm[far].mean()),
        "near_mean_cm": float(mean_errors_cm[~far].mean()),
    }
