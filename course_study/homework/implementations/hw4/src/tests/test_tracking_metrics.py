from __future__ import annotations

import json

import pytest

from companion.tracking_metrics import metrics
from conftest import PROJECT_ROOT


def test_tracking_metrics_matches_saved_hw4_evaluation():
    path = (
        PROJECT_ROOT
        / "tests/fixtures/hw4_tracking_reference.json"
    )
    if not path.is_file():
        pytest.skip("Saved HW4 evaluation is unavailable in this workspace")
    original = json.loads(path.read_text(encoding="utf-8"))["original"]
    rows, expected = original["episodes"], original["metrics"]
    assert len(rows) == 100

    actual = metrics(rows)

    assert actual.keys() == expected.keys()
    assert len(actual) == 8
    count = actual["episodes_final_within_5cm"]
    assert type(count) is int
    assert count == expected["episodes_final_within_5cm"] == 61
    assert actual == pytest.approx(expected, rel=0.0, abs=1e-8)


def test_tracking_metrics_includes_threshold_and_median_ties():
    rows = [
        {
            "target": [0.0, 0.0, distance],
            "episode_return": 0.0,
            "mean_error_m": mean_error,
            "final_error_m": final_error,
        }
        for distance, mean_error, final_error in [
            (1.0, 0.01, 0.05),
            (2.0, 0.02, 0.05000001),
            (-2.0, 0.03, 0.04999999),
            (3.0, 0.04, 0.06),
        ]
    ]

    summary = metrics(rows)

    assert summary["episodes_final_within_5cm"] == 2
    assert summary["median_goal_distance_cm"] == 200.0
    assert summary["near_mean_cm"] == pytest.approx(1.0)
    assert summary["far_mean_cm"] == pytest.approx(3.0)
