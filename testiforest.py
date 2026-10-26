import numpy as np
import pandas as pd
from app import analyze_csv
from iforest_detector import IsolationForestDetector, RollingZScoreDetector
from iforest import build_causal_features, score_series


def make_data():
    np.random.seed(0)
    train = np.random.normal(10, 1, 2000)
    test = np.random.normal(10, 1, 500)
    return train, test


def test_one_finite_score_per_reading():
    train, test = make_data()
    scores = IsolationForestDetector().fit(train).score(test)
    assert len(scores) == len(test)
    assert np.isfinite(scores).all()


def test_spike_gets_a_high_score():
    train, test = make_data()
    test[300] = 40
    scores = IsolationForestDetector().fit(train).score(test)
    rank = (scores > scores[300]).sum() + 1
    print("spike rank:", rank, "of", len(scores))
    assert rank <= 5


def test_no_future_data_used():
    train, test = make_data()
    det = IsolationForestDetector().fit(train)
    before = det.score(test)
    changed = test.copy()
    changed[400:] = 99  # change only the FUTURE part
    after = det.score(changed)
    assert np.allclose(before[:400], after[:400])  # earlier scores must not move


def test_rolling_z_score_flags_spike():
    _, test = make_data()
    test[300] = 40
    detector = RollingZScoreDetector()
    scores = detector.score(test)
    predictions = detector.predict(test)
    assert len(scores) == len(test)
    assert np.isfinite(scores).all()
    assert predictions[300] == -1


def test_rolling_z_score_threshold_changes_sensitivity():
    values = np.tile([9.0, 11.0], 30)
    values[-1] = 13.0
    assert RollingZScoreDetector(threshold=2.0).predict(values)[-1] == -1
    assert RollingZScoreDetector(threshold=3.0).predict(values)[-1] == 1


def test_rolling_z_score_does_not_use_future_data():
    _, test = make_data()
    detector = RollingZScoreDetector()
    before = detector.score(test)
    changed = test.copy()
    changed[400:] = 99
    after = detector.score(changed)
    assert np.allclose(before[:400], after[:400])


def test_iforest_output_contract_and_causal_features():
    values = np.tile([9.0, 11.0], 40)
    values[55] = 30.0
    frame = pd.DataFrame(
        {
            "timestamp": np.arange(len(values)),
            "value": values,
        }
    )

    result = score_series(frame, "value", "timestamp", "sensor-1")
    assert result.columns.tolist() == [
        "series_id",
        "timestamp",
        "observed",
        "baseline_median",
        "robust_z",
        "score",
        "label",
    ]
    assert len(result) == 24
    assert set(result["label"].unique()).issubset({"NORMAL", "ANOMALOUS"})
    assert np.isfinite(result["score"]).all()
    assert np.isfinite(result["robust_z"]).all()

    before = build_causal_features(
        pd.Series(values), window=10, min_periods=5
    )
    changed = values.copy()
    changed[60:] = 100.0
    after = build_causal_features(
        pd.Series(changed), window=10, min_periods=5
    )
    assert np.allclose(
        before.iloc[:60].to_numpy(),
        after.iloc[:60].to_numpy(),
        equal_nan=True,
    )


def test_iforest_holdout_changes_cannot_change_earlier_holdout_scores():
    values = np.tile([9.0, 11.0], 40)
    frame = pd.DataFrame({"timestamp": np.arange(len(values)), "value": values})
    changed = frame.copy()
    changed.loc[65:, "value"] = 50.0

    before = score_series(frame, "value", "timestamp", "sensor-1")
    after = score_series(changed, "value", "timestamp", "sensor-1")
    pd.testing.assert_frame_equal(before.iloc[:9], after.iloc[:9])


def test_p3_adaptation_contract_and_event_detection():
    values = [9.0, 11.0] * 40
    values[45] = 30.0
    csv_lines = ["timestamp,sensor,value"]
    csv_lines.extend(
        f"2026-01-01T00:{index:02d}:00,sensor-1,{value}"
        for index, value in enumerate(values)
    )

    result = analyze_csv("\n".join(csv_lines), "sentinel_adaptive")
    prediction = result["predictions"][45]
    assert result["model"] == "sentinel_adaptive"
    assert prediction["label"] == "ANOMALOUS"
    assert prediction["event_id"] is not None
    assert prediction["incident_id"] is not None
    assert prediction["evidence"]["observed"] == 30.0
    assert prediction["evidence"]["raw_score"] > 0.0
    assert prediction["evidence"]["event_start_timestamp"] == prediction["timestamp"]
    assert set(prediction) == {
        "series_id",
        "timestamp",
        "anomaly_score",
        "label",
        "threshold",
        "event_id",
        "incident_id",
        "detector_version",
        "reason_codes",
        "evidence",
    }
    assert result["results"][45]["label"] == prediction["label"]
    assert result["results"][45]["incident_id"] == prediction["incident_id"]
    assert result["results"][45]["baseline_median"] == prediction["evidence"]["baseline"]
    assert result["sensitivity"] == "moderate"
    assert result["enter_threshold"] == 2.5
    assert result["exit_threshold"] == 1.5


def test_app_isolation_forest_scores_only_chronological_holdout():
    csv_lines = ["timestamp,value"]
    csv_lines.extend(
        f"{index},{10 + (index % 2)}" for index in range(80)
    )

    result = analyze_csv("\n".join(csv_lines), "isolation_forest")

    assert result["input_readings"] == 80
    assert result["scored_readings"] == 24
    assert result["results"][0]["timestamp"] == "56"
    assert result["results"][-1]["timestamp"] == "79"
    assert {row["scoring_scope"] for row in result["results"]} == {
        "chronological_holdout"
    }


if __name__ == "__main__":
    test_one_finite_score_per_reading()
    test_spike_gets_a_high_score()
    test_no_future_data_used()
    test_rolling_z_score_flags_spike()
    test_rolling_z_score_threshold_changes_sensitivity()
    test_rolling_z_score_does_not_use_future_data()
    test_iforest_output_contract_and_causal_features()
    test_iforest_holdout_changes_cannot_change_earlier_holdout_scores()
    test_p3_adaptation_contract_and_event_detection()
    print("All tests passed")