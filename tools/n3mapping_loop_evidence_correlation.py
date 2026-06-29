#!/usr/bin/env python3
"""Compare loop evidence signals against GT-labeled Z failure classes."""

import argparse
import csv
import json
import math
from pathlib import Path


DEFAULT_SIGNALS = [
    "fitness_score",
    "inlier_ratio",
    "vertical_observability_score",
    "source_z_span",
    "target_z_span",
    "z_overlap_ratio_before",
    "z_overlap_ratio_after",
    "source_z_robust_span",
    "target_z_robust_span",
    "z_robust_overlap_ratio_before",
    "z_robust_overlap_ratio_after",
    "source_target_z_centroid_delta_before",
    "source_target_z_centroid_delta_after",
    "vertical_information_ratio",
    "best_z_offset_m",
    "best_z_offset_fitness",
    "zero_z_fitness",
    "fitness_gap_zero_vs_best",
    "z_hypothesis_spread_m",
    "vertical_ambiguity_score",
    "heightmap_overlap_cell_count",
    "heightmap_overlap_ratio",
    "heightmap_ground_dz_median",
    "heightmap_ground_dz_p90",
    "heightmap_ground_dz_max",
    "heightmap_ground_support_ratio",
    "heightmap_vertical_consistency_score",
    "submap_pred_overlap_ratio",
    "submap_pred_support_ratio",
    "submap_pred_consistency_score",
    "submap_measured_overlap_ratio",
    "submap_measured_support_ratio",
    "submap_measured_consistency_score",
    "submap_overlap_gain",
    "graph_trial_residual_x_after",
    "graph_trial_residual_y_after",
    "graph_trial_residual_z_after",
    "graph_trial_residual_roll_after",
    "graph_trial_residual_pitch_after",
    "graph_trial_residual_yaw_after",
    "graph_trial_residual_translation_norm_after",
    "graph_trial_residual_rotation_norm_after",
    "graph_trial_mean_pose_update_translation",
    "graph_trial_max_pose_update_translation",
    "graph_trial_mean_pose_update_rotation",
    "graph_trial_max_pose_update_rotation",
    "graph_trial_existing_loop_residual_delta",
    "graph_trial_odom_residual_delta",
    "graph_trial_consistency_score",
]


DERIVED_ABS_SIGNALS = {
    "abs_source_target_z_centroid_delta_before": "source_target_z_centroid_delta_before",
    "abs_source_target_z_centroid_delta_after": "source_target_z_centroid_delta_after",
    "abs_best_z_offset_m": "best_z_offset_m",
    "abs_graph_trial_existing_loop_residual_delta": "graph_trial_existing_loop_residual_delta",
    "abs_graph_trial_odom_residual_delta": "graph_trial_odom_residual_delta",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize whether loop evidence signals separate bad-Z-after from corrected-Z cases."
    )
    parser.add_argument("--labeled_csv", required=True, help="loop_candidates_labeled.csv from n3mapping_loop_debug_analyze.py")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--signals",
        default="",
        help="Optional comma-separated signal list. Defaults to known runtime evidence fields.",
    )
    return parser.parse_args()


def parse_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes"}


def finite_float(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return parsed if math.isfinite(parsed) else float("nan")


def mean(values):
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else float("nan")


def median(values):
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return float("nan")
    mid = len(finite) // 2
    if len(finite) % 2:
        return finite[mid]
    return 0.5 * (finite[mid - 1] + finite[mid])


def min_or_nan(values):
    finite = [value for value in values if math.isfinite(value)]
    return min(finite) if finite else float("nan")


def max_or_nan(values):
    finite = [value for value in values if math.isfinite(value)]
    return max(finite) if finite else float("nan")


def auc_bad_greater(bad_values, corrected_values):
    bad = [value for value in bad_values if math.isfinite(value)]
    corrected = [value for value in corrected_values if math.isfinite(value)]
    if not bad or not corrected:
        return float("nan")
    wins = 0.0
    total = 0
    for b in bad:
        for c in corrected:
            if b > c:
                wins += 1.0
            elif b == c:
                wins += 0.5
            total += 1
    return wins / total if total else float("nan")


def threshold_errors(bad_values, corrected_values, higher_bad):
    bad = [value for value in bad_values if math.isfinite(value)]
    corrected = [value for value in corrected_values if math.isfinite(value)]
    if not bad or not corrected:
        return float("nan"), 0, 0
    threshold = 0.5 * (mean(bad) + mean(corrected))
    if higher_bad:
        false_positive = sum(1 for value in corrected if value >= threshold)
        false_negative = sum(1 for value in bad if value < threshold)
    else:
        false_positive = sum(1 for value in corrected if value <= threshold)
        false_negative = sum(1 for value in bad if value > threshold)
    return threshold, false_positive, false_negative


def overlap_count(bad_values, corrected_values):
    bad = [value for value in bad_values if math.isfinite(value)]
    corrected = [value for value in corrected_values if math.isfinite(value)]
    if not bad or not corrected:
        return 0
    bad_min, bad_max = min(bad), max(bad)
    corr_min, corr_max = min(corrected), max(corrected)
    bad_inside_corrected = sum(1 for value in bad if corr_min <= value <= corr_max)
    corrected_inside_bad = sum(1 for value in corrected if bad_min <= value <= bad_max)
    return bad_inside_corrected + corrected_inside_bad


def json_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def row_signal_value(row, signal):
    if signal in DERIVED_ABS_SIGNALS:
        return abs(finite_float(row.get(DERIVED_ABS_SIGNALS[signal])))
    return finite_float(row.get(signal))


def collect_rows(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = set(reader.fieldnames or [])
    return rows, fieldnames


def analyze_signal(rows, signal):
    bad_values = []
    corrected_values = []
    for row in rows:
        if not parse_bool(row.get("candidate_accepted")) or not parse_bool(row.get("gt_is_loop")):
            continue
        value = row_signal_value(row, signal)
        if not math.isfinite(value):
            continue
        if parse_bool(row.get("z_after_bad")):
            bad_values.append(value)
        elif parse_bool(row.get("z_corrected")):
            corrected_values.append(value)

    auc = auc_bad_greater(bad_values, corrected_values)
    higher_bad = bool(math.isfinite(auc) and auc >= 0.5)
    oriented_auc = max(auc, 1.0 - auc) if math.isfinite(auc) else float("nan")
    threshold, false_positive, false_negative = threshold_errors(bad_values, corrected_values, higher_bad)
    return {
        "signal_name": signal,
        "sample_count": len(bad_values) + len(corrected_values),
        "bad_z_after_count": len(bad_values),
        "corrected_z_count": len(corrected_values),
        "bad_z_after_mean": mean(bad_values),
        "corrected_z_mean": mean(corrected_values),
        "bad_z_after_median": median(bad_values),
        "corrected_z_median": median(corrected_values),
        "bad_z_after_min": min_or_nan(bad_values),
        "bad_z_after_max": max_or_nan(bad_values),
        "corrected_z_min": min_or_nan(corrected_values),
        "corrected_z_max": max_or_nan(corrected_values),
        "auc_bad_greater": auc,
        "auc_like_score": oriented_auc,
        "direction": "higher_bad" if higher_bad else "lower_bad",
        "threshold_at_mean_midpoint": threshold,
        "overlap_count": overlap_count(bad_values, corrected_values),
        "false_positive_if_thresholded": false_positive,
        "false_negative_if_thresholded": false_negative,
    }


def main():
    args = parse_args()
    rows, fieldnames = collect_rows(args.labeled_csv)
    requested = [s.strip() for s in args.signals.split(",") if s.strip()]
    signals = requested if requested else DEFAULT_SIGNALS + sorted(DERIVED_ABS_SIGNALS.keys())
    available = []
    for signal in signals:
        if signal in DERIVED_ABS_SIGNALS or signal in fieldnames:
            available.append(signal)

    results = [analyze_signal(rows, signal) for signal in available]
    results.sort(
        key=lambda row: (
            0 if math.isfinite(row["auc_like_score"]) else 1,
            -row["auc_like_score"] if math.isfinite(row["auc_like_score"]) else 0.0,
            row["overlap_count"],
            row["signal_name"],
        )
    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "loop_evidence_correlation.csv"
    fieldnames_out = [
        "signal_name",
        "sample_count",
        "bad_z_after_count",
        "corrected_z_count",
        "bad_z_after_mean",
        "corrected_z_mean",
        "bad_z_after_median",
        "corrected_z_median",
        "bad_z_after_min",
        "bad_z_after_max",
        "corrected_z_min",
        "corrected_z_max",
        "auc_bad_greater",
        "auc_like_score",
        "direction",
        "threshold_at_mean_midpoint",
        "overlap_count",
        "false_positive_if_thresholded",
        "false_negative_if_thresholded",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_out)
        writer.writeheader()
        writer.writerows(results)

    json_path = output / "loop_evidence_correlation.json"
    payload = {
        "labeled_csv": str(args.labeled_csv),
        "signal_count": len(results),
        "bad_z_after_definition": "candidate_accepted && gt_is_loop && z_after_bad",
        "corrected_z_definition": "candidate_accepted && gt_is_loop && z_corrected",
        "top_signals": [
            {key: json_value(value) for key, value in row.items()}
            for row in results[:10]
        ],
        "signals": [
            {key: json_value(value) for key, value in row.items()}
            for row in results
        ],
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"correlated signals={len(results)} output={output}")


if __name__ == "__main__":
    main()
