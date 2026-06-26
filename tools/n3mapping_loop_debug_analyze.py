#!/usr/bin/env python3
"""Label n3mapping loop_debug.jsonl candidates with KITTI360 ground truth."""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze loop_debug.jsonl against keyframe ground-truth poses."
    )
    parser.add_argument("--loop_debug", required=True, help="Path to loop_debug.jsonl")
    parser.add_argument("--keyframes_gt", required=True, help="Path to keyframes_gt.csv")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--accepted_loops", default="", help="Optional accepted_loops.csv")
    parser.add_argument("--loop_translation_threshold", type=float, default=5.0)
    parser.add_argument("--loop_yaw_threshold_deg", type=float, default=45.0)
    parser.add_argument("--min_id_gap", type=int, default=20)
    parser.add_argument("--z_drift_threshold", type=float, default=0.5)
    parser.add_argument("--rpy_drift_threshold_deg", type=float, default=10.0)
    return parser.parse_args()


def yaw_from_quat(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_quat(qx, qy, qz, qw):
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 0.0 or not math.isfinite(norm):
        raise RuntimeError("invalid quaternion norm")
    return qx / norm, qy / norm, qz / norm, qw / norm


def quat_to_rot(qx, qy, qz, qw):
    qx, qy, qz, qw = normalize_quat(qx, qy, qz, qw)
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz
    return [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ]


def mat_transpose(m):
    return [[m[j][i] for j in range(3)] for i in range(3)]


def mat_mul(a, b):
    return [
        [sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
        for i in range(3)
    ]


def mat_vec_mul(a, v):
    return [sum(a[i][k] * v[k] for k in range(3)) for i in range(3)]


def rpy_from_rot(r):
    # Matches the conventional xyz fixed-axis approximation used for diagnostics.
    pitch = math.asin(max(-1.0, min(1.0, -r[2][0])))
    cp = math.cos(pitch)
    if abs(cp) > 1e-9:
        roll = math.atan2(r[2][1], r[2][2])
        yaw = math.atan2(r[1][0], r[0][0])
    else:
        roll = math.atan2(-r[1][2], r[1][1])
        yaw = 0.0
    return roll, pitch, yaw


def angle_diff(a, b):
    diff = a - b
    while diff > math.pi:
        diff -= 2.0 * math.pi
    while diff < -math.pi:
        diff += 2.0 * math.pi
    return diff


def finite_float(value, default=float("nan")):
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def mean_finite(values):
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return float("nan")
    return sum(finite) / len(finite)


def min_finite(values):
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return float("nan")
    return min(finite)


def load_keyframes_gt(path):
    poses = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"keyframe_id", "frame_id", "x", "y", "z", "qx", "qy", "qz", "qw"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise RuntimeError(f"{path} is missing required keyframe GT columns")
        for row in reader:
            keyframe_id = int(row["keyframe_id"])
            x = finite_float(row["x"])
            y = finite_float(row["y"])
            z = finite_float(row["z"])
            qx = finite_float(row["qx"])
            qy = finite_float(row["qy"])
            qz = finite_float(row["qz"])
            qw = finite_float(row["qw"])
            if not all(math.isfinite(v) for v in (x, y, z, qx, qy, qz, qw)):
                raise RuntimeError(f"{path} contains non-finite pose for keyframe {keyframe_id}")
            poses[keyframe_id] = {
                "frame_id": int(row["frame_id"]),
                "x": x,
                "y": y,
                "z": z,
                "qx": qx,
                "qy": qy,
                "qz": qz,
                "qw": qw,
                "yaw": yaw_from_quat(qx, qy, qz, qw),
                "rot": quat_to_rot(qx, qy, qz, qw),
            }
    if not poses:
        raise RuntimeError(f"{path} contains no keyframe ground-truth poses")
    return poses


def load_loop_candidates(path):
    candidates = []
    with open(path) as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
            if event.get("record_type") != "candidate":
                continue
            if "query_id" not in event or "match_id" not in event:
                raise RuntimeError(f"{path}:{line_number}: candidate is missing query_id or match_id")
            candidates.append(event)
    return candidates


def load_optimization_summaries(path):
    summaries = []
    with open(path) as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
            if event.get("record_type") != "optimization_summary":
                continue
            summaries.append(event)
    return summaries


def load_accepted_pairs(path):
    pairs = set()
    if not path:
        return pairs
    if not os.path.exists(path):
        raise RuntimeError(f"accepted loops file does not exist: {path}")
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"query_id", "match_id"}.issubset(set(reader.fieldnames)):
            raise RuntimeError(f"{path} is missing query_id/match_id columns")
        for row in reader:
            pairs.add((int(row["query_id"]), int(row["match_id"])))
    return pairs


def gt_pair_metrics(poses, query_id, match_id):
    query = poses.get(query_id)
    match = poses.get(match_id)
    if query is None or match is None:
        return float("nan"), float("nan"), False
    dx = query["x"] - match["x"]
    dy = query["y"] - match["y"]
    dz = query["z"] - match["z"]
    translation = math.sqrt(dx * dx + dy * dy + dz * dz)
    yaw_deg = abs(angle_diff(query["yaw"], match["yaw"])) * 180.0 / math.pi
    return translation, yaw_deg, True


def gt_relative_axes(poses, query_id, match_id):
    query = poses.get(query_id)
    match = poses.get(match_id)
    if query is None or match is None:
        return None
    r_match_t = mat_transpose(match["rot"])
    dt_world = [
        query["x"] - match["x"],
        query["y"] - match["y"],
        query["z"] - match["z"],
    ]
    translation = mat_vec_mul(r_match_t, dt_world)
    rel_rot = mat_mul(r_match_t, query["rot"])
    roll, pitch, yaw = rpy_from_rot(rel_rot)
    return {
        "x": translation[0],
        "y": translation[1],
        "z": translation[2],
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
    }


def measurement_axes(event):
    keys = ("x", "y", "z", "roll", "pitch", "yaw")
    values = {key: event_float(event, f"measurement_{key}") for key in keys}
    if not all(math.isfinite(values[key]) for key in keys):
        return None
    return values


def axis_errors(measured, gt_axes):
    if measured is None or gt_axes is None:
        return {
            "x": float("nan"),
            "y": float("nan"),
            "z": float("nan"),
            "roll": float("nan"),
            "pitch": float("nan"),
            "yaw": float("nan"),
        }
    return {
        "x": measured["x"] - gt_axes["x"],
        "y": measured["y"] - gt_axes["y"],
        "z": measured["z"] - gt_axes["z"],
        "roll": angle_diff(measured["roll"], gt_axes["roll"]),
        "pitch": angle_diff(measured["pitch"], gt_axes["pitch"]),
        "yaw": angle_diff(measured["yaw"], gt_axes["yaw"]),
    }


def is_gt_loop(translation, yaw_deg, translation_threshold, yaw_threshold_deg):
    return (
        math.isfinite(translation)
        and math.isfinite(yaw_deg)
        and translation <= translation_threshold
        and yaw_deg <= yaw_threshold_deg
    )


def gt_heading_class(translation, yaw_deg, translation_threshold, yaw_threshold_deg):
    if not math.isfinite(translation) or not math.isfinite(yaw_deg):
        return "unknown"
    if translation > translation_threshold:
        return "far"
    if yaw_deg <= yaw_threshold_deg:
        return "same_heading"
    if yaw_deg >= 180.0 - yaw_threshold_deg:
        return "opposite_heading"
    return "cross_heading"


def enumerate_gt_loop_pairs(poses, translation_threshold, yaw_threshold_deg, min_id_gap):
    ids = sorted(poses.keys())
    pairs = set()
    for i, match_id in enumerate(ids):
        for query_id in ids[i + 1 :]:
            if abs(query_id - match_id) < min_id_gap:
                continue
            translation, yaw_deg, valid = gt_pair_metrics(poses, query_id, match_id)
            if valid and is_gt_loop(translation, yaw_deg, translation_threshold, yaw_threshold_deg):
                pairs.add((query_id, match_id))
    return pairs


def enumerate_gt_position_pairs(poses, translation_threshold, yaw_threshold_deg, min_id_gap):
    ids = sorted(poses.keys())
    pairs = set()
    for i, match_id in enumerate(ids):
        for query_id in ids[i + 1 :]:
            if abs(query_id - match_id) < min_id_gap:
                continue
            translation, yaw_deg, valid = gt_pair_metrics(poses, query_id, match_id)
            if not valid:
                continue
            heading_class = gt_heading_class(translation, yaw_deg, translation_threshold, yaw_threshold_deg)
            if heading_class in ("same_heading", "opposite_heading", "cross_heading"):
                pairs.add((query_id, match_id))
    return pairs


def increment_nested_counter(container, *keys):
    current = container
    for key in keys[:-1]:
        current = current.setdefault(str(key), {})
    leaf = str(keys[-1])
    current[leaf] = current.get(leaf, 0) + 1


def event_float(event, key):
    return finite_float(event.get(key))


def format_float(value):
    if math.isfinite(value):
        return f"{value:.9g}"
    return ""


def best_by_fitness(rows):
    finite_rows = [r for r in rows if math.isfinite(r["fitness_score"])]
    if finite_rows:
        return min(finite_rows, key=lambda r: r["fitness_score"])
    return rows[0] if rows else None


def exceeds_abs(value, threshold):
    return math.isfinite(value) and abs(value) >= threshold


def build_optimization_after_map(optimization_summaries):
    residuals = {}
    for event in optimization_summaries:
        accepted_edges = event.get("accepted_edges") or []
        for edge in accepted_edges:
            from_id = edge.get("from_id", "")
            to_id = edge.get("to_id", "")
            if not isinstance(from_id, int) or not isinstance(to_id, int):
                continue
            # n3mapping writes loop edges in match->query direction.
            query_id = to_id
            match_id = from_id
            residuals[(query_id, match_id)] = {
                "z": event_float(event, "loop_residual_z_after"),
                "roll": event_float(event, "loop_residual_roll_after"),
                "pitch": event_float(event, "loop_residual_pitch_after"),
                "yaw": event_float(event, "loop_residual_yaw_after"),
            }
    return residuals


def classify_candidate(accepted,
                       gt_loop,
                       heading_class,
                       candidate_selectable,
                       z_bad,
                       roll_pitch_bad):
    if accepted and gt_loop:
        if z_bad:
            return "accepted_true_loop_bad_z"
        if roll_pitch_bad:
            return "accepted_true_loop_bad_roll_pitch"
        return "accepted_true_loop_good"
    if accepted and heading_class == "opposite_heading":
        return "accepted_opposite_heading_loop"
    if accepted and heading_class == "cross_heading":
        return "accepted_cross_heading_loop"
    if accepted and not gt_loop:
        return "accepted_false_loop"
    if gt_loop and candidate_selectable:
        return "true_loop_not_selected"
    if gt_loop:
        return "verification_reject_true_loop"
    if heading_class in ("opposite_heading", "cross_heading"):
        return "position_loop_not_selected" if candidate_selectable else "verification_reject_position_loop"
    return "retrieval_false_positive"


def build_query_summary(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["query_id"], []).append(row)

    summary_rows = []
    for query_id in sorted(grouped):
        query_rows = grouped[query_id]
        accepted_rows = [r for r in query_rows if r["candidate_accepted"]]
        true_rows = [r for r in query_rows if r["gt_is_loop"]]
        position_rows = [r for r in query_rows if r.get("gt_position_loop")]
        selectable_true_rows = [r for r in true_rows if r["candidate_selectable"]]
        selectable_position_rows = [r for r in position_rows if r["candidate_selectable"]]
        accepted = accepted_rows[0] if accepted_rows else None
        best_true = best_by_fitness(true_rows)
        best_selectable_true = best_by_fitness(selectable_true_rows)
        best_candidate = best_by_fitness(query_rows)
        selection_failure = bool(selectable_true_rows and accepted is not None and not accepted["gt_is_loop"])
        missed_true_candidate = bool(true_rows and not selectable_true_rows)
        position_selection_failure = bool(
            selectable_position_rows and accepted is not None and not accepted.get("gt_position_loop")
        )
        missed_position_candidate = bool(position_rows and not selectable_position_rows)

        summary_rows.append(
            {
                "query_id": query_id,
                "candidate_count": len(query_rows),
                "true_candidate_count": len(true_rows),
                "position_candidate_count": len(position_rows),
                "true_selectable_candidate_count": len(selectable_true_rows),
                "position_selectable_candidate_count": len(selectable_position_rows),
                "accepted_match_id": accepted["match_id"] if accepted else -1,
                "accepted_is_gt_loop": accepted["gt_is_loop"] if accepted else False,
                "accepted_is_position_loop": accepted.get("gt_position_loop", False) if accepted else False,
                "accepted_gt_translation_m": accepted["gt_query_match_translation_m"] if accepted else float("nan"),
                "accepted_gt_yaw_deg": accepted["gt_query_match_yaw_deg"] if accepted else float("nan"),
                "accepted_fitness_score": accepted["fitness_score"] if accepted else float("nan"),
                "accepted_inlier_ratio": accepted["inlier_ratio"] if accepted else float("nan"),
                "accepted_residual_z": accepted["residual_z"] if accepted else float("nan"),
                "best_true_match_id": best_true["match_id"] if best_true else -1,
                "best_true_fitness_score": best_true["fitness_score"] if best_true else float("nan"),
                "best_true_inlier_ratio": best_true["inlier_ratio"] if best_true else float("nan"),
                "best_true_gt_translation_m": best_true["gt_query_match_translation_m"] if best_true else float("nan"),
                "best_true_residual_z": best_true["residual_z"] if best_true else float("nan"),
                "best_selectable_true_match_id": best_selectable_true["match_id"] if best_selectable_true else -1,
                "best_selectable_true_fitness_score": best_selectable_true["fitness_score"] if best_selectable_true else float("nan"),
                "best_selectable_true_gt_translation_m": best_selectable_true["gt_query_match_translation_m"] if best_selectable_true else float("nan"),
                "best_selectable_true_residual_z": best_selectable_true["residual_z"] if best_selectable_true else float("nan"),
                "best_candidate_match_id": best_candidate["match_id"] if best_candidate else -1,
                "best_candidate_is_gt_loop": best_candidate["gt_is_loop"] if best_candidate else False,
                "best_candidate_fitness_score": best_candidate["fitness_score"] if best_candidate else float("nan"),
                "selection_failure": selection_failure,
                "missed_true_candidate": missed_true_candidate,
                "position_selection_failure": position_selection_failure,
                "missed_position_candidate": missed_position_candidate,
            }
        )
    return summary_rows


def analyze(args):
    poses = load_keyframes_gt(args.keyframes_gt)
    candidates = load_loop_candidates(args.loop_debug)
    optimization_summaries = load_optimization_summaries(args.loop_debug)
    optimization_after_by_pair = build_optimization_after_map(optimization_summaries)
    accepted_pairs = load_accepted_pairs(args.accepted_loops)
    candidate_pairs = {(int(c["query_id"]), int(c["match_id"])) for c in candidates}
    gt_loop_pairs = enumerate_gt_loop_pairs(
        poses,
        args.loop_translation_threshold,
        args.loop_yaw_threshold_deg,
        args.min_id_gap,
    )
    gt_position_pairs = enumerate_gt_position_pairs(
        poses,
        args.loop_translation_threshold,
        args.loop_yaw_threshold_deg,
        args.min_id_gap,
    )

    rows = []
    stats = {
        "candidate_count": 0,
        "candidate_with_gt_count": 0,
        "accepted_candidate_count": 0,
        "retrieval_true_positive": 0,
        "retrieval_position_positive": 0,
        "retrieval_false_positive": 0,
        "retrieval_miss_estimate": 0,
        "icp_reject_true_loop": 0,
        "true_loop_not_selected": 0,
        "position_loop_not_selected": 0,
        "accepted_false_loop": 0,
        "accepted_far_false_loop": 0,
        "accepted_opposite_heading_loop": 0,
        "accepted_cross_heading_loop": 0,
        "accepted_position_loop": 0,
        "accepted_true_loop": 0,
        "accepted_true_loop_good": 0,
        "accepted_true_loop_bad_z": 0,
        "accepted_true_loop_bad_z_measurement": 0,
        "accepted_true_loop_bad_z_after": 0,
        "accepted_true_loop_corrected_z": 0,
        "accepted_true_loop_bad_roll_pitch": 0,
        "verification_reject_true_loop": 0,
        "verification_reject_position_loop": 0,
        "z_drift_suspect_count": 0,
        "optimization_summary_count": 0,
        "optimization_high_residual_z_after_count": 0,
        "optimization_max_residual_z_after": 0.0,
        "failure_class_counts": {},
        "reject_reason_counts": {},
        "reject_reason_by_source": {},
        "reject_reason_by_heading_class": {},
        "reject_reason_by_source_heading": {},
        "edge_mode_counts": {},
        "accepted_full6dof": 0,
        "accepted_planar_xy_yaw": 0,
        "accepted_vertical_neutral": 0,
        "vertical_hypothesis_candidate_count": 0,
        "vertical_hypothesis_planar_recommendation_count": 0,
        "vertical_hypothesis_full6dof_recommendation_count": 0,
        "accepted_true_loop_bad_z_with_vertical_hypothesis": 0,
        "accepted_true_loop_bad_z_planar_recommended": 0,
        "heightmap_candidate_count": 0,
        "accepted_true_loop_bad_z_heightmap_high": 0,
        "heightmap_separates_bad_z_count": 0,
        "graph_trial_candidate_count": 0,
        "graph_trial_success_count": 0,
        "segment_candidate_count": 0,
        "segment_consistent_count": 0,
        "segment_inconsistent_count": 0,
        "segment_insufficient_count": 0,
        "accepted_true_loop_bad_z_after_graph_trial_score_mean": float("nan"),
        "accepted_true_loop_bad_z_after_graph_trial_score_min": float("nan"),
        "accepted_true_loop_corrected_z_graph_trial_score_mean": float("nan"),
        "accepted_true_loop_corrected_z_graph_trial_score_min": float("nan"),
        "accepted_true_loop_bad_z_after_segment_consensus_ratio_mean": float("nan"),
        "accepted_true_loop_bad_z_after_segment_translation_median_mean": float("nan"),
        "accepted_true_loop_corrected_z_segment_consensus_ratio_mean": float("nan"),
        "accepted_true_loop_corrected_z_segment_translation_median_mean": float("nan"),
    }
    bad_z_after_graph_trial_scores = []
    corrected_z_graph_trial_scores = []
    bad_z_after_segment_ratios = []
    bad_z_after_segment_translations = []
    corrected_z_segment_ratios = []
    corrected_z_segment_translations = []
    rpy_threshold_rad = args.rpy_drift_threshold_deg * math.pi / 180.0
    accepted_pairs_available = bool(args.accepted_loops)

    for event in candidates:
        query_id = int(event["query_id"])
        match_id = int(event["match_id"])
        translation, yaw_deg, has_gt = gt_pair_metrics(poses, query_id, match_id)
        heading_class = gt_heading_class(
            translation,
            yaw_deg,
            args.loop_translation_threshold,
            args.loop_yaw_threshold_deg,
        )
        position_loop = heading_class in ("same_heading", "opposite_heading", "cross_heading")
        gt_loop = is_gt_loop(
            translation,
            yaw_deg,
            args.loop_translation_threshold,
            args.loop_yaw_threshold_deg,
        )
        if accepted_pairs_available:
            accepted = (query_id, match_id) in accepted_pairs
        else:
            accepted = event.get("gate_result") == "accepted"
        candidate_selectable = accepted or event.get("reject_reason", "") == "not_selected"

        residual_z = event_float(event, "residual_z")
        residual_roll = event_float(event, "residual_roll")
        residual_pitch = event_float(event, "residual_pitch")
        residual_yaw = event_float(event, "residual_yaw")
        gt_axes = gt_relative_axes(poses, query_id, match_id)
        measured_axes = measurement_axes(event)
        errors = axis_errors(measured_axes, gt_axes)
        after_residual = optimization_after_by_pair.get(
            (query_id, match_id),
            {
                "z": float("nan"),
                "roll": float("nan"),
                "pitch": float("nan"),
                "yaw": float("nan"),
            },
        )
        residual_z_after = after_residual["z"]
        residual_roll_after = after_residual["roll"]
        residual_pitch_after = after_residual["pitch"]
        residual_yaw_after = after_residual["yaw"]
        edge_mode = str(event.get("edge_mode", "unknown") or "unknown")
        vertical_observability_score = event_float(event, "vertical_observability_score")
        vertical_downweighted = bool(event.get("vertical_downweighted", False))
        source_z_span = event_float(event, "source_z_span")
        target_z_span = event_float(event, "target_z_span")
        z_overlap_ratio_before = event_float(event, "z_overlap_ratio_before")
        z_overlap_ratio_after = event_float(event, "z_overlap_ratio_after")
        source_z_robust_span = event_float(event, "source_z_robust_span")
        target_z_robust_span = event_float(event, "target_z_robust_span")
        z_robust_overlap_ratio_before = event_float(event, "z_robust_overlap_ratio_before")
        z_robust_overlap_ratio_after = event_float(event, "z_robust_overlap_ratio_after")
        source_target_z_centroid_delta_before = event_float(
            event, "source_target_z_centroid_delta_before"
        )
        source_target_z_centroid_delta_after = event_float(
            event, "source_target_z_centroid_delta_after"
        )
        vertical_information_ratio = event_float(event, "vertical_information_ratio")
        vertical_hypothesis_count = int(event.get("vertical_hypothesis_count") or 0)
        best_z_offset_m = event_float(event, "best_z_offset_m")
        best_z_offset_fitness = event_float(event, "best_z_offset_fitness")
        zero_z_fitness = event_float(event, "zero_z_fitness")
        fitness_gap_zero_vs_best = event_float(event, "fitness_gap_zero_vs_best")
        z_hypothesis_spread_m = event_float(event, "z_hypothesis_spread_m")
        vertical_ambiguity_score = event_float(event, "vertical_ambiguity_score")
        vertical_hypothesis_edge_recommendation = str(
            event.get("vertical_hypothesis_edge_recommendation", "not_available") or "not_available"
        )
        heightmap_overlap_cell_count = int(event.get("heightmap_overlap_cell_count") or 0)
        heightmap_overlap_ratio = event_float(event, "heightmap_overlap_ratio")
        heightmap_ground_dz_median = event_float(event, "heightmap_ground_dz_median")
        heightmap_ground_dz_p90 = event_float(event, "heightmap_ground_dz_p90")
        heightmap_ground_dz_max = event_float(event, "heightmap_ground_dz_max")
        heightmap_ground_support_ratio = event_float(event, "heightmap_ground_support_ratio")
        heightmap_vertical_consistency_score = event_float(
            event, "heightmap_vertical_consistency_score"
        )
        graph_trial_success = bool(event.get("graph_trial_success", False))
        graph_trial_residual_x_after = event_float(event, "graph_trial_residual_x_after")
        graph_trial_residual_y_after = event_float(event, "graph_trial_residual_y_after")
        graph_trial_residual_z_after = event_float(event, "graph_trial_residual_z_after")
        graph_trial_residual_roll_after = event_float(event, "graph_trial_residual_roll_after")
        graph_trial_residual_pitch_after = event_float(event, "graph_trial_residual_pitch_after")
        graph_trial_residual_yaw_after = event_float(event, "graph_trial_residual_yaw_after")
        graph_trial_residual_translation_norm_after = event_float(
            event, "graph_trial_residual_translation_norm_after"
        )
        graph_trial_residual_rotation_norm_after = event_float(
            event, "graph_trial_residual_rotation_norm_after"
        )
        graph_trial_mean_pose_update_translation = event_float(
            event, "graph_trial_mean_pose_update_translation"
        )
        graph_trial_max_pose_update_translation = event_float(
            event, "graph_trial_max_pose_update_translation"
        )
        graph_trial_mean_pose_update_rotation = event_float(
            event, "graph_trial_mean_pose_update_rotation"
        )
        graph_trial_max_pose_update_rotation = event_float(
            event, "graph_trial_max_pose_update_rotation"
        )
        graph_trial_existing_loop_residual_delta = event_float(
            event, "graph_trial_existing_loop_residual_delta"
        )
        graph_trial_odom_residual_delta = event_float(event, "graph_trial_odom_residual_delta")
        graph_trial_consistency_score = event_float(event, "graph_trial_consistency_score")
        graph_trial_recommendation = str(
            event.get("graph_trial_recommendation", "not_available") or "not_available"
        )
        segment_pair_count = int(event.get("segment_pair_count") or 0)
        segment_valid_pair_count = int(event.get("segment_valid_pair_count") or 0)
        segment_consensus_inlier_count = int(event.get("segment_consensus_inlier_count") or 0)
        segment_consensus_ratio = event_float(event, "segment_consensus_ratio")
        segment_translation_median = event_float(event, "segment_translation_median")
        segment_translation_std = event_float(event, "segment_translation_std")
        segment_yaw_median = event_float(event, "segment_yaw_median")
        segment_yaw_std = event_float(event, "segment_yaw_std")
        segment_z_std = event_float(event, "segment_z_std")
        segment_roll_pitch_std = event_float(event, "segment_roll_pitch_std")
        segment_direction = str(event.get("segment_direction", "not_available") or "not_available")
        segment_recommendation = str(event.get("segment_recommendation", "not_available") or "not_available")
        candidate_source = str(event.get("candidate_source", "") or "")
        reject_reason = str(event.get("reject_reason", "") or "")
        icp_iterations = int(event.get("icp_iterations") or 0)
        icp_optimizer_error = event_float(event, "icp_optimizer_error")
        icp_termination = str(event.get("icp_termination", "invalid") or "invalid")
        transform_debug_axes = {}
        for prefix in ("pred_match_query", "icp_correction_match", "measured_match_query"):
            for axis in ("x", "y", "z", "roll", "pitch", "yaw"):
                transform_debug_axes[f"{prefix}_{axis}"] = event_float(event, f"{prefix}_{axis}")
        loop_referee_recommendation = str(
            event.get("loop_referee_recommendation", "not_available") or "not_available"
        )
        loop_referee_reason = str(
            event.get("loop_referee_reason", "not_available") or "not_available"
        )
        loop_referee_risk_flags = str(
            event.get("loop_referee_risk_flags", "not_available") or "not_available"
        )
        heightmap_high = bool(
            heightmap_overlap_cell_count > 0
            and math.isfinite(heightmap_ground_dz_p90)
            and heightmap_ground_dz_p90 >= args.z_drift_threshold
        )
        z_measurement_residual_large = bool(
            accepted and gt_loop and exceeds_abs(residual_z, args.z_drift_threshold)
        )
        z_measurement_bad = bool(
            accepted and gt_loop and exceeds_abs(errors["z"], args.z_drift_threshold)
        )
        z_after_bad = bool(
            accepted and gt_loop and exceeds_abs(residual_z_after, args.z_drift_threshold)
        )
        z_corrected = bool(z_measurement_residual_large and not z_after_bad)
        z_drift_suspect = bool(
            accepted
            and (
                (math.isfinite(residual_z) and abs(residual_z) >= args.z_drift_threshold)
                or (math.isfinite(residual_z_after) and abs(residual_z_after) >= args.z_drift_threshold)
                or (math.isfinite(residual_roll) and abs(residual_roll) >= rpy_threshold_rad)
                or (math.isfinite(residual_pitch) and abs(residual_pitch) >= rpy_threshold_rad)
                or (math.isfinite(residual_roll_after) and abs(residual_roll_after) >= rpy_threshold_rad)
                or (math.isfinite(residual_pitch_after) and abs(residual_pitch_after) >= rpy_threshold_rad)
            )
        )
        z_bad = bool(
            accepted
            and gt_loop
            and (
                z_measurement_bad
                or z_after_bad
                or z_measurement_residual_large
            )
        )
        roll_pitch_bad = bool(
            accepted
            and gt_loop
            and (
                exceeds_abs(errors["roll"], rpy_threshold_rad)
                or exceeds_abs(errors["pitch"], rpy_threshold_rad)
                or exceeds_abs(residual_roll_after, rpy_threshold_rad)
                or exceeds_abs(residual_pitch_after, rpy_threshold_rad)
                or exceeds_abs(residual_roll, rpy_threshold_rad)
                or exceeds_abs(residual_pitch, rpy_threshold_rad)
            )
        )
        failure_class = classify_candidate(
            accepted,
            gt_loop,
            heading_class,
            candidate_selectable,
            z_bad,
            roll_pitch_bad,
        )

        if accepted and gt_loop:
            category = "accepted_true_loop"
        elif accepted and heading_class == "opposite_heading":
            category = "accepted_opposite_heading_loop"
        elif accepted and heading_class == "cross_heading":
            category = "accepted_cross_heading_loop"
        elif accepted:
            category = "accepted_false_loop"
        elif gt_loop:
            category = "rejected_true_loop"
        elif position_loop:
            category = "rejected_position_loop"
        else:
            category = "rejected_false_candidate"

        stats["candidate_count"] += 1
        if has_gt:
            stats["candidate_with_gt_count"] += 1
        if accepted:
            stats["accepted_candidate_count"] += 1
            stats["edge_mode_counts"][edge_mode] = stats["edge_mode_counts"].get(edge_mode, 0) + 1
            if edge_mode == "full6dof":
                stats["accepted_full6dof"] += 1
            elif edge_mode == "planar_xy_yaw":
                stats["accepted_planar_xy_yaw"] += 1
            elif edge_mode == "vertical_neutral":
                stats["accepted_vertical_neutral"] += 1
        if vertical_hypothesis_count > 0:
            stats["vertical_hypothesis_candidate_count"] += 1
            if vertical_hypothesis_edge_recommendation == "planar_xy_yaw":
                stats["vertical_hypothesis_planar_recommendation_count"] += 1
            elif vertical_hypothesis_edge_recommendation == "full6dof":
                stats["vertical_hypothesis_full6dof_recommendation_count"] += 1
        if heightmap_overlap_cell_count > 0:
            stats["heightmap_candidate_count"] += 1
        if graph_trial_recommendation != "not_available":
            stats["graph_trial_candidate_count"] += 1
            if graph_trial_success:
                stats["graph_trial_success_count"] += 1
        if segment_recommendation != "not_available":
            stats["segment_candidate_count"] += 1
            if segment_recommendation == "consistent":
                stats["segment_consistent_count"] += 1
            elif segment_recommendation == "inconsistent":
                stats["segment_inconsistent_count"] += 1
            elif segment_recommendation == "insufficient_support":
                stats["segment_insufficient_count"] += 1
        if position_loop:
            stats["retrieval_position_positive"] += 1
        if gt_loop:
            stats["retrieval_true_positive"] += 1
        elif has_gt and not position_loop:
            stats["retrieval_false_positive"] += 1
        if gt_loop and not accepted and candidate_selectable:
            stats["true_loop_not_selected"] += 1
        if gt_loop and not accepted and not candidate_selectable:
            stats["icp_reject_true_loop"] += 1
            stats["verification_reject_true_loop"] += 1
        if position_loop and not gt_loop and not accepted and candidate_selectable:
            stats["position_loop_not_selected"] += 1
        if position_loop and not gt_loop and not accepted and not candidate_selectable:
            stats["verification_reject_position_loop"] += 1
        if accepted and not position_loop:
            stats["accepted_false_loop"] += 1
            stats["accepted_far_false_loop"] += 1
        if accepted and heading_class == "opposite_heading":
            stats["accepted_opposite_heading_loop"] += 1
        if accepted and heading_class == "cross_heading":
            stats["accepted_cross_heading_loop"] += 1
        if accepted and position_loop:
            stats["accepted_position_loop"] += 1
        if accepted and gt_loop:
            stats["accepted_true_loop"] += 1
        if failure_class == "accepted_true_loop_good":
            stats["accepted_true_loop_good"] += 1
        elif failure_class == "accepted_true_loop_bad_z":
            stats["accepted_true_loop_bad_z"] += 1
        elif failure_class == "accepted_true_loop_bad_roll_pitch":
            stats["accepted_true_loop_bad_roll_pitch"] += 1
        if z_measurement_bad:
            stats["accepted_true_loop_bad_z_measurement"] += 1
        if z_after_bad:
            stats["accepted_true_loop_bad_z_after"] += 1
        if z_bad and vertical_hypothesis_count > 0:
            stats["accepted_true_loop_bad_z_with_vertical_hypothesis"] += 1
            if vertical_hypothesis_edge_recommendation == "planar_xy_yaw":
                stats["accepted_true_loop_bad_z_planar_recommended"] += 1
        if z_bad and heightmap_high:
            stats["accepted_true_loop_bad_z_heightmap_high"] += 1
        if z_after_bad and heightmap_high:
            stats["heightmap_separates_bad_z_count"] += 1
        if z_corrected:
            stats["accepted_true_loop_corrected_z"] += 1
        if z_after_bad:
            bad_z_after_graph_trial_scores.append(graph_trial_consistency_score)
            bad_z_after_segment_ratios.append(segment_consensus_ratio)
            bad_z_after_segment_translations.append(segment_translation_median)
        if z_corrected:
            corrected_z_graph_trial_scores.append(graph_trial_consistency_score)
            corrected_z_segment_ratios.append(segment_consensus_ratio)
            corrected_z_segment_translations.append(segment_translation_median)
        if z_drift_suspect:
            stats["z_drift_suspect_count"] += 1
        stats["failure_class_counts"][failure_class] = stats["failure_class_counts"].get(failure_class, 0) + 1
        if not accepted:
            increment_nested_counter(stats["reject_reason_counts"], reject_reason or "none")
            increment_nested_counter(stats["reject_reason_by_source"], candidate_source or "unknown", reject_reason or "none")
            increment_nested_counter(stats["reject_reason_by_heading_class"], heading_class, reject_reason or "none")
            increment_nested_counter(
                stats["reject_reason_by_source_heading"],
                candidate_source or "unknown",
                heading_class,
                reject_reason or "none",
            )

        rows.append(
            {
                "query_id": query_id,
                "match_id": match_id,
                "gt_query_match_translation_m": translation,
                "gt_query_match_yaw_deg": yaw_deg,
                "gt_distance_m": translation,
                "gt_yaw_diff_deg": yaw_deg,
                "gt_is_loop": gt_loop,
                "gt_pose_loop": gt_loop,
                "gt_position_loop": position_loop,
                "gt_place_loop": position_loop,
                "gt_heading_class": heading_class,
                "gt_relative_x": gt_axes["x"] if gt_axes is not None else float("nan"),
                "gt_relative_y": gt_axes["y"] if gt_axes is not None else float("nan"),
                "gt_relative_z": gt_axes["z"] if gt_axes is not None else float("nan"),
                "gt_relative_roll": gt_axes["roll"] if gt_axes is not None else float("nan"),
                "gt_relative_pitch": gt_axes["pitch"] if gt_axes is not None else float("nan"),
                "gt_relative_yaw": gt_axes["yaw"] if gt_axes is not None else float("nan"),
                "candidate_accepted": accepted,
                "candidate_selectable": candidate_selectable,
                "candidate_source": candidate_source,
                "edge_mode": edge_mode,
                "vertical_observability_score": vertical_observability_score,
                "vertical_downweighted": vertical_downweighted,
                "source_z_span": source_z_span,
                "target_z_span": target_z_span,
                "z_overlap_ratio_before": z_overlap_ratio_before,
                "z_overlap_ratio_after": z_overlap_ratio_after,
                "source_z_robust_span": source_z_robust_span,
                "target_z_robust_span": target_z_robust_span,
                "z_robust_overlap_ratio_before": z_robust_overlap_ratio_before,
                "z_robust_overlap_ratio_after": z_robust_overlap_ratio_after,
                "source_target_z_centroid_delta_before": source_target_z_centroid_delta_before,
                "source_target_z_centroid_delta_after": source_target_z_centroid_delta_after,
                "vertical_information_ratio": vertical_information_ratio,
                "vertical_hypothesis_count": vertical_hypothesis_count,
                "best_z_offset_m": best_z_offset_m,
                "best_z_offset_fitness": best_z_offset_fitness,
                "zero_z_fitness": zero_z_fitness,
                "fitness_gap_zero_vs_best": fitness_gap_zero_vs_best,
                "z_hypothesis_spread_m": z_hypothesis_spread_m,
                "vertical_ambiguity_score": vertical_ambiguity_score,
                "vertical_hypothesis_edge_recommendation": vertical_hypothesis_edge_recommendation,
                "heightmap_overlap_cell_count": heightmap_overlap_cell_count,
                "heightmap_overlap_ratio": heightmap_overlap_ratio,
                "heightmap_ground_dz_median": heightmap_ground_dz_median,
                "heightmap_ground_dz_p90": heightmap_ground_dz_p90,
                "heightmap_ground_dz_max": heightmap_ground_dz_max,
                "heightmap_ground_support_ratio": heightmap_ground_support_ratio,
                "heightmap_vertical_consistency_score": heightmap_vertical_consistency_score,
                "graph_trial_success": graph_trial_success,
                "graph_trial_residual_x_after": graph_trial_residual_x_after,
                "graph_trial_residual_y_after": graph_trial_residual_y_after,
                "graph_trial_residual_z_after": graph_trial_residual_z_after,
                "graph_trial_residual_roll_after": graph_trial_residual_roll_after,
                "graph_trial_residual_pitch_after": graph_trial_residual_pitch_after,
                "graph_trial_residual_yaw_after": graph_trial_residual_yaw_after,
                "graph_trial_residual_translation_norm_after": graph_trial_residual_translation_norm_after,
                "graph_trial_residual_rotation_norm_after": graph_trial_residual_rotation_norm_after,
                "graph_trial_mean_pose_update_translation": graph_trial_mean_pose_update_translation,
                "graph_trial_max_pose_update_translation": graph_trial_max_pose_update_translation,
                "graph_trial_mean_pose_update_rotation": graph_trial_mean_pose_update_rotation,
                "graph_trial_max_pose_update_rotation": graph_trial_max_pose_update_rotation,
                "graph_trial_existing_loop_residual_delta": graph_trial_existing_loop_residual_delta,
                "graph_trial_odom_residual_delta": graph_trial_odom_residual_delta,
                "graph_trial_consistency_score": graph_trial_consistency_score,
                "graph_trial_recommendation": graph_trial_recommendation,
                "segment_pair_count": segment_pair_count,
                "segment_valid_pair_count": segment_valid_pair_count,
                "segment_consensus_inlier_count": segment_consensus_inlier_count,
                "segment_consensus_ratio": segment_consensus_ratio,
                "segment_translation_median": segment_translation_median,
                "segment_translation_std": segment_translation_std,
                "segment_yaw_median": segment_yaw_median,
                "segment_yaw_std": segment_yaw_std,
                "segment_z_std": segment_z_std,
                "segment_roll_pitch_std": segment_roll_pitch_std,
                "segment_direction": segment_direction,
                "segment_recommendation": segment_recommendation,
                "icp_iterations": icp_iterations,
                "icp_optimizer_error": icp_optimizer_error,
                "icp_termination": icp_termination,
                **transform_debug_axes,
                "loop_referee_recommendation": loop_referee_recommendation,
                "loop_referee_reason": loop_referee_reason,
                "loop_referee_risk_flags": loop_referee_risk_flags,
                "z_measurement_residual_large": z_measurement_residual_large,
                "z_measurement_bad": z_measurement_bad,
                "z_after_bad": z_after_bad,
                "z_corrected": z_corrected,
                "gate_result": event.get("gate_result", ""),
                "reject_reason": reject_reason,
                "fitness_score": event_float(event, "fitness_score"),
                "inlier_ratio": event_float(event, "inlier_ratio"),
                "residual_z": residual_z,
                "residual_roll": residual_roll,
                "residual_pitch": residual_pitch,
                "residual_yaw": residual_yaw,
                "residual_z_after": residual_z_after,
                "residual_roll_after": residual_roll_after,
                "residual_pitch_after": residual_pitch_after,
                "residual_yaw_after": residual_yaw_after,
                "measurement_x": measured_axes["x"] if measured_axes is not None else float("nan"),
                "measurement_y": measured_axes["y"] if measured_axes is not None else float("nan"),
                "measurement_z": measured_axes["z"] if measured_axes is not None else float("nan"),
                "measurement_roll": measured_axes["roll"] if measured_axes is not None else float("nan"),
                "measurement_pitch": measured_axes["pitch"] if measured_axes is not None else float("nan"),
                "measurement_yaw": measured_axes["yaw"] if measured_axes is not None else float("nan"),
                "icp_error_to_gt_x": errors["x"],
                "icp_error_to_gt_y": errors["y"],
                "icp_error_to_gt_z": errors["z"],
                "icp_error_to_gt_roll": errors["roll"],
                "icp_error_to_gt_pitch": errors["pitch"],
                "icp_error_to_gt_yaw": errors["yaw"],
                "z_drift_suspect": z_drift_suspect,
                "category": category,
                "failure_class": failure_class,
            }
        )

    stats["retrieval_miss_estimate"] = len(gt_loop_pairs - candidate_pairs)
    stats["position_retrieval_miss_estimate"] = len(gt_position_pairs - candidate_pairs)
    stats["accepted_true_loop_bad_z_after_graph_trial_score_mean"] = mean_finite(
        bad_z_after_graph_trial_scores
    )
    stats["accepted_true_loop_bad_z_after_graph_trial_score_min"] = min_finite(
        bad_z_after_graph_trial_scores
    )
    stats["accepted_true_loop_corrected_z_graph_trial_score_mean"] = mean_finite(
        corrected_z_graph_trial_scores
    )
    stats["accepted_true_loop_corrected_z_graph_trial_score_min"] = min_finite(
        corrected_z_graph_trial_scores
    )
    stats["accepted_true_loop_bad_z_after_segment_consensus_ratio_mean"] = mean_finite(
        bad_z_after_segment_ratios
    )
    stats["accepted_true_loop_bad_z_after_segment_translation_median_mean"] = mean_finite(
        bad_z_after_segment_translations
    )
    stats["accepted_true_loop_corrected_z_segment_consensus_ratio_mean"] = mean_finite(
        corrected_z_segment_ratios
    )
    stats["accepted_true_loop_corrected_z_segment_translation_median_mean"] = mean_finite(
        corrected_z_segment_translations
    )
    stats["accepted_pose_loop"] = stats["accepted_true_loop"]
    stats["accepted_place_loop"] = stats["accepted_position_loop"]
    if stats["accepted_candidate_count"] > 0:
        stats["pose_loop_precision"] = (
            stats["accepted_pose_loop"] / stats["accepted_candidate_count"]
        )
        stats["position_loop_precision"] = (
            stats["accepted_position_loop"] / stats["accepted_candidate_count"]
        )
        stats["place_loop_precision"] = (
            stats["accepted_place_loop"] / stats["accepted_candidate_count"]
        )
    else:
        stats["pose_loop_precision"] = float("nan")
        stats["position_loop_precision"] = float("nan")
        stats["place_loop_precision"] = float("nan")
    stats["gt_loop_pair_count"] = len(gt_loop_pairs)
    stats["gt_pose_pair_count"] = len(gt_loop_pairs)
    stats["gt_position_pair_count"] = len(gt_position_pairs)
    stats["gt_place_pair_count"] = len(gt_position_pairs)
    gt_loop_opportunity_queries = {query_id for query_id, _ in gt_loop_pairs}
    gt_position_opportunity_queries = {query_id for query_id, _ in gt_position_pairs}
    candidate_queries = {int(c["query_id"]) for c in candidates}
    candidate_position_queries = {
        row["query_id"] for row in rows if row.get("gt_position_loop")
    }
    stats["gt_loop_opportunity_query_count"] = len(gt_loop_opportunity_queries)
    stats["gt_position_opportunity_query_count"] = len(gt_position_opportunity_queries)
    stats["query_with_any_candidate_count"] = len(candidate_queries)
    stats["query_without_any_candidate_count"] = len(set(poses.keys()) - candidate_queries)
    stats["query_with_any_position_candidate_count"] = len(candidate_position_queries)
    stats["query_without_position_candidate_count"] = len(
        gt_position_opportunity_queries - candidate_position_queries
    )
    stats["candidate_unique_pair_count"] = len(candidate_pairs)
    stats["accepted_pairs_source"] = "accepted_loops_csv" if accepted_pairs_available else "loop_debug_gate_result"
    query_summary_rows = build_query_summary(rows)
    stats["query_count"] = len(query_summary_rows)
    stats["query_with_true_candidate_count"] = sum(
        1 for row in query_summary_rows if row["true_candidate_count"] > 0
    )
    stats["query_with_position_candidate_count"] = sum(
        1 for row in query_summary_rows if row["position_candidate_count"] > 0
    )
    stats["query_with_selectable_true_candidate_count"] = sum(
        1 for row in query_summary_rows if row["true_selectable_candidate_count"] > 0
    )
    stats["query_with_selectable_position_candidate_count"] = sum(
        1 for row in query_summary_rows if row["position_selectable_candidate_count"] > 0
    )
    stats["query_selection_failure_count"] = sum(
        1 for row in query_summary_rows if row["selection_failure"]
    )
    stats["query_position_selection_failure_count"] = sum(
        1 for row in query_summary_rows if row["position_selection_failure"]
    )
    stats["query_missed_true_candidate_count"] = sum(
        1 for row in query_summary_rows if row["missed_true_candidate"]
    )
    stats["query_missed_position_candidate_count"] = sum(
        1 for row in query_summary_rows if row["missed_position_candidate"]
    )
    stats["thresholds"] = {
        "loop_translation_threshold_m": args.loop_translation_threshold,
        "loop_yaw_threshold_deg": args.loop_yaw_threshold_deg,
        "min_id_gap": args.min_id_gap,
        "z_drift_threshold_m": args.z_drift_threshold,
        "rpy_drift_threshold_deg": args.rpy_drift_threshold_deg,
    }

    optimization_rows = []
    for index, event in enumerate(optimization_summaries):
        accepted_edges = event.get("accepted_edges") or []
        if not accepted_edges:
            accepted_edges = [{"from_id": "", "to_id": ""}]
        for edge in accepted_edges:
            from_id = edge.get("from_id", "")
            to_id = edge.get("to_id", "")
            # n3mapping writes loop edges in match->query direction.
            query_id = to_id if isinstance(to_id, int) else ""
            match_id = from_id if isinstance(from_id, int) else ""
            translation, yaw_deg, has_gt = (
                gt_pair_metrics(poses, query_id, match_id)
                if isinstance(query_id, int) and isinstance(match_id, int)
                else (float("nan"), float("nan"), False)
            )
            heading_class = gt_heading_class(
                translation,
                yaw_deg,
                args.loop_translation_threshold,
                args.loop_yaw_threshold_deg,
            )
            gt_loop = is_gt_loop(
                translation,
                yaw_deg,
                args.loop_translation_threshold,
                args.loop_yaw_threshold_deg,
            )
            residual_z_after = event_float(event, "loop_residual_z_after")
            optimization_rows.append(
                {
                    "summary_index": index,
                    "from_id": from_id,
                    "to_id": to_id,
                    "query_id": query_id,
                    "match_id": match_id,
                    "gt_query_match_translation_m": translation,
                    "gt_query_match_yaw_deg": yaw_deg,
                    "gt_is_loop": gt_loop if has_gt else "",
                    "gt_pose_loop": gt_loop if has_gt else "",
                    "gt_position_loop": heading_class in ("same_heading", "opposite_heading", "cross_heading") if has_gt else "",
                    "gt_place_loop": heading_class in ("same_heading", "opposite_heading", "cross_heading") if has_gt else "",
                    "gt_heading_class": heading_class if has_gt else "",
                    "accepted_edge_count": event.get("accepted_edge_count", 0),
                    "loop_residual_x_before": event_float(event, "loop_residual_x_before"),
                    "loop_residual_y_before": event_float(event, "loop_residual_y_before"),
                    "loop_residual_z_before": event_float(event, "loop_residual_z_before"),
                    "loop_residual_x_after": event_float(event, "loop_residual_x_after"),
                    "loop_residual_y_after": event_float(event, "loop_residual_y_after"),
                    "loop_residual_z_after": residual_z_after,
                    "loop_residual_roll_before": event_float(event, "loop_residual_roll_before"),
                    "loop_residual_pitch_before": event_float(event, "loop_residual_pitch_before"),
                    "loop_residual_yaw_before": event_float(event, "loop_residual_yaw_before"),
                    "loop_residual_roll_after": event_float(event, "loop_residual_roll_after"),
                    "loop_residual_pitch_after": event_float(event, "loop_residual_pitch_after"),
                    "loop_residual_yaw_after": event_float(event, "loop_residual_yaw_after"),
                    "mean_pose_update_translation": event_float(event, "mean_pose_update_translation"),
                    "max_pose_update_translation": event_float(event, "max_pose_update_translation"),
                    "mean_pose_update_rotation": event_float(event, "mean_pose_update_rotation"),
                    "max_pose_update_rotation": event_float(event, "max_pose_update_rotation"),
                }
            )
            if math.isfinite(residual_z_after):
                stats["optimization_max_residual_z_after"] = max(
                    stats["optimization_max_residual_z_after"],
                    abs(residual_z_after),
                )
                if abs(residual_z_after) >= args.z_drift_threshold:
                    stats["optimization_high_residual_z_after_count"] += 1
    stats["optimization_summary_count"] = len(optimization_summaries)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "loop_candidates_labeled.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = [
            "query_id",
            "match_id",
            "gt_query_match_translation_m",
            "gt_query_match_yaw_deg",
            "gt_distance_m",
            "gt_yaw_diff_deg",
            "gt_is_loop",
            "gt_pose_loop",
            "gt_position_loop",
            "gt_place_loop",
            "gt_heading_class",
            "gt_relative_x",
            "gt_relative_y",
            "gt_relative_z",
            "gt_relative_roll",
            "gt_relative_pitch",
            "gt_relative_yaw",
            "candidate_accepted",
            "candidate_selectable",
            "candidate_source",
            "edge_mode",
            "vertical_observability_score",
            "vertical_downweighted",
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
            "vertical_hypothesis_count",
            "best_z_offset_m",
            "best_z_offset_fitness",
            "zero_z_fitness",
            "fitness_gap_zero_vs_best",
            "z_hypothesis_spread_m",
            "vertical_ambiguity_score",
            "vertical_hypothesis_edge_recommendation",
            "heightmap_overlap_cell_count",
            "heightmap_overlap_ratio",
            "heightmap_ground_dz_median",
            "heightmap_ground_dz_p90",
            "heightmap_ground_dz_max",
            "heightmap_ground_support_ratio",
            "heightmap_vertical_consistency_score",
            "graph_trial_success",
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
            "graph_trial_recommendation",
            "segment_pair_count",
            "segment_valid_pair_count",
            "segment_consensus_inlier_count",
            "segment_consensus_ratio",
            "segment_translation_median",
            "segment_translation_std",
            "segment_yaw_median",
            "segment_yaw_std",
            "segment_z_std",
            "segment_roll_pitch_std",
            "segment_direction",
            "segment_recommendation",
            "icp_iterations",
            "icp_optimizer_error",
            "icp_termination",
            "pred_match_query_x",
            "pred_match_query_y",
            "pred_match_query_z",
            "pred_match_query_roll",
            "pred_match_query_pitch",
            "pred_match_query_yaw",
            "icp_correction_match_x",
            "icp_correction_match_y",
            "icp_correction_match_z",
            "icp_correction_match_roll",
            "icp_correction_match_pitch",
            "icp_correction_match_yaw",
            "measured_match_query_x",
            "measured_match_query_y",
            "measured_match_query_z",
            "measured_match_query_roll",
            "measured_match_query_pitch",
            "measured_match_query_yaw",
            "loop_referee_recommendation",
            "loop_referee_reason",
            "loop_referee_risk_flags",
            "z_measurement_residual_large",
            "z_measurement_bad",
            "z_after_bad",
            "z_corrected",
            "gate_result",
            "reject_reason",
            "fitness_score",
            "inlier_ratio",
            "residual_z",
            "residual_roll",
            "residual_pitch",
            "residual_yaw",
            "residual_z_after",
            "residual_roll_after",
            "residual_pitch_after",
            "residual_yaw_after",
            "measurement_x",
            "measurement_y",
            "measurement_z",
            "measurement_roll",
            "measurement_pitch",
            "measurement_yaw",
            "icp_error_to_gt_x",
            "icp_error_to_gt_y",
            "icp_error_to_gt_z",
            "icp_error_to_gt_roll",
            "icp_error_to_gt_pitch",
            "icp_error_to_gt_yaw",
            "z_drift_suspect",
            "category",
            "failure_class",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            for key in (
                "gt_query_match_translation_m",
                "gt_query_match_yaw_deg",
                "gt_distance_m",
                "gt_yaw_diff_deg",
                "gt_relative_x",
                "gt_relative_y",
                "gt_relative_z",
                "gt_relative_roll",
                "gt_relative_pitch",
                "gt_relative_yaw",
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
                "heightmap_overlap_ratio",
                "heightmap_ground_dz_median",
                "heightmap_ground_dz_p90",
                "heightmap_ground_dz_max",
                "heightmap_ground_support_ratio",
                "heightmap_vertical_consistency_score",
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
                "segment_consensus_ratio",
                "segment_translation_median",
                "segment_translation_std",
                "segment_yaw_median",
                "segment_yaw_std",
                "segment_z_std",
                "segment_roll_pitch_std",
                "icp_optimizer_error",
                "pred_match_query_x",
                "pred_match_query_y",
                "pred_match_query_z",
                "pred_match_query_roll",
                "pred_match_query_pitch",
                "pred_match_query_yaw",
                "icp_correction_match_x",
                "icp_correction_match_y",
                "icp_correction_match_z",
                "icp_correction_match_roll",
                "icp_correction_match_pitch",
                "icp_correction_match_yaw",
                "measured_match_query_x",
                "measured_match_query_y",
                "measured_match_query_z",
                "measured_match_query_roll",
                "measured_match_query_pitch",
                "measured_match_query_yaw",
                "fitness_score",
                "inlier_ratio",
                "residual_z",
                "residual_roll",
                "residual_pitch",
                "residual_yaw",
                "residual_z_after",
                "residual_roll_after",
                "residual_pitch_after",
                "residual_yaw_after",
                "measurement_x",
                "measurement_y",
                "measurement_z",
                "measurement_roll",
                "measurement_pitch",
                "measurement_yaw",
                "icp_error_to_gt_x",
                "icp_error_to_gt_y",
                "icp_error_to_gt_z",
                "icp_error_to_gt_roll",
                "icp_error_to_gt_pitch",
                "icp_error_to_gt_yaw",
            ):
                formatted[key] = format_float(row[key])
            writer.writerow(formatted)

    optimization_csv_path = output / "loop_optimization_summary.csv"
    with open(optimization_csv_path, "w", newline="") as f:
        fieldnames = [
            "summary_index",
            "from_id",
            "to_id",
            "query_id",
            "match_id",
            "gt_query_match_translation_m",
            "gt_query_match_yaw_deg",
            "gt_is_loop",
            "gt_pose_loop",
            "gt_position_loop",
            "gt_place_loop",
            "gt_heading_class",
            "accepted_edge_count",
            "loop_residual_x_before",
            "loop_residual_y_before",
            "loop_residual_z_before",
            "loop_residual_x_after",
            "loop_residual_y_after",
            "loop_residual_z_after",
            "loop_residual_roll_before",
            "loop_residual_pitch_before",
            "loop_residual_yaw_before",
            "loop_residual_roll_after",
            "loop_residual_pitch_after",
            "loop_residual_yaw_after",
            "mean_pose_update_translation",
            "max_pose_update_translation",
            "mean_pose_update_rotation",
            "max_pose_update_rotation",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in optimization_rows:
            formatted = dict(row)
            for key in (
                "gt_query_match_translation_m",
                "gt_query_match_yaw_deg",
                "loop_residual_x_before",
                "loop_residual_y_before",
                "loop_residual_z_before",
                "loop_residual_x_after",
                "loop_residual_y_after",
                "loop_residual_z_after",
                "loop_residual_roll_before",
                "loop_residual_pitch_before",
                "loop_residual_yaw_before",
                "loop_residual_roll_after",
                "loop_residual_pitch_after",
                "loop_residual_yaw_after",
                "mean_pose_update_translation",
                "max_pose_update_translation",
                "mean_pose_update_rotation",
                "max_pose_update_rotation",
            ):
                formatted[key] = format_float(row[key])
            writer.writerow(formatted)

    summary_csv_path = output / "loop_query_summary.csv"
    with open(summary_csv_path, "w", newline="") as f:
        fieldnames = [
            "query_id",
            "candidate_count",
            "true_candidate_count",
            "position_candidate_count",
            "true_selectable_candidate_count",
            "position_selectable_candidate_count",
            "accepted_match_id",
            "accepted_is_gt_loop",
            "accepted_is_position_loop",
            "accepted_gt_translation_m",
            "accepted_gt_yaw_deg",
            "accepted_fitness_score",
            "accepted_inlier_ratio",
            "accepted_residual_z",
            "best_true_match_id",
            "best_true_fitness_score",
            "best_true_inlier_ratio",
            "best_true_gt_translation_m",
            "best_true_residual_z",
            "best_selectable_true_match_id",
            "best_selectable_true_fitness_score",
            "best_selectable_true_gt_translation_m",
            "best_selectable_true_residual_z",
            "best_candidate_match_id",
            "best_candidate_is_gt_loop",
            "best_candidate_fitness_score",
            "selection_failure",
            "missed_true_candidate",
            "position_selection_failure",
            "missed_position_candidate",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in query_summary_rows:
            formatted = dict(row)
            for key in (
                "accepted_gt_translation_m",
                "accepted_gt_yaw_deg",
                "accepted_fitness_score",
                "accepted_inlier_ratio",
                "accepted_residual_z",
                "best_true_fitness_score",
                "best_true_inlier_ratio",
                "best_true_gt_translation_m",
                "best_true_residual_z",
                "best_selectable_true_fitness_score",
                "best_selectable_true_gt_translation_m",
                "best_selectable_true_residual_z",
                "best_candidate_fitness_score",
            ):
                formatted[key] = format_float(row[key])
            writer.writerow(formatted)

    json_path = output / "loop_diagnosis.json"
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"analyzed candidates={stats['candidate_count']} output={output}")
    return 0


def main():
    try:
        return analyze(parse_args())
    except Exception as exc:  # pragma: no cover - surfaced in CLI tests
        print(f"n3mapping_loop_debug_analyze: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
