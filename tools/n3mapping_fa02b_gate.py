#!/usr/bin/env python3
"""Fail-closed paired gate for correlated-odometry loop correction."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import n3mapping_fa02_gate as fa02


CONTRACT_SCHEMA = "n3mapping_fa02b_acceptance_contract_v1"
RUN_SCHEMA = "n3mapping_fa02b_condition_run_v1"
REPORT_SCHEMA = "n3mapping_fa02b_acceptance_v1"
PASS = "PASS"
FAIL = "FAIL_FA02B_QUALITY"
INVALID = "INVALID_EVIDENCE"
EXIT_CODES = {PASS: 0, FAIL: 1, INVALID: 3}


def drift_config(profile: dict[str, Any], seed: int) -> dict[str, Any]:
    return {
        "enabled": True,
        "seed": seed,
        "translation_scale_error_fraction": float(
            profile["translation_scale_error_fraction"]
        ),
        "yaw_bias_deg_per_meter": float(profile["yaw_bias_deg_per_meter"]),
        "translation_rw_std_m_per_sqrt_meter": float(
            profile["translation_rw_std_m_per_sqrt_meter"]
        ),
        "rotation_rw_std_deg_per_sqrt_meter": float(
            profile["rotation_rw_std_deg_per_sqrt_meter"]
        ),
    }


def same_number(lhs: Any, rhs: Any) -> bool:
    return (
        isinstance(lhs, (int, float))
        and not isinstance(lhs, bool)
        and math.isfinite(float(lhs))
        and math.isclose(float(lhs), float(rhs), rel_tol=0.0, abs_tol=1e-12)
    )


def verify_condition(
    label: str,
    directory: Path,
    expected_condition: str,
    case: dict[str, Any],
    episode: dict[str, Any],
    expected_drift: dict[str, Any],
    provenance: dict[str, str],
    issues: fa02.Issues,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = fa02.load_json(directory / "run_manifest.json")
    if manifest.get("schema") != RUN_SCHEMA or manifest.get("status") != "COMPLETE":
        issues.add("evidence", "run_manifest", f"{label}: incomplete run manifest")
    required = {
        "case_id": str(case["id"]),
        "condition": expected_condition,
        "episode_id": str(episode["id"]),
        "dataset": str(episode["dataset"]),
        "sequence": str(episode["sequence"]),
        "source_commit": provenance["source_commit"],
        "input_manifest_sha256": provenance["input_manifest_sha256"],
        "episode_frames_sha256": provenance["episode_frames_sha256"],
        "process_return_code": 0,
        "missing_artifacts": [],
    }
    for field, expected in required.items():
        if manifest.get(field) != expected:
            issues.add(
                "evidence", "run_provenance",
                f"{label}: {field}={manifest.get(field)!r}, expected {expected!r}",
            )
    if manifest.get("drift_config") != expected_drift:
        issues.add("evidence", "drift_manifest", f"{label}: drift config mismatch")
    evaluator = manifest.get("evaluator")
    if not isinstance(evaluator, dict):
        issues.add("evidence", "evaluator_fingerprint", f"{label}: evaluator missing")
    else:
        path = Path(str(evaluator.get("path", "")))
        expected_hash = evaluator.get("sha256")
        expected_size = evaluator.get("size_bytes")
        if (
            not path.is_file()
            or not isinstance(expected_hash, str)
            or not fa02.SHA256_RE.fullmatch(expected_hash)
            or not isinstance(expected_size, int)
            or path.stat().st_size != expected_size
            or fa02.sha256_file(path) != expected_hash
        ):
            issues.add(
                "evidence", "evaluator_fingerprint",
                f"{label}: evaluator binary changed, missing, or malformed",
            )
    command = manifest.get("command")
    if not isinstance(command, list):
        issues.add("evidence", "evaluator_command", f"{label}: command missing")
    else:
        if not isinstance(evaluator, dict) or command[0] != evaluator.get("path"):
            issues.add(
                "evidence", "evaluator_command",
                f"{label}: command does not use the fingerprinted evaluator",
            )
        disabled = "--disable_loop_closure" in command
        if disabled != (expected_condition == "loop_off"):
            issues.add(
                "evidence", "loop_condition_command",
                f"{label}: loop flag does not match condition",
            )
        if "--enable_correlated_odom_drift" not in command:
            issues.add(
                "evidence", "drift_command",
                f"{label}: correlated drift was not explicitly enabled",
            )

    metrics = fa02.load_json(directory / "metrics.json")
    expected_loop_enabled = expected_condition == "loop_on"
    if metrics.get("loop_closure_enabled") is not expected_loop_enabled:
        issues.add("evidence", "loop_condition_metrics", f"{label}: loop condition mismatch")
    if metrics.get("odom_source") != "correlated_gt_derived" or metrics.get(
        "backend_input_contract"
    ) != "correlated_gt_derived_odom_plus_lidar":
        issues.add("evidence", "backend_contract", f"{label}: backend contract mismatch")
    if metrics.get("frames_processed") != int(episode["max_frames"]):
        issues.add("evidence", "frame_count", f"{label}: frame count mismatch")
    actual_drift = metrics.get("correlated_odom_drift")
    if not isinstance(actual_drift, dict):
        issues.add("evidence", "drift_metrics", f"{label}: drift metrics missing")
    else:
        for field, expected in expected_drift.items():
            actual = actual_drift.get(field)
            matches = actual is expected if isinstance(expected, bool) else same_number(actual, expected)
            if not matches:
                issues.add(
                    "evidence", "drift_metrics",
                    f"{label}: drift {field}={actual!r}, expected {expected!r}",
                )
    return manifest, metrics


def grade_accepted_loops(
    episode: dict[str, Any], directory: Path, oracle: dict[str, Any], issues: fa02.Issues
) -> dict[str, Any]:
    label = str(episode["id"])
    keyframes = fa02.load_keyframes(directory / "keyframes_gt.csv")
    accepted = fa02.load_accepted_pairs(directory / "accepted_loops.csv")
    candidates = fa02.load_candidates(directory / "loop_debug.jsonl")
    authoritative = bool(episode["attitude_authoritative"])
    correct = 0
    catastrophic = 0
    details: list[dict[str, Any]] = []
    for pair in sorted(accepted):
        query_id, match_id = pair
        if query_id not in keyframes or match_id not in keyframes:
            issues.add("evidence", "accepted_pair_missing_gt", f"{label}: {pair} lacks GT")
            continue
        event = candidates.get(pair)
        if event is None or event.get("gate_result") != "accepted":
            issues.add(
                "evidence", "accepted_pair_missing_debug",
                f"{label}: {pair} lacks an accepted debug event",
            )
            continue
        measurement = fa02.measurement_pose(event)
        gt_relative = fa02.relative_pose(keyframes[match_id], keyframes[query_id])
        translation_error, rotation_error = fa02.pose_error(gt_relative, measurement)
        place_distance = fa02.translation_norm(tuple(
            a - b for a, b in zip(
                keyframes[query_id]["translation"], keyframes[match_id]["translation"]
            )
        ))
        measurement_correct = (
            translation_error
            <= float(oracle["correct_measurement_translation_error_m_max"])
            and rotation_error
            <= float(oracle["correct_measurement_rotation_error_deg_max"])
        )
        position_consistent = place_distance <= float(
            oracle["place_translation_threshold_m"]
        )
        pair_correct = measurement_correct if authoritative else position_consistent
        pair_catastrophic = (
            authoritative
            and (
                translation_error
                > float(oracle["catastrophic_measurement_translation_error_m_min"])
                or rotation_error
                > float(oracle["catastrophic_measurement_rotation_error_deg_min"])
            )
        ) or (
            not authoritative
            and place_distance > float(oracle["catastrophic_place_distance_m_min"])
        )
        correct += int(pair_correct)
        catastrophic += int(pair_catastrophic)
        details.append({
            "query_id": query_id,
            "match_id": match_id,
            "correct": pair_correct,
            "catastrophic": pair_catastrophic,
            "place_distance_m": place_distance,
            "measurement_translation_error_m": translation_error,
            "measurement_rotation_error_deg": rotation_error,
        })
    return {
        "accepted_loop_count": len(accepted),
        "correct_accepted_loop_count": correct,
        "catastrophic_false_loop_count": catastrophic,
        "accepted_loops": details,
    }


def improvement_ratio(before: float | None, after: float | None) -> float | None:
    if before is None or after is None or before <= 0.0:
        return None
    return (before - after) / before


def regression_ratio(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    if before <= 1e-12:
        return 1.0 if after <= 1e-12 else math.inf
    return after / before


def evaluate_pair(
    case: dict[str, Any],
    episode: dict[str, Any],
    profile: dict[str, Any],
    case_root: Path,
    oracle: dict[str, Any],
    thresholds: dict[str, Any],
    provenance: dict[str, str],
    map_helpers: Any,
    proto_module: Any,
    issues: fa02.Issues,
) -> dict[str, Any]:
    label = str(case["id"])
    expected_drift = drift_config(profile, int(case["seed"]))
    off_dir = case_root / "loop_off"
    on_dir = case_root / "loop_on"
    off_manifest, off_metrics = verify_condition(
        f"{label}/loop_off", off_dir, "loop_off", case, episode,
        expected_drift, provenance, issues,
    )
    on_manifest, on_metrics = verify_condition(
        f"{label}/loop_on", on_dir, "loop_on", case, episode,
        expected_drift, provenance, issues,
    )
    if off_manifest.get("evaluator") != on_manifest.get("evaluator"):
        issues.add("evidence", "paired_evaluator", f"{label}: evaluator fingerprints differ")
    for artifact in ("trajectory_gt.txt", "trajectory_odom.txt", "keyframes_gt.csv"):
        if fa02.sha256_file(off_dir / artifact) != fa02.sha256_file(on_dir / artifact):
            issues.add("evidence", "paired_input", f"{label}: paired {artifact} differs")

    if int(off_metrics.get("accepted_loop_count", -1)) != 0:
        issues.add("evidence", "loop_off_accepted", f"{label}: loop-off accepted a loop")
    if fa02.load_accepted_pairs(off_dir / "accepted_loops.csv"):
        issues.add("evidence", "loop_off_csv", f"{label}: loop-off CSV is not empty")

    odom = fa02.trajectory_metrics(
        off_dir / "trajectory_gt.txt", off_dir / "trajectory_odom.txt"
    )
    loop_off = fa02.trajectory_metrics(
        off_dir / "trajectory_gt.txt", off_dir / "trajectory_optimized.txt"
    )
    loop_on = fa02.trajectory_metrics(
        on_dir / "trajectory_gt.txt", on_dir / "trajectory_optimized.txt"
    )
    loops = grade_accepted_loops(episode, on_dir, oracle, issues)

    for condition, directory, metrics in (
        ("loop_off", off_dir, off_metrics),
        ("loop_on", on_dir, on_metrics),
    ):
        map_proto = map_helpers.parse_map(directory / "n3map.pbstream", proto_module)
        structure = map_helpers.map_structure(map_proto, proto_module)
        expected_loops = 0 if condition == "loop_off" else loops["accepted_loop_count"]
        invariants = {
            "keyframes": int(metrics["accepted_keyframes"]),
            "odometry": max(0, int(metrics["accepted_keyframes"]) - 1),
            "loop": expected_loops,
            "session_anchor": 0,
            "unknown_edges": 0,
            "dense_trajectory": int(episode["max_frames"]),
            "metadata_match": True,
            "duplicate_keyframes": 0,
            "dangling_edges": 0,
        }
        for field, expected in invariants.items():
            if structure.get(field) != expected:
                issues.add(
                    "quality", "graph_structure",
                    f"{label}/{condition}: map {field}={structure.get(field)!r}, "
                    f"expected {expected!r}",
                )
        stderr = (directory / "stderr.log").read_text(
            encoding="utf-8", errors="replace"
        )
        optimizer_errors = sum(
            bool(re.search(r"(?:optimizer|optimization).*(?:exception|failed|fatal)", line, re.I))
            for line in stderr.splitlines()
        )
        if optimizer_errors > int(thresholds["optimizer_error_count_max"]):
            issues.add(
                "quality", "optimizer_error",
                f"{label}/{condition}: optimizer errors={optimizer_errors}",
            )

    injected_ate = odom["ate_translation_rmse_m"]
    ate_improvement = improvement_ratio(
        loop_off["ate_translation_rmse_m"], loop_on["ate_translation_rmse_m"]
    )
    rpe_translation_ratio = regression_ratio(
        loop_off["rpe_translation_rmse_m"], loop_on["rpe_translation_rmse_m"]
    )
    rpe_rotation_ratio = regression_ratio(
        loop_off["rpe_rotation_rmse_deg"], loop_on["rpe_rotation_rmse_deg"]
    )
    positive = str(case["expected_role"]) == "positive_correction"
    case_pass = True
    if injected_ate is None or injected_ate < float(
        thresholds["minimum_injected_odom_ate_translation_rmse_m"]
    ):
        issues.add("quality", "insufficient_drift", f"{label}: odom ATE={injected_ate}")
        case_pass = False
    if loops["catastrophic_false_loop_count"] > int(
        thresholds["catastrophic_false_loop_count_max"]
    ):
        issues.add(
            "quality", "catastrophic_false_loop",
            f"{label}: catastrophic loops={loops['catastrophic_false_loop_count']}",
        )
        case_pass = False
    if positive:
        if loops["correct_accepted_loop_count"] < int(
            thresholds["minimum_correct_accepted_loops_per_positive_case"]
        ):
            issues.add("quality", "positive_loop_miss", f"{label}: no correct loop")
            case_pass = False
        if ate_improvement is None or ate_improvement < float(
            thresholds["ate_translation_improvement_ratio_min"]
        ):
            issues.add(
                "quality", "ate_improvement",
                f"{label}: translation ATE improvement={ate_improvement}",
            )
            case_pass = False
        if rpe_translation_ratio is None or rpe_translation_ratio > float(
            thresholds["rpe_translation_regression_ratio_max"]
        ):
            issues.add(
                "quality", "rpe_translation_regression",
                f"{label}: translation RPE ratio={rpe_translation_ratio}",
            )
            case_pass = False
        if bool(episode["attitude_authoritative"]) and (
            rpe_rotation_ratio is None
            or rpe_rotation_ratio > float(thresholds["rpe_rotation_regression_ratio_max"])
        ):
            issues.add(
                "quality", "rpe_rotation_regression",
                f"{label}: rotation RPE ratio={rpe_rotation_ratio}",
            )
            case_pass = False
    elif str(case["expected_role"]) == "no_loop_control" and (
        loops["accepted_loop_count"]
        > int(thresholds["accepted_loop_count_max_per_control_case"])
    ):
        issues.add(
            "quality", "control_false_loop",
            f"{label}: accepted loops={loops['accepted_loop_count']}",
        )
        case_pass = False
    return {
        "id": label,
        "episode_id": episode["id"],
        "profile_id": case["profile_id"],
        "seed": int(case["seed"]),
        "expected_role": case["expected_role"],
        "pass": case_pass,
        "drift": expected_drift,
        "odom": odom,
        "loop_off": loop_off,
        "loop_on": loop_on,
        "ate_translation_improvement_ratio": ate_improvement,
        "rpe_translation_ratio_on_over_off": rpe_translation_ratio,
        "rpe_rotation_ratio_on_over_off": rpe_rotation_ratio,
        "loops": loops,
    }


def evaluate(contract_path: Path, root: Path, proto_path: Path) -> dict[str, Any]:
    issues = fa02.Issues()
    contract = fa02.load_json(contract_path)
    if contract.get("schema") != CONTRACT_SCHEMA:
        issues.add("evidence", "contract_schema", "FA-02B contract schema mismatch")
    section = contract.get("fa02b")
    if not isinstance(section, dict):
        raise ValueError("contract has no fa02b section")
    input_manifest, _ = fa02.verify_inputs(
        contract_path, contract, root, issues, section="fa02b"
    )
    provenance = {
        "source_commit": str(input_manifest.get("source_commit", "")),
        "input_manifest_sha256": fa02.sha256_file(root / "input_manifest.json"),
        "episode_frames_sha256": fa02.sha256_file(root / "episode_frames.csv"),
    }
    episodes = {str(item["id"]): item for item in section["episodes"]}
    profiles = {str(item["id"]): item for item in section["drift_profiles"]}
    map_helpers = fa02.load_map_helpers()
    proto_module = map_helpers.load_proto_module(proto_path)
    reports: list[dict[str, Any]] = []
    for case in section["cases"]:
        try:
            reports.append(evaluate_pair(
                case,
                episodes[str(case["episode_id"])],
                profiles[str(case["profile_id"])],
                root / "runs" / str(case["id"]),
                section["oracle"],
                section["thresholds"],
                provenance,
                map_helpers,
                proto_module,
                issues,
            ))
        except Exception as exc:
            issues.add("evidence", "case_exception", f"{case['id']}: {exc}")
    positive = [item for item in reports if item["expected_role"] == "positive_correction"]
    positive_passed = sum(bool(item["pass"]) for item in positive)
    pass_rate = positive_passed / len(positive) if positive else None
    if pass_rate is None or pass_rate < float(
        section["thresholds"]["positive_case_pass_rate_min"]
    ):
        issues.add("quality", "positive_case_pass_rate", f"pass rate={pass_rate}")
    status = INVALID if issues.counts["evidence"] else FAIL if issues.counts["quality"] else PASS
    return {
        "schema": REPORT_SCHEMA,
        "status": status,
        "source_commit": input_manifest.get("source_commit"),
        "aggregate": {
            "case_count": len(reports),
            "positive_case_count": len(positive),
            "positive_case_passed": positive_passed,
            "positive_case_pass_rate": pass_rate,
            "accepted_loop_count": sum(
                item["loops"]["accepted_loop_count"] for item in reports
            ),
            "correct_accepted_loop_count": sum(
                item["loops"]["correct_accepted_loop_count"] for item in reports
            ),
            "catastrophic_false_loop_count": sum(
                item["loops"]["catastrophic_false_loop_count"] for item in reports
            ),
        },
        "cases": reports,
        "issue_counts": dict(issues.counts),
        "issues": issues.items,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--proto", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        report = evaluate(
            args.contract.resolve(strict=True),
            args.evidence_root.resolve(strict=True),
            args.proto.resolve(strict=True),
        )
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps({"status": report["status"], "aggregate": report["aggregate"]}, sort_keys=True))
        return EXIT_CODES[report["status"]]
    except Exception as exc:
        print(f"n3mapping_fa02b_gate: {exc}", file=sys.stderr)
        return EXIT_CODES[INVALID]


if __name__ == "__main__":
    raise SystemExit(main())
