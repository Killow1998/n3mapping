#!/usr/bin/env python3
"""Compare two strict-valid n3mapping v2 evaluation runs.

The tool reads ``paired_comparison`` from a JSON-compatible
``frozen_contract.yaml`` and writes ``paired_diff.csv`` plus
``gate_report.json``. Safety, utility, and resource gates remain separate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from n3mapping_eval_validate import (
    SCHEMA_VERSION,
    load_contract,
    load_json_strict,
    sha256_file,
    validate_run,
)


GATE_NAMES = ("safety", "utility", "resource")
DIRECTIONS = {"higher", "lower", "equal"}
MARGIN_MODES = {"absolute", "relative", "both"}
FORMAL_EVIDENCE_CLASS = "recorded_cross_session"
FORMAL_ODOM_SOURCE = "recorded_lio_odom"
FORMAL_QUERY_SOURCE = "independent_real_scan"
SYNTHETIC_EVIDENCE_CLASS = "synthetic_map_render"
FINGERPRINT_ALGORITHM = "fnv1a64_xyz_intensity_float32_le_v1"


def _dotted_get(value: Any, path: str) -> Any:
    current = value
    for component in path.split("."):
        if not isinstance(current, dict) or component not in current:
            raise KeyError(path)
        current = current[component]
    return current


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _json_dump_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if path.exists() or path.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"refusing to overwrite comparison artifact: {path}")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"comparison artifact appeared during write: {path}")
    temporary.replace(path)


def _csv_dump_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "metric",
        "gate",
        "baseline",
        "candidate",
        "delta",
        "relative_delta",
        "direction",
        "absolute_margin",
        "relative_margin",
        "margin_mode",
        "minimum",
        "maximum",
        "status",
        "reason",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if path.exists() or path.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"refusing to overwrite comparison artifact: {path}")
    with temporary.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"comparison artifact appeared during write: {path}")
    temporary.replace(path)


def _metric_result(
    spec: dict[str, Any],
    baseline_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
) -> tuple[dict[str, Any], str | None]:
    metric = spec.get("metric")
    gate = spec.get("gate")
    direction = spec.get("direction")
    margin_mode = spec.get("margin_mode", "both")
    row: dict[str, Any] = {
        "metric": metric,
        "gate": gate,
        "baseline": None,
        "candidate": None,
        "delta": None,
        "relative_delta": None,
        "direction": direction,
        "absolute_margin": spec.get("absolute_margin"),
        "relative_margin": spec.get("relative_margin"),
        "margin_mode": margin_mode,
        "minimum": spec.get("minimum"),
        "maximum": spec.get("maximum"),
        "status": "INVALID",
        "reason": "",
    }
    if not isinstance(metric, str) or not metric:
        row["reason"] = "metric name must be a non-empty string"
        return row, row["reason"]
    if gate not in GATE_NAMES:
        row["reason"] = f"unsupported gate {gate!r}"
        return row, row["reason"]
    if direction not in DIRECTIONS:
        row["reason"] = f"unsupported direction {direction!r}"
        return row, row["reason"]
    if margin_mode not in MARGIN_MODES:
        row["reason"] = f"unsupported margin_mode {margin_mode!r}"
        return row, row["reason"]
    absolute_margin = _finite_number(spec.get("absolute_margin"))
    relative_margin = _finite_number(spec.get("relative_margin"))
    if absolute_margin is None or absolute_margin < 0.0:
        row["reason"] = "absolute_margin must be finite and non-negative"
        return row, row["reason"]
    if relative_margin is None or relative_margin < 0.0:
        row["reason"] = "relative_margin must be finite and non-negative"
        return row, row["reason"]
    try:
        baseline_raw = _dotted_get(baseline_metrics, metric)
        candidate_raw = _dotted_get(candidate_metrics, metric)
    except KeyError:
        row["reason"] = f"required metric {metric!r} is missing"
        return row, row["reason"]
    baseline = _finite_number(baseline_raw)
    candidate = _finite_number(candidate_raw)
    if baseline is None or candidate is None:
        row["reason"] = f"required metric {metric!r} must be finite and non-null"
        return row, row["reason"]
    delta = candidate - baseline
    relative_delta = delta / abs(baseline) if baseline != 0.0 else None
    row.update(
        {
            "baseline": baseline,
            "candidate": candidate,
            "delta": delta,
            "relative_delta": relative_delta,
        }
    )

    if direction == "higher":
        absolute_pass = delta >= -absolute_margin
        relative_pass = relative_delta is None or relative_delta >= -relative_margin
    elif direction == "lower":
        absolute_pass = delta <= absolute_margin
        relative_pass = relative_delta is None or relative_delta <= relative_margin
    else:
        absolute_pass = abs(delta) <= absolute_margin
        relative_pass = relative_delta is None or abs(relative_delta) <= relative_margin

    if margin_mode == "absolute":
        margin_pass = absolute_pass
    elif margin_mode == "relative":
        if relative_delta is None:
            row["reason"] = "relative margin is undefined at a zero baseline; use absolute or both"
            return row, row["reason"]
        margin_pass = relative_pass
    else:
        # At a zero baseline the pre-registered absolute margin is authoritative.
        margin_pass = absolute_pass and relative_pass

    threshold_pass = True
    minimum = spec.get("minimum")
    maximum = spec.get("maximum")
    if minimum is not None:
        minimum_number = _finite_number(minimum)
        if minimum_number is None:
            row["reason"] = "minimum must be finite when present"
            return row, row["reason"]
        threshold_pass = threshold_pass and candidate >= minimum_number
    if maximum is not None:
        maximum_number = _finite_number(maximum)
        if maximum_number is None:
            row["reason"] = "maximum must be finite when present"
            return row, row["reason"]
        threshold_pass = threshold_pass and candidate <= maximum_number
    passed = margin_pass and threshold_pass
    row["status"] = "PASS" if passed else "FAIL"
    if passed:
        row["reason"] = "within frozen margin and absolute threshold"
    elif not margin_pass:
        row["reason"] = "candidate exceeds frozen non-inferiority margin"
    else:
        row["reason"] = "candidate violates frozen absolute threshold"
    return row, None


def _compatibility_value(manifest: dict[str, Any], field: str) -> Any:
    return _dotted_get(manifest, field)


def _prepare_output(output: Path) -> str | None:
    if output.exists() or output.is_symlink():
        if output.is_symlink() or not output.is_dir():
            return f"comparison output exists and is not a fresh directory: {output}"
        try:
            if next(output.iterdir(), None) is not None:
                return f"comparison output directory is non-empty; refusing to overwrite: {output}"
        except OSError as exc:
            return f"failed to inspect comparison output directory: {exc}"
        return None
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        return f"failed to create comparison output directory: {exc}"
    return None


def _read_query_fingerprints(run_dir: Path) -> tuple[dict[str, tuple[bool, str, str]], list[str]]:
    relative = "raw/resolved_queries.csv"
    path = run_dir / relative
    errors: list[str] = []
    records: dict[str, tuple[bool, str, str]] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            headers = list(reader.fieldnames or [])
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        return {}, [f"failed to read {relative}: {exc}"]
    required = {
        "query_id",
        "query_cloud_fingerprint_available",
        "query_cloud_fingerprint_algorithm",
        "query_cloud_fingerprint_fnv1a64",
    }
    missing = sorted(required - set(headers))
    if missing:
        return {}, [f"{relative} is missing fingerprint fields: {missing}"]
    for row_index, row in enumerate(rows, start=2):
        query_id = (row.get("query_id") or "").strip()
        available_raw = (row.get("query_cloud_fingerprint_available") or "").strip().lower()
        if available_raw in {"true", "1"}:
            available = True
        elif available_raw in {"false", "0"}:
            available = False
        else:
            errors.append(f"{relative}:{row_index} fingerprint availability is not boolean")
            continue
        algorithm = (row.get("query_cloud_fingerprint_algorithm") or "").strip()
        fingerprint = (row.get("query_cloud_fingerprint_fnv1a64") or "").strip()
        if not query_id or query_id in records:
            errors.append(f"{relative}:{row_index} has empty or duplicate query_id")
            continue
        if algorithm != FINGERPRINT_ALGORITHM:
            errors.append(f"{relative}:{row_index} has the wrong fingerprint algorithm")
        if len(fingerprint) != 16 or any(character not in "0123456789abcdef" for character in fingerprint):
            errors.append(f"{relative}:{row_index} fingerprint must be 16 lowercase hex")
        records[query_id] = (available, algorithm, fingerprint)
    return records, errors


def compare_runs(
    baseline_run: Path | str,
    candidate_run: Path | str,
    contract_path: Path | str,
    output_dir: Path | str,
) -> dict[str, Any]:
    baseline_dir = Path(baseline_run).expanduser().resolve()
    candidate_dir = Path(candidate_run).expanduser().resolve()
    contract_file = Path(contract_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    freshness_error = _prepare_output(output)
    if freshness_error is not None:
        return {
            "schema_version": SCHEMA_VERSION,
            "baseline_run": str(baseline_dir),
            "candidate_run": str(candidate_dir),
            "contract": str(contract_file),
            "contract_sha256": sha256_file(contract_file) if contract_file.is_file() else None,
            "verdict": "INVALID",
            "evidence_ceiling": "INVALID",
            "errors": [freshness_error],
            "baseline_validation": None,
            "candidate_validation": None,
            "compatibility": [],
            "gates": {},
        }

    errors: list[str] = []
    if baseline_dir == candidate_dir:
        errors.append("baseline and candidate resolve to the same run directory")
    baseline_validation = validate_run(baseline_dir, strict=True).to_dict()
    candidate_validation = validate_run(candidate_dir, strict=True).to_dict()
    if not baseline_validation["valid"]:
        errors.append("baseline run failed strict validation")
    if not candidate_validation["valid"]:
        errors.append("candidate run failed strict validation")

    try:
        contract = load_contract(contract_file)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        contract = {}
        errors.append(f"failed to load frozen contract: {exc}")
    contract_digest = sha256_file(contract_file) if contract_file.is_file() else None

    try:
        baseline_manifest = load_json_strict(baseline_dir / "manifest.json")
        candidate_manifest = load_json_strict(candidate_dir / "manifest.json")
        baseline_metrics = load_json_strict(baseline_dir / "summary/metrics.json")
        candidate_metrics = load_json_strict(candidate_dir / "summary/metrics.json")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        baseline_manifest = {}
        candidate_manifest = {}
        baseline_metrics = {}
        candidate_metrics = {}
        errors.append(f"failed to load paired run inputs: {exc}")

    if contract_digest is not None:
        for name, manifest in (("baseline", baseline_manifest), ("candidate", candidate_manifest)):
            if isinstance(manifest, dict) and manifest.get("frozen_contract_sha256") != contract_digest:
                errors.append(f"{name} frozen_contract_sha256 does not match --contract")

    comparison = contract.get("paired_comparison") if isinstance(contract, dict) else None
    if not isinstance(comparison, dict):
        comparison = {}
        errors.append("frozen contract is missing paired_comparison object")
    require_equal = comparison.get("require_equal")
    if not isinstance(require_equal, list) or not all(isinstance(item, str) for item in require_equal):
        require_equal = []
        errors.append("paired_comparison.require_equal must be a list of manifest field paths")
    required_compatibility = {
        "schema_version",
        "mode",
        "dataset_id",
        "odom_source",
        "query_source",
        "evidence_class",
        "dataset_manifest_sha256",
        "resolved_config_sha256",
        "candidate_budget",
        "icp_budget",
        "label_thresholds",
    }
    missing_compatibility = sorted(required_compatibility - set(require_equal))
    if missing_compatibility:
        errors.append(f"require_equal omits mandatory compatibility fields: {missing_compatibility}")
    compatibility: list[dict[str, Any]] = []
    for field in require_equal:
        try:
            baseline_value = _compatibility_value(baseline_manifest, field)
            candidate_value = _compatibility_value(candidate_manifest, field)
        except KeyError:
            compatibility.append(
                {"field": field, "equal": False, "baseline": None, "candidate": None}
            )
            errors.append(f"compatibility field {field!r} is missing")
            continue
        equal = baseline_value == candidate_value
        compatibility.append(
            {
                "field": field,
                "equal": equal,
                "baseline": baseline_value,
                "candidate": candidate_value,
            }
        )
        if not equal:
            errors.append(f"paired runs differ in compatibility field {field!r}")
    baseline_run_id = baseline_manifest.get("run_id")
    candidate_run_id = candidate_manifest.get("run_id")
    distinct_run_ids = (
        isinstance(baseline_run_id, str)
        and bool(baseline_run_id)
        and isinstance(candidate_run_id, str)
        and bool(candidate_run_id)
        and baseline_run_id != candidate_run_id
    )
    compatibility.append(
        {
            "field": "run_id_distinct",
            "equal": distinct_run_ids,
            "baseline": baseline_run_id,
            "candidate": candidate_run_id,
        }
    )
    if not distinct_run_ids:
        errors.append("paired runs must have distinct non-empty run_id values")

    synthetic_requirements = contract.get("synthetic_requirements", {})
    if not isinstance(synthetic_requirements, dict):
        synthetic_requirements = {}
        errors.append("synthetic_requirements must be an object when present")
    require_query_fingerprint = synthetic_requirements.get(
        "require_query_cloud_fingerprint_fnv1a64",
        False,
    )
    if not isinstance(require_query_fingerprint, bool):
        errors.append(
            "synthetic_requirements.require_query_cloud_fingerprint_fnv1a64 must be boolean"
        )
        require_query_fingerprint = False
    if require_query_fingerprint:
        baseline_fingerprints, baseline_fingerprint_errors = _read_query_fingerprints(
            baseline_dir
        )
        candidate_fingerprints, candidate_fingerprint_errors = _read_query_fingerprints(
            candidate_dir
        )
        errors.extend(f"baseline {message}" for message in baseline_fingerprint_errors)
        errors.extend(f"candidate {message}" for message in candidate_fingerprint_errors)
        equal_fingerprints = baseline_fingerprints == candidate_fingerprints
        compatibility.append(
            {
                "field": "per_query_fnv1a64",
                "equal": equal_fingerprints,
                "baseline": len(baseline_fingerprints),
                "candidate": len(candidate_fingerprints),
            }
        )
        if not equal_fingerprints:
            mismatched_ids = sorted(
                {
                    query_id
                    for query_id in set(baseline_fingerprints) | set(candidate_fingerprints)
                    if baseline_fingerprints.get(query_id) != candidate_fingerprints.get(query_id)
                }
            )
            errors.append(
                "paired runs differ in contract-required per_query_fnv1a64 evidence: "
                f"{mismatched_ids}"
            )

    metric_specs = comparison.get("metrics")
    if not isinstance(metric_specs, list) or not metric_specs:
        metric_specs = []
        errors.append("paired_comparison.metrics must be a non-empty list")
    rows: list[dict[str, Any]] = []
    for spec in metric_specs:
        if not isinstance(spec, dict):
            errors.append("paired metric entries must be objects")
            continue
        row, config_error = _metric_result(spec, baseline_metrics, candidate_metrics)
        rows.append(row)
        if config_error:
            errors.append(config_error)

    duplicate_metrics = [
        name for name, count in Counter(row["metric"] for row in rows).items() if count > 1
    ]
    if duplicate_metrics:
        errors.append(f"paired metrics contain duplicates: {sorted(duplicate_metrics)}")

    gate_reports: dict[str, dict[str, Any]] = {}
    for gate in GATE_NAMES:
        gate_rows = [row for row in rows if row.get("gate") == gate]
        if not gate_rows:
            errors.append(f"frozen contract has no {gate} metric")
            gate_reports[gate] = {"status": "INVALID", "metric_count": 0, "failed_metrics": []}
            continue
        invalid = [row["metric"] for row in gate_rows if row["status"] == "INVALID"]
        failed = [row["metric"] for row in gate_rows if row["status"] == "FAIL"]
        status = "INVALID" if invalid else ("FAIL" if failed else "PASS")
        gate_reports[gate] = {
            "status": status,
            "metric_count": len(gate_rows),
            "invalid_metrics": invalid,
            "failed_metrics": failed,
        }

    if errors or any(value["status"] == "INVALID" for value in gate_reports.values()):
        verdict = "INVALID"
        evidence_ceiling = "INVALID"
    elif any(value["status"] == "FAIL" for value in gate_reports.values()):
        verdict = "FAIL"
        evidence_ceiling = "FAIL"
    else:
        odom_source = candidate_manifest.get("odom_source")
        evidence_class = candidate_manifest.get("evidence_class")
        gt_runtime_access = candidate_manifest.get("gt_runtime_access")
        if (
            odom_source == FORMAL_ODOM_SOURCE
            and candidate_manifest.get("query_source") == FORMAL_QUERY_SOURCE
            and evidence_class == FORMAL_EVIDENCE_CLASS
            and gt_runtime_access is False
        ):
            verdict = "PASS"
            evidence_ceiling = "PASS"
        else:
            verdict = "SHADOW_ONLY"
            evidence_ceiling = "SHADOW_ONLY"
            if evidence_class == SYNTHETIC_EVIDENCE_CLASS:
                errors_for_note = "synthetic_map_render cannot establish formal product evidence"
            elif odom_source != FORMAL_ODOM_SOURCE:
                errors_for_note = "non-recorded-LIO odometry cannot establish formal product evidence"
            elif candidate_manifest.get("query_source") != FORMAL_QUERY_SOURCE:
                errors_for_note = "formal evidence requires independent_real_scan queries"
            else:
                errors_for_note = "formal evidence requires recorded_cross_session and GT isolation"
            gate_reports["evidence"] = {"status": "SHADOW_ONLY", "reason": errors_for_note}

    report = {
        "schema_version": SCHEMA_VERSION,
        "baseline_run": str(baseline_dir),
        "candidate_run": str(candidate_dir),
        "contract": str(contract_file),
        "contract_sha256": contract_digest,
        "verdict": verdict,
        "evidence_ceiling": evidence_ceiling,
        "errors": errors,
        "baseline_validation": baseline_validation,
        "candidate_validation": candidate_validation,
        "compatibility": compatibility,
        "gates": gate_reports,
    }
    _csv_dump_atomic(output / "paired_diff.csv", rows)
    _json_dump_atomic(output / "gate_report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = compare_runs(args.baseline, args.candidate, args.contract, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["verdict"] in {"PASS", "SHADOW_ONLY"} else 1


if __name__ == "__main__":
    sys.exit(main())
