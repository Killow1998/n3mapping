#!/usr/bin/env python3

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
from types import SimpleNamespace
import unittest
from unittest import mock

from product_test_elf import build_fake_product_elf


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "n3mapping_product_gate.py"
TOOLS_DIRECTORY = ROOT / "tools"
if str(TOOLS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIRECTORY))
import n3mapping_product_gate as GATE
FLOOR7_IDS = (
    "floor7_inside_710_room",
    "floor7_outside_707_room",
    "floor7_outside_710_room",
    "floor7_outside_713_room",
    "floor7_710toMeeting_static_70_75s",
)
POSE = {
    "tx": 1.0,
    "ty": 2.0,
    "tz": 0.3,
    "qx": 0.0,
    "qy": 0.0,
    "qz": 0.0,
    "qw": 1.0,
}
CANDIDATE_COMMIT = "a" * 40
PRODUCT_PROFILE_TEXT = "mode: localization\n"
PRODUCT_PROFILE_SHA256 = hashlib.sha256(
    PRODUCT_PROFILE_TEXT.encode("utf-8")
).hexdigest()
MANIFEST_HEADER = (
    "episode_id",
    "frame_index",
    "stamp_ns",
    "pcd_path",
    "tx",
    "ty",
    "tz",
    "qx",
    "qy",
    "qz",
    "qw",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pcd_set_sha256(manifest: Path) -> str:
    digest = hashlib.sha256()
    with manifest.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            pcd_path = row["pcd_path"]
            path = manifest.parent / pcd_path
            digest.update(pcd_path.encode("utf-8"))
            digest.update(b"\x00")
            digest.update(str(path.stat().st_size).encode("ascii"))
            digest.update(b"\x00")
            digest.update(sha256_file(path).encode("ascii"))
            digest.update(b"\n")
    return digest.hexdigest()


class ProductGateTest(unittest.TestCase):
    def test_evaluator_resources_are_measured_by_gate(self) -> None:
        completed, measured = GATE.run_evaluator_with_resources(
            [
                sys.executable,
                "-c",
                "payload = bytearray(16 * 1024 * 1024); print(len(payload))",
            ]
        )
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stdout.strip(), str(16 * 1024 * 1024))
        self.assertGreater(measured["peak_rss_kib"], 1024)
        self.assertEqual(measured["swap_operations"], 0)

    def bundle_path(self, root: Path, role: str) -> Path:
        return root / "bundles" / role

    def write_executable(self, path: Path, source: str) -> None:
        path.write_text(
            "#!/usr/bin/env python3\n" + textwrap.dedent(source),
            encoding="utf-8",
        )
        path.chmod(0o755)

    def make_tools(self, root: Path) -> tuple[Path, Path]:
        for role, map_bytes in (("floor7", b"floor7-map"), ("b22", b"b22-map")):
            bundle = self.bundle_path(root, role)
            (bundle / "calibration").mkdir(parents=True)
            (bundle / "map.pbstream").write_bytes(map_bytes)
            (bundle / "map.pbstream.localization_atlas.pb").write_bytes(
                role.encode("utf-8") + b"-atlas"
            )
            (bundle / "product_v1.yaml").write_text(
                PRODUCT_PROFILE_TEXT, encoding="utf-8"
            )
            (bundle / "calibration" / "lidar.yaml").write_text(
                "fixed: true\n", encoding="utf-8"
            )
        bundle_tool = root / "fake_bundle_tool.py"
        self.write_executable(
            bundle_tool,
            """
            import pathlib
            import sys
            bundle = pathlib.Path(sys.argv[sys.argv.index("--bundle") + 1])
            if (bundle / "tampered").exists():
                print("bundle content mismatch", file=sys.stderr)
                raise SystemExit(1)
            print('{"status":"verified"}')
            """,
        )
        preflight_tool = root / "fake_preflight_tool.py"
        self.write_executable(
            preflight_tool,
            """
            import pathlib
            import sys
            artifact = pathlib.Path(
                sys.argv[sys.argv.index("--artifact") + 1]
            )
            if not artifact.is_file():
                print("missing preflight artifact", file=sys.stderr)
                raise SystemExit(1)
            print('{"status":"PASS"}')
            """,
        )
        evaluator_impl = root / "fake_evaluator_impl.py"
        self.write_executable(
            evaluator_impl,
            """
            import argparse
            import csv
            import json
            import math
            import pathlib

            parser = argparse.ArgumentParser()
            parser.add_argument("--map", required=True)
            parser.add_argument("--atlas", required=True)
            parser.add_argument("--manifest", required=True)
            parser.add_argument("--output", required=True)
            parser.add_argument("--raw-first-stamp-ns", type=int)
            args = parser.parse_args()
            manifest = pathlib.Path(args.manifest)
            scenario = json.loads(
                manifest.with_name("scenario.json").read_text(encoding="utf-8")
            )
            with manifest.open(newline="", encoding="utf-8") as stream:
                frames = list(csv.DictReader(stream))
            strict = scenario.get(
                "strict_ms", [100.0 + 25.0 * i for i in range(len(frames))]
            )
            if len(strict) != len(frames):
                raise SystemExit("strict/frame count mismatch")
            output = pathlib.Path(args.output)
            output.mkdir(parents=True)
            output.joinpath("alignment.pcd").write_bytes(b"pcd")

            history_locked = scenario.get(
                "history_locked", scenario.get("locked", False)
            )
            hidden_history = scenario.get("hidden_history_lock", False)
            lock_frame = scenario.get("lock_frame", 2)
            seed_id = scenario.get("seed_id", 7)
            support_id = scenario.get("support_id", 9)
            pose = scenario.get(
                "pose",
                {
                    "tx": 1.0, "ty": 2.0, "tz": 0.3,
                    "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
                },
            )
            rows = []
            for index, frame in enumerate(frames):
                if hidden_history:
                    full = index == lock_frame
                else:
                    full = history_locked and index >= lock_frame
                state = "FULL_6DOF_LOCKED" if full else "PROVISIONAL"
                source = "GEOMETRICALLY_CORRECTED" if full else "NONE"
                if scenario.get("recently_lost_frame") == index:
                    state = "RECENTLY_LOST"
                    source = "ODOM_PREDICTED"
                if scenario.get("illegal_state_source") and index == 0:
                    source = "GEOMETRICALLY_CORRECTED"
                authoritative_full = state == "FULL_6DOF_LOCKED"
                usable_global_pose = (
                    authoritative_full or state == "RECENTLY_LOST"
                )
                edge = authoritative_full and index == lock_frame
                if scenario.get("missing_first_lock_edge") and index == lock_frame:
                    edge = False
                preprocessing = strict[index] * 0.4
                backend = strict[index] - preprocessing
                rows.append(
                    {
                        "frame_index": index,
                        "stamp_ns": int(frame["stamp_ns"]),
                        "success": int(usable_global_pose),
                        "relocalization_lock_edge": int(edge),
                        "authoritative_full": int(authoritative_full),
                        "seed_keyframe_id": seed_id if full else -1,
                        "support_keyframe_id": support_id if full else -1,
                        "matched_keyframe_id": support_id if full else -1,
                        "relocalization_state": state,
                        "pose_source": source,
                        "decision": "fake",
                        "preprocessing_ms": preprocessing,
                        "backend_ms": backend,
                        "strict_ms": strict[index],
                        **{f"map_{key}": value for key, value in pose.items()},
                    }
                )
            headers = [
                "frame_index", "stamp_ns", "success",
                "relocalization_lock_edge", "authoritative_full",
                "seed_keyframe_id", "support_keyframe_id",
                "matched_keyframe_id", "relocalization_state", "pose_source",
                "decision", "preprocessing_ms", "backend_ms", "strict_ms",
                "map_tx", "map_ty", "map_tz", "map_qx", "map_qy",
                "map_qz", "map_qw",
            ]
            with output.joinpath("frame_status.csv").open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=headers)
                writer.writeheader()
                writer.writerows(rows)

            result_locked = scenario.get(
                "result_locked", scenario.get("locked", False)
            )
            ordered = sorted(strict)
            position = 0.95 * (len(ordered) - 1)
            lower = int(position)
            upper = min(lower + 1, len(ordered) - 1)
            p95 = ordered[lower] + (position - lower) * (
                ordered[upper] - ordered[lower]
            )
            if scenario.get("nonfinite"):
                p95 = float("nan")
            acquisition_start = (
                args.raw_first_stamp_ns
                if args.raw_first_stamp_ns is not None
                else int(frames[0]["stamp_ns"])
            )
            result_lock_frame = lock_frame if result_locked else -1
            lock_stamp = (
                int(frames[result_lock_frame]["stamp_ns"])
                if result_locked else 0
            )
            final_state = scenario.get(
                "final_state_override", rows[-1]["relocalization_state"]
            )
            final_source = scenario.get(
                "final_source_override", rows[-1]["pose_source"]
            )
            result = {
                "schema": "n3mapping_product_gate_case_result_v1",
                "episode_id": scenario["id"],
                "map_path": str(pathlib.Path(args.map).resolve()),
                "manifest_path": str(manifest.resolve()),
                "input_frame_count": len(frames),
                "processed_frame_count": len(frames),
                "algorithm_lock": result_locked,
                "alignment_pose_authoritative": result_locked,
                "final_state": final_state,
                "final_pose_source": final_source,
                "final_decision": "fake",
                "manual_alignment_correct": None,
                "lock_frame_index": result_lock_frame,
                "lock_stamp_ns": lock_stamp,
                "acquisition_start_stamp_ns": acquisition_start,
                "acquisition_to_lock_s": (
                    scenario.get(
                        "reported_acquisition_s",
                        (lock_stamp - acquisition_start) / 1e9,
                    )
                    if result_locked else -1.0
                ),
                "matched_keyframe_id": support_id if result_locked else -1,
                "relocalization_seed_keyframe_id": (
                    seed_id if result_locked else -1
                ),
                "relocalization_support_keyframe_id": (
                    support_id if result_locked else -1
                ),
                "reported_map_body_pose": pose,
                "final_map_body_pose": pose,
                "reported_map_odom_transform": pose,
                "review_pcd": "review.pcd",
                "alignment_review_pcd": "alignment.pcd",
                "provenance_review_pcd": "provenance.pcd",
                "comparison_review_pcd": "",
                "performance": {
                    "map_load_ms": 10.0,
                    "strict_p95_ms": p95,
                    "strict_max_ms": max(strict),
                    "search_phase_p95_ms": p95,
                    "search_phase_max_ms": max(strict),
                    "runtime_peak_rss_kib": scenario.get("rss_kib", 100000),
                    "runtime_swap_operations": scenario.get("swap", 0),
                },
            }
            output.joinpath("result.json").write_text(
                json.dumps(result, allow_nan=True) + "\\n", encoding="utf-8"
            )
            raise SystemExit(0 if result_locked else 2)
            """,
        )
        runtime_node = build_fake_product_elf(
            root,
            "fake_runtime_node",
            commit=CANDIDATE_COMMIT,
            product_profile_sha256=PRODUCT_PROFILE_SHA256,
            runtime_node=True,
        )
        evaluator = build_fake_product_elf(
            root,
            "fake_evaluator",
            commit=CANDIDATE_COMMIT,
            product_profile_sha256=PRODUCT_PROFILE_SHA256,
            runtime_node=False,
            delegate_script=evaluator_impl,
        )
        for role in ("floor7", "b22"):
            bundle = self.bundle_path(root, role)
            (bundle / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema": "fake_test_bundle",
                        "n3mapping_commit": CANDIDATE_COMMIT,
                        "product_profile_sha256": PRODUCT_PROFILE_SHA256,
                        "atlas_format_version": "test-v1",
                        "runtime_node_sha256": sha256_file(runtime_node),
                        "runtime_node_linked_product_libraries": {
                            "libn3mapping_core.so": sha256_file(
                                root / "libn3mapping_core.so"
                            ),
                            "libn3mapping_humble_wrapper.so": sha256_file(
                                root / "libn3mapping_humble_wrapper.so"
                            ),
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        return bundle_tool, evaluator

    def write_input(
        self, root: Path, name: str, **scenario_values: object
    ) -> Path:
        directory = root / "inputs" / name
        directory.mkdir(parents=True)
        stamp_base = 10_000_000_000
        with (directory / "frames.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(MANIFEST_HEADER)
            for index in range(4):
                pcd_name = f"frame_{index:06d}.pcd"
                (directory / pcd_name).write_bytes(
                    f"{name}-pcd-{index}".encode("utf-8")
                )
                writer.writerow(
                    [
                        name,
                        index,
                        stamp_base + index * 1_000_000_000,
                        pcd_name,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                    ]
                )
        (directory / "scenario.json").write_text(
            json.dumps({"id": name, **scenario_values}) + "\n",
            encoding="utf-8",
        )
        return directory / "frames.csv"

    def baseline(self) -> dict[str, object]:
        return {
            "reported_map_body_pose": POSE,
            "lock_frame_index": 2,
            "relocalization_seed_keyframe_id": 7,
            "relocalization_support_keyframe_id": 9,
        }

    def case(
        self,
        case_id: str,
        manifest: Path,
        case_class: str,
        *,
        source_id: str | None = None,
        map_role: str = "floor7",
        negative_kind: str | None = None,
        baseline: dict[str, object] | None = None,
        raw_first_stamp_ns: int | None = None,
        reference: Path | None = None,
        evidence: Path | None = None,
    ) -> dict[str, object]:
        result: dict[str, object] = {
            "id": case_id,
            "source_id": source_id or case_id,
            "map_role": map_role,
            "manifest": str(manifest.relative_to(manifest.parents[2])),
            "class": case_class,
            "input_manifest_sha256": sha256_file(manifest),
            "input_pcd_set_sha256": pcd_set_sha256(manifest),
        }
        if baseline is not None:
            result["baseline"] = baseline
        if raw_first_stamp_ns is not None:
            result["raw_first_stamp_ns"] = raw_first_stamp_ns
        if negative_kind is not None:
            result["negative_kind"] = negative_kind
        if reference is not None:
            result["reference_trajectory_csv"] = str(
                reference.relative_to(reference.parents[2])
            )
            result["reference_trajectory_sha256"] = sha256_file(reference)
        if evidence is not None:
            result["reference_evidence_json"] = str(
                evidence.relative_to(evidence.parents[2])
            )
            result["reference_evidence_sha256"] = sha256_file(evidence)
        return result

    def write_reference_evidence(
        self, root: Path, manifest: Path
    ) -> tuple[Path, Path]:
        reference = manifest.with_name("reference_trajectory.csv")
        with manifest.open(newline="", encoding="utf-8") as stream:
            frames = list(csv.DictReader(stream))
        with reference.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("stamp_ns", *POSE.keys()))
            for frame in frames:
                writer.writerow((frame["stamp_ns"], *POSE.values()))
        query_bag = root / "query.bag"
        full_bag = root / "full.bag"
        query_bag.write_bytes(b"query raw bag")
        full_bag.write_bytes(b"full raw bag")
        overlap = {
            "reference_class": "same_frame_reference",
            "matched_payload_ratio": 1.0,
            "unique_payload_ratio": 1.0,
            "lidar_message_count": len(frames),
            "matched_payload_count": len(frames),
            "unique_payload_count": len(frames),
        }
        inventory = root / "inventory.json"
        inventory.write_text(
            json.dumps(
                {
                    "schema": "n3mapping_ros1_raw_bag_inventory_v1",
                    "input_directory": str(root.resolve()),
                    "file_hashes_included": True,
                    "full_bag": {
                        "file": full_bag.name,
                        "sha256": sha256_file(full_bag),
                    },
                    "query_bags": [
                        {
                            "file": query_bag.name,
                            "sha256": sha256_file(query_bag),
                            "payload_overlap": overlap,
                        }
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        dense_trajectory = root / "dense_trajectory.csv"
        with dense_trajectory.open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(
                ("stamp_ns", "tx", "ty", "tz", "qx", "qy", "qz", "qw", "seq")
            )
            for index, frame in enumerate(frames):
                writer.writerow(
                    (frame["stamp_ns"], *POSE.values(), index)
                )
        dense_evidence = root / "dense_evidence.json"
        dense_evidence.write_text(
            json.dumps(
                {
                    "schema": "n3mapping_dense_trajectory_evidence_v1",
                    "pbstream": str(
                        (
                            self.bundle_path(root, "floor7")
                            / "map.pbstream"
                        ).resolve()
                    ),
                    "pbstream_sha256": sha256_file(
                        self.bundle_path(root, "floor7") / "map.pbstream"
                    ),
                    "trajectory_csv": str(dense_trajectory.resolve()),
                    "trajectory_csv_sha256": sha256_file(dense_trajectory),
                    "row_count": len(frames),
                    "source": "native",
                    "degraded": False,
                    "pose_convention": "T_world_body",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        evidence = manifest.with_name("reference_evidence.json")
        evidence.write_text(
            json.dumps(
                {
                    "schema": "n3mapping_same_frame_reference_evidence_v1",
                    "reference_class": "same_run_dense_optimized_reference",
                    "inventory_payload_overlap": {
                        "matched_payload_ratio": 1.0,
                        "unique_payload_ratio": 1.0,
                        "lidar_message_count": len(frames),
                    },
                    "query_bag": str(query_bag.resolve()),
                    "query_bag_sha256": sha256_file(query_bag),
                    "inventory_query_bag_sha256": sha256_file(query_bag),
                    "full_bag": str(full_bag.resolve()),
                    "full_bag_sha256": sha256_file(full_bag),
                    "inventory": str(inventory.resolve()),
                    "inventory_sha256": sha256_file(inventory),
                    "dense_trajectory": str(dense_trajectory.resolve()),
                    "dense_trajectory_sha256": sha256_file(dense_trajectory),
                    "dense_evidence": str(dense_evidence.resolve()),
                    "dense_evidence_sha256": sha256_file(dense_evidence),
                    "lio_frames": str(manifest.resolve()),
                    "lio_frames_sha256": sha256_file(manifest),
                    "pbstream_sha256": sha256_file(
                        self.bundle_path(root, "floor7") / "map.pbstream"
                    ),
                    "pbstream": str(
                        (
                            self.bundle_path(root, "floor7")
                            / "map.pbstream"
                        ).resolve()
                    ),
                    "raw_lidar_count": len(frames),
                    "raw_first_header_stamp_ns": int(frames[0]["stamp_ns"]),
                    "raw_last_header_stamp_ns": int(frames[-1]["stamp_ns"]),
                    "lio_manifest_count": len(frames),
                    "skipped_raw_lidar_count": 0,
                    "lio_to_raw_coverage_ratio": 1.0,
                    "reference_pose_count": len(frames),
                    "dense_trajectory_count": len(frames),
                    "timestamp_alignment": {
                        "raw_to_lio_frame_index_inferred": False,
                        "fitted_time_offset": False,
                        "join_tolerance_ns": 512,
                        "acquisition_start_definition": (
                            "first query CustomMsg header.stamp_ns"
                        ),
                        "manifest_minus_dense_ns": {"min": 0, "max": 0},
                    },
                    "pose_convention": "T_world_body",
                    "pose_source": "native pbstream dense trajectory",
                    "output": {
                        "reference_trajectory": reference.name,
                        "reference_trajectory_sha256": sha256_file(reference),
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return reference, evidence

    def make_suite(
        self,
        root: Path,
        *,
        first_reference: bool = False,
    ) -> tuple[list[dict[str, object]], dict[str, Path]]:
        cases: list[dict[str, object]] = []
        manifests: dict[str, Path] = {}
        for index, case_id in enumerate(FLOOR7_IDS):
            manifest = self.write_input(root, case_id, locked=True)
            manifests[case_id] = manifest
            if index == 0 and first_reference:
                reference, evidence = self.write_reference_evidence(
                    root, manifest
                )
                cases.append(
                    self.case(
                        case_id,
                        manifest,
                        "required_lock",
                        reference=reference,
                        evidence=evidence,
                    )
                )
            else:
                cases.append(
                    self.case(
                        case_id,
                        manifest,
                        "required_lock",
                        baseline=self.baseline(),
                        raw_first_stamp_ns=10_000_000_000,
                    )
                )
        for case_id, source_id, map_role, negative_kind in (
            (
                "b22_query_on_floor7_wrong_map",
                "b22_query",
                "floor7",
                "wrong_map",
            ),
            (
                "f7tof9_query_on_b22_no_overlap",
                "f7tof9_query",
                "b22",
                "no_overlap",
            ),
        ):
            manifest = self.write_input(root, case_id, locked=False)
            manifests[case_id] = manifest
            cases.append(
                self.case(
                    case_id,
                    manifest,
                    "must_reject",
                    source_id=source_id,
                    map_role=map_role,
                    negative_kind=negative_kind,
                )
            )
        return cases, manifests

    def run_gate(
        self,
        root: Path,
        bundle_tool: Path,
        evaluator: Path,
        cases: list[dict[str, object]],
        *,
        evaluator_sha256: str | None = None,
        bundle_tool_sha256: str | None = None,
        runtime_preflight_sha256: str | None = None,
        authority_preflight_sha256: str | None = None,
        expected_dataset_sha256: str | None = None,
        authority_schema: str = (
            "n3mapping_product_active_runtime_authority_preflight_v2"
        ),
    ) -> subprocess.CompletedProcess[str]:
        map_artifacts: dict[str, object] = {}
        for role in ("floor7", "b22"):
            bundle = self.bundle_path(root, role)
            calibration = bundle / "calibration" / "lidar.yaml"
            map_artifacts[role] = {
                "map": {
                    "bytes": (bundle / "map.pbstream").stat().st_size,
                    "sha256": sha256_file(bundle / "map.pbstream"),
                },
                "atlas": {
                    "bytes": (
                        bundle / "map.pbstream.localization_atlas.pb"
                    ).stat().st_size,
                    "sha256": sha256_file(
                        bundle / "map.pbstream.localization_atlas.pb"
                    ),
                },
                "atlas_format_version": "test-v1",
                "product_profile_sha256": sha256_file(
                    bundle / "product_v1.yaml"
                ),
                "calibration_files": [
                    {
                        "path": "lidar.yaml",
                        "bytes": calibration.stat().st_size,
                        "sha256": sha256_file(calibration),
                    }
                ],
            }
        dataset_path = root / "dataset.json"
        dataset_path.write_text(
            json.dumps(
                {
                    "schema": "n3mapping_product_gate_dataset_v1",
                    "schema_version": 1,
                    "dataset_id": "unit_product_gate",
                    "dataset_revision": 1,
                    "frozen_at_shanghai": "2026-07-23T12:00:00+08:00",
                    "map_artifacts": map_artifacts,
                    "cases": cases,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        candidate_run_path = root / "candidate_run.json"
        runtime_preflight = root / "runtime_preflight.json"
        runtime_node = root / "fake_runtime_node"
        core_sha256 = sha256_file(root / "libn3mapping_core.so")
        node_libraries = {
            "libn3mapping_core.so": core_sha256,
            "libn3mapping_humble_wrapper.so": sha256_file(
                root / "libn3mapping_humble_wrapper.so"
            ),
        }
        runtime_preflight.write_text(
            json.dumps(
                {
                    "schema": "n3mapping_product_runtime_preflight_v1",
                    "status": "PASS",
                    "candidate_commit": CANDIDATE_COMMIT,
                    "product_profile_sha256": PRODUCT_PROFILE_SHA256,
                    "node": {
                        "path": str(runtime_node.resolve()),
                        "sha256": sha256_file(runtime_node),
                        "product_libraries": node_libraries,
                    },
                    "evaluator": {
                        "path": str(evaluator.resolve()),
                        "sha256": sha256_file(evaluator),
                        "product_libraries": {
                            "libn3mapping_core.so": core_sha256,
                        },
                    },
                    "bundles": [
                        {
                            "path": str(self.bundle_path(root, role).resolve()),
                            "manifest_sha256": sha256_file(
                                self.bundle_path(root, role)
                                / "manifest.json"
                            ),
                        }
                        for role in ("floor7", "b22")
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        authority_preflight = root / "authority_preflight.json"
        authority_preflight.write_text(
            json.dumps(
                {
                    "schema": authority_schema,
                    "schema_version": 2,
                    "evidence_kind": "active_runtime",
                    "status": "PASS",
                    "candidate_commit": CANDIDATE_COMMIT,
                    "product_profile_sha256": PRODUCT_PROFILE_SHA256,
                    "distro": "humble",
                    "source_files": {
                        "runtime_node": {
                            "path": str(runtime_node.resolve()),
                            "sha256": sha256_file(runtime_node),
                        }
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        preflight_tool = root / "fake_preflight_tool.py"
        candidate_run_path.write_text(
            json.dumps(
                {
                    "schema": "n3mapping_product_gate_candidate_run_v1",
                    "candidate_commit": CANDIDATE_COMMIT,
                    "evaluator_sha256": (
                        evaluator_sha256 or sha256_file(evaluator)
                    ),
                    "evaluator_linked_product_libraries": {
                        "libn3mapping_core.so": sha256_file(
                            root / "libn3mapping_core.so"
                        ),
                    },
                    "bundle_tool_sha256": (
                        bundle_tool_sha256 or sha256_file(bundle_tool)
                    ),
                    "preflight_tool_sha256": sha256_file(preflight_tool),
                    "runtime_preflight_json": str(runtime_preflight),
                    "runtime_preflight_sha256": (
                        runtime_preflight_sha256
                        or sha256_file(runtime_preflight)
                    ),
                    "authority_preflight_json": str(authority_preflight),
                    "authority_preflight_sha256": (
                        authority_preflight_sha256
                        or sha256_file(authority_preflight)
                    ),
                    "bundles": {
                        "floor7": str(self.bundle_path(root, "floor7")),
                        "b22": str(self.bundle_path(root, "b22")),
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return subprocess.run(
            [
                sys.executable,
                "-B",
                str(TOOL),
                "--candidate-run",
                str(candidate_run_path),
                "--dataset-contract",
                str(dataset_path),
                "--expected-dataset-sha256",
                expected_dataset_sha256 or sha256_file(dataset_path),
                "--evaluator",
                str(evaluator),
                "--bundle-tool",
                str(bundle_tool),
                "--preflight-tool",
                str(preflight_tool),
                "--output",
                str(root / "gate_output"),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    def load_report(self, root: Path) -> dict[str, object]:
        return json.loads(
            (root / "gate_output" / "gate_result.json").read_text(
                encoding="utf-8"
            )
        )

    def test_complete_suite_passes_and_writes_case_checksum_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle_tool, evaluator = self.make_tools(root)
            cases, _ = self.make_suite(root, first_reference=True)
            completed = self.run_gate(root, bundle_tool, evaluator, cases)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = self.load_report(root)
            self.assertEqual(report["status"], "COMPLETE")
            self.assertTrue(report["pass"])
            self.assertEqual(
                report["summary"]["product_positive_success_rate"], 1.0
            )
            self.assertEqual(
                report["summary"]["negative_false_lock_rate"], 0.0
            )
            first = report["cases"][0]
            self.assertEqual(first["raw_first_stamp_ns"], 10_000_000_000)
            self.assertEqual(
                first["reference_evidence"]["pbstream_sha256"],
                sha256_file(
                    self.bundle_path(root, "floor7") / "map.pbstream"
                ),
            )
            for case_report in report["cases"]:
                checksums = (
                    Path(case_report["output"]) / "checksums.sha256"
                )
                self.assertTrue(checksums.is_file())
                content = checksums.read_text(encoding="utf-8")
                self.assertIn("map.pbstream", content)
                self.assertIn("product_v1.yaml", content)
                self.assertIn("calibration/lidar.yaml", content)
                self.assertIn("frame_000000.pcd", content)
                self.assertIn("frame_status.csv", content)
                self.assertEqual(
                    case_report["checksums"]["outputs"]["checksums.sha256"],
                    sha256_file(checksums),
                )

    def test_consistent_negative_lock_is_a_complete_product_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle_tool, evaluator = self.make_tools(root)
            cases, manifests = self.make_suite(root)
            manifests["f7tof9_query_on_b22_no_overlap"].with_name(
                "scenario.json"
            ).write_text(
                json.dumps(
                    {
                        "id": "f7tof9_query_on_b22_no_overlap",
                        "locked": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            completed = self.run_gate(root, bundle_tool, evaluator, cases)
            self.assertEqual(completed.returncode, 2, completed.stderr)
            report = self.load_report(root)
            self.assertEqual(report["status"], "COMPLETE")
            self.assertEqual(
                report["summary"]["case_counts"]["negative_false_locks"], 1
            )
            negative = next(
                case
                for case in report["cases"]
                if case["id"] == "f7tof9_query_on_b22_no_overlap"
            )
            self.assertIn(
                "false_lock_on_must_reject_case",
                negative["semantic_failure_reasons"],
            )

    def test_hidden_historical_lock_cannot_pass_as_no_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle_tool, evaluator = self.make_tools(root)
            cases, manifests = self.make_suite(root)
            manifests["f7tof9_query_on_b22_no_overlap"].with_name(
                "scenario.json"
            ).write_text(
                json.dumps(
                    {
                        "id": "f7tof9_query_on_b22_no_overlap",
                        "locked": False,
                        "result_locked": False,
                        "hidden_history_lock": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            completed = self.run_gate(root, bundle_tool, evaluator, cases)
            self.assertEqual(completed.returncode, 1)
            report = self.load_report(root)
            self.assertEqual(report["status"], "INCOMPLETE")
            self.assertIn(
                "algorithm_lock disagrees with authoritative FULL lock history",
                report["error"],
            )

    def test_illegal_state_pose_source_is_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle_tool, evaluator = self.make_tools(root)
            cases, manifests = self.make_suite(root)
            target = manifests[FLOOR7_IDS[0]]
            target.with_name("scenario.json").write_text(
                json.dumps(
                    {
                        "id": FLOOR7_IDS[0],
                        "locked": True,
                        "illegal_state_source": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            completed = self.run_gate(root, bundle_tool, evaluator, cases)
            self.assertEqual(completed.returncode, 1)
            self.assertIn(
                "illegal state/pose_source combination",
                self.load_report(root)["error"],
            )

    def test_first_authoritative_full_requires_lock_edge(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle_tool, evaluator = self.make_tools(root)
            cases, manifests = self.make_suite(root)
            target = manifests[FLOOR7_IDS[0]]
            target.with_name("scenario.json").write_text(
                json.dumps(
                    {
                        "id": FLOOR7_IDS[0],
                        "locked": True,
                        "missing_first_lock_edge": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            completed = self.run_gate(root, bundle_tool, evaluator, cases)
            self.assertEqual(completed.returncode, 1)
            self.assertIn(
                "first authoritative FULL row is not a lock edge",
                self.load_report(root)["error"],
            )

    def test_pcd_tamper_breaks_the_frozen_input_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle_tool, evaluator = self.make_tools(root)
            cases, manifests = self.make_suite(root)
            (manifests[FLOOR7_IDS[0]].parent / "frame_000000.pcd").write_bytes(
                b"tampered"
            )
            completed = self.run_gate(root, bundle_tool, evaluator, cases)
            self.assertEqual(completed.returncode, 1)
            self.assertIn(
                "input PCD set SHA-256 mismatch",
                self.load_report(root)["error"],
            )

    def test_reference_evidence_tamper_is_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle_tool, evaluator = self.make_tools(root)
            cases, manifests = self.make_suite(root, first_reference=True)
            evidence = manifests[FLOOR7_IDS[0]].with_name(
                "reference_evidence.json"
            )
            value = json.loads(evidence.read_text(encoding="utf-8"))
            value["output"]["reference_trajectory_sha256"] = "0" * 64
            evidence.write_text(json.dumps(value) + "\n", encoding="utf-8")
            completed = self.run_gate(root, bundle_tool, evaluator, cases)
            self.assertEqual(completed.returncode, 1)
            self.assertIn(
                "reference evidence SHA-256 mismatch",
                self.load_report(root)["error"],
            )

    def test_reference_evidence_cannot_be_reused_for_another_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle_tool, evaluator = self.make_tools(root)
            cases, manifests = self.make_suite(root, first_reference=True)
            evidence = manifests[FLOOR7_IDS[0]].with_name(
                "reference_evidence.json"
            )
            other_manifest = manifests[
                "f7tof9_query_on_b22_no_overlap"
            ]
            value = json.loads(evidence.read_text(encoding="utf-8"))
            value["lio_frames"] = str(other_manifest.resolve())
            value["lio_frames_sha256"] = sha256_file(other_manifest)
            evidence.write_text(json.dumps(value) + "\n", encoding="utf-8")
            completed = self.run_gate(root, bundle_tool, evaluator, cases)
            self.assertEqual(completed.returncode, 1)
            self.assertIn(
                "reference evidence SHA-256 mismatch",
                self.load_report(root)["error"],
            )

    def test_reference_raw_bag_bytes_are_rehashed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle_tool, evaluator = self.make_tools(root)
            cases, _ = self.make_suite(root, first_reference=True)
            (root / "query.bag").write_bytes(b"different raw bag")
            completed = self.run_gate(root, bundle_tool, evaluator, cases)
            self.assertEqual(completed.returncode, 1)
            self.assertIn(
                "query bag SHA-256 mismatch",
                self.load_report(root)["error"],
            )

    def test_raw_acquisition_start_cannot_postdate_first_lio_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle_tool, evaluator = self.make_tools(root)
            cases, _ = self.make_suite(root)
            cases[0]["raw_first_stamp_ns"] = 10_000_000_001
            completed = self.run_gate(root, bundle_tool, evaluator, cases)
            self.assertEqual(completed.returncode, 1)
            self.assertIn(
                "raw acquisition start is later",
                self.load_report(root)["error"],
            )

    def test_evaluator_cannot_lie_about_acquisition_to_lock_duration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle_tool, evaluator = self.make_tools(root)
            cases, manifests = self.make_suite(root)
            target = manifests[FLOOR7_IDS[0]]
            target.with_name("scenario.json").write_text(
                json.dumps(
                    {
                        "id": FLOOR7_IDS[0],
                        "locked": True,
                        "reported_acquisition_s": 0.001,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            completed = self.run_gate(root, bundle_tool, evaluator, cases)
            self.assertEqual(completed.returncode, 1)
            self.assertIn(
                "disagrees with trusted integer-nanosecond timestamps",
                self.load_report(root)["error"],
            )

    def test_suite_must_include_all_floor7_and_both_negative_kinds(self) -> None:
        for removed, expected in (
            (FLOOR7_IDS[-1], "missing required floor7 positive cases"),
            (
                "b22_query_on_floor7_wrong_map",
                "missing required exact negative cases",
            ),
        ):
            with self.subTest(removed=removed):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    bundle_tool, evaluator = self.make_tools(root)
                    cases, _ = self.make_suite(root)
                    cases = [
                        case for case in cases if case["id"] != removed
                    ]
                    completed = self.run_gate(
                        root, bundle_tool, evaluator, cases
                    )
                    self.assertEqual(completed.returncode, 1)
                    self.assertIn(
                        expected, self.load_report(root)["error"]
                    )

    def test_dataset_contract_requires_external_trusted_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle_tool, evaluator = self.make_tools(root)
            cases, _ = self.make_suite(root)
            completed = self.run_gate(
                root,
                bundle_tool,
                evaluator,
                cases,
                expected_dataset_sha256="0" * 64,
            )
            self.assertEqual(completed.returncode, 1)
            self.assertIn(
                "externally supplied trusted digest",
                self.load_report(root)["error"],
            )

    def test_required_negative_source_to_map_pair_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle_tool, evaluator = self.make_tools(root)
            cases, _ = self.make_suite(root)
            target = next(
                case
                for case in cases
                if case["id"] == "b22_query_on_floor7_wrong_map"
            )
            target["map_role"] = "b22"
            completed = self.run_gate(root, bundle_tool, evaluator, cases)
            self.assertEqual(completed.returncode, 1)
            self.assertIn(
                "does not match the frozen product contract",
                self.load_report(root)["error"],
            )

    def test_replay_evidence_path_and_hash_are_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle_tool, evaluator = self.make_tools(root)
            cases, _ = self.make_suite(root, first_reference=True)
            cases[0]["raw_lio_replay_evidence_json"] = "missing.json"
            completed = self.run_gate(root, bundle_tool, evaluator, cases)
            self.assertEqual(completed.returncode, 1)
            self.assertIn(
                "replay evidence path and SHA-256 must be provided together",
                self.load_report(root)["error"],
            )

    def test_negative_replay_evidence_is_verified_without_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle_tool, evaluator = self.make_tools(root)
            cases, manifests = self.make_suite(root)
            case = next(
                item
                for item in cases
                if item["id"] == "b22_query_on_floor7_wrong_map"
            )
            invalid_evidence = manifests[
                "b22_query_on_floor7_wrong_map"
            ].with_name("invalid_raw_lio_replay_evidence.json")
            invalid_evidence.write_text("{}\n", encoding="utf-8")
            case["raw_lio_replay_evidence_json"] = str(
                invalid_evidence.relative_to(root)
            )
            case["raw_lio_replay_evidence_sha256"] = sha256_file(
                invalid_evidence
            )
            completed = self.run_gate(root, bundle_tool, evaluator, cases)
            self.assertEqual(completed.returncode, 1)
            error = self.load_report(root)["error"]
            self.assertIsInstance(error, str, completed.stderr)
            self.assertIn(
                "raw-to-LIO replay evidence verification failed", error
            )
            self.assertNotIn("requires a reference", error)

    def test_negative_replay_evidence_binds_actual_manifest_and_pcd_set(
        self,
    ) -> None:
        class FakeReplayEvidenceError(RuntimeError):
            pass

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.write_input(
                root, "negative_evidence_unit", locked=False
            )
            fingerprint = GATE.fingerprint_input_manifest(manifest)
            evidence = root / "raw_lio_replay_evidence.json"
            source_first = fingerprint["first_stamp_ns"] - 1_000_000
            source_last = fingerprint["last_stamp_ns"] + 1_000_000
            document = {
                "spec": {
                    "source_ros1_bag": str((root / "query.bag").resolve()),
                    "converted_ros2_bag": str(
                        (root / "query_ros2").resolve()
                    ),
                    "lio_output_ros2_bag": str(
                        (root / "query_lio").resolve()
                    ),
                    "extractor_output": str(manifest.parent.resolve()),
                    "source_lidar_topic": "/livox/lidar",
                    "cloud_topic": "/cloud_registered_body",
                },
                "bags": {
                    "source_ros1": {
                        "topics": {
                            "/livox/lidar": {
                                "message_count": (
                                    fingerprint["frame_count"] + 3
                                ),
                                "first_header_stamp_ns": source_first,
                                "last_header_stamp_ns": source_last,
                            }
                        }
                    },
                    "lio_output_ros2": {
                        "topics": {
                            "/cloud_registered_body": {
                                "message_count": fingerprint["frame_count"],
                                "first_header_stamp_ns": fingerprint[
                                    "first_stamp_ns"
                                ],
                                "last_header_stamp_ns": fingerprint[
                                    "last_stamp_ns"
                                ],
                            }
                        }
                    },
                },
                "extractor": {
                    "frames_csv": {
                        "path": str(manifest.resolve()),
                        "sha256": fingerprint["manifest_sha256"],
                    },
                    "pcd_set_sha256": fingerprint["pcd_set_sha256"],
                    "frame_count": fingerprint["frame_count"],
                },
            }
            evidence.write_text(
                json.dumps(document) + "\n", encoding="utf-8"
            )
            fake_module = SimpleNamespace(
                ReplayEvidenceError=FakeReplayEvidenceError,
                verify_evidence=lambda path: {
                    "evidence_sha256": sha256_file(path),
                    "verdict": "PASS",
                },
            )
            with mock.patch.dict(
                sys.modules,
                {"n3mapping_raw_lio_replay_evidence": fake_module},
            ):
                verified = GATE.validate_raw_lio_replay_evidence(
                    evidence, manifest, fingerprint, None
                )
                self.assertEqual(
                    verified["raw_first_stamp_ns"], source_first
                )
                self.assertEqual(
                    verified["raw_lidar_count"],
                    fingerprint["frame_count"] + 3,
                )

                document["extractor"]["pcd_set_sha256"] = "0" * 64
                evidence.write_text(
                    json.dumps(document) + "\n", encoding="utf-8"
                )
                with self.assertRaisesRegex(
                    GATE.GateError,
                    "extractor output is not the Gate input",
                ):
                    GATE.validate_raw_lio_replay_evidence(
                        evidence, manifest, fingerprint, None
                    )

    def test_full_se3_accuracy_rejects_vertical_and_roll_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle_tool, evaluator = self.make_tools(root)
            cases, manifests = self.make_suite(root)
            angle = math.radians(4.0)
            wrong_pose = {
                **POSE,
                "tz": POSE["tz"] + 0.6,
                "qx": math.sin(angle / 2.0),
                "qw": math.cos(angle / 2.0),
            }
            manifests[FLOOR7_IDS[0]].with_name("scenario.json").write_text(
                json.dumps(
                    {
                        "id": FLOOR7_IDS[0],
                        "locked": True,
                        "pose": wrong_pose,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            completed = self.run_gate(root, bundle_tool, evaluator, cases)
            self.assertEqual(completed.returncode, 2, completed.stderr)
            report = self.load_report(root)
            first = report["cases"][0]
            self.assertGreater(
                first["baseline_score"]["pose"]["translation_error_m"], 0.5
            )
            self.assertGreater(
                first["baseline_score"]["pose"]["roll_error_deg"], 2.0
            )
            self.assertIn(
                "locked_pose_outside_accuracy_contract",
                first["semantic_failure_reasons"],
            )

    def test_frozen_lock_frame_and_keyframe_ids_are_gate_authority(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle_tool, evaluator = self.make_tools(root)
            cases, manifests = self.make_suite(root)
            manifests[FLOOR7_IDS[0]].with_name("scenario.json").write_text(
                json.dumps(
                    {
                        "id": FLOOR7_IDS[0],
                        "locked": True,
                        "lock_frame": 1,
                        "seed_id": 8,
                        "support_id": 10,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            completed = self.run_gate(root, bundle_tool, evaluator, cases)
            self.assertEqual(completed.returncode, 2, completed.stderr)
            first = self.load_report(root)["cases"][0]
            self.assertFalse(
                first["baseline_score"]["lock_frame"]["exact_match"]
            )
            self.assertFalse(
                first["baseline_score"][
                    "relocalization_seed_keyframe"
                ]["exact_match"]
            )
            self.assertFalse(
                first["baseline_score"][
                    "relocalization_support_keyframe"
                ]["exact_match"]
            )
            self.assertIn(
                "frozen_baseline_identity_mismatch",
                first["semantic_failure_reasons"],
            )

    def test_recently_lost_does_not_inherit_legacy_success_authority(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle_tool, evaluator = self.make_tools(root)
            cases, manifests = self.make_suite(root)
            manifests[FLOOR7_IDS[0]].with_name("scenario.json").write_text(
                json.dumps(
                    {
                        "id": FLOOR7_IDS[0],
                        "locked": True,
                        "recently_lost_frame": 3,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            completed = self.run_gate(root, bundle_tool, evaluator, cases)
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_bundle_tamper_leaves_incomplete_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle_tool, evaluator = self.make_tools(root)
            cases, _ = self.make_suite(root)
            (
                self.bundle_path(root, "floor7") / "tampered"
            ).write_text(
                "yes\n", encoding="utf-8"
            )
            completed = self.run_gate(root, bundle_tool, evaluator, cases)
            self.assertEqual(completed.returncode, 1)
            report = self.load_report(root)
            self.assertEqual(report["status"], "INCOMPLETE")
            self.assertIn("verification failed", report["error"])

    def test_nonfinite_evaluator_result_is_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle_tool, evaluator = self.make_tools(root)
            cases, manifests = self.make_suite(root)
            manifests[FLOOR7_IDS[0]].with_name("scenario.json").write_text(
                json.dumps(
                    {
                        "id": FLOOR7_IDS[0],
                        "locked": True,
                        "nonfinite": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            completed = self.run_gate(root, bundle_tool, evaluator, cases)
            self.assertEqual(completed.returncode, 1)
            self.assertIn(
                "non-finite JSON number", self.load_report(root)["error"]
            )

    def test_frozen_tool_identity_rejects_a_different_evaluator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle_tool, evaluator = self.make_tools(root)
            cases, _ = self.make_suite(root)
            completed = self.run_gate(
                root,
                bundle_tool,
                evaluator,
                cases,
                evaluator_sha256="0" * 64,
            )
            self.assertEqual(completed.returncode, 1)
            self.assertIn(
                "evaluator SHA-256 does not match",
                self.load_report(root)["error"],
            )

    def test_tampered_preflight_evidence_leaves_gate_incomplete(self) -> None:
        for field in (
            "runtime_preflight_sha256",
            "authority_preflight_sha256",
        ):
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    bundle_tool, evaluator = self.make_tools(root)
                    cases, _ = self.make_suite(root)
                    completed = self.run_gate(
                        root,
                        bundle_tool,
                        evaluator,
                        cases,
                        **{field: "0" * 64},
                    )
                    self.assertEqual(completed.returncode, 1)
                    report = self.load_report(root)
                    self.assertEqual(report["status"], "INCOMPLETE")
                    self.assertIn(
                        "preflight SHA-256 mismatch",
                        report["error"],
                    )

    def test_synthetic_authority_evidence_leaves_gate_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle_tool, evaluator = self.make_tools(root)
            cases, _ = self.make_suite(root)
            completed = self.run_gate(
                root,
                bundle_tool,
                evaluator,
                cases,
                authority_schema="n3mapping_product_authority_preflight_v1",
            )
            self.assertEqual(completed.returncode, 1)
            report = self.load_report(root)
            self.assertEqual(report["status"], "INCOMPLETE")
            self.assertEqual(report["product_verdict"], "INCOMPLETE")
            self.assertIn(
                "active-runtime authority preflight is missing",
                report["error"],
            )

    def test_bundle_commit_must_match_the_frozen_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle_tool, evaluator = self.make_tools(root)
            cases, _ = self.make_suite(root)
            manifest = self.bundle_path(root, "floor7") / "manifest.json"
            value = json.loads(manifest.read_text(encoding="utf-8"))
            value["n3mapping_commit"] = "b" * 40
            manifest.write_text(
                json.dumps(value) + "\n", encoding="utf-8"
            )
            completed = self.run_gate(
                root, bundle_tool, evaluator, cases
            )
            self.assertEqual(completed.returncode, 1)
            self.assertIn(
                "Product Bundle commit does not match",
                self.load_report(root)["error"],
            )

    def test_rejects_algorithm_threshold_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle_tool, evaluator = self.make_tools(root)
            cases, _ = self.make_suite(root)
            cases[0]["matcher_score_threshold"] = 0.5
            completed = self.run_gate(root, bundle_tool, evaluator, cases)
            self.assertEqual(completed.returncode, 1)
            self.assertIn(
                "algorithm scoring/threshold key is forbidden",
                self.load_report(root)["error"],
            )


if __name__ == "__main__":
    unittest.main()
