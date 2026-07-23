#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True

from product_test_elf import build_fake_product_elf


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
TOOL = TOOLS / "n3mapping_product_preflight.py"
sys.path.insert(0, str(TOOLS))
import n3mapping_product_preflight as preflight  # noqa: E402


COMMIT = "7" * 40
PROFILE_BYTES = b"mode: localization\nproduct_profile_v1: true\n"
PROFILE_SHA256 = hashlib.sha256(PROFILE_BYTES).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def run_checked(command: list[str]) -> None:
    completed = subprocess.run(
        command, check=False, capture_output=True, text=True
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stdout + completed.stderr)


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(
            value, indent=2, sort_keys=True, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )


def pose(stamp_ns: int, x: float) -> dict[str, object]:
    return {
        "stamp_ns": stamp_ns,
        "frame_id": "map",
        "position": [x, 0.0, 0.0],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }


def pose_matrix(valid: bool) -> list[float]:
    x_scale = 1.0 if valid else 2.0
    return [
        x_scale,
        0.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        2.0,
        0.0,
        0.0,
        1.0,
        3.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]


def event(
    label: str,
    *,
    backend_state: str,
    backend_source: str,
    backend_valid: bool,
    backend_lock: bool,
    status_state: str | None,
    status_source: str | None,
    epoch: int,
    authority: list[dict[str, object]],
    legacy: list[int],
    global_poses: list[dict[str, object]],
    world_cloud_count: int,
) -> dict[str, object]:
    statuses: list[dict[str, object]] = []
    if status_state is not None and status_source is not None:
        statuses.append(
            {
                "state": status_state,
                "pose_source": status_source,
                "lock_epoch": epoch,
            }
        )
    return {
        "label": label,
        "backend": {
            "state": backend_state,
            "pose_source": backend_source,
            "pose_matrix": pose_matrix(backend_valid),
            "lock_event": backend_lock,
        },
        "observed": {
            "statuses": statuses,
            "authoritative_poses": authority,
            "legacy_lock_epochs": legacy,
            "global_poses": global_poses,
            "world_cloud_count": world_cloud_count,
        },
    }


class ProductPreflightTest(unittest.TestCase):
    def run_tool(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-B", str(TOOL), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )

    def make_binaries(self, root: Path) -> tuple[Path, Path]:
        binaries = root / "install" / "lib" / "n3mapping"
        node = build_fake_product_elf(
            binaries,
            "n3mapping_node",
            commit=COMMIT,
            product_profile_sha256=PROFILE_SHA256,
            runtime_node=True,
        )
        evaluator = build_fake_product_elf(
            binaries,
            "n3mapping_relocalization_manifest_eval",
            commit=COMMIT,
            product_profile_sha256=PROFILE_SHA256,
            runtime_node=False,
        )
        return node, evaluator

    def test_binary_identity_rejects_non_product_compile_flags(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            debug_node = build_fake_product_elf(
                root / "debug",
                "n3mapping_node",
                commit=COMMIT,
                product_profile_sha256=PROFILE_SHA256,
                runtime_node=True,
                build_type="Debug",
            )
            with self.assertRaisesRegex(
                preflight.PreflightError, "not Release"
            ):
                preflight.inspect_binary(debug_node, "runtime node")

            research_node = build_fake_product_elf(
                root / "research",
                "n3mapping_node",
                commit=COMMIT,
                product_profile_sha256=PROFILE_SHA256,
                runtime_node=True,
                research_tools="ON",
            )
            with self.assertRaisesRegex(
                preflight.PreflightError, "includes research tools"
            ):
                preflight.inspect_binary(research_node, "runtime node")

    def make_cmake_cache(
        self,
        root: Path,
        node: Path,
        *,
        research_tools: str = "OFF",
        build_type: str = "Release",
        product_commit: str = COMMIT,
    ) -> tuple[Path, Path]:
        install_prefix = node.parents[2]
        build = root / "build"
        build.mkdir(exist_ok=True)
        cache = build / "CMakeCache.txt"
        cache.write_text(
            "\n".join(
                [
                    "# Product V1 synthetic CMake cache",
                    f"CMAKE_BUILD_TYPE:STRING={build_type}",
                    f"CMAKE_INSTALL_PREFIX:PATH={install_prefix}",
                    "CMAKE_PROJECT_NAME:STATIC=n3mapping",
                    (
                        "N3MAPPING_BUILD_RESEARCH_TOOLS:BOOL="
                        f"{research_tools}"
                    ),
                    f"N3MAPPING_PRODUCT_COMMIT:STRING={product_commit}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return cache, install_prefix

    def make_bundle(self, root: Path, node: Path, name: str = "bundle") -> Path:
        bundle = root / name
        (bundle / "calibration").mkdir(parents=True)
        (bundle / "map.pbstream").write_bytes(b"map")
        (bundle / "map.pbstream.localization_atlas.pb").write_bytes(b"atlas")
        (bundle / "product_v1.yaml").write_bytes(PROFILE_BYTES)
        (bundle / "calibration" / "mid360.yaml").write_text(
            "extrinsic: identity\n", encoding="utf-8"
        )
        node_evidence = preflight.inspect_binary(node, "runtime node")
        files = []
        for path in sorted(bundle.rglob("*")):
            if path.is_file():
                files.append(
                    {
                        "path": path.relative_to(bundle).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
        manifest = {
            "schema": "n3mapping_product_map_bundle_v1",
            "schema_version": 1,
            "created_at": "2026-07-23T12:00:00+08:00",
            "n3mapping_commit": COMMIT,
            "runtime_node_sha256": sha256_file(node),
            "runtime_node_linked_product_libraries": node_evidence[
                "product_libraries"
            ],
            "lidar_model": "Mid-360",
            "lidar_message_type": "livox_ros_driver/CustomMsg",
            "atlas_format_version": "n3mapping_localization_atlas_v1",
            "atlas_generation_command": "n3mapping atlas compile",
            "product_profile_sha256": PROFILE_SHA256,
            "required_roles": {
                "map": "map.pbstream",
                "atlas": "map.pbstream.localization_atlas.pb",
                "config": "product_v1.yaml",
            },
            "files": files,
        }
        write_json(bundle / "manifest.json", manifest)
        return bundle

    def make_observation(
        self,
        root: Path,
        node: Path,
        *,
        distro: str = "humble",
        trusted_runner: bool = False,
    ) -> tuple[Path, dict[str, object]]:
        harness = (
            TOOLS / "n3mapping_product_authority_runner.py"
            if trusted_runner
            else root / "authority_harness.py"
        )
        log = root / "authority_raw.jsonl"
        if not trusted_runner:
            harness.write_text("# caller supplied harness\n", encoding="utf-8")
        log.write_text('{"event":"raw"}\n', encoding="utf-8")

        first_pose = pose(100, 1.0)
        steady_pose = pose(200, 1.1)
        degraded_pose = pose(300, 1.2)
        recovered_pose = pose(400, 1.3)
        extension_pose = pose(500, 2.0)
        observation: dict[str, object] = {
            "schema": "n3mapping_authority_observation_v1",
            "schema_version": 1,
            "distro": distro,
            "candidate_commit": COMMIT,
            "product_profile_sha256": PROFILE_SHA256,
            "files": {
                "runtime_node": {
                    "path": str(node),
                    "sha256": sha256_file(node),
                },
                "harness": {
                    "path": str(harness),
                    "sha256": sha256_file(harness),
                },
                "log": {
                    "path": str(log),
                    "sha256": sha256_file(log),
                },
            },
            "topics": preflight.expected_topics(distro),
            "scenarios": [
                {
                    "id": "localization_authority_sequence",
                    "run_mode": "LOCALIZATION",
                    "events": [
                        event(
                            "searching",
                            backend_state="SEARCHING",
                            backend_source="NONE",
                            backend_valid=True,
                            backend_lock=False,
                            status_state="SEARCHING",
                            status_source="NONE",
                            epoch=0,
                            authority=[],
                            legacy=[],
                            global_poses=[],
                            world_cloud_count=0,
                        ),
                        event(
                            "region_hypothesis",
                            backend_state="REGION_HYPOTHESIS",
                            backend_source="NONE",
                            backend_valid=True,
                            backend_lock=False,
                            status_state="REGION_HYPOTHESIS",
                            status_source="NONE",
                            epoch=0,
                            authority=[],
                            legacy=[],
                            global_poses=[],
                            world_cloud_count=0,
                        ),
                        event(
                            "first_full_lock",
                            backend_state="FULL_6DOF_LOCKED",
                            backend_source="GEOMETRICALLY_CORRECTED",
                            backend_valid=True,
                            backend_lock=True,
                            status_state="FULL_6DOF_LOCKED",
                            status_source="GEOMETRICALLY_CORRECTED",
                            epoch=1,
                            authority=[copy.deepcopy(first_pose)],
                            legacy=[1],
                            global_poses=[copy.deepcopy(first_pose)],
                            world_cloud_count=1,
                        ),
                        event(
                            "steady_full_lock",
                            backend_state="FULL_6DOF_LOCKED",
                            backend_source="GEOMETRICALLY_CORRECTED",
                            backend_valid=True,
                            backend_lock=False,
                            status_state="FULL_6DOF_LOCKED",
                            status_source="GEOMETRICALLY_CORRECTED",
                            epoch=1,
                            authority=[],
                            legacy=[],
                            global_poses=[steady_pose],
                            world_cloud_count=1,
                        ),
                        event(
                            "degraded_tracking",
                            backend_state="DEGRADED_TRACKING",
                            backend_source="ODOM_PREDICTED",
                            backend_valid=True,
                            backend_lock=False,
                            status_state="DEGRADED_TRACKING",
                            status_source="ODOM_PREDICTED",
                            epoch=1,
                            authority=[],
                            legacy=[],
                            global_poses=[degraded_pose],
                            world_cloud_count=1,
                        ),
                        event(
                            "recovered_full_lock",
                            backend_state="FULL_6DOF_LOCKED",
                            backend_source="GEOMETRICALLY_CORRECTED",
                            backend_valid=True,
                            backend_lock=True,
                            status_state="FULL_6DOF_LOCKED",
                            status_source="GEOMETRICALLY_CORRECTED",
                            epoch=2,
                            authority=[copy.deepcopy(recovered_pose)],
                            legacy=[2],
                            global_poses=[copy.deepcopy(recovered_pose)],
                            world_cloud_count=1,
                        ),
                    ],
                },
                {
                    "id": "invalid_pose_suppression",
                    "run_mode": "LOCALIZATION",
                    "events": [
                        event(
                            "invalid_full_pose",
                            backend_state="FULL_6DOF_LOCKED",
                            backend_source="GEOMETRICALLY_CORRECTED",
                            backend_valid=False,
                            backend_lock=True,
                            status_state="SEARCHING",
                            status_source="NONE",
                            epoch=0,
                            authority=[],
                            legacy=[],
                            global_poses=[],
                            world_cloud_count=0,
                        )
                    ],
                },
                {
                    "id": "map_extension_legacy_compatibility",
                    "run_mode": "MAP_EXTENSION",
                    "events": [
                        event(
                            "map_extension_lock",
                            backend_state="FULL_6DOF_LOCKED",
                            backend_source="GEOMETRICALLY_CORRECTED",
                            backend_valid=True,
                            backend_lock=True,
                            status_state=None,
                            status_source=None,
                            epoch=0,
                            authority=[],
                            legacy=[1],
                            global_poses=[extension_pose],
                            world_cloud_count=1,
                        )
                    ],
                },
            ],
        }
        observation_path = root / "authority_observation.json"
        write_json(observation_path, observation)
        return observation_path, observation

    def test_runtime_create_verify_records_exact_elf_and_bundle_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            node, evaluator = self.make_binaries(root)
            cache, install_prefix = self.make_cmake_cache(root, node)
            first_bundle = self.make_bundle(root, node, "bundle_a")
            second_bundle = self.make_bundle(root, node, "bundle_b")
            artifact = root / "runtime_preflight.json"
            created = self.run_tool(
                "runtime-create",
                "--node",
                str(node),
                "--evaluator",
                str(evaluator),
                "--bundle",
                str(first_bundle),
                "--bundle",
                str(second_bundle),
                "--cmake-cache",
                str(cache),
                "--install-prefix",
                str(install_prefix),
                "--candidate-commit",
                COMMIT,
                "--product-profile-sha256",
                PROFILE_SHA256,
                "--output",
                str(artifact),
            )
            self.assertEqual(created.returncode, 0, created.stderr)
            evidence = json.loads(artifact.read_text(encoding="utf-8"))
            self.assertEqual(evidence["status"], "PASS")
            self.assertTrue(evidence["created_at"].endswith("+08:00"))
            self.assertEqual(len(evidence["bundles"]), 2)
            self.assertEqual(
                evidence["cmake_cache"]["required_entries"][
                    "N3MAPPING_BUILD_RESEARCH_TOOLS"
                ],
                {"type": "BOOL", "value": "OFF"},
            )
            installed_paths = {
                entry["path"] for entry in evidence["install_prefix"]["files"]
            }
            self.assertIn(
                node.relative_to(install_prefix).as_posix(),
                installed_paths,
            )
            self.assertIn(
                evaluator.relative_to(install_prefix).as_posix(),
                installed_paths,
            )
            self.assertTrue(evidence["node"]["ldd_lines"])
            self.assertEqual(
                evidence["node"]["product_libraries"][
                    "libn3mapping_core.so"
                ],
                evidence["evaluator"]["product_libraries"][
                    "libn3mapping_core.so"
                ],
            )
            for dependency in evidence["node"]["dependencies"]:
                self.assertEqual(
                    dependency["sha256"],
                    sha256_file(Path(dependency["path"])),
                )

            verified = self.run_tool(
                "runtime-verify", "--artifact", str(artifact)
            )
            self.assertEqual(verified.returncode, 0, verified.stderr)

            added_install_file = (
                install_prefix / "share" / "n3mapping" / "allowed_extra.txt"
            )
            added_install_file.parent.mkdir(parents=True, exist_ok=True)
            added_install_file.write_text("changed inventory\n", encoding="utf-8")
            rejected = self.run_tool(
                "runtime-verify", "--artifact", str(artifact)
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("evidence changed", rejected.stderr)

    def test_runtime_rejects_different_core_forbidden_dependency_and_symlink(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            node, evaluator = self.make_binaries(root)
            cache, install_prefix = self.make_cmake_cache(root, node)
            bundle = self.make_bundle(root, node)

            other = install_prefix / "other"
            other.mkdir()
            core_source = other / "core.c"
            core_source.write_text(
                "int n3mapping_fake_core(void) { return 11; }\n",
                encoding="utf-8",
            )
            run_checked(
                [
                    "cc",
                    "-shared",
                    "-fPIC",
                    "-Wl,-soname,libn3mapping_core.so",
                    "-o",
                    str(other / "libn3mapping_core.so"),
                    str(core_source),
                ]
            )
            other_evaluator = build_fake_product_elf(
                other,
                "different_core_eval",
                commit=COMMIT,
                product_profile_sha256=PROFILE_SHA256,
                runtime_node=False,
            )
            rejected_core = self.run_tool(
                "runtime-create",
                "--node",
                str(node),
                "--evaluator",
                str(other_evaluator),
                "--bundle",
                str(bundle),
                "--cmake-cache",
                str(cache),
                "--install-prefix",
                str(install_prefix),
                "--candidate-commit",
                COMMIT,
                "--product-profile-sha256",
                PROFILE_SHA256,
                "--output",
                str(root / "different_core.json"),
            )
            self.assertNotEqual(rejected_core.returncode, 0)
            self.assertIn("same core", rejected_core.stderr)

            forbidden_root = root / "external_dependency"
            forbidden_root.mkdir()
            forbidden_source = forbidden_root / "checkpoint.c"
            forbidden_source.write_text(
                "int checkpoint_runtime(void) { return 0; }\n",
                encoding="utf-8",
            )
            forbidden_library = forbidden_root / "libcheckpoint_runtime.so"
            run_checked(
                [
                    "cc",
                    "-shared",
                    "-fPIC",
                    "-Wl,-soname,libcheckpoint_runtime.so",
                    "-o",
                    str(forbidden_library),
                    str(forbidden_source),
                ]
            )
            identity = json.dumps(
                {
                    "schema": "n3mapping_product_build_identity_v2",
                    "commit": COMMIT,
                    "product_profile_sha256": PROFILE_SHA256,
                    "build_type": "Release",
                    "research_tools": "OFF",
                    "verified": True,
                },
                sort_keys=True,
            )
            evaluator_source = node.parent / "forbidden_eval.c"
            evaluator_source.write_text(
                "#include <stdio.h>\n"
                "#include <string.h>\n"
                "extern int n3mapping_fake_core(void);\n"
                "extern int checkpoint_runtime(void);\n"
                "int main(int argc, char **argv) {\n"
                "  (void)n3mapping_fake_core();\n"
                "  (void)checkpoint_runtime();\n"
                "  if (argc == 2 && strcmp(argv[1], "
                '"--build-identity-json") == 0) {\n'
                f"    puts({json.dumps(identity)}); return 0;\n"
                "  }\n"
                "  return 0;\n"
                "}\n",
                encoding="utf-8",
            )
            forbidden_evaluator = node.parent / "forbidden_eval"
            run_checked(
                [
                    "cc",
                    "-Wl,--disable-new-dtags",
                    f"-Wl,-rpath,$ORIGIN:{forbidden_root}",
                    "-L",
                    str(node.parent),
                    "-L",
                    str(forbidden_root),
                    "-o",
                    str(forbidden_evaluator),
                    str(evaluator_source),
                    "-ln3mapping_core",
                    "-lcheckpoint_runtime",
                ]
            )
            rejected_dependency = self.run_tool(
                "runtime-create",
                "--node",
                str(node),
                "--evaluator",
                str(forbidden_evaluator),
                "--bundle",
                str(bundle),
                "--cmake-cache",
                str(cache),
                "--install-prefix",
                str(install_prefix),
                "--candidate-commit",
                COMMIT,
                "--product-profile-sha256",
                PROFILE_SHA256,
                "--output",
                str(root / "forbidden.json"),
            )
            self.assertNotEqual(rejected_dependency.returncode, 0)
            self.assertIn("forbidden Product V1", rejected_dependency.stderr)

            symlink = root / "evaluator_link"
            symlink.symlink_to(evaluator)
            rejected_symlink = self.run_tool(
                "runtime-create",
                "--node",
                str(node),
                "--evaluator",
                str(symlink),
                "--bundle",
                str(bundle),
                "--cmake-cache",
                str(cache),
                "--install-prefix",
                str(install_prefix),
                "--candidate-commit",
                COMMIT,
                "--product-profile-sha256",
                PROFILE_SHA256,
                "--output",
                str(root / "symlink.json"),
            )
            self.assertNotEqual(rejected_symlink.returncode, 0)
            self.assertIn("symlink", rejected_symlink.stderr)

    def test_runtime_rejects_research_build_and_forbidden_install_entry(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            node, evaluator = self.make_binaries(root)
            cache, install_prefix = self.make_cmake_cache(
                root, node, research_tools="ON"
            )
            bundle = self.make_bundle(root, node)
            common = [
                "runtime-create",
                "--node",
                str(node),
                "--evaluator",
                str(evaluator),
                "--bundle",
                str(bundle),
                "--cmake-cache",
                str(cache),
                "--install-prefix",
                str(install_prefix),
                "--candidate-commit",
                COMMIT,
                "--product-profile-sha256",
                PROFILE_SHA256,
            ]
            research_on = self.run_tool(
                *common,
                "--output",
                str(root / "research_on.json"),
            )
            self.assertNotEqual(research_on.returncode, 0)
            self.assertIn(
                "N3MAPPING_BUILD_RESEARCH_TOOLS mismatch",
                research_on.stderr,
            )

            cache, install_prefix = self.make_cmake_cache(root, node)
            forbidden = (
                install_prefix / "lib" / "n3mapping" / "n3mapping_kitti360_eval"
            )
            forbidden.write_bytes(b"model")
            forbidden_install = self.run_tool(
                "runtime-create",
                "--node",
                str(node),
                "--evaluator",
                str(evaluator),
                "--bundle",
                str(bundle),
                "--cmake-cache",
                str(cache),
                "--install-prefix",
                str(install_prefix),
                "--candidate-commit",
                COMMIT,
                "--product-profile-sha256",
                PROFILE_SHA256,
                "--output",
                str(root / "forbidden_install.json"),
            )
            self.assertNotEqual(forbidden_install.returncode, 0)
            self.assertIn("research runtime file is forbidden", forbidden_install.stderr)

    def test_caller_supplied_authority_observation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            node, _ = self.make_binaries(root)
            observation, _ = self.make_observation(root, node)
            artifact = root / "authority_preflight.json"

            created = self.run_tool(
                "authority-create",
                "--observation",
                str(observation),
                "--output",
                str(artifact),
            )
            self.assertNotEqual(created.returncode, 0)
            self.assertIn(
                "the following arguments are required: --node, --evidence-dir",
                created.stderr,
            )
            self.assertFalse(artifact.exists())

            with self.assertRaisesRegex(
                preflight.PreflightError,
                "verifier-owned runner",
            ):
                preflight.validate_authority_observation(observation)

    def test_trusted_active_observation_artifact_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            node, _ = self.make_binaries(root)
            observation, _ = self.make_observation(
                root, node, trusted_runner=True
            )
            evidence = preflight.validate_authority_observation(observation)
            artifact = preflight.authority_artifact(evidence)
            artifact_path = root / "authority_preflight.json"
            write_json(artifact_path, artifact)

            verified = self.run_tool(
                "authority-verify", "--artifact", str(artifact_path)
            )
            self.assertEqual(verified.returncode, 0, verified.stderr)
            self.assertEqual(
                json.loads(verified.stdout),
                {
                    "artifact": str(artifact_path),
                    "distro": "humble",
                    "kind": "authority",
                    "status": "PASS",
                },
            )

            Path(evidence["source_files"]["log"]["path"]).write_text(
                '{"tampered":true}\n', encoding="utf-8"
            )
            rejected = self.run_tool(
                "authority-verify", "--artifact", str(artifact_path)
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("hash mismatch", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
