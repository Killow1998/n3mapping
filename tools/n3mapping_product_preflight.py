#!/usr/bin/env python3
"""Create and verify fail-closed Product V1 preflight evidence.

The runtime artifact binds the exact node, evaluator, product libraries, and
Map Bundles to one candidate identity.  Authority certification is deliberately
disabled until a verifier-owned active-runtime observer exists; caller-supplied
scenario matrices are not runtime evidence.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any


BUILD_IDENTITY_SCHEMA = "n3mapping_product_build_identity_v2"
BUNDLE_SCHEMA = "n3mapping_product_map_bundle_v1"
RUNTIME_SCHEMA = "n3mapping_product_runtime_preflight_v1"
AUTHORITY_OBSERVATION_SCHEMA = "n3mapping_authority_observation_v2"
ACTIVE_RUNTIME_AUTHORITY_SCHEMA = (
    "n3mapping_product_active_runtime_authority_preflight_v2"
)
SCHEMA_VERSION = 1
AUTHORITY_SCHEMA_VERSION = 2

COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_DEPENDENCY_RE = re.compile(
    r"(?:learned|checkpoint|cuda|gpu|research|torch|tensorflow|onnx|"
    r"tensorrt|cudnn|p1[-_ ]?[k-q](?:\b|_))",
    re.IGNORECASE,
)
FORBIDDEN_INSTALL_PATH_RE = re.compile(
    r"(?:^|[/_.-])(?:learned|checkpoint|ckpt|train[a-z0-9]*|research|"
    r"p1[-_]?[k-q])(?:$|[/_.-])",
    re.IGNORECASE,
)
LDD_ADDRESS_RE = re.compile(r"\(0x[0-9a-fA-F]+\)")
PRODUCT_WRAPPERS = {
    "libn3mapping_humble_wrapper.so",
    "libn3mapping_noetic_wrapper.so",
}
KNOWN_RESEARCH_RUNTIME_BASENAMES = {
    "n3mapping_dataset_readiness.py",
    "n3mapping_episode_benchmark.py",
    "n3mapping_episode_diagnose.py",
    "n3mapping_episode_freeze.py",
    "n3mapping_episode_review_prepare.py",
    "n3mapping_eval_compare.py",
    "n3mapping_eval_matrix.py",
    "n3mapping_eval_validate.py",
    "n3mapping_frozen_bev_verifier_audit.py",
    "n3mapping_keyframe_surface_support_audit.py",
    "n3mapping_kitti360_eval",
    "n3mapping_kitti360_reader",
    "n3mapping_loop_candidate_benchmark.py",
    "n3mapping_loop_debug_analyze.py",
    "n3mapping_m2dgr_eval",
    "n3mapping_multiview_free_space_audit.py",
    "n3mapping_runtime_signal_audit.py",
    "n3mapping_surface_overlap_audit.py",
    "n3mapping_synthetic_eval_gate.py",
    "n3mapping_synthetic_relocalization_eval",
    "n3mapping_synthetic_relocalization_publisher",
    "n3mapping_synthetic_relocalization_visualizer",
    "n3mapping_verifier_pair_freeze.py",
    "n3mapping_verifier_split_freeze.py",
    "run_synthetic_relocalization_matrix.py",
    "synthetic_frozen_contract.yaml",
}
BUNDLE_REQUIRED_ROLES = {
    "map": "map.pbstream",
    "atlas": "map.pbstream.localization_atlas.pb",
    "config": "product_v1.yaml",
}


class PreflightError(RuntimeError):
    """A fail-closed preflight contract violation."""


def _reject_duplicate_key(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PreflightError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise PreflightError(f"non-finite JSON number: {value}")


def _require_finite_tree(value: Any, location: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise PreflightError(f"non-finite JSON number at {location}")
    if isinstance(value, dict):
        for key, child in value.items():
            _require_finite_tree(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _require_finite_tree(child, f"{location}[{index}]")


def strict_json_loads(text: str, source: str) -> Any:
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_key,
            parse_constant=_reject_json_constant,
        )
    except PreflightError:
        raise
    except json.JSONDecodeError as error:
        raise PreflightError(f"cannot parse JSON {source}: {error}") from error
    _require_finite_tree(value)
    return value


def _absolute_without_resolving(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def require_no_symlink_components(path: Path, *, include_leaf: bool = True) -> Path:
    absolute = _absolute_without_resolving(path)
    parts = absolute.parts
    cursor = Path(parts[0])
    final_index = len(parts) if include_leaf else max(1, len(parts) - 1)
    for part in parts[1:final_index]:
        cursor /= part
        try:
            mode = cursor.lstat().st_mode
        except FileNotFoundError as error:
            raise PreflightError(f"path component does not exist: {cursor}") from error
        if stat.S_ISLNK(mode):
            raise PreflightError(f"symlink path component is forbidden: {cursor}")
    return absolute


def require_regular_file(path: Path, role: str) -> Path:
    absolute = require_no_symlink_components(path)
    try:
        file_stat = absolute.stat()
    except OSError as error:
        raise PreflightError(f"cannot stat {role}: {absolute}: {error}") from error
    if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size <= 0:
        raise PreflightError(
            f"{role} must be a non-empty regular file: {absolute}"
        )
    return absolute


def require_runtime_dependency(path: Path, role: str) -> Path:
    """Return the canonical file behind a loader-reported dependency.

    Distribution library paths conventionally traverse `/lib` and SONAME
    symlinks.  Those are not caller-controlled evidence paths.  We therefore
    record and hash the canonical target while retaining the untouched `ldd`
    line; every later verification resolves and hashes it again.
    """
    try:
        resolved = path.resolve(strict=True)
        file_stat = resolved.stat()
    except OSError as error:
        raise PreflightError(f"cannot resolve {role}: {path}: {error}") from error
    if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size <= 0:
        raise PreflightError(
            f"{role} must resolve to a non-empty regular file: {path}"
        )
    return resolved


def require_directory(path: Path, role: str) -> Path:
    absolute = require_no_symlink_components(path)
    try:
        mode = absolute.stat().st_mode
    except OSError as error:
        raise PreflightError(f"cannot stat {role}: {absolute}: {error}") from error
    if not stat.S_ISDIR(mode):
        raise PreflightError(f"{role} must be a directory: {absolute}")
    return absolute


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json_file(path: Path, role: str) -> tuple[Path, Any]:
    regular = require_regular_file(path, role)
    try:
        text = regular.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise PreflightError(f"cannot read {role}: {regular}: {error}") from error
    return regular, strict_json_loads(text, str(regular))


def require_exact_keys(
    value: Any, expected: set[str], location: str
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PreflightError(f"{location} must be an object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise PreflightError(
            f"{location} fields mismatch; missing={missing} extra={extra}"
        )
    return value


def require_sha256(value: Any, location: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise PreflightError(f"{location} must be a lowercase SHA-256")
    return value


def require_commit(value: Any, location: str) -> str:
    if not isinstance(value, str) or not COMMIT_RE.fullmatch(value):
        raise PreflightError(f"{location} must be a lowercase 40-hex commit")
    return value


def created_at_utc8() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(
        timespec="seconds"
    )


def validate_utc8_timestamp(value: Any, location: str) -> str:
    if not isinstance(value, str):
        raise PreflightError(f"{location} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise PreflightError(f"{location} is not ISO-8601") from error
    if parsed.utcoffset() != timedelta(hours=8):
        raise PreflightError(f"{location} must use Asia/Shanghai UTC+08:00")
    return value


def _run(command: list[str], role: str) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["LC_ALL"] = "C"
    try:
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
    except OSError as error:
        raise PreflightError(f"cannot run {role}: {error}") from error


def read_build_identity(executable: Path) -> dict[str, Any]:
    completed = _run(
        [str(executable), "--build-identity-json"],
        f"identity probe for {executable}",
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise PreflightError(
            f"build identity probe failed for {executable}: {detail}"
        )
    identity = strict_json_loads(completed.stdout, f"identity from {executable}")
    identity = require_exact_keys(
        identity,
        {
            "schema",
            "commit",
            "product_profile_sha256",
            "build_type",
            "research_tools",
            "verified",
        },
        f"identity from {executable}",
    )
    if identity["schema"] != BUILD_IDENTITY_SCHEMA:
        raise PreflightError(f"build identity schema mismatch: {executable}")
    require_commit(identity["commit"], f"identity commit for {executable}")
    require_sha256(
        identity["product_profile_sha256"],
        f"identity profile for {executable}",
    )
    if identity["build_type"] != "Release":
        raise PreflightError(
            f"build identity is not Release: {executable}"
        )
    if identity["research_tools"] != "OFF":
        raise PreflightError(
            f"build identity includes research tools: {executable}"
        )
    if identity["verified"] is not True:
        raise PreflightError(f"build identity is not verified: {executable}")
    return identity


def inspect_elf_dependencies(executable: Path) -> tuple[list[str], list[dict[str, str]]]:
    with executable.open("rb") as stream:
        if stream.read(4) != b"\x7fELF":
            raise PreflightError(f"Product V1 executable is not ELF: {executable}")
    completed = _run(["ldd", str(executable)], f"ldd for {executable}")
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise PreflightError(f"ldd failed for {executable}: {detail}")
    raw_lines = completed.stdout.splitlines()
    if not raw_lines:
        raise PreflightError(f"ldd returned no dependency observations: {executable}")

    dependencies: list[dict[str, str]] = []
    seen_names: set[str] = set()
    for raw_line in raw_lines:
        stripped = raw_line.strip()
        if not stripped:
            raise PreflightError(f"ldd emitted an empty line: {executable}")
        if "not found" in stripped:
            raise PreflightError(f"unresolved runtime dependency: {stripped}")
        if "=>" in stripped:
            name, remainder = stripped.split("=>", 1)
            name = name.strip()
            path_text = remainder.strip().split(maxsplit=1)[0]
            if not path_text.startswith("/"):
                raise PreflightError(f"unresolved ldd line: {stripped}")
        else:
            first = stripped.split(maxsplit=1)[0]
            if not first.startswith("/"):
                # linux-vdso is a kernel-provided pseudo dependency.  Its raw
                # line remains in the artifact, but there is no file to hash.
                if first.startswith("linux-vdso"):
                    continue
                raise PreflightError(f"unparsed ldd line: {stripped}")
            path_text = first
            name = Path(path_text).name
        if not name or name in seen_names:
            raise PreflightError(f"duplicate or empty ldd dependency: {stripped}")
        seen_names.add(name)
        dependency_path = require_runtime_dependency(
            Path(path_text), f"runtime dependency {name}"
        )
        forbidden_text = f"{name} {dependency_path}"
        match = FORBIDDEN_DEPENDENCY_RE.search(forbidden_text)
        if match:
            raise PreflightError(
                "forbidden Product V1 runtime dependency "
                f"{match.group(0)!r}: {stripped}"
            )
        dependencies.append(
            {
                "name": name,
                "path": str(dependency_path),
                "sha256": sha256_file(dependency_path),
                "ldd_line": raw_line,
            }
        )
    return raw_lines, dependencies


def inspect_binary(path: Path, role: str) -> dict[str, Any]:
    executable = require_regular_file(path, role)
    identity = read_build_identity(executable)
    ldd_lines, dependencies = inspect_elf_dependencies(executable)
    product_libraries = {
        dependency["name"]: dependency["sha256"]
        for dependency in dependencies
        if dependency["name"].startswith("libn3mapping")
    }
    if "libn3mapping_core.so" not in product_libraries:
        raise PreflightError(f"{role} does not resolve libn3mapping_core.so")
    wrappers = PRODUCT_WRAPPERS.intersection(product_libraries)
    if role == "runtime node" and len(wrappers) != 1:
        raise PreflightError(
            "runtime node must resolve exactly one supported ROS wrapper"
        )
    if role == "evaluator" and wrappers:
        raise PreflightError("evaluator must not resolve a ROS wrapper library")
    return {
        "path": str(executable),
        "sha256": sha256_file(executable),
        "identity": identity,
        "ldd_lines": ldd_lines,
        "dependencies": dependencies,
        "product_libraries": dict(sorted(product_libraries.items())),
    }


def preflight_tool_identity() -> dict[str, str]:
    tool = require_regular_file(Path(__file__), "Product V1 preflight tool")
    return {"path": str(tool), "sha256": sha256_file(tool)}


def authority_runner_identity() -> dict[str, str]:
    runner = require_regular_file(
        Path(__file__).with_name("n3mapping_product_authority_runner.py"),
        "Product V1 authority runner",
    )
    return {"path": str(runner), "sha256": sha256_file(runner)}


def is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def inspect_cmake_cache(
    cache_path: Path,
    *,
    expected_commit: str,
    install_prefix: Path,
) -> dict[str, Any]:
    cache = require_regular_file(cache_path, "Product V1 CMakeCache")
    try:
        lines = cache.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise PreflightError(f"cannot read CMakeCache: {cache}: {error}") from error
    entries: dict[str, dict[str, str]] = {}
    entry_re = re.compile(r"^([^:#=]+):([^=]+)=(.*)$")
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line or raw_line.startswith(("#", "//")):
            continue
        match = entry_re.fullmatch(raw_line)
        if not match:
            raise PreflightError(
                f"malformed CMakeCache line {line_number}: {raw_line!r}"
            )
        key, value_type, value = match.groups()
        if key in entries:
            raise PreflightError(f"duplicate CMakeCache key: {key}")
        entries[key] = {"type": value_type, "value": value}

    required = {
        "CMAKE_BUILD_TYPE": {"type": "STRING", "value": "Release"},
        "N3MAPPING_BUILD_RESEARCH_TOOLS": {"type": "BOOL", "value": "OFF"},
        "N3MAPPING_PRODUCT_COMMIT": {
            "type": "STRING",
            "value": expected_commit,
        },
        "CMAKE_INSTALL_PREFIX": {
            "type": "PATH",
            "value": str(install_prefix),
        },
    }
    for key, expected in required.items():
        if entries.get(key) != expected:
            raise PreflightError(
                f"CMakeCache {key} mismatch; expected "
                f"{expected['type']}={expected['value']!r}, "
                f"observed={entries.get(key)!r}"
            )
    return {
        "path": str(cache),
        "sha256": sha256_file(cache),
        "required_entries": required,
    }


def inspect_install_prefix(
    prefix_path: Path,
    *,
    node: dict[str, Any],
    evaluator: dict[str, Any],
) -> dict[str, Any]:
    prefix = require_directory(prefix_path, "Product V1 install prefix")
    for role, binary in (("runtime node", node), ("evaluator", evaluator)):
        binary_path = Path(binary["path"])
        if not is_within(binary_path, prefix):
            raise PreflightError(
                f"{role} is outside the frozen install prefix: {binary_path}"
            )

    files: list[dict[str, Any]] = []
    for path in sorted(prefix.rglob("*")):
        require_no_symlink_components(path)
        relative = path.relative_to(prefix).as_posix()
        forbidden = FORBIDDEN_INSTALL_PATH_RE.search(relative)
        if forbidden:
            raise PreflightError(
                "forbidden Product V1 install entry "
                f"{forbidden.group(0)!r}: {relative}"
            )
        if path.name in KNOWN_RESEARCH_RUNTIME_BASENAMES:
            raise PreflightError(
                f"research runtime file is forbidden in Product V1: {relative}"
            )
        mode = path.stat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise PreflightError(
                f"non-regular Product V1 install entry: {path}"
            )
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not files:
        raise PreflightError("Product V1 install prefix contains no files")
    installed_paths = {entry["path"] for entry in files}
    for role, binary in (("runtime node", node), ("evaluator", evaluator)):
        relative = Path(binary["path"]).relative_to(prefix).as_posix()
        if relative not in installed_paths:
            raise PreflightError(f"{role} is missing from install inventory")
    return {"path": str(prefix), "files": files}


def _bundle_regular_files(bundle: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(bundle.rglob("*")):
        require_no_symlink_components(path)
        mode = path.stat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise PreflightError(f"non-regular Bundle entry is forbidden: {path}")
        if path == bundle / "manifest.json":
            continue
        if path.stat().st_size <= 0:
            raise PreflightError(f"empty Bundle file is forbidden: {path}")
        files.append(path)
    return files


def inspect_bundle(
    bundle_path: Path,
    *,
    expected_commit: str,
    expected_profile: str,
    node: dict[str, Any],
) -> dict[str, Any]:
    bundle = require_directory(bundle_path, "Product Map Bundle")
    manifest_path, manifest = strict_json_file(
        bundle / "manifest.json", "Product Map Bundle manifest"
    )
    manifest = require_exact_keys(
        manifest,
        {
            "schema",
            "schema_version",
            "created_at",
            "n3mapping_commit",
            "runtime_node_sha256",
            "runtime_node_linked_product_libraries",
            "lidar_model",
            "lidar_message_type",
            "atlas_format_version",
            "atlas_generation_command",
            "product_profile_sha256",
            "required_roles",
            "files",
        },
        f"Bundle manifest {manifest_path}",
    )
    if manifest["schema"] != BUNDLE_SCHEMA or manifest["schema_version"] != 1:
        raise PreflightError(f"Bundle schema mismatch: {manifest_path}")
    validate_utc8_timestamp(manifest["created_at"], f"{manifest_path}.created_at")
    if manifest["n3mapping_commit"] != expected_commit:
        raise PreflightError(f"Bundle candidate commit mismatch: {bundle}")
    if manifest["product_profile_sha256"] != expected_profile:
        raise PreflightError(f"Bundle product profile mismatch: {bundle}")
    if manifest["runtime_node_sha256"] != node["sha256"]:
        raise PreflightError(f"Bundle runtime node hash mismatch: {bundle}")
    if manifest["runtime_node_linked_product_libraries"] != node[
        "product_libraries"
    ]:
        raise PreflightError(
            f"Bundle runtime product library hashes mismatch: {bundle}"
        )
    if manifest["required_roles"] != BUNDLE_REQUIRED_ROLES:
        raise PreflightError(f"Bundle required role mapping mismatch: {bundle}")
    for field in (
        "lidar_model",
        "lidar_message_type",
        "atlas_format_version",
        "atlas_generation_command",
    ):
        if not isinstance(manifest[field], str) or not manifest[field].strip():
            raise PreflightError(f"Bundle {field} is empty: {bundle}")

    entries = manifest["files"]
    if not isinstance(entries, list):
        raise PreflightError(f"Bundle files must be a list: {bundle}")
    recorded: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for index, entry in enumerate(entries):
        entry = require_exact_keys(
            entry, {"path", "bytes", "sha256"}, f"{manifest_path}.files[{index}]"
        )
        relative_text = entry["path"]
        if not isinstance(relative_text, str):
            raise PreflightError(f"Bundle file path is not a string: {bundle}")
        relative = Path(relative_text)
        if (
            relative.is_absolute()
            or relative_text != relative.as_posix()
            or ".." in relative.parts
            or relative_text in seen_paths
            or relative_text == "manifest.json"
        ):
            raise PreflightError(f"unsafe or duplicate Bundle file path: {relative_text}")
        seen_paths.add(relative_text)
        if not isinstance(entry["bytes"], int) or isinstance(entry["bytes"], bool):
            raise PreflightError(f"Bundle file byte count is invalid: {relative_text}")
        require_sha256(entry["sha256"], f"Bundle file hash {relative_text}")
        file_path = require_regular_file(bundle / relative, "Bundle file")
        current = {
            "path": relative_text,
            "bytes": file_path.stat().st_size,
            "sha256": sha256_file(file_path),
        }
        if entry != current:
            raise PreflightError(f"Bundle file mismatch: {file_path}")
        recorded.append(current)

    actual_paths = {
        path.relative_to(bundle).as_posix() for path in _bundle_regular_files(bundle)
    }
    if actual_paths != seen_paths:
        raise PreflightError(
            f"Bundle file set mismatch: missing={sorted(seen_paths - actual_paths)} "
            f"extra={sorted(actual_paths - seen_paths)}"
        )
    for role, relative in BUNDLE_REQUIRED_ROLES.items():
        if relative not in seen_paths:
            raise PreflightError(f"Bundle is missing {role}: {relative}")
    if not any(path.startswith("calibration/") for path in seen_paths):
        raise PreflightError(f"Bundle has no calibration evidence: {bundle}")
    config_entry = next(
        entry for entry in recorded if entry["path"] == BUNDLE_REQUIRED_ROLES["config"]
    )
    if config_entry["sha256"] != expected_profile:
        raise PreflightError(f"Bundle config is not the frozen profile: {bundle}")
    return {
        "path": str(bundle),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest": manifest,
        "verified_files": recorded,
    }


def build_runtime_evidence(
    node_path: Path,
    evaluator_path: Path,
    bundle_paths: list[Path],
    cmake_cache_path: Path,
    install_prefix_path: Path,
    expected_commit: str,
    expected_profile: str,
) -> dict[str, Any]:
    expected_commit = require_commit(expected_commit, "candidate commit")
    expected_profile = require_sha256(expected_profile, "product profile")
    if not bundle_paths:
        raise PreflightError("at least one Product Map Bundle is required")
    node = inspect_binary(node_path, "runtime node")
    evaluator = inspect_binary(evaluator_path, "evaluator")
    for role, binary in (("runtime node", node), ("evaluator", evaluator)):
        identity = binary["identity"]
        if identity["commit"] != expected_commit:
            raise PreflightError(f"{role} candidate commit mismatch")
        if identity["product_profile_sha256"] != expected_profile:
            raise PreflightError(f"{role} product profile mismatch")
    if (
        node["product_libraries"]["libn3mapping_core.so"]
        != evaluator["product_libraries"]["libn3mapping_core.so"]
    ):
        raise PreflightError("runtime node and evaluator do not use the same core")
    install_prefix = require_directory(
        install_prefix_path, "Product V1 install prefix"
    )
    cmake_cache = inspect_cmake_cache(
        cmake_cache_path,
        expected_commit=expected_commit,
        install_prefix=install_prefix,
    )
    install_inventory = inspect_install_prefix(
        install_prefix, node=node, evaluator=evaluator
    )
    bundles = [
        inspect_bundle(
            path,
            expected_commit=expected_commit,
            expected_profile=expected_profile,
            node=node,
        )
        for path in bundle_paths
    ]
    bundle_names = [bundle["path"] for bundle in bundles]
    if len(bundle_names) != len(set(bundle_names)):
        raise PreflightError("duplicate Product Map Bundle path")
    return {
        "candidate_commit": expected_commit,
        "product_profile_sha256": expected_profile,
        "preflight_tool": preflight_tool_identity(),
        "cmake_cache": cmake_cache,
        "install_prefix": install_inventory,
        "node": node,
        "evaluator": evaluator,
        "bundles": bundles,
    }


def runtime_artifact(evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": RUNTIME_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_at": created_at_utc8(),
        "status": "PASS",
        **evidence,
    }


def normalize_ldd_addresses(value: Any, key: str = "") -> Any:
    """Remove only ASLR loader addresses before evidence comparison."""
    if isinstance(value, dict):
        return {
            child_key: normalize_ldd_addresses(child, child_key)
            for child_key, child in value.items()
        }
    if isinstance(value, list):
        return [normalize_ldd_addresses(child, key) for child in value]
    if isinstance(value, str) and key in {"ldd_line", "ldd_lines"}:
        return LDD_ADDRESS_RE.sub("(ADDR)", value)
    return value


def validate_runtime_artifact_shape(value: Any) -> dict[str, Any]:
    artifact = require_exact_keys(
        value,
        {
            "schema",
            "schema_version",
            "created_at",
            "status",
            "candidate_commit",
            "product_profile_sha256",
            "preflight_tool",
            "cmake_cache",
            "install_prefix",
            "node",
            "evaluator",
            "bundles",
        },
        "runtime preflight artifact",
    )
    if artifact["schema"] != RUNTIME_SCHEMA or artifact["schema_version"] != 1:
        raise PreflightError("runtime preflight artifact schema mismatch")
    if artifact["status"] != "PASS":
        raise PreflightError("runtime preflight artifact is not PASS")
    validate_utc8_timestamp(artifact["created_at"], "artifact.created_at")
    require_commit(artifact["candidate_commit"], "artifact.candidate_commit")
    require_sha256(
        artifact["product_profile_sha256"],
        "artifact.product_profile_sha256",
    )
    if not isinstance(artifact["bundles"], list) or not artifact["bundles"]:
        raise PreflightError("runtime artifact has no Bundle evidence")
    return artifact


def expected_topics(distro: str) -> dict[str, dict[str, Any]]:
    if distro == "humble":
        def topic(name: str, message_type: str, depth: int, durability: str) -> dict[str, Any]:
            return {
                "name": name,
                "type": message_type,
                "qos": {
                    "reliability": "reliable",
                    "durability": durability,
                    "depth": depth,
                },
            }

        return {
            "relocalization_status": topic(
                "/n3mapping/relocalization_status",
                "n3mapping/msg/RelocalizationStatus",
                1,
                "transient_local",
            ),
            "authoritative_pose": topic(
                "/n3mapping/relocalization_pose",
                "geometry_msgs/msg/PoseStamped",
                1,
                "volatile",
            ),
            "legacy_lock": topic(
                "/n3mapping/relocalization_lock",
                "std_msgs/msg/UInt32",
                10,
                "volatile",
            ),
            "global_odometry": topic(
                "/n3mapping/odometry",
                "nav_msgs/msg/Odometry",
                10,
                "volatile",
            ),
            "global_world_cloud": topic(
                "/n3mapping/cloud_world",
                "sensor_msgs/msg/PointCloud2",
                10,
                "volatile",
            ),
        }
    if distro == "noetic":
        def topic(name: str, message_type: str, queue: int, latch: bool) -> dict[str, Any]:
            return {
                "name": name,
                "type": message_type,
                "qos": {
                    "transport": "TCPROS",
                    "queue_size": queue,
                    "latch": latch,
                },
            }

        return {
            "relocalization_status": topic(
                "/n3mapping/relocalization_status",
                "n3mapping/RelocalizationStatus",
                1,
                True,
            ),
            "authoritative_pose": topic(
                "/n3mapping/relocalization_pose",
                "geometry_msgs/PoseStamped",
                1,
                False,
            ),
            "legacy_lock": topic(
                "/n3mapping/relocalization_lock",
                "std_msgs/UInt32",
                10,
                False,
            ),
            "global_odometry": topic(
                "/n3mapping/odometry",
                "nav_msgs/Odometry",
                10,
                False,
            ),
            "global_world_cloud": topic(
                "/n3mapping/cloud_world",
                "sensor_msgs/PointCloud2",
                10,
                False,
            ),
        }
    raise PreflightError(f"unsupported ROS distro: {distro!r}")


def validate_pose(value: Any, location: str) -> dict[str, Any]:
    pose = require_exact_keys(
        value,
        {"stamp_ns", "frame_id", "position", "orientation_xyzw"},
        location,
    )
    if (
        not isinstance(pose["stamp_ns"], int)
        or isinstance(pose["stamp_ns"], bool)
        or pose["stamp_ns"] <= 0
    ):
        raise PreflightError(f"{location}.stamp_ns must be a positive integer")
    if not isinstance(pose["frame_id"], str) or not pose["frame_id"]:
        raise PreflightError(f"{location}.frame_id must be non-empty")
    for field, length in (("position", 3), ("orientation_xyzw", 4)):
        sequence = pose[field]
        if not isinstance(sequence, list) or len(sequence) != length:
            raise PreflightError(f"{location}.{field} must have length {length}")
        if any(
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(float(number))
            for number in sequence
        ):
            raise PreflightError(f"{location}.{field} must be finite numeric data")
    quaternion = [float(number) for number in pose["orientation_xyzw"]]
    norm_squared = sum(component * component for component in quaternion)
    if abs(norm_squared - 1.0) > 1e-6:
        raise PreflightError(f"{location} quaternion is not normalized")
    return pose


def matrix_is_finite_rigid(value: Any, location: str) -> bool:
    if not isinstance(value, list) or len(value) != 16:
        raise PreflightError(f"{location} must contain 16 row-major values")
    matrix: list[float] = []
    for index, number in enumerate(value):
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(float(number))
        ):
            raise PreflightError(f"{location}[{index}] must be finite numeric data")
        matrix.append(float(number))
    tolerance = 1e-6
    if any(
        abs(matrix[index] - expected) > tolerance
        for index, expected in zip((12, 13, 14, 15), (0.0, 0.0, 0.0, 1.0))
    ):
        return False
    rotation = [
        [matrix[row * 4 + column] for column in range(3)]
        for row in range(3)
    ]
    for left in range(3):
        for right in range(3):
            dot = sum(
                rotation[row][left] * rotation[row][right]
                for row in range(3)
            )
            expected = 1.0 if left == right else 0.0
            if abs(dot - expected) > tolerance:
                return False
    determinant = (
        rotation[0][0]
        * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1]
        * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2]
        * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    return abs(determinant - 1.0) <= tolerance


def validate_status(
    value: Any,
    location: str,
    *,
    state: str,
    pose_source: str,
    lock_epoch: int,
) -> None:
    status = require_exact_keys(
        value, {"state", "pose_source", "lock_epoch"}, location
    )
    if status != {
        "state": state,
        "pose_source": pose_source,
        "lock_epoch": lock_epoch,
    }:
        raise PreflightError(
            f"{location} mismatch: expected state={state} "
            f"pose_source={pose_source} lock_epoch={lock_epoch}"
        )


def validate_event_shape(value: Any, location: str) -> dict[str, Any]:
    event = require_exact_keys(value, {"label", "backend", "observed"}, location)
    if not isinstance(event["label"], str) or not event["label"]:
        raise PreflightError(f"{location}.label must be non-empty")
    require_exact_keys(
        event["backend"],
        {"state", "pose_source", "pose_matrix", "lock_event"},
        f"{location}.backend",
    )
    matrix_is_finite_rigid(
        event["backend"]["pose_matrix"], f"{location}.backend.pose_matrix"
    )
    observed = require_exact_keys(
        event["observed"],
        {
            "statuses",
            "authoritative_poses",
            "legacy_lock_epochs",
            "global_poses",
            "world_cloud_count",
        },
        f"{location}.observed",
    )
    for field in (
        "statuses",
        "authoritative_poses",
        "legacy_lock_epochs",
        "global_poses",
    ):
        if not isinstance(observed[field], list):
            raise PreflightError(f"{location}.observed.{field} must be a list")
    if (
        not isinstance(observed["world_cloud_count"], int)
        or isinstance(observed["world_cloud_count"], bool)
        or observed["world_cloud_count"] < 0
    ):
        raise PreflightError(
            f"{location}.observed.world_cloud_count must be a nonnegative integer"
        )
    for index, pose in enumerate(observed["authoritative_poses"]):
        validate_pose(pose, f"{location}.observed.authoritative_poses[{index}]")
    for index, pose in enumerate(observed["global_poses"]):
        validate_pose(pose, f"{location}.observed.global_poses[{index}]")
    epochs = observed["legacy_lock_epochs"]
    if any(
        not isinstance(epoch, int) or isinstance(epoch, bool) or epoch <= 0
        for epoch in epochs
    ):
        raise PreflightError(
            f"{location}.observed.legacy_lock_epochs must be positive integers"
        )
    return event


def _expect_backend(
    event: dict[str, Any],
    location: str,
    *,
    state: str,
    pose_source: str,
    pose_valid: bool,
    lock_event: bool,
) -> None:
    expected_scalars = {
        "state": state,
        "pose_source": pose_source,
        "lock_event": lock_event,
    }
    actual_scalars = {
        key: event["backend"][key] for key in expected_scalars
    }
    if actual_scalars != expected_scalars:
        raise PreflightError(
            f"{location}.backend mismatch; expected {expected_scalars}"
        )
    actual_pose_valid = matrix_is_finite_rigid(
        event["backend"]["pose_matrix"], f"{location}.backend.pose_matrix"
    )
    if actual_pose_valid is not pose_valid:
        raise PreflightError(
            f"{location}.backend pose validity mismatch; expected {pose_valid}"
        )


def _expect_localization_event(
    event: dict[str, Any],
    location: str,
    *,
    label: str,
    backend_state: str,
    backend_source: str,
    backend_valid: bool,
    backend_lock: bool,
    status_state: str,
    status_source: str,
    epoch: int,
    authority_count: int,
    legacy_epochs: list[int],
    global_count: int,
    world_cloud_count: int,
) -> None:
    if event["label"] != label:
        raise PreflightError(f"{location}.label must be {label!r}")
    _expect_backend(
        event,
        location,
        state=backend_state,
        pose_source=backend_source,
        pose_valid=backend_valid,
        lock_event=backend_lock,
    )
    observed = event["observed"]
    if len(observed["statuses"]) != 1:
        raise PreflightError(f"{location} must emit exactly one status")
    validate_status(
        observed["statuses"][0],
        f"{location}.observed.statuses[0]",
        state=status_state,
        pose_source=status_source,
        lock_epoch=epoch,
    )
    if len(observed["authoritative_poses"]) != authority_count:
        raise PreflightError(
            f"{location} authoritative initialization pose count mismatch"
        )
    if observed["legacy_lock_epochs"] != legacy_epochs:
        raise PreflightError(f"{location} legacy lock epoch observation mismatch")
    if len(observed["global_poses"]) != global_count:
        raise PreflightError(f"{location} global pose count mismatch")
    if observed["world_cloud_count"] != world_cloud_count:
        raise PreflightError(f"{location} world cloud count mismatch")
    if authority_count == 1:
        if observed["authoritative_poses"][0] != observed["global_poses"][0]:
            raise PreflightError(
                f"{location} initialization and first global pose differ"
            )


def validate_localization_sequence(events: Any, location: str) -> None:
    if not isinstance(events, list) or len(events) != 7:
        raise PreflightError(f"{location} must contain seven ordered events")
    parsed = [
        validate_event_shape(event, f"{location}[{index}]")
        for index, event in enumerate(events)
    ]
    specifications = (
        dict(
            label="searching",
            backend_state="SEARCHING",
            backend_source="NONE",
            backend_valid=True,
            backend_lock=False,
            status_state="SEARCHING",
            status_source="NONE",
            epoch=0,
            authority_count=0,
            legacy_epochs=[],
            global_count=0,
            world_cloud_count=0,
        ),
        dict(
            label="provisional",
            backend_state="PROVISIONAL",
            backend_source="NONE",
            backend_valid=True,
            backend_lock=False,
            status_state="PROVISIONAL",
            status_source="NONE",
            epoch=0,
            authority_count=0,
            legacy_epochs=[],
            global_count=0,
            world_cloud_count=0,
        ),
        dict(
            label="first_full_lock",
            backend_state="FULL_6DOF_LOCKED",
            backend_source="GEOMETRICALLY_CORRECTED",
            backend_valid=True,
            backend_lock=True,
            status_state="FULL_6DOF_LOCKED",
            status_source="GEOMETRICALLY_CORRECTED",
            epoch=1,
            authority_count=1,
            legacy_epochs=[1],
            global_count=1,
            world_cloud_count=1,
        ),
        dict(
            label="steady_full_lock",
            backend_state="FULL_6DOF_LOCKED",
            backend_source="GEOMETRICALLY_CORRECTED",
            backend_valid=True,
            backend_lock=False,
            status_state="FULL_6DOF_LOCKED",
            status_source="GEOMETRICALLY_CORRECTED",
            epoch=1,
            authority_count=0,
            legacy_epochs=[],
            global_count=1,
            world_cloud_count=1,
        ),
        dict(
            label="recently_lost",
            backend_state="RECENTLY_LOST",
            backend_source="ODOM_PREDICTED",
            backend_valid=True,
            backend_lock=False,
            status_state="RECENTLY_LOST",
            status_source="ODOM_PREDICTED",
            epoch=1,
            authority_count=0,
            legacy_epochs=[],
            global_count=1,
            world_cloud_count=1,
        ),
        dict(
            label="lost",
            backend_state="LOST",
            backend_source="NONE",
            backend_valid=True,
            backend_lock=False,
            status_state="LOST",
            status_source="NONE",
            epoch=1,
            authority_count=0,
            legacy_epochs=[],
            global_count=0,
            world_cloud_count=0,
        ),
        dict(
            label="recovered_full_lock",
            backend_state="FULL_6DOF_LOCKED",
            backend_source="GEOMETRICALLY_CORRECTED",
            backend_valid=True,
            backend_lock=True,
            status_state="FULL_6DOF_LOCKED",
            status_source="GEOMETRICALLY_CORRECTED",
            epoch=2,
            authority_count=1,
            legacy_epochs=[2],
            global_count=1,
            world_cloud_count=1,
        ),
    )
    for index, specification in enumerate(specifications):
        _expect_localization_event(
            parsed[index], f"{location}[{index}]", **specification
        )


def validate_invalid_pose_sequence(events: Any, location: str) -> None:
    if not isinstance(events, list) or len(events) != 1:
        raise PreflightError(f"{location} must contain one event")
    event = validate_event_shape(events[0], f"{location}[0]")
    _expect_localization_event(
        event,
        f"{location}[0]",
        label="invalid_full_pose",
        backend_state="FULL_6DOF_LOCKED",
        backend_source="GEOMETRICALLY_CORRECTED",
        backend_valid=False,
        backend_lock=True,
        status_state="SEARCHING",
        status_source="NONE",
        epoch=0,
        authority_count=0,
        legacy_epochs=[],
        global_count=0,
        world_cloud_count=0,
    )


def validate_map_extension_sequence(events: Any, location: str) -> None:
    if not isinstance(events, list) or len(events) != 1:
        raise PreflightError(f"{location} must contain one event")
    event = validate_event_shape(events[0], f"{location}[0]")
    if event["label"] != "map_extension_lock":
        raise PreflightError(f"{location}[0].label must be 'map_extension_lock'")
    _expect_backend(
        event,
        f"{location}[0]",
        state="FULL_6DOF_LOCKED",
        pose_source="GEOMETRICALLY_CORRECTED",
        pose_valid=True,
        lock_event=True,
    )
    observed = event["observed"]
    if observed["statuses"]:
        raise PreflightError("MAP_EXTENSION must not emit localization status")
    if observed["authoritative_poses"]:
        raise PreflightError(
            "MAP_EXTENSION must not emit a localization initialization pose"
        )
    if observed["legacy_lock_epochs"] != [1]:
        raise PreflightError("MAP_EXTENSION must emit exactly legacy epoch 1")
    if len(observed["global_poses"]) != 1:
        raise PreflightError("MAP_EXTENSION must emit one normal global pose")
    if observed["world_cloud_count"] != 1:
        raise PreflightError("MAP_EXTENSION must emit one normal world cloud")


def resolve_declared_file(
    descriptor: Any, location: str, base: Path
) -> dict[str, str]:
    descriptor = require_exact_keys(descriptor, {"path", "sha256"}, location)
    if not isinstance(descriptor["path"], str) or not descriptor["path"]:
        raise PreflightError(f"{location}.path must be non-empty")
    declared_hash = require_sha256(descriptor["sha256"], f"{location}.sha256")
    path = Path(descriptor["path"])
    if not path.is_absolute():
        path = base / path
    regular = require_regular_file(path, location)
    current_hash = sha256_file(regular)
    if current_hash != declared_hash:
        raise PreflightError(f"{location} hash mismatch: {regular}")
    return {"path": str(regular), "sha256": current_hash}


def validate_authority_observation(
    observation_path: Path,
) -> dict[str, Any]:
    evidence = _validate_future_runner_observation(observation_path)
    expected_runner = authority_runner_identity()
    if evidence["source_files"]["harness"] != expected_runner:
        raise PreflightError(
            "authority observation was not produced by the verifier-owned runner"
        )
    return evidence


def _validate_future_runner_observation(
    observation_path: Path,
) -> dict[str, Any]:
    """Parse and structurally validate active-runtime runner output."""
    observation_file, observation = strict_json_file(
        observation_path, "authority raw observation"
    )
    observation = require_exact_keys(
        observation,
        {
            "schema",
            "schema_version",
            "distro",
            "candidate_commit",
            "product_profile_sha256",
            "files",
            "topics",
            "scenarios",
        },
        "authority raw observation",
    )
    if (
        observation["schema"] != AUTHORITY_OBSERVATION_SCHEMA
        or observation["schema_version"] != AUTHORITY_SCHEMA_VERSION
    ):
        raise PreflightError("authority raw observation schema mismatch")
    distro = observation["distro"]
    topics = expected_topics(distro)
    if observation["topics"] != topics:
        raise PreflightError(f"{distro} topic/QoS observation mismatch")
    candidate_commit = require_commit(
        observation["candidate_commit"], "observation.candidate_commit"
    )
    profile = require_sha256(
        observation["product_profile_sha256"],
        "observation.product_profile_sha256",
    )
    files = require_exact_keys(
        observation["files"],
        {"runtime_node", "harness", "log"},
        "observation.files",
    )
    resolved_files = {
        role: resolve_declared_file(
            files[role], f"observation.files.{role}", observation_file.parent
        )
        for role in ("runtime_node", "harness", "log")
    }
    node = inspect_binary(Path(resolved_files["runtime_node"]["path"]), "runtime node")
    expected_wrapper = (
        "libn3mapping_humble_wrapper.so"
        if distro == "humble"
        else "libn3mapping_noetic_wrapper.so"
    )
    resolved_wrappers = PRODUCT_WRAPPERS.intersection(
        node["product_libraries"]
    )
    if resolved_wrappers != {expected_wrapper}:
        raise PreflightError(
            f"{distro} observation uses the wrong ROS wrapper: "
            f"{sorted(resolved_wrappers)}"
        )
    if node["identity"]["commit"] != candidate_commit:
        raise PreflightError("authority node candidate commit mismatch")
    if node["identity"]["product_profile_sha256"] != profile:
        raise PreflightError("authority node product profile mismatch")

    scenarios = observation["scenarios"]
    if not isinstance(scenarios, list) or len(scenarios) != 3:
        raise PreflightError("authority observation must contain three scenarios")
    by_id: dict[str, dict[str, Any]] = {}
    for index, scenario in enumerate(scenarios):
        scenario = require_exact_keys(
            scenario,
            {"id", "run_mode", "events"},
            f"observation.scenarios[{index}]",
        )
        scenario_id = scenario["id"]
        if not isinstance(scenario_id, str) or scenario_id in by_id:
            raise PreflightError("authority scenario IDs must be unique strings")
        by_id[scenario_id] = scenario
    expected_scenarios = {
        "localization_authority_sequence",
        "invalid_pose_suppression",
        "map_extension_legacy_compatibility",
    }
    if set(by_id) != expected_scenarios:
        raise PreflightError("authority scenario set mismatch")
    localization = by_id["localization_authority_sequence"]
    invalid = by_id["invalid_pose_suppression"]
    map_extension = by_id["map_extension_legacy_compatibility"]
    if localization["run_mode"] != "LOCALIZATION":
        raise PreflightError("localization sequence run mode mismatch")
    if invalid["run_mode"] != "LOCALIZATION":
        raise PreflightError("invalid-pose sequence run mode mismatch")
    if map_extension["run_mode"] != "MAP_EXTENSION":
        raise PreflightError("map-extension sequence run mode mismatch")
    validate_localization_sequence(
        localization["events"],
        "localization_authority_sequence.events",
    )
    validate_invalid_pose_sequence(
        invalid["events"], "invalid_pose_suppression.events"
    )
    validate_map_extension_sequence(
        map_extension["events"],
        "map_extension_legacy_compatibility.events",
    )
    return {
        "observation_path": str(observation_file),
        "observation_sha256": sha256_file(observation_file),
        "distro": distro,
        "candidate_commit": candidate_commit,
        "product_profile_sha256": profile,
        "preflight_tool": preflight_tool_identity(),
        "source_files": resolved_files,
        "runtime_node_identity": node,
        "checks": [
            "topic_qos_contract",
            "searching_and_provisional_fail_closed",
            "first_full_single_authority_edge",
            "steady_full_no_duplicate_edge",
            "recently_lost_odom_predicted_no_init_edge",
            "lost_suppresses_global_output",
            "recovery_full_new_authority_edge",
            "invalid_pose_no_authority",
            "map_extension_legacy_only",
        ],
    }


def authority_artifact(evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": ACTIVE_RUNTIME_AUTHORITY_SCHEMA,
        "schema_version": AUTHORITY_SCHEMA_VERSION,
        "created_at": created_at_utc8(),
        "status": "PASS",
        "evidence_kind": "active_runtime",
        "distro": evidence["distro"],
        "candidate_commit": evidence["candidate_commit"],
        "product_profile_sha256": evidence["product_profile_sha256"],
        "preflight_tool": evidence["preflight_tool"],
        "observation": {
            "path": evidence["observation_path"],
            "sha256": evidence["observation_sha256"],
        },
        "source_files": evidence["source_files"],
        "runtime_node_identity": evidence["runtime_node_identity"],
        "checks": evidence["checks"],
    }


def validate_authority_artifact_shape(value: Any) -> dict[str, Any]:
    artifact = require_exact_keys(
        value,
        {
            "schema",
            "schema_version",
            "created_at",
            "status",
            "evidence_kind",
            "distro",
            "candidate_commit",
            "product_profile_sha256",
            "preflight_tool",
            "observation",
            "source_files",
            "runtime_node_identity",
            "checks",
        },
        "authority preflight artifact",
    )
    if (
        artifact["schema"] != ACTIVE_RUNTIME_AUTHORITY_SCHEMA
        or artifact["schema_version"] != AUTHORITY_SCHEMA_VERSION
    ):
        raise PreflightError("authority preflight artifact schema mismatch")
    if (
        artifact["status"] != "PASS"
        or artifact["evidence_kind"] != "active_runtime"
    ):
        raise PreflightError("authority preflight artifact is not active PASS")
    validate_utc8_timestamp(artifact["created_at"], "artifact.created_at")
    require_commit(artifact["candidate_commit"], "artifact.candidate_commit")
    require_sha256(
        artifact["product_profile_sha256"],
        "artifact.product_profile_sha256",
    )
    require_exact_keys(
        artifact["observation"],
        {"path", "sha256"},
        "artifact.observation",
    )
    if artifact["distro"] not in {"humble", "noetic"}:
        raise PreflightError("authority preflight distro is unsupported")
    if not isinstance(artifact["checks"], list) or not artifact["checks"]:
        raise PreflightError("authority preflight checks are missing")
    return artifact


def run_authority_observer(
    node_path: Path, evidence_dir_path: Path
) -> dict[str, Any]:
    node = require_regular_file(node_path, "authority runtime node")
    runner = authority_runner_identity()
    evidence_dir = _absolute_without_resolving(evidence_dir_path)
    require_no_symlink_components(evidence_dir, include_leaf=False)
    if evidence_dir.exists() or evidence_dir.is_symlink():
        raise PreflightError(
            f"authority evidence directory already exists: {evidence_dir}"
        )
    try:
        evidence_dir.mkdir(mode=0o700)
    except OSError as error:
        raise PreflightError(
            f"cannot create authority evidence directory: {error}"
        ) from error

    observation = evidence_dir / "authority_observation.json"
    raw_log = evidence_dir / "authority_raw.jsonl"
    command = [
        sys.executable,
        "-B",
        runner["path"],
        "--node",
        str(node),
        "--observation",
        str(observation),
        "--log",
        str(raw_log),
    ]
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise PreflightError(
            f"cannot run active-runtime authority observer: {error}"
        ) from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise PreflightError(
            f"active-runtime authority observer failed: {detail}"
        )
    evidence = validate_authority_observation(observation)
    if Path(evidence["source_files"]["runtime_node"]["path"]) != node:
        raise PreflightError(
            "active-runtime authority observer used a different runtime node"
        )
    return evidence


def write_artifact(path: Path, value: dict[str, Any], overwrite: bool) -> Path:
    absolute = _absolute_without_resolving(path)
    parent = require_directory(absolute.parent, "artifact output directory")
    output = parent / absolute.name
    if output.exists() or output.is_symlink():
        if output.is_symlink():
            raise PreflightError(f"artifact output must not be a symlink: {output}")
        if not overwrite:
            raise PreflightError(f"artifact already exists: {output}")
        require_regular_file(output, "existing artifact output")
    serialized = json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ) + "\n"
    temporary = parent / f".{output.name}.{os.getpid()}.tmp"
    if temporary.exists() or temporary.is_symlink():
        raise PreflightError(f"temporary artifact path already exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def verify_runtime(path: Path) -> dict[str, Any]:
    artifact_path, loaded = strict_json_file(path, "runtime preflight artifact")
    artifact = validate_runtime_artifact_shape(loaded)
    evidence = build_runtime_evidence(
        Path(artifact["node"]["path"]),
        Path(artifact["evaluator"]["path"]),
        [Path(bundle["path"]) for bundle in artifact["bundles"]],
        Path(artifact["cmake_cache"]["path"]),
        Path(artifact["install_prefix"]["path"]),
        artifact["candidate_commit"],
        artifact["product_profile_sha256"],
    )
    expected = runtime_artifact(evidence)
    expected["created_at"] = artifact["created_at"]
    if normalize_ldd_addresses(expected) != normalize_ldd_addresses(artifact):
        raise PreflightError(
            f"runtime preflight evidence changed since creation: {artifact_path}"
        )
    return artifact


def verify_authority(path: Path) -> dict[str, Any]:
    artifact_path, loaded = strict_json_file(
        path, "authority preflight artifact"
    )
    artifact = validate_authority_artifact_shape(loaded)
    observation = resolve_declared_file(
        artifact["observation"],
        "artifact.observation",
        artifact_path.parent,
    )
    evidence = validate_authority_observation(Path(observation["path"]))
    expected = authority_artifact(evidence)
    expected["created_at"] = artifact["created_at"]
    if normalize_ldd_addresses(expected) != normalize_ldd_addresses(artifact):
        raise PreflightError(
            f"authority preflight evidence changed since creation: "
            f"{artifact_path}"
        )
    return artifact


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create or verify n3mapping Product V1 preflight evidence."
    )
    commands = parser.add_subparsers(dest="action", required=True)

    runtime_create = commands.add_parser(
        "runtime-create", help="audit candidate ELFs and Product Map Bundles"
    )
    runtime_create.add_argument("--node", type=Path, required=True)
    runtime_create.add_argument("--evaluator", type=Path, required=True)
    runtime_create.add_argument(
        "--bundle", type=Path, action="append", required=True
    )
    runtime_create.add_argument("--cmake-cache", type=Path, required=True)
    runtime_create.add_argument("--install-prefix", type=Path, required=True)
    runtime_create.add_argument("--candidate-commit", required=True)
    runtime_create.add_argument("--product-profile-sha256", required=True)
    runtime_create.add_argument("--output", type=Path, required=True)
    runtime_create.add_argument("--overwrite", action="store_true")

    runtime_verify = commands.add_parser(
        "runtime-verify", help="recompute a runtime preflight artifact"
    )
    runtime_verify.add_argument("--artifact", type=Path, required=True)

    authority_create = commands.add_parser(
        "authority-create",
        help="run the verifier-owned observer against an exact Humble node",
    )
    authority_create.add_argument("--node", type=Path, required=True)
    authority_create.add_argument(
        "--evidence-dir", type=Path, required=True
    )
    authority_create.add_argument("--output", type=Path, required=True)
    authority_create.add_argument("--overwrite", action="store_true")

    authority_verify = commands.add_parser(
        "authority-verify",
        help="recompute an active-runtime authority preflight artifact",
    )
    authority_verify.add_argument("--artifact", type=Path, required=True)
    return parser


def main() -> int:
    args = make_parser().parse_args()
    try:
        if args.action == "runtime-create":
            evidence = build_runtime_evidence(
                args.node,
                args.evaluator,
                args.bundle,
                args.cmake_cache,
                args.install_prefix,
                args.candidate_commit,
                args.product_profile_sha256,
            )
            artifact = runtime_artifact(evidence)
            output = write_artifact(args.output, artifact, args.overwrite)
            result = {"artifact": str(output), "kind": "runtime", "status": "PASS"}
        elif args.action == "runtime-verify":
            artifact = verify_runtime(args.artifact)
            result = {
                "artifact": str(_absolute_without_resolving(args.artifact)),
                "bundle_count": len(artifact["bundles"]),
                "kind": "runtime",
                "status": "PASS",
            }
        elif args.action == "authority-create":
            evidence = run_authority_observer(
                args.node, args.evidence_dir
            )
            artifact = authority_artifact(evidence)
            output = write_artifact(args.output, artifact, args.overwrite)
            result = {
                "artifact": str(output),
                "distro": artifact["distro"],
                "kind": "authority",
                "status": "PASS",
            }
        else:
            artifact = verify_authority(args.artifact)
            result = {
                "artifact": str(_absolute_without_resolving(args.artifact)),
                "distro": artifact["distro"],
                "kind": "authority",
                "status": "PASS",
            }
    except (PreflightError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
