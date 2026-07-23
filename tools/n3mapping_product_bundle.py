#!/usr/bin/env python3
"""Create or verify a fail-closed n3mapping Product Map Bundle V1 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any
from datetime import datetime, timedelta, timezone

from n3mapping_product_identity import (
    ProductIdentityError,
    read_product_build_identity,
    require_runtime_node_identity,
)


SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"
REQUIRED_FILES = {
    "map": "map.pbstream",
    "atlas": "map.pbstream.localization_atlas.pb",
    "config": "product_v1.yaml",
}
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class BundleError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regular_files(bundle: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(bundle.rglob("*")):
        if path.name == MANIFEST_NAME and path.parent == bundle:
            continue
        if path.is_symlink():
            raise BundleError(f"bundle must not contain symlinks: {path}")
        if path.is_file():
            files.append(path)
    return files


def canonical_product_config() -> Path:
    script = Path(__file__).resolve()
    candidates = (
        script.parents[1] / "config" / "product_v1.yaml",
        script.parents[2] / "share" / "n3mapping" / "config" / "product_v1.yaml",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise BundleError("cannot locate installed canonical product_v1.yaml")


def validate_layout(bundle: Path) -> None:
    if not bundle.is_dir():
        raise BundleError(f"bundle directory does not exist: {bundle}")
    for role, relative in REQUIRED_FILES.items():
        path = bundle / relative
        if not path.is_file() or path.is_symlink():
            raise BundleError(f"missing regular {role} file: {relative}")
        if path.stat().st_size == 0:
            raise BundleError(f"{role} file is empty: {relative}")
    calibration = bundle / "calibration"
    if not calibration.is_dir():
        raise BundleError("missing calibration directory")
    calibration_files = [
        path for path in regular_files(calibration) if path.stat().st_size > 0
    ]
    if not calibration_files:
        raise BundleError("calibration directory contains no non-empty files")
    canonical_config = canonical_product_config()
    if sha256_file(bundle / REQUIRED_FILES["config"]) != sha256_file(
        canonical_config
    ):
        raise BundleError(
            "bundle product_v1.yaml does not match the installed canonical "
            "product profile"
        )


def file_entries(bundle: Path) -> list[dict[str, Any]]:
    entries = []
    for path in regular_files(bundle):
        relative = path.relative_to(bundle).as_posix()
        entries.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return entries


def create_manifest(args: argparse.Namespace) -> dict[str, Any]:
    bundle = args.bundle.resolve()
    validate_layout(bundle)
    try:
        identity = read_product_build_identity(args.runtime_node)
        require_runtime_node_identity(identity)
    except ProductIdentityError as error:
        raise BundleError(str(error)) from error
    canonical_profile_sha256 = sha256_file(canonical_product_config())
    if identity["product_profile_sha256"] != canonical_profile_sha256:
        raise BundleError(
            "identity executable product profile does not match the "
            "installed canonical product_v1.yaml"
        )
    for option, value in (
        ("--lidar-model", args.lidar_model),
        ("--lidar-message-type", args.lidar_message_type),
        ("--atlas-format-version", args.atlas_format_version),
        ("--atlas-generation-command", args.atlas_generation_command),
    ):
        if not value.strip():
            raise BundleError(f"{option} must be non-empty")
    output = bundle / MANIFEST_NAME
    if output.exists() and not args.overwrite:
        raise BundleError(f"manifest already exists: {output}")

    manifest = {
        "schema": "n3mapping_product_map_bundle_v1",
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(
            timespec="seconds"
        ),
        "n3mapping_commit": identity["commit"],
        "runtime_node_sha256": identity["executable_sha256"],
        "runtime_node_linked_product_libraries": identity[
            "linked_product_libraries"
        ],
        "lidar_model": args.lidar_model,
        "lidar_message_type": args.lidar_message_type,
        "atlas_format_version": args.atlas_format_version,
        "atlas_generation_command": args.atlas_generation_command,
        "product_profile_sha256": canonical_profile_sha256,
        "required_roles": REQUIRED_FILES,
        "files": file_entries(bundle),
    }
    serialized = json.dumps(
        manifest, indent=2, sort_keys=True, ensure_ascii=False
    ) + "\n"
    temporary = bundle / f".{MANIFEST_NAME}.{os.getpid()}.tmp"
    try:
        temporary.write_text(serialized, encoding="utf-8")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return manifest


def load_manifest(bundle: Path) -> dict[str, Any]:
    path = bundle / MANIFEST_NAME
    if not path.is_file() or path.is_symlink():
        raise BundleError(f"missing regular manifest: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BundleError(f"cannot parse manifest: {error}") from error
    if manifest.get("schema") != "n3mapping_product_map_bundle_v1":
        raise BundleError("manifest schema mismatch")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise BundleError("manifest schema version mismatch")
    if manifest.get("required_roles") != REQUIRED_FILES:
        raise BundleError("manifest required role mapping mismatch")
    commit = manifest.get("n3mapping_commit")
    if not isinstance(commit, str) or not COMMIT_RE.fullmatch(commit):
        raise BundleError("manifest n3mapping_commit is invalid")
    product_profile_sha256 = manifest.get("product_profile_sha256")
    if (
        not isinstance(product_profile_sha256, str)
        or not SHA256_RE.fullmatch(product_profile_sha256)
        or product_profile_sha256
        != sha256_file(canonical_product_config())
    ):
        raise BundleError(
            "manifest product profile does not match the installed canonical "
            "product_v1.yaml"
        )
    runtime_node_sha256 = manifest.get(
        "runtime_node_sha256"
    )
    if (
        not isinstance(runtime_node_sha256, str)
        or not SHA256_RE.fullmatch(runtime_node_sha256)
    ):
        raise BundleError("manifest runtime node SHA-256 is invalid")
    runtime_libraries = manifest.get(
        "runtime_node_linked_product_libraries"
    )
    if (
        not isinstance(runtime_libraries, dict)
        or "libn3mapping_core.so" not in runtime_libraries
    ):
        raise BundleError(
            "manifest runtime node linked product libraries are invalid"
        )
    wrappers = {
        "libn3mapping_humble_wrapper.so",
        "libn3mapping_noetic_wrapper.so",
    }
    if len(wrappers.intersection(runtime_libraries)) != 1:
        raise BundleError(
            "manifest runtime node must contain exactly one supported ROS "
            "wrapper library"
        )
    for name, digest in runtime_libraries.items():
        if (
            not isinstance(name, str)
            or not name.startswith("libn3mapping")
            or not isinstance(digest, str)
            or not SHA256_RE.fullmatch(digest)
        ):
            raise BundleError(
                "manifest runtime node linked product libraries are invalid"
            )
    for field in (
        "lidar_model",
        "lidar_message_type",
        "atlas_format_version",
        "atlas_generation_command",
        "created_at",
    ):
        if not isinstance(manifest.get(field), str) or not manifest[field]:
            raise BundleError(f"manifest {field} is missing")
    return manifest


def verify_manifest(args: argparse.Namespace) -> dict[str, Any]:
    bundle = args.bundle.resolve()
    validate_layout(bundle)
    manifest = load_manifest(bundle)
    expected_entries = manifest.get("files")
    if not isinstance(expected_entries, list):
        raise BundleError("manifest files must be a list")
    current_entries = file_entries(bundle)
    if expected_entries != current_entries:
        expected = {
            entry.get("path"): entry
            for entry in expected_entries
            if isinstance(entry, dict)
        }
        current = {entry["path"]: entry for entry in current_entries}
        changed = sorted(
            path
            for path in set(expected) | set(current)
            if expected.get(path) != current.get(path)
        )
        raise BundleError("bundle content mismatch: " + ", ".join(changed))
    return manifest


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        description="Create or verify an n3mapping Product Map Bundle V1."
    )
    subparsers = command.add_subparsers(dest="action", required=True)

    create = subparsers.add_parser("create", help="write manifest.json")
    create.add_argument("--bundle", type=Path, required=True)
    create.add_argument(
        "--runtime-node",
        type=Path,
        required=True,
        help=(
            "Verified installed Product V1 runtime node whose exact bytes, "
            "linked product libraries, commit, and profile are recorded"
        ),
    )
    create.add_argument("--lidar-model", required=True)
    create.add_argument("--lidar-message-type", required=True)
    create.add_argument("--atlas-format-version", required=True)
    create.add_argument("--atlas-generation-command", required=True)
    create.add_argument("--overwrite", action="store_true")

    verify = subparsers.add_parser("verify", help="verify all bundle hashes")
    verify.add_argument("--bundle", type=Path, required=True)
    return command


def main() -> int:
    args = parser().parse_args()
    try:
        if args.action == "create":
            manifest = create_manifest(args)
        else:
            manifest = verify_manifest(args)
    except (BundleError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "bundle": str(args.bundle.resolve()),
                "file_count": len(manifest["files"]),
                "n3mapping_commit": manifest["n3mapping_commit"],
                "status": "created" if args.action == "create" else "verified",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
