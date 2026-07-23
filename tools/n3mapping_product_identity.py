#!/usr/bin/env python3
"""Read the fail-closed compile-time identity of a Product V1 executable."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
IDENTITY_SCHEMA = "n3mapping_product_build_identity_v2"


class ProductIdentityError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_for_executable(path: Path) -> list[str]:
    if path.suffix == ".py":
        return [sys.executable, "-B", str(path)]
    return [str(path)]


def linked_product_library_hashes(executable: Path) -> dict[str, str]:
    with executable.open("rb") as stream:
        if stream.read(4) != b"\x7fELF":
            raise ProductIdentityError(
                "Product V1 identity executable must be an ELF binary"
            )
    try:
        completed = subprocess.run(
            ["ldd", str(executable)],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise ProductIdentityError(
            f"cannot inspect Product V1 linked libraries: {error}"
        ) from error
    if completed.returncode != 0:
        raise ProductIdentityError(
            "cannot inspect Product V1 linked libraries: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    libraries: dict[str, str] = {}
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if not line.startswith("libn3mapping"):
            continue
        name, separator, remainder = line.partition("=>")
        name = name.strip()
        if not separator or remainder.strip().startswith("not found"):
            raise ProductIdentityError(
                f"Product V1 linked library is unresolved: {line}"
            )
        path_text = remainder.strip().split(maxsplit=1)[0]
        path = Path(path_text)
        if not path.is_file():
            raise ProductIdentityError(
                f"Product V1 linked library is missing: {path}"
            )
        if name in libraries:
            raise ProductIdentityError(
                f"duplicate Product V1 linked library: {name}"
            )
        libraries[name] = sha256_file(path.resolve())
    if "libn3mapping_core.so" not in libraries:
        raise ProductIdentityError(
            "Product V1 ELF does not resolve libn3mapping_core.so"
        )
    return dict(sorted(libraries.items()))


def require_runtime_node_identity(identity: dict[str, Any]) -> None:
    libraries = identity.get("linked_product_libraries")
    if not isinstance(libraries, dict):
        raise ProductIdentityError(
            "Product V1 runtime node library identity is missing"
        )
    wrappers = {
        "libn3mapping_humble_wrapper.so",
        "libn3mapping_noetic_wrapper.so",
    }
    resolved_wrappers = wrappers.intersection(libraries)
    if len(resolved_wrappers) != 1:
        raise ProductIdentityError(
            "Product V1 runtime node must resolve exactly one supported ROS "
            "wrapper library"
        )


def read_product_build_identity(executable: Path) -> dict[str, Any]:
    if executable.is_symlink():
        raise ProductIdentityError(
            f"identity executable must not be a symlink: {executable}"
        )
    resolved = executable.resolve()
    if (
        not resolved.is_file()
        or not resolved.stat().st_size
    ):
        raise ProductIdentityError(
            f"identity executable is not a non-empty regular file: {resolved}"
        )
    command = [
        *command_for_executable(resolved),
        "--build-identity-json",
    ]
    try:
        completed = subprocess.run(
            command, check=False, capture_output=True, text=True
        )
    except OSError as error:
        raise ProductIdentityError(
            f"cannot query Product V1 build identity: {error}"
        ) from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ProductIdentityError(
            "Product V1 build identity probe failed: " + detail
        )
    try:
        identity = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ProductIdentityError(
            "Product V1 build identity is not one JSON object"
        ) from error
    if not isinstance(identity, dict) or set(identity) != {
        "schema",
        "commit",
        "product_profile_sha256",
        "build_type",
        "research_tools",
        "verified",
    }:
        raise ProductIdentityError(
            "Product V1 build identity has unexpected fields"
        )
    if identity["schema"] != IDENTITY_SCHEMA:
        raise ProductIdentityError("Product V1 build identity schema mismatch")
    if identity["build_type"] != "Release":
        raise ProductIdentityError(
            "Product V1 build identity is not a Release build"
        )
    if identity["research_tools"] != "OFF":
        raise ProductIdentityError(
            "Product V1 build identity includes research tools"
        )
    if identity["verified"] is not True:
        raise ProductIdentityError("Product V1 build identity is unverified")
    if not isinstance(identity["commit"], str) or not COMMIT_RE.fullmatch(
        identity["commit"]
    ):
        raise ProductIdentityError("Product V1 build commit is invalid")
    profile = identity["product_profile_sha256"]
    if not isinstance(profile, str) or not SHA256_RE.fullmatch(profile):
        raise ProductIdentityError(
            "Product V1 profile SHA-256 is invalid"
        )
    identity["executable"] = str(resolved)
    identity["executable_sha256"] = sha256_file(resolved)
    identity["linked_product_libraries"] = linked_product_library_hashes(
        resolved
    )
    return identity
