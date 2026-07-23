#!/usr/bin/env python3
"""Verify a Product Map Bundle before exec'ing the localization node."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from n3mapping_product_bundle import BundleError, verify_manifest
from n3mapping_product_identity import (
    ProductIdentityError,
    read_product_build_identity,
    require_runtime_node_identity,
)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        description=(
            "Fail-closed launcher for n3mapping localization Product V1."
        )
    )
    command.add_argument("--bundle", required=True, type=Path)
    command.add_argument(
        "--node-executable",
        type=Path,
        help="Explicit node executable (test/development override).",
    )
    command.add_argument(
        "--ros-version",
        choices=("1", "2"),
        default=os.environ.get("ROS_VERSION"),
    )
    command.add_argument("--dry-run", action="store_true")
    return command


def node_command(
    bundle: Path,
    node_executable: Path,
    ros_version: str,
    passthrough: list[str],
) -> list[str]:
    map_path = bundle / "map.pbstream"
    atlas_path = bundle / "map.pbstream.localization_atlas.pb"
    if ros_version == "2":
        injected = [
            "--ros-args",
            "-p",
            "product_profile_v1:=true",
            "-p",
            f"map_path:={map_path}",
            "-p",
            f"reloc_atlas_path:={atlas_path}",
        ]
    elif ros_version == "1":
        injected = [
            "_product_profile_v1:=true",
            f"_map_path:={map_path}",
            f"_reloc_atlas_path:={atlas_path}",
        ]
    else:
        raise BundleError(
            "ROS_VERSION must be 1 or 2 for the product runtime"
        )
    return [str(node_executable), *passthrough, *injected]


def reject_parameter_overrides(ros_version: str, passthrough: list[str]) -> None:
    if ros_version == "2":
        forbidden_exact = {"-p", "--param", "--params-file"}
        for token in passthrough:
            if token in forbidden_exact or token.startswith(
                ("--param=", "--params-file=")
            ):
                raise BundleError(
                    "Product V1 runtime forbids ROS 2 parameter overrides"
                )
    elif ros_version == "1":
        for token in passthrough:
            if token.startswith("_") and not token.startswith("__") and ":=" in token:
                raise BundleError(
                    "Product V1 runtime forbids ROS 1 private parameter "
                    "overrides"
                )


def main() -> int:
    args, passthrough = parser().parse_known_args()
    bundle = args.bundle.resolve()
    try:
        manifest = verify_manifest(
            argparse.Namespace(bundle=bundle)
        )
        node_argument = (
            args.node_executable
            if args.node_executable is not None
            else Path(__file__).resolve().with_name("n3mapping_node")
        )
        node_executable = node_argument.resolve()
        if (
            node_argument.is_symlink()
            or not node_executable.is_file()
            or not os.access(node_executable, os.X_OK)
        ):
            raise BundleError(
                f"node executable is not an executable regular file: "
                f"{node_executable}"
            )
        reject_parameter_overrides(args.ros_version, passthrough)
        identity = read_product_build_identity(node_executable)
        require_runtime_node_identity(identity)
        if identity["commit"] != manifest["n3mapping_commit"]:
            raise BundleError(
                "node build commit does not match Product Bundle"
            )
        if (
            identity["product_profile_sha256"]
            != manifest["product_profile_sha256"]
        ):
            raise BundleError(
                "node product profile does not match Product Bundle"
            )
        if (
            identity["executable_sha256"]
            != manifest["runtime_node_sha256"]
        ):
            raise BundleError(
                "node executable does not match Product Bundle identity"
            )
        if (
            identity["linked_product_libraries"]
            != manifest["runtime_node_linked_product_libraries"]
        ):
            raise BundleError(
                "node linked product libraries do not match Product Bundle "
                "identity"
            )
        command = node_command(
            bundle, node_executable, args.ros_version, passthrough
        )
    except (BundleError, ProductIdentityError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    evidence = {
        "bundle": str(bundle),
        "n3mapping_commit": manifest["n3mapping_commit"],
        "product_profile_sha256": manifest["product_profile_sha256"],
        "node_build_identity": identity,
        "node_command": command,
        "status": "verified",
    }
    print(json.dumps(evidence, sort_keys=True), flush=True)
    if args.dry_run:
        return 0
    os.execv(node_executable, command)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
