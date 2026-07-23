#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from product_test_elf import build_fake_product_elf


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_TOOL = ROOT / "tools" / "n3mapping_product_bundle.py"
RUNTIME = ROOT / "tools" / "n3mapping_product_runtime.py"
COMMIT = "2" * 40


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


class ProductRuntimeTest(unittest.TestCase):
    def make_node(
        self, root: Path, *, commit: str = COMMIT, verified: bool = True
    ) -> Path:
        return build_fake_product_elf(
            root,
            "n3mapping_node",
            commit=commit,
            product_profile_sha256=sha256_file(
                ROOT / "config" / "product_v1.yaml"
            ),
            runtime_node=True,
            verified=verified,
        )

    def make_bundle(self, root: Path, identity: Path) -> Path:
        bundle = root / "bundle"
        (bundle / "calibration").mkdir(parents=True)
        (bundle / "map.pbstream").write_bytes(b"map")
        (bundle / "map.pbstream.localization_atlas.pb").write_bytes(b"atlas")
        (bundle / "product_v1.yaml").write_bytes(
            (ROOT / "config" / "product_v1.yaml").read_bytes()
        )
        (bundle / "calibration" / "lidar.yaml").write_text(
            "fixed: true\n", encoding="utf-8"
        )
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                str(BUNDLE_TOOL),
                "create",
                "--bundle",
                str(bundle),
                "--runtime-node",
                str(identity),
                "--lidar-model",
                "Mid-360",
                "--lidar-message-type",
                "livox_ros_driver2/msg/CustomMsg",
                "--atlas-format-version",
                "n3mapping_localization_atlas_v1",
                "--atlas-generation-command",
                "n3mapping_localization_atlas_compile --map map.pbstream",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return bundle

    def test_verified_bundle_injects_authoritative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            node = self.make_node(root)
            bundle = self.make_bundle(root, node)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(RUNTIME),
                    "--bundle",
                    str(bundle),
                    "--node-executable",
                    str(node),
                    "--ros-version",
                    "2",
                    "--dry-run",
                    "--ros-args",
                    "-r",
                    "__node:=n3mapping_node",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(completed.stdout)
            command = report["node_command"]
            self.assertIn(f"map_path:={bundle / 'map.pbstream'}", command)
            self.assertIn(
                "product_profile_v1:=true",
                command,
            )
            self.assertNotIn("reloc_atlas_enable:=true", command)
            self.assertEqual(report["status"], "verified")

    def test_tampered_bundle_never_executes_node(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            node = self.make_node(root)
            bundle = self.make_bundle(root, node)
            (bundle / "product_v1.yaml").write_text(
                "tampered: true\n", encoding="utf-8"
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(RUNTIME),
                    "--bundle",
                    str(bundle),
                    "--node-executable",
                    str(node),
                    "--ros-version",
                    "1",
                    "--dry-run",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 1)
            self.assertIn("canonical product profile", completed.stderr)

    def test_ros2_parameter_override_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            node = self.make_node(root)
            bundle = self.make_bundle(root, node)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(RUNTIME),
                    "--bundle",
                    str(bundle),
                    "--node-executable",
                    str(node),
                    "--ros-version",
                    "2",
                    "--dry-run",
                    "--ros-args",
                    "-p",
                    "reloc_min_inlier_ratio:=0.1",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 1)
            self.assertIn("forbids ROS 2 parameter overrides", completed.stderr)

    def test_ros1_private_parameter_override_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            node = self.make_node(root)
            bundle = self.make_bundle(root, node)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(RUNTIME),
                    "--bundle",
                    str(bundle),
                    "--node-executable",
                    str(node),
                    "--ros-version",
                    "1",
                    "--dry-run",
                    "_reloc_min_inlier_ratio:=0.1",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 1)
            self.assertIn("forbids ROS 1 private", completed.stderr)

    def test_node_identity_must_match_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identity = self.make_node(root)
            bundle = self.make_bundle(root, identity)
            mismatched = root / "other"
            mismatched.mkdir()
            node = self.make_node(mismatched, commit="3" * 40)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(RUNTIME),
                    "--bundle",
                    str(bundle),
                    "--node-executable",
                    str(node),
                    "--ros-version",
                    "2",
                    "--dry-run",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 1)
            self.assertIn("commit does not match", completed.stderr)


if __name__ == "__main__":
    unittest.main()
