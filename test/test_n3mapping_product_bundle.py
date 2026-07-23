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
TOOL = ROOT / "tools" / "n3mapping_product_bundle.py"
COMMIT = "1" * 40


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def flat_ros_parameters(path: Path) -> dict[str, str]:
    parameters: dict[str, str] = {}
    inside_parameters = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == "ros__parameters:":
            inside_parameters = True
            continue
        if not inside_parameters or raw_line[:4] != "    ":
            continue
        key, separator, value = stripped.partition(":")
        if separator:
            parameters[key] = value.strip()
    return parameters


class ProductBundleTest(unittest.TestCase):
    def make_bundle(self, root: Path) -> Path:
        bundle = root / "bundle"
        (bundle / "calibration").mkdir(parents=True)
        (bundle / "map.pbstream").write_bytes(b"map")
        (bundle / "map.pbstream.localization_atlas.pb").write_bytes(b"atlas")
        (bundle / "product_v1.yaml").write_bytes(
            (ROOT / "config" / "product_v1.yaml").read_bytes()
        )
        (bundle / "calibration" / "lidar_imu.yaml").write_text(
            "extrinsic: identity\n", encoding="utf-8"
        )
        return bundle

    def make_identity_executable(
        self, root: Path, *, verified: bool = True
    ) -> Path:
        return build_fake_product_elf(
            root,
            "n3mapping_node",
            commit=COMMIT,
            product_profile_sha256=sha256_file(
                ROOT / "config" / "product_v1.yaml"
            ),
            runtime_node=True,
            verified=verified,
        )

    def run_tool(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-B", str(TOOL), *args],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_create_and_verify(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = self.make_bundle(Path(temporary))
            identity = self.make_identity_executable(Path(temporary))
            created = self.run_tool(
                "create",
                "--bundle",
                str(bundle),
                "--runtime-node",
                str(identity),
                "--lidar-model",
                "Mid-360",
                "--lidar-message-type",
                "livox_ros_driver/CustomMsg",
                "--atlas-format-version",
                "n3mapping_localization_atlas_v1",
                "--atlas-generation-command",
                "n3mapping_localization_atlas_compile --map map.pbstream",
            )
            self.assertEqual(created.returncode, 0, created.stderr)
            manifest = json.loads(
                (bundle / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["schema"], "n3mapping_product_map_bundle_v1"
            )
            self.assertTrue(manifest["created_at"].endswith("+08:00"))
            self.assertEqual(
                manifest["atlas_format_version"],
                "n3mapping_localization_atlas_v1",
            )
            self.assertEqual(len(manifest["files"]), 4)

            verified = self.run_tool("verify", "--bundle", str(bundle))
            self.assertEqual(verified.returncode, 0, verified.stderr)
            self.assertEqual(json.loads(verified.stdout)["status"], "verified")

    def test_verify_rejects_modified_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = self.make_bundle(Path(temporary))
            identity = self.make_identity_executable(Path(temporary))
            created = self.run_tool(
                "create",
                "--bundle",
                str(bundle),
                "--runtime-node",
                str(identity),
                "--lidar-model",
                "Mid-360",
                "--lidar-message-type",
                "livox_ros_driver/CustomMsg",
                "--atlas-format-version",
                "n3mapping_localization_atlas_v1",
                "--atlas-generation-command",
                "n3mapping_localization_atlas_compile --map map.pbstream",
            )
            self.assertEqual(created.returncode, 0, created.stderr)
            (bundle / "product_v1.yaml").write_text(
                "mode: mapping\n", encoding="utf-8"
            )
            verified = self.run_tool("verify", "--bundle", str(bundle))
            self.assertNotEqual(verified.returncode, 0)
            self.assertIn("product_v1.yaml", verified.stderr)

    def test_create_requires_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = self.make_bundle(Path(temporary))
            identity = self.make_identity_executable(Path(temporary))
            (bundle / "calibration" / "lidar_imu.yaml").unlink()
            created = self.run_tool(
                "create",
                "--bundle",
                str(bundle),
                "--runtime-node",
                str(identity),
                "--lidar-model",
                "Mid-360",
                "--lidar-message-type",
                "livox_ros_driver/CustomMsg",
                "--atlas-format-version",
                "n3mapping_localization_atlas_v1",
                "--atlas-generation-command",
                "n3mapping_localization_atlas_compile --map map.pbstream",
            )
            self.assertNotEqual(created.returncode, 0)
            self.assertIn("calibration", created.stderr)

    def test_create_rejects_unverified_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = self.make_bundle(root)
            identity = self.make_identity_executable(root, verified=False)
            created = self.run_tool(
                "create",
                "--bundle",
                str(bundle),
                "--runtime-node",
                str(identity),
                "--lidar-model",
                "Mid-360",
                "--lidar-message-type",
                "livox_ros_driver/CustomMsg",
                "--atlas-format-version",
                "n3mapping_localization_atlas_v1",
                "--atlas-generation-command",
                "n3mapping_localization_atlas_compile --map map.pbstream",
            )
            self.assertNotEqual(created.returncode, 0)
            self.assertIn("unverified", created.stderr)

    def test_product_profile_only_changes_product_invariants(self) -> None:
        development = flat_ros_parameters(ROOT / "config" / "n3mapping.yaml")
        product = flat_ros_parameters(ROOT / "config" / "product_v1.yaml")
        self.assertEqual(set(development), set(product))
        differences = {
            key: (development[key], product[key])
            for key in development
            if development[key] != product[key]
        }
        self.assertEqual(
            differences,
            {
                "mode": ('"mapping"', '"localization"'),
                "reloc_atlas_enable": ("false", "true"),
                "save_global_map_on_shutdown": ("true", "false"),
            },
        )


if __name__ == "__main__":
    unittest.main()
