#!/usr/bin/env python3
"""Support exports must preserve original runs and exclude unrelated data."""

import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "support_bundle", ROOT / "tools" / "n3mapping_support_bundle.py")
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


class SupportBundleTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = self.root / "input.yaml"
        self.config.write_text('n3mapping_node:\n  ros__parameters:\n    map_path: /reference/map.pbstream\n')
        self.run = self.root / "run"
        self.output = self.root / "feedback.zip"

    def prepare(self, diagnostics=False):
        identity = {"commit": "a" * 40, "verified": True}
        args = SimpleNamespace(config=self.config, run_dir=self.run,
                               node_executable=self.root / "node", mode="localization",
                               diagnostics=diagnostics)
        with patch.object(TOOL.subprocess, "run", return_value=SimpleNamespace(
                stdout=json.dumps(identity))) as invoke:
            TOOL.prepare(args)
        self.assertEqual(invoke.call_args.args[0][-1], "--build-identity-json")
        return args

    def pack(self, file_mib=8, total_mib=64):
        TOOL.pack(SimpleNamespace(run_dir=self.run, output=self.output,
                                  max_file_mib=file_mib, max_total_mib=total_mib))
        return zipfile.ZipFile(self.output)

    def test_prepare_snapshots_without_changing_original_or_defaults(self):
        args = self.prepare()
        self.assertEqual((self.run / "config.yaml").read_bytes(), self.config.read_bytes())
        params = json.loads((self.run / "overrides.yaml").read_text())["n3mapping_node"]["ros__parameters"]
        self.assertNotIn("reloc_debug_enable", params)
        self.assertNotIn("map_path", params)
        self.assertEqual(params["map_save_path"], str(self.run))
        with patch.object(TOOL.subprocess, "run", return_value=SimpleNamespace(
                stdout='{"commit":"a"}')):
            with self.assertRaises(FileExistsError):
                TOOL.prepare(args)

    def test_diagnostics_is_explicit_and_has_no_algorithm_overrides(self):
        self.prepare(diagnostics=True)
        params = json.loads((self.run / "overrides.yaml").read_text())["n3mapping_node"]["ros__parameters"]
        self.assertTrue(params["reloc_debug_enable"])
        self.assertTrue(params["loop_debug_enable"])
        self.assertEqual(len(params), 6)

    def test_allowlist_excludes_maps_bags_private_files_and_symlinks(self):
        self.prepare()
        (self.run / "n3mapping.log").write_text("startup\nlock\n")
        for name in ("map.pbstream", "cloud.pcd", "data.db3", ".env", "private.txt"):
            (self.run / name).write_text("do not include")
        (self.run / "frontend.log").symlink_to(self.config)
        ros = self.run / "ros_log"
        ros.mkdir()
        (ros / "node.log").write_text("warning")
        (ros / "external").symlink_to(self.root, target_is_directory=True)
        with self.pack() as archive:
            names = archive.namelist()
            self.assertIn("logs/n3mapping.log", names)
            self.assertIn("logs/ros_log/node.log", names)
            self.assertNotIn("logs/frontend.log", names)
            self.assertFalse(any("map.pbstream" in n or "private" in n for n in names))
            report = json.loads(archive.read("bundle_report.json"))
            self.assertTrue(report["incomplete"])
        self.assertTrue((self.run / "map.pbstream").exists())

    def test_limits_are_reported_and_logs_remain_unchanged(self):
        self.prepare()
        payload = b"first\n" + b"x" * (2 * TOOL.MIB) + b"\nlast\n"
        log = self.run / "n3mapping.log"
        log.write_bytes(payload)
        (self.run / "frontend.log").write_bytes(b"f" * (2 * TOOL.MIB))
        with self.pack(file_mib=1, total_mib=1) as archive:
            self.assertNotIn("logs/n3mapping.log", archive.namelist())
            data = archive.read("logs/n3mapping.log.excerpt")
            self.assertTrue(data.startswith(b"first\n"))
            self.assertTrue(data.endswith(b"\nlast\n"))
            self.assertLessEqual(sum(i.file_size for i in archive.infolist()
                                     if i.filename.startswith("logs/")), TOOL.MIB)
            report = json.loads(archive.read("bundle_report.json"))
            self.assertTrue(report["incomplete"])
            self.assertTrue(any(e.get("truncated") for e in report["files"]))
        self.assertEqual(log.read_bytes(), payload)

    def test_existing_output_is_never_overwritten(self):
        self.prepare()
        self.output.write_bytes(b"keep existing")
        with self.assertRaises(FileExistsError):
            self.pack()
        self.assertEqual(self.output.read_bytes(), b"keep existing")

    def test_missing_evidence_is_not_a_success_claim(self):
        self.run.mkdir()
        with self.pack() as archive:
            report = json.loads(archive.read("bundle_report.json"))
            self.assertTrue(report["incomplete"])
            self.assertIn("build_identity.json", report["missing"])

    def test_complete_small_run_and_output_inside_run_is_rejected(self):
        self.prepare()
        (self.run / "n3mapping.log").write_text("normal shutdown\n")
        with self.pack() as archive:
            report = json.loads(archive.read("bundle_report.json"))
            self.assertFalse(report["incomplete"])
            self.assertEqual(report["missing"], [])
        self.output = self.run / "feedback.zip"
        with self.assertRaises(ValueError):
            self.pack()
        self.assertFalse(self.output.exists())

    def test_pack_cli_needs_no_ros_or_network(self):
        self.run.mkdir()
        result = subprocess.run([
            "python3", "-B", str(ROOT / "tools" / "n3mapping_support_bundle.py"),
            "pack", "--run-dir", str(self.run), "--output", str(self.output)],
            capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(zipfile.is_zipfile(self.output))


if __name__ == "__main__":
    unittest.main()
