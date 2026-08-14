#!/usr/bin/env python3
"""Regression tests for the CMake source-tree build guard."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


REPOSITORY = Path(__file__).resolve().parents[1]
GUARD = REPOSITORY / "cmake" / "n3mapping_product_identity.cmake"
REFUSAL = "Refusing to build inside the n3mapping source tree"


class SourceTreeGuardTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        (self.source / "cmake").mkdir(parents=True)
        (self.source / "config").mkdir()
        shutil.copy2(GUARD, self.source / "cmake" / GUARD.name)
        (self.source / "config" / "product_v1.yaml").write_text(
            "profile: source-tree-guard-test\n", encoding="utf-8"
        )
        (self.source / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.10)\n"
            "project(n3mapping_source_tree_guard_test LANGUAGES NONE)\n"
            "set(N3MAPPING_ROOT \"${CMAKE_CURRENT_SOURCE_DIR}\")\n"
            "set(N3MAPPING_BUILD_RESEARCH_TOOLS OFF)\n"
            "include(${N3MAPPING_ROOT}/cmake/n3mapping_product_identity.cmake)\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def configure(self, build: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["cmake", "-S", str(self.source), "-B", str(build)],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_out_of_tree_build_is_allowed(self) -> None:
        result = self.configure(self.root / "build")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_nested_build_is_refused(self) -> None:
        result = self.configure(self.source / "build")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(REFUSAL, result.stdout + result.stderr)

    def test_symlink_inside_source_is_refused(self) -> None:
        outside = self.root / "outside"
        outside.mkdir()
        link = self.source / "build-link"
        link.symlink_to(outside, target_is_directory=True)
        result = self.configure(link)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(REFUSAL, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
