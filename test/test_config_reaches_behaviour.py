#!/usr/bin/env python3
"""Audit the shared parameter surface and detect inert configuration fields.

ROS adapters and canonical provenance consume one field visitor. Runtime
adapter behaviour is covered by the native-node E2E checks.
"""

from __future__ import annotations

from pathlib import Path
import re
import unittest


REPOSITORY = Path(__file__).resolve().parents[1]

HEADER = REPOSITORY / "include/n3mapping/config.h"
SUMMARY = REPOSITORY / "src/config.cpp"
SCHEMA = REPOSITORY / "include/n3mapping/config_fields.h"
HUMBLE = REPOSITORY / "humble/src/config_humble.cpp"
NOETIC = REPOSITORY / "noetic/src/config_noetic.cpp"

# The plumbing is where a field is declared, registered, and loaded. Reading it
# in any of these does not make it do anything.
PLUMBING = {HEADER, SCHEMA, SUMMARY, HUMBLE, NOETIC}

FIELD_PATTERN = re.compile(
    r"^\s*(?:bool|int|double|float|std::size_t|size_t|std::string)\s+(\w+)\s*=",
    re.M,
)


def sources() -> list[Path]:
    """Production translation units and headers, excluding configuration bindings."""
    found: list[Path] = []
    for pattern in ("*.cpp", "*.h", "*.hpp"):
        for path in REPOSITORY.rglob(pattern):
            relative = path.relative_to(REPOSITORY)
            if relative.parts[0] in ("test", "docs", "build", "install"):
                continue
            if path in PLUMBING:
                continue
            found.append(path)
    return found


class ConfigReachesBehaviour(unittest.TestCase):
    def setUp(self) -> None:
        self.fields = sorted(set(FIELD_PATTERN.findall(
            HEADER.read_text(encoding="utf-8"))))
        self.assertTrue(self.fields, "No configuration fields were parsed")

    def test_every_field_is_read_outside_the_plumbing(self) -> None:
        corpus = "\n".join(path.read_text(encoding="utf-8") for path in sources())
        inert = [
            name for name in self.fields
            if not re.search(r"\b" + re.escape(name) + r"\b", corpus)
        ]
        self.assertEqual(
            inert, [],
            "Configuration fields without production consumers: wire them up or remove them",
        )

    def test_shared_surface_matches_declared_fields(self) -> None:
        bindings = re.findall(r'visit\s*\(\s*"(\w+)"\s*,\s*config\.(\w+)\s*\)',
                              SCHEMA.read_text(encoding="utf-8"))
        self.assertEqual(len(bindings), len(set(bindings)))
        self.assertTrue(all(name == member for name, member in bindings))
        self.assertEqual(sorted(name for name, _ in bindings), self.fields)


if __name__ == "__main__":
    unittest.main()
