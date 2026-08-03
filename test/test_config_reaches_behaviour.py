#!/usr/bin/env python3
"""Every config field must reach behaviour, and be settable on both builds.

Five defects in this package have had the same shape: a field declared in
config.h, registered for the runtime summary, read from yaml, printed -- and read
nowhere that decides anything. loop_noise_position never reached the loop edges;
vertical_observability_score was a hardcoded 1.0 written into the debug stream as
if measured; loop_closest_id_th and loop_min_id_interval were parsed, validated
and never used; the provenance key list had gone stale; foreground_conflict_ratio
is computed and read by nothing. Each was found by accident, months apart.

They are mechanically detectable, so this detects them. A field that appears only
in the config plumbing does nothing, however complete its wiring looks. A field
missing from either ROS build's loader cannot be set there, which is how a
mechanism ships and is silently unreachable on one of the two targets.

This does not check that a field changes the right behaviour -- only that
something outside the plumbing reads it at all. That is the floor, not the
ceiling.
"""

from __future__ import annotations

from pathlib import Path
import re
import unittest


REPOSITORY = Path(__file__).resolve().parents[1]

HEADER = REPOSITORY / "include/n3mapping/config.h"
SUMMARY = REPOSITORY / "src/config.cpp"
HUMBLE = REPOSITORY / "humble/src/config_humble.cpp"
NOETIC = REPOSITORY / "noetic/src/config_noetic.cpp"

# The plumbing is where a field is declared, registered, and loaded. Reading it
# in any of these does not make it do anything.
PLUMBING = {HEADER, SUMMARY, HUMBLE, NOETIC}

FIELD_PATTERN = re.compile(
    r"^\s{4}(?:bool|int|double|float|std::size_t|size_t|std::string)\s+(\w+)\s*=",
    re.M,
)


def declared_fields() -> list[str]:
    return sorted(set(FIELD_PATTERN.findall(HEADER.read_text(encoding="utf-8"))))


def sources() -> list[Path]:
    """Production translation units and headers, tests excluded.

    A field exercised only from a test is still inert in the product, and one of
    the defects above was exactly that: a test that set
    loop_spatial_candidate_min_id_gap, asserted a behaviour, passed, and named
    itself after a gate the production path had stopped consulting.
    """
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
        self.fields = declared_fields()
        self.assertGreater(len(self.fields), 100,
                           "the field pattern stopped matching; fix the pattern "
                           "rather than the assertion")
        self.corpus = "\n".join(
            path.read_text(encoding="utf-8", errors="ignore") for path in sources()
        )

    def test_every_field_is_read_outside_the_plumbing(self) -> None:
        inert = [
            name for name in self.fields
            if not re.search(r"\b" + re.escape(name) + r"\b", self.corpus)
        ]
        self.assertEqual(
            inert, [],
            "these config fields are declared, registered and loadable, and no "
            "production code reads them. Either wire them up or delete them; "
            "leaving them makes every params file and provenance record that "
            "sets them a lie about what the run was configured with.",
        )

    def test_every_field_is_in_the_runtime_summary(self) -> None:
        text = SUMMARY.read_text(encoding="utf-8")
        registered = set(re.findall(r"N3MAPPING_CONFIG_FIELD\((\w+)\);", text))
        missing = sorted(set(self.fields) - registered)
        self.assertEqual(
            missing, [],
            "these fields are missing from runtimeConfigCanonical, so a run "
            "configured with them records no evidence of it",
        )

    def test_every_field_is_settable_on_both_builds(self) -> None:
        for label, path in (("humble", HUMBLE), ("noetic", NOETIC)):
            loaded = set(re.findall(r'gets?\("(\w+)"', path.read_text(encoding="utf-8")))
            missing = sorted(set(self.fields) - loaded)
            with self.subTest(build=label):
                self.assertEqual(
                    missing, [],
                    f"these fields cannot be set from yaml on the {label} build, "
                    "so a mechanism can ship and be unreachable on one target",
                )


if __name__ == "__main__":
    unittest.main()
