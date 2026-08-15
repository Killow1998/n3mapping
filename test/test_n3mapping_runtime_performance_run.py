#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "n3mapping_runtime_performance_run.py"
SPEC = importlib.util.spec_from_file_location("runtime_performance_run", TOOL_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {TOOL_PATH}")
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


class RuntimePerformanceRunTest(unittest.TestCase):
    def test_counts_only_requested_jsonl_record_type(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            path = Path(raw_dir) / "runtime.jsonl"
            path.write_text(
                '{"record_type":"runtime_frame","frame_index":1}\n'
                '{"record_type":"loop_cycle","cycle_index":1}\n'
                '{"record_type":"runtime_frame","frame_index":2}\n',
                encoding="utf-8",
            )
            self.assertEqual(TOOL.count_record_type(path, "runtime_frame"), 2)
            self.assertEqual(TOOL.count_record_type(path, "loop_cycle"), 1)

    def test_bag_identity_hashes_metadata_but_not_large_payload(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            bag = Path(raw_dir) / "bag"
            bag.mkdir()
            (bag / "metadata.yaml").write_text("rosbag2_bagfile_information: {}\n")
            (bag / "data_0.db3").write_bytes(b"payload")
            identity = TOOL.bag_identity(bag)
            self.assertEqual(identity["payloads"][0]["size_bytes"], 7)
            self.assertNotIn("sha256", identity["payloads"][0])
            self.assertEqual(len(identity["metadata"]["sha256"]), 64)

    def test_log_scan_is_machine_readable(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            path = Path(raw_dir) / "node.log"
            path.write_text(
                "Tracking failed (x1)\nMessage Filter dropping message\n",
                encoding="utf-8",
            )
            result = TOOL.scan_log(path)
            self.assertEqual(result["tracking_failed"], 1)
            self.assertEqual(result["message_filter_drop"], 1)
            json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
