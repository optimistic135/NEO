import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.engine.benchmark_report import summarize


class ReportTest(unittest.TestCase):
    def test_empty_run_does_not_invent_scores(self):
        with TemporaryDirectory() as path:
            root = Path(path)
            (root / "configuration.json").write_text(json.dumps({
                "protocol": {"corruptions": ["gaussian_noise"], "seeds": [1234, 2020, 9999],
                             "severity": 5, "batch_size": 64},
                "checkpoint": {"path": "/checkpoint", "sha256": "mock", "preprocessing": {}},
                "methods": {"no_adapt": {"description": "source"}},
                "device": "cpu", "cuda": None, "git": "mock", "dependencies": {},
            }))
            summarize(root)
            self.assertIn("| no_adapt | 0/3 | — | — |", (root/"report.md").read_text())
            self.assertEqual(len((root/"summary/overall_metrics.csv").read_text().splitlines()), 1)
