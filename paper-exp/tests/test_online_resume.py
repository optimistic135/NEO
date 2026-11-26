from copy import deepcopy
import unittest

from src.engine.online_resume import _PREVIOUS, can_resume_without_cotta


class OnlineResumeTest(unittest.TestCase):
    def test_only_known_repair_reuses_flow_results(self):
        old = dict(code_git=_PREVIOUS, batch_size=64, methods=dict(cotta={}, roid=dict(eps="1e-05"), sar={}))
        new = deepcopy(old)
        new["code_git"] = "new"
        del new["methods"]["cotta"]
        new["methods"]["roid"]["eps"] = .00001
        self.assertTrue(can_resume_without_cotta(old, new, False))
        self.assertFalse(can_resume_without_cotta(old, new, True))
        self.assertFalse(can_resume_without_cotta(old, dict(new, batch_size=128), False))
