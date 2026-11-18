import unittest

from src.utils.settings import config


class ConfigTests(unittest.TestCase):
    def test_production_seed_and_corruption_matrix(self):
        cfg = config("datasets/corruptions.yaml")
        self.assertEqual(cfg["seeds"], [137, 2029, 8671])
        self.assertEqual(len(set(cfg["corruptions"])), 15)
        self.assertEqual(cfg["severity"], 5)
