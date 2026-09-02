"""Unit tests for the Arctan ML scoring API."""

import unittest

from fastapi.testclient import TestClient

from arctan.main import app


class TestHealthEndpoints(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_healthz(self) -> None:
        response = self.client.get("/healthz")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["service"], "arctan-ml")

    def test_readyz(self) -> None:
        response = self.client.get("/readyz")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn(data["status"], ["ok", "not_trained"])


class TestScoringEndpoints(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_score_single_entity(self) -> None:
        response = self.client.get("/api/v1/scores/test-entity-001")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["success"])
        self.assertIn("entityId", body["data"])
        self.assertIn("riskScore", body["data"])
        self.assertIn("fraudProbability", body["data"])
        self.assertIn("explanation", body["data"])

    def test_score_batch(self) -> None:
        response = self.client.post(
            "/api/v1/scores/batch",
            json={"entity_ids": ["entity-1", "entity-2", "entity-3"]},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["success"])
        self.assertEqual(len(body["data"]), 3)

    def test_batch_max_limit(self) -> None:
        too_many = [f"entity-{i}" for i in range(101)]
        response = self.client.post(
            "/api/v1/scores/batch",
            json={"entity_ids": too_many},
        )
        self.assertEqual(response.status_code, 422)


class TestRiskLevelMapping(unittest.TestCase):
    def test_risk_levels(self) -> None:
        from arctan.inference import FraudScorer

        self.assertEqual(FraudScorer._get_risk_level(900), "critical")
        self.assertEqual(FraudScorer._get_risk_level(600), "high")
        self.assertEqual(FraudScorer._get_risk_level(300), "medium")
        self.assertEqual(FraudScorer._get_risk_level(100), "low")


if __name__ == "__main__":
    unittest.main()
