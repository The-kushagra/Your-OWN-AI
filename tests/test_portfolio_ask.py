"""Local integration checks for the read-only /portfolio/ask endpoint.

Run with the server and Ollama available:
  PORTFOLIO_ASK_URL=http://127.0.0.1:8080/portfolio/ask python3 -m unittest discover -s tests
"""
import json
import os
import unittest
from urllib.request import Request, urlopen


ENDPOINT = os.environ.get("PORTFOLIO_ASK_URL", "http://127.0.0.1:8080/portfolio/ask")


def ask(question, k=1):
    payload = json.dumps({"question": question, "k": k}).encode()
    request = Request(ENDPOINT, data=payload, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=90) as response:
        return json.loads(response.read().decode())


class PortfolioAskIntegrationTests(unittest.TestCase):
    def test_hiring_claims_are_qualified_and_profile_only(self):
        result = ask("Why should Kushagra be hired for a Java Developer role?")
        answer = result["answer"]
        self.assertIn("Verified facts:", answer)
        self.assertIn("Potential role relevance (inference, not a verified qualification)", answer)
        self.assertIn("does not document professional experience, proficiency levels, or achievements", answer)
        self.assertTrue(result["sources"])
        self.assertTrue(all(source["title"].startswith("Profile |") for source in result["sources"]))

    def test_three_project_comparison_keeps_each_project_separate(self):
        result = ask("Compare all three projects and Kushagra's contributions", k=1)
        answer = result["answer"]
        for name in ("NextHire AI", "UPI Without Internet", "Your-OWN-AI"):
            self.assertIn(name, answer)
        self.assertIn("Next.js 16, TypeScript, React 19, Tailwind CSS v4", answer)
        self.assertIn("Java 17, Spring Boot 3.3.5", answer)
        self.assertIn("C++, HNSW, KD-Tree, brute-force search, Ollama embeddings", answer)
        for section in answer.split("\n\n"):
            self.assertIn("- Technologies:", section)
            self.assertIn("- Status:", section)
            self.assertIn("- Verified contributions by Kushagra:", section)
        self.assertIn("Individual contributions by Kushagra are not specified", answer)
        self.assertEqual(len(result["sources"]), 3)
        self.assertTrue(all(any(source["title"].startswith(name + " |") for source in result["sources"])
                            for name in ("NextHire AI", "UPI Without Internet", "Your-OWN-AI")))

    def test_missing_education_is_explicit(self):
        result = ask("What education has Kushagra completed?")
        self.assertIn("don't have verified information", result["answer"])
        self.assertIn("education details", result["answer"])
        self.assertTrue(result["sources"])
        self.assertTrue(all(source["title"].startswith("Profile |") for source in result["sources"]))

    def test_named_project_does_not_inherit_another_stack(self):
        result = ask("Does NextHire AI use Spring Boot?")
        self.assertTrue(all(source["title"].startswith("NextHire AI |") for source in result["sources"]))
        source_text = " ".join(source["text"] for source in result["sources"])
        answer = result["answer"].lower()
        self.assertIn("next.js 16", source_text.lower())
        self.assertNotIn("Java 17, Spring Boot 3.3.5", source_text)
        self.assertTrue(answer.startswith("no."))
        self.assertIn("java", answer)
        self.assertIn("spring boot", answer)

    def test_upi_explanation_is_conversational_and_covers_contribution_limits(self):
        result = ask(
            "Explain the UPI Without Internet project, my specific contribution, and its limitations."
        )
        answer = result["answer"]
        answer_lower = answer.lower()
        self.assertTrue(answer.startswith("UPI Without Internet is a demo"))
        self.assertIn("RSA-OAEP and AES-256-GCM", answer)
        self.assertIn("simulated devices pass packets in gossip rounds", answer_lower)
        self.assertIn("records demo settlements", answer)
        self.assertIn("GET /api/transactions/{vpa}", answer)
        self.assertIn("latest 20 transactions", answer)
        self.assertIn("sender or recipient", answer)
        self.assertIn("an API test", answer)
        self.assertIn("Pratyush Narain", answer)
        self.assertIn("real Bluetooth transfer", answer)
        self.assertIn("live UPI/payment rails", answer)
        self.assertIn("production readiness", answer)
        self.assertNotIn("- Technologies:", answer)
        self.assertTrue(result["sources"])
        self.assertTrue(all(source["title"].startswith("UPI Without Internet |")
                            for source in result["sources"]))


if __name__ == "__main__":
    unittest.main()
