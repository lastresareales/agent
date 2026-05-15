import tempfile
import unittest
from pathlib import Path

from memory_store import learn_from_extraction, memory_summary, save_message, search_facts, search_messages


class MemoryStoreTest(unittest.TestCase):
    def test_learns_and_searches_graph_facts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "memory.sqlite3"
            extraction = {
                "text": "Satya Nadella leads Microsoft from Redmond.",
                "entities": [
                    {
                        "text": "Satya Nadella",
                        "label": "PER",
                        "confidence": 0.99,
                    },
                    {
                        "text": "Microsoft",
                        "label": "ORG",
                        "confidence": 0.97,
                    },
                ],
                "graph": {
                    "edges": [
                        {"source": "Satya Nadella", "relation": "LEADS", "target": "Microsoft"},
                    ]
                },
            }

            learned = learn_from_extraction(extraction, db_path=db_path)
            learn_from_extraction(extraction, db_path=db_path)

            self.assertEqual(
                learned,
                [{"source": "Satya Nadella", "relation": "LEADS", "target": "Microsoft", "confidence": 0.97}],
            )
            self.assertEqual(memory_summary(db_path=db_path), {"facts": 1, "occurrences": 2, "messages": 0})

            facts = search_facts("microsoft", db_path=db_path)
            self.assertEqual(len(facts), 1)
            self.assertEqual(facts[0]["source"], "Satya Nadella")
            self.assertEqual(facts[0]["relation"], "LEADS")
            self.assertEqual(facts[0]["target"], "Microsoft")
            self.assertEqual(facts[0]["occurrences"], 2)

    def test_saves_and_searches_raw_messages(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "memory.sqlite3"

            saved = save_message("user", "My favorite programming language is Python.", db_path=db_path)

            self.assertEqual(saved["role"], "user")
            self.assertGreaterEqual(saved["importance"], 0.5)
            self.assertEqual(memory_summary(db_path=db_path), {"facts": 0, "occurrences": 0, "messages": 1})

            messages = search_messages("python", db_path=db_path)
            self.assertEqual(len(messages), 1)
            self.assertEqual(messages[0]["content"], "My favorite programming language is Python.")


if __name__ == "__main__":
    unittest.main()
