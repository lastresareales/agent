import unittest
from unittest.mock import patch

from entities import ExtractedEntity
from recognition_service import extract_text


class RecognitionServiceTest(unittest.TestCase):
    def test_fallback_extracts_entities_and_graph(self):
        result = extract_text("Linus Torvalds created Linux in Helsinki.", learn=False)

        self.assertEqual(result["engine"], "fallback")
        self.assertEqual(result["text"], "Linus Torvalds created Linux in Helsinki.")

        entities = result["entities"]
        self.assertEqual([entity["text"] for entity in entities], ["Linus Torvalds", "Linux", "Helsinki"])
        self.assertEqual([entity["label"] for entity in entities], ["PER", "MISC", "LOC"])

        graph = result["graph"]
        self.assertEqual(
            graph["edges"],
            [
                {"source": "Linus Torvalds", "relation": "CREATED", "target": "Linux"},
                {"source": "Linus Torvalds", "relation": "MENTIONED_IN", "target": "Helsinki"},
            ],
        )

    def test_model_path_uses_offsets_and_confidence(self):
        class FakeModel:
            def extract(self, text):
                return [
                    ExtractedEntity(word="Satya Nadella", label="PER", start=0, end=13, confidence=0.99),
                    ExtractedEntity(word="Microsoft", label="ORG", start=20, end=29, confidence=0.97),
                    ExtractedEntity(word="Redmond", label="LOC", start=35, end=42, confidence=0.95),
                ]

        with patch("recognition_service.get_model", return_value=FakeModel()):
            result = extract_text("Satya Nadella leads Microsoft from Redmond.", learn=False)

        self.assertEqual(result["engine"], "bert")
        self.assertEqual(
            result["entities"],
            [
                {
                    "id": "Satya Nadella-0",
                    "text": "Satya Nadella",
                    "label": "PER",
                    "start": 0,
                    "end": 13,
                    "confidence": 0.99,
                },
                {
                    "id": "Microsoft-20",
                    "text": "Microsoft",
                    "label": "ORG",
                    "start": 20,
                    "end": 29,
                    "confidence": 0.97,
                },
                {
                    "id": "Redmond-35",
                    "text": "Redmond",
                    "label": "LOC",
                    "start": 35,
                    "end": 42,
                    "confidence": 0.95,
                },
            ],
        )
        self.assertIn({"source": "Satya Nadella", "relation": "LEADS", "target": "Microsoft"}, result["graph"]["edges"])
        self.assertIn({"source": "Microsoft", "relation": "LOCATED_IN", "target": "Redmond"}, result["graph"]["edges"])

    def test_model_path_repairs_contiguous_split_word_labels(self):
        class FakeModel:
            def extract(self, text):
                return [
                    ExtractedEntity(word="Red", label="PER", start=15, end=18, confidence=0.70),
                    ExtractedEntity(word="mond", label="LOC", start=18, end=22, confidence=0.95),
                ]

        with patch("recognition_service.get_model", return_value=FakeModel()):
            result = extract_text("Microsoft from Redmond.", learn=False)

        self.assertEqual(
            result["entities"],
            [
                {
                    "id": "Redmond-15",
                    "text": "Redmond",
                    "label": "LOC",
                    "start": 15,
                    "end": 22,
                    "confidence": 0.70,
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
