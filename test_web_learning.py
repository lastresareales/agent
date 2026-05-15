import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from web_learning import learn_url


class WebLearningTest(unittest.TestCase):
    def test_learns_from_fetched_page_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            previous_db = os.environ.get("ENTITY_RECOGNITION_MEMORY_DB")
            os.environ["ENTITY_RECOGNITION_MEMORY_DB"] = str(Path(temp_dir) / "memory.sqlite3")
            try:
                with patch(
                    "web_learning.fetch_url_text",
                    return_value="Linus Torvalds created Linux in Helsinki.",
                ):
                    response = learn_url("https://example.com/linux")

                self.assertEqual(response["url"], "https://example.com/linux")
                self.assertEqual(response["characters_read"], 41)
                self.assertEqual(response["extraction"]["entities"][0]["text"], "Linus Torvalds")
                self.assertTrue(response["extraction"]["learned_facts"])
            finally:
                if previous_db is None:
                    os.environ.pop("ENTITY_RECOGNITION_MEMORY_DB", None)
                else:
                    os.environ["ENTITY_RECOGNITION_MEMORY_DB"] = previous_db


if __name__ == "__main__":
    unittest.main()
