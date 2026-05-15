import os
import tempfile
import unittest
from pathlib import Path

from chat_service import chat


class ChatServiceTest(unittest.TestCase):
    def test_chat_learns_and_retrieves_memory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            previous_db = os.environ.get("ENTITY_RECOGNITION_MEMORY_DB")
            os.environ["ENTITY_RECOGNITION_MEMORY_DB"] = str(Path(temp_dir) / "memory.sqlite3")
            try:
                response = chat("Linus Torvalds created Linux in Helsinki.")

                self.assertIn("I learned this:", response["reply"])
                self.assertEqual(response["backend"], "rules")
                self.assertEqual(response["extraction"]["engine"], "fallback")
                self.assertTrue(response["memory"])
                self.assertTrue(response["message_memory"])
                self.assertEqual(response["saved_message"]["role"], "assistant")
                self.assertEqual(response["memory"][0]["source"], "Linus Torvalds")
            finally:
                if previous_db is None:
                    os.environ.pop("ENTITY_RECOGNITION_MEMORY_DB", None)
                else:
                    os.environ["ENTITY_RECOGNITION_MEMORY_DB"] = previous_db

    def test_chat_uses_ollama_when_available(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            previous_db = os.environ.get("ENTITY_RECOGNITION_MEMORY_DB")
            os.environ["ENTITY_RECOGNITION_MEMORY_DB"] = str(Path(temp_dir) / "memory.sqlite3")
            try:
                from unittest.mock import patch

                with patch("chat_service.ask_ollama", return_value=("I can talk with more context now.", None)):
                    response = chat("Linus Torvalds created Linux in Helsinki.")

                self.assertEqual(response["reply"], "I can talk with more context now.")
                self.assertEqual(response["backend"], "ollama")
                self.assertIsNone(response["backend_error"])
                self.assertTrue(response["memory"])
                self.assertTrue(response["message_memory"])
            finally:
                if previous_db is None:
                    os.environ.pop("ENTITY_RECOGNITION_MEMORY_DB", None)
                else:
                    os.environ["ENTITY_RECOGNITION_MEMORY_DB"] = previous_db


if __name__ == "__main__":
    unittest.main()
