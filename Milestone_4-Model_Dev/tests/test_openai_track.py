from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from _bootstrap import ensure_src_path


M4_ROOT = ensure_src_path()

from m4_model_dev.models.model_registry import resolve_candidate_spec
from m4_model_dev.training.openai_finetune import _to_openai_chat_record


class OpenAITrackTests(unittest.TestCase):
    def test_openai_base_candidate_is_available(self) -> None:
        spec = resolve_candidate_spec("openai_gpt41_mini_base")

        self.assertEqual(spec.backend, "openai")
        self.assertEqual(spec.model_name, "gpt-4.1-mini-2025-04-14")
        self.assertTrue(spec.enabled)

    def test_openai_finetuned_candidate_can_be_enabled_from_environment(self) -> None:
        with patch.dict("os.environ", {"OPENAI_FINE_TUNED_MODEL": "ft:gpt-4o-mini:test::abc123"}, clear=False):
            spec = resolve_candidate_spec("openai_gpt41_mini_finetuned")

        self.assertEqual(spec.backend, "openai")
        self.assertTrue(spec.enabled)
        self.assertEqual(spec.model_name, "ft:gpt-4o-mini:test::abc123")

    def test_sft_record_converts_to_openai_chat_format(self) -> None:
        converted = _to_openai_chat_record({"prompt": "write solver", "response": "def solve(instance_path): return {}"})

        self.assertEqual(len(converted["messages"]), 3)
        self.assertEqual(converted["messages"][0]["role"], "system")
        self.assertEqual(converted["messages"][1]["role"], "user")
        self.assertEqual(converted["messages"][2]["role"], "assistant")


if __name__ == "__main__":
    unittest.main()
