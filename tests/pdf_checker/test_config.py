from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from apps.pdf_checker.config import load_pdf_checker_config


class PDFCheckerConfigTests(unittest.TestCase):
    def test_defaults_without_bearer_token(self) -> None:
        config = load_pdf_checker_config(
            env={"CITATION_CHECKER_CONFIG_PATH": "/tmp/does_not_exist_config.json"}
        )
        self.assertEqual(config.entry_extraction.mode, "heuristic")
        self.assertEqual(config.entry_extraction.provider, "bedrock")
        self.assertIsNone(config.entry_extraction.bedrock.model_id)
        self.assertIsNone(config.entry_extraction.bedrock.bearer_token)
        self.assertIsNone(config.entry_extraction.local.model_path)
        self.assertIsNone(config.connectors.dblp_sqlite_path)

    def test_token_only_enables_model_mode_with_default_model_id(self) -> None:
        config = load_pdf_checker_config(
            env={
                "CITATION_CHECKER_CONFIG_PATH": "/tmp/does_not_exist_config.json",
                "AWS_BEARER_TOKEN_BEDROCK": "token_value",
            }
        )
        self.assertEqual(config.entry_extraction.mode, "model")
        self.assertEqual(config.entry_extraction.provider, "bedrock")
        self.assertEqual(config.entry_extraction.bedrock.model_id, "qwen.qwen3-vl-235b-a22b")
        self.assertEqual(config.entry_extraction.bedrock.bearer_token, "token_value")

    def test_explicit_mode_and_model_override(self) -> None:
        config = load_pdf_checker_config(
            env={
                "CITATION_CHECKER_CONFIG_PATH": "/tmp/does_not_exist_config.json",
                "AWS_BEARER_TOKEN_BEDROCK": "token_value",
                "CITATION_CHECKER_PDF_ENTRY_EXTRACTION": "heuristic",
                "CITATION_CHECKER_BEDROCK_MODEL_ID": "custom-model",
            }
        )
        self.assertEqual(config.entry_extraction.mode, "heuristic")
        self.assertEqual(config.entry_extraction.provider, "bedrock")
        self.assertEqual(config.entry_extraction.bedrock.model_id, "custom-model")
        self.assertEqual(config.entry_extraction.bedrock.bearer_token, "token_value")

    def test_json_config_file_loading(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            config_path = Path(tempdir) / "config.json"
            config_path.write_text(
                """{
	  "connectors": {
	    "cache_path": "/tmp/custom_cache.sqlite",
	    "dblp_mirror_path": "/tmp/custom_mirror.jsonl",
	    "dblp_sqlite_path": "/tmp/custom_dblp.sqlite",
	    "semantic_scholar_api_key": "s2_key"
	  },
	  "entry_extraction": {
	    "mode": "model",
	    "provider": "bedrock",
	    "chunk_chars": 9000,
	    "bedrock": {
	      "region": "us-west-2",
      "model_id": "my-model",
      "bearer_token": "json_token"
    }
  }
}""",
                encoding="utf-8",
            )
            config = load_pdf_checker_config(
                env={"CITATION_CHECKER_CONFIG_PATH": str(config_path)}
            )

        self.assertEqual(str(config.connectors.cache_path), "/tmp/custom_cache.sqlite")
        self.assertEqual(str(config.connectors.dblp_mirror_path), "/tmp/custom_mirror.jsonl")
        self.assertEqual(str(config.connectors.dblp_sqlite_path), "/tmp/custom_dblp.sqlite")
        self.assertEqual(config.connectors.semantic_scholar_api_key, "s2_key")
        self.assertEqual(config.entry_extraction.mode, "model")
        self.assertEqual(config.entry_extraction.provider, "bedrock")
        self.assertEqual(config.entry_extraction.chunk_chars, 9000)
        self.assertEqual(config.entry_extraction.bedrock.region, "us-west-2")
        self.assertEqual(config.entry_extraction.bedrock.model_id, "my-model")
        self.assertEqual(config.entry_extraction.bedrock.bearer_token, "json_token")

    def test_env_overrides_json_config(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            config_path = Path(tempdir) / "config.json"
            config_path.write_text(
                """{
  "entry_extraction": {
    "mode": "heuristic",
    "bedrock": {
      "model_id": "json-model",
      "bearer_token": "json-token"
    }
  }
}""",
                encoding="utf-8",
            )
            config = load_pdf_checker_config(
                env={
                    "CITATION_CHECKER_CONFIG_PATH": str(config_path),
                    "CITATION_CHECKER_PDF_ENTRY_EXTRACTION": "model",
                    "CITATION_CHECKER_PDF_MODEL_PROVIDER": "bedrock",
                    "CITATION_CHECKER_BEDROCK_MODEL_ID": "env-model",
                    "AWS_BEARER_TOKEN_BEDROCK": "env-token",
                }
            )

        self.assertEqual(config.entry_extraction.mode, "model")
        self.assertEqual(config.entry_extraction.provider, "bedrock")
        self.assertEqual(config.entry_extraction.bedrock.model_id, "env-model")
        self.assertEqual(config.entry_extraction.bedrock.bearer_token, "env-token")

    def test_token_only_in_json_enables_model_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            config_path = Path(tempdir) / "config.json"
            config_path.write_text(
                """{
  "entry_extraction": {
    "bedrock": {
      "bearer_token": "json-token-only"
    }
  }
}""",
                encoding="utf-8",
            )
            config = load_pdf_checker_config(
                env={"CITATION_CHECKER_CONFIG_PATH": str(config_path)}
            )

        self.assertEqual(config.entry_extraction.mode, "model")
        self.assertEqual(config.entry_extraction.provider, "bedrock")
        self.assertEqual(config.entry_extraction.bedrock.model_id, "qwen.qwen3-vl-235b-a22b")
        self.assertEqual(config.entry_extraction.bedrock.bearer_token, "json-token-only")

    def test_local_model_path_auto_enables_local_provider(self) -> None:
        config = load_pdf_checker_config(
            env={
                "CITATION_CHECKER_CONFIG_PATH": "/tmp/does_not_exist_config.json",
                "CITATION_CHECKER_LOCAL_MODEL_PATH": "/tmp/local-model",
            }
        )
        self.assertEqual(config.entry_extraction.mode, "model")
        self.assertEqual(config.entry_extraction.provider, "local")
        self.assertEqual(config.entry_extraction.local.model_path, "/tmp/local-model")


if __name__ == "__main__":
    unittest.main()
