"""Tests for Config class, particularly client resolution."""

import os
from unittest.mock import patch, MagicMock

import openai
import pytest

from llmcomp.config import Config, _get_api_keys, _discover_url_key_pairs


class TestGetApiKeys:
    """Tests for _get_api_keys function."""

    def test_returns_env_var_name_and_key(self):
        """Test that _get_api_keys returns (env_var_name, key) tuples."""
        with patch.dict(os.environ, {"TEST_API_KEY": "secret123"}, clear=False):
            result = _get_api_keys("TEST_API_KEY")
            assert result == [("TEST_API_KEY", "secret123")]

    def test_includes_suffixed_variants(self):
        """Test that suffixed variants are included."""
        env = {
            "TEST_API_KEY": "key0",
            "TEST_API_KEY_A": "keyA",
            "TEST_API_KEY_B": "keyB",
        }
        with patch.dict(os.environ, env, clear=False):
            result = _get_api_keys("TEST_API_KEY")
            # Should include base and suffixed variants
            assert ("TEST_API_KEY", "key0") in result
            assert ("TEST_API_KEY_A", "keyA") in result
            assert ("TEST_API_KEY_B", "keyB") in result
            assert len(result) == 3

    def test_excludes_suffixed_when_disabled(self):
        """Test that suffixed variants can be excluded."""
        env = {
            "TEST_API_KEY": "key0",
            "TEST_API_KEY_A": "keyA",
        }
        with patch.dict(os.environ, env, clear=False):
            result = _get_api_keys("TEST_API_KEY", include_suffixed=False)
            assert result == [("TEST_API_KEY", "key0")]

    def test_skips_missing_keys(self):
        """Test that missing env vars are skipped."""
        with patch.dict(os.environ, {}, clear=True):
            result = _get_api_keys("NONEXISTENT_KEY")
            assert result == []


class TestDiscoverUrlKeyPairs:
    """Tests for _discover_url_key_pairs function."""

    def test_returns_tuples_with_env_name(self):
        """Test that url_key_pairs include env var names."""
        env = {"OPENAI_API_KEY": "sk-test123"}
        with patch.dict(os.environ, env, clear=True):
            result = _discover_url_key_pairs()
            assert len(result) == 1
            url, key, env_name = result[0]
            assert url == "https://api.openai.com/v1"
            assert key == "sk-test123"
            assert env_name == "OPENAI_API_KEY"


class TestClientForModel:
    """Tests for Config.client_for_model and key selection logic."""

    @pytest.fixture(autouse=True)
    def reset_config(self):
        """Reset Config before and after each test."""
        Config.reset()
        yield
        Config.reset()

    def test_selects_lexicographically_lowest_env_var(self):
        """Test that when multiple keys work, the one with lowest env var name is selected."""
        # Set up multiple API keys
        env = {
            "OPENAI_API_KEY_Z": "key-z",
            "OPENAI_API_KEY_A": "key-a",
            "OPENAI_API_KEY": "key-base",
        }

        # Track which client was created with which key
        created_clients = {}

        def mock_test_url_key_pair(model, url, key):
            """Mock that returns a client for all keys."""
            client = MagicMock(spec=openai.OpenAI)
            client.api_key = key
            created_clients[key] = client
            return client

        with patch.dict(os.environ, env, clear=True):
            Config._url_key_pairs = None  # Force rediscovery
            with patch.object(Config, "_test_url_key_pair", side_effect=mock_test_url_key_pair):
                client = Config.client_for_model("gpt-4")

        # Should select the key from OPENAI_API_KEY (lexicographically lowest)
        assert client.api_key == "key-base"

    def test_selects_lowest_among_working_keys(self):
        """Test that only working keys are considered for selection."""
        env = {
            "OPENAI_API_KEY_Z": "key-z",
            "OPENAI_API_KEY_A": "key-a",  # This will "fail"
            "OPENAI_API_KEY": "key-base",  # This will "fail"
        }

        def mock_test_url_key_pair(model, url, key):
            """Mock that only returns client for key-z."""
            if key == "key-z":
                client = MagicMock(spec=openai.OpenAI)
                client.api_key = key
                return client
            return None

        with patch.dict(os.environ, env, clear=True):
            Config._url_key_pairs = None
            with patch.object(Config, "_test_url_key_pair", side_effect=mock_test_url_key_pair):
                client = Config.client_for_model("gpt-4")

        # Should select key-z since it's the only one that works
        assert client.api_key == "key-z"

    def test_deterministic_selection_with_multiple_providers(self):
        """Test deterministic selection across different providers."""
        env = {
            "OPENROUTER_API_KEY": "or-key",
            "OPENAI_API_KEY": "oai-key",
        }

        def mock_test_url_key_pair(model, url, key):
            """Mock that returns client for all keys."""
            client = MagicMock(spec=openai.OpenAI)
            client.api_key = key
            client.base_url = url
            return client

        with patch.dict(os.environ, env, clear=True):
            Config._url_key_pairs = None
            with patch.object(Config, "_test_url_key_pair", side_effect=mock_test_url_key_pair):
                client = Config.client_for_model("some-model")

        # OPENAI_API_KEY < OPENROUTER_API_KEY lexicographically
        assert client.api_key == "oai-key"



