"""Tests for the training module."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.train import compute_asymmetric_score, flatten_dict, get_git_hash, hash_file


class TestAsymmetricScore:
    """Tests for the PHM asymmetric scoring function."""

    def test_perfect_prediction_scores_zero(self) -> None:
        """Perfect predictions should score 0."""
        y_true = np.array([50, 100, 25])
        y_pred = np.array([50, 100, 25])
        score = compute_asymmetric_score(y_true, y_pred)
        assert abs(score) < 1e-10

    def test_late_predictions_penalized_more(self) -> None:
        """Late predictions should be penalized more than early ones."""
        y_true = np.array([50.0])

        # Early by 10 cycles
        early_score = compute_asymmetric_score(y_true, np.array([40.0]))

        # Late by 10 cycles
        late_score = compute_asymmetric_score(y_true, np.array([60.0]))

        assert late_score > early_score

    def test_non_negative_score(self) -> None:
        """Asymmetric score should always be non-negative."""
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 125, 100).astype(float)
        y_pred = y_true + rng.normal(0, 10, 100)
        score = compute_asymmetric_score(y_true, y_pred)
        assert score >= 0


class TestFlattenDict:
    """Tests for dictionary flattening utility."""

    def test_flat_dict_unchanged(self) -> None:
        """Already flat dict should be returned as-is."""
        d = {"a": 1, "b": "hello"}
        assert flatten_dict(d) == d

    def test_nested_dict_flattened(self) -> None:
        """Nested dicts should be flattened with dot separators."""
        d = {"model": {"xgboost": {"max_depth": 6}}}
        result = flatten_dict(d)
        assert result["model.xgboost.max_depth"] == 6

    def test_list_converted_to_string(self) -> None:
        """List values should be converted to strings."""
        d = {"sensors": [1, 2, 3]}
        result = flatten_dict(d)
        assert result["sensors"] == "[1, 2, 3]"


class TestHashFile:
    """Tests for file hashing utility."""

    def test_hash_existing_file(self, tmp_path: Path) -> None:
        """Hash of a file should be a non-empty string."""
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        result = hash_file(f)
        assert len(result) == 12
        assert isinstance(result, str)

    def test_hash_missing_file(self) -> None:
        """Missing file should return sentinel value."""
        result = hash_file("nonexistent.txt")
        assert result == "file_not_found"

    def test_same_content_same_hash(self, tmp_path: Path) -> None:
        """Same file content should produce same hash."""
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("identical content")
        f2.write_text("identical content")
        assert hash_file(f1) == hash_file(f2)


class TestGetGitHash:
    """Tests for Git hash retrieval."""

    def test_returns_string(self) -> None:
        """Should return a string (either hash or 'unknown')."""
        result = get_git_hash()
        assert isinstance(result, str)
        assert len(result) > 0
