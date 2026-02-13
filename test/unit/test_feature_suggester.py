"""Tests for feature suggester module with Gemini."""

import pytest
import json
from src.feature_suggester import FeatureSuggestion, FeatureSuggester
from src.dataset_analyzer import DatasetAnalyzer
import pandas as pd


@pytest.fixture
def sample_suggestion():
    """Create sample suggestion."""
    return FeatureSuggestion(
        name="age_squared",
        formula="age ** 2",
        rationale="Capture non-linear age effects",
        feature_type="numerical",
        python_code="df['age_squared'] = df['age'] ** 2",
    )


@pytest.fixture
def sample_dataframe():
    """Create sample dataframe."""
    return pd.DataFrame(
        {
            "age": [20, 30, 40, 50, 60],
            "income": [30000, 50000, 70000, 90000, 100000],
            "target": [0, 1, 1, 1, 0],
        }
    )


class TestFeatureSuggestion:
    """Test FeatureSuggestion class."""

    def test_creation(self, sample_suggestion):
        """Test suggestion creation."""
        assert sample_suggestion.name == "age_squared"
        assert sample_suggestion.feature_type == "numerical"

    def test_to_dict(self, sample_suggestion):
        """Test conversion to dict."""
        data = sample_suggestion.to_dict()
        assert isinstance(data, dict)
        assert data["name"] == "age_squared"
        assert "python_code" in data

    def test_from_dict(self, sample_suggestion):
        """Test creation from dict."""
        data = sample_suggestion.to_dict()
        reconstructed = FeatureSuggestion.from_dict(data)
        assert reconstructed.name == sample_suggestion.name


class TestFeatureSuggesterMocking:
    """Test FeatureSuggester with mocking."""

    def test_parse_suggestions(self, sample_dataframe):
        """Test suggestion parsing."""
        suggester = FeatureSuggester()

        mock_response = json.dumps(
            [
                {
                    "name": "age_squared",
                    "formula": "age ** 2",
                    "rationale": "Test",
                    "feature_type": "numerical",
                    "python_code": "df['age_squared'] = df['age'] ** 2",
                }
            ]
        )

        suggestions = suggester._parse_suggestions(mock_response)

        assert len(suggestions) == 1
        assert suggestions[0].name == "age_squared"

    def test_parse_invalid_json(self):
        """Test parsing invalid JSON."""
        suggester = FeatureSuggester()

        suggestions = suggester._parse_suggestions("invalid json {]")

        assert len(suggestions) == 0
