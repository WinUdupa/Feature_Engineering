"""LLM-based feature suggestion engine using Google Gemini."""

import json
from typing import List, Dict, Any, Optional
from google import genai
from src.logger import setup_logger
from src.config import settings
from src.dataset_analyzer import DatasetAnalyzer


logger = setup_logger(__name__)

# Initialize Gemini client globally
try:
    client = genai.Client(api_key=settings.google_api_key)
except Exception as e:
    logger.warning(f"Failed to initialize Gemini client: {e}")
    client = None


class FeatureSuggestion:
    """Represents a suggested feature."""

    def __init__(
        self,
        name: str,
        formula: str,
        rationale: str,
        feature_type: str,
        python_code: str,
    ):
        self.name = name
        self.formula = formula
        self.rationale = rationale
        self.feature_type = feature_type
        self.python_code = python_code

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "formula": self.formula,
            "rationale": self.rationale,
            "feature_type": self.feature_type,
            "python_code": self.python_code,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureSuggestion":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            formula=data["formula"],
            rationale=data["rationale"],
            feature_type=data["feature_type"],
            python_code=data["python_code"],
        )


class FeatureSuggester:
    """Generates feature suggestions using Google Gemini API."""

    def __init__(self):
        """Initialize the suggester."""
        global client

        if not settings.google_api_key:
            raise ValueError("GOOGLE_API_KEY not set in environment")

        if client is None:
            client = genai.Client(api_key=settings.google_api_key)

        self.client = client
        self.conversation_history = []
        self.suggestions: List[FeatureSuggestion] = []

    def suggest_features(
        self,
        analyzer: DatasetAnalyzer,
        num_suggestions: int = 10,
        task_type: str = "classification",
    ) -> List[FeatureSuggestion]:
        """
        Generate feature suggestions for a dataset.

        Args:
            analyzer: DatasetAnalyzer instance
            num_suggestions: Number of features to suggest
            task_type: 'classification' or 'regression'

        Returns:
            List of FeatureSuggestion objects
        """

        logger.info(f"Generating {num_suggestions} feature suggestions using Gemini")

        # Build prompt
        prompt = self._build_prompt(analyzer, num_suggestions, task_type)

        # Call Gemini
        response = self._call_gemini(prompt)

        # Parse suggestions
        self.suggestions = self._parse_suggestions(response)

        logger.info(f"Generated {len(self.suggestions)} suggestions")

        return self.suggestions

    def _build_prompt(
        self, analyzer: DatasetAnalyzer, num_suggestions: int, task_type: str
    ) -> str:
        """Build the LLM prompt."""

        summary = analyzer.get_summary_for_llm()
        numeric_features = list(analyzer.metadata["numeric_features"].keys())
        categorical_features = list(analyzer.metadata["categorical_features"].keys())

        prompt = f"""You are an expert data scientist specializing in feature engineering.

DATASET INFORMATION:
{summary}

NUMERIC FEATURES:
{', '.join(numeric_features[:15])}

CATEGORICAL FEATURES:
{', '.join(categorical_features[:15])}

TASK: {task_type.upper()}

Your task is to suggest {num_suggestions} NEW FEATURES that could improve model performance.

IMPORTANT REQUIREMENTS:
1. Features must be mathematically sound and interpretable
2. Features should leverage relationships between existing features
3. Avoid creating features that are just copies of existing ones
4. Include diverse feature types: ratios, polynomials, interactions, aggregations
5. Return ONLY valid JSON, no markdown or extra text

For each feature, provide:
- name: A descriptive name (snake_case)
- formula: Mathematical formula for clarity
- rationale: Why this feature helps (1-2 sentences)
- feature_type: 'numerical', 'categorical', or 'binary'
- python_code: Valid pandas/numpy code that creates the feature

PYTHON CODE GUIDELINES:
- Use 'df' to refer to the dataframe
- Use 'pd' for pandas and 'np' for numpy
- Handle edge cases (division by zero, missing values)
- Example: df['age_squared'] = df['age'] ** 2
- Example: df['age_income_ratio'] = df['age'] / (df['income'] + 1)

Return a JSON array with exactly {num_suggestions} feature objects:
[
  {{
    "name": "feature_name",
    "formula": "mathematical formula",
    "rationale": "why this helps",
    "feature_type": "numerical",
    "python_code": "df['feature_name'] = ..."
  }},
  ...
]

Only return the JSON array, nothing else."""

        return prompt

    def _call_gemini(self, user_message: str) -> str:
        """Call Gemini API."""

        logger.info("Calling Gemini API...")

        try:
            response = self.client.models.generate_content(
                model=settings.google_model,
                contents=user_message,
                config=genai.types.GenerateContentConfig(
                    max_output_tokens=settings.google_max_tokens,
                    temperature=0.7,
                ),
            )

            assistant_message = response.text

            logger.info("Gemini API call successful")

            return assistant_message

        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            raise

    def _parse_suggestions(self, response: str) -> List[FeatureSuggestion]:
        """Parse JSON response into FeatureSuggestion objects."""

        try:
            # Extract JSON from response
            json_str = response.strip()
            if json_str.startswith("```"):
                json_str = json_str[json_str.find("[") : json_str.rfind("]") + 1]

            features_data = json.loads(json_str)

            suggestions = []
            for feature_data in features_data:
                try:
                    suggestion = FeatureSuggestion.from_dict(feature_data)
                    suggestions.append(suggestion)
                except Exception as e:
                    logger.warning(f"Failed to parse feature: {e}")
                    continue

            return suggestions

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response: {response}")
            return []

    def refine_suggestions(self, feedback: str) -> List[FeatureSuggestion]:
        """Refine suggestions based on feedback."""

        logger.info("Refining suggestions based on feedback")

        refinement_prompt = f"""Based on the features you suggested earlier, here is some feedback:

{feedback}

Please suggest {len(self.suggestions)} improved features that address this feedback.
Return the same JSON format as before."""

        response = self._call_gemini(refinement_prompt)
        self.suggestions = self._parse_suggestions(response)

        return self.suggestions

    def save_suggestions(self, output_path: str) -> None:
        """Save suggestions to JSON file."""

        data = [s.to_dict() for s in self.suggestions]

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Suggestions saved to {output_path}")

    def load_suggestions(self, input_path: str) -> List[FeatureSuggestion]:
        """Load suggestions from JSON file."""

        with open(input_path, "r") as f:
            data = json.load(f)

        self.suggestions = [FeatureSuggestion.from_dict(item) for item in data]

        logger.info(f"Loaded {len(self.suggestions)} suggestions from {input_path}")

        return self.suggestions
