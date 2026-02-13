"""Safe feature generation from suggestions."""

import re
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from src.logger import setup_logger
from src.feature_suggester import FeatureSuggestion
from src.config import settings
from typing import Dict, Any

logger = setup_logger(__name__)


class FeatureGenerationError(Exception):
    """Raised when feature generation fails."""

    pass


class FeatureValidator:
    """Validates suggested features before generation."""

    @staticmethod
    def validate_code(code: str) -> bool:
        """Check if code looks safe to execute."""

        # Dangerous patterns
        dangerous_patterns = [
            r"__import__",
            r"exec\(",
            r"eval\(",
            r"compile\(",
            r"open\(",
            r"subprocess",
            r"os\.system",
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, code):
                logger.warning(f"Dangerous pattern detected: {pattern}")
                return False

        return True

    @staticmethod
    def validate_dataframe_operation(code: str) -> bool:
        """Check if code uses only dataframe operations."""

        # Should reference df
        if "df" not in code:
            logger.warning("Code doesn't reference 'df'")
            return False

        # Should have assignment
        if "=" not in code:
            logger.warning("Code has no assignment")
            return False

        return True


class FeatureGenerator:
    """Generates features from suggestions safely."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize generator.

        Args:
            df: Input dataframe
        """
        self.original_df = df.copy()
        self.df = df.copy()
        self.generated_features: Dict[str, pd.Series] = {}
        self.generation_log: List[Dict[str, Any]] = []

    def generate_feature(
        self, suggestion: FeatureSuggestion
    ) -> Tuple[bool, Optional[str]]:
        """
        Generate a single feature safely.

        Args:
            suggestion: FeatureSuggestion object

        Returns:
            Tuple of (success: bool, error_message: Optional[str])
        """

        logger.info(f"Generating feature: {suggestion.name}")

        # Validate code
        if not FeatureValidator.validate_code(suggestion.python_code):
            error_msg = f"Code failed security validation: {suggestion.name}"
            logger.error(error_msg)
            return False, error_msg

        if not FeatureValidator.validate_dataframe_operation(suggestion.python_code):
            error_msg = f"Invalid dataframe operation: {suggestion.name}"
            logger.error(error_msg)
            return False, error_msg

        try:
            # Create local namespace for safe execution
            local_namespace = {
                "df": self.df,
                "pd": pd,
                "np": np,
            }

            # Execute code
            exec(suggestion.python_code, {"__builtins__": {}}, local_namespace)

            # Get the generated feature (last assigned variable)
            # For simple assignment df['feature_name'] = ...
            feature_name = suggestion.name

            if feature_name in self.df.columns:
                feature_series = self.df[feature_name]

                # Validate feature
                if len(feature_series) != len(self.df):
                    error_msg = f"Feature length mismatch for {feature_name}"
                    logger.error(error_msg)
                    return False, error_msg

                # Store generated feature
                self.generated_features[feature_name] = feature_series

                # Log success
                self.generation_log.append(
                    {
                        "feature_name": feature_name,
                        "success": True,
                        "error": None,
                        "type": suggestion.feature_type,
                    }
                )

                logger.info(f"✓ Generated feature: {feature_name}")
                return True, None
            else:
                error_msg = f"Feature {feature_name} not found after execution"
                logger.error(error_msg)
                return False, error_msg

        except Exception as e:
            error_msg = f"Error generating {suggestion.name}: {str(e)}"
            logger.error(error_msg)

            self.generation_log.append(
                {
                    "feature_name": suggestion.name,
                    "success": False,
                    "error": str(e),
                    "type": suggestion.feature_type,
                }
            )

            return False, error_msg

    def generate_all_features(
        self, suggestions: List[FeatureSuggestion], stop_on_error: bool = False
    ) -> Tuple[int, int]:
        """
        Generate all features from suggestions.

        Args:
            suggestions: List of FeatureSuggestion objects
            stop_on_error: Stop if any feature fails

        Returns:
            Tuple of (successful_count, failed_count)
        """

        logger.info(f"Generating {len(suggestions)} features")

        successful = 0
        failed = 0

        for suggestion in suggestions:
            success, error = self.generate_feature(suggestion)

            if success:
                successful += 1
            else:
                failed += 1
                if stop_on_error:
                    logger.error(f"Stopping due to error: {error}")
                    break

        logger.info(f"Generation complete: {successful} successful, {failed} failed")

        return successful, failed

    def get_enriched_dataframe(self) -> pd.DataFrame:
        """
        Get dataframe with generated features.

        Returns:
            Original dataframe + generated features
        """

        result = self.original_df.copy()

        for feature_name, feature_series in self.generated_features.items():
            result[feature_name] = feature_series.values

        logger.info(f"Enriched dataframe shape: {result.shape}")

        return result

    def get_new_features_only(self) -> pd.DataFrame:
        """Get only the newly generated features."""

        if not self.generated_features:
            return pd.DataFrame()

        return pd.DataFrame(self.generated_features)

    def validate_features(self) -> Dict[str, Any]:
        """Validate generated features for quality."""

        validation_report: Dict[str, Any] = {
            'total_features': len(self.generated_features),
            'valid_features': 0,
            'invalid_features': [],
            'high_variance_features': [],
            'constant_features': [],
            'high_null_features': [],
        }

        for feature_name, feature_series in self.generated_features.items():
            # Check for constant values
            if feature_series.nunique() == 1:
                validation_report["constant_features"].append(feature_name)
                continue

            # Check for high null percentage
            null_pct = feature_series.isnull().sum() / len(feature_series) * 100
            if null_pct > 30:
                validation_report["high_null_features"].append(
                    {"feature": feature_name, "null_percentage": null_pct}
                )
                continue

            # Check for high variance (might indicate outliers)
            if feature_series.dtype in ["int64", "float64"]:
                cv = feature_series.std() / (abs(feature_series.mean()) + 1e-8)
                if cv > 10:
                    validation_report["high_variance_features"].append(
                        {"feature": feature_name, "coefficient_of_variation": cv}
                    )

            validation_report["valid_features"] += 1

        logger.info(
            f"Validation report: {validation_report['valid_features']}/{validation_report['total_features']} valid"
        )

        return validation_report

    def get_generation_log(self) -> List[Dict[str, Any]]:
        """Get log of all generation attempts."""
        return self.generation_log
