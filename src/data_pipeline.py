"""End-to-end data pipeline orchestration."""

from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
from sklearn.model_selection import train_test_split
from src.logger import setup_logger
from src.dataset_analyzer import DatasetAnalyzer
from src.feature_suggester import FeatureSuggester
from src.feature_generator import FeatureGenerator
from src.config import settings

logger = setup_logger(__name__)


class DataPipeline:
    """Orchestrates the complete data pipeline."""

    def __init__(self, data_path: Path, target_col: str):
        """
        Initialize pipeline.

        Args:
            data_path: Path to dataset
            target_col: Name of target column
        """
        self.data_path = Path(data_path)
        self.target_col = target_col

        self.raw_df: Optional[pd.DataFrame] = None
        self.analyzer: Optional[DatasetAnalyzer] = None
        self.suggester: Optional[FeatureSuggester] = None
        self.generator: Optional[FeatureGenerator] = None
        self.enriched_df: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
        """Load raw dataset."""
        logger.info(f"Loading data from {self.data_path}")

        if self.data_path.suffix == ".csv":
            self.raw_df = pd.read_csv(self.data_path)
        elif self.data_path.suffix == ".parquet":
            self.raw_df = pd.read_parquet(self.data_path)
        else:
            raise ValueError(f"Unsupported format: {self.data_path.suffix}")

        logger.info(f"Data loaded: {self.raw_df.shape}")
        return self.raw_df

    def analyze_dataset(self) -> DatasetAnalyzer:
        """Analyze dataset and extract metadata."""
        if self.raw_df is None:
            self.load_data()

        assert self.raw_df is not None

        logger.info("Analyzing dataset...")
        self.analyzer = DatasetAnalyzer(self.raw_df, self.target_col)
        self.analyzer.analyze()

        return self.analyzer

    def suggest_features(self, num_suggestions: int = 10) -> list:
        """Generate feature suggestions."""
        if self.analyzer is None:
            self.analyze_dataset()

        assert self.analyzer is not None

        logger.info("Generating feature suggestions...")
        self.suggester = FeatureSuggester()

        # Determine task type
        task_type = self.analyzer.metadata["target_info"]["type"]

        suggestions = self.suggester.suggest_features(
            self.analyzer, num_suggestions=num_suggestions, task_type=task_type
        )

        return suggestions

    def generate_features(self, suggestions: list) -> Tuple[int, int]:
        """Generate features from suggestions."""
        if self.raw_df is None:
            self.load_data()

        assert self.raw_df is not None

        logger.info("Generating features...")
        self.generator = FeatureGenerator(self.raw_df)

        successful, failed = self.generator.generate_all_features(suggestions)

        self.enriched_df = self.generator.get_enriched_dataframe()

        return successful, failed

    def save_enriched_data(self, output_path: Path) -> None:
        """Save enriched dataset."""
        if self.enriched_df is None:
            raise ValueError(
                "No enriched data to save. Run generate_features first."
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == ".csv":
            self.enriched_df.to_csv(output_path, index=False)
        elif output_path.suffix == ".parquet":
            self.enriched_df.to_parquet(output_path, index=False)

        logger.info(f"Enriched data saved to {output_path}")

    def split_data(
        self, test_size: float = 0.2, random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train/test.

        Returns:
            X_train, X_test, y_train, y_test
        """
        if self.enriched_df is None:
            raise ValueError(
                "No enriched data to split. Run generate_features first."
            )

        assert self.enriched_df is not None

        X = self.enriched_df.drop(columns=[self.target_col])
        y = self.enriched_df[self.target_col]

        # Determine if stratified split should be used
        stratify = None
        if (
            self.analyzer is not None
            and self.analyzer.metadata["target_info"]["type"] == "classification"
        ):
            stratify = y

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )

        logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

        return X_train, X_test, y_train, y_test

    def run_full_pipeline(self, num_suggestions: int = 10) -> dict:
        """Execute full pipeline end-to-end."""

        logger.info("Starting full pipeline execution...")

        # Load
        self.load_data()

        # Analyze
        self.analyze_dataset()

        # Suggest
        suggestions = self.suggest_features(num_suggestions)

        # Generate
        successful, failed = self.generate_features(suggestions)

        # Validate
        if self.generator is not None:
            validation = self.generator.validate_features()
        else:
            validation = {}

        # Save
        output_path = settings.data_processed_path / "enriched_data.csv"
        self.save_enriched_data(output_path)

        result = {
            "original_shape": self.raw_df.shape if self.raw_df is not None else None,
            "enriched_shape": (
                self.enriched_df.shape if self.enriched_df is not None else None
            ),
            "suggestions_count": len(suggestions),
            "generated_successful": successful,
            "generated_failed": failed,
            "validation": validation,
            "enriched_data_path": str(output_path),
        }

        logger.info("Pipeline execution complete")

        return result