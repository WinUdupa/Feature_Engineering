"""Analyze datasets and extract metadata for LLM."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import json
from src.logger import setup_logger
from src.config import settings
from typing import Dict, Any

logger = setup_logger(__name__)


class DatasetAnalyzer:
    """Analyzes dataset and extracts metadata for feature suggestions."""

    def __init__(self, df: pd.DataFrame, target_col: str):
        """
        Initialize analyzer.

        Args:
            df: Input dataframe
            target_col: Name of target column
        """
        self.df = df
        self.target_col = target_col
        self.metadata: Dict[str, Any] = {}

    def analyze(self) -> Dict[str, Any]:
        """Extract comprehensive dataset metadata."""

        logger.info(f"Analyzing dataset with shape {self.df.shape}")

        self.metadata = {
            "basic_info": self._get_basic_info(),
            "numeric_features": self._analyze_numeric(),
            "categorical_features": self._analyze_categorical(),
            "target_info": self._analyze_target(),
            "missing_data": self._analyze_missing(),
            "correlations": self._analyze_correlations(),
        }

        return self.metadata

    def _get_basic_info(self) -> Dict[str, Any]:
        """Get basic dataset information."""
        return {
            "shape": {"rows": self.df.shape[0], "columns": self.df.shape[1]},
            "columns": self.df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
        }

    def _analyze_numeric(self) -> Dict[str, Any]:
        """Analyze numeric features."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            return {}

        analysis = {}
        for col in numeric_cols:
            analysis[col] = {
                "type": "numeric",
                "min": float(self.df[col].min()),
                "max": float(self.df[col].max()),
                "mean": float(self.df[col].mean()),
                "median": float(self.df[col].median()),
                "std": float(self.df[col].std()),
                "skewness": float(self.df[col].skew()),
                "kurtosis": float(self.df[col].kurtosis()),
            }

        return analysis

    def _analyze_categorical(self) -> Dict[str, Any]:
        """Analyze categorical features."""
        categorical_cols = self.df.select_dtypes(include=["object"]).columns.tolist()

        analysis = {}
        for col in categorical_cols:
            unique_vals = self.df[col].unique().tolist()
            analysis[col] = {
                "type": "categorical",
                "unique_count": len(unique_vals),
                "unique_values": unique_vals[:10],  # First 10
                "mode": str(self.df[col].mode().values[0])
                if not self.df[col].mode().empty
                else None,
            }

        return analysis

    def _analyze_target(self) -> Dict[str, Any]:
        """Analyze target variable."""
        target_dtype = self.df[self.target_col].dtype

        if np.issubdtype(target_dtype, np.number):
            return {
                "type": "regression",
                "min": float(self.df[self.target_col].min()),
                "max": float(self.df[self.target_col].max()),
                "mean": float(self.df[self.target_col].mean()),
            }
        else:
            return {
                "type": "classification",
                "classes": self.df[self.target_col].unique().tolist(),
                "class_distribution": self.df[self.target_col].value_counts().to_dict(),
            }

    def _analyze_missing(self) -> Dict[str, float]:
        """Analyze missing values."""
        missing = self.df.isnull().sum()
        return {
            col: float(missing[col] / len(self.df) * 100)
            for col in missing[missing > 0].index
        }

    def _analyze_correlations(self) -> Dict[str, list]:
        """Get top correlations."""
        numeric_df = self.df.select_dtypes(include=[np.number])

        if numeric_df.shape[1] < 2:
            return {}

        corr_matrix = numeric_df.corr()

        # Get top correlations
        high_corr = {}
        for col in corr_matrix.columns:
            correlations = corr_matrix[col][corr_matrix[col].abs() > 0.7]
            if len(correlations) > 1:
                high_corr[col] = [
                    (str(idx), float(val))
                    for idx, val in correlations.items()
                    if idx != col
                ]

        return high_corr

    def get_summary_for_llm(self) -> str:
        """Generate text summary of dataset for LLM prompt."""

        if not self.metadata:
            self.analyze()

        summary = f"""
DATASET SUMMARY
===============

Basic Info:
- Shape: {self.metadata['basic_info']['shape']['rows']} rows × {self.metadata['basic_info']['shape']['columns']} columns
- Total columns: {self.metadata['basic_info']['shape']['columns']}

Numeric Features ({len(self.metadata['numeric_features'])}):
"""

        for col, stats in list(self.metadata["numeric_features"].items())[:10]:
            summary += f"\n- {col}: range [{stats['min']:.2f}, {stats['max']:.2f}], mean={stats['mean']:.2f}"

        summary += f"\n\nCategorical Features ({len(self.metadata['categorical_features'])}):\n"

        for col, stats in list(self.metadata["categorical_features"].items())[:10]:
            summary += f"\n- {col}: {stats['unique_count']} unique values"

        summary += f"\n\nTarget Variable: {self.target_col}"
        summary += f"\n- Type: {self.metadata['target_info']['type']}"

        if self.metadata["target_info"]["type"] == "classification":
            summary += f"\n- Classes: {self.metadata['target_info']['classes']}"
        else:
            summary += f"\n- Range: [{self.metadata['target_info']['min']:.2f}, {self.metadata['target_info']['max']:.2f}]"

        return summary

    def save_metadata(self, output_path: Path) -> None:
        """Save metadata to JSON file."""
        if not self.metadata:
            self.analyze()

        with open(output_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

        logger.info(f"Metadata saved to {output_path}")


def load_dataset(path: Path, target_col: str) -> tuple[pd.DataFrame, DatasetAnalyzer]:
    """Load dataset and return with analyzer."""

    logger.info(f"Loading dataset from {path}")

    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    logger.info(f"Dataset loaded: {df.shape}")

    analyzer = DatasetAnalyzer(df, target_col)
    analyzer.analyze()

    return df, analyzer
