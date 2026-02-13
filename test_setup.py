"""Test pipeline without API calls."""

import pandas as pd
from src.data_pipeline import DataPipeline

print("Testing offline pipeline...")

# Create test data
df = pd.DataFrame({
    'age': range(20, 70, 5),
    'income': [30000 + i*5000 for i in range(10)],
    'experience': range(0, 100, 10),
    'target': [0, 1, 1, 1, 0, 1, 0, 1, 1, 0]
})
df.to_csv('data/raw/offline_test.csv', index=False)

# Test each step
pipeline = DataPipeline('data/raw/offline_test.csv', target_col='target')

# Step 1: Load
df = pipeline.load_data()
assert df.shape[0] == 10, "Data not loaded correctly"
print("✓ Load data")

# Step 2: Analyze
analyzer = pipeline.analyze_dataset()
assert analyzer.metadata is not None, "Analysis failed"
print("✓ Analyze dataset")

# Step 3: Split (without feature generation)
pipeline.enriched_df = pipeline.raw_df.copy()
X_train, X_test, y_train, y_test = pipeline.split_data()
assert len(X_train) + len(X_test) == 10, "Split failed"
print("✓ Split data")

print("\n✅ All offline pipeline tests passed!")