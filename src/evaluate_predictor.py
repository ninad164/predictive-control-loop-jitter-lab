"""Evaluate a trained deadline miss predictor on unseen runtime telemetry."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


DEFAULT_MODEL_PATH = Path("models/deadline_predictor.joblib")
DEFAULT_OUTPUT_PATH = Path("results/reports/predictions_unseen.csv")
FEATURE_COLUMNS = [
    "loop_execution_time_ms",
    "loop_period_ms",
    "jitter_ms",
    "cpu_percent",
    "memory_percent",
]
TARGET_COLUMN = "deadline_miss"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for predictor evaluation."""

    parser = argparse.ArgumentParser(
        description="Evaluate the trained deadline miss predictor on a runtime log."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the runtime telemetry CSV to evaluate.",
    )
    return parser.parse_args()


def load_evaluation_data(input_path: Path) -> pd.DataFrame:
    """Load and validate a runtime telemetry CSV file."""

    dataframe = pd.read_csv(input_path)
    required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        missing_display = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"CSV file '{input_path}' is missing required columns: {missing_display}"
        )
    return dataframe


def prepare_evaluation_data(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare features and target values for model evaluation."""

    evaluation_dataframe = dataframe.copy()
    evaluation_dataframe["actual_deadline_miss"] = (
        evaluation_dataframe[TARGET_COLUMN]
        .astype(str)
        .str.lower()
        .map({"true": 1, "false": 0})
    )
    evaluation_dataframe = evaluation_dataframe.dropna(
        subset=FEATURE_COLUMNS + ["actual_deadline_miss"]
    )

    if evaluation_dataframe.empty:
        raise ValueError("No valid samples remain after dropping missing values.")

    feature_dataframe = evaluation_dataframe[FEATURE_COLUMNS].copy()
    evaluation_dataframe["actual_deadline_miss"] = evaluation_dataframe[
        "actual_deadline_miss"
    ].astype(int)
    return evaluation_dataframe, feature_dataframe


def save_prediction_report(
    evaluation_dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save prediction results for each evaluated sample."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_dataframe.to_csv(output_path, index=False)


def main() -> None:
    """Load the trained model, evaluate it, and save predictions."""

    args = parse_args()
    model = joblib.load(DEFAULT_MODEL_PATH)

    input_path = Path(args.input)
    raw_dataframe = load_evaluation_data(input_path)
    evaluation_dataframe, feature_dataframe = prepare_evaluation_data(raw_dataframe)

    actual = evaluation_dataframe["actual_deadline_miss"]
    predicted = model.predict(feature_dataframe)

    evaluation_dataframe["predicted_deadline_miss"] = predicted.astype(int)
    evaluation_dataframe["prediction_correct"] = (
        evaluation_dataframe["actual_deadline_miss"]
        == evaluation_dataframe["predicted_deadline_miss"]
    )

    accuracy = accuracy_score(actual, predicted)
    precision = precision_score(actual, predicted, zero_division=0)
    recall = recall_score(actual, predicted, zero_division=0)
    f1 = f1_score(actual, predicted, zero_division=0)
    matrix = confusion_matrix(actual, predicted)
    report = classification_report(actual, predicted, zero_division=0)

    save_prediction_report(evaluation_dataframe, DEFAULT_OUTPUT_PATH)

    print(f"Number of samples: {len(evaluation_dataframe)}")
    print(f"Number of actual deadline misses: {int(actual.sum())}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 score: {f1:.4f}")
    print("Confusion matrix:")
    print(matrix)
    print("Classification report:")
    print(report)


if __name__ == "__main__":
    main()
