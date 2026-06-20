"""Train a deadline miss predictor from runtime telemetry CSV logs."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


DEFAULT_INPUT_DIR = Path("data/runtime_logs")
DEFAULT_MODEL_PATH = Path("models/deadline_predictor.joblib")
DEFAULT_CONFUSION_MATRIX_PATH = Path(
    "results/plots/deadline_predictor_confusion_matrix.png"
)
DEFAULT_FEATURE_IMPORTANCE_PATH = Path(
    "results/plots/feature_importance_random_forest.png"
)
FEATURE_COLUMNS = [
    "loop_execution_time_ms",
    "loop_period_ms",
    "jitter_ms",
    "cpu_percent",
    "memory_percent",
]
TARGET_COLUMN = "deadline_miss"
TIMESTAMP_COLUMN = "timestamp"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for model training."""

    parser = argparse.ArgumentParser(
        description="Train a deadline miss predictor from runtime telemetry logs."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing runtime log CSV files. Default: data/runtime_logs",
    )
    return parser.parse_args()


def load_runtime_logs(input_dir: Path) -> pd.DataFrame:
    """Load and combine all runtime log CSV files from a directory."""

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in '{input_dir}'.")

    dataframes: list[pd.DataFrame] = []
    required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN, TIMESTAMP_COLUMN])

    for csv_file in csv_files:
        dataframe = pd.read_csv(csv_file)
        missing_columns = required_columns.difference(dataframe.columns)
        if missing_columns:
            missing_display = ", ".join(sorted(missing_columns))
            raise ValueError(
                f"CSV file '{csv_file}' is missing required columns: "
                f"{missing_display}"
            )
        dataframe = dataframe.copy()
        dataframe[TIMESTAMP_COLUMN] = pd.to_datetime(dataframe[TIMESTAMP_COLUMN])
        dataframe = dataframe.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
        dataframes.append(dataframe)

    combined_dataframe = pd.concat(dataframes, ignore_index=True)
    return combined_dataframe


def prepare_training_data(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create one-step-ahead samples and drop incomplete rows.

    The model uses telemetry features at timestep ``t`` to predict
    ``deadline_miss`` at timestep ``t + 1``.
    """

    prepared_dataframe = dataframe[
        [TIMESTAMP_COLUMN] + FEATURE_COLUMNS + [TARGET_COLUMN]
    ].copy()
    prepared_dataframe[TARGET_COLUMN] = (
        prepared_dataframe[TARGET_COLUMN].astype(str).str.lower().map(
            {"true": 1, "false": 0}
        )
    )
    prepared_dataframe = prepared_dataframe.sort_values(TIMESTAMP_COLUMN).reset_index(
        drop=True
    )
    prepared_dataframe["next_deadline_miss"] = prepared_dataframe[TARGET_COLUMN].shift(
        -1
    )
    prepared_dataframe = prepared_dataframe.iloc[:-1].copy()
    prepared_dataframe = prepared_dataframe.dropna()

    if prepared_dataframe.empty:
        raise ValueError("No valid samples remain after dropping missing values.")
    if prepared_dataframe["next_deadline_miss"].nunique() < 2:
        raise ValueError(
            "Training data must contain both deadline miss and non-miss samples."
        )

    features = prepared_dataframe[FEATURE_COLUMNS]
    target = prepared_dataframe["next_deadline_miss"].astype(int)
    return features, target


def train_model(features: pd.DataFrame, target: pd.Series) -> tuple[
    RandomForestClassifier,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """Split the dataset and train a Random Forest classifier."""

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=42,
        stratify=target,
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(x_train, y_train)
    return model, x_train, x_test, y_train, y_test


def save_confusion_matrix_plot(
    y_true: pd.Series,
    y_pred: pd.Series,
    output_path: Path,
) -> None:
    """Save a confusion matrix plot for the trained model."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 6))
    matrix = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=[0, 1])
    display.plot(ax=axis, cmap="Blues", colorbar=False)
    axis.set_title("Deadline Miss Predictor Confusion Matrix")
    axis.grid(False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_feature_importance_plot(
    model: RandomForestClassifier,
    output_path: Path,
) -> pd.Series:
    """Save a feature importance bar chart and return the ranked importances."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
    importances = importances.sort_values(ascending=False)

    figure, axis = plt.subplots(figsize=(9, 6))
    bars = axis.bar(importances.index, importances.values, color="steelblue")
    axis.bar_label(bars, fmt="%.3f", padding=3)
    axis.set_title("Random Forest Feature Importance")
    axis.set_xlabel("Feature")
    axis.set_ylabel("Importance")
    axis.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.setp(axis.get_xticklabels(), rotation=20, ha="right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return importances


def main() -> None:
    """Train the deadline miss classifier and save model artifacts."""

    args = parse_args()
    input_dir = Path(args.input_dir)

    combined_dataframe = load_runtime_logs(input_dir)
    features, target = prepare_training_data(combined_dataframe)
    model, _x_train, x_test, _y_train, y_test = train_model(features, target)

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    DEFAULT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, DEFAULT_MODEL_PATH)
    save_confusion_matrix_plot(y_test, y_pred, DEFAULT_CONFUSION_MATRIX_PATH)
    feature_importances = save_feature_importance_plot(
        model, DEFAULT_FEATURE_IMPORTANCE_PATH
    )

    print(f"Number of samples: {len(features)}")
    print(f"Number of deadline misses: {int(target.sum())}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 score: {f1:.4f}")
    print("Confusion matrix:")
    print(matrix)
    print("Classification report:")
    print(report)
    print("Feature Importance:")
    for rank, (feature_name, importance) in enumerate(
        feature_importances.items(),
        start=1,
    ):
        print(f"{rank}. {feature_name} : {importance:.2f}")


if __name__ == "__main__":
    main()
