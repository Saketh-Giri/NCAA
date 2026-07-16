"""
Final Project model workflow for the college basketball dataset.

Research question:
Given a tournament team's seed and regular-season efficiency metrics, how many
NCAA tournament games will it win?

Run:
    python final_project_models.py

The script saves slide-ready tables and figures in the outputs/ folder.
"""

from __future__ import annotations

import os
from pathlib import Path

# Keep Matplotlib from trying to write a font cache in the user's home folder.
DATA_PATH = Path("cbb.csv")
OUTPUT_DIR = Path("outputs")
OFFICIAL_BRACKET_PATH = Path("bracket_site/official_bracket_seeds.csv")
MPL_CACHE_DIR = OUTPUT_DIR / ".matplotlib_cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR.resolve()))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TEST_SIZE = 0.2


FEATURES = [
    "CONF",
    "W",
    "WIN_RATE",
    "ADJOE",
    "ADJDE",
    "BARTHAG",
    "EFG_O",
    "EFG_D",
    "TOR",
    "TORD",
    "ORB",
    "DRB",
    "FTR",
    "FTRD",
    "2P_O",
    "2P_D",
    "3P_O",
    "3P_D",
    "ADJ_T",
    "WAB",
    "SEED_NUM",
]

SELECTION_FEATURES = [feature for feature in FEATURES if feature != "SEED_NUM"]


def load_and_prepare_data(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the dataset and return both all teams and tournament teams only."""
    data = pd.read_csv(path)
    data["SEED_NUM"] = pd.to_numeric(data["SEED"], errors="coerce")
    data["WIN_RATE"] = data["W"] / data["G"]
    data["MADE_TOURNAMENT"] = data["SEED_NUM"].notna().astype(int)

    tournament = data[data["MADE_TOURNAMENT"] == 1].copy()
    tournament["SEED_NUM"] = tournament["SEED_NUM"].astype(int)

    return data, tournament


def build_preprocessor(features: list[str] | None = None) -> ColumnTransformer:
    """Create preprocessing for numeric features and conference labels."""
    selected_features = FEATURES if features is None else features
    numeric_features = [feature for feature in selected_features if feature != "CONF"]
    categorical_features = [feature for feature in selected_features if feature == "CONF"]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = [("num", numeric_pipeline, numeric_features)]
    if categorical_features:
        transformers.append(("cat", categorical_pipeline, categorical_features))

    return ColumnTransformer(transformers=transformers)


def build_models() -> dict[str, object]:
    """Define the two models used in the final presentation."""
    return {
        "Ridge Regression": Ridge(alpha=2.0),
        "Gradient Boosting Regression": GradientBoostingRegressor(
            max_depth=2,
            learning_rate=0.03,
            n_estimators=400,
            subsample=0.85,
        ),
    }


def build_selection_models() -> dict[str, object]:
    """Define models that predict whether a team makes March Madness."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Gradient Boosting Classification": GradientBoostingClassifier(
            max_depth=2,
            learning_rate=0.03,
            n_estimators=400,
            subsample=0.85,
        ),
    }


def random_split(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Create a fresh randomized train/test split each time the script runs."""
    train_data, test_data = train_test_split(
        data,
        test_size=TEST_SIZE,
        shuffle=True,
        stratify=data["Tour_Wins"],
    )

    X_train = train_data[FEATURES]
    X_test = test_data[FEATURES]
    y_train = train_data["Tour_Wins"]
    y_test = test_data["Tour_Wins"]

    return X_train, X_test, y_train, y_test, test_data


def selection_split(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Create a randomized split for predicting NCAA tournament selection."""
    train_data, test_data = train_test_split(
        data,
        test_size=TEST_SIZE,
        shuffle=True,
        stratify=data["MADE_TOURNAMENT"],
    )

    X_train = train_data[SELECTION_FEATURES]
    X_test = test_data[SELECTION_FEATURES]
    y_train = train_data["MADE_TOURNAMENT"]
    y_test = test_data["MADE_TOURNAMENT"]

    return X_train, X_test, y_train, y_test, test_data


def evaluate_models(
    models: dict[str, object],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Pipeline], dict[str, np.ndarray]]:
    """Fit all models and compute test-year regression metrics."""
    rows = []
    fitted_models: dict[str, Pipeline] = {}
    predictions: dict[str, np.ndarray] = {}

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                ("model", model),
            ]
        )
        pipeline.fit(X_train, y_train)
        y_pred = np.clip(pipeline.predict(X_test), 0, 6)

        fitted_models[model_name] = pipeline
        predictions[model_name] = y_pred
        rows.append(
            {
                "model": model_name,
                "mae": mean_absolute_error(y_test, y_pred),
                "rmse": mean_squared_error(y_test, y_pred) ** 0.5,
                "r2": r2_score(y_test, y_pred),
            }
        )

    metrics = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
    return metrics, fitted_models, predictions


def evaluate_selection_models(
    models: dict[str, object],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Pipeline], dict[str, np.ndarray]]:
    """Fit selection classifiers and compute held-out classification metrics."""
    rows = []
    fitted_models: dict[str, Pipeline] = {}
    probabilities: dict[str, np.ndarray] = {}

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor(SELECTION_FEATURES)),
                ("model", model),
            ]
        )
        pipeline.fit(X_train, y_train)
        y_probability = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_probability >= 0.5).astype(int)

        fitted_models[model_name] = pipeline
        probabilities[model_name] = y_probability
        rows.append(
            {
                "model": model_name,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_probability),
            }
        )

    metrics = (
        pd.DataFrame(rows)
        .sort_values(["f1", "roc_auc"], ascending=False)
        .reset_index(drop=True)
    )
    return metrics, fitted_models, probabilities


def save_descriptive_outputs(all_teams: pd.DataFrame, tournament: pd.DataFrame) -> None:
    """Save concise descriptive tables for slide creation."""
    selected_columns = [
        "Tour_Wins",
        "G",
        "W",
        "WIN_RATE",
        "ADJOE",
        "ADJDE",
        "BARTHAG",
        "EFG_O",
        "EFG_D",
        "ADJ_T",
        "WAB",
        "SEED_NUM",
    ]
    tournament[selected_columns].describe().round(3).to_csv(
        OUTPUT_DIR / "descriptive_statistics_tournament_teams.csv"
    )

    seed_summary = (
        tournament.groupby("SEED_NUM")["Tour_Wins"]
        .agg(["count", "mean", "median", "max"])
        .round(3)
    )
    seed_summary.to_csv(OUTPUT_DIR / "tour_wins_by_seed.csv")

    dataset_summary = pd.DataFrame(
        [
            {"item": "total_team_seasons", "value": len(all_teams)},
            {"item": "tournament_team_seasons", "value": len(tournament)},
            {"item": "first_year", "value": int(all_teams["YEAR"].min())},
            {"item": "last_year", "value": int(all_teams["YEAR"].max())},
            {"item": "unique_conferences", "value": all_teams["CONF"].nunique()},
        ]
    )
    dataset_summary.to_csv(OUTPUT_DIR / "dataset_summary.csv", index=False)


def save_predictions(
    test_data: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    best_model_name: str,
) -> pd.DataFrame:
    """Save randomized test-set forecasts from the best model."""
    prediction_data = test_data[
        ["TEAM", "CONF", "YEAR", "SEED_NUM", "Tour_Wins", "POSTSEASON"]
    ].copy()
    prediction_data["Predicted_Tour_Wins"] = predictions[best_model_name]
    prediction_data["Absolute_Error"] = (
        prediction_data["Predicted_Tour_Wins"] - prediction_data["Tour_Wins"]
    ).abs()
    prediction_data = prediction_data.sort_values("Predicted_Tour_Wins", ascending=False)
    prediction_data.round(3).to_csv(
        OUTPUT_DIR / "random_test_predictions.csv", index=False
    )
    return prediction_data


def save_yearly_tournament_predictions(
    tournament: pd.DataFrame,
    model_template: object,
    model_name: str,
) -> pd.DataFrame:
    """Save tournament-win predictions for each year from models trained on other years."""
    rows = []

    for year in sorted(tournament["YEAR"].unique()):
        train_data = tournament[tournament["YEAR"] != year]
        test_data = tournament[tournament["YEAR"] == year]
        pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                ("model", clone(model_template)),
            ]
        )
        pipeline.fit(train_data[FEATURES], train_data["Tour_Wins"])
        y_pred = np.clip(pipeline.predict(test_data[FEATURES]), 0, 6)

        prediction_data = test_data[
            ["TEAM", "CONF", "YEAR", "SEED_NUM", "Tour_Wins", "POSTSEASON"]
        ].copy()
        prediction_data["Predicted_Tour_Wins"] = y_pred
        prediction_data["Absolute_Error"] = (
            prediction_data["Predicted_Tour_Wins"] - prediction_data["Tour_Wins"]
        ).abs()
        prediction_data["Prediction_Model"] = model_name
        rows.append(prediction_data)

    yearly_predictions = pd.concat(rows, ignore_index=True).sort_values(
        ["YEAR", "SEED_NUM", "Predicted_Tour_Wins"],
        ascending=[False, True, False],
    )
    yearly_predictions.round(4).to_csv(
        OUTPUT_DIR / "yearly_tournament_predictions.csv", index=False
    )
    return yearly_predictions


def save_official_bracket_predictions(
    yearly_predictions: pd.DataFrame,
    official_bracket_path: Path = OFFICIAL_BRACKET_PATH,
) -> pd.DataFrame:
    """Attach year-held-out predictions to official NCAA bracket seed slots."""
    official_bracket = pd.read_csv(official_bracket_path)
    bracket_predictions = official_bracket.merge(
        yearly_predictions,
        on=["YEAR", "TEAM"],
        how="left",
        suffixes=("", "_MODEL_DATA"),
    )
    missing_predictions = bracket_predictions[
        bracket_predictions["Predicted_Tour_Wins"].isna()
    ]
    if not missing_predictions.empty:
        missing_text = missing_predictions[["YEAR", "BRACKET_SEED", "TEAM"]].to_string(
            index=False
        )
        raise ValueError(f"Missing model predictions for official bracket teams:\n{missing_text}")

    bracket_predictions = bracket_predictions.drop(
        columns=["SEED_NUM_MODEL_DATA"],
        errors="ignore",
    )
    bracket_predictions = bracket_predictions.sort_values(
        ["YEAR", "REGION_CODE", "SEED_NUM", "BRACKET_SEED"]
    )
    bracket_predictions.round(4).to_csv(
        OUTPUT_DIR / "official_bracket_predictions.csv", index=False
    )
    return bracket_predictions


def save_selection_predictions(
    test_data: pd.DataFrame,
    probabilities: dict[str, np.ndarray],
    best_model_name: str,
) -> pd.DataFrame:
    """Save held-out predictions for making or missing March Madness."""
    prediction_data = test_data[
        [
            "TEAM",
            "CONF",
            "YEAR",
            "G",
            "W",
            "WIN_RATE",
            "WAB",
            "MADE_TOURNAMENT",
            "SEED",
            "POSTSEASON",
        ]
    ].copy()
    prediction_data["Predicted_Tournament_Probability"] = probabilities[best_model_name]
    prediction_data["Predicted_Made_Tournament"] = (
        prediction_data["Predicted_Tournament_Probability"] >= 0.5
    ).astype(int)
    prediction_data["Prediction"] = np.where(
        prediction_data["Predicted_Made_Tournament"] == 1,
        "Make March Madness",
        "Miss March Madness",
    )
    prediction_data["Actual"] = np.where(
        prediction_data["MADE_TOURNAMENT"] == 1,
        "Made March Madness",
        "Missed March Madness",
    )
    prediction_data["Correct"] = (
        prediction_data["Predicted_Made_Tournament"]
        == prediction_data["MADE_TOURNAMENT"]
    )
    prediction_data = prediction_data.sort_values(
        ["YEAR", "Predicted_Tournament_Probability"],
        ascending=[False, False],
    )
    prediction_data.round(4).to_csv(
        OUTPUT_DIR / "march_madness_selection_predictions.csv", index=False
    )
    return prediction_data


def save_all_selection_predictions(data: pd.DataFrame, model: Pipeline) -> pd.DataFrame:
    """Save tournament selection probabilities for every team-season."""
    prediction_data = data[
        [
            "TEAM",
            "CONF",
            "YEAR",
            "G",
            "W",
            "WIN_RATE",
            "WAB",
            "MADE_TOURNAMENT",
            "SEED",
            "POSTSEASON",
        ]
    ].copy()
    prediction_data["Predicted_Tournament_Probability"] = model.predict_proba(
        data[SELECTION_FEATURES]
    )[:, 1]
    prediction_data["Predicted_Made_Tournament"] = (
        prediction_data["Predicted_Tournament_Probability"] >= 0.5
    ).astype(int)
    prediction_data["Prediction"] = np.where(
        prediction_data["Predicted_Made_Tournament"] == 1,
        "Make March Madness",
        "Miss March Madness",
    )
    prediction_data["Actual"] = np.where(
        prediction_data["MADE_TOURNAMENT"] == 1,
        "Made March Madness",
        "Missed March Madness",
    )
    prediction_data["Correct"] = (
        prediction_data["Predicted_Made_Tournament"]
        == prediction_data["MADE_TOURNAMENT"]
    )
    prediction_data = prediction_data.sort_values(
        ["YEAR", "Predicted_Tournament_Probability"],
        ascending=[False, False],
    )
    prediction_data.round(4).to_csv(
        OUTPUT_DIR / "march_madness_selection_all_teams.csv", index=False
    )
    return prediction_data


def save_ridge_coefficients(model: Pipeline) -> pd.DataFrame:
    """Save Ridge coefficients for the interpretable baseline model."""
    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    coefficients = model.named_steps["model"].coef_
    coefficient_table = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "absolute_coefficient": np.abs(coefficients),
        }
    ).sort_values("absolute_coefficient", ascending=False)
    coefficient_table.round(4).to_csv(OUTPUT_DIR / "ridge_coefficients.csv", index=False)
    return coefficient_table


def save_permutation_importance(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    best_model_name: str,
) -> pd.DataFrame:
    """Save feature importance as test MAE increase after shuffling each feature."""
    importance = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=30,
        scoring="neg_mean_absolute_error",
    )
    importance_table = pd.DataFrame(
        {
            "feature": FEATURES,
            "mae_increase_when_shuffled": importance.importances_mean,
            "std": importance.importances_std,
        }
    ).sort_values("mae_increase_when_shuffled", ascending=False)

    safe_model_name = best_model_name.lower().replace(" ", "_")
    importance_table.round(4).to_csv(
        OUTPUT_DIR / "feature_importance_best_model.csv", index=False
    )
    importance_table.round(4).to_csv(
        OUTPUT_DIR / f"feature_importance_{safe_model_name}.csv", index=False
    )
    return importance_table


def save_figures(
    tournament: pd.DataFrame,
    metrics: pd.DataFrame,
    predictions_table: pd.DataFrame,
    importance_table: pd.DataFrame,
) -> None:
    """Create slide-ready charts for the final presentation."""
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))
    tour_counts = tournament["Tour_Wins"].value_counts().sort_index()
    ax = sns.barplot(x=tour_counts.index, y=tour_counts.values, color="#4C78A8")
    ax.set_title("Tournament Wins Are Highly Concentrated at Zero and One")
    ax.set_xlabel("Tournament wins")
    ax.set_ylabel("Tournament-team seasons")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tour_wins_distribution.png", dpi=200)
    plt.close()

    plt.figure(figsize=(9, 5))
    seed_means = tournament.groupby("SEED_NUM")["Tour_Wins"].mean().reset_index()
    ax = sns.barplot(data=seed_means, x="SEED_NUM", y="Tour_Wins", color="#72B7B2")
    ax.set_title("Higher Seeds Tend to Win More Tournament Games")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Average tournament wins")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "seed_vs_average_tour_wins.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    metrics_long = metrics.melt(id_vars="model", value_vars=["mae", "rmse"])
    ax = sns.barplot(data=metrics_long, x="model", y="value", hue="variable")
    ax.set_title("Model Error on Randomized Test Set")
    ax.set_xlabel("")
    ax.set_ylabel("Error in tournament wins")
    ax.tick_params(axis="x", rotation=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_performance.png", dpi=200)
    plt.close()

    top_features = importance_table.head(10).copy()
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        data=top_features,
        y="feature",
        x="mae_increase_when_shuffled",
        color="#F58518",
    )
    ax.set_title("Most Important Features for the Best Model")
    ax.set_xlabel("Increase in MAE when feature is shuffled")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=200)
    plt.close()

    top_predictions = predictions_table.head(12).copy()
    top_predictions = top_predictions.sort_values("Predicted_Tour_Wins")
    y_positions = np.arange(len(top_predictions))
    plt.figure(figsize=(9, 6))
    plt.barh(
        y_positions - 0.18,
        top_predictions["Predicted_Tour_Wins"],
        height=0.35,
        label="Predicted",
        color="#4C78A8",
    )
    plt.barh(
        y_positions + 0.18,
        top_predictions["Tour_Wins"],
        height=0.35,
        label="Actual",
        color="#54A24B",
    )
    plt.yticks(y_positions, top_predictions["TEAM"])
    plt.xlabel("Tournament wins")
    plt.title("Top Forecasted Teams vs. Actual Results")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "top_predictions_vs_actual.png", dpi=200)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    for stale_path in [
        OUTPUT_DIR / "2019_tournament_predictions.csv",
        *OUTPUT_DIR.glob("feature_importance_*_regression.csv"),
    ]:
        stale_path.unlink(missing_ok=True)

    all_teams, tournament = load_and_prepare_data()
    X_train, X_test, y_train, y_test, test_data = random_split(tournament)
    (
        selection_X_train,
        selection_X_test,
        selection_y_train,
        selection_y_test,
        selection_test_data,
    ) = selection_split(all_teams)

    metrics, fitted_models, predictions = evaluate_models(
        build_models(), X_train, X_test, y_train, y_test
    )
    best_model_name = str(metrics.iloc[0]["model"])
    selection_metrics, selection_models, selection_probabilities = evaluate_selection_models(
        build_selection_models(),
        selection_X_train,
        selection_X_test,
        selection_y_train,
        selection_y_test,
    )
    selection_best_model_name = str(selection_metrics.iloc[0]["model"])
    yearly_predictions_table = save_yearly_tournament_predictions(
        tournament,
        build_models()[best_model_name],
        best_model_name,
    )
    official_bracket_predictions = save_official_bracket_predictions(
        yearly_predictions_table
    )

    save_descriptive_outputs(all_teams, tournament)
    metrics.round(4).to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
    selection_metrics.round(4).to_csv(
        OUTPUT_DIR / "selection_model_metrics.csv", index=False
    )
    predictions_table = save_predictions(test_data, predictions, best_model_name)
    selection_predictions_table = save_selection_predictions(
        selection_test_data,
        selection_probabilities,
        selection_best_model_name,
    )
    save_all_selection_predictions(
        all_teams,
        selection_models[selection_best_model_name],
    )
    save_ridge_coefficients(fitted_models["Ridge Regression"])
    importance_table = save_permutation_importance(
        fitted_models[best_model_name], X_test, y_test, best_model_name
    )
    save_figures(tournament, metrics, predictions_table, importance_table)

    print("Final Project model workflow complete.")
    print(f"Dataset: {len(all_teams)} team-seasons from {all_teams['YEAR'].min()}-{all_teams['YEAR'].max()}")
    print(f"Tournament modeling sample: {len(tournament)} team-seasons")
    print(f"Random train/test split: {len(X_train)} train rows, {len(X_test)} test rows")
    print()
    print("Model performance on randomized test set:")
    print(metrics.round(3).to_string(index=False))
    print()
    print(f"Best model: {best_model_name}")
    print(
        f"Official bracket prediction rows: {len(official_bracket_predictions)} "
        f"across {official_bracket_predictions['YEAR'].nunique()} years"
    )
    print()
    print("Top forecasts:")
    print(
        predictions_table[
            ["TEAM", "CONF", "SEED_NUM", "Tour_Wins", "POSTSEASON", "Predicted_Tour_Wins", "Absolute_Error"]
        ]
        .head(10)
        .round(3)
        .to_string(index=False)
    )
    print()
    print("March Madness selection model performance:")
    print(selection_metrics.round(3).to_string(index=False))
    print()
    print(f"Best selection model: {selection_best_model_name}")
    print("Most likely held-out teams to make March Madness:")
    print(
        selection_predictions_table[
            [
                "TEAM",
                "CONF",
                "YEAR",
                "W",
                "WAB",
                "Predicted_Tournament_Probability",
                "Prediction",
                "Actual",
                "Correct",
            ]
        ]
        .head(12)
        .round(3)
        .to_string(index=False)
    )
    print()
    print(f"Saved outputs to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
