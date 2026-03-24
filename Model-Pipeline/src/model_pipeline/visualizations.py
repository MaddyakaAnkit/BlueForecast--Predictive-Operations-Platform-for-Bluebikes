"""
Result visualizations for BlueForecast model pipeline.

Generates plots required by submission guidelines (Section 4):
  1. Feature importance bar chart (SHAP vs XGBoost gain)
  2. Predicted vs Actual scatter plot
  3. Residual distribution histogram
  4. Bias disparity bar chart across slice dimensions
  5. SHAP summary beeswarm plot

All plots saved as PNG to GCS + logged as MLflow artifacts.
"""

import io
import logging
import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from google.cloud import storage

logger = logging.getLogger("model_pipeline.visualizations")
logger.setLevel(logging.INFO)

BUCKET = "bluebikes-demand-predictor-data"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_plot_to_gcs(fig: plt.Figure, run_id: str, filename: str) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    gcs_path = f"processed/models/{run_id}/plots/{filename}"
    blob = storage.Client().bucket(BUCKET).blob(gcs_path)
    blob.upload_from_file(buf, content_type="image/png")
    uri = f"gs://{BUCKET}/{gcs_path}"
    logger.info("Plot saved → %s", uri)
    plt.close(fig)
    return uri


def _log_plot_as_mlflow_artifact(fig: plt.Figure, run_id: str, filename: str) -> None:
    import mlflow.tracking
    client = mlflow.tracking.MlflowClient()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, filename)
        fig.savefig(path, format="png", dpi=150, bbox_inches="tight")
        client.log_artifact(run_id, path, artifact_path="plots")


def _log_uris_to_mlflow(run_id: str, plot_uris: dict[str, str]) -> None:
    import mlflow.tracking
    client = mlflow.tracking.MlflowClient()
    for tag_key, uri in plot_uris.items():
        client.set_tag(run_id, f"plot_{tag_key}", uri)
    logger.info("Plot URIs logged to MLflow run %s", run_id)


# ---------------------------------------------------------------------------
# 1. Feature Importance
# ---------------------------------------------------------------------------

def plot_feature_importance(
    shap_importance: dict[str, float],
    gain_importance: dict[str, float],
    run_id: str,
    top_n: int = 15,
) -> str:
    top_features = list(shap_importance.keys())[:top_n]
    shap_vals = [shap_importance[f] for f in top_features]
    gain_vals = [gain_importance.get(f, 0) for f in top_features]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    y_pos = np.arange(len(top_features))

    ax1.barh(y_pos, shap_vals[::-1], color="#2196F3", alpha=0.85)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(top_features[::-1])
    ax1.set_xlabel("Mean |SHAP value|")
    ax1.set_title("SHAP Feature Importance")

    ax2.barh(y_pos, gain_vals[::-1], color="#FF9800", alpha=0.85)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_features[::-1])
    ax2.set_xlabel("XGBoost Gain")
    ax2.set_title("XGBoost Gain Importance")

    fig.suptitle(f"Feature Importance — Run {run_id[:8]}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    _log_plot_as_mlflow_artifact(fig, run_id, "feature_importance.png")
    return _save_plot_to_gcs(fig, run_id, "feature_importance.png")


# ---------------------------------------------------------------------------
# 2. Predicted vs Actual
# ---------------------------------------------------------------------------

def plot_predicted_vs_actual(
    y_true: np.ndarray, y_pred: np.ndarray, run_id: str, n_sample: int = 50_000,
) -> str:
    from sklearn.metrics import mean_squared_error, r2_score

    rng = np.random.default_rng(42)
    if len(y_true) > n_sample:
        idx = rng.choice(len(y_true), size=n_sample, replace=False)
        y_t, y_p = y_true[idx], y_pred[idx]
    else:
        y_t, y_p = y_true, y_pred

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_t, y_p, alpha=0.08, s=4, color="#1976D2", rasterized=True)

    max_val = max(y_t.max(), y_p.max())
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=1.5, label="Perfect prediction")

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    ax.text(0.05, 0.92, f"RMSE: {rmse:.4f}\nR²: {r2:.4f}",
            transform=ax.transAxes, fontsize=12, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    ax.set_xlabel("Actual Demand", fontsize=12)
    ax.set_ylabel("Predicted Demand", fontsize=12)
    ax.set_title(f"Predicted vs Actual — Run {run_id[:8]}", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    fig.tight_layout()

    _log_plot_as_mlflow_artifact(fig, run_id, "predicted_vs_actual.png")
    return _save_plot_to_gcs(fig, run_id, "predicted_vs_actual.png")


# ---------------------------------------------------------------------------
# 3. Residual Distribution
# ---------------------------------------------------------------------------

def plot_residual_distribution(
    y_true: np.ndarray, y_pred: np.ndarray, run_id: str,
) -> str:
    residuals = y_true - y_pred

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(residuals, bins=100, color="#4CAF50", alpha=0.75, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero residual")

    mean_r, std_r = float(np.mean(residuals)), float(np.std(residuals))
    ax.axvline(mean_r, color="orange", linestyle="-", linewidth=1.2, label=f"Mean: {mean_r:.4f}")

    ax.text(0.95, 0.92,
            f"Mean: {mean_r:.4f}\nStd: {std_r:.4f}\nMedian: {float(np.median(residuals)):.4f}",
            transform=ax.transAxes, fontsize=11, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    ax.set_xlabel("Residual (Actual − Predicted)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Residual Distribution — Run {run_id[:8]}", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left")
    fig.tight_layout()

    _log_plot_as_mlflow_artifact(fig, run_id, "residual_distribution.png")
    return _save_plot_to_gcs(fig, run_id, "residual_distribution.png")


# ---------------------------------------------------------------------------
# 4. Bias Disparity
# ---------------------------------------------------------------------------

def plot_bias_disparity(bias_report: dict, run_id: str) -> str:
    dimensions, ratios = [], []
    for dim_name, dim_data in bias_report.get("dimensions", {}).items():
        if "disparity_ratio" in dim_data and dim_data["disparity_ratio"] is not None:
            dimensions.append(dim_name)
            ratios.append(dim_data["disparity_ratio"])

    if not dimensions:
        logger.warning("No disparity data in bias report — skipping plot.")
        return ""

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#F44336" if r >= 3.0 else "#FF9800" if r >= 2.5 else "#4CAF50" for r in ratios]
    bars = ax.bar(dimensions, ratios, color=colors, alpha=0.85, edgecolor="white", linewidth=1.2)

    ax.axhline(y=3.0, color="red", linestyle="--", linewidth=2, label="Block threshold (3.0×)")

    for bar, ratio in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
                f"{ratio:.2f}×", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xlabel("Slice Dimension", fontsize=12)
    ax.set_ylabel("RMSE Disparity Ratio (max/min)", fontsize=12)
    ax.set_title(f"Bias Detection — Run {run_id[:8]}", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right")
    ax.set_ylim(0, max(ratios) * 1.3)
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()

    _log_plot_as_mlflow_artifact(fig, run_id, "bias_disparity.png")
    return _save_plot_to_gcs(fig, run_id, "bias_disparity.png")


# ---------------------------------------------------------------------------
# 5. SHAP Summary Beeswarm
# ---------------------------------------------------------------------------

def plot_shap_summary(
    forecaster, X_test: pd.DataFrame, feature_cols: list[str],
    run_id: str, n_sample: int = 5_000,
) -> str:
    sample = X_test.sample(n=min(n_sample, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(forecaster._model)
    shap_values = explainer.shap_values(sample.values)

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, sample, feature_names=feature_cols,
                      show=False, max_display=15)
    plt.title(f"SHAP Summary — Run {run_id[:8]}", fontsize=14, fontweight="bold")
    fig = plt.gcf()
    fig.tight_layout()

    _log_plot_as_mlflow_artifact(fig, run_id, "shap_summary.png")
    return _save_plot_to_gcs(fig, run_id, "shap_summary.png")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_all_plots(
    forecaster,
    X_test: pd.DataFrame,
    y_test,
    feature_cols: list[str],
    run_id: str,
    shap_importance: dict[str, float] | None = None,
    gain_importance: dict[str, float] | None = None,
    bias_report: dict | None = None,
) -> dict[str, str]:
    """Generate all submission-required visualizations. Returns {name: gcs_uri}."""
    logger.info("=== Generating Result Visualizations ===")
    plot_uris = {}

    y_true = y_test.values if hasattr(y_test, "values") else y_test
    y_pred = forecaster.predict(X_test.values if hasattr(X_test, "values") else X_test)

    # 1. Feature importance
    if shap_importance and gain_importance:
        logger.info("[1/5] Feature importance bar chart...")
        plot_uris["feature_importance"] = plot_feature_importance(
            shap_importance, gain_importance, run_id)
    else:
        logger.info("[1/5] Skipped feature importance (no pre-computed data).")

    # 2. Predicted vs actual
    logger.info("[2/5] Predicted vs actual scatter...")
    plot_uris["predicted_vs_actual"] = plot_predicted_vs_actual(y_true, y_pred, run_id)

    # 3. Residual distribution
    logger.info("[3/5] Residual distribution histogram...")
    plot_uris["residual_distribution"] = plot_residual_distribution(y_true, y_pred, run_id)

    # 4. Bias disparity
    if bias_report:
        logger.info("[4/5] Bias disparity bar chart...")
        uri = plot_bias_disparity(bias_report, run_id)
        if uri:
            plot_uris["bias_disparity"] = uri
    else:
        logger.info("[4/5] Skipped bias disparity (no report).")

    # 5. SHAP summary
    logger.info("[5/5] SHAP summary beeswarm plot...")
    plot_uris["shap_summary"] = plot_shap_summary(forecaster, X_test, feature_cols, run_id)

    _log_uris_to_mlflow(run_id, plot_uris)
    logger.info("All visualizations generated: %d plots.", len(plot_uris))
    return plot_uris