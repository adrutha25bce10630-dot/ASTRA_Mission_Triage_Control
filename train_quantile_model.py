from pathlib import Path
import joblib
from make_dummy_data import generate_dummy_rtn_dataset
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_pinball_loss
from sklearn.model_selection import train_test_split

FEATURE_COLS = [
    "delta_t_hours",
    "altitude_km",
    "inclination_deg",
    "eccentricity",
    "bstar",
    "rcs_m2",
    "f10_7",
    "kp",
]

TARGETS = {
    "R": "abs_err_r_km",
    "T": "abs_err_t_km",
    "N": "abs_err_n_km",
}

# Lower, median, and upper quantiles for uncertainty bounds
QUANTILES = [0.10, 0.50, 0.90]


def load_or_create_data(csv_path: Path = Path("dummy_rtn_errors.csv")) -> pd.DataFrame:
  if csv_path.exists():
    return pd.read_csv(csv_path)
  return generate_dummy_rtn_dataset()


def train_all_quantile_models(df: pd.DataFrame) -> dict:
  X = df[FEATURE_COLS].copy()
  models = {}

  X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

  print("=== Quantile Error Model Evaluation (Test Set) ===")
  for axis, target_col in TARGETS.items():
    y = df[target_col].copy()
    y_train = y.loc[X_train.index]
    y_test = y.loc[X_test.index]

    models[axis] = {}
    preds = {}

    for q in QUANTILES:
      reg = HistGradientBoostingRegressor(
          loss="quantile",
          quantile=q,
          max_iter=150,
          learning_rate=0.06,
          max_depth=6,
          min_samples_leaf=25,
          random_state=42,
      )
      reg.fit(X_train, y_train)
      y_pred = np.maximum(0.0, reg.predict(X_test))
      models[axis][q] = reg
      preds[q] = y_pred

    # Ensure non-crossing quantiles across [q_low, q_med, q_high]
    q_low, q_med, q_high = QUANTILES
    low_arr = np.minimum(preds[q_low], preds[q_med])
    high_arr = np.maximum(preds[q_high], preds[q_med])

    empirical_coverage = float(
        np.mean((y_test.to_numpy() >= low_arr) & (y_test.to_numpy() <= high_arr))
    )
    nominal_coverage = float(q_high - q_low)
    mae_median = float(mean_absolute_error(y_test, preds[q_med]))
    pinball_high = float(mean_pinball_loss(y_test, high_arr, alpha=q_high))

    print(
        f"Axis {axis} ({target_col}): "
        f"Median MAE={mae_median:.4f} km | "
        f"Pinball(q={q_high})={pinball_high:.4f} | "
        f"Interval [{q_low:.2f}, {q_high:.2f}] Coverage={empirical_coverage:.1%} "
        f"(Target ~{nominal_coverage:.0%})"
    )

  return models


if __name__ == "__main__":
  df = load_or_create_data()
  trained_models = train_all_quantile_models(df)

  bundle = {
      "features": FEATURE_COLS,
      "quantiles": QUANTILES,
      "models": trained_models,
  }
  out_model = Path("astra_quantile_models.joblib")
  joblib.dump(bundle, out_model)
  print(f"\nSaved trained quantile model bundle to {out_model.resolve()}")