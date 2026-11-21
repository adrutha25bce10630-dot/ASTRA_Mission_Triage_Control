from datetime import datetime, timezone
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from risk_engine import ObjectState, assess_conjunction

MODEL_PATH = Path("astra_quantile_models.joblib")
OUT_JSON_PATH = Path("triage_results.json")


def predict_rtn_bounds(bundle: dict, features_dict: dict) -> dict:
  """
    Predict [q_low, q_med, q_high] error magnitudes (km) for R, T, and N
    from a single object's feature dictionary.
    """
  feature_cols = bundle["features"]
  q_low, q_med, q_high = bundle["quantiles"]
  models = bundle["models"]

  row_df = pd.DataFrame(
      [{col: features_dict[col] for col in feature_cols}], columns=feature_cols
  )

  rtn_bounds = {}
  for axis in ("R", "T", "N"):
    p_low = float(np.maximum(0.0, models[axis][q_low].predict(row_df)[0]))
    p_med = float(np.maximum(0.0, models[axis][q_med].predict(row_df)[0]))
    p_high = float(np.maximum(0.0, models[axis][q_high].predict(row_df)[0]))

    # Enforce non-crossing order
    p_low = min(p_low, p_med)
    p_high = max(p_high, p_med)

    rtn_bounds[axis] = {
        "low": round(p_low, 5),
        "median": round(p_med, 5),
        "high": round(p_high, 5),
    }

  return rtn_bounds


def run_sample_conjunctions(bundle: dict) -> list[dict]:
  """
    Run conjunction triage across sample LEO encounters.
    Replace or expand these records with real SGP4 TCA states + CelesTrak/NOAA inputs.
    """
  scenarios = [
      {
          "event_id": "ASTRA-2026-001",
          "tca_utc": "2026-11-23T08:15:00Z",
          "primary": {
              "name": "SENTINEL-2A (NORAD 40697)",
              "r_km": np.array([7156.20, 0.0, 0.0]),
              "v_km_s": np.array([0.0, 7.46, 0.35]),
              "hard_body_radius_m": 6.5,
              "features": {
                  "delta_t_hours": 18.0,
                  "altitude_km": 786.0,
                  "inclination_deg": 98.5,
                  "eccentricity": 0.00012,
                  "bstar": 1.2e-5,
                  "rcs_m2": 12.0,
                  "f10_7": 125.0,
                  "kp": 2.1,
              },
          },
          "secondary": {
              "name": "COSMOS-2251 DEB (NORAD 34427)",
              "r_km": np.array([7156.95, 0.04, -0.02]),
              "v_km_s": np.array([0.0, -7.46, -0.35]),
              "hard_body_radius_m": 2.5,
              "features": {
                  "delta_t_hours": 18.0,
                  "altitude_km": 786.0,
                  "inclination_deg": 74.0,
                  "eccentricity": 0.0032,
                  "bstar": 4.5e-5,
                  "rcs_m2": 0.25,
                  "f10_7": 125.0,
                  "kp": 2.1,
              },
          },
      },
      {
          "event_id": "ASTRA-2026-002",
          "tca_utc": "2026-11-23T14:40:00Z",
          "primary": {
              "name": "ISS-ZARYA (NORAD 25544)",
              "r_km": np.array([6792.00, 0.0, 0.0]),
              "v_km_s": np.array([0.0, 7.66, 0.0]),
              "hard_body_radius_m": 15.0,
              "features": {
                  "delta_t_hours": 54.0,
                  "altitude_km": 418.0,
                  "inclination_deg": 51.6,
                  "eccentricity": 0.0005,
                  "bstar": 2.4e-4,
                  "rcs_m2": 420.0,
                  "f10_7": 195.0,
                  "kp": 5.4,
              },
          },
          "secondary": {
              "name": "FENGYUN-1C DEB (NORAD 31980)",
              "r_km": np.array([6792.18, 0.01, 0.01]),
              "v_km_s": np.array([0.0, -7.66, 0.12]),
              "hard_body_radius_m": 3.0,
              "features": {
                  "delta_t_hours": 54.0,
                  "altitude_km": 418.0,
                  "inclination_deg": 98.8,
                  "eccentricity": 0.0041,
                  "bstar": 3.1e-4,
                  "rcs_m2": 0.40,
                  "f10_7": 195.0,
                  "kp": 5.4,
              },
          },
      },
  ]

  outputs = []
  for sc in scenarios:
    p_info = sc["primary"]
    s_info = sc["secondary"]

    p_bounds = predict_rtn_bounds(bundle, p_info["features"])
    s_bounds = predict_rtn_bounds(bundle, s_info["features"])

    # Use median predicted RTN magnitudes as 1-sigma scale proxy, and low/high for bounds
    p_sig_med = np.array([
        max(p_bounds["R"]["median"], 1e-4),
        max(p_bounds["T"]["median"], 1e-4),
        max(p_bounds["N"]["median"], 1e-4),
    ])
    p_sig_low = np.array([
        max(p_bounds["R"]["low"], 1e-4),
        max(p_bounds["T"]["low"], 1e-4),
        max(p_bounds["N"]["low"], 1e-4),
    ])
    p_sig_high = np.array([
        max(p_bounds["R"]["high"], 1e-4),
        max(p_bounds["T"]["high"], 1e-4),
        max(p_bounds["N"]["high"], 1e-4),
    ])

    s_sig_med = np.array([
        max(s_bounds["R"]["median"], 1e-4),
        max(s_bounds["T"]["median"], 1e-4),
        max(s_bounds["N"]["median"], 1e-4),
    ])
    s_sig_low = np.array([
        max(s_bounds["R"]["low"], 1e-4),
        max(s_bounds["T"]["low"], 1e-4),
        max(s_bounds["N"]["low"], 1e-4),
    ])
    s_sig_high = np.array([
        max(s_bounds["R"]["high"], 1e-4),
        max(s_bounds["T"]["high"], 1e-4),
        max(s_bounds["N"]["high"], 1e-4),
    ])

    primary_obj = ObjectState(
        name=p_info["name"],
        r_km=p_info["r_km"],
        v_km_s=p_info["v_km_s"],
        sigma_rtn_km=p_sig_med,
        hard_body_radius_m=p_info["hard_body_radius_m"],
    )
    secondary_obj = ObjectState(
        name=s_info["name"],
        r_km=s_info["r_km"],
        v_km_s=s_info["v_km_s"],
        sigma_rtn_km=s_sig_med,
        hard_body_radius_m=s_info["hard_body_radius_m"],
    )

    triage = assess_conjunction(
        primary_obj,
        secondary_obj,
        sigma_rtn_low_1=p_sig_low,
        sigma_rtn_high_1=p_sig_high,
        sigma_rtn_low_2=s_sig_low,
        sigma_rtn_high_2=s_sig_high,
    )

    record = {
        "event_id": sc["event_id"],
        "tca_utc": sc["tca_utc"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        **triage,
        "primary_rtn_bounds_km": p_bounds,
        "secondary_rtn_bounds_km": s_bounds,
    }
    outputs.append(record)

  return outputs


if __name__ == "__main__":
  if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Missing {MODEL_PATH}. Run train_quantile_model.py first."
    )

  model_bundle = joblib.load(MODEL_PATH)
  results = run_sample_conjunctions(model_bundle)

  OUT_JSON_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
  print(f"Wrote {len(results)} conjunction assessments to {OUT_JSON_PATH.resolve()}\n")

  for r in results:
    print(
        f"[{r['event_id']}] {r['primary']} vs {r['secondary']} | "
        f"Miss={r['miss_distance_km']} km | Pc={r['pc']:.3e} | Action={r['action']}"
    )