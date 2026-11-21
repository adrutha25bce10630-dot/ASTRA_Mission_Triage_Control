from datetime import datetime, timezone
import json
from pathlib import Path
from find_tca_conjunctions import screen_conjunctions
from flask import Flask, jsonify, request, send_from_directory
import joblib
import numpy as np
from risk_engine import ObjectState, assess_conjunction
from run_triage_pipeline import MODEL_PATH, OUT_JSON_PATH, predict_rtn_bounds

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(BASE_DIR))

_MODEL_BUNDLE = None


def get_model_bundle() -> dict:
  global _MODEL_BUNDLE
  if _MODEL_BUNDLE is None:
    if not MODEL_PATH.exists():
      raise FileNotFoundError(
          f"Missing {MODEL_PATH}. Run train_quantile_model.py first."
      )
    _MODEL_BUNDLE = joblib.load(MODEL_PATH)
  return _MODEL_BUNDLE


@app.route("/")
def serve_dashboard():
  dashboard_file = BASE_DIR / "astra_dashboard.html"
  if dashboard_file.exists():
    return send_from_directory(str(BASE_DIR), "astra_dashboard.html")
  return jsonify({
      "status": "ok",
      "message": "ASTRA API running. Place astra_dashboard.html in this directory to serve the UI.",
  })


@app.route("/api/results", methods=["GET"])
def get_saved_results():
  if not OUT_JSON_PATH.exists():
    return jsonify({"results": [], "note": "Run run_triage_pipeline.py to generate triage_results.json."})
  data = json.loads(OUT_JSON_PATH.read_text(encoding="utf-8"))
  return jsonify({"results": data})


@app.route("/api/triage", methods=["POST"])
def run_live_triage():
  """
    Expects JSON payload:
    {
      "primary": {"name": "...", "line1": "...", "line2": "...", "hard_body_radius_m": 8.0, "rcs_m2": 10.0},
      "secondary": {"name": "...", "line1": "...", "line2": "...", "hard_body_radius_m": 3.0, "rcs_m2": 0.5},
      "horizon_hours": 48.0,
      "distance_filter_km": 25.0,
      "f10_7": 135.0,
      "kp": 2.8
    }
    """
  try:
    payload = request.get_json(force=True) or {}
    p_in = payload["primary"]
    s_in = payload["secondary"]

    horizon_hours = float(payload.get("horizon_hours", 48.0))
    distance_filter_km = float(payload.get("distance_filter_km", 25.0))
    f10_7 = float(payload.get("f10_7", 135.0))
    kp = float(payload.get("kp", 2.8))

    start_dt = datetime.now(timezone.utc)
    candidates = screen_conjunctions(
        name1=p_in["name"],
        line1_a=p_in["line1"],
        line2_a=p_in["line2"],
        name2=s_in["name"],
        line1_b=s_in["line1"],
        line2_b=s_in["line2"],
        start_dt=start_dt,
        horizon_hours=horizon_hours,
        coarse_step_seconds=60,
        distance_filter_km=distance_filter_km,
        f10_7=f10_7,
        kp=kp,
        rcs1_m2=float(p_in.get("rcs_m2", 10.0)),
        rcs2_m2=float(s_in.get("rcs_m2", 0.5)),
    )

    bundle = get_model_bundle()
    assessments = []

    for idx, cand in enumerate(candidates, start=1):
      p_bounds = predict_rtn_bounds(bundle, cand["primary"]["features"])
      s_bounds = predict_rtn_bounds(bundle, cand["secondary"]["features"])

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
          name=cand["primary"]["name"],
          r_km=np.array(cand["primary"]["r_km"], dtype=float),
          v_km_s=np.array(cand["primary"]["v_km_s"], dtype=float),
          sigma_rtn_km=p_sig_med,
          hard_body_radius_m=float(p_in.get("hard_body_radius_m", 8.0)),
      )
      secondary_obj = ObjectState(
          name=cand["secondary"]["name"],
          r_km=np.array(cand["secondary"]["r_km"], dtype=float),
          v_km_s=np.array(cand["secondary"]["v_km_s"], dtype=float),
          sigma_rtn_km=s_sig_med,
          hard_body_radius_m=float(s_in.get("hard_body_radius_m", 3.0)),
      )

      triage = assess_conjunction(
          primary_obj,
          secondary_obj,
          sigma_rtn_low_1=p_sig_low,
          sigma_rtn_high_1=p_sig_high,
          sigma_rtn_low_2=s_sig_low,
          sigma_rtn_high_2=s_sig_high,
      )

      assessments.append({
          "event_id": f"ASTRA-LIVE-{idx:03d}",
          "tca_utc": cand["tca_utc"],
          **triage,
          "primary_rtn_bounds_km": p_bounds,
          "secondary_rtn_bounds_km": s_bounds,
      })

    return jsonify({
        "candidate_count": len(candidates),
        "assessments": assessments,
    })

  except KeyError as exc:
    return jsonify({"error": f"Missing required field: {exc}"}), 400
  except Exception as exc:
    return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
  app.run(host="127.0.0.1", port=5000, debug=True)