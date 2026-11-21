from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sgp4.api import Satrec, jday

EARTH_RADIUS_KM = 6378.137
MU_EARTH_KM3_S2 = 398600.4418


def parse_tle_epoch_utc(line1: str) -> datetime:
  """
    Parse TLE Line 1 epoch (columns 19-32: YYDDD.DDDDDDDD) into a timezone-aware UTC datetime.
    """
  epoch_str = line1[18:32].strip()
  year_2d = int(epoch_str[:2])
  day_of_year = float(epoch_str[2:])
  year = 2000 + year_2d if year_2d < 57 else 1900 + year_2d
  jan1 = datetime(year, 1, 1, tzinfo=timezone.utc)
  return jan1 + timedelta(days=day_of_year - 1.0)


def dt_to_jd_fr(dt_utc: datetime) -> tuple[float, float]:
  """Convert UTC datetime to (jd, fr) for sgp4."""
  dt_utc = dt_utc.astimezone(timezone.utc)
  return jday(
      dt_utc.year,
      dt_utc.month,
      dt_utc.day,
      dt_utc.hour,
      dt_utc.minute,
      dt_utc.second + dt_utc.microsecond * 1e-6,
  )


def propagate_teme(
    line1: str, line2: str, dt_utc: datetime
) -> tuple[np.ndarray, np.ndarray]:
  """Propagate TLE to dt_utc and return TEME position (km) and velocity (km/s)."""
  sat = Satrec.twoline2rv(line1, line2)
  jd, fr = dt_to_jd_fr(dt_utc)
  err, r_km, v_km_s = sat.sgp4(jd, fr)
  if err != 0:
    raise RuntimeError(f"SGP4 error {err} for NORAD {sat.satnum}")
  return np.array(r_km, dtype=float), np.array(v_km_s, dtype=float)


def teme_to_rtn(
    r_ref: np.ndarray, v_ref: np.ndarray, delta_r_teme: np.ndarray
) -> np.ndarray:
  """Rotate TEME position offset into reference RTN frame [Radial, In-Track, Cross-Track]."""
  r_hat = r_ref / np.linalg.norm(r_ref)
  h_vec = np.cross(r_ref, v_ref)
  n_hat = h_vec / np.linalg.norm(h_vec)
  t_hat = np.cross(n_hat, r_hat)
  t_hat /= np.linalg.norm(t_hat)
  rot = np.vstack([r_hat, t_hat, n_hat])
  return rot @ delta_r_teme


def extract_tle_elements(line1: str, line2: str) -> dict:
  """Extract orbital parameters from a TLE for training features."""
  sat = Satrec.twoline2rv(line1, line2)
  # Mean motion in rad/min -> rev/day -> semi-major axis approximation
  n_rad_per_min = sat.no_kozai
  n_rad_per_sec = n_rad_per_min / 60.0
  a_km = (MU_EARTH_KM3_S2 / (n_rad_per_sec**2)) ** (1.0 / 3.0)
  alt_km = a_km - EARTH_RADIUS_KM

  return {
      "norad_id": int(sat.satnum),
      "altitude_km": float(alt_km),
      "inclination_deg": float(np.degrees(sat.inclo)),
      "eccentricity": float(sat.ecco),
      "bstar": float(sat.bstar),
  }


def match_space_weather(
    wx_df: pd.DataFrame, target_dt: datetime
) -> tuple[float, float]:
  """
    Given a space weather DataFrame with columns ['timestamp_utc', 'f10_7', 'kp'],
    return nearest (f10_7, kp) for target_dt.
    """
  if wx_df.empty:
    return 120.0, 2.0

  diffs = (wx_df["timestamp_utc"] - target_dt).abs()
  idx = diffs.idxmin()
  row = wx_df.loc[idx]
  return float(row["f10_7"]), float(row["kp"])


def build_rtn_error_table(
    tle_df: pd.DataFrame,
    wx_df: pd.DataFrame,
    horizons_hours: list[float] = (6.0, 12.0, 24.0, 48.0, 72.0),
    max_ref_gap_hours: float = 6.0,
    rcs_lookup: dict[int, float] | None = None,
) -> pd.DataFrame:
  """
    tle_df expected columns: ['norad_id', 'epoch_utc', 'line1', 'line2']
    wx_df expected columns: ['timestamp_utc', 'f10_7', 'kp']
    """
  rcs_lookup = rcs_lookup or {}
  rows = []

  tle_df = tle_df.copy()
  tle_df["epoch_utc"] = pd.to_datetime(tle_df["epoch_utc"], utc=True)

  wx_clean = wx_df.copy()
  if not wx_clean.empty:
    wx_clean["timestamp_utc"] = pd.to_datetime(
        wx_clean["timestamp_utc"], utc=True
    )

  for norad_id, grp in tle_df.groupby("norad_id"):
    grp = grp.sort_values("epoch_utc").reset_index(drop=True)
    epochs = grp["epoch_utc"].tolist()

    for i in range(len(grp)):
      old_row = grp.iloc[i]
      t0 = old_row["epoch_utc"]
      elems = extract_tle_elements(old_row["line1"], old_row["line2"])

      for h in horizons_hours:
        target_dt = t0 + timedelta(hours=float(h))

        # Find closest newer TLE near target_dt within max_ref_gap_hours
        best_j = None
        best_gap_h = None
        for j in range(i + 1, len(grp)):
          gap_h = abs((grp.iloc[j]["epoch_utc"] - target_dt).total_seconds()) / 3600.0
          if gap_h <= max_ref_gap_hours and (
              best_gap_h is None or gap_h < best_gap_h
          ):
            best_j = j
            best_gap_h = gap_h

        if best_j is None:
          continue

        ref_row = grp.iloc[best_j]
        try:
          r_pred, _ = propagate_teme(
              old_row["line1"], old_row["line2"], target_dt
          )
          r_ref, v_ref = propagate_teme(
              ref_row["line1"], ref_row["line2"], target_dt
          )
        except Exception:
          continue

        err_rtn = teme_to_rtn(r_ref, v_ref, r_pred - r_ref)
        f10_7, kp = match_space_weather(wx_clean, target_dt)

        rows.append({
            "norad_id": int(norad_id),
            "epoch_utc": t0.isoformat(),
            "target_utc": target_dt.isoformat(),
            "ref_gap_hours": round(float(best_gap_h), 3),
            "delta_t_hours": float(h),
            "altitude_km": round(elems["altitude_km"], 3),
            "inclination_deg": round(elems["inclination_deg"], 4),
            "eccentricity": round(elems["eccentricity"], 6),
            "bstar": elems["bstar"],
            "rcs_m2": float(rcs_lookup.get(int(norad_id), 1.0)),
            "f10_7": f10_7,
            "kp": kp,
            "err_r_km": float(err_rtn[0]),
            "err_t_km": float(err_rtn[1]),
            "err_n_km": float(err_rtn[2]),
            "abs_err_r_km": float(abs(err_rtn[0])),
            "abs_err_t_km": float(abs(err_rtn[1])),
            "abs_err_n_km": float(abs(err_rtn[2])),
        })

  return pd.DataFrame(rows)


if __name__ == "__main__":
  tle_file = Path("downloaded_tles.csv")
  wx_file = Path("space_weather.csv")
  out_file = Path("real_rtn_errors.csv")

  if not tle_file.exists():
    print(
        f"Waiting for {tle_file.resolve()}. "
        "Create a CSV with columns ['norad_id', 'epoch_utc', 'line1', 'line2'] once your download finishes."
    )
  else:
    tle_data = pd.read_csv(tle_file)
    wx_data = (
        pd.read_csv(wx_file)
        if wx_file.exists()
        else pd.DataFrame(columns=["timestamp_utc", "f10_7", "kp"])
    )
    dataset = build_rtn_error_table(tle_data, wx_data)
    dataset.to_csv(out_file, index=False)
    print(f"Saved {len(dataset)} real RTN training rows to {out_file.resolve()}")