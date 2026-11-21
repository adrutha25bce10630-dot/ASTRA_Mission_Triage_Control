from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
from sgp4.api import Satrec, jday

EARTH_RADIUS_KM = 6378.137
MU_EARTH_KM3_S2 = 398600.4418


def parse_tle_epoch_utc(line1: str) -> datetime:
  """Parse TLE Line 1 epoch (YYDDD.DDDDDDDD) into UTC datetime."""
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


def propagate_state(
    sat: Satrec, dt_utc: datetime
) -> tuple[np.ndarray, np.ndarray]:
  """Propagate Satrec to dt_utc and return (r_km, v_km_s) in TEME."""
  jd, fr = dt_to_jd_fr(dt_utc)
  err, r_km, v_km_s = sat.sgp4(jd, fr)
  if err != 0:
    raise RuntimeError(f"SGP4 error {err} for NORAD {sat.satnum}")
  return np.array(r_km, dtype=float), np.array(v_km_s, dtype=float)


def tle_features(
    line1: str,
    line2: str,
    tca_utc: datetime,
    f10_7: float = 130.0,
    kp: float = 2.5,
    rcs_m2: float = 1.0,
) -> dict:
  """Extract model feature dictionary for a TLE propagated to tca_utc."""
  sat = Satrec.twoline2rv(line1, line2)
  epoch_utc = parse_tle_epoch_utc(line1)
  delta_t_hours = max(0.1, (tca_utc - epoch_utc).total_seconds() / 3600.0)

  n_rad_per_sec = sat.no_kozai / 60.0
  a_km = (MU_EARTH_KM3_S2 / (n_rad_per_sec**2)) ** (1.0 / 3.0)
  alt_km = a_km - EARTH_RADIUS_KM

  return {
      "delta_t_hours": round(delta_t_hours, 3),
      "altitude_km": round(alt_km, 2),
      "inclination_deg": round(float(np.degrees(sat.inclo)), 4),
      "eccentricity": float(sat.ecco),
      "bstar": float(sat.bstar),
      "rcs_m2": float(rcs_m2),
      "f10_7": float(f10_7),
      "kp": float(kp),
  }


def refine_tca(
    sat1: Satrec,
    sat2: Satrec,
    center_dt: datetime,
    window_seconds: int = 120,
    step_seconds: int = 5,
) -> tuple[datetime, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Refine closest approach around center_dt at finer time resolution."""
  best_dt = center_dt
  best_dist = float("inf")
  best_states = None

  n_steps = int((2 * window_seconds) / step_seconds) + 1
  start_dt = center_dt - timedelta(seconds=window_seconds)

  for k in range(n_steps):
    dt_curr = start_dt + timedelta(seconds=k * step_seconds)
    r1, v1 = propagate_state(sat1, dt_curr)
    r2, v2 = propagate_state(sat2, dt_curr)
    dist = float(np.linalg.norm(r2 - r1))
    if dist < best_dist:
      best_dist = dist
      best_dt = dt_curr
      best_states = (r1, v1, r2, v2)

  assert best_states is not None
  r1, v1, r2, v2 = best_states
  return best_dt, best_dist, r1, v1, r2, v2


def screen_conjunctions(
    name1: str,
    line1_a: str,
    line2_a: str,
    name2: str,
    line1_b: str,
    line2_b: str,
    start_dt: datetime,
    horizon_hours: float = 48.0,
    coarse_step_seconds: int = 60,
    distance_filter_km: float = 25.0,
    f10_7: float = 135.0,
    kp: float = 2.8,
    rcs1_m2: float = 10.0,
    rcs2_m2: float = 0.5,
) -> list[dict]:
  """
    Screen two objects over [start_dt, start_dt + horizon_hours] for local
    minima in separation distance within `distance_filter_km`.
    """
  sat1 = Satrec.twoline2rv(line1_a, line2_a)
  sat2 = Satrec.twoline2rv(line1_b, line2_b)

  total_seconds = int(horizon_hours * 3600)
  n_points = max(3, total_seconds // coarse_step_seconds + 1)
  times = [
      start_dt + timedelta(seconds=i * coarse_step_seconds)
      for i in range(n_points)
  ]
  dists = np.empty(n_points, dtype=float)

  for i, t_curr in enumerate(times):
    r1, _ = propagate_state(sat1, t_curr)
    r2, _ = propagate_state(sat2, t_curr)
    dists[i] = float(np.linalg.norm(r2 - r1))

  candidates = []
  # Detect interior local minima in distance
  for i in range(1, n_points - 1):
    if dists[i] < dists[i - 1] and dists[i] <= dists[i + 1]:
      if dists[i] <= distance_filter_km:
        tca_dt, min_dist_km, r1, v1, r2, v2 = refine_tca(
            sat1,
            sat2,
            times[i],
            window_seconds=coarse_step_seconds,
            step_seconds=5,
        )
        candidates.append({
            "tca_utc": tca_dt.isoformat(),
            "miss_distance_km": round(min_dist_km, 4),
            "primary": {
                "name": name1,
                "r_km": r1.tolist(),
                "v_km_s": v1.tolist(),
                "features": tle_features(
                    line1_a, line2_a, tca_dt, f10_7=f10_7, kp=kp, rcs_m2=rcs1_m2
                ),
            },
            "secondary": {
                "name": name2,
                "r_km": r2.tolist(),
                "v_km_s": v2.tolist(),
                "features": tle_features(
                    line1_b, line2_b, tca_dt, f10_7=f10_7, kp=kp, rcs_m2=rcs2_m2
                ),
            },
        })

  return candidates


if __name__ == "__main__":
  # Example TLE pair for quick screening test
  p_name = "ISS (ZARYA)"
  p_l1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  30123-3 0  9993"
  p_l2 = "2 25544  51.6412  15.2345 0006789  45.1234  80.5678 15.49812345432101"

  s_name = "SAMPLE DEBRIS"
  s_l1 = "1 40001U 98067B   24001.50000000  .00021000  00000-0  41000-3 0  9991"
  s_l2 = "2 40001  51.6420  15.2360 0007100  46.0000  79.9000 15.49950000123456"

  t_start = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
  hits = screen_conjunctions(
      p_name,
      p_l1,
      p_l2,
      s_name,
      s_l1,
      s_l2,
      start_dt=t_start,
      horizon_hours=48.0,
      coarse_step_seconds=60,
      distance_filter_km=50.0,
  )
  print(f"Found {len(hits)} close-approach candidate(s).")
  for h in hits[:5]:
    print(f"  TCA={h['tca_utc']} | Miss={h['miss_distance_km']} km")