from pathlib import Path
import numpy as np
import pandas as pd


def generate_dummy_rtn_dataset(
    n_samples: int = 3000, seed: int = 42
) -> pd.DataFrame:
  rng = np.random.default_rng(seed)

  # 1. Input features matching ASTRA's physical regime
  delta_t_hours = rng.uniform(2.0, 96.0, size=n_samples)  # 2h to 4 days
  altitude_km = rng.uniform(
      380.0, 900.0, size=n_samples
  )  # LEO altitude range
  inclination_deg = rng.choice(
      [51.6, 53.2, 74.0, 86.4, 97.6], size=n_samples
  )  # Common LEO inclinations
  eccentricity = rng.uniform(0.0001, 0.015, size=n_samples)
  bstar = rng.lognormal(
      mean=-9.5, sigma=1.1, size=n_samples
  )  # Drag / radiation proxy
  rcs_m2 = rng.lognormal(
      mean=-0.5, sigma=1.2, size=n_samples
  )  # Object radar cross-section
  f10_7 = rng.uniform(70.0, 240.0, size=n_samples)  # Solar radio flux (sfu)
  kp = rng.uniform(0.5, 7.5, size=n_samples)  # Geomagnetic index

  # 2. Physical scaling factors (stronger drag lower down & during solar/geomagnetic activity)
  alt_drag_factor = np.exp(-(altitude_km - 380.0) / 160.0)
  bstar_norm = np.clip(bstar / 1e-4, 0.05, 25.0)
  solar_factor = np.clip((f10_7 - 65.0) / 100.0, 0.5, 2.5)
  geomag_factor = np.exp(np.clip((kp - 2.0) / 4.5, -0.5, 1.5))
  time_growth_t = (delta_t_hours / 12.0) ** 1.35
  time_growth_rn = (delta_t_hours / 12.0) ** 0.85

  # Base scale (in km) per RTN axis
  base_scale = 0.02 + 0.35 * alt_drag_factor * np.sqrt(
      bstar_norm
  ) * solar_factor * geomag_factor

  scale_r = (
      base_scale * 0.25 * time_growth_rn * rng.uniform(0.7, 1.3, size=n_samples)
  )
  scale_t = (
      base_scale * 1.40 * time_growth_t * rng.uniform(0.7, 1.3, size=n_samples)
  )
  scale_n = (
      base_scale * 0.20 * time_growth_rn * rng.uniform(0.7, 1.3, size=n_samples)
  )

  # Signed residuals (km) + absolute magnitudes (km)
  err_r_km = rng.normal(0.0, scale_r)
  err_t_km = rng.normal(0.0, scale_t)
  err_n_km = rng.normal(0.0, scale_n)

  df = pd.DataFrame({
      "delta_t_hours": np.round(delta_t_hours, 3),
      "altitude_km": np.round(altitude_km, 2),
      "inclination_deg": inclination_deg,
      "eccentricity": np.round(eccentricity, 6),
      "bstar": bstar,
      "rcs_m2": np.round(rcs_m2, 4),
      "f10_7": np.round(f10_7, 2),
      "kp": np.round(kp, 2),
      "err_r_km": err_r_km,
      "err_t_km": err_t_km,
      "err_n_km": err_n_km,
      "abs_err_r_km": np.abs(err_r_km),
      "abs_err_t_km": np.abs(err_t_km),
      "abs_err_n_km": np.abs(err_n_km),
  })
  return df


if __name__ == "__main__":
  out_path = Path("dummy_rtn_errors.csv")
  dataset = generate_dummy_rtn_dataset()
  dataset.to_csv(out_path, index=False)
  print(f"Saved {len(dataset)} rows to {out_path.resolve()}")
  print(
      dataset[[
          "delta_t_hours",
          "altitude_km",
          "f10_7",
          "kp",
          "abs_err_r_km",
          "abs_err_t_km",
          "abs_err_n_km",
      ]].describe()
  )