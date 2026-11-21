from dataclasses import dataclass
import numpy as np
from scipy.integrate import dblquad


@dataclass
class ObjectState:
  name: str
  r_km: np.ndarray  # Position vector in TEME/ECI (km), shape (3,)
  v_km_s: np.ndarray  # Velocity vector in TEME/ECI (km/s), shape (3,)
  sigma_rtn_km: (
      np.ndarray
  )  # 1-sigma uncertainty [sigma_R, sigma_T, sigma_N] (km), shape (3,)
  hard_body_radius_m: float  # Effective hard-body radius in meters


def rtn_to_eci_matrix(r_km: np.ndarray, v_km_s: np.ndarray) -> np.ndarray:
  """
    Return 3x3 rotation matrix R_rtn_to_eci whose columns are [R_hat, T_hat, N_hat]
    expressed in the object's TEME/ECI frame.
    """
  r_norm = np.linalg.norm(r_km)
  if r_norm == 0:
    raise ValueError("Position vector cannot be zero.")
  r_hat = r_km / r_norm

  h_vec = np.cross(r_km, v_km_s)
  h_norm = np.linalg.norm(h_vec)
  if h_norm == 0:
    raise ValueError("Angular momentum vector cannot be zero.")
  n_hat = h_vec / h_norm

  t_hat = np.cross(n_hat, r_hat)
  t_hat = t_hat / np.linalg.norm(t_hat)

  # Columns are R, T, N basis vectors in ECI/TEME
  return np.column_stack([r_hat, t_hat, n_hat])


def build_eci_covariance(obj: ObjectState) -> np.ndarray:
  """Build 3x3 diagonal covariance in RTN from [sigma_R, sigma_T, sigma_N] and rotate to ECI."""
  sigmas = np.asarray(obj.sigma_rtn_km, dtype=float)
  if sigmas.shape != (3,) or np.any(sigmas <= 0):
    raise ValueError("sigma_rtn_km must have 3 positive values [R, T, N].")
  cov_rtn = np.diag(sigmas**2)
  rot = rtn_to_eci_matrix(obj.r_km, obj.v_km_s)
  return rot @ cov_rtn @ rot.T


def encounter_plane_2d_params(
    primary: ObjectState, secondary: ObjectState
) -> tuple[np.ndarray, np.ndarray, float]:
  """
    Project relative miss vector (secondary - primary) and combined 3x3 covariance
    onto the 2D encounter plane perpendicular to relative velocity v_rel.

    Returns:
      mu_2d: projected miss vector [x, y] in km, shape (2,)
      cov_2d: projected 2x2 covariance matrix in km^2, shape (2, 2)
      combined_radius_km: sum of hard-body radii in km
    """
  cov1 = build_eci_covariance(primary)
  cov2 = build_eci_covariance(secondary)
  cov_comb_3d = cov1 + cov2

  rel_pos = secondary.r_km - primary.r_km
  rel_vel = secondary.v_km_s - primary.v_km_s
  v_rel_norm = np.linalg.norm(rel_vel)
  if v_rel_norm == 0:
    raise ValueError("Relative velocity cannot be zero.")

  # Orthonormal encounter frame:
  # z_hat along relative velocity; x_hat and y_hat span the 2D encounter plane
  z_hat = rel_vel / v_rel_norm
  ref_vec = (
      np.array([1.0, 0.0, 0.0])
      if abs(z_hat[0]) < 0.9
      else np.array([0.0, 1.0, 0.0])
  )
  x_hat = np.cross(ref_vec, z_hat)
  x_hat /= np.linalg.norm(x_hat)
  y_hat = np.cross(z_hat, x_hat)

  # 2x3 projection matrix onto the (x_hat, y_hat) encounter plane
  proj_2x3 = np.vstack([x_hat, y_hat])

  mu_2d = proj_2x3 @ rel_pos
  cov_2d = proj_2x3 @ cov_comb_3d @ proj_2x3.T
  combined_radius_km = (
      primary.hard_body_radius_m + secondary.hard_body_radius_m
  ) / 1000.0

  return mu_2d, cov_2d, combined_radius_km


def compute_2d_pc(
    mu_2d: np.ndarray, cov_2d: np.ndarray, radius_km: float
) -> float:
  """
    Integrate a 2D bivariate Gaussian N(mu_2d, cov_2d) over a circle of radius `radius_km`
    centered at the origin of the encounter plane (Foster 2D encounter-plane formulation).
    """
  det = np.linalg.det(cov_2d)
  if det <= 0:
    raise ValueError("Projected 2D covariance matrix must be positive definite.")
  inv_cov = np.linalg.inv(cov_2d)
  norm_const = 1.0 / (2.0 * np.pi * np.sqrt(det))

  def pdf_2d(y: float, x: float) -> float:
    diff = np.array([x - mu_2d[0], y - mu_2d[1]])
    exponent = -0.5 * float(diff.T @ inv_cov @ diff)
    return norm_const * np.exp(exponent)

  # Integrate over disk x^2 + y^2 <= R^2
  r = radius_km
  prob, _ = dblquad(
      pdf_2d,
      -r,
      r,
      lambda x: -np.sqrt(max(0.0, r * r - x * x)),
      lambda x: np.sqrt(max(0.0, r * r - x * x)),
      epsabs=1e-13,
      epsrel=1e-8,
  )
  return float(np.clip(prob, 0.0, 1.0))


def triage_action(
    pc: float,
    monitor_threshold: float = 1e-6,
    act_threshold: float = 1e-4,
) -> str:
  """Map Pc into Ignore / Monitor / Act."""
  if pc >= act_threshold:
    return "Act"
  if pc >= monitor_threshold:
    return "Monitor"
  return "Ignore"


def assess_conjunction(
    primary: ObjectState,
    secondary: ObjectState,
    sigma_rtn_low_1: np.ndarray | None = None,
    sigma_rtn_high_1: np.ndarray | None = None,
    sigma_rtn_low_2: np.ndarray | None = None,
    sigma_rtn_high_2: np.ndarray | None = None,
    monitor_threshold: float = 1e-6,
    act_threshold: float = 1e-4,
) -> dict:
  """
    Full conjunction assessment returning nominal Pc, triage decision,
    miss distance, and optional Pc uncertainty bounds.
    """
  miss_km = float(np.linalg.norm(secondary.r_km - primary.r_km))
  mu_2d, cov_2d, r_comb_km = encounter_plane_2d_params(primary, secondary)
  pc_nominal = compute_2d_pc(mu_2d, cov_2d, r_comb_km)
  action = triage_action(pc_nominal, monitor_threshold, act_threshold)

  result = {
      "primary": primary.name,
      "secondary": secondary.name,
      "miss_distance_km": round(miss_km, 4),
      "combined_radius_m": round(r_comb_km * 1000.0, 2),
      "pc": pc_nominal,
      "action": action,
  }

  # Optional bounds from lower/upper RTN uncertainty estimates
  if (
      sigma_rtn_low_1 is not None
      and sigma_rtn_high_1 is not None
      and sigma_rtn_low_2 is not None
      and sigma_rtn_high_2 is not None
  ):
    low_1 = ObjectState(
        primary.name,
        primary.r_km,
        primary.v_km_s,
        sigma_rtn_low_1,
        primary.hard_body_radius_m,
    )
    low_2 = ObjectState(
        secondary.name,
        secondary.r_km,
        secondary.v_km_s,
        sigma_rtn_low_2,
        secondary.hard_body_radius_m,
    )
    high_1 = ObjectState(
        primary.name,
        primary.r_km,
        primary.v_km_s,
        sigma_rtn_high_1,
        primary.hard_body_radius_m,
    )
    high_2 = ObjectState(
        secondary.name,
        secondary.r_km,
        secondary.v_km_s,
        sigma_rtn_high_2,
        secondary.hard_body_radius_m,
    )

    _, cov_low, _ = encounter_plane_2d_params(low_1, low_2)
    _, cov_high, _ = encounter_plane_2d_params(high_1, high_2)
    pc_low = compute_2d_pc(mu_2d, cov_low, r_comb_km)
    pc_high = compute_2d_pc(mu_2d, cov_high, r_comb_km)
    result["pc_low"] = min(pc_low, pc_high)
    result["pc_high"] = max(pc_low, pc_high)

  return result


if __name__ == "__main__":
  # Example test case at TCA (~550 km LEO)
  sat = ObjectState(
      name="SAT-LEO-104",
      r_km=np.array([6920.0, 0.0, 0.0]),
      v_km_s=np.array([0.0, 7.58, 0.0]),
      sigma_rtn_km=np.array([0.08, 0.45, 0.06]),
      hard_body_radius_m=8.0,
  )
  debris = ObjectState(
      name="DEB-FENGYUN-882",
      r_km=np.array([6920.35, 0.02, 0.01]),
      v_km_s=np.array([0.0, -7.58, 0.20]),
      sigma_rtn_km=np.array([0.12, 0.80, 0.09]),
      hard_body_radius_m=4.0,
  )

  outcome = assess_conjunction(
      sat,
      debris,
      sigma_rtn_low_1=np.array([0.04, 0.22, 0.03]),
      sigma_rtn_high_1=np.array([0.16, 0.95, 0.12]),
      sigma_rtn_low_2=np.array([0.06, 0.40, 0.05]),
      sigma_rtn_high_2=np.array([0.25, 1.50, 0.18]),
  )
  for k, v in outcome.items():
    print(f"{k}: {v}")