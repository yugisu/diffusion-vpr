import numpy as np


def flat_earth_dist_m(lat1: float, lon1: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
  """Approximate distances in metres from one point to an array of points."""
  dlat = (lats - lat1) * 111_111
  dlon = (lons - lon1) * 111_111 * np.cos(np.radians(lat1))
  return np.sqrt(dlat**2 + dlon**2)
