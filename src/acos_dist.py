from parameters import R
import autograd.numpy as np
from math import pi

def acos_dist(coord_1, coord_2, R=R):
  coord_1 *= pi

  lat_1 = coord_1[0]
  lat_2 = coord_2[0]
  long_1 = coord_1[1]
  long_2 = coord_2[1]

  f = np.sin((lat_2 - lat_1) / 2) ** 2 + np.cos(lat_1) * np.cos(lat_2) * np.sin(
    (long_1 - long_2) / 2) ** 2
  return 2 * R * np.arcsin(np.sqrt(f))

def acosdist(coord_1_list, coord_2_list, R=R):
  coords = list(zip(coord_1_list, coord_2_list))
  distances = np.array([acos_dist(coord1, coord_2, R)
                        for coord1,coord_2 in coords])
  return np.sum(distances) / distances.shape[0]