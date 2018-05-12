from acos_dist import acos_dist
import numpy as np
from numpy import argmin
from parameters import R
from math import pi

class Coordinates :
  def __init__(self, all_coords):
    self.all_coords = list(all_coords)
    self.unique_coords = self.to_unique(self.all_coords)

  @staticmethod
  def to_unique(all_coords):
    all_coords = [tuple(coord) for coord in all_coords]
    unique_list = list(set(all_coords))
    return [np.array(coord) for coord in unique_list]

  def min_coord(self, coord):
    all_distances = [self.acos_dist(coord, coord_i)
                     for coord_i in self.unique_coords]
    min_idx = argmin(all_distances)
    return self.unique_coords[min_idx]

  def min_coords(self, coords):
    return [list(self.min_coord(coord)) for coord in coords]

  @staticmethod
  def acos_dist(coord_1, coord_2, R=R):
    coord_1 *= pi

    lat_1 = coord_1[0]
    lat_2 = coord_2[0]
    long_1 = coord_1[1]
    long_2 = coord_2[1]

    f = np.sin((lat_2 - lat_1) / 2) ** 2 + np.cos(lat_1) * np.cos(
      lat_2) * np.sin(
        (long_1 - long_2) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(f))