import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def closest_points_on_lines(p1, d1, p2, d2):
  """
  Calculate the closest points on two skew lines in 3D.
  
  Parameters:
  p1: numpy array of shape (3,), a point on the first line
  d1: numpy array of shape (3,), the direction vector of the first line
  p2: numpy array of shape (3,), a point on the second line
  d2: numpy array of shape (3,), the direction vector of the second line
  
  Returns:
  r1: numpy array of shape (3,), the closest point on the first line
  r2: numpy array of shape (3,), the closest point on the second line
  """
  # Normalize direction vectors (optional, but generally good practice)
  d1 = d1 / np.linalg.norm(d1)
  d2 = d2 / np.linalg.norm(d2)

  # Vector between points on the two lines
  p12 = p1 - p2

  # Coefficients for the system of equations
  a = np.dot(d1, d1)
  b = np.dot(d1, d2)
  c = np.dot(d2, d2)
  d = np.dot(d1, p12)
  e = np.dot(d2, p12)

  # Solve the system of linear equations
  denom = a * c - b * b
  if np.abs(denom) < 1e-6:
    raise ValueError("Lines are parallel or nearly parallel.")

  t1 = (b * e - c * d) / denom
  t2 = (a * e - b * d) / denom

  # Closest points on each line
  r1 = p1 + t1 * d1  # Closest point on line 1
  r2 = p2 + t2 * d2  # Closest point on line 2

  return r1, r2

def midpoint(r1, r2):
  """
  Calculate the midpoint between two points in 3D.
  
  Parameters:
  r1: numpy array of shape (3,), the first point
  r2: numpy array of shape (3,), the second point
  
  Returns:
  midpoint: numpy array of shape (3,), the midpoint between r1 and r2
  """
  return (r1 + r2) / 2

def plot_lines_and_closest_point(p1, d1, p2, d2, r1, r2, midpoint):
  """
  Plot two lines in 3D and their closest points.
  
  Parameters:
  p1, d1: point and direction vector of the first line
  p2, d2: point and direction vector of the second line
  r1: closest point on the first line
  r2: closest point on the second line
  midpoint: midpoint between r1 and r2
  """
  # Create a range of values to plot the lines
  t_range = np.linspace(-10, 10, 100)

  # Line 1 points
  line1 = np.array([p1 + t * d1 for t in t_range])

  # Line 2 points
  line2 = np.array([p2 + t * d2 for t in t_range])

  # Create a 3D plot
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  # Plot line 1
  ax.plot(line1[:, 0], line1[:, 1], line1[:, 2], label="Line 1", color='blue')

  # Plot line 2
  ax.plot(line2[:, 0], line2[:, 1], line2[:, 2], label="Line 2", color='green')

  # Plot closest points
  ax.scatter(r1[0], r1[1], r1[2], color='red', label="Closest point on Line 1", s=100)
  ax.scatter(r2[0], r2[1], r2[2], color='orange', label="Closest point on Line 2", s=100)

  # Plot the midpoint
  ax.scatter(midpoint[0], midpoint[1], midpoint[2], color='purple', label="Midpoint", s=100)

  # Draw a line between the closest points
  ax.plot([r1[0], r2[0]], [r1[1], r2[1]], [r1[2], r2[2]], color='black', linestyle='--', label="Distance between closest points")

  # Set labels
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  ax.legend()

  plt.show()

# Example usage:
p1 = np.array([1, 2, 3])  # A point on line 1
d1 = np.array([4, 5, 6])  # Direction vector of line 1

p2 = np.array([7, 8, 9])  # A point on line 2
d2 = np.array([10, 11, -12])  # Direction vector of line 2

# Get the closest points on both lines
r1, r2 = closest_points_on_lines(p1, d1, p2, d2)

# Find the midpoint of the closest points
r_closest = midpoint(r1, r2)

# Plot the lines and the closest points
plot_lines_and_closest_point(p1, d1, p2, d2, r1, r2, r_closest)
