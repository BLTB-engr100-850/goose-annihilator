import numpy as np
import matplotlib.pyplot as plt

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

def screen_to_ray(x, y, FOV_h, FOV_v):
  """ Convert screen space coordinates to ray direction in 3D. """
  theta_x = (x * FOV_h) / 2
  theta_y = (y * FOV_v) / 2
  # Compute the direction vector
  ray_dir = np.array([
    np.sin(theta_x),
    np.sin(theta_y),
    np.cos(theta_x) * np.cos(theta_y)
  ])
  return ray_dir / np.linalg.norm(ray_dir)

def compute_aim_angles(P_obj):
  """ Compute the horizontal and vertical angles to aim the laser. """
  x_obj, y_obj, z_obj = P_obj
  theta_h = np.arctan2(y_obj, x_obj)
  theta_v = np.arcsin(z_obj / np.linalg.norm(P_obj))
  return np.degrees(theta_h), np.degrees(theta_v)

def screen_to_ray(x, y, FOV_h, FOV_v):
  """ Convert screen space coordinates to ray direction in 3D. """
  theta_x = (x * FOV_h) / 2
  theta_y = (y * FOV_v) / 2
  # Compute the direction vector
  ray_dir = np.array([
    np.sin(theta_x),
    np.sin(theta_y),
    np.cos(theta_x) * np.cos(theta_y)
  ])
  return ray_dir / np.linalg.norm(ray_dir)

def visualize_scene(C1, C2, ray1, ray2, P_obj, laser_origin=np.array([0, 0, 0])):
  """
  Visualize the cameras, rays, laser, and object in 3D space.
  
  Parameters:
  C1: numpy array, Camera 1 position
  C2: numpy array, Camera 2 position
  ray1: numpy array, direction of ray from Camera 1
  ray2: numpy array, direction of ray from Camera 2
  P_obj: numpy array, the 3D position of the object
  laser_origin: numpy array, the position of the laser origin
  """
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  # Plot Camera 1
  ax.scatter(C1[0], C1[1], C1[2], color='blue', s=100, label="Camera 1")
  ax.text(C1[0], C1[1], C1[2], 'Camera 1', color='blue')

  # Plot Camera 2
  ax.scatter(C2[0], C2[1], C2[2], color='green', s=100, label="Camera 2")
  ax.text(C2[0], C2[1], C2[2], 'Camera 2', color='green')

  # Plot Laser Origin
  ax.scatter(laser_origin[0], laser_origin[1], laser_origin[2], color='red', s=100, label="Laser Origin")
  ax.text(laser_origin[0], laser_origin[1], laser_origin[2], 'Laser Origin', color='red')

  # Plot Object
  ax.scatter(P_obj[0], P_obj[1], P_obj[2], color='purple', s=200, label="Object")
  ax.text(P_obj[0], P_obj[1], P_obj[2], 'Object', color='purple')

  # Draw rays from Camera 1 to Object
  t_values = np.linspace(0, 10, 100)
  ray1_path = np.array([C1 + t * ray1 for t in t_values])
  ax.plot(ray1_path[:, 0], ray1_path[:, 1], ray1_path[:, 2], color='blue', linestyle='--', label="Ray 1")

  # Draw rays from Camera 2 to Object
  ray2_path = np.array([C2 + t * ray2 for t in t_values])
  ax.plot(ray2_path[:, 0], ray2_path[:, 1], ray2_path[:, 2], color='green', linestyle='--', label="Ray 2")

  # Draw laser from origin to Object
  laser_path = np.array([laser_origin + t * (P_obj - laser_origin) for t in np.linspace(0, 1, 100)])
  ax.plot(laser_path[:, 0], laser_path[:, 1], laser_path[:, 2], color='red', label="Laser")

  # Set plot limits
  ax.set_xlim([-10, 10])
  ax.set_ylim([-10, 10])
  ax.set_zlim([-10, 10])

  # Set labels
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  
  # Add legend
  ax.legend()

  # Show plot
  plt.show()

if __name__ == "__main__":
  # Given FOV in degrees for horizontal and vertical
  FOV_h = 90  # horizontal FOV in degrees
  FOV_v = 60  # vertical FOV in degrees

  # Convert FOV to radians
  FOV_h = np.radians(FOV_h)
  FOV_v = np.radians(FOV_v)

  # Example values for camera positions
  C1 = np.array([-10, 0, 0])  # Camera 1 position
  C2 = np.array([10, 0, 0])  # Camera 2 position

  # Screen space coordinates from camera 1 and camera 2, in the range [-1, 1], so make sure to normalize for resolution
  x1, y1 = 0.5, 0  # From camera 1
  x2, y2 = -0.5, 0  # From camera 2

  # Convert screen coordinates to rays
  ray1 = screen_to_ray(x1, y1, FOV_h, FOV_v)
  ray2 = screen_to_ray(x2, y2, FOV_h, FOV_v)

  r1, r2 = closest_points_on_lines(C1, ray1, C2, ray2)
  P_obj = midpoint(r1, r2)

  # Compute aiming angles for the laser
  theta_h, theta_v = compute_aim_angles(P_obj)

  print("Horizontal angle to aim the laser:", theta_h)
  print("Vertical angle to aim the laser:", theta_v)

  visualize_scene(C1, C2, ray1, ray2, P_obj)