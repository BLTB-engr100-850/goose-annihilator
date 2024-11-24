import numpy as np

desired_vec = np.array([0, -0.2, 1])
desired_vec = desired_vec / np.linalg.norm(desired_vec)

mirror_normal = (desired_vec - np.array([1, 0, 0]))
# print(mirror_normal)
mirror_normal = mirror_normal/np.linalg.norm(mirror_normal)
print(mirror_normal)
# theta = np.arctan2(mirror_normal[0], mirror_normal[2])
# phi = np.arcsin(mirror_normal[1])

# theta = int(1020 + theta * 1100 / (np.pi/2)) # converting to servo microseconds
# phi = int(1135 + phi * 2000 / np.pi) # converting to servo microseconds

# print(theta, phi)
v_out = np.array([.073279, -.63902, 3.82972017])
v_out = v_out / np.linalg.norm(v_out)
print(v_out)

n = mirror_normal

v_in = v_out - 2 * np.dot(v_out, n) * n
print(v_in)