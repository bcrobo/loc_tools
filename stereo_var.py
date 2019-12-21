#!/usr/bin/python

import cv2
import numpy as np
import collections
import so3
from scipy.stats import chi2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# baseline
b = 0.3
# focal length
f = 585.04
fb = f*b

# Let's p be a point in camera frame
# defined by p = [u, v, d]
# where u and v represent the pixel
# coordinate in the image and d the 
# disparity value associated to that point

# Jacobian of the reconstruction of a point
# given u, v and the disparity
def Jg(K, p):
  u = p[0]
  v = p[1]
  d = p[2]
  cx = K[0,2]
  cy = K[1,2]
  d_sqr = d * d
  return np.array([
#    [b / d, 0, b * (cx - u) / d_sqr],
#    [0, b / d, b * (cy - v) / d_sqr],
    [b / d, 0, 0],
    [0, b / d, 0],
    [0, 0, -fb/d_sqr]])

# Jacobian of the function that
# convert a point from the opencv
# coordinate system to the ros
# coordinate system
def Jf():
  return np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])

# Jacobian of the overall function
# using the chain rule
def J(K, p):
  return np.dot(Jf(), Jg(K, p))

# Backproject a point u, v
def backproject(K, uv):
  u = uv[0]
  v = uv[1]
  fx = K[0,0]
  fy = K[1,1]
  cx = K[0,2]
  cy = K[1,2]
  return np.array([(u - cx) / fx, (v - cy) / fy, 1.0])

# Reconstruct a from u, v, and disparity value a 3d point
# in the camera frame
def reconstruct(K, p):
  disp = p[2]
  z = fb / disp
  xy = backproject(K, p[0:2])
  # move to ros coordinate system
  return np.array([z, -xy[0], -xy[1]])
  
# Return the disparity value
# corresponding to the given depth
def disparity(z):
  return fb / z

# Return points representing the uncertainty
# in 3 dimensions
def ellipsoid(center, cov, confidence=0.95):
  # find the rotation matrix and radii of the axes
  U, s, rotation = np.linalg.svd(cov)
  print("Scales")
  print(s)
  print("Rotation")
  print(rotation)
  
  chi_sqr_val = np.sqrt(chi2.ppf(confidence, cov.shape[0]))
  radii = np.sqrt(chi_sqr_val * s)
  u = np.linspace(0.0, 2.0 * np.pi, 100)
  v = np.linspace(0.0, np.pi, 100)
  x = radii[0] * np.outer(np.cos(u), np.sin(v))
  y = radii[1] * np.outer(np.sin(u), np.sin(v))
  z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
  for i in range(len(x)):
      for j in range(len(x)):
          [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation) + center
  return x, y, z


# Std of the 2d point on u, v, and disparity
sig = np.array([1, 1, 0.5])
# Variance of the 3d point
var = np.diag(np.power(sig, 2))

# Camera parameters
K = np.array([[f, 0, 533.09], [0, f, 418.08], [0, 0, 1]])

# Pixel
u = 256.0
v = 192.0

# Plot
fig = plt.figure()
ax = fig.gca(projection='3d')
for d in np.arange(200, 2, -50):
  # uv, d point
  p = np.array([u, v, disparity(d)])

  # Propagate uncertainty
  Jac = J(K, p)
  var_3d = np.linalg.multi_dot((Jac, var, np.transpose(Jac)))

  # Project the point onto the image plane
  xyz = reconstruct(K, p)

  # Compute ellipsoid confidence
  x, y, z = ellipsoid(xyz, var_3d)

  # Unit vectors
  ax.plot([0, xyz[0]], [0, xyz[1]], zs=[0, xyz[2]])
  ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color='b', alpha=0.2)
  ax.scatter(xyz[0], xyz[1], xyz[2])
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()




