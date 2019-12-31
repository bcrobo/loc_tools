#!/usr/bin/python

import cv2
import numpy as np
import collections
import so3
from scipy.stats import chi2
import matplotlib.pyplot as plt

# f function projects a 3d point
# on the image plane

# g function transform a 3d point
# of world coordinate into camera
# frame

# P is a 3d point in the world frame
# expressed using opencv conventions

# We look for the jacobian that will
# propagate the 3d uncertainty of a
# 3d point onto the image plane
# Using the chain rule we got
# J_2x3 = Jf(g(P)) * Jg(P)
# then we can propagate the uncertainty
# of the 3d point using
# J_2d = J * Var_3d * J^T

ROS_R_CV = np.array([
    [0,0,1],
    [-1,0,0],
    [0,-1,0]])
CV_R_ROS = np.transpose(ROS_R_CV)

# Compute the confidence ellipse
def confidence_ellipse(pose, cov, confidence=0.95):
  # Compute eigenvalues and eigenvectors
  eig_val, eig_vec = np.linalg.eig(cov)
  # Sort in ascending order
  idx = eig_val.argsort()[::-1]   
  eig_val = eig_val[idx]
  eig_vec = eig_vec[:,idx]
  # Get the angle between major
  # axis and x axis
  phi = np.arctan2(eig_vec[-1][1], eig_vec[-1][0])
  # Clip angle
  if phi < 0: phi = phi + 2*np.pi
  # 
  chi_sqr_val = np.sqrt(chi2.ppf(confidence, cov.shape[0]))
  # 2d rotation matrix
  R = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])
  # Compute axis length
  a = chi_sqr_val * np.sqrt(eig_val[-1]) 
  b = chi_sqr_val * np.sqrt(eig_val[0])
  theta_grid = np.linspace(0, 2*np.pi)
  ellipse_x = a * np.cos(theta_grid)
  ellipse_y = b * np.sin(theta_grid)
  ellipse = np.dot(R, np.array([ellipse_x, ellipse_y]))
  return ellipse[0,:] + pose[0], ellipse[1,:] + pose[1]

# Jacobian of the projection on image plane
def Jf(K, P):
  X = P[0]
  Y = P[1]
  Z = P[2]
  fx = K[0,0]
  fy = K[1,1]

  return np.array([
    [fx/Z, 0, -X*fx/Z*Z],
    [0, fy/Z, -Y*fy/Z*Z]])

# Jacobian of the rigid transformation
def Jg(pose):
  return cv2.Rodrigues(pose.rvec)[0]

# 2 by 3 jacobian of the projection
def J(K, pose, P):
  R = cv2.Rodrigues(pose.rvec)[0]
  P_cam = np.dot(R, P) + pose.tvec
  return np.linalg.multi_dot((Jf(K, P_cam), CV_R_ROS, Jg(pose)))

# Project a 3d point onto the image plane
def project(P, pose, K, res_y):
  R = cv2.Rodrigues(pose.rvec)[0]
  P_cam = np.dot(R, P) + pose.tvec
  P_camX = P_cam[0] / P_cam[2]
  P_camY = P_cam[1] / P_cam[2]
  P_pix = np.dot(K, np.array([P_camX, P_camY, 1.0]))
  P_pix[1] = res_y - P_pix[1]
  return P_pix[0:2]

# Opencv pose
Pose = collections.namedtuple('Pose', ['rvec', 'tvec'])

# Std of the 3d point
sig = np.array([0.0, 0.1, 0.2])
# Variance of the 3d point
var = np.diag(np.power(sig, 2))

# Camera parameters
res_x = 1024
res_y = 768
K = np.array([[579.71, 0, 511.5], [0, 579.71, 383.5], [0, 0, 1]])

# 3d point
P = np.array([0, 0, 2])

# Pose
angle, axis = so3.log_map_rot_matx(so3.eulerZYX_to_rot_matx(0.1, 0.1, 0.1))
rvec = angle * axis
tvec = np.array([0,0,0])
pose = Pose(rvec, tvec)

# Propagate uncertainty
Jac = J(K, pose, P)
var_pix = np.linalg.multi_dot((Jac, var, np.transpose(Jac)))

# Project the point onto the image plane
uv = project(P, pose, K, res_y)
xe95, ye95 = confidence_ellipse(uv, var_pix)
xe99, ye99 = confidence_ellipse(uv, var_pix, confidence=0.99)

# Plot
plt.figure(0)
plt.xlim(0, res_x)
plt.ylim(0, res_y)
plt.xlabel("u (pix)")
plt.ylabel("v (pix)")
plt.scatter(uv[0], uv[1], marker='+')
plt.plot(xe95, ye95)
plt.plot(xe99, ye99)
plt.grid()
plt.show()
