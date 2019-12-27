#!/usr/bin/python

import cv2
import numpy as np
import collections
import se3
import so3
from scipy.stats import chi2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

Pose = collections.namedtuple('Pose', ['rvec', 'tvec'])
Feature = collections.namedtuple('Feature', ['feature', 'cov_inv_depth', 'cov_xyz'])

# Pass a point from OpenCV
# to ROS coordinate system
ROS_R_CV = np.array([
    [0,0,1],
    [-1,0,0],
    [0,-1,0]])
CV_R_ROS = np.transpose(ROS_R_CV)

# Compute the unit vector indicating
# the direction of the current feature
def direction(azimuth, elevation):
  return np.array([
    np.cos(elevation) * np.cos(azimuth),
    np.cos(elevation) * np.sin(azimuth),
    np.sin(elevation)])

# Jacobian of the direction function
# wrt azimuth and elevation angles
def J_direction(azimuth, elevation):
  ce = np.cos(elevation)
  ca = np.cos(azimuth)
  se = np.sin(elevation)
  sa = np.sin(azimuth)
  return np.array([
    [-ce*sa, -se*ca],
    [ce*ca, -se*sa],
    [0, ce]])
    
# Compute the inverse depth representation
# of a feature 
def inverse_depth(pose, K, u, v, rho=0.1):
  # Normalized feature coordinate
  cx = K[0,2]
  cy = K[1,2]
  fx = K[0,0]
  fy = K[1,1]
  h_cam = np.array([
    (u - cx) / fx,
    (v - cy) / fy,
    1.0])

  # Opencv camera pose
  R = cv2.Rodrigues(pose.rvec)[0]
  # Express the feature in world coordinate
  h_world = np.linalg.multi_dot((ROS_R_CV, R, h_cam))
  x_w = h_world[0]
  y_w = h_world[1]
  z_w = h_world[2]
  # Retrieve ray from world origin
  # to feature location
  azimuth = np.arctan(y_w / x_w)
  elevation = np.arctan(z_w / np.sqrt(x_w*x_w + y_w*y_w))
  # Express camera center in world coordinate
  t_w = np.dot(ROS_R_CV, pose.tvec)
  return np.array([
    t_w[0],
    t_w[1],
    t_w[2],
    azimuth,
    elevation,
    rho])

# Compute xyz point representation in world coordinate
# given a point expressed in inverse depth representation
def point_from_inverse_depth(inverse_depth_point, r):
  # Camera center from which the feature
  # was observed the first time
  obs_initial_position = inverse_depth_point[0:3]
  # Direction of the feature
  azimuth = inverse_depth_point[3]
  elevation = inverse_depth_point[4]
  # Inverse depth
  rho = inverse_depth_point[5]
  # Retrieve feature point
  return obs_initial_position + (1/rho) * direction(elevation, azimuth)
  #return rho * (obs_initial_position - r) + direction(elevation, azimuth)

# Jacobian of the function that
# for an inverse depth representation
# retrieves the feature xyz coordinates
def J_reconstruct(feature, pose):
  xi = feature[0]
  yi = feature[1]
  zi = feature[2]
  azimuth = feature[3]
  elevation = feature[4]
  rho = feature[5]
  ce = np.cos(elevation)
  ca = np.cos(azimuth)
  se = np.sin(elevation)
  sa = np.sin(azimuth)
  rho_sq = rho * rho

  J = np.zeros((3,6))
  J[0:3, 0:3] = np.eye(3,3)
  #J[0:3, 0:3] = np.dot(rho, np.eye(3,3))
  # dfx
  J[0,3] = -sa * ce / rho
  J[0,4] = -se * ca / rho
  J[0,5] = -ce * ca / rho_sq
  # dfy
  J[1,3] = ca * ce / rho
  J[1,4] = -se * sa / rho
  J[1,5] = -ce * sa / rho_sq
  # dfz
  J[2,3] = 0
  J[2,4] = ce / rho
  J[2,5] = -se / rho_sq

  #J[0:3, 3:5] = J_direction(azimuth, elevation)
  #J[0:3, 5] = np.array([xi, yi, zi]) - pose.tvec
  return J

# Jacobian of measurement equation
# of a point in inverse depth wrt 
# the position, orientation and a 
# single feature parametrized in
# inverse depth
def J_inv_depth(pose, feature):
  p = pose.tvec
  R = cv2.Rodrigues(pose.rvec)[0]
  xi = feature[0]
  yi = feature[1]
  zi = feature[2]
  azimuth = feature[3]
  elevation = feature[4]
  rho = feature[5]
  ti = np.array([xi, yi, zi])
  di = direction(azimuth, elevation)

  J = np.zeros((3, 6))
  J[0:3, 0:3] = np.dot(rho, np.transpose(R))
  J[0:3, 3:5] = np.dot(np.transpose(R), J_direction(azimuth, elevation))
  J[0:3, 5] = np.dot(np.transpose(R), ti - p)
  return J

# Jacobian of the projection on image plane
def Jp(K, P):
  X = P[0]
  Y = P[1]
  Z = P[2]
  fx = K[0,0]
  fy = K[1,1]

  return np.array([
    [fx/Z, 0, -X*fx/Z*Z],
    [0, fy/Z, -Y*fy/Z*Z]])


def Jh(K, pose, feature):
  hw = point_from_inverse_depth(feature, pose.tvec)
  W_R_C = cv2.Rodrigues(pose.rvec)[0]
  C_R_W = np.transpose(W_R_C)
  hc = np.dot(C_R_W, hw) - np.dot(C_R_W, pose.tvec)
  hc_cv = np.dot(CV_R_ROS, hc)
  return np.linalg.multi_dot((Jp(K, hc_cv), CV_R_ROS, J_inv_depth(pose, feature)))

# Project a 3d point onto the image plane
def project(P, pose, K):
  R = cv2.Rodrigues(pose.rvec)[0]
  P_cam = np.dot(R, P) + pose.tvec
  P_camX = P_cam[0] / P_cam[2]
  P_camY = P_cam[1] / P_cam[2]
  P_pix = np.dot(K, np.array([P_camX, P_camY, 1.0]))
  return P_pix[0:2]

# Convert a pose to opencv coordinate
def toOpencv(pose):
  return  Pose(
      rvec=np.array([-pose.rvec[1], -pose.rvec[2], pose.rvec[0]]),
      tvec=np.array([-pose.tvec[1], -pose.tvec[2], pose.tvec[0]]))

# Return points representing the uncertainty
# in 3 dimensions
def ellipsoid(center, cov, confidence=0.95):
  # find the rotation matrix and radii of the axes
  U, s, rotation = np.linalg.svd(cov)
  
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

def plot_frame(R, t, ax, s=0.02):
  basis = np.eye(3,3)
  R_basis = np.dot(R, basis)
  tx = t[0]
  ty = t[1]
  tz = t[2]
  axis = np.dot(s, R_basis[0:3, 0]) + t
  ax.plot([tx, axis[0]], [ty, axis[1]], [tz, axis[2]], color='r')
  axis = np.dot(s, R_basis[0:3, 1]) + t
  ax.plot([tx, axis[0]], [ty, axis[1]], [tz, axis[2]], color='g')
  axis = np.dot(s, R_basis[0:3, 2]) + t
  ax.plot([tx, axis[0]], [ty, axis[1]], [tz, axis[2]], color='b')

def update_next_pose(yaw, P):
  Rp = so3.eulerZYX_to_rot_matx(np.pi, 0.0, 0.0)
  Rz = so3.eulerZYX_to_rot_matx(yaw, 0.0, 0.0)
  R = np.linalg.multi_dot((np.transpose(Rp), Rz, Rp, np.eye(3,3)))

  toOrigin = np.dot(Rp, P)
  x_axis = np.array([1.0, 0.0, 0.0])
  toPose = np.linalg.norm(P) * np.linalg.multi_dot((Rz, Rp, x_axis))
  t = toPose - toOrigin
  return Pose(rvec=cv2.Rodrigues(R)[0], tvec=t)
  
if __name__ == "__main__":
  # Camera intrinsics
  K = np.array([[579.71, 0, 511.5], [0, 579.71, 383.5], [0, 0, 1]])
  # Collection of poses
  rvec = np.array([0.0,0.0,0.0])
  tvec = np.array([0.0,0,0])
  pose = Pose(rvec, tvec)
  # Feature point in world coordinate
  P = np.array([3, 0, 0])
  feature_history = []
  initialized = False
  # Speed at which we evolves
  num_pose = 3
  max_yaw = np.pi
  yaw_step = max_yaw / num_pose
  yaw = 0
  trajectory = []
  for i in range(num_pose):
    trajectory.append(pose)
    yaw = yaw + yaw_step
    pose = update_next_pose(yaw, P)
  
  # Measurement noise
  sig_u = 0.4
  sig_v = 0.4
  R = np.diag(np.power(np.array([sig_u, sig_v]), 2))
  for pose in trajectory:
    # If the feature is not initialized
    # create an inverse depth representation
    # of it
    if not initialized:
      uv = project(np.dot(CV_R_ROS, P), toOpencv(pose), K)
      # Inverse depth representation of the point
      feature_vector = inverse_depth(toOpencv(pose), K, uv[0], uv[1], rho=1/np.linalg.norm(P))
      # (Facultative) Point in world coordinate for comparison only
      P_w = point_from_inverse_depth(feature_vector, pose.tvec)
      # Initial sigma on inverse depth representation
      covariance_inv_depth = np.diag(np.power(np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.5]),2))
      # Propagate the inverse depth on the xyz depth
      J = J_reconstruct(feature_vector, pose)
      covariance_xyz = np.linalg.multi_dot((J, covariance_inv_depth, np.transpose(J)))
      # Save both covariances
      feature_history.append(Feature(feature=feature_vector, cov_inv_depth=covariance_inv_depth, cov_xyz=covariance_xyz))
      initialized = True
    else:
      # Implement one step kalman filter for measurement
      # and see the effect on the covariance
      f = feature_history[-1]
      H = Jh(K, pose, f.feature)
      S = np.linalg.multi_dot((H, f.cov_inv_depth, np.transpose(H))) + R
      Sinv = np.linalg.inv(S)
      # Kalman gain
      Kgain = np.linalg.multi_dot((f.cov_inv_depth, np.transpose(H), Sinv))
      covariance_inv_depth = np.dot(np.eye(6,6) - np.dot(Kgain, H), f.cov_inv_depth)
      J = J_reconstruct(f.feature, pose)
      covariance_xyz = np.linalg.multi_dot((J, covariance_inv_depth, np.transpose(J)))
      feature_history.append(Feature(feature=feature_vector, cov_inv_depth=covariance_inv_depth, cov_xyz=covariance_xyz))
  # Plot
  alpha_max = 0.8
  alpha_min = 0.05
  alpha_range = np.arange(alpha_min, alpha_max, (alpha_max - alpha_min) / len(trajectory))
  fig = plt.figure()
  colors = [np.random.rand(3,) for i in range(len(trajectory))]
  ax = fig.gca(projection='3d')
#  ax.set_aspect(aspect=1)
  # Point to represent the feature
  # ax.scatter(P_w[0], P_w[1], P_w[2])
  for i in range(len(trajectory)):
    rvec = trajectory[i].rvec
    tvec = trajectory[i].tvec
    cov_xyz = feature_history[i].cov_xyz
    # Plot frame
    plot_frame(cv2.Rodrigues(rvec)[0], tvec, ax)
    # Plot direction of the feature
    ax.plot([tvec[0], P_w[0]], [tvec[1], P_w[1]], zs=[tvec[2], P_w[2]], color=colors[i])
    # Plot xyz uncertainty
    #x0, y0, z0 = ellipsoid(P_w[0:3], feature_history[-1].cov_xyz)
    x0, y0, z0 = ellipsoid(P_w[0:3], feature_history[i].cov_xyz)
    ax.plot_wireframe(x0, y0, z0,  rstride=4, cstride=4, alpha=alpha_range[i], color=colors[i])
  ax.set_xlabel('X axis')
  ax.set_ylabel('Y axis')
  ax.set_zlabel('Z axis')
  ax.set_xlim(-0.5, 4)
  ax.set_ylim(-0.5, 4)
  ax.set_zlim(-0.5, 0.5)
  plt.show()
