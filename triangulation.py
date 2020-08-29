#!/usr/bin/python

import cv2
import numpy as np
import collections
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

# Convert yaw, pitch and roll to rotation matrix
def eulerZYX_to_rot_matx(yaw, pitch, roll):
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    return np.array([
        [cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
        [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
        [-sp, sr * cp, cr * cp]
    ])

# Project a 3d point onto the image plane
def project(P, pose, K):
  R = cv2.Rodrigues(pose.rvec)[0]
  P_cam = np.dot(R, P) + pose.tvec
  P_camX = P_cam[0] / P_cam[2]
  P_camY = P_cam[1] / P_cam[2]
  P_pix = np.dot(K, np.array([P_camX, P_camY, 1.0]))
  return P_pix

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

def projection_matrix(rvec, tvec, K):
  R = cv2.Rodrigues(rvec)[0]
  T = np.zeros((3,4))
  T[:3, :3] = R
  T[:3, 3] = tvec.T
  return np.dot(K, T)

def update_next_pose(yaw, P):
  Rp = eulerZYX_to_rot_matx(np.pi, 0.0, 0.0)
  Rz = eulerZYX_to_rot_matx(yaw, 0.0, 0.0)
  R = np.linalg.multi_dot((np.transpose(Rp), Rz, Rp, np.eye(3,3)))

  toOrigin = np.dot(Rp, P)
  x_axis = np.array([1.0, 0.0, 0.0])
  toPose = np.linalg.norm(P) * np.linalg.multi_dot((Rz, Rp, x_axis))
  t = toPose - toOrigin
  return Pose(rvec=cv2.Rodrigues(R)[0], tvec=t)

def S(v):
  S = np.zeros((2, 3))
  S[0,1] = -v[2]
  S[0,2] = v[1]
  S[1,0] = v[2]
  S[1,2] = -v[0]
  return S

def triangulate(uv1, uv2, P1, P2):
  A1 = np.dot(S(uv1), P1)
  A2 = np.dot(S(uv2), P2)
  A = np.vstack((A1, A2))
  w, v = np.linalg.eig(np.dot(A.T, A))
  c = np.argmin(w)
  X = v[:, c]
  return A, X

def estimateCov(C1, C2, A, P1, P2, K, X_hat):
  S1 = -S(np.dot(P1, X_hat))
  S2 = -S(np.dot(P2, X_hat))
  B1 = np.hstack((S1, np.zeros((2,3))))
  B2 = np.hstack((np.zeros((2,3)), S2))
  B = np.vstack((B1, B2))

  C = np.zeros((6,6))
  C[:2,:2] = C1
  C[3:5,3:5] = C2
  
  N = np.zeros((5,5))
  N[:4,:4] = np.linalg.multi_dot((A.T, np.linalg.inv(np.linalg.multi_dot((B, C, B.T))), A))
  N[:4, 4] = X_hat
  N[4, :4] = X_hat.T
  C_h = np.linalg.inv(N)[:4, :4]

  Je = np.zeros((3,4))
  Je[:3, :3] = np.eye(3,3)
  Je[:3,3] = (-X_hat / X_hat[3])[:3]
  Je = np.dot(1./X_hat[3], Je)
  C_xyz = np.linalg.multi_dot((Je, C_h, Je.T))

  w, v = np.linalg.eig(C_xyz)
  min_eig_val = w[np.argmin(w)]
  max_eig_val = w[np.argmax(w)]
  r = np.sqrt(min_eig_val / max_eig_val)
  return C_xyz, r

if __name__ == "__main__":
  # Camera intrinsics
  K = np.array([[579.71, 0, 511.5], [0, 579.71, 383.5], [0, 0, 1]])
  # Feature point in opencv coordinate
  X = np.array([3, 0, 0])
  X_cv = np.dot(CV_R_ROS, X)
  max_yaw = np.radians(10)
  num_pose = 2
  yaw_step = max_yaw / num_pose
  yaw = 0
  trajectory = []
  for i in range(num_pose):
    yaw = yaw + yaw_step
    pose = update_next_pose(yaw, X)
    trajectory.append(pose)

  # Noise in measurements (assume ~1pixel)
  z_sigma = 1.
  z_var = z_sigma*z_sigma
  C = np.diag(np.array([z_var, z_var]))

  # Collection of poses
  rvec = np.array([0.0,0.0,0.0])
  tvec = np.array([0.0,0.0,0.0])
  pose1 = Pose(rvec, tvec)
  pose1_cv = toOpencv(pose1)
  P1 = projection_matrix(pose1_cv.rvec, pose1_cv.tvec, K)
  uv1 = project(X_cv, pose1_cv, K)


  rs = []
  for i in range(len(trajectory)):
      posei = trajectory[i]
      posei_cv = toOpencv(posei)
      uvi = project(X_cv, posei_cv, K)
      Pi = projection_matrix(posei_cv.rvec, posei_cv.tvec, K)

      A, X_hat_cv = triangulate(uv1, uvi, P1, Pi)
      X_hat = np.dot(ROS_R_CV, X_hat_cv[:3]/X_hat_cv[3])

      C_xyz, r = estimateCov(C, C, A, P1, Pi, K, X_hat_cv)
      rs.append(r)

      fig = plt.figure()
      ax = fig.gca(projection='3d')
      plot_frame(cv2.Rodrigues(pose1.rvec)[0], pose1.tvec, ax)
      plot_frame(cv2.Rodrigues(posei.rvec)[0], posei.tvec, ax)
      x0, y0, z0 = ellipsoid(X_hat, np.linalg.multi_dot((ROS_R_CV, C_xyz, ROS_R_CV.T)))
      ax.plot_wireframe(x0, y0, z0, rstride=4, cstride=4) 
      ax.set_xlabel('X axis') 
      ax.set_ylabel('Y axis') 
      ax.set_zlabel('z axis') 
      ax.set_xlim(-0.5, 4)
      ax.set_ylim(-0.5, 4)
      ax.set_zlim(-0.5, 0.5)
  #plt.title("R")
  #plt.plot(rs)
  plt.show()

  # Plot
  #alpha_max = 0.8
  #alpha_min = 0.05
  #alpha_range = np.arange(alpha_min, alpha_max, (alpha_max - alpha_min) / len(trajectory))
  #colors = [np.random.rand(3,) for i in range(len(trajectory))]
# # ax.set_aspect(aspect=1)
  ## Point to represent the feature
  ## ax.scatter(P_w[0], P_w[1], P_w[2])
  #for i in range(len(trajectory)):
  #  fig = plt.figure()
  #  ax = fig.gca(projection='3d')
  #  rvec = trajectory[i].rvec
  #  tvec = trajectory[i].tvec
  #  cov_xyz = feature_history[i].cov_xyz
  #  # Plot frame
  #  plot_frame(cv2.Rodrigues(rvec)[0], tvec, ax)
  #  # Plot direction of the feature
  #  #ax.plot([tvec[0], P_w[0]], [tvec[1], P_w[1]], zs=[tvec[2], P_w[2]], color=colors[i])
  #  # Plot xyz uncertainty
  #  #x0, y0, z0 = ellipsoid(P_w[0:3], feature_history[-1].cov_xyz)
  #  x0, y0, z0 = ellipsoid(P_w[0:3], feature_history[i].cov_xyz)
  #  ax.plot_wireframe(x0, y0, z0,  rstride=4, cstride=4, alpha=alpha_range[i], color='b')
  #  ax.set_xlabel('X axis')
  #  ax.set_ylabel('Y axis')
  #  ax.set_zlabel('Z axis')
  #  ax.set_xlim(-0.5, 4)
  #  ax.set_ylim(-0.5, 4)
  #  ax.set_zlim(-0.5, 0.5)
  #plt.show()
