#!/usr/bin/python

import numpy as np
import se3
import collections
import matplotlib.pyplot as plt
from scipy.stats import chi2

PoseWithCov = collections.namedtuple('PoseWithCov', 'pose cov')

tau = np.array([0,0,0,0,0,0])
sigma = 0.03

# Initial state (mean)
r = 1
T0 = se3.exp(tau)
E0 = np.zeros((6,6))
T0[0,3] = r

# Uncertainty on in-plane rotation
def get_uncertainty(sigma):
  sigma_sq = np.power(sigma, 2)
  E = np.diag(sigma_sq)
  return sigma, E

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
  
  
  
# Simulate 1000 samples for 100 steps
N = 1000
K = 100
samples = [[PoseWithCov for j in range(K)] for i in range(N)]
for i in range(N):
  Tk = np.eye(4,4)
  Ek = E0
  samples[i][0] = PoseWithCov(pose=Tk, cov=Ek) 
  random_sigs = np.random.uniform(low=-sigma, high=sigma, size=(K))
  for j in range(1, K):
    # Random sampling
    Psi, E = get_uncertainty(np.array([0.1, 0, 0, 0, 0, random_sigs[j]]))
    Adj = se3.adj(Tk)
    # Update mean
    Tk = np.linalg.multi_dot((Tk, se3.exp(Psi), T0))
    # Update covariances
    Ek = samples[i][j-1].cov + np.linalg.multi_dot((Adj, E, np.transpose(Adj)))
    samples[i][j] = PoseWithCov(pose=Tk, cov=Ek)

# Draw
plt.figure(0)
for i in range(N-1, N):
  x = []
  y = []
  for j in range(K):
    pose2d = samples[i][j].pose[0:2,3]
    cov2d = samples[i][j].cov[0:2, 0:2]
    x.append(pose2d[0])
    y.append(pose2d[1])
    xe, ye = confidence_ellipse(pose2d, cov2d)
    plt.plot(xe, ye)
  plt.scatter(x,y)
plt.xlim((-5, 105))
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.grid()
plt.show()

