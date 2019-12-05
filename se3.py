#!/usr/bin/python

import so3
import numpy as np

def V(angle, axis):
  if angle * angle == 0:
    return np.eye(3,3)
  else:
    A = (1 - np.cos(angle)) / (angle * angle)
    B = (angle - np.sin(angle)) / (angle * angle * angle)
    return np.eye(3,3) + np.dot(A, so3.skew(axis)) + np.dot(B, np.power(so3.skew(axis), 2))

def Vinv(angle, axis):
  Vinv = np.eye(3,3) - np.dot(0.5, so3.skew(axis))
  if angle * angle != 0:
    Vinv = Vinv + (1/ (angle * angle)) * (1 - 

def exp(tau):
  u = tau[0:3]
  rvec = tau[3:7]
  angle, axis = so3.rotv_to_angle_axis(rvec)
  R = so3.rodrigues(angle, axis)
  Vt = np.dot(V(angle,axis), u)
  T = np.zeros((4,4))
  T[0:3, 0:3] = R
  T[0:3, 3] = Vt
  return T

def log(T):
  R = so3.log_map_rot_matx(T[0:3, 0,3])
  angle, axis = se3.log_map_rot_matx(R)
  
