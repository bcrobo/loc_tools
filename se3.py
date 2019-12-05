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
    A = np.sin(angle) / angle
    B = (1 - np.cos(angle)) / (angle * angle)
    Vinv = Vinv + np.dot((1/ angle*angle), (1 - (A / (2*B)) * np.power(axis, 2)))
  return Vinv

def exp(tau):
  u = tau[0:3]
  w = tau[3:6]
  angle, axis = so3.rotv_to_angle_axis(w)
  R = so3.rodrigues(angle, axis)
  Vt = np.dot(V(angle,axis), u)
  T = np.zeros((4,4))
  T[0:3, 0:3] = R
  T[0:3, 3] = Vt
  return T

def log(T):
  angle, axis = so3.log_map_rot_matx(T[0:3, 0:3])
  t = T[0:3, 3]
  V_inv = Vinv(angle, axis)
  tau = np.zeros(6)
  tau[0:3] = np.dot(V_inv, t)
  tau[3:7] = angle * axis
  return tau

def adj(T):
  Adj = np.zeros((6,6))
  R = T[0:3, 0:3]
  t = T[0:3, 3]
  Adj[0:3, 0:3] = R
  Adj[0:3, 3:6] = np.dot(so3.skew(t), R)
  Adj[3:6, 3:6] = R
  return Adj
