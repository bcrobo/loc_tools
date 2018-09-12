#! /usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.spatial.distance as ssd
import scipy.optimize as so
import shapely.geometry as geom
import tf
import pyproj

from rosbag import Bag
from scipy import interpolate
from scipy.stats import chi2
from yaml import load # extract params
import os

gnss_topicname = ["/gps_position", "/gps_bestpos"]
lidar_topicnames = ["/lidar_pose", "/lms_pose", "/vlp_pose"]
fusion_topicname = ["/vehicle_pose"]
fusion_details_topicname = ["/fusion/fusion_details"]
attitude_topicname = ["/vehicle_attitude"]
tf_topicname = ["/tf"]

gps2vhc_x = -0.29
yaw_threshold = np.radians(2)

longlat = pyproj.Proj("+proj=longlat +datum=WGS84")

class Traj:
  def __init__(self):
    self.time = np.array([])
    self.x = np.array([])
    self.y = np.array([])
    self.geometry = geom.LineString()

  def distance(self, x, y):
    p = geom.Point(x, y)
    return p.distance(self.geometry)

def read_gps_origin(bag):
  messages = [msg for _topic, msg, _t in bag.read_messages(topics="/rosparam_dump")]
  if len(messages):
    msg = messages[0]
    values = load(msg.data)
    if "origin_latitude" in values and "origin_longitude" in values:
      return values["origin_latitude"], values["origin_longitude"]
  return 0.0, 0.0

def fit_spline(x, y):
  tck, u = interpolate.splprep([x, y], s=0)
  xnew, ynew = interpolate.splev(u, tck)
  xy = np.transpose( np.array([xnew, ynew]) )
  return geom.LineString(xy)

def filter_yaws(x, y, yaw_tolerance):
  # Compute yaws from dx and dy
  yaws = [] 
  for i in xrange(x.size):
      if i:
          dx = x[i] - x[i-1]
          dy = y[i] - y[i-1]
          yaw = math.atan2(dy,dx) + np.pi
          yaws.append(yaw)

 
  # Filter yaw changes > 2 degrees
  valid_yaws = np.empty([0])
  valid_indexes = []

  for i in xrange(len(yaws)):
    if i:
      yaw_diff = yaws[i] - yaws[i-1]
      if abs(yaw_diff) < yaw_tolerance:
        valid_yaws = np.hstack((valid_yaws, np.array([yaws[i]])))
        valid_indexes.append(i)
 
  return valid_indexes, valid_yaws

def filtered_vehicle_in_gnss(x_vhc, y_vhc):
  indexes, yaws = filter_yaws(x_vhc, y_vhc, np.radians(2))

  x_gnss = np.array([])
  y_gnss = np.array([])
  for i in xrange(len(yaws)):

      idx = indexes[i]
      x = x_vhc[idx] - gps2vhc_x * math.cos(yaws[i])
      y = y_vhc[idx] - gps2vhc_x * math.sin(yaws[i])

      x_gnss = np.concatenate([x_gnss, np.array([x])])
      y_gnss = np.concatenate([y_gnss, np.array([y])])

  return x_gnss, y_gnss 

def filtered_track(x_track, y_track, time, x_leverarm=0.0):
  size = x_track.size

  # Not enough points
  if size < 3:
      print("- Error moving gnss trajectory to vhc got less than 3 pts")
      sys.exit(-1)

  indexes, yaws = filter_yaws(x_track, y_track, np.radians(2))

  # Compute gnss position in vehicle frame
  x_filtered = np.array([])
  y_filtered = np.array([])
  time_filtered = np.array([])
  dx = dy = 0.0
  for i in xrange(len(yaws)):

      idx = indexes[i]
      dx = x_leverarm * math.cos(yaws[i])
      dy = x_leverarm * math.sin(yaws[i])
      #if x_leverarm:
        #print "dx: " + str(dx) + " dy: " + str(dy)
      x = x_track[idx] - dx
      y = y_track[idx] - dy

      x_filtered = np.concatenate([x_filtered, np.array([x])])
      y_filtered = np.concatenate([y_filtered, np.array([y])])
      time_filtered = np.concatenate([time_filtered, np.array([time[idx]])])
  return x_filtered, y_filtered, time_filtered
    
def load_lat_long_traj(bag, topic, cartesian, with_fit=False):

  coord_x = np.array([])
  coord_y = np.array([])
  time = np.array([])
  for topic, msg, t in bag.read_messages(topics=topic):
    if msg.longitude and msg.latitude and msg.longitude < 300 and msg.latitude < 300:
      #print "longi: " + str(msg.longitude) + " lati: " + str(msg.latitude)
      x, y = pyproj.transform(longlat, cartesian, msg.longitude, msg.latitude)
      coord_x = np.concatenate([coord_x, np.array([x])])
      coord_y = np.concatenate([coord_y, np.array([y])])
      time = np.concatenate([time, np.array([t.to_sec()])])

  traj = Traj()
  if with_fit:
    x_gnss, y_gnss, time_gnss = filtered_track(coord_x, coord_y, time)
    traj.x = x_gnss
    traj.y = y_gnss
    traj.time = time_gnss
    traj_geometry = fit_spline(x_gnss, y_gnss)
    traj.geometry = traj_geometry
  else:
    traj.x = coord_x
    traj.y = coord_y
    traj.time = time 
  return traj

def load_pose_traj(bag, topic, x_leverarm=0.0):
  coord_x = np.array([])
  coord_y = np.array([])
  time = np.array([])
  for topic, msg, t in bag.read_messages(topics=topic):
    coord_x = np.concatenate([coord_x, np.array([msg.pose.pose.position.x])])
    coord_y = np.concatenate([coord_y, np.array([msg.pose.pose.position.y])])
    time = np.concatenate([time, np.array([t.to_sec()])])
    
  coord_x, coord_y, filtered_time = filtered_track(coord_x, coord_y, time, x_leverarm)

  traj = Traj()
  traj.x = coord_x
  traj.y = coord_y
  traj.time = filtered_time
  return traj 

def load_cm_bagfile(bag, cartesian):
  cm_traj = load_lat_long_traj(bag, "/planes", cartesian)
  return cm_traj
   
def load_em_bagfile(bag, cartesian):
  em_gnss_traj = load_lat_long_traj(bag, "/gps_bestpos", cartesian, with_fit=True)
  em_gnss_pos_traj = load_pose_traj(bag, "/gps_position")
  em_vhc_pos_traj = load_pose_traj(bag, "/vehicle_pose", x_leverarm=gps2vhc_x)
  return em_gnss_traj, em_gnss_pos_traj, em_vhc_pos_traj

def is_cm_bagfile(bag):
  for topic, info in bag.get_type_and_topic_info().topics.iteritems():
    if topic == "/reference":
      return True
  return False
 
 
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Compare lidar data with gnss trajectory')
  parser.add_argument("-b", "--bag", required=True, nargs="*", help="input bag files")

  args = parser.parse_args()

  cm_traj = Traj()
  em_gnss_traj = Traj()
  em_gnss_pos_traj = Traj()
  bags = []
  for b in args.bag:
    with Bag(b, 'r') as bag:
      if is_cm_bagfile(bag):
        bags.append(b)
      else:
        bags.insert(0, b)
        # Read origin of site
        origin_latitude, origin_longitude = read_gps_origin(bag)
        if origin_latitude and origin_longitude:
          cartesian = pyproj.Proj("+proj=tmerc +datum=WGS84 +ellps=WGS84 +k_0=1 +lon_0={:.8f} +lat_0={:.8f} +x_0=0 +y_0=0 +axis=enu +units=m".format(origin_longitude, origin_latitude))

  with Bag(bags[0], 'r') as bag:
    em_gnss_traj, em_gnss_pos_traj, em_vhc_pos_traj = load_em_bagfile(bag, cartesian)

  cm_trajs = {b:Traj() for b in bags[1:]}
  for b, cm_traj in cm_trajs.iteritems():
    with Bag(b, 'r') as bag:
      cm_trajs[b] = load_cm_bagfile(bag, cartesian)

  cm_traj_distances = {b:np.array([]) for b in bags[1:]}
  for b, cm_traj in cm_trajs.iteritems():
    distances_cm = np.array([])
    for i in xrange(cm_traj.x.size):
      dist = em_gnss_traj.distance(cm_traj.x[i], cm_traj.y[i])
      distances_cm = np.concatenate([distances_cm, np.array([dist])])
    cm_traj_distances[b] = distances_cm
    

  distances_em = np.array([])
  for i in xrange(em_vhc_pos_traj.x.size):
    dist = em_gnss_traj.distance(em_vhc_pos_traj.x[i], em_vhc_pos_traj.y[i])
    distances_em = np.concatenate([distances_em, np.array([dist])])

  fig = plt.figure(1)
  ## Plot trajectories
  tracks = fig.add_subplot(211)
  tracks.grid()
  patches = []
  colors = []
  for b, cm_traj in cm_trajs.iteritems():
    p = tracks.scatter(cm_traj.x, cm_traj.y, s=0.8)
    c = p.get_facecolor()[0]
    patches.append( mpatches.Patch(color=c, label="CM " + os.path.split(b)[-1]) )
    colors.append(c)
  p = tracks.scatter(em_gnss_traj.x, em_gnss_traj.y, s=0.8)
  c = p.get_facecolor()[0]
  patches.append( mpatches.Patch(color=c, label="EM RTK") )
  colors.append(c)
  p = tracks.scatter(em_vhc_pos_traj.x, em_vhc_pos_traj.y, s=0.8)
  c = p.get_facecolor()[0]
  patches.append( mpatches.Patch(color=c, label="EM Fusion") )
  colors.append(c)
  tracks.legend(handles=patches)
  tracks.set_aspect(aspect='equal')
  tracks.set_xlabel("X (m)")
  tracks.set_ylabel("Y (m)")
  tracks.set_title("Trajectories")
  # Plot distances
  dist = fig.add_subplot(212)
  dist.grid()
  patches = []
  i = 0
  for b, cm_dist in cm_traj_distances.iteritems():
    p = dist.scatter(cm_trajs[b].time, cm_dist, s=0.8, c=colors[i])
    patches.append(mpatches.Patch(color=colors[i], label="CM " + os.path.split(b)[-1] + " mean: " + str(np.mean(cm_dist))))
    i += 1

  p = dist.plot(em_vhc_pos_traj.time, distances_em, c=colors[-1])
  patches.append(mpatches.Patch(color=colors[-1], label="EM Fusion mean: " + str(np.mean(distances_em))))
  dist.legend(handles=patches)
  dist.set_xlabel("Time (s)")
  dist.set_ylabel("Orth. error (m)")
  dist.set_title("Orthogonal error with EM RTK")
  plt.show()
