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

from rosbag import Bag
from scipy import interpolate
from scipy.stats import chi2

gnss_topicname = ["/gps_position"]
lidar_topicnames = ["/lidar_pose", "/lms_pose", "/vlp_pose"]
fusion_topicname = ["/vehicle_pose"]

vhc2gps_x = 0.642
yaw_threshold = np.radians(2)

'''
BAG DATA
'''
class Trajectory2D:
    def __init__(self):
        self.time = np.empty([0])
        self.x = np.empty([0])
        self.y = np.empty([0])
        self.yaw = np.empty([0])
        self.cov = np.empty((0,2,2))

    def size(self):
        return self.x.size

class BagData:
    def __init__(self):
        self.gnss_traj = Trajectory2D()
        self.fusion_traj = Trajectory2D()
        self.lidar_traj = {l:Trajectory2D() for l in lidar_topicnames}


class Statistics:
    def __init__(self):
        self.lidar_distances = {l:np.empty([0]) for l in lidar_topicnames}
        self.lidar_interp_trajectories = {l:Trajectory2D() for l in lidar_topicnames}
        self.gnss2vhc_traj = Trajectory2D()

class BagBundle:
    def __init__(self):
        self.bag_data = BagData()
        self.statistics = Statistics()

class ClosestPointOptimizer:
    def __init__(self, tck, XY, init_guess = np.empty([0])):
        # spline parameters (coefficients, ...)
        self.tck = tck
        # 2D array of 2xN
        self.XY = XY
        if not init_guess.size:
            # Auto computed parameters for lidar trajectory
            self.init_guess = self.compute_parameters(self.XY)
        else:
            self.init_guess = init_guess
        # Current point to optimize
        self.p = np.empty([0])
        self.closest_x = np.empty([0])
        self.closest_y = np.empty([0])

        # Array for all distances
        self.distances = np.empty([0])

    def compute_parameters(self, XY):
        M = XY.shape[1] # Cols
        v = np.array([0])

        for i in xrange(1, M):
            vi = v[i-1] + ssd.euclidean(XY[:,i], XY[:,i-1])
            v = np.hstack((v, np.array([vi])))

        u = np.empty([0])
        for i in xrange(0, M):
            ui = v[i] / v[M - 1]
            u = np.hstack((u, np.array([ui])))
        return u
    
    def distanceToPoint(self, u):
        s = interpolate.splev(u, self.tck)
        return ssd.euclidean(self.p, np.array([s[0][0], s[1][0]]))

    def optimize(self):
        M = self.XY.shape[1] # Cols
        self.distances = np.empty([0])
        for i in xrange(0,M):
            self.p = self.XY[:,i]

            minimum = so.fmin(self.distanceToPoint, 0)
            closest_point = interpolate.splev(minimum, self.tck)
            dist = ssd.euclidean(closest_point, self.p)

            self.closest_x = np.hstack((self.closest_x, np.array([closest_point[0][0]])))
            self.closest_y = np.hstack((self.closest_y, np.array([closest_point[1][0]])))
            self.distances = np.hstack((self.distances, np.array([dist])))

def filterYawValuesGreaterThan(yaws, max_angle):
    
    prev_yaw = 0
    valid_yaws = np.empty([0])
    valid_indexes = []

    for i in xrange(len(yaws)):
        yaw_2pi = yaws[i] + np.pi
        if abs(yaw_2pi - prev_yaw) < max_angle and i != 0:
            valid_yaws = np.hstack((valid_yaws, [yaws[i]]))
            valid_indexes.append(i)
        prev_yaw = yaw_2pi

    return valid_indexes, valid_yaws

def moveGnssToVhc(gnss_trajectory):
    length = gnss_trajectory.size()

    # Not enough points
    if length < 3:
        print("- Error moving gnss trajectory to vhc got less than 3 pts")
        sys.exit(-1)

    # Compute yaws from dx and dy
    yaws = [] 
    for i in xrange(length):
        if i:
            dx = gnss_trajectory.x[i] - gnss_trajectory.x[i-1]
            dy = gnss_trajectory.y[i] - gnss_trajectory.y[i-1]
            yaw = math.atan2(dy,dx)
            yaws.append(yaw)

    # Filter yaw changes > 5 degrees
    vhc_traj = Trajectory2D()
    valid_yaw_indexes, valid_yaw = filterYawValuesGreaterThan(yaws, yaw_threshold)

    # Compute gnss position in vehicle frame
    for i in xrange(len(valid_yaw)):

        idx = valid_yaw_indexes[i]
        x = gnss_trajectory.x[idx] - vhc2gps_x * math.cos(valid_yaw[i])
        y = gnss_trajectory.y[idx] - vhc2gps_x * math.sin(valid_yaw[i])

        vhc_traj.x = np.hstack((vhc_traj.x, np.array([x])))
        vhc_traj.y = np.hstack((vhc_traj.y, np.array([y])))
        vhc_traj.yaw = np.hstack((vhc_traj.yaw, np.array([valid_yaw[i]])))

        vhc_traj.time = np.hstack((vhc_traj.time, np.array([gnss_trajectory.time[idx]])))
        vhc_traj.cov = np.vstack((vhc_traj.cov, gnss_trajectory.cov[idx,:,:].reshape(1,2,2)))

    return vhc_traj

def syncLidarTrajectory(gnss_traj, lidar_traj):
    first_time = gnss_traj.time[0]
    last_time = gnss_traj.time[-1]

    sync_lidar_traj = Trajectory2D()
    for i in xrange(0,lidar_traj.size()):
        if lidar_traj.time[i] >= first_time and lidar_traj.time[i] <= last_time:
            sync_lidar_traj.x = np.hstack((sync_lidar_traj.x, np.array([lidar_traj.x[i]])))
            sync_lidar_traj.y = np.hstack((sync_lidar_traj.y, np.array([lidar_traj.y[i]])))
            sync_lidar_traj.time = np.hstack((sync_lidar_traj.time, np.array([lidar_traj.time[i]])))
    return sync_lidar_traj


def computeDist(gnss_vhc_traj, lidar_traj):

    # Synchronize lidar trajectory to keep only record on the gnss trajectory
    lidar_sync_traj = syncLidarTrajectory(gnss_vhc_traj, lidar_traj)

    # Fit spline to gnss trajectory
    tck, u = interpolate.splprep([gnss_vhc_traj.x, gnss_vhc_traj.y], s=0)
    xnew, ynew = interpolate.splev(u, tck)

    # Find closest point on interpolated trajectory
    xy = np.transpose( np.array([xnew, ynew]) )
    gnss_vhc_interp = geom.LineString(xy)
    point_on_traj_x = np.empty([0])
    point_on_traj_y = np.empty([0])
    distances = np.empty([0])
    for i in xrange(0,lidar_sync_traj.size()):
        p = geom.Point(lidar_sync_traj.x[i], lidar_sync_traj.y[i])
        distances = np.hstack((distances, np.array([p.distance(gnss_vhc_interp)])))
        point_on_traj = gnss_vhc_interp.interpolate(gnss_vhc_interp.project(p))
        point_on_traj_x = np.hstack((point_on_traj_x, np.array([point_on_traj.x])))
        point_on_traj_y = np.hstack((point_on_traj_y, np.array([point_on_traj.y])))

#    # Display
#    plt.plot(xnew, ynew)
#    plt.scatter(lidar_sync_traj.x, lidar_sync_traj.y, c='r')
#    plt.scatter(gnss_vhc_traj.x, gnss_vhc_traj.y, c='m')
#    for i in xrange(0, lidar_sync_traj.size()):
#        plt.plot(np.array([point_on_traj_x[i], lidar_sync_traj.x[i]]), np.array([point_on_traj_y[i], lidar_sync_traj.y[i]]), 'k-')
#    plt.axis('equal')
#    plt.show()
    return lidar_sync_traj, distances

def loadBagData(bag):

    bdata = BagData()

    # Load gnss data
    for topic, msg, t in bag.read_messages(topics=gnss_topicname):
        bdata.gnss_traj.time = np.hstack((bdata.gnss_traj.time, np.array([msg.header.stamp.to_sec()])))
        bdata.gnss_traj.x = np.hstack((bdata.gnss_traj.x, np.array([msg.pose.pose.position.x])))
        bdata.gnss_traj.y = np.hstack((bdata.gnss_traj.y, np.array([msg.pose.pose.position.y])))
        xx = msg.pose.covariance[0]
        xy = msg.pose.covariance[1]
        yx = msg.pose.covariance[6]
        yy = msg.pose.covariance[7]
        cov = np.array([[xx, xy],[yx, yy]])
        bdata.gnss_traj.cov = np.vstack((bdata.gnss_traj.cov, cov.reshape(1,2,2)))

    if bdata.gnss_traj.size() < 3:
        print("- Error : GNSS trajectory too small or non-existent in bag")
        sys.exit(-1)

    # Load lidar data
    for topic, msg, t in bag.read_messages(topics=lidar_topicnames):
        bdata.lidar_traj[topic].time = np.hstack((bdata.lidar_traj[topic].time, np.array([msg.header.stamp.to_sec()])))
        bdata.lidar_traj[topic].x = np.hstack((bdata.lidar_traj[topic].x, np.array([msg.pose.pose.position.x])))
        bdata.lidar_traj[topic].y = np.hstack((bdata.lidar_traj[topic].y, np.array([msg.pose.pose.position.y])))
        xx = msg.pose.covariance[0]
        xy = msg.pose.covariance[1]
        yx = msg.pose.covariance[6]
        yy = msg.pose.covariance[7]
        cov = np.array([[xx, xy],[yx, yy]])
        bdata.lidar_traj[topic].cov = np.vstack((bdata.lidar_traj[topic].cov, cov.reshape(1,2,2)))

    # Load fusion data
    for topic, msg, t in bag.read_messages(topics=fusion_topicname):
        bdata.fusion_traj.time = np.hstack((bdata.fusion_traj.time, np.array([msg.header.stamp.to_sec()])))
        bdata.fusion_traj.x = np.hstack((bdata.fusion_traj.x, np.array([msg.pose.pose.position.x])))
        bdata.fusion_traj.y = np.hstack((bdata.fusion_traj.y, np.array([msg.pose.pose.position.y])))
        xx = msg.pose.covariance[0]
        xy = msg.pose.covariance[1]
        yx = msg.pose.covariance[6]
        yy = msg.pose.covariance[7]
        cov = np.array([[xx, xy],[yx, yy]])
        bdata.fusion_traj.cov = np.vstack((bdata.fusion_traj.cov, cov.reshape(1,2,2)))

    return bdata

def computeEllipse(cov,mass_level=0.99):
        eig_vec,eig_val,u = np.linalg.svd(cov)
        #make sure 0th eigenvector has positive x-coordinate
        if eig_vec[0][0] < 0:
            eig_vec[0] *= -1
        semi_maj = np.sqrt(eig_val[0])
        semi_min = np.sqrt(eig_val[1])

        distances = np.linspace(0,20,20001)
        chi2_cdf = chi2.cdf(distances,df=2)
	multiplier = np.sqrt(distances[np.where(np.abs(chi2_cdf-mass_level)==np.abs(chi2_cdf-mass_level).min())[0][0]])
        semi_maj *= multiplier
        semi_min *= multiplier

	phi = np.arccos(np.dot(eig_vec[0],np.array([1,0])))
        if eig_vec[0][1] < 0 and phi > 0:
            phi *= -1

       	return semi_maj, semi_min, phi

def plotEllipse(semimaj=1,semimin=1,phi=0,xc=0,yc=0,theta_num=1e3,fill=False,data_out=False,ax=None,a_color='b'): 
        #generate data for ellipse
        theta = np.linspace(0,2*np.pi,theta_num)
        r = 1 / np.sqrt((np.cos(theta))**2 + (np.sin(theta))**2)
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        data = np.array([x,y])
        S = np.array([[semimaj,0],[0,semimin]])
        R = np.array([[np.cos(phi),-np.sin(phi)],[np.sin(phi),np.cos(phi)]])
        T = np.dot(R,S)
        data = np.dot(T,data)
        data[0] += xc
        data[1] += yc

         # Output data
        if data_out == True:
                return data

        # Plot
	return_fig = False
        if ax is None:
                return_fig = True
                fig,ax = plt.subplots()

        ax.plot(data[0],data[1],color=a_color,linestyle='-')

        if fill == True:
                ax.fill(data[0],data[1],**fill_kwargs)

        if return_fig == True:
                return fig


def plotTrajectoryEllipses(traj, mass_level=0.99, theta_num=1e3, fill=False, data_out=False,ax=None, a_color='b'):
    for i in xrange(0,traj.size()):
        semi_maj, semi_min, phi = computeEllipse(traj.cov[i,:,:], mass_level) 
        plotEllipse(semi_maj, semi_min, phi, traj.x[i], traj.y[i], theta_num,fill,data_out,ax,a_color)

'''
PLOTTING FUNCTIONS
'''
def plotDistances(bags_bundle):
    plt.figure(1)
    plt.subplot(111)

    patches = []
    colors = []
    for bag, bag_bundle in bags_bundle.iteritems():
        for lidar in lidar_topicnames:
            if bag_bundle.bag_data.lidar_traj[lidar].size() > 0:
                p = plt.plot(bag_bundle.statistics.lidar_interp_trajectories[lidar].time, bag_bundle.statistics.lidar_distances[lidar])
                bag_name = bag.split("/")[-1]
                colors.append(p[0].get_color())
                mean_dist = np.mean(bag_bundle.statistics.lidar_distances[lidar])
                patches.append( mpatches.Patch(color=p[0].get_color(), label=lidar + " on " + bag_name + " mean: " + str(mean_dist)) )

    plt.xlabel("Time (s)")
    plt.ylabel("Distance (m)")
    plt.grid(True)
    plt.legend(handles=patches)
    plt.title("Distance between lidars and GNSS")
    plt.show()

def plotTrajectories(bags_bundle, with_ellipses=False):
    plt.figure(1)
    plt.subplot(111)

    patches = []
    colors = []
    for bag, bag_bundle in bags_bundle.iteritems():
        bag_name = bag.split("/")[-1]
        for lidar in lidar_topicnames:
            if bag_bundle.bag_data.lidar_traj[lidar].size() > 0:
                p = plt.plot(bag_bundle.bag_data.lidar_traj[lidar].x, bag_bundle.bag_data.lidar_traj[lidar].y)
                colors.append(p[0].get_color())
                patches.append( mpatches.Patch(color=p[0].get_color(), label=lidar + " on " + bag_name))

                if with_ellipses:
                    plotTrajectoryEllipses(bag_bundle.bag_data.lidar_traj[lidar], 0.99, 1e3,False,False,plt,p[0].get_color())

        p = plt.plot(bag_bundle.statistics.gnss2vhc_traj.x, bag_bundle.statistics.gnss2vhc_traj.y)
        if with_ellipses:
            plotTrajectoryEllipses(bag_bundle.statistics.gnss2vhc_traj, 0.99, 1e3,False,False,plt,p[0].get_color())
        patches.append( mpatches.Patch(color=p[0].get_color(), label=gnss_topicname[0] + " on " + bag_name))


    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.grid(True)
    plt.axis('equal')
    plt.legend(handles=patches)
    plt.title("GNSS and Lidars trajectories")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare lidar data with gnss trajectory')
    parser.add_argument("-b", "--bag", required=True, nargs="*", help="input bag files")
    parser.add_argument("-d", "--distance", default=False, action='store_true', help="Plot distance from lidars to gnss trajectory")
    parser.add_argument("-t", "--trajectory", default=False, action='store_true', help="Plot trajectories from lidar and gnss")
    args = parser.parse_args()

    bags_bundle = {b:BagBundle() for b in args.bag}
    for b in args.bag:
        with Bag(b, 'r') as bag:
            # Load all bag data
            bdata = loadBagData(bag)
            stats = Statistics()

            # Move gnss to vehicle frame (and remove first / last record due to wrong yaw)
            gnss_vhc_traj = moveGnssToVhc(bdata.gnss_traj)
            # Save new gnss trajectory
            stats.gnss2vhc_traj = gnss_vhc_traj

            for lidar_topic in lidar_topicnames:
                if bdata.lidar_traj[lidar_topic].size() > 0:
                    lidar_sync_traj, distances = computeDist(gnss_vhc_traj, bdata.lidar_traj[lidar_topic])
                    stats.lidar_distances[lidar_topic] = distances
                    stats.lidar_interp_trajectories[lidar_topic] = lidar_sync_traj


            bags_bundle[b].bag_data = bdata 
            bags_bundle[b].statistics = stats 

    if args.distance:
        plotDistances(bags_bundle)
    if args.trajectory:
        plotTrajectories(bags_bundle)
