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

from rosbag import Bag
from scipy import interpolate
from scipy.stats import chi2

gnss_topicname = ["/gps_position"]
lidar_topicnames = ["/lidar_pose", "/lms_pose", "/vlp_pose"]
fusion_topicname = ["/vehicle_pose"]
#fusion_details_topicname = ["/localization/fusion_details"]
fusion_details_topicname = ["/fusion/fusion_details"]
attitude_topicname = ["/vehicle_attitude"]
tf_topicname = ["/tf"]

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

class Trajectory3D:
    def __init__(self):
        self.child_frame = ""
        self.time = np.empty([0])
        self.x = np.empty([0])
        self.y = np.empty([0])
        self.z = np.empty([0])
        self.roll = np.empty([0])
        self.pitch = np.empty([0])
        self.yaw = np.empty([0])

class BagData:
    def __init__(self, transforms={}):
        self.gnss_traj = Trajectory2D()
        self.fusion_traj = Trajectory2D()
        self.lidar_traj = {l:Trajectory2D() for l in lidar_topicnames}
        self.roll = np.empty([0])
        self.pitch = np.empty([0])
        self.yaw = np.empty([0])
        self.attitude_time = np.empty([0])
        self.tf_frames = {parent:Trajectory3D() for parent,child in transforms.iteritems()}

class Statistics:
    def __init__(self):
        self.lidar_distances = {l:np.empty([0]) for l in lidar_topicnames}
        self.lidar_interp_trajectories = {l:Trajectory2D() for l in lidar_topicnames}
        self.fusion_distances = np.empty([0])
        self.fusion_interp_traj = Trajectory2D()
        self.gnss2vhc_traj = Trajectory2D()
        self.mahalanobis = {l:np.empty([0]) for l in lidar_topicnames + gnss_topicname}
        self.mahalanobis_time = {l:np.empty([0]) for l in lidar_topicnames + gnss_topicname}
        self.mir = {l:np.empty([0]) for l in lidar_topicnames}
        self.mir_time = {l:np.empty([0]) for l in lidar_topicnames}

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

#    # Display ortho distances
#    plt.plot(xnew, ynew)
#    plt.scatter(lidar_sync_traj.x, lidar_sync_traj.y, c='r')
#    plt.scatter(gnss_vhc_traj.x, gnss_vhc_traj.y, c='m')
#    for i in xrange(0, lidar_sync_traj.size()):
#        plt.plot(np.array([point_on_traj_x[i], lidar_sync_traj.x[i]]), np.array([point_on_traj_y[i], lidar_sync_traj.y[i]]), 'k-')
#    plt.axis('equal')
#    plt.show()
    return lidar_sync_traj, distances

def sensorNameToTopic(sensor_name):
    r = [x for x in lidar_topicnames if sensor_name in x]
    if len(r) > 0:
        return r[0]

    r = [x for x in gnss_topicname if sensor_name in x]
    if len(r) > 0:
        return r[0]

    return ""

def convertToDictionnary(transforms):
    if len(transforms) % 2 != 0:
        print "- Error even number of tf transforms"
        return

    transforms_dict = {}
    for i in xrange(0, len(transforms), 2):
        parent_frame = transforms[i]
        transforms_dict[parent_frame] = transforms[i+1]
    return transforms_dict

def loadBagData(bag, transforms):

    # Read tf transforms if needed
    parent_child_frames = {}
    if transforms:
        parent_child_frames.update(convertToDictionnary(transforms))

    bdata = BagData(parent_child_frames)
    stats = Statistics()

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
        stats.mir[topic] = np.hstack((stats.mir[topic], np.array([msg.detail.matched_impact_ratio])))
        stats.mir_time[topic] = np.hstack((stats.mir_time[topic], np.array([msg.header.stamp.to_sec()])))

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
        r, p, y = tf.transformations.euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
        bdata.yaw = np.hstack((bdata.yaw, np.degrees(y)))

    # Load fusion details
    for topic, msg, t in bag.read_messages(topics=fusion_details_topicname):
        sensor_topic = sensorNameToTopic(msg.sensor_name)
        if sensor_topic:
            stats.mahalanobis[sensor_topic] = np.hstack((stats.mahalanobis[sensor_topic], np.array([msg.mahalanobis_dist])))
            stats.mahalanobis_time[sensor_topic] = np.hstack((stats.mahalanobis_time[sensor_topic], np.array([msg.header.stamp.to_sec()])))

    # Load roll and pitch
    for topic, msg, t in bag.read_messages(topics=attitude_topicname):
        bdata.attitude_time = np.hstack((bdata.attitude_time, np.array([msg.header.stamp.to_sec()])))
        r, p, y = tf.transformations.euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
        r = np.degrees(r)
        p = np.degrees(p)
        bdata.roll = np.hstack((bdata.roll, r))
        bdata.pitch = np.hstack((bdata.pitch, p))

    # Load tf transformations if needed
    if transforms:
        for topic, msg, t in bag.read_messages(topics=tf_topicname):
            
            asked = False
            for transform in msg.transforms:
                parent_frame = transform.header.frame_id
                child_frame = transform.child_frame_id

                if parent_frame in parent_child_frames and child_frame == parent_child_frames[parent_frame]:
                    bdata.tf_frames[parent_frame].child_frame = child_frame
                    bdata.tf_frames[parent_frame].time = np.hstack((bdata.tf_frames[parent_frame].time, np.array([transform.header.stamp.to_sec()])))
                    bdata.tf_frames[parent_frame].x = np.hstack((bdata.tf_frames[parent_frame].x, np.array([transform.transform.translation.x])))
                    bdata.tf_frames[parent_frame].y = np.hstack((bdata.tf_frames[parent_frame].y, np.array([transform.transform.translation.y])))
                    bdata.tf_frames[parent_frame].z = np.hstack((bdata.tf_frames[parent_frame].z, np.array([transform.transform.translation.z])))
                    r, p, y = tf.transformations.euler_from_quaternion([transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w])
                    bdata.tf_frames[parent_frame].roll = np.hstack((bdata.tf_frames[parent_frame].roll, np.array([np.degrees(r)]))) 
                    bdata.tf_frames[parent_frame].pitch = np.hstack((bdata.tf_frames[parent_frame].pitch, np.array([np.degrees(p)]))) 
                    bdata.tf_frames[parent_frame].yaw = np.hstack((bdata.tf_frames[parent_frame].yaw, np.array([np.degrees(y)])))

    return bdata, stats

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
def plotDistances(bags_bundle, with_fusion=False):
    plt.figure(1)
    plt.subplot(111)

    patches = []
    colors = []
    for bag, bag_bundle in bags_bundle.iteritems():
        bag_name = bag.split("/")[-1]
        for lidar in lidar_topicnames:
            if bag_bundle.bag_data.lidar_traj[lidar].size() > 0:
                p = plt.plot(bag_bundle.statistics.lidar_interp_trajectories[lidar].time, bag_bundle.statistics.lidar_distances[lidar])
                colors.append(p[0].get_color())
                mean_dist = np.mean(bag_bundle.statistics.lidar_distances[lidar])
                patches.append( mpatches.Patch(color=p[0].get_color(), label=lidar + " on " + bag_name + " mean: " + str(mean_dist)) )

        for fusion_topic in fusion_topicname:
            if with_fusion:
                p = plt.plot(bag_bundle.statistics.fusion_interp_traj.time, bag_bundle.statistics.fusion_distances)
                colors.append(p[0].get_color())
                mean_dist = np.mean(bag_bundle.statistics.fusion_distances)
                patches.append( mpatches.Patch(color=p[0].get_color(), label=fusion_topic + " on " + bag_name + " mean: " + str(mean_dist)) )

    plt.xlabel("Time (s)")
    plt.ylabel("Distance (m)")
    plt.grid(True)
    plt.legend(handles=patches)
    plt.title("Distance between lidars and GNSS")
    plt.show()

def plotMahalanobis(bags_bundle):
    plt.figure(1)
    plt.subplot(111)

    patches = []
    colors = []
    for bag, bag_bundle in bags_bundle.iteritems():
        for (topic, md), (topic, time) in zip(bag_bundle.statistics.mahalanobis.iteritems(), bag_bundle.statistics.mahalanobis_time.iteritems()):
            if md.size > 0:
                p = plt.plot(time, md)
                bag_name = bag.split("/")[-1]
                colors.append(p[0].get_color())
                mean_md = np.mean(md)
                patches.append( mpatches.Patch(color=p[0].get_color(), label=topic + " on " + bag_name + " mean: " + str(mean_md)) )

    plt.xlabel("Time (s)")
    plt.ylabel("Mahalanobis distance")
    plt.grid(True)
    plt.legend(handles=patches)
    plt.title("Mahalanobis distances")
    plt.show()

def plotMir(bags_bundle):
    plt.figure(1)
    plt.subplot(111)

    patches = []
    colors = []
    for bag, bag_bundle in bags_bundle.iteritems():
        for (topic, mir), (topic, time) in zip(bag_bundle.statistics.mir.iteritems(), bag_bundle.statistics.mir_time.iteritems()):
            if mir.size > 0:
                p = plt.plot(time, mir)
                bag_name = bag.split("/")[-1]
                colors.append(p[0].get_color())
                mean_mir = np.mean(mir)
                patches.append( mpatches.Patch(color=p[0].get_color(), label=topic + " on " + bag_name + " mean: " + str(mean_mir)) )

    plt.xlabel("Time (s)")
    plt.ylabel("Mean Impact Ratio")
    plt.grid(True)
    plt.legend(handles=patches)
    plt.title("Mean Impact Ratio")
    plt.show()

def plotAttitude(bags_bundle, with_yaw=False):
    plt.figure(1)
    plt.subplot(111)

    patches = []
    colors = []
    for bag, bag_bundle in bags_bundle.iteritems():
        bag_name = bag.split("/")[-1]
        # Roll
        if bag_bundle.bag_data.roll.size > 0:
            p = plt.plot(bag_bundle.bag_data.attitude_time, bag_bundle.bag_data.roll)
            colors.append(p[0].get_color())
            patches.append( mpatches.Patch(color=p[0].get_color(), label="Roll on " + bag_name) )

        # Pitch
        if bag_bundle.bag_data.pitch.size > 0:
            p = plt.plot(bag_bundle.bag_data.attitude_time, bag_bundle.bag_data.pitch)
            colors.append(p[0].get_color())
            patches.append( mpatches.Patch(color=p[0].get_color(), label="Pitch on " + bag_name) )

        # Yaw
        if with_yaw:
            if bag_bundle.bag_data.yaw.size > 0:
                p = plt.plot(bdata.fusion_traj.time, bag_bundle.bag_data.yaw)
                colors.append(p[0].get_color())
                patches.append( mpatches.Patch(color=p[0].get_color(), label="Yaw on " + bag_name) )

    plt.xlabel("Time (s)")
    plt.ylabel("Angle roll, pitch (deg)")
    plt.grid(True)
    plt.legend(handles=patches)
    plt.title("Vehicle attitude")
    plt.show()

def plotTransformsAttitude(bags_bundle, with_yaw=False):
    plt.figure(1)
    plt.subplot(111)

    patches = []
    colors = []
    for bag, bag_bundle in bags_bundle.iteritems():
        bag_name = bag.split("/")[-1]
        for frame_id, traj3d in bag_bundle.bag_data.tf_frames.iteritems():
            # Roll
            if traj3d.x.size > 0:
                p = plt.plot(traj3d.time, traj3d.roll)
                colors.append(p[0].get_color())
                patches.append( mpatches.Patch(color=p[0].get_color(), label=frame_id + " to "  + traj3d.child_frame + " roll on " + bag_name) )

            # Pitch
            if traj3d.y.size > 0:
                p = plt.plot(traj3d.time, traj3d.pitch)
                colors.append(p[0].get_color())
                patches.append( mpatches.Patch(color=p[0].get_color(), label=frame_id + " to "  + traj3d.child_frame + " pitch on " + bag_name) )

            if with_yaw:
                # Yaw
                if traj3d.y.size > 0:
                    p = plt.plot(traj3d.time, traj3d.yaw)
                    colors.append(p[0].get_color())
                    patches.append( mpatches.Patch(color=p[0].get_color(), label=frame_id + " to "  + traj3d.child_frame + " yaw on " + bag_name) )


    plt.xlabel("Time (s)")
    plt.ylabel("Angle roll, pitch (deg)")
    plt.grid(True)
    plt.legend(handles=patches)
    plt.title("TF frames attitude")
    plt.show()


def plotTrajectories(bags_bundle, with_ellipses=False, with_fusion=False):
    plt.figure(1)
    plt.subplot(111)

    patches = []
    colors = []
    for bag, bag_bundle in bags_bundle.iteritems():
        bag_name = bag.split("/")[-1]
        # Plot each lidar trajectories
        for lidar in lidar_topicnames:
            if bag_bundle.bag_data.lidar_traj[lidar].size() > 0:
                p = plt.plot(bag_bundle.bag_data.lidar_traj[lidar].x, bag_bundle.bag_data.lidar_traj[lidar].y)
                colors.append(p[0].get_color())
                patches.append( mpatches.Patch(color=p[0].get_color(), label=lidar + " on " + bag_name))

                if with_ellipses:
                    plotTrajectoryEllipses(bag_bundle.bag_data.lidar_traj[lidar], 0.99, 1e3,False,False,plt,p[0].get_color())

        # Plot GNSS in vehicle frame
        p = plt.plot(bag_bundle.statistics.gnss2vhc_traj.x, bag_bundle.statistics.gnss2vhc_traj.y)
        if with_ellipses:
            plotTrajectoryEllipses(bag_bundle.statistics.gnss2vhc_traj, 0.99, 1e3,False,False,plt,p[0].get_color())
        patches.append( mpatches.Patch(color=p[0].get_color(), label=gnss_topicname[0] + " on " + bag_name))

        # Plot fusion pose if asked
        if with_fusion:
            for fusion_topic in fusion_topicname:
                if bag_bundle.bag_data.fusion_traj.size() > 0:
                    p = plt.plot(bag_bundle.bag_data.fusion_traj.x, bag_bundle.bag_data.fusion_traj.y)
                    colors.append(p[0].get_color())
                    patches.append( mpatches.Patch(color=p[0].get_color(), label=fusion_topic + " on " + bag_name))
                    if with_ellipses:
                        plotTrajectoryEllipses(bag_bundle.bag_data.fusion_traj, 0.99, 1e3,False,False,plt,p[0].get_color())



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
    parser.add_argument("-md", "--mahalanobis", default=False, action='store_true', help="Plot mahalanbis distances of lidars and gnss")
    parser.add_argument("-mir", "--meanimpactratio", default=False, action='store_true', help="Plot mean impact ratio of lidars")
    parser.add_argument("-cov", "--covariance", default=False, action='store_true', help="Plot covariances on trajectory plot")
    parser.add_argument("-fusion", "--withfusion", default=False, action='store_true', help="Plot fusion topic as well")
    parser.add_argument("-rp", "--attitude", default=False, action='store_true', help="Plot roll pitch of the vehicle")
    parser.add_argument("-y", "--yaw", default=False, action='store_true', help="Plot additionally the yaw")
    parser.add_argument("-tf", "--transforms", default=False, nargs="*", help="Plot attitude of asked tf frames")
    args = parser.parse_args()

    bags_bundle = {b:BagBundle() for b in args.bag}
    for b in args.bag:
        with Bag(b, 'r') as bag:
            # Load all bag data
            bdata, stats = loadBagData(bag, args.transforms)

            # Move gnss to vehicle frame (and remove first / last record due to wrong yaw)
            gnss_vhc_traj = moveGnssToVhc(bdata.gnss_traj)
            # Save new gnss trajectory
            stats.gnss2vhc_traj = gnss_vhc_traj

            # For each lidar topics
            for lidar_topic in lidar_topicnames:
                if bdata.lidar_traj[lidar_topic].size() > 0:
                    lidar_sync_traj, distances = computeDist(gnss_vhc_traj, bdata.lidar_traj[lidar_topic])
                    stats.lidar_distances[lidar_topic] = distances
                    stats.lidar_interp_trajectories[lidar_topic] = lidar_sync_traj

            # For fusion topic
            for fusion_topic in fusion_topicname:
                if bdata.fusion_traj.size() > 0:
                    fusion_sync_traj, distances = computeDist(gnss_vhc_traj, bdata.fusion_traj)
                    stats.fusion_distances = distances
                    stats.fusion_interp_traj = fusion_sync_traj

            bags_bundle[b].bag_data = bdata 
            bags_bundle[b].statistics = stats 

    if args.distance:
        plotDistances(bags_bundle, args.withfusion)
    if args.trajectory:
        plotTrajectories(bags_bundle, args.covariance, args.withfusion)
    if args.mahalanobis:
        plotMahalanobis(bags_bundle)
    if args.meanimpactratio:
        plotMir(bags_bundle)
    if args.attitude:
        plotAttitude(bags_bundle, args.yaw)
    if args.transforms:
        plotTransformsAttitude(bags_bundle, args.yaw)
