#!/usr/bin/python

import numpy as np
import math

LASER_PITCH = np.radians(-10)
LASER_YAW = np.radians(0)
SENSOR_POSITION = np.array([2,0,2])
SENSOR_RPY =np.array([np.radians(0), np.radians(-10), np.radians(0)])

PLANE_NORMAL = np.array([0,0,1])
# Point on the plane
P0 = np.array([2,0,0])

VEHICLE_LENGTH = 4

def rotm(rpy):
    roll = rpy[0]
    pitch = rpy[1]
    yaw = rpy[2]

    yawMatrix = np.matrix([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
        ])

    pitchMatrix = np.matrix([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
        ])

    rollMatrix = np.matrix([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]
        ])

    return yawMatrix * pitchMatrix * rollMatrix

def sensorToVehicle(P_sensor):
    R = rotm(SENSOR_RPY)
    return np.dot(np.transpose(R), P_sensor) + SENSOR_POSITION

def sensorPoint(r):
    return np.array([r * math.cos(LASER_PITCH) * math.cos(LASER_YAW), -r * math.cos(LASER_PITCH) * math.sin(LASER_YAW), r * math.sin(LASER_PITCH)])

def intersection():
    Ps1 = sensorPoint(1)
    Ps2 = sensorPoint(2)

    Pv1 = sensorToVehicle(Ps1)
    Pv2 = sensorToVehicle(Ps2)

    l = Pv2 - Pv1
    l0 = Pv1

    down = np.dot(l, PLANE_NORMAL)
    if down == 0:
        print("+ Line and plane are parallel")
        return

    up = np.dot((P0 - l0), PLANE_NORMAL)
    if up == 0:
        print("+ Line contained in the plane")
        return

    d = up / down
    return d * l + l0

P = intersection()
P = P - np.array([VEHICLE_LENGTH/2, 0, 0])
print "+ First lidar impact with ground at " + str(P) + " from front of the vehicle"
     
