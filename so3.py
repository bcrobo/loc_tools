import numpy as np


def skew_sym_matx(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def v_operator(skew_matx):
    return np.array([skew_matx[2, 1], skew_matx[0, 2], skew_matx[1, 0]])


def angle_axis_to_rotv(angle, axis):
    return angle * axis


def rotv_to_angle_axis(rotv):
    angle = np.linalg.norm(rotv)
    axis = np.zeros(3)
    if angle != 0:
        axis = rotv / angle
    return angle, axis


def rodrigues(angle, axis):
    skew_square = skew_sym_matx(np.square(axis))
    skew = skew_sym_matx(axis)
    return np.identity(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * skew_square


def exp_map_rot_matx(rotv):
    angle, axis = rotv_to_angle_axis(rotv)
    return rodrigues(angle, axis)


def exp_map_quat(rotv):
    angle, axis = rotv_to_angle_axis(rotv)
    half_angle = angle / 2
    u = np.sin(half_angle) * axis
    return np.array([np.cos(half_angle), u[0], u[1], u[2]])


def log_map_rot_matx(R):
    angle = np.arccos((np.trace(R) - 1) / 2)
    axis = np.zeros(3)
    if angle != 0:
        axis = v_operator(R - np.transpose(R)) / (2 * np.sin(angle))
    return angle, axis


def log_map_quat(q):
    qv = q[1:3]
    axis = qv / np.linalg.norm(qv)
    angle = np.arctan(np.linalg.norm(qv), q[0])
    return angle, axis


# roll - [-pi pi]
# pitch [-pi/2 pi/2]
# yaw [0 2pi]
def euler_to_rot_matx(roll, pitch, yaw):
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


def quat_to_rot_matx(q):
    a = q[0]  # w
    b = q[1]  # x
    c = q[2]  # y
    d = q[3]  # z

    return np.array([
        [a ** 2 + b ** 2 - c ** 2 - d ** 2, 2 * (b * c + a * d), 2 * (b * d + a * c)],
        [2 * (b * c - a * d), a ** 2 - b ** 2 + c ** 2 - d ** 2, 2 * (c * d + a * b)],
        [2 * (b * d - a * c), 2 * (c * d - a * b), a ** 2 - b ** 2 - c ** 2 + d ** 2]
    ])


def rot_matx_to_euler(R):
    r31 = R[2, 0]
    r32 = R[2, 1]
    r33 = R[2, 2]
    r21 = R[1, 0]
    r11 = R[0, 0]
    roll = np.arctan(r32 / r33)
    pitch = np.arcsin(-r31)
    yaw = np.arctan(r21 / r11)
    return roll, pitch, yaw


def quat_to_euler(q):
    a = q[0]  # w
    b = q[1]  # x
    c = q[2]  # y
    d = q[3]  # z
    roll = np.arctan((2 * c * d + 2 * a * b) / (a ** 2 - b ** 2 - c ** 2 + d ** 2))
    pitch = np.arcsin(-2 * b * d + 2 * a * c)
    yaw = np.arctan((2 * b * c + 2 * a * d) / (a ** 2 + b ** 2 - c ** 2 - d ** 2))
    return roll, pitch, yaw


def euler_to_quat(roll, pitch, yaw):
    R = euler_to_rot_matx(roll, pitch, yaw)
    return rot_matx_to_quat(R)


def rot_matx_to_quat(R):
    T = 1 + np.trace(R)

    if T > 1.e-9:
        S = 2 * np.sqrt(T)
        a = S / 4
        b = (R[2, 1] - R[1, 2]) / S
        c = (R[0, 2] - R[2, 0]) / S
        d = (R[1, 0] - R[0, 1]) / S
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = 2 * np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
            a = (R[1, 2] - R[2, 1]) / S
            b = -S / 4
            c = (R[1, 0] - R[0, 1]) / S
            d = (R[0, 2] - R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = 2 * np.sqrt(1 - R[0, 0] + R[1, 1] - R[2, 2])
            a = (R[2, 0] - R[0, 2]) / S
            b = (R[1, 0] - R[0, 1]) / S
            c = -S / 4
            d = (R[2, 1] - R[1, 2]) / S
        else:
            S = 2 * np.sqrt(1 - R[0, 0] - R[1, 1] + R[2, 2])
            a = (R[0, 1] - R[1, 0]) / S
            b = (R[0, 2] - R[2, 0]) / S
            c = (R[2, 1] - R[1, 2]) / S
            d = -S / 4

    return np.array([a, b, c, d])
