import numpy as np

# [v]x
def skew(v):
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
    axis_x_sq = skew(np.square(axis))
    axis_x = skew(axis)
    return np.identity(3) + np.sin(angle) * axis_x + (1 - np.cos(angle)) * axis_x_sq


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


def rot_matx_to_eulerZYX(R):
    r31 = R[2, 0]
    r32 = R[2, 1]
    r33 = R[2, 2]
    r21 = R[1, 0]
    r11 = R[0, 0]
    r12 = R[0, 1]
    r22 = R[1, 1]

    eps = 1e-6
    if not ((1.0 - np.abs(r31)) < eps):
        pitch1 = -np.arcsin(r31)
        pitch2 = np.pi - pitch1
        roll1 = np.arctan2(r32/np.cos(pitch1), r33/np.cos(pitch1))
        roll2 = np.arctan2(r32/np.cos(pitch2), r33/np.cos(pitch2))
        yaw1 = np.arctan2(r21/np.cos(pitch1), r11/np.cos(pitch1))
        yaw2 = np.arctan2(r21/np.cos(pitch2), r11/np.cos(pitch2))
        return (yaw1, pitch1, roll1), (yaw2, pitch2, roll2)
    else:
        # if r31 == -1.0 (then pitch is pi/2)
        yaw = 0
        if (r31 + 1.0) < eps:
            pitch = np.pi/2.0
            roll = np.arctan2(r12, r22)
        # then r31 == 1.0 (then pitch is -pi/2)
        else:
            pitch = -np.pi/2.0
            roll = np.arctan2(r12, -r22)
        return yaw, pitch, roll


def quat_to_euler(q):
    return rot_matx_to_eulerZYX(quat_to_rot_matx(q))


def eulerZYX_to_quat(yaw, pitch, roll):
    return rot_matx_to_quat(euler_to_rot_matx(roll, pitch, yaw))


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
