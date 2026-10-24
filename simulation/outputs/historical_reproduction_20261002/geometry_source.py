"""Archived camera geometry functions, loaded by ncsim.historical_backend.

The loader supplies numpy, math, compose and axangle2mat. The four function
bodies retain the original abstract syntax trees; training entry points are
not part of this geometry-only module.
"""

def get_Tacm(theta):
    T1 = compose([0.0, 0.0, 0.0], axangle2mat([0, 0, 1], math.radians(theta[1])), np.ones(3))
    T2 = compose([0.0, 0.7, 0.0], axangle2mat([0, 0, 1], math.radians(theta[2])), np.ones(3))
    T3 = compose([0.0, 0.5, 0.0], axangle2mat([0, 0, 1], math.radians(theta[3])), np.ones(3))
    T4 = compose([0.0, 0.4, 0.0], axangle2mat([0, 1, 0], -math.radians(theta[5])), np.ones(3))
    T5 = compose([0.0, 0.1405, 0.0], axangle2mat([1, 0, 0], math.radians(theta[6])), np.ones(3))
    T6 = compose([0.0, 0.0, 0.0], axangle2mat([1, 0, 0], math.radians(180.0)), np.ones(3))
    T_cam = np.dot(np.dot(np.dot(np.dot(np.dot(T1, T2), T3), T4), T5), T6)
    return T_cam

def get_T13(theta):
    T1 = compose([0.0, 0.0, 0.0], axangle2mat([0, 0, 1], math.radians(theta[1])), np.ones(3))
    T2 = compose([0.0, 0.0, 0.0], axangle2mat([0, 0, 1], math.radians(theta[2])), np.ones(3))
    T3 = compose([0.0, 0.0, 0.0], axangle2mat([0, 0, 1], math.radians(theta[3])), np.ones(3))
    T_13 = np.dot(np.dot(T1, T2), T3)
    T_13 = np.linalg.inv(T_13)
    return T_13

def get_paltformangle(endpos, endz, targetpoint, T_13):
    direction = [targetpoint[0] - endpos[0], targetpoint[1] - endpos[1], targetpoint[2] - endz]
    magnitude = math.sqrt(direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2)
    if magnitude != 0:
        unit_direction = [component / magnitude for component in direction]
    else:
        unit_direction = [0, 0, 0]
    target_direction = T_13.dot(np.array([unit_direction[0], unit_direction[1], unit_direction[2], 1]))
    magnitude = math.sqrt(target_direction[0] ** 2 + target_direction[1] ** 2 + target_direction[2] ** 2)
    angle1 = -math.atan(target_direction[0] / magnitude / (target_direction[2] / magnitude))
    angle2 = math.asin(target_direction[1] / magnitude)
    return (math.degrees(angle1), math.degrees(angle2))

def get_pyramid(ratio, height, T_cam, endz):
    centre = np.array([0, 0, 0, 1])
    points = np.array([[0.621 / ratio, 0.349 / ratio, 1, 1], [-0.621 / ratio, 0.349 / ratio, 1, 1], [-0.621 / ratio, -0.349 / ratio, 1, 1], [0.621 / ratio, -0.349 / ratio, 1, 1]])
    centre = T_cam.dot(centre)
    centre[2] = centre[2] + endz
    for i in range(4):
        points[i] = T_cam.dot(points[i])
        points[i][2] = points[i][2] + endz
        points[i] = centre + (points[i] - centre) / (points[i][2] - centre[2]) * (height - centre[2])
        points[i][3] = 1
    return (centre, points)
