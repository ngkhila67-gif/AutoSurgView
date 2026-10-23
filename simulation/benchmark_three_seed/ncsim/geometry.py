"""Original viewing geometry; FK, IK and collision use archived native sources."""
import math
import numpy as np
from .native_backend import forward, inverse, footprint_shapes, shapes_overlap, self_collision

LINKS = np.array([0.7, 0.5, 0.5145])
INITIAL_Q = np.deg2rad([-50.0, 100.0, -50.0])
HEIGHT = 1.8


def transform(translation, axis, angle):
    a = np.asarray(axis, dtype=float)
    a /= np.linalg.norm(a)
    x, y, z = a
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    R = np.eye(3) + np.sin(angle) * K + (1-np.cos(angle)) * K @ K
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, translation
    return T


def camera_geometry(q, xy, target, ratio=4.0):
    direction = np.asarray(target) - np.r_[xy, HEIGHT]
    direction /= np.linalg.norm(direction)
    total = q.sum()
    c, s = np.cos(total), np.sin(total)
    local = np.array([c*direction[0]+s*direction[1], -s*direction[0]+c*direction[1], direction[2]])
    pan = -np.arctan(local[0]/local[2])
    tilt = np.arcsin(np.clip(local[1], -1, 1))
    T = (transform([0,0,0], [0,0,1], q[0]) @
         transform([0,.7,0], [0,0,1], q[1]) @
         transform([0,.5,0], [0,0,1], q[2]) @
         transform([0,.4035,0], [0,1,0], -pan) @
         transform([0,.111,0], [1,0,0], tilt) @
         transform([0,0,0], [1,0,0], np.pi))
    center = T[:3,3] + [0,0,HEIGHT]
    rays = np.array([[.621,.349,ratio],[-.621,.349,ratio],[-.621,-.349,ratio],[.621,-.349,ratio]]) / ratio
    rays = rays @ T[:3,:3].T
    corners = center + rays * ((target[2]-center[2])/rays[:,2])[:,None]
    return T, center, T[:3,2].copy(), corners


def classify_and_score(points, xy, target, q):
    T, center, axis, corners = camera_geometry(q, xy, target)
    if len(points) == 0:
        return np.zeros(0,dtype=bool), 0.0, np.zeros(2), (T,center,axis,corners)
    projected = center + (points-center) * ((target[2]-center[2])/(points[:,2]-center[2]))[:,None]
    # Same horizontal ray crossing convention as the source implementation.
    crossings = np.zeros(len(points), dtype=int)
    px,py = projected[:,0], projected[:,1]
    for i in range(4):
        a,b = corners[i],corners[(i+1)%4]
        if abs(b[1]-a[1]) < 1e-14:
            continue
        yvalid = ((py-b[1])*(py-a[1]) <= 0) & (py != b[1])
        xhit = a[0]+(b[0]-a[0])*(py-a[1])/(b[1]-a[1])
        crossings += yvalid & (xhit >= px)
    inside = crossings % 2 == 1
    p = points[inside]
    if not len(p):
        return inside,0.0,np.zeros(2),(T,center,axis,corners)
    v = p - np.r_[xy,HEIGHT]
    denom = v @ axis
    if np.any(np.abs(denom) < 1e-12):
        raise ValueError("Singular axial projection; scene requires review")
    projected_vector = v / denom[:,None] - axis
    dist = np.linalg.norm(projected_vector,axis=1)
    # Epsilon is a documented numerical guard for a point exactly on the axis.
    score = np.sum(1.0 / np.maximum(dist,1e-9))
    planar = projected_vector[:,:2]
    norms = np.linalg.norm(planar,axis=1)
    unit = np.divide(planar,norms[:,None],out=np.zeros_like(planar),where=norms[:,None]>1e-12)
    guide = -(unit * dist[:,None]).sum(axis=0)
    guide_norm = np.linalg.norm(guide)
    if guide_norm > 1e-12:
        guide /= guide_norm
    return inside,float(score),guide,(T,center,axis,corners)


def sample_points(points, target, T, count, rng):
    if len(points) == 0:
        return np.zeros((count,3),dtype=np.float32)
    # Preserve the original float32 cast BEFORE subtracting the camera target.
    inv = np.linalg.inv(T)
    camera_points = ((points-[0,0,HEIGHT]) @ inv[:3,:3].T + inv[:3,3]).astype(np.float32)
    target_camera = (np.r_[np.asarray(target)-[0,0,HEIGHT],1.0] @ inv.T)[:3]
    xyz = camera_points - target_camera
    n = len(xyz)
    if n > count:
        xyz = xyz[rng.choice(n,count,replace=False)]
    elif n < count:
        xyz = np.vstack((xyz,xyz[rng.choice(n,count-n,replace=True)]))
    return xyz.astype(np.float32)


