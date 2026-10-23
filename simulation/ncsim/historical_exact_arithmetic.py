"""Preserve the archived Python degree state and operation order exactly.

The public radians are only a key into the original degree state. This avoids
round-trip conversion changing near-singular KDL status on subsequent steps.
"""
from collections import OrderedDict
import types
import numpy as np
from . import historical_backend as hb, native_backend as nb, geometry
from .paper_evaluation import PaperEnv

_DEGREES=OrderedDict()


def original_degrees(q):
    key=tuple(q)
    return _DEGREES.get(key,np.rad2deg(q)).copy()


def inverse(xy,start_q):
    theta=nb.vector(original_degrees(start_q),3);target=nb.vector(xy,2);result=np.zeros(4)
    status=hb._LIB.nc_ik(nb.pointer(theta),nb.pointer(target),nb.pointer(result),100,1e-8)
    if status!=0:return np.array(start_q),False
    # Exact source: result[i] / 3.1415926 * 180; never convert back from radians.
    degrees=result[:3]/nb.SOURCE_PI*180
    q=np.deg2rad(degrees)
    _DEGREES[tuple(q)]=degrees
    if len(_DEGREES)>4096:_DEGREES.popitem(last=False)
    return q,True


def camera_geometry(q,xy,target,ratio=4.):
    theta=[0,*original_degrees(q),0,0,0]
    theta[5:7]=hb._CAMERA['get_paltformangle'](xy,1.8,target,hb._CAMERA['get_T13'](theta))
    T=hb._CAMERA['get_Tacm'](theta)
    center,corners=hb._CAMERA['get_pyramid'](ratio,target[2],T,1.8)
    return T,center[:3],T[:3,2],corners[:,:3]


classify=types.FunctionType(geometry.classify_and_score.__code__,dict(geometry.classify_and_score.__globals__,camera_geometry=camera_geometry))


class DegreeStateEnv(hb.HistoricalEnv):
    def _measure(self):
        self.inside,self.score,self.guide,self.geometry=classify(self.points,self.xy,self.scene.target,self.q)
    step=types.FunctionType(PaperEnv.step.__code__,dict(PaperEnv.step.__globals__,inverse=inverse))
