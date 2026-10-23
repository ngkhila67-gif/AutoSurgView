"""Explicit archived 0.5405 m source variant, isolated from the frozen new run."""
import ast,ctypes,json,math,os,types
from pathlib import Path
import numpy as np
from .data import ROOT,sha256
from . import geometry,native_backend
from .paper_evaluation import PaperEnv

_HANDLES=[]
_LIB=None
_CAMERA=None


def configure(out):
    global _LIB,_CAMERA
    out=Path(out)
    m=json.loads((out/'historical_backend_manifest.json').read_text(encoding='utf-8'))
    path=out/('historical_backend_05405.dll' if os.name=='nt' else 'historical_backend_05405.so')
    if sha256(path)!=m['dll_sha256'] or sha256(m['source'])!=m['source_sha256']:
        raise ValueError('Historical variant source or binary checksum changed')
    if os.name == 'nt':
        _HANDLES.append(os.add_dll_directory(m['runtime_dll_directory']))
    _LIB=ctypes.CDLL(str(path))
    ptr=ctypes.POINTER(ctypes.c_double)
    _LIB.nc_ik.argtypes=[ptr,ptr,ptr,ctypes.c_int,ctypes.c_double];_LIB.nc_ik.restype=ctypes.c_int
    source=Path(m['source']).read_text(encoding='utf-8-sig')
    tree=ast.parse(source)
    names={'get_Tacm','get_T13','get_paltformangle','get_pyramid'}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==4
    def compose(t,R,s):
        T=np.eye(4);T[:3,:3]=R@np.diag(s);T[:3,3]=t;return T
    ns=dict(np=np,math=math,compose=compose,axangle2mat=lambda axis,a:geometry.transform([0,0,0],axis,a)[:3,:3])
    exec(compile(ast.Module(body=nodes,type_ignores=[]),m['source'],'exec'),ns)
    _CAMERA=ns


def inverse(xy,start_q):
    if _LIB is None:
        raise RuntimeError('Configure the historical variant explicitly')
    theta=native_backend.degrees(start_q);target=native_backend.vector(xy,2);result=np.zeros(4)
    status=_LIB.nc_ik(native_backend.pointer(theta),native_backend.pointer(target),native_backend.pointer(result),100,1e-8)
    return (result[:3]*np.pi/native_backend.SOURCE_PI,True) if status==0 else (np.array(start_q),False)


def camera_geometry(q,xy,target,ratio=4.):
    theta=[0,*np.rad2deg(q),0,0,0]
    theta[5:7]=_CAMERA['get_paltformangle'](xy,1.8,target,_CAMERA['get_T13'](theta))
    T=_CAMERA['get_Tacm'](theta)
    center,corners=_CAMERA['get_pyramid'](ratio,target[2],T,1.8)
    return T,center[:3],T[:3,2],corners[:,:3]


_scope=dict(geometry.classify_and_score.__globals__,camera_geometry=camera_geometry)
classify=types.FunctionType(geometry.classify_and_score.__code__,_scope)


class HistoricalEnv(PaperEnv):
    def _measure(self):
        self.inside,self.score,self.guide,self.geometry=classify(self.points,self.xy,self.scene.target,self.q)

    # Same source-preserving step body, only bind the explicitly selected chain.
    step=types.FunctionType(PaperEnv.step.__code__,dict(PaperEnv.step.__globals__,inverse=inverse))


def replay_initializer(out):
    configure(out)
    from . import historical_reproduction as audit
    audit.PaperEnv=HistoricalEnv
    audit.inverse=inverse
