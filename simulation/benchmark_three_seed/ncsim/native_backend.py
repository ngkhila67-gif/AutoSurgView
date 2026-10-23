"""Thin Python binding to original archived KDL and NC collision C++ sources.

No numerical IK or collision fallback is provided. Missing/changed binaries fail.
Public joint arrays use normal radians; the source environment stores degrees.
"""
import ctypes
from functools import lru_cache
import json
import os
from pathlib import Path
import numpy as np
from .data import ROOT, sha256

ENGINE='archived_cpp_native_v2'
SOURCE_PI=3.1415926
_DLL_DIRECTORIES=[]

@lru_cache(maxsize=1)
def library():
    folder=ROOT/'native'
    manifest=folder/'build_manifest.json'
    path=folder/('original_backend.dll' if os.name=='nt' else 'original_backend.so')
    if not manifest.exists() or not path.exists():
        raise RuntimeError('Original native backend missing. Run native/build_native.py; no fallback is allowed.')
    metadata=json.loads(manifest.read_text(encoding='utf-8'))
    if sha256(path)!=metadata['dll_sha256']:
        raise RuntimeError('Native backend checksum mismatch; rebuild and verify before running.')
    if os.name == 'nt':
        _DLL_DIRECTORIES.append(os.add_dll_directory(metadata['runtime_dll_directory']))
    lib=ctypes.CDLL(str(path))
    ptr=ctypes.POINTER(ctypes.c_double)
    signatures=dict(nc_fk=[ptr,ptr],nc_ik=[ptr,ptr,ptr,ctypes.c_int,ctypes.c_double],
                    nc_self_collision=[ptr],nc_obstacle_collision=[ptr,ptr,ctypes.c_int],
                    nc_footprints=[ptr,ptr],nc_pair_collision=[ptr,ptr])
    for name,args in signatures.items():
        getattr(lib,name).argtypes=args
        getattr(lib,name).restype=ctypes.c_int
    return lib

def pointer(array):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

def vector(values,size):
    array=np.ascontiguousarray(values,dtype=np.float64)
    if array.shape!=(size,) or not np.isfinite(array).all():
        raise ValueError(f'Expected {size} finite numbers')
    return array

def degrees(q):
    return vector(np.rad2deg(vector(q,3)),3)

def forward(q):
    theta=degrees(q);result=np.zeros(8,dtype=np.float64)
    code=library().nc_fk(pointer(theta),pointer(result))
    if code!=0:
        raise RuntimeError(f'Original KDL FK returned {code}')
    return result.reshape(4,2)

def inverse_details(xy,start_q,max_iter=100,tolerance=1e-8):
    theta=degrees(start_q);target=vector(xy,2);result=np.zeros(4,dtype=np.float64)
    if max_iter<1 or tolerance<=0:
        raise ValueError('Invalid KDL solver settings')
    status=library().nc_ik(pointer(theta),pointer(target),pointer(result),max_iter,tolerance)
    # Myclass.py accepts status exactly zero, converts first three solved joints
    # with 180 / 3.1415926, and discards the redundant fourth joint.
    nominal_radians=result[:3]*np.pi/SOURCE_PI
    return nominal_radians,int(status),result

def inverse(xy,start_q,max_iter=100,tolerance=1e-8):
    q,status,_=inverse_details(xy,start_q,max_iter,tolerance)
    return (q,True) if status==0 else (np.asarray(start_q,dtype=float).copy(),False)

def self_collision(q):
    theta=degrees(q)
    return bool(library().nc_self_collision(pointer(theta)))

def obstacle_collision(q,rectangles):
    """Original collision.cpp; Nx5 (x_m,y_m,width_m,height_m,angle_degrees)."""
    theta=degrees(q)
    rectangles=np.ascontiguousarray(rectangles,dtype=np.float64).reshape(-1,5)
    if not np.isfinite(rectangles).all() or np.any(rectangles[:,2:4]<=0):
        raise ValueError('Invalid obstacle rectangles')
    return bool(library().nc_obstacle_collision(pointer(theta),pointer(rectangles),len(rectangles)))

def footprint_shapes(q):
    theta=degrees(q);records=np.empty((7,8),dtype=np.float64)
    count=library().nc_footprints(pointer(theta),pointer(records))
    if count!=7:
        raise RuntimeError('Unexpected archived collision model geometry')
    groups=[[] for _ in range(4)]
    for group,kind,x,y,r,w,h,angle in records[:count]:
        a=np.deg2rad(angle);c,s=np.cos(a),np.sin(a)
        groups[int(group)].append(('circle' if kind==0 else 'rect',np.array([x,y]),
            r if kind==0 else w,0. if kind==0 else h,np.array([[c,-s],[s,c]])))
    return groups

def shapes_overlap(a,b):
    def packed(shape):
        kind,center,w,h,R=shape
        return vector([0 if kind=='circle' else 1,*center,w,h,np.rad2deg(np.arctan2(R[1,0],R[0,0]))],6)
    a,b=packed(a),packed(b)
    return bool(library().nc_pair_collision(pointer(a),pointer(b)))

def provenance():
    library()
    path=ROOT/'native/build_manifest.json'
    m=json.loads(path.read_text(encoding='utf-8'))
    return dict(engine=ENGINE,kdl_version=m['kdl_version'],dll_sha256=m['dll_sha256'],
                build_manifest=str(path),build_manifest_sha256=sha256(path),
                kdl_source=m['kdl_source'],collision_source=m['collision_source'],scope=m['scope'])
