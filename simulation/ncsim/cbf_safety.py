"""Simulation-only CBF-QP extension; never used to rewrite the frozen benchmark.

Original .5405 KDL, seven STL assemblies and NC self-collision remain in use.
New equipment uses declared geometric envelopes, not measured hospital CAD.
The CBF is a static-equipment clearance filter, not an occlusion-value barrier.
"""
from pathlib import Path
import ctypes
import numpy as np
import osqp
from scipy import sparse
from . import historical_backend as hb, native_backend as nb
from .data import ROOT, configuration, sha256
from .manuscript_viewer import PARTS, assembly

DT = 0.1
DECISION_TIME = 1.0
V_MAX = 0.35
D_SAFE = 0.04
ALPHA = 2.0


def forward(q):
    """Call the actual archived .5405 m KDL library, not a replacement FK."""
    if hb._LIB is None:
        raise RuntimeError('Historical backend must be configured')
    ptr = ctypes.POINTER(ctypes.c_double)
    hb._LIB.nc_fk.argtypes = [ptr, ptr]
    hb._LIB.nc_fk.restype = ctypes.c_int
    out = np.zeros(8)
    if hb._LIB.nc_fk(nb.pointer(nb.degrees(q)), nb.pointer(out)) != 0:
        raise RuntimeError('Original KDL forward kinematics failed')
    return out.reshape(4, 2)[-1].copy()


def equipment_scene(target, layout):
    """Metre-scale parametric OR fixtures. All rendered solids have OBB envelopes.

    Layout parameters are fixed before policy comparison. Layout 0 is spacious;
    layout 1 lowers the lamp; layout 2 additionally moves a monitor inward.
    These are constructed stress tests and not reconstructions of captured rooms.
    """
    gx, gy, gz = map(float, target)
    items = []
    def add(kind, name, x, y, yaw, specs):
        parts = []
        c, s = np.cos(yaw), np.sin(yaw)
        for label, p, size, color, shape in specs:
            px, py, pz = p
            parts.append(dict(id=kind+'_'+label, center=[x+c*px-s*py, y+s*px+c*py, pz],
                              size=list(size), yaw=float(yaw), color=color, shape=shape))
        items.append(dict(id=kind, label=name, parts=parts))
    def box(label, xyz, size, color='#d5dee3'):
        return (label, xyz, size, color, 'box')
    def post(label, xyz, diameter, height, color='#b6c6ce'):
        return (label, xyz, [diameter, diameter, height], color, 'cylinder')
    floor = -.1245
    add('table', '手术床', gx, gy, 0, [
        box('base', [0,0,floor+.12],[.72,.48,.20],'#c2cbd3'),
        box('column',[0,0,.31],[.28,.26,.6]),
        box('deck',[0,0,gz-.095],[2.02,.64,.08],'#a7bac4'),
        box('mattress',[0,0,gz-.035],[2.04,.66,.055],'#408c9c'),
        box('rail_near',[0,-.36,gz-.09],[1.7,.025,.06]),
        box('rail_far',[0,.36,gz-.09],[1.7,.025,.06])])
    add('anaesthesia', '麻醉工作站', gx-1.40, gy+.75, 0, [
        box('base',[0,0,.01],[.70,.60,.15],'#b8c6cf'),
        box('body',[0,0,.54],[.58,.5,.9],'#e3e9ed'),
        box('screen',[0,-.04,1.19],[.46,.13,.34],'#354f61'),
        box('control',[0,-.26,.83],[.52,.16,.1],'#667e8f'),
        post('tank',[.31,.11,.55],.13,.85,'#85b6b7')])
    mx = gx-.99 if layout<2 else gx-.53
    add('monitor', '移动监护仪', mx, gy-.07, 0, [
        box('base',[0,0,-.035],[.50,.44,.11],'#a8bdca'),
        post('stand',[0,0,.55],.06,1.15),
        box('screen',[0,0,1.49],[.46,.13,.36],'#3f6378'),
        box('display',[0,-.072,1.49],[.4,.018,.29],'#122e3d')])
    add('instruments','器械车',gx+.05,gy+1.08,0,[
        box('top',[0,0,.84],[.78,.48,.045],'#7ca9ad'),
        box('shelf',[0,0,.26],[.74,.44,.03],'#c1cdd3'),
        *[post(f'leg{k}',[x,y,.34],.035,.84) for k,(x,y) in enumerate([(-.33,-.18),(.33,-.18),(-.33,.18),(.33,.18)])]])
    add('iv','输液架',gx+1.18,gy+.38,0,[
        box('base',[0,0,-.035],[.40,.40,.06],'#abbcc6'),
        post('pole',[0,0,.83],.03,1.83),
        box('hook',[0,0,1.78],[.34,.025,.025]),
        box('bag',[.14,0,1.55],[.12,.055,.22],'#b8d9dd')])
    lamp_z=2.40 if layout==0 else 1.95
    add('lamp','手术无影灯',gx+.27,gy+.025,0,[
        box('head',[0,0,lamp_z],[.38,.30,.10],'#dbe8ec'),
        box('light',[0,0,lamp_z-.055],[.30,.23,.015],'#fff2c7'),
        post('neck',[0,0,(2.65+lamp_z+.05)/2],.045,2.65-lamp_z-.05),
        box('arm',[.46,0,2.67],[.96,.055,.055]),
        post('pole',[.91,0,1.24],.055,2.85)])
    return dict(layout=layout,label=['宽裕布局','低位手术灯','低位手术灯与靠近的监护仪'][layout],
                units='m',ground_z=floor,equipment=items,
                provenance='New parametric geometric equipment; dimensions/poses are simulation assumptions, not measured hospital equipment',
                occlusion_scope='Original recorded point cloud only; added fixture surfaces are NOT inserted into the policy observation or occlusion score')


class RobotCapsules:
    """Conservative capsules enclosing every vertex AND triangle of each STL.

    Endpoints span the longest axis of the original local AABB; radius is the
    half-diagonal of the other two axes. AABB containment implies containment of
    all triangles. This is intentionally conservative, especially for the base.
    """
    def __init__(self):
        self.a=[]; self.b=[]; self.r=[]; self.manifest=[]
        for key,label,name in PARTS:
            p=Path(configuration()['archive_root'])/'mytest'/name
            raw=p.read_bytes()
            dt=np.dtype([('normal','<f4',3),('v','<f4',(3,3)),('attr','<u2')])
            v=np.frombuffer(raw[84:],dtype=dt)['v'].reshape(-1,3).astype(float)*.001
            lo,hi=v.min(0),v.max(0); ax=int(np.argmax(hi-lo)); a=(lo+hi)/2; b=a.copy()
            a[ax]=lo[ax];b[ax]=hi[ax]
            radius=float(np.linalg.norm(np.delete((hi-lo)/2,ax)))+1e-7
            perp=np.delete(v-(lo+hi)/2,ax,axis=1)
            assert np.max(np.linalg.norm(perp,axis=1))<=radius
            self.a.append(a);self.b.append(b);self.r.append(radius)
            self.manifest.append(dict(key=key,source=str(p),sha256=sha256(p),a=a.tolist(),b=b.tolist(),radius_m=radius,vertices=len(v)))
        self.a,self.b,self.r=map(np.asarray,(self.a,self.b,self.r))

    def world(self,q,target):
        T=np.array(assembly(q,forward(q),target));R=T[:,:3,:3];t=T[:,:3,3]
        return np.einsum('ijk,ik->ij',R,self.a)+t,np.einsum('ijk,ik->ij',R,self.b)+t


def segment_box_distance(a,b,half):
    """Exact Euclidean segment-to-AABB distances, batched over leading axis.

    Squared distance is a convex piecewise quadratic. Split at crossings of
    each slab, minimize each quadratic on its interval, then take the minimum.
    """
    a,b,half=np.broadcast_arrays(np.atleast_2d(a),np.atleast_2d(b),np.atleast_2d(half))
    d=b-a
    with np.errstate(divide='ignore',invalid='ignore'):
        roots=np.concatenate([(-half-a)/d,(half-a)/d],axis=1)
    roots=np.nan_to_num(roots,nan=0.,posinf=1.,neginf=0.)
    ts=np.sort(np.c_[np.zeros(len(a)),np.ones(len(a)),np.clip(roots,0,1)],axis=1)
    left,right=ts[:,:-1],ts[:,1:];mid=(left+right)/2
    x=a[:,None,:]+mid[:,:,None]*d[:,None,:]
    active=np.abs(x)>half[:,None,:]
    bound=np.where(x>0,half[:,None,:],-half[:,None,:])
    offset=(a[:,None,:]-bound)*active
    den=(d[:,None,:]**2*active).sum(2)
    num=(d[:,None,:]*offset).sum(2)
    opt=np.divide(-num,den,out=mid.copy(),where=den>1e-24)
    opt=np.clip(opt,left,right)
    p=a[:,None,:]+opt[:,:,None]*d[:,None,:]
    sq=np.maximum(np.abs(p)-half[:,None,:],0)**2
    return np.sqrt(sq.sum(2).min(1))


class Clearance:
    def __init__(self,capsules,scene,target):
        self.robot=capsules;self.target=np.asarray(target)
        self.parts=[p for e in scene['equipment'] for p in e['parts']]
        self.c=np.array([p['center'] for p in self.parts]);self.half=np.array([p['size'] for p in self.parts])/2
        self.rot=[]
        for p in self.parts:
            c,s=np.cos(p['yaw']),np.sin(p['yaw']);self.rot.append([[c,s,0],[-s,c,0],[0,0,1]])
        self.rot=np.array(self.rot)

    def distances(self,q):
        a,b=self.robot.world(q,self.target)
        aa=np.einsum('ojk,roj->rok',np.transpose(self.rot,(0,2,1)),a[:,None,:]-self.c[None,:,:])
        bb=np.einsum('ojk,roj->rok',np.transpose(self.rot,(0,2,1)),b[:,None,:]-self.c[None,:,:])
        half=np.broadcast_to(self.half,aa.shape)
        d=segment_box_distance(aa.reshape(-1,3),bb.reshape(-1,3),half.reshape(-1,3))
        return d-np.repeat(self.robot.r,len(self.parts))

    def h_and_gradient(self,q):
        h=self.distances(q)-D_SAFE
        jac=np.column_stack([(self.distances(q+np.eye(3)[j]*1e-5)-self.distances(q-np.eye(3)[j]*1e-5))/(2e-5) for j in range(3)])
        return h,jac

    def closest(self,q):
        ds=self.distances(q);ix=int(ds.argmin());r,o=divmod(ix,len(self.parts))
        return float(ds[ix]),PARTS[r][0]+' / '+self.parts[o]['id']


def filter_velocity(q,u_nom,clearance):
    """Hard clearance CBF-QP, followed by sampled nonlinear feasibility checks.

    qdot = u; h=d_capsule,OBB-d_safe; grad(h) u + alpha*h >= 0.
    No safety slack. No undocumented hardware joint bounds are invented.
    Numerical failure returns a hold, not unchecked nominal motion.
    """
    h,jac=clearance.h_and_gradient(q)
    if h.min() < -1e-6 or nb.self_collision(q):
        return np.zeros(3),dict(status='unsafe_initial_state',correction=float(np.linalg.norm(u_nom)),residual=None,backtrack=0)
    A=sparse.csc_matrix(np.vstack([jac,np.eye(3)]));low=np.r_[-ALPHA*h,np.full(3,-V_MAX)];up=np.r_[np.full(len(h),np.inf),np.full(3,V_MAX)]
    solver=osqp.OSQP()
    solver.setup(P=sparse.eye(3,format='csc'),q=-np.asarray(u_nom),A=A,l=low,u=up,
                 verbose=False,eps_abs=1e-8,eps_rel=1e-8,max_iter=10000,polishing=False)
    result=solver.solve(raise_error=False)
    solved=result.info.status_val in (1,2) and result.x is not None and np.isfinite(result.x).all()
    u=np.array(result.x) if solved else np.zeros(3)
    if solved and (np.min(jac@u+ALPHA*h)<-2e-6 or np.max(np.abs(u))>V_MAX+2e-6):solved=False;u[:]=0
    k=0
    # Model checks at five points within each 0.1 s integration interval.
    # This is a numerical safeguard, not a continuous-time proof.
    for k in range(13):
        ok=all(clearance.distances(q+t*DT*u).min()>=D_SAFE-1e-7 and not nb.self_collision(q+t*DT*u) for t in [.2,.4,.6,.8,1.])
        if ok:break
        u*=.5
    else:u[:]=0
    residual=float(np.min(jac@u+ALPHA*h))
    return u,dict(status=result.info.status if solved else 'QP_failure_hold',correction=float(np.linalg.norm(u-u_nom)),residual=residual,backtrack=k)
