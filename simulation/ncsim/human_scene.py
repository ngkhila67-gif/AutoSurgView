"""Declared, synthetic OR humans. Rays and CBF use these saved capsules.

These are parametric adults, not patient scans or measured clinical anatomy.
Original robot meshes, FK and self-collision are provided by the archive.
Detailed CAD display meshes are a separate appearance layer, not this geometry.
"""
import numpy as np
from .cbf_safety import Clearance, equipment_scene, D_SAFE
from .manuscript_viewer import PARTS

HUMAN_MARGIN = .08
SCENARIOS = ['医生静态探身', '医生动态探身', '双人协作与动态探身']


def segment_segment_distance(a, b, c, d):
    """Exact batched segment distance, including parallel and degenerate cases.

    Minimize the convex quadratic on [0,1]^2: four edges and the interior
    stationary point are a complete candidate set.
    """
    a,b,c,d=np.broadcast_arrays(*[np.atleast_2d(x) for x in (a,b,c,d)])
    u,v,w=b-a,d-c,a-c
    A=(u*u).sum(-1); B=(u*v).sum(-1); C=(v*v).sum(-1)
    D=(u*w).sum(-1); E=(v*w).sum(-1)
    div=lambda n,z: np.divide(n,z,out=np.zeros_like(n),where=np.abs(z)>1e-24)
    ss=[np.zeros_like(A),np.ones_like(A),np.clip(div(-D,A),0,1),np.clip(div(B-D,A),0,1)]
    tt=[np.clip(div(E,C),0,1),np.clip(div(E+B,C),0,1),np.zeros_like(A),np.ones_like(A)]
    det=A*C-B*B
    s=div(B*E-C*D,det);t=div(A*E-B*D,det)
    valid=(det>1e-14*A*C)&(s>=0)&(s<=1)&(t>=0)&(t<=1)
    ds=[np.sum((w+x[...,None]*u-y[...,None]*v)**2,axis=-1) for x,y in zip(ss,tt)]
    ds.append(np.where(valid,np.sum((w+s[...,None]*u-t[...,None]*v)**2,axis=-1),np.inf))
    return np.sqrt(np.maximum(0,np.min(ds,axis=0)))


class HumanScene:
    def __init__(self,target,scenario):
        self.target=np.array(target,dtype=float);self.scenario=int(scenario)
        self.equipment=equipment_scene(target,0)
        self.equipment['label']=SCENARIOS[scenario]
        # Mattress supports the patient's back, while G is the anterior chest.
        for item in self.equipment['equipment']:
            if item['id']=='table':
                for p in item['parts']:
                    if p['id'].split('_',1)[1] in ('deck','mattress','rail_near','rail_far'):
                        p['center'][2]-=.28
                    elif p['id']=='table_column':p['center'][2]=.20;p['size'][2]=.38
            if item['id']=='instruments':
                for p in item['parts']:p['center'][0]+=.85
        self.equipment['occlusion_scope']='Synthetic surfaces of the same humans and equipment; archived anonymous cloud is not mixed in'
        self._equipment_points=self.equipment_points()

    def pose(self,t):
        G=self.target;parts=[]
        def cap(person,name,a,b,r,color,category):
            parts.append(dict(id=person+'_'+name,person=person,category=category,
                a=np.asarray(a).tolist(),b=np.asarray(b).tolist(),radius_m=r,color=color))
        def patient(name,a,b,r,color='#4b9ca5',category='patient'):
            cap('patient',name,G+np.array(a),G+np.array(b),r,color,category)
        patient('torso',[-.25,0,-.14],[.18,0,-.14],.14)
        patient('head',[-.54,0,-.10],[-.54,0,-.10],.105,'#d7a282')
        patient('neck',[-.46,0,-.13],[-.32,0,-.13],.06,'#d7a282')
        for sign in (-1,1):
            patient('arm'+str(sign),[-.25,sign*.21,-.15],[.28,sign*.21,-.15],.055)
            patient('leg'+str(sign),[.20,sign*.09,-.17],[.78,sign*.09,-.17],.075)
        def clinician(person,far=False):
            # Smooth, prescribed motion is independent of the controller.
            wave=(1-np.cos(2*np.pi*t/8))/2 if self.scenario else 0.
            sign=-1 if far else 1
            x=G[0]+(-.43 if far else .01)
            y=G[1]+.65 if far else max(.63,G[1]-.65)
            shift=np.array([(.04 if far else -.025)*wave,(-.055 if far else -.065)*wave,.025*wave])
            pelvis=np.array([x,y,.73]); chest=np.array([x,y+sign*.23,1.16])+shift*.45
            head=np.array([x,G[1]+(.16 if far else -.16),1.50])+shift
            color='#7291af' if far else '#4c8dba'
            cap(person,'torso',pelvis,chest,.155,color,'torso')
            cap(person,'neck',chest+[0,0,.03],head-[0,0,.06],.063,'#d7a282','neck')
            cap(person,'head',head,head,.12,'#dbad90','head')
            for side in (-1,1):
                shoulder=chest+np.array([side*.16,0,.025])
                elbow=np.array([x+side*.25,y+sign*.38,1.035])+shift*.35
                hand=np.array([x+side*.12,G[1]+(-.08 if not far else .08),G[2]+.10])+shift*.15
                cap(person,'upper_arm'+str(side),shoulder,elbow,.052,color,'arm')
                cap(person,'forearm'+str(side),elbow,hand,.043,color,'arm')
                cap(person,'hand'+str(side),hand,hand+np.array([0,sign*.035,-.015]),.035,'#dfcdb5','hand')
                hip=pelvis+np.array([side*.105,0,-.045]);knee=np.array([x+side*.12,y,.34]);foot=np.array([x+side*.13,y+sign*.045,-.035])
                cap(person,'thigh'+str(side),hip,knee,.075,'#44768f','leg')
                cap(person,'shin'+str(side),knee,foot,.055,'#44768f','leg')
        clinician('surgeon')
        if self.scenario==2:clinician('assistant',True)
        return parts

    @staticmethod
    def surface(parts):
        # Deterministic equal count per capsule (not a simulated depth camera).
        n=72;k=np.arange(n);z=1-2*(k+.5)/n;angle=k*np.pi*(3-np.sqrt(5))
        sphere=np.c_[np.sqrt(1-z*z)*np.cos(angle),np.sqrt(1-z*z)*np.sin(angle),z]
        cloud=[]
        for p in parts:
            a,b=np.array(p['a']),np.array(p['b']);axis=b-a;norm=np.linalg.norm(axis)
            if norm<1e-12:cloud.append(a+p['radius_m']*sphere);continue
            axis/=norm;radial=sphere-(sphere@axis)[:,None]*axis
            radial/=np.maximum(np.linalg.norm(radial,axis=1)[:,None],1e-12)
            # Half the samples on the cylinder, half on its two hemispheres.
            cyl=a+((np.arange(n//2)+.5)/(n//2))[:,None]*(b-a)+p['radius_m']*radial[:n//2]
            dirs=sphere[n//2:];ends=np.where((dirs@axis)[:,None]>=0,b,a)
            cloud.extend([cyl,ends+p['radius_m']*dirs])
        return np.concatenate(cloud)

    def equipment_points(self):
        result=[]
        for item in self.equipment['equipment']:
            for p in item['parts']:
                # Rendered cylinders have box envelopes; use box-envelope
                # surfaces for the synthetic observation, declared explicitly.
                half=np.array(p['size'])/2;points=[]
                for axis in range(3):
                    rest=[i for i in range(3) if i!=axis]
                    for side in (-1,1):
                        for u in (-.65,.65):
                            for v in (-.65,.65):
                                x=np.zeros(3);x[axis]=side*half[axis];x[rest]=[u*half[rest[0]],v*half[rest[1]]];points.append(x)
                c,s=np.cos(p['yaw']),np.sin(p['yaw']);R=np.array([[c,-s,0],[s,c,0],[0,0,1]])
                result.append(np.array(points)@R.T+np.array(p['center']))
        return np.concatenate(result)

    def points(self,t):
        p=np.concatenate([self.surface(self.pose(t)),self._equipment_points])
        return p[(p[:,2]>=self.target[2]+.2)&(p[:,2]<=1.75)]

    def occlusion(self,camera,t):
        # 315 equally weighted rays to a declared 200 x 120 mm surgical field.
        xx,yy=np.meshgrid(np.linspace(-.10,.10,21),np.linspace(-.06,.06,15))
        ends=self.target+np.c_[xx.ravel(),yy.ravel(),np.zeros(xx.size)]
        starts=np.broadcast_to(camera,ends.shape);masked=np.zeros(len(ends),bool);head=masked.copy()
        for p in self.pose(t):
            hit=segment_segment_distance(starts,ends,p['a'],p['b'])<p['radius_m']-1e-7
            masked|=hit
            if p['category']=='head':head|=hit
        for item in self.equipment['equipment']:
            for p in item['parts']:
                c,s=np.cos(p['yaw']),np.sin(p['yaw']);R=np.array([[c,s,0],[-s,c,0],[0,0,1]])
                a=(starts-p['center'])@R.T;v=(ends-starts)@R.T;half=np.array(p['size'])/2
                parallel=np.abs(v)<1e-12
                with np.errstate(divide='ignore',invalid='ignore'):
                    lo=(-half-a)/v;hi=(half-a)/v
                low=np.where(parallel,-np.inf,np.minimum(lo,hi));high=np.where(parallel,np.inf,np.maximum(lo,hi))
                enter=np.maximum(0,low.max(1));leave=np.minimum(1,high.min(1))
                hit=(enter<leave-1e-9)&~np.any(parallel&(np.abs(a)>half),axis=1)
                masked|=hit
        return dict(field_occlusion_pct=float(masked.mean()*100),head_occlusion_pct=float(head.mean()*100))


class HumanClearance:
    def __init__(self,robot,scene):
        self.robot=robot;self.scene=scene;self.target=scene.target
        self.equipment=Clearance(robot,scene.equipment,scene.target)
        self.n_equipment=len(robot.r)*len(self.equipment.parts)
        self.parts=scene.pose(0)
        self.margins=np.r_[np.full(self.n_equipment,D_SAFE),np.full(len(robot.r)*len(self.parts),HUMAN_MARGIN)]

    def distances(self,q,t):
        a,b=self.robot.world(q,self.target);parts=self.scene.pose(t)
        c=np.array([p['a'] for p in parts]);d=np.array([p['b'] for p in parts]);r=np.array([p['radius_m'] for p in parts])
        ds=segment_segment_distance(a[:,None,:],b[:,None,:],c[None,:,:],d[None,:,:])-self.robot.r[:,None]-r[None,:]
        return np.r_[self.equipment.distances(q),ds.ravel()]

    def h(self,q,t):return self.distances(q,t)-self.margins

    def derivatives(self,q,t):
        eps=1e-5;eye=np.eye(3)*eps;h=self.h(q,t)
        jac=np.column_stack([(self.h(q+d,t)-self.h(q-d,t))/(2*eps) for d in eye])
        ht=(self.h(q,t+eps)-self.h(q,t-eps))/(2*eps)
        return h,jac,ht

    def metrics(self,q,t):
        ds=self.distances(q,t);de=ds[:self.n_equipment];dh=ds[self.n_equipment:]
        er,eo=divmod(int(de.argmin()),len(self.equipment.parts));hr,ho=divmod(int(dh.argmin()),len(self.parts))
        return dict(equipment_clearance_m=float(de.min()),human_clearance_m=float(dh.min()),
            closest_equipment=PARTS[er][0]+' / '+self.equipment.parts[eo]['id'],
            closest_human=PARTS[hr][0]+' / '+self.parts[ho]['id'])
