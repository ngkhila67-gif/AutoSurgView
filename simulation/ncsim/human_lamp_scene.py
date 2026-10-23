"""User-requested lower surgical lamp, keeping the v1 scene reproducible."""
import numpy as np
from .human_scene import HumanScene
from .cbf_safety import segment_box_distance


class NearFieldHumanScene(HumanScene):
    def __init__(self,target,scenario):
        super().__init__(target,scenario)
        gx,gy,gz=map(float,target)
        lamp=next(e for e in self.equipment['equipment'] if e['id']=='lamp')
        parts={p['id']:p for p in lamp['parts']}
        # Head moves 350 mm down and from (Gx+.27,Gy+.025) to directly over G.
        # Keep its original geometry, pole and upper joint height.
        parts['lamp_head']['center']=[gx,gy,2.05]
        parts['lamp_light']['center']=[gx,gy,1.995]
        parts['lamp_neck']['center']=[gx,gy,(2.65+2.10)/2]
        parts['lamp_neck']['size'][2]=2.65-2.10
        pole=np.array(parts['lamp_pole']['center'][:2]);head=np.array([gx,gy]);v=pole-head
        parts['lamp_arm']['center']=[*((pole+head)/2).tolist(),2.67]
        parts['lamp_arm']['size'][0]=float(np.linalg.norm(v))+.05
        parts['lamp_arm']['yaw']=float(np.arctan2(v[1],v[0]))
        self.equipment['lamp_adjustment']=dict(head_center=[gx,gy,2.05],lowered_m=.35,
            horizontal_offset_from_target_m=0.,light_surface_above_target_m=1.995-.0075-gz,
            provenance='User-requested simulation layout change; no measured clinical lamp geometry or illumination model')
        self._equipment_points=self.equipment_points()

    def lamp_human_clearance(self,t):
        people=self.pose(t)
        lamp=next(e for e in self.equipment['equipment'] if e['id']=='lamp')
        a=np.array([p['a'] for p in people]);b=np.array([p['b'] for p in people]);radius=np.array([p['radius_m'] for p in people])
        candidates=[]
        for p in lamp['parts']:
            c,s=np.cos(p['yaw']),np.sin(p['yaw']);R=np.array([[c,s,0],[-s,c,0],[0,0,1]])
            ds=segment_box_distance((a-p['center'])@R.T,(b-p['center'])@R.T,np.array(p['size'])/2)-radius
            i=int(ds.argmin());candidates.append((float(ds[i]),p['id']+' / '+people[i]['id']))
        return min(candidates)
