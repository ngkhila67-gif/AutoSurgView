"""Second requested lamp adjustment; archived previous lamp layout unchanged."""
import numpy as np
from .human_lamp_scene import NearFieldHumanScene


class CloseLampHumanScene(NearFieldHumanScene):
    def __init__(self,target,scenario):
        super().__init__(target,scenario)
        lamp=next(e for e in self.equipment['equipment'] if e['id']=='lamp')
        parts={p['id']:p for p in lamp['parts']}
        parts['lamp_head']['center'][2]=1.90
        parts['lamp_light']['center'][2]=1.845
        parts['lamp_neck']['center'][2]=(2.65+1.95)/2
        parts['lamp_neck']['size'][2]=2.65-1.95
        # The archived camera/arm occupies the near side at z~1.8 m. Move the
        # lower head toward the far side instead of intersecting that workspace.
        for key in ('lamp_head','lamp_light','lamp_neck'):parts[key]['center'][1]+=.20
        head=np.array(parts['lamp_head']['center'][:2]);pole=np.array(parts['lamp_pole']['center'][:2]);v=pole-head
        parts['lamp_arm']['center'][:2]=((head+pole)/2).tolist()
        parts['lamp_arm']['size'][0]=float(np.linalg.norm(v))+.05
        parts['lamp_arm']['yaw']=float(np.arctan2(v[1],v[0]))
        self.equipment['lamp_adjustment'].update(head_center=parts['lamp_head']['center'].copy(),
            lowered_m=.50,lowered_from_previous_m=.15,horizontal_offset_from_target_m=.20,
            light_surface_above_target_m=1.845-.0075-float(target[2]),
            light_bottom_center_to_target_m=float(np.hypot(1.845-.0075-float(target[2]),.20)))
        self._equipment_points=self.equipment_points()
