"""Requested camera/lamp separation and connected monitor supports.

Changes belong to a new, explicit scene version. All added/edited solids are
used both in the display and in the equipment clearance/observation model.
"""
import numpy as np
from .human_lamp_close_scene import CloseLampHumanScene


class ClearLampHumanScene(CloseLampHumanScene):
    def __init__(self, target, scenario):
        super().__init__(target, scenario)
        equipment = {item['id']: item for item in self.equipment['equipment']}
        parts = {p['id']: p for p in equipment['lamp']['parts']}
        # Retain z=1.90 m; move horizontally away from the camera workspace.
        for key in ('lamp_head', 'lamp_light', 'lamp_neck'):
            parts[key]['center'][1] += .40
        head = np.array(parts['lamp_head']['center'][:2])
        pole = np.array(parts['lamp_pole']['center'][:2])
        vector = pole - head
        parts['lamp_arm']['center'][:2] = ((head + pole) / 2).tolist()
        parts['lamp_arm']['size'][0] = float(np.linalg.norm(vector)) + .05
        parts['lamp_arm']['yaw'] = float(np.arctan2(vector[1], vector[0]))
        self.equipment['lamp_adjustment'].update(
            head_center=parts['lamp_head']['center'].copy(),
            horizontal_offset_from_target_m=.60,
            horizontal_move_from_previous_m=.40,
            lowered_from_previous_m=0.,
            light_bottom_center_to_target_m=float(np.hypot(1.8375-float(target[2]), .60)))

        monitor = {p['id']: p for p in equipment['monitor']['parts']}
        stand, screen = monitor['monitor_stand'], monitor['monitor_screen']
        bottom = stand['center'][2] - stand['size'][2] / 2
        # The upright now reaches the screen centre, rather than stopping
        # 185 mm below the screen. It overlaps the casing to form a joint.
        top = screen['center'][2]
        stand['center'][2] = (bottom + top) / 2
        stand['size'][2] = top - bottom
        monitor_joint = dict(id='monitor_mount', shape='box', yaw=0.,
            center=[screen['center'][0], screen['center'][1]+.055, top-.045],
            size=[.12,.10,.12], color='#b6c6ce')
        equipment['monitor']['parts'].append(monitor_joint)

        anaesthesia = {p['id']: p for p in equipment['anaesthesia']['parts']}
        body, screen = anaesthesia['anaesthesia_body'], anaesthesia['anaesthesia_screen']
        bottom = body['center'][2] + body['size'][2]/2 - .015
        top = screen['center'][2] - screen['size'][2]/2 + .06
        equipment['anaesthesia']['parts'].append(dict(
            id='anaesthesia_screen_mount', shape='box', yaw=0.,
            center=[screen['center'][0], screen['center'][1]+.03, (bottom+top)/2],
            size=[.12,.10,top-bottom], color='#b6c6ce'))
        self.equipment['support_adjustment'] = {
            'monitor': 'Upright extends to screen centre; rear mounting block overlaps upright and screen.',
            'anaesthesia': 'Mounting block overlaps workstation top and screen bottom.',
            'scope': 'Parametric simulation fixtures, not measured device CAD.'}
        self._equipment_points = self.equipment_points()
