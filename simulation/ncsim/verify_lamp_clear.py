"""Check saved v4 display connections, paired starts and moved-lamp geometry."""
import json
from pathlib import Path
import numpy as np
from .data import ROOT, sha256
from .cbf_safety import segment_box_distance


def run():
    old=ROOT/'outputs/cbf_human_lamp_close_20261002'
    out=ROOT/'outputs/cbf_human_lamp_clear_20261008'
    cat=json.loads((out/'catalog.json').read_text(encoding='utf-8'))['cases']
    protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    previous=json.loads((old/'protocol.json').read_text(encoding='utf-8'))
    assert protocol['selected']==previous['selected']
    reports=[]
    def connected(a,b):
        assert a['yaw']==b['yaw']==0
        return bool(np.all(np.abs(np.array(a['center'])-b['center']) <= (np.array(a['size'])+b['size'])/2+1e-9))
    def inside(point,box):
        c,s=np.cos(box['yaw']),np.sin(box['yaw'])
        rot=np.array([[c,s,0],[-s,c,0],[0,0,1]])
        return bool(np.all(np.abs(rot@(np.array(point)-box['center'])) <= np.array(box['size'])/2+1e-9))
    for entry in cat:
        case=json.loads((out/entry['file']).read_text(encoding='utf-8'))
        before=json.loads((old/entry['file']).read_text(encoding='utf-8'))
        parts={p['id']:p for e in case['equipment']['equipment'] for p in e['parts']}
        oldparts={p['id']:p for e in before['equipment']['equipment'] for p in e['parts']}
        for a,b in [('monitor_stand','monitor_screen'),('monitor_stand','monitor_mount'),('monitor_mount','monitor_screen'),
                    ('anaesthesia_body','anaesthesia_screen_mount'),('anaesthesia_screen_mount','anaesthesia_screen')]:
            assert connected(parts[a],parts[b]),(entry['case_id'],a,b)
        neck=parts['lamp_neck'];a=np.array(neck['center']);b=a.copy()
        a[2]-=neck['size'][2]/2;b[2]+=neck['size'][2]/2
        assert inside(a,parts['lamp_head']) and inside(b,parts['lamp_arm'])
        pole=parts['lamp_pole'];assert inside(np.array(pole['center'])+[0,0,pole['size'][2]/2],parts['lamp_arm'])
        assert np.allclose(np.array(parts['lamp_head']['center'])-oldparts['lamp_head']['center'],[0,.4,0])
        measure={}
        for method,run_data in case['runs'].items():
            assert run_data['trace'][0]['q']==before['runs'][method]['trace'][0]['q']
            assert run_data['summary']['integration_steps']==120
            old_by_time={f['simulation_time_s']:f for f in before['runs'][method]['trace']}
            for f in run_data['trace']:
                if f['simulation_time_s'] in old_by_time:
                    assert f['humans']==old_by_time[f['simulation_time_s']]['humans']
            distances={}
            for version,data,geometry in [('old',before['runs'][method],oldparts),('new',run_data,parts)]:
                camera=np.array([f['camera'] for f in data['trace']])
                lamp=geometry['lamp_head'];local=camera-lamp['center']
                distances[version]=float(segment_box_distance(local,local,np.array(lamp['size'])/2).min())
            assert distances['new']>distances['old'],(entry['case_id'],method,distances)
            measure[method]=distances
        reports.append(dict(case_id=entry['case_id'],connected_supports=True,lamp_move_m=[0,.4,0],
                            min_camera_origin_to_lamp_head_box_m=measure))
    old_hashes=json.loads((ROOT/'outputs/human_detail_20261008/data_hashes_before.json').read_text(encoding='utf-8'))
    for record in old_hashes:assert sha256(ROOT/record['file'])==record['sha256']
    result=dict(cases=len(cat),paired_runs=2*len(cat),same_starts_and_motion=True,
                supports_connected=True,old_data_files_unchanged=len(old_hashes),checks=reports,
                distance_note='Layout check uses camera optical origin to lamp-head box; full robot envelopes are checked by the CBF evaluation.')
    (out/'layout_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))


if __name__=='__main__':run()
