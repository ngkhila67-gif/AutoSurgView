"""Whole-dataset playback, with the archived mesh version matching the .5405 chain."""
from functools import lru_cache
from pathlib import Path
import base64,gzip,json,struct
import numpy as np
from .data import ROOT,configuration,sha256,load_scene
from .geometry import transform
from . import historical_backend as hb
from .paper_evaluation import read_scene,dump
from .manuscript_evaluation import HISTORY

PARTS=[('base','底盘','base.STL'),('lift','升降立柱','lift.STL'),('arm1','第一机械臂','arm1.STL'),('arm2','第二机械臂','arm2.STL'),('arm3','第三机械臂','arm3.STL'),('platform1','一级云台','platform1.STL'),('platform2','二级云台','platform2.STL')]


def prepare_assets(out):
    source=Path(configuration()['archive_root'])/'mytest';parts=[];payload=[]
    for key,label,name in PARTS:
        p=source/name;raw=p.read_bytes();n=struct.unpack_from('<I',raw,80)[0]
        assert len(raw)==84+50*n
        parts.append(dict(key=key,label=label,source=str(p),sha256=sha256(p),triangles=n))
        payload.append(dict(key=key,label=label,triangles=n,gzip_base64=base64.b64encode(gzip.compress(raw,mtime=0)).decode('ascii')))
    manifest=dict(parts=parts,total_triangles=sum(p['triangles'] for p in parts),
        assembly_source=str(source.parent/'myenv/Myclass_修改模型备份.py'),
        assembly_source_sha256=sha256(source.parent/'myenv/Myclass_修改模型备份.py'),
        assembly_lines=[1153,1202],coordinate_conversion='Original VTK (x,z,-.232-y) converted back to world (x,y,z); source units mm converted to m',
        geometry_note='Uses original meshes named by the archived .5405 environment. No mesh simplification. Original display pan anchor .4285 differs from scoring .4; retained and disclosed.')
    dump(out/'robot_model_manifest.json',manifest);dump(out/'robot_assets.json',payload)
    return manifest


def assembly(q,xy,target):
    """Literal translations/rotation order in archived render(), then rigid basis conversion."""
    theta=[0,*np.rad2deg(q),0,0,0]
    pan,tilt=hb._CAMERA['get_paltformangle'](xy,1.8,target,hb._CAMERA['get_T13'](theta))
    a=np.cumsum(q);s=np.sin(a);c=np.cos(a)
    lift=1.8-(.142+1.019+.0875+.0165)
    y1=lift+.142+1.019;y2=y1+.0875;y3=y2+.0165
    p2=[-.7*s[0],y2,-.232-.7*c[0]]
    p3=[-.7*s[0]-.5*s[1],y2,-.232-.7*c[0]-.5*c[1]]
    p4=[p3[0]-.4285*s[2],y3,p3[2]-.4285*c[2]]
    p5=[p3[0]-.5405*s[2],y3,p3[2]-.5405*c[2]]
    pos=[[0,0,0],[0,lift+.142,-.232],[0,y1,-.232],p2,p3,p4,p5]
    ang=[0,0,*a,a[-1],a[-1]]
    conversion=np.array([[1,0,0,0],[0,0,-1,-.232],[0,1,0,0],[0,0,0,1]],float)
    matrices=[]
    for i,(p,t) in enumerate(zip(pos,ang)):
        M=transform(p,[0,1,0],t)
        if i>=5:M=M@transform([0,0,0],[0,0,1],np.deg2rad(pan))
        if i==6:M=M@transform([0,0,0],[1,0,0],np.deg2rad(tilt))
        matrices.append((conversion@M).tolist())
    return matrices


class PlaybackStore:
    def __init__(self,out):
        self.out=Path(out);hb.configure(HISTORY)
        self.rows=json.loads((self.out/'ordered_dataset.json').read_text(encoding='utf-8'))[1592:]

    @lru_cache(maxsize=8)
    def record(self,i):return read_scene(self.out/'scenes'/f'scene_{i+1:03d}.json.gz')

    @lru_cache(maxsize=8)
    def scene(self,i):return load_scene(self.rows[i])

    @lru_cache(maxsize=24)
    def case(self,i,j):
        if not 0<=i<398 or not 0<=j<40:raise ValueError('Invalid scene or start')
        rec=self.record(i);scene=self.scene(i)
        points=scene.points[(scene.points[:,2]>=scene.target[2]+.2)&(scene.points[:,2]<=1.75)]
        case=dict(case_id=f'scene_{i+1:03d}_start_{j+1:02d}',scene_id=scene.scene_id,source=scene.source,
            source_sha256=scene.source_sha256,points=points.tolist(),runs={})
        for ep in rec['episodes']:
            if ep['summary']['start_index']!=j:continue
            frames=[]
            for f in ep['trace']:
                xy=np.array(f['xy']);q=np.array(f['q']);inside,score,_,geom=hb.classify(points,xy,scene.target,q)
                if not np.isclose(score,f['score'],rtol=1e-12,atol=1e-9):raise AssertionError('Replay geometry differs from evaluated geometry')
                frames.append(dict(**f,camera=geom[1].tolist(),corners=geom[3].tolist(),target=scene.target.tolist(),
                    inside_indices=np.flatnonzero(inside).tolist(),robot_transforms=assembly(q,xy,scene.target)))
            case['runs'][ep['summary']['method']]=dict(summary=ep['summary'],trace=frames)
        assert len(case['runs'])==3
        return case


def catalog(out):
    out=Path(out);rows=json.loads((out/'ordered_dataset.json').read_text(encoding='utf-8'))[1592:]
    first=None;clear=None;failure=None;entries=[]
    for i,row in enumerate(rows):
        r=read_scene(out/'scenes'/f'scene_{i+1:03d}.json.gz')
        f=[e['summary'] for e in r['episodes'] if e['summary']['method']=='fusion']
        for s in f:
            if first is None and s['success'] and 4<=s['decision_steps']<12:first=[i,s['start_index']]
            if clear is None and s['initially_clear']:clear=[i,s['start_index']]
            if failure is None and s['eligible'] and not s['success']:failure=[i,s['start_index']]
        entries.append(dict(index=i,scene_id=row['scene_id'],starts=[dict(index=s['start_index'],initial_score=s['initial_score'],reason=s['termination_reason']) for s in f]))
    return dict(scenes=entries,default_case=first or [0,0],clear_example=clear,failure_example=failure,
        summary=json.loads((out/'summary.json').read_text(encoding='utf-8')))


def publish(out):
    out=Path(out).resolve();manifest=prepare_assets(out);c=catalog(out);dump(out/'catalog.json',c)
    template=(ROOT/'viewer/manuscript.html').read_text(encoding='utf-8')
    template=template.replace('__CATALOG__',json.dumps(c,ensure_ascii=False).replace('</','<\\/'))
    (out/'正文对齐仿真.html').write_text(template,encoding='utf-8')
    dump(ROOT/'outputs/latest_manuscript.json',dict(directory=str(out),viewer=str(out/'正文对齐仿真.html')))
    return manifest
