"""Original seven-part STL assembly from Myclass.py; visualization only.

No mesh simplification or replacement geometry is used. The compressed payload
contains the exact archived STL bytes, and assembly retains the source offsets.
"""
import base64
import gzip
import json
import struct
from pathlib import Path
import numpy as np
from .data import ROOT, configuration, sha256
from .geometry import HEIGHT, transform

PARTS = (
    ("base", "底盘与固定立柱", "base.STL"),
    ("lift", "升降立柱", "lift.STL"),
    ("arm1", "第一节机械臂", "arm1.STL"),
    ("arm2", "第二节机械臂", "arm2.STL"),
    ("arm3", "第三节机械臂", "arm3.STL"),
    ("platform1", "一级云台", "platform1new.STL"),
    ("platform2", "二级云台", "platform2new.STL"),
)
MANIFEST = ROOT / "data" / "robot_model_manifest.json"
PAYLOAD = ROOT / "data" / "robot_model_payload.json"


def assembly_transforms(q, xy, target, height=HEIGHT):
    """Seven row-major matrices in metres, matching source VTK T*Rz*Ry*Rx."""
    q = np.asarray(q, dtype=float)
    angles = np.cumsum(q)
    direction = np.asarray(target, dtype=float) - np.r_[xy, height]
    direction /= np.linalg.norm(direction)
    c, s = np.cos(angles[2]), np.sin(angles[2])
    local = np.array([c*direction[0]+s*direction[1],
                      -s*direction[0]+c*direction[1], direction[2]])
    pan = -np.arctan(local[0]/local[2])
    tilt = np.arcsin(np.clip(local[1], -1, 1))
    lift = height - (.142 + 1.0165 + .0625 + .001 - .0245)
    j2 = np.array([-.7*np.sin(angles[0]), .7*np.cos(angles[0])])
    j3 = j2 + [-.5*np.sin(angles[1]), .5*np.cos(angles[1])]
    p1 = j3 + [-.4035*np.sin(angles[2]), .4035*np.cos(angles[2])]
    p2 = j3 + [-.5145*np.sin(angles[2]), .5145*np.cos(angles[2])]
    z1 = lift + .142 + 1.0165
    z2 = z1 + .0625
    z3 = z2 + .001
    positions = [[0,0,0], [0,0,lift+.142], [0,0,z1],
                 [*j2,z2], [*j3,z3], [*p1,z3-.0245], [*p2,z3-.0245]]
    rotations = [0, 0, *angles, angles[2], angles[2]]
    matrices = [transform(t, [0,0,1], a) for t,a in zip(positions,rotations)]
    matrices[5] = matrices[5] @ transform([0,0,0], [0,1,0], -pan)
    matrices[6] = matrices[6] @ transform([0,0,0], [0,1,0], -pan) @ transform([0,0,0], [1,0,0], tilt)
    return [m.tolist() for m in matrices]


def prepare_robot_assets():
    archive = Path(configuration()["archive_root"])
    source = archive / "myenv" / "Myclass.py"
    directory = archive / "mytest" / "robot_model0412"
    source_hashes = {name:sha256(directory/name) for _,_,name in PARTS}
    if MANIFEST.exists() and PAYLOAD.exists():
        cached = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if (cached["source_sha256"] == sha256(source)
            and {p["filename"]:p["sha256"] for p in cached["parts"]} == source_hashes
            and cached["payload_sha256"] == sha256(PAYLOAD)):
            return cached
    parts, payload = [], []
    dtype = np.dtype([("normal","<f4",(3,)),("vertices","<f4",(3,3)),("attribute","<u2")])
    for key, label, name in PARTS:
        path = directory/name
        raw = path.read_bytes()
        triangles = struct.unpack_from("<I",raw,80)[0]
        if len(raw) != 84+triangles*50:
            raise ValueError(f"Not a complete binary STL: {path}")
        vertices = np.frombuffer(raw,dtype=dtype,offset=84)["vertices"].reshape(-1,3)
        parts.append(dict(key=key,label=label,filename=name,source=str(path),sha256=source_hashes[name],
                          bytes=len(raw),triangles=triangles,
                          bounds_mm=[vertices.min(axis=0).tolist(),vertices.max(axis=0).tolist()]))
        payload.append(dict(key=key,label=label,triangles=triangles,
                            gzip_base64=base64.b64encode(gzip.compress(raw,compresslevel=6,mtime=0)).decode("ascii")))
    PAYLOAD.write_text(json.dumps(payload,ensure_ascii=False,separators=(",",":")),encoding="utf-8")
    manifest = dict(source=str(source),source_sha256=sha256(source),
                    assembly_lines=[1221,1270],source_units="mm",viewer_units="m",parts=parts,
                    total_triangles=sum(p["triangles"] for p in parts),payload_sha256=sha256(PAYLOAD),
                    mesh_processing="Lossless gzip of original STL bytes; all triangles retained; mm to m at display.",
                    assembly="Original Myclass.py render translations and VTK default T*Rz*Ry*Rx; original camera pan and tilt.",
                    scope="Visual assembly only. Original native kinematics and collision unchanged. Display colours and lighting are illustrative, not STL source colours.")
    MANIFEST.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    return manifest


def render_viewer(payload, output):
    manifest = prepare_robot_assets()
    template = (ROOT/"viewer"/"template.html").read_text(encoding="utf-8")
    bundle = (ROOT/"viewer"/"robot-viewer.bundle.js").read_text(encoding="utf-8")
    values = {"__SIMULATION_DATA__":json.dumps(payload,ensure_ascii=False,allow_nan=False).replace("</","<\\/"),
              "__ROBOT_MODEL_DATA__":PAYLOAD.read_text(encoding="utf-8"),
              "__PAPER_FIGURE_DATA__":"data:image/png;base64,"+base64.b64encode((ROOT/"data"/"paper_figure12.png").read_bytes()).decode("ascii"),
              "__ROBOT_VIEWER_BUNDLE__":bundle.replace("</script", "<\\/script")}
    for marker,value in values.items():
        template = template.replace(marker,value)
    Path(output).write_text(template,encoding="utf-8")
    Path(output).with_name("robot_model_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    return manifest
