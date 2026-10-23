from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import re
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def configuration():
    return json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def natural_key(value):
    return tuple(int(p) if p.isdigit() else p for p in re.split(r"(\d+)", str(value)))


def read_pcd(path):
    """Read ASCII / uncompressed binary PCD, validating record count and fields."""
    with Path(path).open("rb") as f:
        header = {}
        while True:
            raw = f.readline()
            if not raw:
                raise ValueError(f"Missing DATA header: {path}")
            line = raw.decode("ascii").strip()
            if not line or line.startswith("#"):
                continue
            key, *values = line.split()
            header[key] = values
            if key == "DATA":
                break
        payload = f.read()
    fields = header["FIELDS"]
    sizes = list(map(int, header["SIZE"]))
    types = header["TYPE"]
    counts = list(map(int, header.get("COUNT", ["1"] * len(fields))))
    n = int(header.get("POINTS", [str(int(header["WIDTH"][0]) * int(header["HEIGHT"][0]))])[0])
    if not all(k in fields for k in ("x", "y", "z")):
        raise ValueError(f"PCD lacks xyz: {path}")
    if header["DATA"][0] == "binary":
        kind = {"F": "f", "I": "i", "U": "u"}
        dtype = np.dtype([(name, "<" + kind[t] + str(size), (count,))
                          for name, size, t, count in zip(fields, sizes, types, counts)])
        if len(payload) != n * dtype.itemsize:
            raise ValueError(f"PCD size mismatch: {path}")
        records = np.frombuffer(payload, dtype=dtype, count=n)
        xyz = np.column_stack([records[k][:, 0] for k in ("x", "y", "z")])
    elif header["DATA"][0] == "ascii":
        records = np.loadtxt(payload.decode("ascii").splitlines(), ndmin=2)
        if len(records) != n:
            raise ValueError(f"PCD count mismatch: {path}")
        offsets = np.cumsum([0] + counts[:-1])
        xyz = records[:, [offsets[fields.index(k)] for k in ("x", "y", "z")]]
    else:
        raise ValueError("Compressed PCD is not supported; no silent conversion.")
    if not np.isfinite(xyz).all():
        raise ValueError(f"PCD contains non-finite coordinates: {path}")
    return np.asarray(xyz, dtype=np.float64)


@dataclass
class Scene:
    scene_id: str
    session: str
    capture_group: str
    points: np.ndarray
    target: np.ndarray
    source: str
    source_sha256: str


def prepare_index(config=None):
    config = config or configuration()
    root = Path(config["archive_root"]) / "mytest" / "pointdata"
    rows, seen_hash = [], {}
    split_lookup = {session: split for split, sessions in config["new_training_split"].items() for session in sessions}
    for session in config["scene_sessions"]:
        for folder in sorted((root / session).glob("data*"), key=natural_key):
            if not folder.is_dir():
                continue
            labels = list(folder.glob("*.txt"))
            if len(labels) != 1:
                raise ValueError(f"Expected one target label: {folder}")
            target = np.loadtxt(labels[0]).reshape(-1)
            if target.shape != (3,) or not np.isfinite(target).all():
                raise ValueError(f"Invalid target: {labels[0]}")
            for pcd in sorted(folder.glob("*.pcd"), key=natural_key):
                digest = sha256(pcd)
                points = read_pcd(pcd)
                duplicate = seen_hash.get(digest)
                scene_id = f"{session}/{folder.name}/{pcd.stem}"
                rows.append(dict(scene_id=scene_id, session=session, capture_group=f"{session}/{folder.name}",
                                 source=str(pcd), source_sha256=digest, target=target.tolist(),
                                 target_source=str(labels[0]), target_sha256=sha256(labels[0]),
                                 point_count=len(points), duplicate_of=duplicate,
                                 new_training_split=split_lookup[session]))
                seen_hash.setdefault(digest, scene_id)
    # Hash duplicates across new splits would leak geometry and must be resolved first.
    memberships = {}
    for r in rows:
        memberships.setdefault(r["source_sha256"], set()).add(r["new_training_split"])
    if any(len(s) > 1 for s in memberships.values()):
        raise ValueError("Duplicate geometry crosses the proposed training split.")
    out = ROOT / "data"
    out.mkdir(exist_ok=True)
    manifest = dict(version=1, purpose="new_portable_simulation", scenes=rows,
                    archived_model_test_exposure="unknown: these scenes may have trained archived models",
                    split_note="Session-folder split for NEW training only; acquisition independence requires verification.")
    (out / "scene_index.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def scene_index():
    p = ROOT / "data" / "scene_index.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else prepare_index()


def load_scene(row):
    if sha256(row["source"]) != row["source_sha256"]:
        raise ValueError(f"Source changed since indexing: {row['source']}")
    if sha256(row["target_source"]) != row["target_sha256"]:
        raise ValueError("Target changed since indexing")
    return Scene(row["scene_id"], row["session"], row["capture_group"], read_pcd(row["source"]),
                 np.asarray(row["target"], dtype=float), row["source"], row["source_sha256"])


def demo_rows(per_session=3):
    rows = scene_index()["scenes"]
    selected = []
    for session in configuration()["scene_sessions"]:
        candidates = [r for r in rows if r["session"] == session and r["duplicate_of"] is None]
        groups = sorted({r["capture_group"] for r in candidates}, key=natural_key)
        for i in np.linspace(0, len(groups) - 1, min(per_session, len(groups)), dtype=int):
            group = [r for r in candidates if r["capture_group"] == groups[i]]
            selected.append(group[len(group)//2])
    return selected
