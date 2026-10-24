"""Check packaged kinematics, collision and camera geometry without robot hardware."""
import json
from pathlib import Path
import sys
import numpy as np

SIM=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SIM))
from ncsim import native_backend,historical_backend
from ncsim.geometry import INITIAL_Q

def main():
    historical_backend.configure(SIM/'outputs/historical_reproduction_20261002')
    footprints=native_backend.forward(INITIAL_Q)
    assert footprints.shape==(4,2) and np.isfinite(footprints).all()
    q,ok=historical_backend.inverse([0.1,1.2],INITIAL_Q)
    assert ok and np.isfinite(q).all(),'Historical inverse kinematics failed'
    camera=historical_backend.camera_geometry(q,[0.1,1.2],np.array([0.0,0.9,0.8]))
    assert all(np.isfinite(x).all() for x in camera)
    collision=native_backend.self_collision(q)
    print(json.dumps({'native_loading':'passed','historical_ik':'passed','camera_geometry':'passed',
                      'self_collision':collision,'backend':native_backend.provenance()},indent=2))

if __name__=='__main__':main()
