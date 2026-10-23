"""Time-varying human + static equipment CBF, simulation only."""
import numpy as np
import osqp
from scipy import sparse
from .cbf_safety import DT,V_MAX,ALPHA
from . import native_backend as nb


def filter_velocity(q,u_nom,clearance,t):
    h,jac,ht=clearance.derivatives(q,t)
    diag=dict(status='unsafe_initial_state',correction=float(np.linalg.norm(u_nom)),residual=None,backtrack=0,feasible=False)
    if h.min()<-1e-7 or nb.self_collision(q):return np.zeros(3),diag
    solver=osqp.OSQP()
    solver.setup(P=sparse.eye(3,format='csc'),q=-np.asarray(u_nom),
        A=sparse.csc_matrix(np.vstack([jac,np.eye(3)])),
        l=np.r_[-ALPHA*h-ht,np.full(3,-V_MAX)],u=np.r_[np.full(len(h),np.inf),np.full(3,V_MAX)],
        verbose=False,eps_abs=1e-8,eps_rel=1e-8,max_iter=10000,polishing=False)
    result=solver.solve(raise_error=False)
    if result.info.status_val not in (1,2) or result.x is None or not np.isfinite(result.x).all():
        diag['status']='QP_infeasible_or_failed';return np.zeros(3),diag
    candidate=np.array(result.x)
    for k in range(13):
        u=candidate/(2**k);residual=float(np.min(jac@u+ht+ALPHA*h))
        if residual<-2e-6 or np.max(np.abs(u))>V_MAX+2e-6:continue
        safe=all(clearance.h(q+f*DT*u,t+f*DT).min()>=-1e-7 and not nb.self_collision(q+f*DT*u) for f in (.2,.4,.6,.8,1.))
        if safe:return u,dict(status='solved',correction=float(np.linalg.norm(u-u_nom)),residual=residual,backtrack=k,feasible=True)
    # A hold is NOT a certified safe response to an approaching person.
    diag['status']='no_verified_dynamic_command';return np.zeros(3),diag
