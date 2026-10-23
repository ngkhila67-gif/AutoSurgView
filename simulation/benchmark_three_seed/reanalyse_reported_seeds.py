"""Reanalyse the current reported three-seed benchmark using read-only numerical records."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'outputs/reanalysis')
args=parser.parse_args()
ROOT=Path(__file__).resolve().parent
W=args.out.resolve();W.mkdir(parents=True,exist_ok=True)
SIM=ROOT.parent
sys.path.insert(0,str(SIM))
from ncsim import publication_figures as pf
OLD=ROOT/'results';V2=OLD
OUT=W/'analysis';D=W/'data';D.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True);(OUT/'source_data').mkdir(exist_ok=True)
SEEDS=[29,47,83];METHODS=['pointcloud','robotstate','fusion']
NAMES={'mmppo':'fusion','pc':'pointcloud','state':'robotstate'}
REV={v:k for k,v in NAMES.items()}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
inputs=[OLD/'episodes.csv.gz',OLD/'seed_level.csv',V2/'analysis_v3.json',V2/'final_state_O.csv.gz']
digests={str(p):sha(p) for p in inputs}

def tint(x):
 x=np.asarray(x,float);h=stats.t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x))
 return [float(x.mean()-h),float(x.mean()+h)]

def seed_level(e):
 rows=[]
 for (m,s),g in e.groupby(['method','training_seed']):
  rows.append(dict(method=m,seed=int(s),n=len(g),macro_success=100*g.groupby('scene_index').success.mean().mean(),macro_improved=100*g.groupby('scene_index').improved.mean().mean(),macro_strict_success=100*g.groupby('scene_index').strict_success.mean().mean(),mean_final=g.final_score.mean(),mean_reduction_pct=100*g.reduction_fraction.mean(),mean_decisions=g.decision_steps.mean(),self_collision=int((g.termination_reason=='self_collision').sum()),step_limit=int((g.termination_reason=='step_limit').sum()),repeated_unreachable=int((g.termination_reason=='repeated_unreachable').sum()),mean_path_all=g.path_length_m.mean(),mean_path_success=g[g.success].path_length_m.mean(),improved_episodes=int(g.improved.sum())))
 return pd.DataFrame(rows)

d=pd.read_csv(OLD/'episodes.csv.gz');d=d[d.training_seed.isin(SEEDS)].copy()
assert len(d)==15920*9 and d.groupby(['method','training_seed']).size().eq(15920).all()
for c in ['eligible','success','strict_success','self_collision','improved']:
 if d[c].dtype!=bool:d[c]=d[c].astype(str).eq('True')
e=d[d.eligible].copy();assert len(e)==11647*9
assert set(e.method)==set(METHODS) and set(e.training_seed)==set(SEEDS)
groups=sorted(e.capture_group.unique());assert len(groups)==47
assert e.groupby(['method','training_seed']).size().eq(11647).all()
d.to_csv(OUT/'episodes.csv.gz',index=False,compression='gzip')
S=seed_level(e);S.to_csv(OUT/'seed_level.csv',index=False)
original=pd.read_csv(OLD/'seed_level.csv');original=original[original.seed.isin(SEEDS)].sort_values(['method','seed']).reset_index(drop=True)
assert (S[['method','seed','n']].to_numpy()==original[['method','seed','n']].to_numpy()).all()
assert np.allclose(S.drop(columns=['method','seed','n']),original.drop(columns=['method','seed','n']),rtol=1e-11,atol=1e-11)
BOOT=np.random.default_rng(pf.SEED).multinomial(47,np.full(47,1/47),size=pf.B)
pf.OUT=OUT;pf.style()
for fn in [pf.fig15,pf.fig16,pf.fig17]:plt.close(fn(e,BOOT))
per=e.groupby(['method','scene_index']).agg(eligible=('case_id','nunique'),success_percent=('success','mean'),improved_percent=('improved','mean')).reset_index()
per[['success_percent','improved_percent']]*=100;per.to_csv(OUT/'per_scene_summary.csv',index=False)
summary={m:{k:dict(mean=float(g[k].mean()),sd=float(g[k].std(ddof=1))) for k in S.columns if k not in ['method','seed','n']} for m,g in S.groupby('method')}
agg=e.groupby(['method','scene_index','group_index'])[['strict_success','success']].mean().reset_index()
cf={};bt={}
for m in METHODS:
 a=agg[agg.method==m];num=a.groupby('group_index').strict_success.sum().reindex(range(47),fill_value=0).to_numpy();den=a.groupby('group_index').size().reindex(range(47),fill_value=0).to_numpy()
 bt[m]=100*(BOOT@num)/(BOOT@den)
 cf[m]=dict(estimate=float(100*a.strict_success.mean()),ci95=pf.ci(bt[m]).tolist(),seeds=S[S.method==m].macro_strict_success.tolist())
gains={}
for m in ['pointcloud','robotstate']:
 x=S[S.method=='fusion'].set_index('seed').macro_strict_success-S[S.method==m].set_index('seed').macro_strict_success
 cf['fusion_minus_'+m]=dict(estimate=float(x.mean()),ci95=pf.ci(bt['fusion']-bt[m]).tolist(),t95=tint(x),per_seed=x.tolist(),seeds_favouring_mmppo=int((x>0).sum()))
 y=S[S.method=='fusion'].set_index('seed').macro_success-S[S.method==m].set_index('seed').macro_success
 gains[m]=dict(per_seed=y.tolist(),mean=float(y.mean()),t95=tint(y),seeds_favouring_mmppo=int((y>0).sum()))
pooled={m:dict(success=float(100*g.success.mean()),strict=float(100*g.strict_success.mean())) for m,g in e.groupby('method')}
benchmark=dict(analysis_scope='Current reported analysis',selected_seeds=SEEDS,n_policies=9,n_eligible_per_policy=11647,summary=summary,gains_over_single_inputs=gains,collision_free=cf,pooled_rates=pooled,fig15=pd.read_csv(OUT/'source_data/Fig15_estimates_and_intervals.csv').to_dict('records'))
(OUT/'benchmark_retrained.json').write_text(json.dumps(benchmark,indent=2),'utf8')

# Secondary image-area endpoints use the recorded final states, not new controller runs.
oldA=json.loads((V2/'analysis_v3.json').read_text('utf8'))
R=pd.read_csv(V2/'final_state_O.csv.gz');R=R[(R.source=='retrained')&R.training_seed.isin(SEEDS)].copy()
R=R.drop(columns=['self_collision','group_index'],errors='ignore')
R['configuration']=R.method.map(NAMES)
R=R.merge(e[['case_id','method','training_seed','self_collision','group_index']],left_on=['case_id','configuration','training_seed'],right_on=['case_id','method','training_seed'],validate='one_to_one')
assert len(R)==11647*9
R['rec']=R.O_final<=20;R['rec_cf']=R.rec&~R.self_collision
def endpoint(key,old_key):
 out=dict(per_seed={},summary={},paired={},archived=oldA[old_key]['archived'])
 bootstrap={}
 for short,m in NAMES.items():
  data=R[R.configuration==m]
  values=[float(100*data[data.training_seed==s].groupby('scene_id')[key].mean().mean()) for s in SEEDS]
  out['per_seed'][short]=values;out['summary'][short]=dict(mean=float(np.mean(values)),sd=float(np.std(values,ddof=1)))
  a=data.groupby(['scene_id','group_index'])[key].mean().reset_index();num=a.groupby('group_index')[key].sum().reindex(range(47),fill_value=0).to_numpy();den=a.groupby('group_index').size().reindex(range(47),fill_value=0).to_numpy()
  bootstrap[short]=100*(BOOT@num)/(BOOT@den)
 for other in ['pc','state']:
  x=np.asarray(out['per_seed']['mmppo'])-np.asarray(out['per_seed'][other])
  out['paired'][other]=dict(per_seed=x.tolist(),mean=float(x.mean()),t95=tint(x),boot95=pf.ci(bootstrap['mmppo']-bootstrap[other]).tolist(),favour=int((x>0).sum()))
 return out
A=dict(analysis_scope='Current reported analysis',selected_seeds=SEEDS,per_seed={},summary={},paired={},archived=oldA['archived'])
for m in ['mmppo','pc','state']:
 A['per_seed'][m]=[row for row in oldA['per_seed'][m] if row['seed'] in SEEDS]
 df=pd.DataFrame(A['per_seed'][m]);assert len(df)==3
 A['summary'][m]={k:dict(mean=float(df[k].mean()),sd=float(df[k].std(ddof=1))) for k in df.columns if k not in ['seed','n']}
 for s in SEEDS:
  src=S[(S.method==NAMES[m])&(S.seed==s)].iloc[0]
  rec=next(r for r in A['per_seed'][m] if r['seed']==s)
  assert abs(src.macro_strict_success-rec['recovery_cf'])<1e-10 and abs(src.mean_final-rec['final_score'])<1e-10
A['secondary_O_le_20']=endpoint('rec','secondary_O_le_20')
A['posthoc_O_le_20_collision_free']=endpoint('rec_cf','posthoc_O_le_20_collision_free')
(OUT/'analysis_v3.json').write_text(json.dumps(A,indent=2),'utf8')
R.to_csv(OUT/'final_state_O_selected.csv',index=False)
rows=[]
for panel,key in [('a','recovery_cf'),('b','final_score'),('c','O')]:
 for short,m in NAMES.items():
  for s in SEEDS:
   value=A['secondary_O_le_20']['per_seed'][short][SEEDS.index(s)] if key=='O' else next(r[key] for r in A['per_seed'][short] if r['seed']==s)
   rows.append(dict(panel=panel,measure=key,configuration=m,seed=s,value=value))
pd.DataFrame(rows).to_csv(D/'Fig3a-c.csv',index=False)
strata=pd.read_csv(OUT/'source_data/Fig17_stratified_estimates.csv');strata['starts_per_policy']=strata.n//3
assert (strata.n%3==0).all();strata.to_csv(D/'Fig3d.csv',index=False)
fast={m:float(100*((g.decision_steps<12)&g.success).mean()) for m,g in e.groupby('method')}
check=dict(selected_seeds=SEEDS,n_policies=9,n_candidate_episodes=len(d),n_analysis_episodes=len(e),n_per_configuration=11647*3,primary=cf,fast_lt12=fast,secondary=A['secondary_O_le_20'],sensitivity=A['posthoc_O_le_20_collision_free'],final_checkpoint_cf={m:A['summary'][m]['final_ckpt_recovery_cf'] for m in ['mmppo','pc','state']},source_sha256=digests)
assert all(sha(Path(p))==digest for p,digest in digests.items())
(OUT/'checks.json').write_text(json.dumps(check,indent=2),'utf8')
print(json.dumps({k:check[k] for k in ['selected_seeds','n_policies','n_candidate_episodes','n_analysis_episodes','primary','fast_lt12','final_checkpoint_cf']},indent=2),flush=True)

