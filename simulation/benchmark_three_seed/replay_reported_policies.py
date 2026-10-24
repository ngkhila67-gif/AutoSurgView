"""Rerun the supplied manuscript policies on their fixed benchmark starts."""
import argparse
import json
from pathlib import Path
import torch
import modality3_study as study

ROOT=Path(__file__).resolve().parent
MODELS=ROOT.parents[1]/'models/benchmark_three_seed'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'outputs/policy_replay')
    parser.add_argument('--method',choices=study.MODES)
    parser.add_argument('--seed',type=int,choices=study.SEEDS)
    parser.add_argument('--scene-limit',type=int,help='Optional software check on the first N of 398 test scenes')
    args=parser.parse_args()
    if args.scene_limit is not None and not 1<=args.scene_limit<=398:parser.error('--scene-limit must be 1..398')
    if args.out.exists():parser.error('Output directory already exists; choose a new --out path')
    torch.set_num_threads(study.THREADS)
    study.configure(study.HISTORY)
    rows=study.read(ROOT/'inputs/ordered_dataset.json')[1592:]
    if args.scene_limit is not None:rows=rows[:args.scene_limit]
    cases=[];starts=study.read(ROOT/'inputs/starts.json')
    for row in rows:
        record={k:row[k] for k in ('scene_id','session','capture_group','source_sha256','target_sha256','target')}
        record['source_rel'],record['target_rel']=study.rel(row['source']),study.rel(row['target_source'])
        resolved=study.resolve(record)
        for name,hash_key in [('source','source_sha256'),('target_source','target_sha256')]:
            if study.sha256(resolved[name])!=row[hash_key]:raise ValueError('Point-cloud or target checksum mismatch: '+row['scene_id'])
        for start in starts:
            if start['scene_id']==row['scene_id']:cases.append(study.make_case(record,start))
    assert len(cases)==len(rows)*40
    methods=[args.method] if args.method else list(study.MODES)
    seeds=[args.seed] if args.seed else study.SEEDS
    args.out.mkdir(parents=True)
    results=[]
    for method in methods:
        for seed in seeds:
            path=MODELS/f'{method}_seed_{seed}/best_model.zip'
            policy=study.PPO.load(path,device='cpu')
            result=study.evaluate(policy,cases,args.out/f'{method}_seed_{seed}')
            results.append({'method':method,'seed':seed,'model_sha256':study.sha256(path),**result})
            print(json.dumps({'method':method,'seed':seed,'candidates':result['candidates'],'complete':True}),flush=True)
    study.dump(args.out/'replay_summary.json',{'test_scenes':len(rows),'starts_per_scene':40,'policies':len(results),
        'full_manuscript_benchmark':len(rows)==398 and len(results)==9,'training_performed':False,'results':results})

if __name__=='__main__':main()
