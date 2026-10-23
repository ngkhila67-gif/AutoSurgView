import argparse
import json
import webbrowser
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description="NC surgical-camera simulation")
    commands=parser.add_subparsers(dest="command",required=True)
    commands.add_parser("prepare",help="Index and verify original point-cloud sources")
    for name in ("demo","evaluate"):
        p=commands.add_parser(name,help="Run matched starts across archived policies and references")
        p.add_argument("--scenes-per-session",type=int,default=3)
        p.add_argument("--starts",type=int,default=2)
    p=commands.add_parser("train",help="Train a NEW policy; never overwrites original models")
    p.add_argument("--steps",type=int,default=2048)
    p.add_argument("--mode",choices=["fusion","only_pc","only_state"],default="fusion")
    p.add_argument("--seed",type=int,default=20261001)
    p.add_argument("--smoke",action="store_true")
    commands.add_parser("open",help="Open the latest interactive replay")
    commands.add_parser("open-paper",help="Open the complete 398 x 40 evaluation report")
    commands.add_parser("open-historical",help="Open the recovered historical paper results and replication audit")
    commands.add_parser("open-figures",help="Open the manuscript figures based only on the new full evaluation")
    p=commands.add_parser("paper-evaluate",help="Frozen 398 scenes x 40 starts x 3 archived models")
    p.add_argument("--output",type=str)
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--scene-limit",type=int,help="Pilot only; omit for all 398 scenes")
    args=parser.parse_args()
    if args.command=="prepare":
        from ncsim.data import prepare_index
        m=prepare_index()
        print(f"Verified {len(m['scenes'])} scenes")
    elif args.command in ("demo","evaluate"):
        if args.scenes_per_session<1 or args.starts<1:
            parser.error("Counts must be positive")
        from ncsim.evaluate import run_evaluation
        out,summary=run_evaluation(args.scenes_per_session,args.starts)
        print(json.dumps(summary,ensure_ascii=False,indent=2))
        print("Output:",out)
    elif args.command=="paper-evaluate":
        if args.workers<1 or (args.scene_limit is not None and not 1<=args.scene_limit<=398):
            parser.error("Invalid workers or scene limit")
        from ncsim.paper_evaluation import run
        output=run(args.output,args.workers,args.scene_limit)
        if args.scene_limit is None:
            from ncsim.paper_report import publish
            publish(output)
        print("Output:",output)
    elif args.command=="train":
        if args.steps<1:
            parser.error("Steps must be positive")
        from ncsim.training import train
        print("Output:",train(args.steps,args.mode,args.seed,args.smoke))
    else:
        from ncsim.data import ROOT
        report_commands={"open-paper":"latest_paper.json","open-historical":"latest_historical.json","open-figures":"latest_figures.json"}
        p=ROOT/"outputs"/report_commands.get(args.command,"latest.json")
        if not p.exists():
            parser.error(f"No saved result found: {p}")
        viewer=Path(json.loads(p.read_text(encoding="utf-8"))["report" if args.command in report_commands else "viewer"])
        webbrowser.open(viewer.as_uri())


if __name__=="__main__":
    main()
