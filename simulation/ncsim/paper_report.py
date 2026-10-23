"""Audit, statistics and presentation for the frozen full-scale evaluation."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import csv
import gzip
import json
import numpy as np
from .data import ROOT, sha256, load_scene
from .paper_evaluation import METHODS, dump, write_rows, read_scene, validate_scene, PaperEnv
from .policies import LABELS


PAPER = dict(fusion=dict(success=.968,improved=.997,mean_final=11.2,mean_reduction=.933,mean_steps=7.),
             pointcloud=dict(success=.857,improved=.954,mean_final=26.1,mean_steps=7.8),
             robotstate=dict(success=.924,improved=.990,mean_steps=9.))


def statistics(values):
    n=len(values)
    eligible=[r for r in values if r["eligible"]]
    improved=[r for r in eligible if r["improved"]]
    success=[r for r in eligible if r["success"]]
    def mean(rows,key):
        return float(np.mean([r[key] for r in rows])) if rows else None
    result=dict(candidates=n,initially_clear=n-len(eligible),eligible=len(eligible),
        successes=len(success),success_fraction=mean(eligible,"success"),
        strict_success_fraction=mean(eligible,"strict_success"),improved=len(improved),
        improved_fraction=mean(eligible,"improved"),mean_final_score=mean(eligible,"final_score"),
        mean_reduction_fraction=mean(eligible,"reduction_fraction"),mean_decision_steps=mean(eligible,"decision_steps"),
        mean_final_score_improved=mean(improved,"final_score"),
        mean_reduction_fraction_improved=mean(improved,"reduction_fraction"),
        mean_decision_steps_improved=mean(improved,"decision_steps"),
        mean_decision_steps_success=mean(success,"decision_steps"),
        self_collision_cases=sum(r["self_collision"] for r in eligible),
        initial_self_collision_candidates=sum(r["initial_self_collision"] for r in values),
        repeated_unreachable_cases=sum(r["termination_reason"]=="repeated_unreachable" for r in eligible),
        step_limit_cases=sum(r["termination_reason"]=="step_limit" for r in eligible),
        fast_success_cases=sum(r["decision_steps"]<12 for r in success),
        fast_success_fraction=sum(r["decision_steps"]<12 for r in success)/len(eligible) if eligible else None,
        success_fraction_horizon41=mean(eligible,"success_horizon41"),
        additional_successes_horizon41=sum(r["success_horizon41"] and not r["success"] for r in eligible))
    return result


def summarize(out):
    out=Path(out)
    digest=sha256(out/"protocol.json")
    rows=[]
    checksums={}
    step_count=0
    with gzip.open(out/"steps.csv.gz","wt",encoding="utf-8-sig",newline="") as sf:
        writer=csv.DictWriter(sf,fieldnames=["case_id","scene_id","method","step","score","x_m","y_m",
            "q1_rad","q2_rad","q3_rad","action_x","action_y","reward","ik_ok","self_collision","reason"])
        writer.writeheader()
        for i in range(398):
            p=out/"scenes"/f"scene_{i+1:03d}.json.gz"
            record=read_scene(p)
            validate_scene(record,i,digest)
            checksums[p.name]=sha256(p)
            for ep in record["episodes"]:
                r=ep["summary"]
                rows.append(r)
                for f in ep["trace"]:
                    a=f["action"] or [None,None]
                    writer.writerow(dict(case_id=r["case_id"],scene_id=r["scene_id"],method=r["method"],
                        step=f["step"],score=f["score"],x_m=f["xy"][0],y_m=f["xy"][1],
                        q1_rad=f["q"][0],q2_rad=f["q"][1],q3_rad=f["q"][2],action_x=a[0],action_y=a[1],
                        reward=f["reward"],ik_ok=f["ik_ok"],self_collision=f["self_collision"],reason=f["reason"]))
                    step_count+=1
    if len(rows)!=47760:
        raise AssertionError("Missing trial records")
    paired={}
    for r in rows:
        paired.setdefault(r["case_id"],[]).append((r["method"],r["initial_x_m"],r["initial_y_m"],r["initial_score"]))
    assert len(paired)==15920
    assert all(len(v)==3 and len({x[1:] for x in v})==1 and {x[0] for x in v}==set(METHODS) for v in paired.values())
    summary=[]
    by_scene=[]
    for method in METHODS:
        values=[r for r in rows if r["method"]==method]
        s=dict(method=method,**statistics(values))
        scene_stats=[]
        for i in range(398):
            group=[r for r in values if r["scene_index"]==i]
            stats=dict(method=method,scene_index=i,scene_id=group[0]["scene_id"],**statistics(group))
            by_scene.append(stats)
            if stats["eligible"]:
                scene_stats.append(stats)
        s["scenes_with_occluded_starts"]=len(scene_stats)
        s["macro_scene_success_fraction"]=float(np.mean([r["success_fraction"] for r in scene_stats]))
        s["macro_scene_improved_fraction"]=float(np.mean([r["improved_fraction"] for r in scene_stats]))
        summary.append(s)
    write_rows(out/"episodes.csv",rows)
    write_rows(out/"summary.csv",summary)
    write_rows(out/"per_scene_summary.csv",by_scene)
    dump(out/"summary.json",summary)
    dump(out/"verification.json",dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        scene_files=398,candidate_starts=15920,models=3,model_trial_records=len(rows),
        trace_frames=step_count,actual_decision_steps=sum(r["decision_steps"] for r in rows),
        all_pairs_identical_initial_state=True,all_traces_match_summaries=True,
        maximum_primary_decision_steps=max(r["decision_steps"] for r in rows),
        protocol_sha256=digest,scene_file_sha256=checksums,
        summary_csv_sha256=sha256(out/"summary.csv"),episodes_csv_sha256=sha256(out/"episodes.csv"),
        steps_csv_gz_sha256=sha256(out/"steps.csv.gz")))
    dump(out/"progress.json",dict(status="evaluation_complete",completed_scenes=398,total_scenes=398,
        completed_model_trials=47760,updated_utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    return rows,summary


def audit_historical_logs(out):
    root=Path(configuration_archive()).parent/"主动规避遮挡"/"SCI对应"
    records=[]
    for method,name in dict(fusion="simulation_point_robot_occlusion_log.txt",
            pointcloud="simulation_only_point_occlusion_log.txt",robotstate="simulation_only_robot_occlusion_log.txt").items():
        path=root/name
        if not path.exists():
            continue
        traces=[]
        current=None
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if "遮挡规避开始" in line:
                current=[]
            elif "遮挡规避结束" in line:
                if current:
                    traces.append(current)
                current=None
            elif current is not None and "遮挡值:" in line:
                current.append(float(line.split(":")[-1]))
        duplicate_pairs=sum(len(t)>2 and (len(t)-1)%2==0 and all(t[i]==t[i+1] for i in range(1,len(t),2)) for t in traces)
        records.append(dict(method=method,path=str(path),sha256=sha256(path),logged_episodes=len(traces),
            final_below20_fraction=float(np.mean([t[-1]<20 for t in traces])) if traces else None,
            improved_fraction=float(np.mean([t[-1]<t[0] for t in traces])) if traces else None,
            duplicated_pair_pattern_episodes=duplicate_pairs,max_logged_values=max(map(len,traces),default=0),
            limitation="Append-only scores without scene IDs, seeds, model hash or version; cannot identify a single 398x40 run or infer true action counts."))
    dump(Path(out)/"historical_log_audit.json",records)
    return records


def configuration_archive():
    from .data import configuration
    return configuration()["archive_root"]


def build_figures(out,rows,summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    names=["MM-PPO","Point cloud only","Robot state only"]
    colors=["#087e87","#8181bc","#cc9e47"]
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout="constrained")
    for ax,key,ref,title in zip(axes,["success_fraction","improved_fraction"],["success","improved"],["Occlusion avoidance success","Any occlusion reduction"]):
        vals=[s[key]*100 for s in summary]
        ax.bar(names,vals,color=colors,width=.55)
        ax.scatter(names,[100*PAPER[s["method"]][ref] for s in summary],marker="D",color="#283c49",label="Paper reported (historical)",zorder=4)
        for i,v in enumerate(vals):
            reference=100*PAPER[summary[i]["method"]][ref]
            ax.text(i,max(v,reference)+2.1,f"{v:.2f}%",ha="center",fontsize=10)
        ax.set_ylim(0,109);ax.set_ylabel("Initially occluded trials (%)");ax.set_title(title)
        ax.legend(loc="lower left",frameon=False,fontsize=8)
    fig.suptitle(f"Reconstructed 398 scenes x 40 starts | {summary[0]['eligible']:,} initially occluded starts",fontsize=13)
    fig.savefig(out/"01_success_comparison.png",dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4.2),layout="constrained")
    for s,name,color in zip(summary,names,colors):
        values=[r for r in rows if r["method"]==s["method"] and r["eligible"]]
        final=np.sort([r["final_score"] for r in values])
        axes[0].plot(final,np.arange(1,len(final)+1)/len(final),color=color,label=name)
        improved=[r for r in values if r["improved"]]
        ratios=np.sort([r["reduction_fraction"]*100 for r in improved])
        steps=np.sort([r["decision_steps"] for r in improved])
        axes[1].plot(ratios,np.arange(1,len(ratios)+1)/len(ratios),color=color)
        axes[2].step(steps,np.arange(1,len(steps)+1)/len(steps),color=color,where="post")
    axes[0].set(xscale="symlog",xlabel="Final occlusion score",ylabel="Cumulative fraction",title="All initially occluded trials")
    axes[0].axvline(20,color="#666",ls="--",lw=1);axes[0].legend(fontsize=8,frameon=False)
    axes[1].set(xlabel="Reduction (%)",title="Improved trials only")
    axes[2].set(xlabel="Actual decisions (not seconds)",title="Improved trials only",xlim=(0,41))
    axes[2].axvline(12,color="#666",ls="--",lw=1)
    for ax in axes:
        ax.set_ylim(0,1.02);ax.grid(alpha=.18)
    fig.savefig(out/"02_outcome_distributions.png",dpi=180);plt.close(fig)


def build_replay(out,rows,case_ids=None):
    from .robot_model import render_viewer, prepare_robot_assets
    selected={}
    eligible=[r for r in rows if r["method"]=="fusion" and r["eligible"]]
    def pick(label,options):
        for r in options:
            if r["case_id"] not in selected:
                selected[r["case_id"]]=label
                break
    if case_ids:
        selected={c:"指定起点" for c in case_ids}
    else:
        pick("首个融合成功",[r for r in eligible if r["success"]])
        pick("融合成功中最长轨迹",sorted([r for r in eligible if r["success"]],key=lambda r:-r["decision_steps"]))
        for reason in ["step_limit","self_collision","repeated_unreachable"]:
            pick("融合终止："+reason,[r for r in eligible if r["termination_reason"]==reason])
        pick("最高初始遮挡",sorted(eligible,key=lambda r:-r["initial_score"]))
        by_case={}
        for r in rows:
            by_case.setdefault(r["case_id"],{})[r["method"]]=r
        pick("融合成功、仅点云失败",[r for r in eligible if r["success"] and not by_case[r["case_id"]]["pointcloud"]["success"]])
        pick("融合失败、仅状态成功",[r for r in eligible if not r["success"] and by_case[r["case_id"]]["robotstate"]["success"]])
        pick("初始已达标",[r for r in rows if r["method"]=="fusion" and r["initially_clear"]])
    dataset=json.loads((out/"ordered_dataset.json").read_text(encoding="utf-8"))[1592:]
    cases=[]
    for case_id,label in selected.items():
        i=int(case_id.split("_")[1])-1
        row=dataset[i];scene=load_scene(row)
        env=PaperEnv([scene])
        env.scene=scene
        env.points=scene.points[(scene.points[:,2]>=scene.target[2]+.2)&(scene.points[:,2]<=1.75)]
        case=dict(case_id=case_id,scene_id=scene.scene_id,session=scene.session,source=scene.source,
            source_sha256=scene.source_sha256,selection_reason=label,points=env.points.tolist(),runs={})
        record=read_scene(out/"scenes"/f"scene_{i+1:03d}.json.gz")
        for ep in record["episodes"]:
            if ep["summary"]["case_id"]!=case_id:
                continue
            frames=[]
            for f in ep["trace"]:
                env.xy=np.array(f["xy"]);env.q=np.array(f["q"])
                env.steps=f["step"];env.reason=f["reason"];env.path_length=f["path_length_m"]
                env._measure()
                if not np.isclose(env.score,f["score"],rtol=1e-12,atol=1e-9):
                    raise AssertionError("Replay score differs from saved measurement")
                frames.append(env.snapshot())
            case["runs"][ep["summary"]["method"]]=dict(trace=frames,summary=ep["summary"])
        if len(case["runs"])!=3:
            raise ValueError(f"Unknown case {case_id}")
        cases.append(case)
    payload=dict(title="398 × 40 完整测试选例",threshold=20,horizon=40,full_evaluation=True,
        labels={m:LABELS[m] for m in METHODS},cases=cases,robot_model=prepare_robot_assets())
    dump(out/"replay_selection.json",[dict(case_id=c["case_id"],reason=c["selection_reason"]) for c in cases])
    render_viewer(payload,out/"完整测试选例回放.html")
    return len(cases)


def build_report(out,rows,summary,replay_count,history):
    import html
    protocol=json.loads((out/"protocol.json").read_text(encoding="utf-8"))
    verification=json.loads((out/"verification.json").read_text(encoding="utf-8"))
    e=summary[0]["eligible"]
    clear=summary[0]["initially_clear"]
    body=[]
    def tr(cells):
        return '<tr>'+''.join('<td>'+str(c)+'</td>' for c in cells)+'</tr>'
    comparison=''.join(tr([LABELS[s['method']],f"{100*s['success_fraction']:.2f}%",f"{100*PAPER[s['method']]['success']:.1f}%",
        f"{100*(s['success_fraction']-PAPER[s['method']]['success']):+.2f}",f"{100*s['improved_fraction']:.2f}%",
        f"{100*PAPER[s['method']]['improved']:.1f}%",f"{s['mean_final_score']:.2f}",f"{s['mean_decision_steps']:.2f}"]) for s in summary)
    conditioned=''.join(tr([LABELS[s['method']],s['improved'],f"{s['mean_final_score_improved']:.2f}",
        f"{100*s['mean_reduction_fraction_improved']:.2f}%",f"{s['mean_decision_steps_improved']:.2f}",
        f"{s['mean_decision_steps_success']:.2f}"]) for s in summary)
    macro=''.join(tr([LABELS[s['method']],f"{100*s['macro_scene_success_fraction']:.2f}%",
        f"{100*PAPER[s['method']]['success']:.1f}%",f"{100*(s['macro_scene_success_fraction']-PAPER[s['method']]['success']):+.2f}",
        f"{100*s['macro_scene_improved_fraction']:.2f}%",f"{100*PAPER[s['method']]['improved']:.1f}%"]) for s in summary)
    failure=''.join(tr([LABELS[s['method']],s['successes'],s['self_collision_cases'],s['repeated_unreachable_cases'],s['step_limit_cases'],
        s['fast_success_cases'],s['additional_successes_horizon41'],f"{100*s['macro_scene_success_fraction']:.2f}%"]) for s in summary)
    history_table=''.join(tr([LABELS[h['method']],f"{h['logged_episodes']:,}",h['duplicated_pair_pattern_episodes'],h['max_logged_values']]) for h in history)
    notes=[
        "数据：按原加载函数在当前归档中的文件顺序重建 1,990 份点云；前 1,592 份为训练部分，后 398 份为测试部分。文件与目标标注均保存了校验值。历史文件顺序及存档模型是否接触过这些文件仍无法由现存记录确认。",
        "398 个文件均来自 TR0822，分属 47 个采集文件夹。data31 同时落在重建训练和测试部分。这是文件级切分，不能称为 398 次独立手术，也不能据此证明采集级独立泛化。",
        "起点：原 Python random 的矩形采样范围和 KDL 可达性筛选；种子 1–15,920。每个起点固定初始逆解猜测，三个模型共享完整初始状态；没有额外剔除初始碰撞姿态。旧 reset 会继承上局关节角和不可达计数，本次清零以得到配对、独立的回合。",
        "点云抽样：沿用原抽样和补点规则，但固定 NumPy RandomState。旧代码没有保存该随机数状态，因此不能声称逐步恢复历史轨迹。没有依据论文数值筛种子、调模型或更改失败记录。",
        "模型：20 / 21 / 22 三份存档权重，分别为融合、仅点云、仅状态；确定性推理，不重新训练，不使用后来覆盖的归一化文件。",
        "运动学与碰撞：实际执行归档 Orocos KDL 1.5.1 与 NC 部署工程 collision.cpp，没有替代求解器或简化碰撞后备方案。原 Python 碰撞模块与部署版本的历史对应关系尚未确认。原 NR+pinv 求解器本身没有启用论文文字所述的 NR_JL 关节限位。",
        "步数：主结果严格最多 40 次决策，每次调用只计一步。原 interaction>40 会执行第 41 步，本次对达到 40 步上限的回合另存第 41 步敏感性结果，不混入主结果。",
        "旧评估代码的 dataset[-j] 在 j=0 时选中训练文件，且漏掉首个测试文件；本次使用全部 398 个测试文件一次。旧汇总会排除零成功率场景，本次所有有效回合与场景均保留。旧源码重复记录同一步，历史日志则可能混合不同版本，不能直接把日志行数当决策步数。",
        "论文图14文字描述的均值来自有改善回合，因此单独列出该条件统计。当前找到的 simulation_plot.m 还要求终值低于 20，并用含初始值的记录长度计步，与正文口径不同；本报告没有将这几种分母混用。",
        "本次复现静态实测点云上的运动学决策测试，不模拟动态人员、机械动力学或真实运动秒数；完整 STL 只负责显示。成功定义为终值低于 20，同时另存排除自碰撞终止的严格成功指标。"]
    note_html=''.join('<li>'+html.escape(t)+'</li>' for t in notes)
    doc=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>398 × 40 完整测试报告</title>
<style>*{{box-sizing:border-box}}body{{margin:0;background:#f3f6f7;color:#233e48;font:15px/1.8 "Segoe UI","Microsoft YaHei",sans-serif}}main{{max-width:1240px;margin:auto;padding:35px 30px}}header{{background:#153e49;color:white;padding:36px;border-radius:16px}}header p{{color:#cee3e6}}h1{{font-size:30px;margin:4px 0}}h2{{font-size:21px;margin:5px 0 15px}}.tag{{font-size:12px;letter-spacing:2px;color:#88d8d3}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:15px;margin:22px 0}}.card,section{{background:white;padding:24px;border-radius:12px;border:1px solid #dce6e8}}.card strong{{display:block;font-size:30px}}.card span{{color:#6a7e85;font-size:13px}}section{{margin:20px 0}}.note{{padding:17px 22px;background:#fff3d9;border-left:4px solid #b2873e;border-radius:7px}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap}}th,td{{padding:12px 10px;border-bottom:1px solid #e4ebed;text-align:right}}th{{background:#f1f6f6;color:#577079}}td:first-child,th:first-child{{text-align:left}}img{{max-width:100%;height:auto}}a{{color:#087e87}}.links{{display:flex;flex-wrap:wrap;gap:15px}}.button{{padding:12px 20px;border-radius:7px;background:#087e87;color:white;text-decoration:none}}small,.muted{{color:#647b84}}li{{margin:10px 0}}footer{{color:#71858c;font-size:12px}}@media(max-width:750px){{.cards{{grid-template-columns:1fr 1fr}}main{{padding:16px}}header{{padding:24px}}h1{{font-size:24px}}}}</style><main>
<header><div class="tag">OR VISION LAB · FULL EVALUATION</div><h1>论文 398 场景 × 40 起点完整测试</h1><p>原始点云 · 三个存档模型 · 原始 KDL 与碰撞 C++ · 可核查逐步记录</p><div>完整计算已完成；历史数值一致性尚未确认</div></header>
<div class="cards"><div class="card"><strong>398 × 40</strong><span>15,920 个候选起点</span></div><div class="card"><strong>47,760</strong><span>三个模型的完整回合记录</span></div><div class="card"><strong>{e:,}</strong><span>每模型初始有遮挡起点</span></div><div class="card"><strong>{clear:,}</strong><span>每模型初始已达标起点</span></div></div>
<div class="note">本轮是依据现存论文、数据、权重和源码重建的完整规模测试。历史随机状态、文件顺序和碰撞模块对应关系存在缺口；新结果不等同于历史实验的逐项认证。论文值与本轮值均直接列出，未为追平论文数值调整参数。</div>
<section><h2>三个模型与论文结果对照</h2><p>本次成功率与改善率的分母均为每模型 <b>{e:,}</b> 个初始遮挡评分 ≥ 20 的起点；初始已达标起点不计为规避成功。主结果按回合合并统计，包含零成功率场景。论文历史比例的聚合细节存在缺口，差值仅作核查，不作显著性结论。</p><div class="scroll"><table><thead><tr><th>模型</th><th>本次成功率</th><th>论文成功率</th><th>差值 / 百分点</th><th>本次改善率</th><th>论文改善率</th><th>本次平均终值</th><th>本次平均步数</th></tr></thead><tbody>{comparison}</tbody></table></div><img src="01_success_comparison.png" alt="本次成功率和改善率与论文报告值对照"></section>
<section><h2>按场景等权汇总的对照</h2><p>原测试脚本先计算每场景比例，再对场景求平均，因此这里也列出相应口径。本次保留所有 398 个场景；每个场景均至少有一个初始遮挡起点。场景等权与上表回合合并使用同一批原始记录，差别仅在各场景权重。</p><div class="scroll"><table><thead><tr><th>模型</th><th>本次场景等权成功率</th><th>论文成功率</th><th>差值 / 百分点</th><th>本次场景等权改善率</th><th>论文改善率</th></tr></thead><tbody>{macro}</tbody></table></div></section>
<section><h2>有改善回合的条件统计</h2><p>以下只纳入终值小于初始值的回合，对应论文图14正文的条件口径。它与上表“全部初始遮挡回合”的均值不同。</p><div class="scroll"><table><thead><tr><th>模型</th><th>改善回合数</th><th>平均终值</th><th>平均下降比例</th><th>平均决策步数</th><th>仅成功回合平均步数</th></tr></thead><tbody>{conditioned}</tbody></table></div><p class="muted">论文列出的融合模型终值 / 下降比例 / 步数为 11.2 / 93.3% / 7.0；仅点云终值 / 步数为 26.1 / 7.8；仅状态步数为 9.0。此处不补造正文中未核实的数值。</p><img src="02_outcome_distributions.png" alt="遮挡终值、改善比例及决策步数累计分布"></section>
<section><h2>失败类型与步数核查</h2><div class="scroll"><table><thead><tr><th>模型</th><th>成功数</th><th>自碰撞终止</th><th>连续不可达</th><th>40步上限</th><th>少于12步成功</th><th>多1步新增成功</th><th>场景等权成功率</th></tr></thead><tbody>{failure}</tbody></table></div><p class="muted">“场景等权”对存在初始遮挡起点的场景分别计算成功率再取平均；无有效回合场景没有定义成功率，未赋成零。第41步仅为源代码边界差异的敏感性检查。碰撞终止与低于阈值可能同一步发生，严格成功数见完整汇总。</p></section>
<section><h2>机器人回放与完整数据</h2><p>整机回放选取 {replay_count} 组起点，覆盖成功、失败与模型差异，便于检查动作。选例不代表总体分布。全部 47,760 条回合及其逐步记录均已保存在本目录。</p><p><a class="button" href="完整测试选例回放.html">打开完整机器人选例回放</a></p><div class="links"><a href="episodes.csv" download>全部回合数据</a><a href="summary.csv" download>完整指标汇总</a><a href="per_scene_summary.csv" download>398场景逐场汇总</a><a href="steps.csv.gz" download>全部逐步轨迹（压缩）</a><a href="protocol.json">冻结测试方案</a><a href="verification.json">完整性核查</a><a href="starts.json">全部起点与种子</a><a href="ordered_dataset.json">数据划分清单</a></div></section>
<section><h2>与原论文、旧代码的对应关系</h2><ol>{note_html}</ol></section>
<section><h2>历史日志核查</h2><p>找到的三个日志只有遮挡值，没有场景、起点种子与模型校验值，且使用追加写入。因此不能把这些日志直接认定为同一次 398 × 40 测试。</p><div class="scroll"><table><thead><tr><th>文件对应模型</th><th>日志回合数</th><th>具有相邻重复对模式</th><th>最长记录值数量</th></tr></thead><tbody>{history_table}</tbody></table></div><p><a href="historical_log_audit.json">查看历史文件来源与校验值</a></p></section>
<section><h2>复现记录</h2><p>398 个场景文件齐全；每场景 40 个起点、每起点 3 个模型；全部起始评分与位置配对一致。共保存 {verification['trace_frames']:,} 帧（含初始帧），实际执行 {verification['actual_decision_steps']:,} 次主测试决策；最多 40 步。逐回合终值均与轨迹末帧一致。</p><p>计算环境：Python {protocol['versions']['python']}，NumPy {protocol['versions']['numpy']}，PyTorch {protocol['versions']['torch']}，CPU 确定性模型推理。继续运行方法见<a href="复现说明.txt">复现说明</a>。</p></section><footer>生成时间：{datetime.now(timezone.utc).isoformat()} · 研究仿真数据，不代表实体手术或临床效果。</footer></main></html>'''
    (out/"完整测试报告.html").write_text(doc,encoding="utf-8")
    explanation="\n".join(["398 场景 × 40 起点 × 3 模型完整测试", "结果入口：完整测试报告.html", "整机回放：完整测试选例回放.html", "",
        "在仿真平台目录运行（复用已完成场景，检查代码/模型/数据校验值）：",
        '.\\.venv\\Scripts\\python.exe run.py paper-evaluate --output outputs\\paper398_reproduction_20261001 --workers 4',
        "从头复跑：将 --output 改为一个尚不存在的新目录。",
        "汇总及重新生成报告：", '.\\.venv\\Scripts\\python.exe -m ncsim.paper_report outputs\\paper398_reproduction_20261001',
        "指定任意已完成起点生成完整机器人回放（会更新选例回放文件，不改统计）：",
        '.\\.venv\\Scripts\\python.exe -m ncsim.paper_report outputs\\paper398_reproduction_20261001 --case scene_001_start_01',
        "完整逐步轨迹同时存于 scenes/scene_001.json.gz 至 scene_398.json.gz；包括第41步单独敏感性记录。",
        "步骤与结果不换算为真实时间。报告中说明了历史条件无法完全确认的范围。", "", *notes])
    (out/"复现说明.txt").write_text(explanation,encoding="utf-8")


def publish(output,case_ids=None):
    out=Path(output).resolve()
    # Re-read authoritative checkpoints instead of lossy CSV coercion.
    rows=[e["summary"] for i in range(398) for e in read_scene(out/"scenes"/f"scene_{i+1:03d}.json.gz")["episodes"]]
    summary=json.loads((out/"summary.json").read_text(encoding="utf-8"))
    build_figures(out,rows,summary)
    history=audit_historical_logs(out)
    n=build_replay(out,rows,case_ids)
    build_report(out,rows,summary,n,history)
    dump(out/"report_manifest.json",dict(generated_utc=datetime.now(timezone.utc).isoformat(),
        report_code_sha256=sha256(Path(__file__)),viewer_template_sha256=sha256(ROOT/"viewer/template.html"),
        outputs_sha256={p.name:sha256(p) for p in out.iterdir() if p.is_file() and p.name not in {"report_manifest.json","progress.json"}}))
    dump(ROOT/"outputs/latest_paper.json",dict(directory=str(out),report=str(out/"完整测试报告.html"),viewer=str(out/"完整测试选例回放.html")))
    print(f"REPORT {out/'完整测试报告.html'}",flush=True)


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--case",action="append",dest="case_ids")
    args=parser.parse_args()
    publish(args.output,args.case_ids)


if __name__=="__main__":
    main()
