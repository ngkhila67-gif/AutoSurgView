"""Present recovered paper results separately from independently regenerated runs."""
from pathlib import Path
from datetime import datetime,timezone
import json,shutil
import numpy as np
from .data import ROOT,sha256
from .paper_evaluation import dump,write_rows
from .policies import LABELS


def publish(out):
    out=Path(out).resolve()
    def read(name):return json.loads((out/name).read_text(encoding='utf-8'))
    metrics=read('historical_metrics.json');cross=read('matlab_crosscheck.json')
    replay=read('action_replay_backup_audit.json')['summary']
    new=read('regenerated/summary.json')
    sources=read('original_files.json')
    for s in sources:
        if sha256(out/s['snapshot'])!=s['sha256']:
            raise AssertionError('Historical MAT snapshot changed')
    assert len(sources)==1194 and all(s['all_five_rounded_values_match'] for s in metrics)
    assert all(r['maximum_python_matlab_difference']<1e-10 for r in cross)
    originals=out/'paper_original_figures';originals.mkdir(exist_ok=True)
    for filename in ['image32.png','image33.png','image34.png','image35.png']:
        shutil.copy2(ROOT/'outputs/paper_source_review'/filename,originals/filename)
    table=[]
    keys=['macro_success_percent','macro_improvement_percent','mean_final_improved','mean_reduction_percent_improved','mean_plotted_steps_improved']
    metric_names=['成功率（场景等权，%）','改善率（场景等权，%）','终止遮挡均值（改善回合）','下降比例均值（改善回合，%）','图中步数均值（含初始帧）']
    for m in metrics:
        for k,label,old in zip(keys,metric_names,m['paper_values']):
            table.append(dict(method=m['method'],metric=label,recomputed=m[k],paper_value=old,
                              rounded_recomputed=float(f"{m[k]:.1f}"),matches_paper=float(f"{m[k]:.1f}")==old))
    write_rows(out/'paper_metric_comparison.csv',table)
    audit=dict(checked_utc=datetime.now(timezone.utc).isoformat(),original_files=1194,
        per_model_seed_batches=398,seeds_covered_by_filenames=[1,15920],
        saved_occluded_episodes=sum(m['logged_occluded_episodes'] for m in metrics),
        recovered_trace_frames=sum(m['frames'] for m in replay),
        all_15_paper_displayed_metrics_match=True,
        max_python_matlab_difference=max(r['maximum_python_matlab_difference'] for r in cross),
        original_figures_rerun_in='Installed MATLAB, original source scripts with input paths and exports adapted',
        historical_geometry_source='Myclass_修改模型备份.py, 0.5405 m chain and .4+.1405 m camera',
        independently_regenerated_candidates=sum(m['candidates'] for m in new),
        final_status='Historical statistics reproduced; independent policy and trajectory regeneration not proven identical',
        originals_verified_unchanged=True)
    for s in sources:
        assert sha256(s['source'])==s['sha256']
    dump(out/'acceptance.json',audit)
    def row(values):return '<tr>'+''.join(f'<td>{v}</td>' for v in values)+'</tr>'
    main_rows=''.join(row([LABELS[m['method']],*[f'{v:.1f}' for v in m['rounded_values']],'<b class="good">5 / 5 一致</b>']) for m in metrics)
    precise=''.join(row([LABELS[m['method']],*[f'{m[k]:.8f}' for k in keys]]) for m in metrics)
    counts=''.join(row([LABELS[m['method']],398,f"{m['logged_occluded_episodes']:,}",f"{m['unlogged_candidate_seeds']:,}",m['zero_success_batches'],f"{m['mean_actual_actions_improved']:.4f}"]) for m in metrics)
    native=''.join(row([LABELS[m['method']],f"{m['frames']:,}",f"{100*m['matching_xy_frames']/m['frames']:.2f}%",f"{100*m['matching_score_frames']/m['frames']:.2f}%",f"{m['max_xy_error_m']:.6f}",f"{m['max_score_error']:.4f}"]) for m in replay)
    generated=''.join(row([LABELS[m['method']],m['candidates'],m['eligible'],*[f'{m[k]:.4f}' for k in keys]]) for m in new)
    figure_blocks=[]
    for num,original,reproduced,description in [
        (14,'image32.png','fig14_15_2','改善回合的终值、下降比例、记录行数。均值、分位数与分布来自完整历史数据。'),
        (15,'image33.png','fig14_15_1','每个MAT文件计算成功率和改善率，再对398个文件等权平均。'),
        (16,'image34.png','fig16_1','所有已记录遮挡回合的终值累计分布，直接运行原CDF脚本。'),
        (17,'image35.png','fig17_1','按初始评分20–40、40–60、60–80、80–100、≥100分组。原图最后一组标为>100，实际代码包含100。')]:
        figure_blocks.append(f'''<section><h2>图{num} · 原图与重生成图</h2><p>{description}</p><details><summary>展开论文原图</summary><img src="paper_original_figures/{original}" alt="论文原图{num}"></details><img src="matlab_reproduction/{reproduced}.png" alt="使用历史数据和原MATLAB脚本重生成图{num}"><div class="links"><a href="matlab_reproduction/{reproduced}.png">PNG</a><a href="matlab_reproduction/{reproduced}.svg">矢量SVG</a><a href="matlab_reproduction/{reproduced}.fig">可编辑MATLAB图</a></div></section>''')
    doc=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>论文历史结果复现核验</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f1f5f6;color:#233f48;font:15px/1.8 "Segoe UI","Microsoft YaHei",sans-serif}}main{{max-width:1260px;margin:auto;padding:30px}}header{{background:#153e49;color:white;padding:35px;border-radius:16px}}h1{{font-size:30px;margin:5px 0}}header p{{color:#c4e0e2}}.eyebrow{{font-size:12px;letter-spacing:2px;color:#9bd6d2}}h2{{font-size:21px;margin:0 0 14px}}section{{background:white;padding:25px;margin:20px 0;border:1px solid #dce7e9;border-radius:12px}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:15px;margin:20px 0}}.card{{background:white;padding:20px;border-radius:10px;border:1px solid #dce7e9}}.card b{{display:block;font-size:29px}}.card span{{font-size:13px;color:#67808a}}.good{{color:#087963}}.callout{{background:#e8f5f0;padding:20px;border-left:4px solid #087963;border-radius:8px}}.amber{{background:#fff4df;border-color:#b18b3d}}.scroll{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;white-space:nowrap;font-size:13px}}td,th{{padding:11px 10px;text-align:right;border-bottom:1px solid #e4ecee}}th{{background:#f2f7f7}}td:first-child,th:first-child{{text-align:left}}img{{display:block;max-width:100%;height:auto;margin:18px auto}}.links{{display:flex;flex-wrap:wrap;gap:18px}}a{{color:#087d88}}li{{margin:9px 0}}details{{border:1px solid #dce7e9;border-radius:7px;padding:12px;margin:12px 0}}summary{{cursor:pointer}}.muted,footer{{color:#6c8188;font-size:13px}}code{{overflow-wrap:anywhere;white-space:normal}}@media(max-width:750px){{main{{padding:16px}}.cards{{grid-template-columns:1fr 1fr}}header{{padding:23px}}h1{{font-size:25px}}}}</style><main>
<header><div class="eyebrow">HISTORICAL EXPERIMENT · RECOVERED & VERIFIED</div><h1>论文历史结果已从原始记录算回</h1><p>三套完整 MAT 批次 · 原 MATLAB 绘图脚本 · 15 项图示数值一致</p><div>统计与图表复现已完成；从模型重新生成每条历史轨迹仍有差异。</div></header>
<div class="cards"><div class="card"><b>1,194</b><span>原始MAT文件：398 × 3</span></div><div class="card"><b>{audit['saved_occluded_episodes']:,}</b><span>实际保存的遮挡回合</span></div><div class="card"><b>15 / 15</b><span>图14、15显示数值一致</span></div><div class="card"><b>4 张</b><span>图14—17由原脚本重生成</span></div></div>
<div class="callout"><b>本次新增证据修正了上轮结论：</b>原工程根目录下的 Matlab_Data、Matlab_Data_only_PC、Matlab_Data_only_state 保存了完整的三套 398 批历史测试。上轮只核查了文本日志，没有发现这些带种子、位置、动作和奖励的 MAT 文件。现已逐文件校验并保存快照，没有改写原数据或为贴合论文调参。</div>
<section><h2>论文五项指标：三个模型全部对上</h2><p>下表是从原 MAT 数据重新计算、按论文精度保留一位小数的结果。15 个数值与论文图14、图15一致；MATLAB 原脚本与独立 Python 重算的最大差值为 {audit['max_python_matlab_difference']:.2e}。</p><div class="scroll"><table><thead><tr><th>模型</th><th>成功率 %</th><th>改善率 %</th><th>平均终值</th><th>下降比例 %</th><th>图中步数</th><th>核对</th></tr></thead><tbody>{main_rows}</tbody></table></div><details><summary>查看未四舍五入的重算值</summary><div class="scroll"><table><thead><tr><th>模型</th><th>成功率 %</th><th>改善率 %</th><th>平均终值</th><th>下降比例 %</th><th>图中步数</th></tr></thead><tbody>{precise}</tbody></table></div></details><p><a href="paper_metric_comparison.csv">下载逐项核对表</a> · <a href="matlab_crosscheck.json">MATLAB / Python 独立交叉验证</a></p></section>
{''.join(figure_blocks)}
<section><h2>完整记录覆盖了什么</h2><p>每个模型都有398个批次文件，文件名中的种子区间连续覆盖1–15,920，每批40个候选起点。原脚本只保存初始遮挡值≥20的回合，三模型合计实际保存34,490个回合、286,373帧。其余候选起点没有逐步记录，不能把它们补写成成功轨迹。</p><div class="scroll"><table><thead><tr><th>模型</th><th>批次</th><th>已保存遮挡回合</th><th>未保存候选起点</th><th>零成功批次</th><th>改善回合平均实际动作</th></tr></thead><tbody>{counts}</tbody></table></div></section>
<section><h2>投稿前应修正的实验口径</h2><ol><li><b>“7.0步”实际含初始状态：</b>xaiorong.m 使用 <code>size(curr_ep,1)</code>，因此比实际动作次数多1。融合、仅点云、仅状态的实际动作均值分别为5.9508、6.8151、7.9796；若写“决策次数”，应明确改用这一口径。MAT文件中的Step_ID为0、2、4…，也不能直接当动作次数。</li><li><b>历史最大动作数为41：</b>原终止条件为 interaction&gt;40，对应绘图最多42行。正文“最多40步”与实际代码不一致。</li><li><b>成功率为场景等权均值：</b>不是把所有回合简单合并。图14的三个均值仅针对有改善回合，也不是全部测试回合的均值。</li><li><b>15,920是候选起点数：</b>初始已达标起点被跳过，且三模型保留下来的有效回合数并不完全相同，不能写成15,920条全部有记录、严格配对的有效规避轨迹。</li><li><b>测试索引存在边界错误：</b>按现存数据顺序和训练划分，原循环 dataset[-j] 在j=0时取第一份训练部分文件，漏掉测试部分的首份文件。历史首批动作与该场景相符；历史模型的真实训练暴露仍需原训练记录确认。因此不能在修正前照搬“398份全部严格未见”的表述。</li></ol><p class="muted">本报告忠实复算旧结果；上轮保留完整测试集、统一起点并严格40步的结果仍单独保留，两套实验口径不混用。</p></section>
<section><h2>已找到影响轨迹的机器人版本差异</h2><p>当前 Myclass.py 使用0.5145 m末端链长，而同目录 Myclass_修改模型备份.py 使用0.5405 m，云台平移为0.4+0.1405 m。按这份原始备份重新编译同一个KDL求解器，并直接执行备份中的相机变换函数，首条历史轨迹的位置和评分完全对齐到浮点精度。没有拟合几何参数或采用替代求解器。</p><p>将全部历史记录动作输入这一版本后，得到下表。它核验的是“同一动作的运动学与评分重放”，不会把历史动作当作模型重新预测的结果。</p><div class="scroll"><table><thead><tr><th>模型</th><th>核对帧数</th><th>位置误差&lt;10⁻¹⁰m</th><th>评分误差&lt;10⁻⁶</th><th>最大位置差 / m</th><th>最大评分差</th></tr></thead><tbody>{native}</tbody></table></div><p><a href="historical_backend_manifest.json">旧尺寸源码与编译记录</a> · <a href="action_replay_backup_audit.json">完整动作核对记录</a></p></section>
<section><h2>独立重新推理：尚未达到逐轨迹一致</h2><p>另用三个存档模型实际重新运行398×40，共47,760条候选回合。使用已找回的0.5405 m版本、原场景顺序、保留上一回合关节初值、原41步终止条件。动作由模型重新输出，未从历史记录读取。旧点云抽样随机状态未保存，本次在运行前固定为每模型一条从0开始的随机序列，没有搜种子。</p><div class="scroll"><table><thead><tr><th>模型</th><th>候选回合</th><th>初始遮挡</th><th>新成功率 %</th><th>新改善率 %</th><th>新平均终值</th><th>新下降比例 %</th><th>新图中步数</th></tr></thead><tbody>{generated}</tbody></table></div><p class="muted">本表是新推理结果，不是上表历史MAT的重算值。随机点云样本、历史PyKDL/碰撞二进制身份与边界处数值行为尚未全部恢复，因此不能宣称模型已逐步重现所有历史动作。</p><p><a href="regenerated/protocol.json">独立重跑方案</a> · <a href="regenerated/summary.json">独立重跑完整核对</a></p></section>
<section><h2>文件与复现入口</h2><div class="links"><a href="historical_episodes.csv">历史逐回合数据</a><a href="historical_batches.csv">每批统计</a><a href="historical_steps.csv.gz">历史全部逐步轨迹</a><a href="historical_metrics.json">高精度指标</a><a href="original_files.json">1,194份源文件清单与校验值</a><a href="acceptance.json">验收记录</a><a href="复现说明.txt">复现说明</a></div><p>original_mat 保留原MAT字节快照；original_scripts 保留原绘图程序；matlab_reproduction 保存改路径后的可运行脚本、PNG、SVG、FIG和计算结果。原数据文件已再次校验，全部未改动。</p></section><footer>生成于 {datetime.now(timezone.utc).isoformat()} · 历史统计结果可复现，不等同于模型逐步再生成已经全部一致。</footer></main></html>'''
    (out/'历史完整复现报告.html').write_text(doc,encoding='utf-8')
    note='''论文历史复现包

入口：历史完整复现报告.html
原始数据：original_mat（3套，每套398个MAT文件）
原始脚本：original_scripts
本地MATLAB复现脚本：matlab_reproduction/run_all.m
图14：matlab_reproduction/fig14_15_2.png / .svg / .fig
图15：matlab_reproduction/fig14_15_1.png / .svg / .fig
图16：matlab_reproduction/fig16_1.png / .svg / .fig
图17：matlab_reproduction/fig17_1.png / .svg / .fig

在本机MATLAB中运行 run_all.m 可重新生成上述图和指标。
脚本沿用原统计与绘图公式，仅改输入路径、输出文件和散点随机种子。
散点展示的抽样与抖动固定为20261002；它不影响均值、密度或成功率。
恢复的是历史数据的数值和图形含义，不声称截图逐像素一致。

在仿真平台目录重新生成报告：
.\\.venv\\Scripts\\python.exe -m ncsim.historical_report outputs\\historical_reproduction_20261002

historical_episodes.csv / historical_steps.csv.gz 为历史MAT导出，不是重新预测的动作。
regenerated 为三个存档模型独立重跑结果，包含逐回合和逐步轨迹。
action_replay* 为原记录动作在现有原源码内核上的一致性核查。
degree_arithmetic_probe 为保留原角度运算顺序的追加4批检查，仍发现残差，未用于修改历史指标。

结果：论文图14、15全部15项显示数值一致，图14—17由原MATLAB脚本重生成。
范围：从模型重新生成每一条历史轨迹尚不能认定完全一致，报告已说明缺失条件与实测残差。
上轮完整新测试保留在 ../paper398_reproduction_20261001，没有覆盖或伪装成历史结果。
'''
    (out/'复现说明.txt').write_text(note,encoding='utf-8')
    dump(ROOT/'outputs/latest_historical.json',dict(directory=str(out),report=str(out/'历史完整复现报告.html')))
    dump(out/'report_manifest.json',dict(code_sha256={str(p):sha256(p) for p in (ROOT/'ncsim').glob('historical*.py')},
        files_sha256={p.relative_to(out).as_posix():sha256(p) for p in out.rglob('*') if p.is_file() and p.name!='report_manifest.json'}))
    print(out/'历史完整复现报告.html')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('output');a=p.parse_args();publish(a.output)
