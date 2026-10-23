"""Source-grounded manuscript/platform crosswalk; original Word remains untouched."""
from pathlib import Path
from datetime import datetime,timezone
import html,json
from .data import ROOT,sha256
from .paper_evaluation import dump,write_rows
from .policies import LABELS


def publish(out):
    out=Path(out);p=json.loads((out/'protocol.json').read_text(encoding='utf-8'));s=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    history=json.loads((ROOT/'outputs/historical_reproduction_20261002/historical_metrics.json').read_text(encoding='utf-8'))
    mapping=[
        ('§3.3 / §4.1','1,990份点云，80/20划分，398个测试场景','使用现存原加载顺序后398份文件；修正dataset[-0]，每份测试文件恰好出现一次','已对齐文件级测试清单；历史模型实际训练暴露、采集级独立性仍未确认'),
        ('§4.1 Experimental Setup','每场景40次随机初始化','完整398×40×3；三个模型共享相同起点、初始关节解和采样种子','已执行并验证全部47,760个候选回合'),
        ('§4.1 Experimental Setup','最大40次交互','严格0为初始帧，动作1—40；不执行第41次动作','已对齐'),
        ('§4.1 / Fig14(c)','平均决策步数、Fast Zone <12','显示实际动作数；状态帧数单独列出；快速成功为成功且实际动作数<12','新测试已对齐；历史图14保留含初始帧的原统计并注明'),
        ('§4.1 / Fig12','完整机器人、红/白点云、绿色视锥、蓝色目标、相机轨迹','7个原STL及原装配变换；原观察方向；全398场景×40起点均可播放','显示语义已对齐；原图确切场景和起点未保存，不能保证截图逐像素一致'),
        ('原Myclass备份','历史运动链和相机变换','0.5405 m原KDL链，直接执行备份相机函数；使用备份直接指定的mytest/*.STL','源码版本已统一；原显示云台锚点0.4285与评分锚点0.4不同，保留原偏置'),
        ('§4.1 / Fig15','成功阈值20；按场景等权成功率和改善率','所有初始遮挡回合纳入，场景等权；同时导出按回合合并结果','已对齐统计定义；零成功场景不丢弃'),
        ('§4.1指标定义 / Fig14图注','正文全回合均值，图14有改善回合分布','分别导出全初始遮挡回合及有改善回合的终值、下降比例、动作均值','两套分母明确列出，不互相代用'),
        ('Fig14—17','正文历史图表和数值','保留原MAT＋原MATLAB重算图；图14、15的15项显示值全部一致','历史图表已复现；本次新测试另表呈现'),
        ('§4.1','严格未见、关节限位','不新增来源不明的限位，也不声称已证实存档模型的训练暴露','待补训练清单与真实关节限位；当前原NR+pinv仍只验证可达性'),
        ('§4.1','运动学仿真','直接调用原KDL和NC碰撞C++；每回合固定实测点云，回放秒数仅为显示','已对齐运动学范围；历史Python碰撞二进制身份仍待确认'),
    ]
    rows=[dict(reference=a,manuscript=b,implementation=c,status=d) for a,b,c,d in mapping]
    write_rows(out/'正文逐项对照.csv',rows)
    def row(vals):return '<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in vals)+'</tr>'
    cross=''.join(row(v) for v in mapping)
    comparison=''.join(row([LABELS[n['method']],f"{h['macro_success_percent']:.2f}",f"{n['macro_success_percent']:.2f}",f"{n['macro_improved_percent']:.2f}",f"{n['mean_actions_improved']:.2f}",f"{n['strict_success_percent']:.2f}"]) for n,h in zip(s,history))
    text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>正文与仿真逐项对齐说明</title><style>body{{margin:0;background:#f1f5f6;color:#263f49;font:15px/1.8 "Segoe UI","Microsoft YaHei",sans-serif}}main{{max-width:1220px;margin:auto;padding:30px}}header{{padding:28px;background:#183f4c;color:white;border-radius:12px}}h1{{font-size:27px;margin:0}}h2{{font-size:20px}}section{{background:white;border:1px solid #dce6e9;border-radius:10px;padding:23px;margin:20px 0}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border-bottom:1px solid #e0e8ea;padding:12px;text-align:left;vertical-align:top}}th{{background:#f1f6f7}}a{{color:#087b8b}}.scroll{{overflow:auto}}.note{{background:#fff4df;padding:15px}}li{{margin:8px 0}}@media(max-width:700px){{main{{padding:14px}}td,th{{min-width:170px}}}}</style><main><header><h1>正文与仿真逐项对齐</h1><p>依据 JFR0511.docx 第3.3、4.1节、图12及图14—17；原Word保持不变。</p></header><p><a href="正文对齐仿真.html">打开全部398×40回放</a> · <a href="正文逐项对照.csv">导出对照表</a></p><section><h2>已经完成的修正</h2><ul><li>补齐全部398场景、每场景40起点的播放入口，包含成功、未成功及初始已达标的记录。</li><li>运动链与显示整机统一到找回的0.5405 m原版本；原7个STL全部保留，共98,438个三角面，未简化网格。旧1,750,394面模型属于后续robot_model0412版本，旧回放另存保留。</li><li>测试集索引、配对起点、40次动作上限及实际动作计数已修正；重新跑完47,760个候选回合、298,116帧。</li><li>历史图14—17与正文对应；新测试数据使用单独标记，初始帧及均值分母明确区分。</li></ul></section><section><h2>完整新测试与历史结果</h2><p>成功率和改善率按场景等权，实际动作均值仅针对有改善回合；严格成功率是按有效回合合并且排除自碰撞终止的补充指标，分母与场景等权率不同。</p><div class="scroll"><table><tr><th>模型</th><th>历史成功率%</th><th>新成功率%</th><th>新改善率%</th><th>新实际动作均值</th><th>严格成功率%</th></tr>{comparison}</table></div><p>每模型15,920个候选起点，其中11,647个初始遮挡、4,273个初始已达标；所有模型共享相同起点。新测试反映正文规则修正后的实际结果，没有要求其四舍五入后强行等于旧图数值。</p></section><section><h2>正文逐项核对</h2><div class="scroll"><table><tr><th>位置</th><th>正文要求</th><th>平台实现</th><th>核对结果</th></tr>{cross}</table></div></section><section><h2>尚不能确认的内容</h2><p class="note">正文同时要求最多40次动作、平均真实决策次数，又保留旧代码41次动作和初始帧计数形成的图表。平台分别保留两套有明确来源的结果，无法在不改正文或原数据的情况下使这两套口径完全相同。</p><p>另有历史训练暴露、真实关节限位、历史Python碰撞模块及图12确切截图帧仍缺少可确认资料。没有添加推测的物理参数；报告不会将这些内容标为已验证。</p></section><section><h2>复核与文件</h2><p><a href="protocol.json">冻结运行方案</a> · <a href="verification.json">完整规模验收</a> · <a href="robot_model_manifest.json">整机原始文件与装配来源</a> · <a href="summary.csv">新统计结果</a> · <a href="/historical_reproduction_20261002/历史完整复现报告.html">历史完整复现报告</a></p><p>Word校验值：{sha256(ROOT.parent/'JFR0511.docx')}。本次平台修改前后与上一轮保存值一致。</p></section><p>{datetime.now(timezone.utc).isoformat()}</p></main></html>'''
    (out/'正文对齐说明.html').write_text(text,encoding='utf-8')
    (out/'使用说明.txt').write_text('正文对齐仿真\n\n双击仿真平台目录的“打开正文对齐仿真.cmd”。新版默认“打开仿真.cmd”也进入此平台。\n选择场景、起点、模型后点击开始回放。全部398×40组起点可用。\n状态0为初始帧，最多40次实际动作。初始已达标的起点无动作。\n正文图14—17仅使用新测试数据，高清图组见仿真平台目录的“打开正文图组.cmd”。原Word和历史数据不变。\n\n复算：\n.\\.venv\\Scripts\\python.exe -m ncsim.manuscript_evaluation 新的输出目录 --workers 4\n重新生成当前页面：\n.\\.venv\\Scripts\\python.exe -c "from ncsim.manuscript_viewer import publish; publish(\'outputs/manuscript_aligned_20261002\')"\n启动本地服务：\n.\\.venv\\Scripts\\python.exe -m ncsim.manuscript_server --open\n\n全部运行方案、源码校验、逐步记录和原模型位置保存在本目录。\n',encoding='utf-8')
    dump(out/'presentation_manifest.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        code_sha256={str(x):sha256(x) for x in [ROOT/'ncsim/manuscript_report.py',ROOT/'ncsim/manuscript_viewer.py',ROOT/'ncsim/manuscript_server.py',ROOT/'viewer/manuscript.html',ROOT/'viewer/robot-viewer.js',ROOT/'viewer/robot-viewer.bundle.js']},
        output_sha256={x.name:sha256(x) for x in out.iterdir() if x.is_file() and x.name not in ['presentation_manifest.json','playback_server.log']}))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('output');a=p.parse_args();publish(a.output)
