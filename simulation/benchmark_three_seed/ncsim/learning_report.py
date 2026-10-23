"""Render an independent report from frozen multimodal comparison artifacts.

No training, evaluation, checkpoint selection, or manuscript files are changed.
Run: python -m ncsim.learning_report --out <study directory>
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path
from urllib.parse import quote

import numpy as np

COLORS = {"ppo": "#3C80BF", "sac": "#FF5A50", "td3": "#F6B049", "tqc": "#4BBEB6", "ppo_legacy": "#8E59C6"}
LABELS = {"ppo": "MM-PPO-v2", "sac": "MM-SAC", "td3": "MM-TD3", "tqc": "MM-TQC", "ppo_legacy": "PPO reward\nablation"}
SHORT_LABELS = {"ppo": "PPO-v2", "sac": "SAC", "td3": "TD3", "tqc": "TQC", "ppo_legacy": "PPO\nold reward"}
MODES = ("raw", "shared_filter")
METRICS = (
    ("success_pct", "Collision-free recovery", "Success (%)", "Higher is better"),
    ("mean_final_score", "Final occlusion score", "Occlusion score", "Lower is better"),
    ("mean_path_m", "Mean path length", "Path length (m)", "Read with success and collisions"),
    ("collision_pct", "Endpoint self-collision", "Episodes with collision (%)", "Lower is better"),
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def normalize(table):
    """Validate per-seed data without fabricating missing measurements."""
    lookup = {}
    for entry in table:
        method, mode = entry["method"], entry["mode"]
        if method not in COLORS or mode not in MODES:
            raise ValueError(f"Unrecognized method or mode: {method}/{mode}")
        if (method, mode) in lookup:
            raise ValueError("Repeated comparison table identity")
        seeds = []
        seen = set()
        for record in entry.get("per_seed", []):
            seed = record["seed"]
            if seed in seen:
                raise ValueError("Repeated training seed")
            seen.add(seed)
            n = record.get("n")
            if not number(n) or n <= 0 or int(n) != n:
                raise ValueError("Each completed seed needs a positive eligible episode count")
            row = dict(record)
            for key in ("success_pct", "mean_final_score", "mean_path_m", "self_collision_episodes"):
                if not number(row.get(key)):
                    raise ValueError(f"Missing/nonfinite per-seed metric: {key}")
            if not 0 <= row["success_pct"] <= 100 or not 0 <= row["self_collision_episodes"] <= n:
                raise ValueError("Invalid success or collision count")
            if int(row["self_collision_episodes"]) != row["self_collision_episodes"]:
                raise ValueError("Collision count must be integral")
            if row["mean_path_m"] < 0:
                raise ValueError("Path length cannot be negative")
            row["collision_pct"] = 100 * row["self_collision_episodes"] / n
            seeds.append(row)
        if seeds:
            lookup[method, mode] = seeds
    return lookup


def plot_comparison(lookup, protocol, destination):
    if not lookup:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.findfont("Times New Roman", fallback_to_default=False)
    settings = {"font.family": "Times New Roman", "font.size": 10,
                "axes.labelsize": 10, "axes.titlesize": 11, "xtick.labelsize": 9,
                "ytick.labelsize": 9, "axes.linewidth": .7, "pdf.fonttype": 42,
                "ps.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white"}
    methods = [m for m in COLORS if any((m, mode) in lookup for mode in MODES)]
    with plt.rc_context(settings):
        fig, axes = plt.subplots(2, 4, figsize=(13.2, 6.15), sharey="col")
        fig.subplots_adjust(left=.085, right=.985, bottom=.19, top=.815, wspace=.36, hspace=.65)
        smoke = protocol.get("stage") == "smoke"
        heading = "SOFTWARE CHECK ONLY" if smoke else "FIXED INTERACTION-BUDGET PILOT"
        fig.text(.5, .96, heading, ha="center", va="top", size=15, weight="bold", color="#273746")
        budget = protocol.get("steps_per_method_seed")
        budget_label = f"{budget:,}" if isinstance(budget, int) else "Not recorded"
        fig.text(.5, .919, f"Multimodal policies | {budget_label} training interactions per method and seed | validation selection", ha="center", size=10)
        fig.text(.5, .885, "Dots: training seeds  |  horizontal line: mean  |  whiskers: sample SD only with >1 seed", ha="center", size=9, color="#555555")
        for row, mode in enumerate(MODES):
            for col, (key, title, ylabel, direction) in enumerate(METRICS):
                ax = axes[row, col]
                ax.spines[["top", "right"]].set_visible(False)
                ax.tick_params(direction="out", width=.7, length=3)
                ax.grid(axis="y", color="#E9E9E9", lw=.6, zorder=0)
                ax.set_axisbelow(True)
                ax.text(-.18, 1.075, chr(97 + row*4 + col), transform=ax.transAxes, weight="bold", size=13)
                if row == 0:
                    ax.set_title(title, pad=13, weight="bold")
                ax.set_ylabel(ylabel)
                ax.set_xlim(-.6, len(methods)-.4)
                ax.set_xticks(range(len(methods)), [SHORT_LABELS[m] for m in methods])
                ax.tick_params(axis="x", length=0, pad=7)
                present = False
                for x, method in enumerate(methods):
                    values = np.array([s[key] for s in lookup.get((method, mode), [])], dtype=float)
                    if not values.size:
                        ax.text(x, .035, "pending", transform=ax.get_xaxis_transform(), size=8, ha="center", color="#777777")
                        continue
                    present = True
                    offsets = np.linspace(-.13, .13, len(values)) if len(values) > 1 else np.array([0.])
                    color = COLORS[method]
                    mean = float(values.mean())
                    if len(values) > 1:
                        ax.errorbar(x, mean, yerr=values.std(ddof=1), color=color, elinewidth=1.35, capsize=4,
                                    capthick=1.1, fmt="none", zorder=3)
                    ax.scatter(x + offsets, values, s=35, c=color, edgecolors="white", linewidths=.7, zorder=4, clip_on=False)
                    ax.plot([x-.21, x+.21], [mean, mean], color="#333333", lw=1.2, zorder=5, clip_on=False)
                if present:
                    lo, hi = ax.get_ylim()
                    if key == "success_pct":
                        ax.set_ylim(min(0, lo), max(105, hi))
                    else:
                        ax.set_ylim(min(0, lo), max(hi, 1.0 if key == "collision_pct" else .01))
                else:
                    ax.text(.5, .5, "No completed evaluation", ha="center", va="center", transform=ax.transAxes, size=9, color="#666666")
                if row == 1:
                    ax.text(.5, -.32, direction, transform=ax.transAxes, ha="center", size=8, color="#666666")
        fig.text(.019, .655, "Raw policy", rotation=90, va="center", size=12, weight="bold")
        fig.text(.019, .33, "Shared sampled-path filter", rotation=90, va="center", size=12, weight="bold")
        seed_counts = sorted({len(s) for s in lookup.values()})
        seed_note = ("Single training seed: no SD or confidence interval." if seed_counts == [1]
                     else f"Completed training seeds per method/mode: {', '.join(map(str, seed_counts))}; seeds shown individually.")
        fig.text(.085, .075, seed_note, size=9)
        fig.text(.085, .046, "All metrics use initially occluded trials. Short paths can reflect early failures. No SOTA or clinical-safety claim.", size=9)
        fig.text(.085, .017, "The shared filter checks sampled native self-collision; it is not human-contact CBF or a continuous safety guarantee.", size=9)
        files = []
        for extension in ("pdf", "svg", "png"):
            path = destination / f"multimodal_comparison.{extension}"
            fig.savefig(path, dpi=600 if extension == "png" else 300, metadata={"Creator": "ncsim.learning_report"})
            files.append(path)
        plt.close(fig)
    return files


def esc(value):
    return html.escape(str(value))


def fmt(value, places=2):
    return f"{value:.{places}f}" if number(value) else "—"


def link(path, label, root):
    path = Path(path)
    if not path.is_file():
        return ""
    relative = Path("..") / path.relative_to(root)
    return f'<a href="{quote(relative.as_posix())}">{esc(label)}</a>'


def render(out):
    out = Path(out).resolve()
    protocol_path, comparison_path = out / "protocol.json", out / "comparison.json"
    protocol = read(protocol_path)
    comparison = read(comparison_path) if comparison_path.exists() else {"table": [], "paired_comparisons": [], "incomplete": []}
    if comparison.get("stage", protocol.get("stage")) != protocol.get("stage"):
        raise ValueError("Comparison stage differs from the frozen protocol")
    if comparison.get("claim", protocol.get("claim")) != protocol.get("claim"):
        raise ValueError("Comparison claim differs from the frozen protocol")
    lookup = normalize(comparison.get("table", []))
    destination = out / "report"
    destination.mkdir(exist_ok=True)
    files = plot_comparison(lookup, protocol, destination)
    smoke = protocol.get("stage") == "smoke"
    stage = "程序验证（仅检验流程）" if smoke else "固定交互预算初步比较"
    scope = ("本次为 smoke 程序验证。图中的数值只用于检查训练、保存、加载和评估链路，不能作为算法性能结论。" if smoke else
             "本次为固定交互预算试验；结果描述该预算下的表现，不等于训练收敛，不构成 SOTA 结论。")
    budget = protocol.get("steps_per_method_seed", "未记录")
    all_seeds = sorted({s["seed"] for records in lookup.values() for s in records})
    seed_note = ("当前只有一个训练种子；不绘制标准差、置信区间或显著性符号。" if len(all_seeds) == 1 else
                 "每个散点表示一个训练种子，横线表示均值；只有完成多个种子的组合才绘制样本标准差，标准差不等于置信区间。")
    if not all_seeds:
        seed_note = "暂无完成的测试组合，暂不生成数据图。缺失结果不填零。"
    downloads = " · ".join(f'<a href="{quote(p.name)}">{p.suffix[1:].upper()}</a>' for p in files)
    visual = ((destination / "multimodal_comparison.svg").read_text(encoding="utf-8")
              if files else '<p class="empty">尚无可绘制结果。请在评估完成并更新 comparison.json 后重新运行本报告。</p>')
    # Inline SVG makes the HTML itself viewable without any remote dependencies.
    if files:
        visual = visual[visual.index("<svg"):]
    table_rows = []
    for mode in MODES:
        for method in COLORS:
            for s in lookup.get((method, mode), []):
                table_rows.append(f'<tr><td><span class="dot" style="background:{COLORS[method]}"></span>{esc(LABELS[method].replace(chr(10), " "))}</td>'
                                  f'<td>{"原始策略" if mode == "raw" else "共享过滤"}</td><td>{esc(s["seed"])}</td><td>{s["n"]}</td>'
                                  f'<td>{fmt(s["success_pct"])}%</td><td>{fmt(s["mean_final_score"])}</td><td>{fmt(s["mean_path_m"], 3)}</td>'
                                  f'<td>{int(s["self_collision_episodes"])}/{int(s["n"])} ({fmt(s["collision_pct"])}%)</td></tr>')
    pairs = []
    for row in comparison.get("paired_comparisons", []):
        pairs.append(f'<tr><td>{esc(row.get("baseline"))}</td><td>{esc(row.get("mode"))}</td><td>{esc(row.get("seed"))}</td>'
                     f'<td>{esc(row.get("paired_cases"))}</td><td>{fmt(row.get("ppo_minus_baseline_success_pp"))}</td>'
                     f'<td>{esc(row.get("common_success_cases"))}</td><td>{fmt(row.get("ppo_minus_baseline_path_on_common_success_m"), 4)}</td>'
                     f'<td>{fmt(row.get("ppo_minus_baseline_final_score_all"), 3)}</td></tr>')
    source_links = [link(out / name, label, out) for name, label in (
        ("protocol.json", "冻结协议"), ("comparison.json", "比较与配对 JSON"),
        ("all_test_episodes.csv", "全部逐次测试 CSV"), ("test_cases.json", "固定测试起点"),
        ("status.json", "训练状态"), ("实验进度.html", "实时进度页面"))]
    job_links = []
    for method in protocol.get("methods", []):
        for seed in protocol.get("seeds", []):
            job = out / f"{method}_seed_{seed}"
            available = [link(job / name, label, out) for name, label in (
                ("learning_curves/progress.csv", "学习曲线 CSV"),
                ("monitor.monitor.csv", "训练回合 CSV"),
                ("best_validation.json", "验证集模型选择"),
                ("validation_history.json", "验证学习曲线 JSON"),
                ("training_manifest.json", "训练与模型记录"),
                ("test_raw/episodes.csv", "原始策略测试 CSV"),
                ("test_shared_filter/episodes.csv", "共享过滤测试 CSV"))]
            available = [v for v in available if v]
            if available:
                job_links.append(f'<li><strong>{esc(method)} / seed {esc(seed)}</strong>：' + " · ".join(available) + "</li>")
    incomplete = comparison.get("incomplete", [])
    pending = (f"还有 {len(incomplete)} 个算法／种子／模式组合未完成；图和表仅显示已完成组合。" if incomplete else
               "请结合冻结协议与训练状态核对计划种子数和完成情况。")
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>多模态算法比较 · {stage}</title>
<style>body{{margin:0;background:#f4f6f8;color:#273746;font:15px/1.65 "Times New Roman","Microsoft YaHei",serif}}main{{max-width:1240px;margin:32px auto;padding:0 24px}}h1{{font-size:29px;margin:0 0 8px}}h2{{font-size:20px;margin-top:0}}.box{{padding:24px;background:white;border:1px solid #dce2e7;margin:18px 0;border-radius:8px}}.tag{{display:inline-block;font-size:13px;font-weight:bold;border-left:4px solid #F6B049;padding:3px 12px;background:#fff5e5}}.notice{{color:#854e19}}.meta{{color:#65727b}}.chart svg{{width:100%;height:auto}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}}td,th{{padding:9px 10px;text-align:left;border-bottom:1px solid #e4e8eb;white-space:nowrap}}th{{background:#f5f7f9}}a{{color:#256aa1;text-decoration:none}}a:hover{{text-decoration:underline}}.dot{{width:9px;height:9px;border-radius:50%;display:inline-block;margin-right:7px}}.empty{{padding:36px;text-align:center;color:#75838c}}ul{{padding-left:22px}}.foot{{font-size:13px;color:#65727b}}@media print{{body{{background:white}}main{{margin:0;max-width:none;padding:0}}.box{{break-inside:avoid;border:0}}}}</style></head><body><main>
<div class="tag">{stage}</div><h1>多模态 PPO、SAC、TD3 与 TQC 比较</h1><p class="meta">每算法、每训练种子预算：{esc(budget)} 次交互。独立报告，不覆盖历史结果或论文图。</p>
<section class="box"><p class="notice"><strong>{scope}</strong></p><p>{seed_note}</p><p>{pending}</p><p>同一组固定测试起点分别评估原始策略和共享碰撞过滤。过滤使用原始 native 自碰撞检测对关节路径采样，<strong>不等同人体 CBF，也不提供连续安全保证</strong>。PPO reward ablation 是原奖励消融，不是历史 checkpoint。</p><p>{" · ".join(v for v in source_links if v)}</p></section>
<section class="box chart"><h2>两种执行模式 × 四项指标</h2><p>{downloads}</p>{visual}<p class="foot">仅统计初始遮挡试次；成功要求恢复视野且无终点或采样路径自碰撞。右侧自碰撞率专指 native 终点检测；额外的采样路径碰撞见逐次数据。最终遮挡分数不是人体 CAD 像素遮挡百分比。路径均值包含失败试次，较短路径可能来自提前终止。</p></section>
<section class="box"><h2>逐训练种子结果</h2><div class="scroll"><table><thead><tr><th>方法</th><th>模式</th><th>训练种子</th><th>初始遮挡试次</th><th>无碰撞恢复率</th><th>最终遮挡分数</th><th>平均路径 m</th><th>终点自碰撞</th></tr></thead><tbody>{''.join(table_rows) or '<tr><td colspan="8">暂无完成结果</td></tr>'}</tbody></table></div></section>
<section class="box"><h2>相同起点的配对比较</h2><p>所有差值均为 MM-PPO-v2 减去对照；恢复率为百分点，路径只比较双方都成功的起点。无共同成功试次时留空，不用 0 代替。</p><div class="scroll"><table><thead><tr><th>对照</th><th>模式</th><th>种子</th><th>配对起点</th><th>恢复率差 pp</th><th>共同成功数</th><th>配对路径差 m</th><th>最终遮挡差</th></tr></thead><tbody>{''.join(pairs) or '<tr><td colspan="8">暂无完成的配对结果</td></tr>'}</tbody></table></div></section>
<section class="box"><h2>学习曲线与数据追溯</h2><p>学习曲线使用原始训练日志；原奖励与新奖励的回报尺度不同，不直接用训练回报高低判断算法优劣。</p><ul>{''.join(job_links) or '<li>训练日志尚未生成。</li>'}</ul></section>
<p class="foot">报告源：comparison.json；protocol.json。报告只读取实验产物，不运行训练、不筛选场景、不重算或改写已有科学结果。导出图文字为 Times New Roman，PDF 内嵌字体，SVG 保留文本。</p></main></body></html>'''
    report = destination / "算法比较报告.html"
    report.write_text(page, encoding="utf-8")
    manifest = dict(stage=protocol.get("stage"), claim=protocol.get("claim"),
                    protocol_sha256=digest(protocol_path),
                    comparison_sha256=digest(comparison_path) if comparison_path.exists() else None,
                    source_sha256=digest(Path(__file__)), method_mode_combinations=len(lookup),
                    seed_counts={f"{m}/{mode}": len(records) for (m, mode), records in lookup.items()},
                    files=[dict(name=p.name, sha256=digest(p)) for p in [report, *files]],
                    note="Read-only rendering; input scientific artifacts are unchanged.")
    (destination / "report_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Existing learning_benchmark study directory")
    args = parser.parse_args()
    print(json.dumps(render(args.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
