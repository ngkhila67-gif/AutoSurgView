"""Manuscript figures from the corrected full evaluation only. No historical results."""
from pathlib import Path
from datetime import datetime,timezone
import csv,hashlib,json,zipfile
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from .data import ROOT,sha256

SOURCE=ROOT/'outputs/manuscript_aligned_20261002'
OUT=ROOT/'outputs/manuscript_figures_20261002'
METHODS=['pointcloud','robotstate','fusion']
NAMES={'pointcloud':'Point cloud','robotstate':'Robot state','fusion':'MM-PPO'}
COLORS={'pointcloud':'#4B78A8','robotstate':'#CB9845','fusion':'#118579'}
INK='#243946';MUTED='#617481';GRID='#E3E9EC';GREEN='#EAF4EF';SEED=20261002;B=2000


def style():
    plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.labelsize':8,
        'axes.titlesize':9,'axes.titleweight':'semibold','axes.edgecolor':'#8B9BA3',
        'axes.linewidth':.65,'axes.labelcolor':INK,'text.color':INK,'xtick.color':MUTED,
        'ytick.color':MUTED,'xtick.labelsize':7,'ytick.labelsize':7,
        'xtick.major.size':3,'ytick.major.size':3,'xtick.major.width':.6,'ytick.major.width':.6,
        'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,
        'legend.fontsize':7.5,'figure.facecolor':'white','axes.facecolor':'white',
        'savefig.facecolor':'white','pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
        'path.simplify':False,'mathtext.fontset':'dejavusans'})


def panel(ax,letter,title):
    ax.text(-.14,1.105,letter,transform=ax.transAxes,fontweight='bold',fontsize=11,va='bottom')
    ax.set_title(title,loc='left',pad=17)
    ax.set_axisbelow(True)


def legend(fig,methods=METHODS,y=.955):
    handles=[Line2D([],[],color=COLORS[m],marker=['o','s','D'][i],lw=1.5,ms=3.6,label=NAMES[m]) for i,m in enumerate(methods)]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,y),ncol=len(methods),handlelength=1.8,columnspacing=2.5)


def ci(samples):return np.nanpercentile(samples,[2.5,97.5],axis=0)


def export_csv(name,rows):
    pd.DataFrame(rows).to_csv(OUT/'source_data'/name,index=False,encoding='utf-8-sig',float_format='%.12g')


def save(fig,name,pages):
    fig.savefig(OUT/f'{name}.png',dpi=600)
    fig.savefig(OUT/f'{name}.svg')
    fig.savefig(OUT/f'{name}.pdf',metadata={'Title':name,'Subject':'Corrected 398 x 40 paired evaluation; new results only'})
    fig.savefig(OUT/f'{name}.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    pages.savefig(fig)


def load():
    d=pd.read_csv(SOURCE/'episodes.csv')
    assert len(d)==47760 and d.decision_steps.max()==40
    assert set(d.method)==set(METHODS)
    grouped=d.groupby('case_id')
    assert len(grouped)==15920 and grouped.size().eq(3).all()
    assert grouped.initial_score.nunique().eq(1).all()
    assert grouped.initial_x_m.nunique().eq(1).all() and grouped.initial_y_m.nunique().eq(1).all()
    d['capture_group']=d.scene_id.str.rsplit('/',n=1).str[0]
    groups=sorted(d.capture_group.unique());assert len(groups)==47
    d['group_index']=d.capture_group.map({g:i for i,g in enumerate(groups)})
    e=d[d.eligible].copy();assert len(e)==11647*3
    assert e.groupby('method').size().eq(11647).all()
    W=np.random.default_rng(SEED).multinomial(len(groups),np.full(len(groups),1/len(groups)),size=B)
    return d,e,groups,W


def raincloud(ax,values,position,color,transform,mean,rng,label_scale=1):
    v=transform(values);grid=np.linspace(v.min(),v.max(),256)
    dens=gaussian_kde(v,bw_method=.20)(grid);dens=dens/dens.max()*.34
    ax.fill_betweenx(grid,position-dens-.015,position-.015,color=color,alpha=.26,lw=0)
    ax.plot(position-dens-.015,grid,color=color,lw=.7,alpha=.85)
    sample=np.unique(np.r_[rng.choice(len(v),min(450,len(v)),replace=False),np.argmin(v),np.argmax(v)])
    ax.scatter(position+.06+rng.uniform(0,.25,len(sample)),v[sample],s=2.2,c=color,alpha=.23,lw=0,rasterized=False)
    q1,med,q3=np.quantile(v,[.25,.5,.75]);ax.plot([position,position],[q1,q3],color=INK,lw=2.5,zorder=4)
    ax.plot([position-.055,position+.055],[med,med],color='white',lw=1.2,zorder=5)
    ax.scatter([position],[transform(mean)],s=29,facecolor='white',edgecolor=color,lw=1.25,zorder=6)
    ax.annotate(f'{mean*label_scale:.2f}',(position,transform(mean)),xytext=(7,-3),textcoords='offset points',
        fontsize=7,fontweight='bold',color=color,zorder=7,bbox=dict(facecolor='white',alpha=.88,edgecolor='none',pad=.6))


def fig14(e):
    fig,axs=plt.subplots(1,3,figsize=(180/25.4,94/25.4));fig.subplots_adjust(left=.075,right=.99,bottom=.22,top=.80,wspace=.40)
    specs=[('final_score','Terminal occlusion',np.log1p),('reduction_fraction','Occlusion reduction',lambda x:np.asarray(x)*100),('decision_steps','Decision count',lambda x:np.asarray(x))]
    rows=[];rng=np.random.default_rng(SEED)
    for ax,(key,title,tf),letter in zip(axs,specs,'abc'):
        panel(ax,letter,title);ax.grid(axis='y',color=GRID,lw=.6)
        if key=='final_score':
            ax.axhspan(0,np.log1p(20),color=GREEN,zorder=0);ax.axhline(np.log1p(20),color='#7BA391',ls=(0,(3,3)),lw=.8)
            ticks=np.array([0,5,20,100,500]);ax.set_yticks(np.log1p(ticks),[str(v) for v in ticks]);ax.set_ylabel('Final occlusion score');ax.set_ylim(-.12,np.log1p(700));ax.text(.98,np.log1p(20)+.10,'Threshold 20',transform=ax.get_yaxis_transform(),ha='right',fontsize=6.3,color='#597E6A')
        elif key=='reduction_fraction':ax.set_ylim(-3,106);ax.set_yticks([0,25,50,75,100]);ax.set_ylabel('Reduction (%)')
        else:
            ax.axhspan(0,12,color=GREEN,zorder=0);ax.axhline(12,color='#7BA391',ls=(0,(3,3)),lw=.8);ax.set_ylim(-1,43);ax.set_yticks([0,10,20,30,40]);ax.set_ylabel('Actual decisions');ax.text(.98,12.6,'Fast zone: <12',transform=ax.get_yaxis_transform(),ha='right',fontsize=6.3,color='#597E6A')
        labs=[]
        for x,m in enumerate(METHODS):
            a=e[(e.method==m)&e.improved];v=a[key].to_numpy();mean=float(v.mean());raincloud(ax,v,x,COLORS[m],tf,mean,rng,label_scale=100 if key=='reduction_fraction' else 1)
            labs.append(NAMES[m].replace(' ','\n')+'\n'+f'n={len(v):,}')
            rows.append(dict(panel=letter,method=m,metric=key,n=len(v),mean=mean,median=np.median(v),q25=np.quantile(v,.25),q75=np.quantile(v,.75),minimum=v.min(),maximum=v.max()))
        ax.set_xlim(-.49,2.53);ax.set_xticks(range(3),labs);ax.tick_params(axis='x',length=0,pad=7)
    fig.text(.5,.065,'Improved episodes only  |  White circles: means  |  Dark bars: interquartile ranges',ha='center',fontsize=6.7,color=MUTED)
    fig.text(.5,.024,'All observations define the distributions; displayed dots are a fixed random subset. Panel a uses a log(1 + score) axis.',ha='center',fontsize=6.5,color=MUTED)
    export_csv('Fig14_distribution_summaries.csv',rows)
    e[e.improved][['case_id','scene_index','capture_group','method','final_score','reduction_fraction','decision_steps']].to_csv(OUT/'source_data/Fig14_episode_data.csv.gz',index=False,compression='gzip')
    return fig


def macro_estimates(e,W):
    agg=e.groupby(['method','scene_index','group_index'])[['success','improved']].mean().reset_index()
    estimates={};boot={}
    for m in METHODS:
        a=agg[agg.method==m];num=a.groupby('group_index')[['success','improved']].sum().reindex(range(47),fill_value=0).to_numpy()
        n=a.groupby('group_index').size().reindex(range(47),fill_value=0).to_numpy()
        estimates[m]=a[['success','improved']].mean().to_numpy()*100
        boot[m]=(W@num)/(W@n)[:,None]*100
    export_csv('Fig15_scene_data.csv',agg.to_dict('records'))
    return estimates,boot


def fig15(e,W):
    estimates,boot=macro_estimates(e,W)
    fig,axs=plt.subplots(1,3,figsize=(180/25.4,80/25.4),gridspec_kw={'width_ratios':[1.1,1.1,1]});fig.subplots_adjust(left=.14,right=.98,bottom=.24,top=.77,wspace=.58)
    rows=[]
    for j,(ax,title,xlim) in enumerate(zip(axs[:2],['Success rate','Improvement rate'],[(76,101.5),(90,100.8)])):
        panel(ax,chr(97+j),title);ax.grid(axis='x',color=GRID,lw=.6)
        for y,m in enumerate(METHODS):
            val=estimates[m][j];lo,hi=ci(boot[m][:,j]);ax.errorbar(val,y,xerr=[[val-lo],[hi-val]],fmt='o',color=COLORS[m],ms=5.1,lw=1.5,capsize=3)
            ax.annotate(f'{val:.2f}%',(val,y),xytext=(-2,10),textcoords='offset points',ha='center',fontsize=8,fontweight='bold',color=COLORS[m])
            rows.append(dict(method=m,metric=['macro_success_percent','macro_improved_percent'][j],estimate=val,ci95_low=lo,ci95_high=hi,scenes=398))
        ax.set_xlim(*xlim);ax.set_ylim(2.6,-.75);ax.set_yticks(range(3),[NAMES[m] for m in METHODS] if j==0 else []);ax.tick_params(axis='y',length=0);ax.spines['left'].set_visible(False);ax.set_xlabel('Scene-equal rate (%)');ax.set_xticks([80,90,100] if j==0 else [90,95,100])
    ax=axs[2];panel(ax,'c','Fusion advantage');ax.grid(axis='x',color=GRID,lw=.6);ax.axvline(0,color='#9CAAB0',lw=.7,ls=(0,(3,3)))
    for y,m in enumerate(METHODS[:2]):
        val=estimates['fusion'][0]-estimates[m][0];lo,hi=ci(boot['fusion'][:,0]-boot[m][:,0]);ax.plot([0,val],[y,y],color=COLORS[m],alpha=.25,lw=3)
        ax.errorbar(val,y,xerr=[[val-lo],[hi-val]],fmt='D',color=COLORS[m],ms=5,lw=1.5,capsize=3)
        ax.annotate(f'+{val:.2f} pp',(val,y),xytext=(0,10),textcoords='offset points',ha='center',fontsize=8,fontweight='bold',color=COLORS[m])
        rows.append(dict(method='fusion_minus_'+m,metric='paired_macro_success_gain_pp',estimate=val,ci95_low=lo,ci95_high=hi,scenes=398))
    ax.set_yticks([0,1],['vs point\ncloud','vs robot\nstate']);ax.tick_params(axis='y',length=0);ax.spines['left'].set_visible(False);ax.set_ylim(1.7,-.75);ax.set_xlim(-.6,17);ax.set_xticks([0,5,10,15]);ax.set_xlabel('Success-rate gain (pp)')
    fig.text(.5,.082,'398 scenes  |  40 paired candidate starts per scene  |  11,647 initially occluded starts per model',ha='center',fontsize=6.7,color=MUTED)
    fig.text(.5,.035,'Intervals: 95% paired acquisition-folder bootstrap (47 folders; 2,000 resamples).',ha='center',fontsize=6.7,color=MUTED)
    export_csv('Fig15_estimates_and_intervals.csv',rows)
    return fig


def count_bootstrap(e,W,m,grid,key):
    a=e[e.method==m];den=a.groupby('group_index').size().reindex(range(47),fill_value=0).to_numpy()
    counts=[]
    for g in range(47):
        v=a[a.group_index==g]
        if key=='decision_steps':v=v[v.success]
        values=np.sort(v[key].to_numpy());counts.append(np.searchsorted(values,grid,side='right'))
    counts=np.array(counts);estimate=counts.sum(axis=0)/den.sum()*100
    samples=(W@counts)/(W@den)[:,None]*100
    return estimate,ci(samples)


def fig16(e,W):
    fig,axs=plt.subplots(1,2,figsize=(180/25.4,96/25.4));fig.subplots_adjust(left=.075,right=.965,bottom=.23,top=.75,wspace=.30);legend(fig,y=.98)
    ax=axs[0];panel(ax,'a','Terminal occlusion distribution');ax.grid(color=GRID,lw=.6);ax.axvspan(0,np.log1p(20),color=GREEN,zorder=0)
    ax.axvline(np.log1p(20),color='#799886',lw=.8,ls=(0,(3,3)));ax.axhline(95,color='#ADB7BC',lw=.7,ls=(0,(3,3)))
    ax.text(np.log1p(20)+.13,8,'Threshold 20',rotation=90,color='#63826E',fontsize=6.5)
    grid=np.unique(np.r_[np.linspace(0,20,100),np.geomspace(20.0001,e.final_score.max(),180)])
    source=[]
    for m in METHODS:
        a=e[e.method==m];x=np.sort(a.final_score.to_numpy());y=np.arange(1,len(x)+1)/len(x)*100
        est,(lo,hi)=count_bootstrap(e,W,m,grid,'final_score');ax.fill_between(np.log1p(grid),lo,hi,color=COLORS[m],alpha=.10,lw=0,step='post');ax.step(np.r_[0,np.log1p(x)],np.r_[0,y],where='post',color=COLORS[m],lw=1.55)
        source += [dict(panel='a',method=m,x=t,estimate=z,ci95_low=l,ci95_high=h) for t,z,l,h in zip(grid,est,lo,hi)]
    ticks=[0,5,20,100,500,1500];ax.set_xticks(np.log1p(ticks),['0','5','20','100','500','1,500']);ax.set_xlim(-.06,np.log1p(e.final_score.max())*1.015);ax.set_ylim(0,102);ax.set_ylabel('Cumulative episodes (%)');ax.set_xlabel('Final occlusion score, log(1 + score) axis')
    ax=axs[1];panel(ax,'b','Success within a decision budget');ax.grid(color=GRID,lw=.6);ax.axvspan(0,12,color=GREEN,zorder=0);ax.axvline(12,color='#799886',lw=.8,ls=(0,(3,3)))
    ax.text(6,8,'Fast zone\n<12 decisions',ha='center',fontsize=6.5,color='#63826E')
    grid=np.arange(41)
    for m in METHODS:
        est,(lo,hi)=count_bootstrap(e,W,m,grid,'decision_steps');ax.fill_between(grid,lo,hi,color=COLORS[m],alpha=.10,lw=0,step='post');ax.step(grid,est,where='post',color=COLORS[m],lw=1.55)
        ax.text(40.8,est[-1],f'{est[-1]:.1f}',va='center',fontsize=6.7,color=COLORS[m],fontweight='bold')
        source += [dict(panel='b',method=m,x=int(t),estimate=z,ci95_low=l,ci95_high=h) for t,z,l,h in zip(grid,est,lo,hi)]
    ax.set_xlim(0,45);ax.set_xticks([0,10,20,30,40]);ax.set_ylim(0,102);ax.set_ylabel('Episodes reaching the threshold (%)');ax.set_xlabel('Decision budget (actual actions)')
    fig.text(.5,.083,'All initially occluded starts (n = 11,647 per model); curves use episode-pooled denominators.',ha='center',fontsize=6.7,color=MUTED)
    fig.text(.5,.036,'Shaded bands: 95% paired acquisition-folder bootstrap intervals. The 95% horizontal line is a CDF level.',ha='center',fontsize=6.5,color=MUTED)
    export_csv('Fig16_curves_and_intervals.csv',source)
    return fig


def fig17(e,W):
    bins=np.array([20,40,60,80,100,np.inf]);a=e.copy();a['bin']=pd.cut(a.initial_score,bins,right=False,labels=False)
    estimates={};boots={};ns=[]
    for m in METHODS:
        v=a[a.method==m];est=[];boot=[]
        for k in range(5):
            subset=v[v.bin==k];n=subset.groupby('group_index').size().reindex(range(47),fill_value=0).to_numpy();num=subset.groupby('group_index').success.sum().reindex(range(47),fill_value=0).to_numpy()
            est.append(num.sum()/n.sum()*100);den=W@n;boot.append(np.divide(W@num,den,out=np.full(B,np.nan),where=den>0)*100)
            if m==METHODS[0]:ns.append(len(subset))
        estimates[m]=np.array(est);boots[m]=np.array(boot).T
    fig,axs=plt.subplots(1,2,figsize=(180/25.4,94/25.4),gridspec_kw={'width_ratios':[1.4,1]});fig.subplots_adjust(left=.075,right=.975,bottom=.27,top=.75,wspace=.32);legend(fig,y=.98)
    labels=['20-40','40-60','60-80','80-100',r'$\geq100$'];x=np.arange(5);source=[]
    ax=axs[0];panel(ax,'a','Success across initial severity');ax.grid(axis='y',color=GRID,lw=.6)
    for k,m in enumerate(METHODS):
        est=estimates[m];lo,hi=ci(boots[m]);ax.errorbar(x+(k-1)*.065,est,yerr=[est-lo,hi-est],marker=['o','s','D'][k],color=COLORS[m],lw=1.4,ms=4,capsize=2)
        source += [dict(method=m,bin=labels[j],n=ns[j],estimate=est[j],ci95_low=lo[j],ci95_high=hi[j]) for j in range(5)]
    lower=min(np.nanmin(ci(boots[m])[0]) for m in METHODS)
    ax.set_ylim(max(0,np.floor((lower-4)/10)*10),103);ax.set_ylabel('Success rate (%)');ax.set_xlim(-.25,4.25);ax.set_xticks(x,[f'{label}\nn={n:,}' for label,n in zip(labels,ns)]);ax.set_xlabel('Initial occlusion score',labelpad=7)
    ax=axs[1];panel(ax,'b','Paired gain from fusion');ax.grid(axis='y',color=GRID,lw=.6);ax.axhline(0,color='#9AA9B1',lw=.85,ls=(0,(3,3)))
    for k,m in enumerate(METHODS[:2]):
        est=estimates['fusion']-estimates[m];lo,hi=ci(boots['fusion']-boots[m]);ax.errorbar(x+(k-.5)*.065,est,yerr=[est-lo,hi-est],marker=['o','s'][k],color=COLORS[m],lw=1.4,ms=4,capsize=2,label='vs '+NAMES[m].lower())
        source += [dict(method='fusion_minus_'+m,bin=labels[j],n=ns[j],estimate=est[j],ci95_low=lo[j],ci95_high=hi[j]) for j in range(5)]
    ax.set_ylabel('Success-rate gain (pp)');ax.set_xticks(x,labels);ax.set_xlim(-.25,4.25);ax.set_xlabel('Initial occlusion score',labelpad=7);ax.legend(loc='upper left',fontsize=6.6,handlelength=1.4)
    fig.text(.5,.083,'Shared initial states and shared bin membership across all models; n denotes paired starts in each bin.',ha='center',fontsize=6.6,color=MUTED)
    fig.text(.5,.035,'Bins are left-inclusive and right-exclusive. Intervals: 95% paired acquisition-folder bootstrap.',ha='center',fontsize=6.6,color=MUTED)
    export_csv('Fig17_stratified_estimates.csv',source)
    return fig


CAPTIONS={
'Fig14_distributions':('图14｜遮挡改善质量与决策效率','Distributions of occlusion reduction quality and decision efficiency. The panels show (a) terminal occlusion score, (b) relative occlusion reduction and (c) actual decision count for episodes with a net reduction in occlusion. The point-cloud, robot-state and MM-PPO models include 11,207, 11,541 and 11,631 improved episodes, respectively. Shaded half-violins use every observation; dots show a fixed random subsample of up to 450 observations with the observed extrema additionally retained. White circles denote arithmetic means on the original measurement scale; dark bars span the interquartile range, with white median marks. Panel a uses a log(1 + score) display and density-estimation coordinate, retains zero scores, and includes the full observed range. Green regions mark the occlusion threshold of 20 and the fast-zone boundary at fewer than 12 actual decisions. The initial state is not counted as a decision.'),
'Fig15_performance':('图15｜整体性能与融合增益','Scene-equal performance in the corrected full evaluation. Each of 398 point-cloud scenes was evaluated from 40 paired candidate starts using each archived model. The panels report (a) threshold-attainment success rate, (b) net occlusion-reduction rate and (c) the paired difference in success rate between MM-PPO and each single-modality model. Within each scene, rates use initially occluded starts only; the reported estimates give scenes equal weight. The dataset contains 11,647 initially occluded starts and 4,273 initially clear starts per model. Success is defined as final occlusion score <20; it is not a collision-free-success claim. Error bars are percentile 95% intervals from 2,000 resamples of the 47 acquisition folders, retaining all scenes and starts within sampled folders and using the same resamples across models. The intervals describe dataset sampling under this grouping, not variability across model training seeds or independent patients. pp, percentage points.'),
'Fig16_visibility_efficiency':('图16｜终止遮挡分布与决策预算','Terminal visibility and decision-budget performance across all initially occluded starts (n=11,647 per model). (a) Empirical cumulative distributions of final occlusion score, including zero scores and the full long tail. The horizontal coordinate uses log(1 + score). The green region marks scores below 20; the dashed horizontal line denotes the 95% cumulative-probability level, not a confidence level. (b) The fraction of initially occluded episodes that reached the threshold by each decision budget. Episodes that did not reach the threshold remain in the denominator. The budget is capped at 40 actual decisions, with the initial state excluded from the count. Fast-zone success requires fewer than 12 decisions. Curves are episode-pooled and therefore need not equal the scene-equal rates in Figure 15. Bands are 95% paired acquisition-folder bootstrap intervals from 2,000 resamples.'),
'Fig17_robustness':('图17｜不同初始遮挡程度下的表现','Performance stratified by initial occlusion severity. (a) Episode-pooled success rates in five severity bins: [20,40), [40,60), [60,80), [80,100) and [100,infinity). The bins contain 939, 679, 558, 502 and 8,969 paired starts, respectively, with identical membership across models. (b) The paired percentage-point gain of MM-PPO relative to each single-modality baseline. Error bars are 95% percentile intervals from the same 2,000 paired acquisition-folder resamples used throughout the figure set. Counts and error bars make the unequal support of the severity strata explicit. These results describe the evaluated dataset and do not establish independence of the archived model training data.')}


def gallery():
    blocks=[]
    for name,(title,caption) in CAPTIONS.items():
        links=' · '.join(f'<a href="{name}.{ext}">{label}</a>' for ext,label in [('png','600 dpi PNG'),('pdf','矢量PDF'),('svg','可编辑SVG'),('tiff','600 dpi TIFF')])
        blocks.append(f'<section><h2>{title}</h2><img src="{name}.png" alt="{title}"><p class="links">{links}</p><details><summary>英文图注与统计口径</summary><p>{caption}</p></details></section>')
    doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>新版数据 · 正文图组</title><style>*{box-sizing:border-box}body{margin:0;background:#f3f6f7;color:#243946;font:15px/1.75 "Segoe UI","Microsoft YaHei",sans-serif}main{max-width:1200px;margin:auto;padding:30px}header{background:#203e4a;color:white;padding:30px;border-radius:13px}h1{font-size:28px;margin:0}header p{color:#cee1e4}h2{font-size:20px;margin:0 0 14px}section{background:white;padding:24px;margin:23px 0;border:1px solid #dbe5e7;border-radius:12px}img{display:block;width:100%;height:auto}a{color:#117f79}nav{display:flex;flex-wrap:wrap;gap:22px;margin:20px 0}.links{margin:12px 0}details{font-size:13px;color:#647981;border-top:1px solid #e5ecee;padding-top:12px}summary{cursor:pointer}.note{padding:17px;background:#edf5f2;border-left:3px solid #118579}.small{font-size:12px;color:#71838b}@media(max-width:650px){main{padding:14px}section{padding:15px}header{padding:20px}}</style><main><header><h1>新版完整测试 · 正文图组</h1><p>398场景 × 40配对起点 × 3模型 · 白底 · 统一字体与配色 · 全部来自新版评估</p></header><nav><a href="正文图组_矢量合集.pdf">四图PDF合集</a><a href="正文图组_交付包.zip">下载完整图包</a><a href="英文图注.txt">英文图注</a><a href="图表口径与数据来源.txt">统计口径说明</a><a href="figure_manifest.json">来源与校验记录</a></nav><div class="note">图组仅使用 manuscript_aligned_20261002 的完整新测试。图14显示有改善回合；图15按场景等权；图16、17按有效回合合并。所有分母、长尾及不确定性在图注中说明。</div>'''+''.join(blocks)+'''<p class="small">原Word及历史记录保持不变。训练暴露尚未核实，图中不使用“严格未见独立测试”或临床泛化结论。误差线不代表不同训练种子的差异。</p></main></html>'''
    (OUT/'图组预览.html').write_text(doc,encoding='utf-8')


def run():
    OUT.mkdir(exist_ok=True);(OUT/'source_data').mkdir(exist_ok=True);style()
    d,e,groups,W=load()
    with PdfPages(OUT/'正文图组_矢量合集.pdf') as pages:
        figs=[]
        for name,fn in zip(CAPTIONS,[lambda:fig14(e),lambda:fig15(e,W),lambda:fig16(e,W),lambda:fig17(e,W)]):
            fig=fn();save(fig,name,pages);figs.append((name,fig));print('Saved '+name,flush=True)
    # A raster overview is for rapid review only; individual vector figures are the publication files.
    overview,axes=plt.subplots(2,2,figsize=(15,8.5));overview.subplots_adjust(left=.015,right=.985,bottom=.025,top=.95,wspace=.04,hspace=.10)
    for ax,(name,_) in zip(axes.flat,figs):
        ax.imshow(plt.imread(OUT/f'{name}.png'));ax.axis('off');ax.set_title(name.replace('_',' '),loc='left',fontsize=11,pad=4)
    overview.savefig(OUT/'图组总览.png',dpi=180)
    for _,fig in figs:plt.close(fig)
    plt.close(overview)
    (OUT/'英文图注.txt').write_text('\n\n'.join(name+'\n'+cap for name,(_,cap) in CAPTIONS.items()),encoding='utf-8')
    notes='''正文图组说明

数据：仅使用 outputs/manuscript_aligned_20261002/episodes.csv；未使用旧MAT、旧结果或旧图曲线。
规模：398场景×40起点×3模型。每模型11,647个初始遮挡起点，4,273个初始达标起点。
图14：有改善回合，三个模型样本数分别11,207 / 11,541 / 11,631；半小提琴与统计量使用全部数据，散点为固定随机展示子样本并额外保留最小最大值。第一轴用log(1+score)展示与估计密度，保留零值和全范围。
图15：先算各场景有效回合成功率/改善率，再对398场景等权；末图为配对成功率差。成功为终值<20，不自动等同无碰撞成功。
图16：全部初始遮挡回合，采用合并分母；决策预算曲线的分母包含未成功回合，初始帧不计步，最多40动作。CDF的95%水平线不是置信水平。
图17：初始遮挡分组[20,40)、[40,60)、[60,80)、[80,100)、[100,∞)，分组由同一起点确定，三模型一致。
区间：47个采集文件夹为重采样组，成组保留其场景和起点；2,000次配对百分位bootstrap，95%区间，随机种子20261002。该区间不代表患者独立性、重复训练或不同训练种子的变异。
外观：三模型固定蓝/金/青绿，Arial字体，180 mm宽，600 dpi PNG/TIFF，可编辑文字SVG，嵌入字体矢量PDF。图中没有显著性星号或未经计算的P值。
输入数据和Word原文件保持不变。历史模型训练暴露与采集级独立性仍待核验，图组不宣称已经证明严格未见泛化。
英文图注可以随新图替换进后续稿件；图14的实际动作均值和旧“含初始帧步数”不同，应同步更新正文。

复算：在仿真平台目录运行 .\\.venv\\Scripts\\python.exe -m ncsim.publication_figures
source_data 提供每图汇总数值、区间、曲线以及必要的逐回合数据。
'''
    (OUT/'图表口径与数据来源.txt').write_text(notes,encoding='utf-8')
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),input_directory=str(SOURCE),
        input_sha256={p:sha256(SOURCE/p) for p in ['episodes.csv','summary.json','protocol.json','verification.json']},
        code_sha256=sha256(Path(__file__)),original_word_sha256=sha256(ROOT.parent/'JFR0511.docx'),
        methods=METHODS,colors=COLORS,bootstrap=dict(unit='acquisition folder',groups=groups,resamples=B,seed=SEED,paired=True,interval='percentile 2.5/97.5'),
        candidates=47760,eligible_per_model=11647,source_scope='Corrected new evaluation only; no historical outcomes imported')
    (OUT/'figure_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    gallery()
    (ROOT/'outputs/latest_figures.json').write_text(json.dumps(dict(directory=str(OUT),report=str(OUT/'图组预览.html')),ensure_ascii=False,indent=2),encoding='utf-8')
    with zipfile.ZipFile(OUT/'正文图组_交付包.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob('*'):
            if p.is_file() and p.suffix!='.zip':z.write(p,p.relative_to(OUT))
    print(OUT,flush=True)


if __name__=='__main__':run()
