"""Scientific chart from saved paired trials; no generated or invented data."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .human_evaluation import OUT

def draw():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':.6,'axes.labelcolor':'#263e4b','text.color':'#263e4b','xtick.color':'#526b78','ytick.color':'#526b78','svg.fonttype':'none'})
    cat=json.loads((OUT/'catalog.json').read_text(encoding='utf-8'))['cases']
    example=next(x for x in cat if x['summaries']['cbf']['scenario']==2 and x['summaries']['cbf']['corrected_steps']>0)
    case=json.loads((OUT/example['file']).read_text(encoding='utf-8'))
    colors={'nominal':'#c47a5a','cbf':'#268a81'};labels={'nominal':'MM-PPO','cbf':'MM-PPO + CBF'}
    fig,axs=plt.subplots(2,2,figsize=(7.3,5.5));fig.subplots_adjust(left=.10,right=.98,bottom=.15,top=.93,hspace=.63,wspace=.36)
    for j,(key,scale,ylabel) in enumerate([('human_clearance_m',1000,'Minimum human clearance (mm)'),('field_occlusion_pct',1,'Simulated field occlusion (%)')]):
        ax=axs[0,j]
        for m,run in case['runs'].items():ax.plot([f['simulation_time_s'] for f in run['trace']],[f[key]*scale for f in run['trace']],color=colors[m],lw=1.5,label=labels[m])
        if j==0:ax.axhline(80,color='#9c8c74',ls='--',lw=.8);ax.text(11.8,82,'80 mm',ha='right',fontsize=7,color='#9c8c74')
        ax.set(xlabel='Simulation time (s)',ylabel=ylabel,xlim=(0,12),xticks=[0,4,8,12]);ax.grid(axis='y',lw=.4,alpha=.2)
        ax.set_title(['Dynamic approach: clearance','Dynamic approach: visibility'][j],loc='left',fontsize=9,pad=9)
    axs[0,1].legend(frameon=False,fontsize=7,loc='upper right')
    ax=axs[1,0]
    for m in colors:
        values=[x['summaries'][m]['min_human_clearance_m']*1000 for x in cat]
        ax.plot(np.arange(1,13),values,'o-',ms=3.3,lw=.8,color=colors[m],label=labels[m])
    ax.axhline(80,color='#9c8c74',ls='--',lw=.8);ax.set(xlabel='Paired case index',ylabel='Trial-minimum human clearance (mm)',xticks=[1,3,6,9,12]);ax.grid(axis='y',lw=.4,alpha=.2)
    ax.set_title('All twelve paired cases',loc='left',fontsize=9,pad=9)
    ax=axs[1,1];means=[]
    for i,m in enumerate(colors):
        count=sum(x['summaries'][m]['complete_safe_trial'] for x in cat)
        ax.bar(i,count,color=colors[m],width=.5);ax.text(i,count+.3,f'{count}/12',ha='center',fontsize=9)
    ax.set(xticks=[0,1],xticklabels=['MM-PPO','MM-PPO + CBF'],ylim=(0,14),yticks=[0,4,8,12],ylabel='Complete trials satisfying margins')
    ax.set_title('Twelve-second clearance checks',loc='left',fontsize=9,pad=9)
    for letter,ax in zip('abcd',axs.flat):ax.text(-.22,1.12,letter,transform=ax.transAxes,fontweight='bold',fontsize=12)
    fig.text(.10,.045,'Synthetic humans and equipment; human margin 80 mm, equipment margin 40 mm.\nPaired inputs and prescribed motion. Discrete geometric checks; no hardware validation.',fontsize=7,color='#637b88',linespacing=1.5)
    fig.savefig(OUT/'人体安全对照.png',dpi=300);fig.savefig(OUT/'人体安全对照.svg');plt.close(fig)

if __name__=='__main__':draw()
