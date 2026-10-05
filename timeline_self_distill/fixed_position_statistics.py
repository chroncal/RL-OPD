"""CPU-only reuse of fixed2000 raw boxes; capped6 excluded for all summaries."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from .prepare import ROOT
from .eval import evaluate
from .full_chain_control import save_once

POSITIONS=list(range(10,100,10))
CELLS=('position_correct__final_correct','position_correct__final_wrong','position_wrong__final_correct','position_wrong__final_wrong')


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    source=a.source.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    rows=[json.loads(s) for s in (ROOT/'data/train2000.jsonl').read_text().splitlines()];assert len(rows)==2000
    files={f.stem:f for f in (source/'sample/raw').glob('*.json')};assert set(files)=={r['sample_id'] for r in rows}
    included=[];excluded=[];source_classification={r['sample_id']:r for r in json.loads((source/'classification2000.json').read_text())}
    for row in rows:
        raw=json.loads(files[row['sample_id']].read_text());assert raw['sample_id']==row['sample_id'] and raw['ground_truth_in_prompt'] is False
        cuts={c['trunc_pct']:c for c in raw['cuts']};assert set(cuts)==set(range(0,101,10))
        if raw['length_capped']:
            assert len(raw['reasoning_token_ids'])==8192 and not raw['reasoning_closed']
            excluded.append({'sample_id':row['sample_id'],'reasoning_length':8192,'previous_any_position_cell':source_classification[row['sample_id']]['cell']})
            continue
        flags=[];ious=[]
        for pct in POSITIONS:
            c=cuts[pct];assert c['cut_token']==len(raw['reasoning_token_ids'])*pct//100
            assert 0<c['cut_token']<len(raw['reasoning_token_ids']), 'All included fixed positions must be valid nonzero prefixes'
            value=evaluate(c['text'],row);assert value==source_classification[row['sample_id']]['probes'][pct//10]['evaluation']
            flags.append(value['correct']);ious.append(value['iou'])
        final=evaluate(cuts[100]['text'],row);assert final==source_classification[row['sample_id']]['final_readout']['evaluation']
        included.append({'sample_id':row['sample_id'],'image_id':Path(row['image_path']).stem,'position_correct':flags,'position_iou':ious,'final_correct':final['correct'],'final_iou':final['iou'],'correct_position_count':sum(flags)})
    assert len(excluded)==6 and len(included)==1994
    n=len(included);final_count=sum(r['final_correct'] for r in included);table=[]
    for j,pct in enumerate(POSITIONS):
        counts=Counter(('position_correct' if r['position_correct'][j] else 'position_wrong')+('__final_correct' if r['final_correct'] else '__final_wrong') for r in included)
        assert sum(counts.values())==n
        correct=counts[CELLS[0]]+counts[CELLS[1]]
        assert counts[CELLS[0]]+counts[CELLS[2]]==final_count
        table.append({'position_pct':pct,'n':n,**{c:counts[c] for c in CELLS},'position_correct_count':correct,'accuracy_pct':100*correct/n,'delta_vs_final_pp':100*(correct-final_count)/n})
    any_rows=[r for r in included if r['correct_position_count']>0]
    histogram={str(k):sum(r['correct_position_count']==k for r in included) for k in range(10)}
    summary={'source':str(source),'original_n':2000,'included_n':n,'excluded_reason':'reasoning reaches8192-token cap; these same6 excluded from every position/final/any-position statistic','excluded_rows':excluded,
        'positions':POSITIONS,'correct_threshold':'IoU>=0.5 against unchanged original GT; parse failures count as wrong',
        'final_definition':'Saved greedy full-reasoning100% prefix schema readout; no fresh reads/natural-final substitution',
        'fixed_position_tables':table,'final':{'n':n,'correct_count':final_count,'wrong_count':n-final_count,'accuracy_pct':100*final_count/n},
        'any_position':{'n':len(any_rows),'mean_correct_positions':sum(r['correct_position_count'] for r in any_rows)/len(any_rows),'only_one_correct_position':histogram['1'],'only_one_fraction_of_any':int(histogram['1'])/len(any_rows),'correct_position_count_histogram':histogram,'maximum_positions':9},
        'heldout_read':False,'new_generation':False,'new_box_readout':False,'training_updates':0}
    save_once(out/'statistics.json',summary);save_once(out/'per_sample.json',included)
    with (out/'fixed_position_tables.csv').open('x') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    plot(summary,out)


def plot(s,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc')
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    x=s['positions']+[100];y=[r['accuracy_pct'] for r in s['fixed_position_tables']]+[s['final']['accuracy_pct']]
    fig,ax=plt.subplots(figsize=(7.8,4.1));ax.plot(x,y,'o-',color='#0072B2',lw=1.8,ms=5)
    ax.plot([100],[y[-1]],'s',color='#222222',ms=6)
    for j,(xp,yp) in enumerate(zip(x,y)):
        ax.annotate(f'{yp:.2f}%',(xp,yp),xytext=(0,12 if j%2==0 else -19),textcoords='offset points',ha='center',fontsize=9)
    ax.set_xticks(x,[str(v) if v!=100 else '100\n（最终）' for v in x],fontproperties=font)
    ax.set_xlim(6,104);ax.set_ylim(0,100);ax.set_yticks(range(0,101,20));ax.grid(axis='y',color='#dddddd',lw=.6)
    ax.set_xlabel('推理前缀位置（%）',fontproperties=font);ax.set_ylabel('定位准确率（%）',fontproperties=font)
    fig.text(.5,.015,'同一批1,994条轨迹；排除6条达到8,192 token上限的轨迹；正确标准：IoU ≥ 0.5',ha='center',fontproperties=font,fontsize=9)
    fig.subplots_adjust(left=.1,right=.985,top=.97,bottom=.22)
    for extension in ('pdf','png'):
        fig.savefig(out/f'accuracy_by_position.{extension}',dpi=300,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':main()
