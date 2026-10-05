"""Verify complete three-arm records before scoring frozen original-image GT."""
import argparse
import json
from pathlib import Path
import numpy as np
from .prepare import ROOT
from .eval import evaluate, parse_bbox
from .select import select_boxes
from .control_report import comparison
from .full_chain_control import save_once, split_natural
from .phase_control import INPUTS, config
from .phase_views import ARMS, CAP, ANSWER_MAX, next_natural_limit, view_index, inverse_box, bbox_text

def verify_record(record,row,arm,closing):
    sid=row['sample_id']
    assert record['sample_id']==sid and record['arm']==arm and not record['ground_truth_in_prompt']
    candidates=record['candidates'];assert len(candidates)>=3
    assert [c['candidate_index'] for c in candidates]==list(range(len(candidates)))
    remaining=CAP;bad=False
    for j,c in enumerate(candidates):
        key='original' if arm=='original' else f'p{view_index(arm,j)}'
        assert c['view_key']==key and c['view']==row['views'][key]
        n,a=c['natural'],c['answer'];base=row['views'][key]['expanded_base_ids']
        assert n['view_key']==key
        for obj in [n,a]:
            assert obj['sample_id']==sid and obj['arm']==arm and obj['candidate_index']==obj['seed']==j
            assert not obj['ground_truth_in_prompt']
        limit=next_natural_limit(remaining,len(base),j)
        assert limit is not None and n['natural_limit']==limit and n['remaining_before_pair']==remaining
        assert n['base_unexpanded_token_ids']==row['base_unexpanded_ids'] and n['expanded_base_token_ids']==base
        full=n['natural_full_token_ids'];think,answer,closed=split_natural(full,closing)
        assert len(full)<=limit
        assert n['reasoning_token_ids']==think and n['natural_answer_token_ids_including_eos']==answer
        assert n['reasoning_closed']==closed and n['length_capped']==(n['finish_reason']=='length')
        assert a['expanded_prompt_token_ids']==base+think+[closing]
        assert a['prefill_tokens']==len(a['expanded_prompt_token_ids'])
        assert len(a['generated_token_ids_including_eos'])<=ANSWER_MAX
        assert a['bbox_2d']==parse_bbox(a['text'])
        mapped=inverse_box(a['text'],c['view'])
        assert a['mapped_bbox_2d']==mapped and a['mapped_text']==bbox_text(mapped)
        cost=len(base)+len(full)+a['prefill_tokens']+len(a['generated_token_ids_including_eos'])
        assert cost==c['accounted_tokens'] and cost<=2*len(base)+2*limit+129 and cost<=remaining
        remaining-=cost
        affected=bool(n['length_capped'] or not closed or a['finish_reason']=='length' or mapped is None)
        assert c['affected']==affected;bad=bad or affected
    stop_key='original' if arm=='original' else f'p{view_index(arm,len(candidates))}'
    assert next_natural_limit(remaining,len(row['views'][stop_key]['expanded_base_ids']),len(candidates)) is None
    assert record['stop_remaining_tokens']==remaining and record['next_request_limit'] is None
    assert record['total_accounted_tokens']==CAP-remaining
    texts=[c['answer']['mapped_text'] for c in candidates]
    choice,support=select_boxes(texts)
    selected={'sample_id':sid,'candidate_index':choice,'support':support,
        'text':texts[choice] if choice is not None else '{}','fallback':choice is None,'labels_used_for_selection':False}
    assert record['selection']==selected
    return selected,bad

def contrast(first,second,groups,affected):
    out=comparison(first,second,groups,affected)
    # The reused historical500-only gate is not the new pilot's gate.
    del out['robust_pass']
    delta=first-second
    out['point_identification_interval_pp']=[float(np.where(affected,-1,delta).mean()*100),
                                           float(np.where(affected,1,delta).mean()*100)]
    out['ordinary_and_cluster_lower_positive']=bool(out['paired_ci95_pp'][0]>0 and out['image_cluster_ci95_pp'][0]>0)
    out['sensitivity_lower_positive']=bool(out['bootstrap_sensitivity_envelope_pp'][0]>0 and out['cluster_sensitivity_envelope_pp'][0]>0)
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    run=args.run.resolve();directory=run/'worker'
    assert json.loads((directory/'run_config.json').read_text())==config('pilot')
    worker=json.loads((directory/'worker.json').read_text())
    assert worker['state']=='complete' and worker['completed_samples']==100
    inputs=[json.loads(line) for line in INPUTS.read_text().splitlines()];assert len(inputs)==100
    expected={r['sample_id'] for r in inputs}
    for arm in ARMS:assert {f.stem for f in (directory/'raw'/arm).glob('*.json')}==expected
    token_cfg=json.loads((Path(config('pilot')['model']['path'])/'tokenizer_config.json').read_text())
    close=[int(k) for k,v in token_cfg['added_tokens_decoder'].items() if v['content']=='</think>'];assert len(close)==1
    records={a:[json.loads((directory/'raw'/a/f'{r["sample_id"]}.json').read_text()) for r in inputs] for a in ARMS}
    predictions={};anomalies={}
    for arm in ARMS:
        verified=[verify_record(rec,row,arm,close[0]) for rec,row in zip(records[arm],inputs)]
        predictions[arm]=[v[0] for v in verified];anomalies[arm]=[v[1] for v in verified]
    # Every deployed selection above was checked using GT-free inputs before labels are opened.
    rows=[json.loads(line) for line in (ROOT/'data/train2000.jsonl').read_text().splitlines()][500:600]
    for row,inp in zip(rows,inputs):
        assert all(row[k]==inp[k] for k in ('sample_id','image_path','expression','width','height'))
    evaluations={a:[evaluate(p['text'],r) for p,r in zip(predictions[a],rows)] for a in ARMS}
    scores={a:np.array([float(e['correct']) for e in evaluations[a]]) for a in ARMS}
    coverage={a:np.array([float(any(evaluate(c['answer']['mapped_text'],r)['correct'] for c in rec['candidates'][:3]))
        for r,rec in zip(rows,records[a])]) for a in ARMS}
    paths=sorted({r['image_path'] for r in rows})
    groups=[np.array([i for i,r in enumerate(rows) if r['image_path']==path]) for path in paths]
    pair_pc=[x or y for x,y in zip(anomalies['phase'],anomalies['canonical'])]
    pair_po=[x or y for x,y in zip(anomalies['phase'],anomalies['original'])]
    contrasts={'phase_minus_canonical':contrast(scores['phase'],scores['canonical'],groups,pair_pc),
        'phase_minus_original':contrast(scores['phase'],scores['original'],groups,pair_po),
        'first3_coverage_phase_minus_canonical':contrast(coverage['phase'],coverage['canonical'],groups,pair_pc)}
    positive=bool(contrasts['phase_minus_original']['delta_pp']>=2 and
        all(c['ordinary_and_cluster_lower_positive'] and c['sensitivity_lower_positive'] for c in contrasts.values()))
    costs={a:{'candidates':sum(len(r['candidates']) for r in records[a]),
        'accounted_tokens':sum(r['total_accounted_tokens'] for r in records[a]),
        'used_tokens_min_max':[min(r['total_accounted_tokens'] for r in records[a]),max(r['total_accounted_tokens'] for r in records[a])],
        'candidate_count_min_max':[min(len(r['candidates']) for r in records[a]),max(len(r['candidates']) for r in records[a])],
        'natural_prefill_tokens':sum(len(c['natural']['expanded_base_token_ids']) for r in records[a] for c in r['candidates']),
        'natural_generated_tokens_including_eos':sum(len(c['natural']['natural_full_token_ids']) for r in records[a] for c in r['candidates']),
        'readout_prefill_tokens':sum(c['answer']['prefill_tokens'] for r in records[a] for c in r['candidates']),
        'answer_generated_tokens_including_eos':sum(len(c['answer']['generated_token_ids_including_eos']) for r in records[a] for c in r['candidates']),
        'visual_prefill_tokens_both_passes':sum(2*c['view']['visual_tokens'] for r in records[a] for c in r['candidates']),
        'affected_rows':sum(anomalies[a]),'natural_caps':sum(c['natural']['length_capped'] for r in records[a] for c in r['candidates']),
        'unclosed_naturals':sum(not c['natural']['reasoning_closed'] for r in records[a] for c in r['candidates']),
        'answer_caps':sum(c['answer']['finish_reason']=='length' for r in records[a] for c in r['candidates']),
        'unusable_mapped_answers':sum(c['answer']['mapped_bbox_2d'] is None for r in records[a] for c in r['candidates'])} for a in ARMS}
    summary={'n':100,'image_clusters':len(groups),'seed':0,'bootstrap_resamples':2000,
        'accuracies':{a:float(scores[a].mean()) for a in ARMS},'correct_counts':{a:int(scores[a].sum()) for a in ARMS},
        'first3_any_correct_coverage':{a:float(coverage[a].mean()) for a in ARMS},'contrasts':contrasts,
        'joint_positive_gate':positive,'costs':costs,'worker':worker,
        'affected_ids':{a:[r['sample_id'] for r,b in zip(rows,anomalies[a]) if b] for a in ARMS},
        'heldout_read':False,'original_gates_changed':False,'verified_paper_direction':False,
        'interpretation':'Exploration of whole input-view treatment; token cap not equal FLOPs/latency; no isolated aliasing or cross-model claim.'}
    out=run/'comparison';out.mkdir(exist_ok=False)
    save_once(out/'metrics.json',summary)
    for a in ARMS:
        (out/f'{a}_predictions.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in predictions[a]))
    detail=[{'sample_id':r['sample_id'],'evaluations':{a:evaluations[a][i] for a in ARMS},
        'first3_coverage':{a:int(coverage[a][i]) for a in ARMS},
        'candidate_scores':{a:[evaluate(c['answer']['mapped_text'],r) for c in records[a][i]['candidates']] for a in ARMS}} for i,r in enumerate(rows)]
    save_once(out/'per_sample.json',detail)
    (out/'REPORT.md').write_text('# 三视图探索结果\n\n'+
        '\n'.join(f'- {a}: {summary["correct_counts"][a]}/100；首3覆盖{coverage[a].mean():.0%}；候选{costs[a]["candidates"]}。' for a in ARMS)+
        f'\n\n联合预定门槛：{positive}。原样保留全部100行及异常；不是论文已成立或等价结论。\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
