"""Separate vectorized reconstruction of the completed cross-trajectory comparison."""
import argparse
import json
import random
from pathlib import Path
import numpy as np


def reconstruct(cuts, row):
    boxes=[]
    for cut in cuts:
        try:
            b=json.loads(cut['text'])
            assert set(b)=={'bbox_2d'}
            b=b['bbox_2d']
            assert len(b)==4 and all(type(v)==int and 0<=v<=1000 for v in b)
        except (ValueError,TypeError,AssertionError):
            b=[0,0,0,0]
        boxes.append(b)
    boxes=np.array(boxes,dtype=float)
    areas=np.prod(np.maximum(0,boxes[:,2:]-boxes[:,:2]),axis=1)
    intersection=np.prod(np.maximum(0,np.minimum(boxes[:,None,2:],boxes[None,:,2:])-np.maximum(boxes[:,None,:2],boxes[None,:,:2])),axis=2)
    union=areas[:,None]+areas[None,:]-intersection
    matrix=np.divide(intersection,union,out=np.zeros_like(intersection),where=union>0)
    np.fill_diagonal(matrix,0)
    scores=matrix.sum(axis=1)/(len(cuts)-1)
    winner=int(np.flatnonzero(np.isclose(scores,scores.max(),rtol=0,atol=1e-14))[0])
    boxes*=np.array([row['width'],row['height'],row['width'],row['height']])/1000
    gt=np.array(row['gt_xyxy'])
    intersection=np.prod(np.maximum(0,np.minimum(boxes[:,2:],gt[2:])-np.maximum(boxes[:,:2],gt[:2])),axis=1)
    union=np.prod(np.maximum(0,boxes[:,2:]-boxes[:,:2]),axis=1)+np.prod(np.maximum(0,gt[2:]-gt[:2]))-intersection
    values=np.divide(intersection,union,out=np.zeros_like(intersection),where=union>0)
    return winner,values


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parent; output=a.output.resolve()
    manifest=json.loads((output/'manifest300.json').read_text())
    old_stats=json.loads((root/'results/timeline_vote_statistics_1994_20261005/per_sample.json').read_text())
    assert [r['sample_id'] for r in manifest]==random.Random(0).sample([r['sample_id'] for r in old_stats],300)
    train={r['sample_id']:r for r in map(json.loads,(root/'data/train2000.jsonl').read_text().splitlines())}
    report=json.loads((output/'comparison.json').read_text())
    per_sample=json.loads((output/'comparison_per_sample.json').read_text())
    assert [r['sample_id'] for r in manifest]==[r['sample_id'] for r in per_sample]
    token_totals={k:0 for k in report['tokens'] if k!='cross_to_timeline_ratio'}
    for expected in per_sample:
        sid=expected['sample_id']; row=train[sid]
        old=json.loads((root/'results/fixed_train2000_classification_20261004/sample/raw'/f'{sid}.json').read_text())
        new=[json.loads((output/'sample/raw'/f'{sid}__r{i}.json').read_text()) for i in range(1,10)]
        old_cuts={c['trunc_pct']:c for c in old['cuts']}
        cross=[old_cuts[100]]+[r['cuts'][0] for r in new]
        timeline=[old_cuts[p] for p in range(10,101,10)]
        winner,values=reconstruct(cross,row)
        t_winner,t_values=reconstruct(timeline,row)
        assert winner==expected['chosen_replicate']
        assert bool(values[winner]>=.5)==expected['cross_correct']
        assert bool(np.any(values>=.5))==expected['any_final_correct']
        assert bool(t_values[t_winner]>=.5)==expected['timeline_correct']
        assert bool(values[0]>=.5)==expected['final_correct']
        assert np.allclose(values,[e['iou'] for e in expected['final_evaluations']],rtol=0,atol=1e-12)
        for i,r in enumerate(new,1):
            assert r['sample_id']==sid and r['seed']==r['replicate']==i
            assert r['base_unexpanded_token_ids']==old['base_unexpanded_token_ids']
            assert r['expanded_base_token_ids']==old['expanded_base_token_ids']
            cut=r['cuts'][0]
            assert cut['generated_token_ids']==cut['token_ids']+([248046] if cut['finish_reason']=='stop' else [])
        counts=dict(original_natural_tokens=len(old['natural_full_token_ids']),
                    original_final_readout_tokens=len(old_cuts[100]['token_ids'])+int(old_cuts[100]['finish_reason']=='stop'),
                    timeline_readout_tokens=sum(len(c['token_ids'])+int(c['finish_reason']=='stop') for c in timeline),
                    new_natural_tokens=sum(len(r['natural_full_token_ids']) for r in new),
                    new_final_readout_tokens=sum(len(r['cuts'][0]['generated_token_ids']) for r in new))
        counts['timeline_total_tokens']=counts['original_natural_tokens']+counts['timeline_readout_tokens']
        counts['cross_total_tokens']=sum(counts[k] for k in ('original_natural_tokens','original_final_readout_tokens','new_natural_tokens','new_final_readout_tokens'))
        for k,v in counts.items():
            assert v==expected[k]
            token_totals[k]+=v
    for k,v in token_totals.items(): assert v==report['tokens'][k]
    for m in report['metrics']:
        key=m['method']
        assert m['correct']==sum(r[key] for r in per_sample)
        assert m['rescued']==sum(not r['final_correct'] and r[key] for r in per_sample)
        assert m['harmed']==sum(r['final_correct'] and not r[key] for r in per_sample)
    receipt=dict(status='PASS',n=300,new_trajectories=2700,medoids_recomputed=600,
                 GT_IoUs_recomputed=6000,selection_reproduced=True,all_token_totals_match=True,
                 scope='Separate NumPy implementation, same primary agent; no model calls')
    with (output/'independent_verification.json').open('x') as f:json.dump(receipt,f,indent=2)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__': main()
