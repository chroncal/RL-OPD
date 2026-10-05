"""Frozen cohort, saved-output stratification and paired timeline/cross-path report."""
import argparse
import json
import random
from fractions import Fraction
from pathlib import Path
from .prepare import ROOT
from .full_chain_control import save_once
from .eval import evaluate, parse_bbox
from .timeline_vote_statistics import exact_iou

SAVED = ROOT/'results/timeline_vote_statistics_1994_20261005/per_sample.json'
ORIGINAL = ROOT/'results/fixed_train2000_classification_20261004/sample/raw'


def select(boxes):
    scores = [sum((exact_iou(a,b) for j,b in enumerate(boxes) if i != j), Fraction(0)) / (len(boxes)-1)
              for i,a in enumerate(boxes)]
    return max(range(len(boxes)), key=lambda i:scores[i])


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    saved = json.loads(SAVED.read_text())
    assert len(saved) == 1994
    bins = []
    for lo,hi in [(0,.4),(.4,.6),(.6,.8),(.8,.9),(.9,1)]:
        rows = [r for r in saved if lo <= r['mean_pair_iou'] and (r['mean_pair_iou'] < hi or hi == 1)]
        final = sum(r['final_correct'] for r in rows)
        voted = sum(r['votes']['all']['evaluation']['correct'] for r in rows)
        rescue = sum(not r['final_correct'] and r['votes']['all']['evaluation']['correct'] for r in rows)
        harm = sum(r['final_correct'] and not r['votes']['all']['evaluation']['correct'] for r in rows)
        bins.append(dict(lower=lo,upper=hi,n=len(rows),final_correct=final,final_accuracy_pct=100*final/len(rows),
                         vote_correct=voted,vote_accuracy_pct=100*voted/len(rows),rescued=rescue,harmed=harm))
    assert sum(r['n'] for r in bins)==1994
    save_once(output/'stratified1994.json',bins)
    # Uniform sample without replacement, never inspect correctness to select IDs.
    ids = random.Random(0).sample([r['sample_id'] for r in saved],300)
    train = {r['sample_id']:r for r in map(json.loads,(ROOT/'data/train2000.jsonl').read_text().splitlines())}
    rows = [{k:train[sid][k] for k in ('sample_id','image_path','expression')} for sid in ids]
    save_once(output/'manifest300.json',rows)
    save_once(output/'selection.json',dict(population=1994,n=300,algorithm='Python random.Random(0).sample, ordered frozen1994 population',
        sample_ids=ids,cohort_source=str(SAVED),original_raw=str(ORIGINAL),replicate_seeds=list(range(1,10)),
        caps_policy='Keep all new capped trajectories, use their saved final100 readout as in original infer.py; no resampling',
        token_accounting='Generated output tokens including EOS: natural_full_token_ids + required schema readouts; exclude prompts/prefills/forced closing token and unused0% readout. Shared original trajectory charged once to each method.',
        training=False))


def old_cut_tokens(cut):
    # infer.py strips only EOS from each schema readout. No other stop rules set.
    return len(cut['token_ids']) + int(cut['finish_reason']=='stop')


def report(output):
    manifest = json.loads((output/'manifest300.json').read_text())
    train = {r['sample_id']:r for r in map(json.loads,(ROOT/'data/train2000.jsonl').read_text().splitlines())}
    saved = {r['sample_id']:r for r in json.loads(SAVED.read_text())}
    expected = {f"{r['sample_id']}__r{rep}" for r in manifest for rep in range(1,10)}
    assert {p.stem for p in (output/'sample/raw').glob('*.json')} == expected
    records=[]
    for row in manifest:
        sid=row['sample_id']; gt=train[sid]
        old=json.loads((ORIGINAL/f'{sid}.json').read_text())
        cuts={c['trunc_pct']:c for c in old['cuts']}
        assert not old['length_capped']
        newer=[json.loads((output/'sample/raw'/f'{sid}__r{rep}.json').read_text()) for rep in range(1,10)]
        for rep,r in enumerate(newer,1):
            assert r['sample_id']==sid and r['replicate']==r['seed']==rep
            assert r['ground_truth_in_prompt'] is False and len(r['cuts'])==1 and r['cuts'][0]['trunc_pct']==100
            assert r['base_unexpanded_token_ids']==old['base_unexpanded_token_ids']
            assert r['expanded_base_token_ids']==old['expanded_base_token_ids']
            assert r['natural_generated_token_count']==len(r['natural_full_token_ids'])
            assert r['cuts'][0]['generated_token_count']==len(r['cuts'][0]['generated_token_ids'])==old_cut_tokens(r['cuts'][0])
        finals=[cuts[100]]+[r['cuts'][0] for r in newer]
        boxes=[parse_bbox(c['text']) for c in finals]
        chosen=select(boxes)  # Must precede GT evaluation.
        ev=[evaluate(c['text'],gt) for c in finals]
        timeline=saved[sid]['votes']['all']['evaluation']
        assert ev[0]['correct']==saved[sid]['final_correct']
        # Natural output includes reasoning, sampled answer and any EOS, even if unused.
        old_natural=len(old['natural_full_token_ids'])
        old_final=old_cut_tokens(cuts[100])
        timeline_probes=sum(old_cut_tokens(cuts[p]) for p in range(10,101,10))
        new_natural=sum(r['natural_generated_token_count'] for r in newer)
        new_final=sum(r['cuts'][0]['generated_token_count'] for r in newer)
        records.append(dict(sample_id=sid,final_correct=ev[0]['correct'],timeline_correct=timeline['correct'],
            cross_correct=ev[chosen]['correct'],any_final_correct=any(e['correct'] for e in ev),chosen_replicate=chosen,
            final_evaluations=ev,final_boxes=boxes,
            cap_replicates=[r['replicate'] for r in newer if r['length_capped']],
            unclosed_replicates=[r['replicate'] for r in newer if not r['reasoning_closed']],
            parse_failure_replicates=[i for i,e in enumerate(ev) if not e['parse_valid']],
            original_natural_tokens=old_natural,original_final_readout_tokens=old_final,
            timeline_readout_tokens=timeline_probes,new_natural_tokens=new_natural,new_final_readout_tokens=new_final,
            timeline_total_tokens=old_natural+timeline_probes,
            cross_total_tokens=old_natural+old_final+new_natural+new_final))
    n=len(records); assert n==300
    table=[]
    for key in ('final_correct','timeline_correct','cross_correct','any_final_correct'):
        count=sum(r[key] for r in records)
        table.append(dict(method=key,n=n,correct=count,accuracy_pct=100*count/n,
            rescued=sum(not r['final_correct'] and r[key] for r in records),
            harmed=sum(r['final_correct'] and not r[key] for r in records)))
    token_keys=('original_natural_tokens','original_final_readout_tokens','timeline_readout_tokens','new_natural_tokens',
                'new_final_readout_tokens','timeline_total_tokens','cross_total_tokens')
    tokens={k:sum(r[k] for r in records) for k in token_keys}
    tokens['cross_to_timeline_ratio']=tokens['cross_total_tokens']/tokens['timeline_total_tokens']
    summary=dict(n=n,metrics=table,tokens=tokens,new_trajectories=2700,
                 new_capped=sum(len(r['cap_replicates']) for r in records),
                 new_unclosed=sum(len(r['unclosed_replicates']) for r in records),
                 final_parse_failures=sum(len(r['parse_failure_replicates']) for r in records),
                 scope='Single frozen random300 subset of original noncapped training trajectories, not heldout generalization',
                 training_updates=0)
    save_once(output/'comparison_per_sample.json',records)
    save_once(output/'comparison.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    (prepare if a.prepare else report)(a.output.resolve())
