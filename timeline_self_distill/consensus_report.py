"""Frozen val500 baseline and four-system final-checkpoint comparison."""
import argparse
import json
from pathlib import Path
import numpy as np
from .prepare import ROOT
from .full_chain_control import save_once
from .eval import evaluate,parse_bbox
from .timeline_vote_statistics import vote


def rows():
    values=[json.loads(x) for x in (ROOT/'data/val500.jsonl').read_text().splitlines()]
    assert len(values)==500
    return values


def baseline(output):
    output.mkdir(parents=True,exist_ok=False)
    values=[]
    for row in rows():
        raw=json.loads((ROOT/'results/e0_formal/raw'/f"{row['sample_id']}.json").read_text())
        cuts={c['trunc_pct']:c for c in raw['cuts']}
        chosen,stability=vote([parse_bbox(cuts[p]['text']) for p in range(10,101,10)])
        selected=chosen['all']['position_pct']
        final=evaluate(cuts[100]['text'],row)
        voted=evaluate(cuts[selected]['text'],row)
        values.append(dict(sample_id=row['sample_id'],image_id=Path(row['image_path']).stem,
            final_correct=final['correct'],vote_correct=voted['correct'],final_iou=final['iou'],vote_iou=voted['iou'],
            vote_position=selected,stability=stability,cap=raw['length_capped'],reasoning_length=raw['reasoning_length']))
    final=sum(r['final_correct'] for r in values);voted=sum(r['vote_correct'] for r in values)
    assert final==274 and voted==290
    summary=dict(n=500,base_final_correct=final,base_vote_correct=voted,base_final_accuracy_pct=100*final/500,
        base_vote_accuracy_pct=100*voted/500,rescued=sum(not r['final_correct'] and r['vote_correct'] for r in values),
        harmed=sum(r['final_correct'] and not r['vote_correct'] for r in values),caps_retained=sum(r['cap'] for r in values),
        source=str(ROOT/'results/e0_formal/raw'),training_reference_vote_pct=68.60581745235707)
    train_ids={Path(r['image_path']).stem for r in map(json.loads,(ROOT/'data/train2000.jsonl').read_text().splitlines())}
    assert not train_ids.intersection(r['image_id'] for r in values)
    save_once(output/'baseline.json',summary);save_once(output/'baseline_per_sample.json',values)
    print(json.dumps(summary,indent=2))


def ci(a,b,images):
    diff=np.array(a,dtype=float)-np.array(b,dtype=float);rng=np.random.default_rng(0)
    means=diff[rng.integers(0,len(diff),size=(2000,len(diff)))].mean(axis=1)
    groups={sid:np.flatnonzero(np.array(images)==sid) for sid in sorted(set(images))}
    keys=list(groups);cluster=[]
    for _ in range(2000):
        ix=np.concatenate([groups[keys[j]] for j in rng.integers(0,len(keys),size=len(keys))])
        cluster.append(diff[ix].mean())
    return dict(delta_pp=100*float(diff.mean()),paired95_pp=(100*np.percentile(means,[2.5,97.5])).tolist(),
                image_cluster95_pp=(100*np.percentile(cluster,[2.5,97.5])).tolist())


def report(root):
    old=json.loads((root/'baseline/baseline_per_sample.json').read_text())
    base=np.array([r['final_correct'] for r in old]);voted=np.array([r['vote_correct'] for r in old])
    table=[];per_sample=[];flags={}
    for name in ('grpo','joint','sd_only'):
        path=root/f'{name}_eval';raws={p.stem:p for p in (path/'raw').glob('*.json')}
        assert set(raws)=={r['sample_id'] for r in old}
        values=[];caps=0;invalid=0;length=[]
        for row in rows():
            raw=json.loads(raws[row['sample_id']].read_text())
            assert len(raw['cuts'])==1 and raw['cuts'][0]['trunc_pct']==100
            ev=evaluate(raw['cuts'][0]['text'],row)
            values.append(ev['correct']);caps+=raw['length_capped'];invalid+=not ev['parse_valid'];length.append(raw['reasoning_length'])
            per_sample.append(dict(arm=name,sample_id=row['sample_id'],evaluation=ev,cap=raw['length_capped'],reasoning_length=raw['reasoning_length']))
        flags[name]=values;correct=int(np.sum(values));current=np.array(values)
        table.append(dict(arm=name,n=500,correct=correct,accuracy_pct=100*correct/500,
            repaired=int(np.sum(current & ~base)),harmed=int(np.sum(~current & base)),
            delta_vs_base_pp=100*float((current.astype(float)-base).mean()),
            delta_vs_val_vote_pp=100*float((current.astype(float)-voted).mean()),
            reaches_68_6=100*correct/500>=68.6,caps=caps,invalid=invalid,mean_reasoning_length=float(np.mean(length))))
    images=[r['image_id'] for r in old]
    comparisons={'joint_vs_grpo':ci(flags['joint'],flags['grpo'],images),
                 'sd_only_vs_base':ci(flags['sd_only'],base,images),
                 'joint_vs_val_vote':ci(flags['joint'],voted,images)}
    training=[];exposure_orders={}
    for arm in ('grpo','joint','sd_only'):
        path=root/'training'/arm
        complete=json.loads((path/'completion.json').read_text());steps=complete['steps']
        files=sorted((path/'records').glob('rollout_step_*.json'));assert len(files)==steps
        active=teacher=cap=natural_tokens=final_tokens=middle_tokens=0
        exposures=[]
        for file in files:
            step=json.loads(file.read_text());assert len(step['samples'])==128
            ids=[r['sample_id'] for r in step['samples']]
            assert all(len(set(ids[j:j+8]))==1 for j in range(0,128,8))
            exposures.extend(ids[::8])
            assert step['teacher_generations']==(0 if arm=='grpo' else 1152)
            active+=step['active'];teacher+=step['teacher_generations'];cap+=sum(r['natural_cap'] for r in step['samples'])
            natural_tokens+=sum(r['natural_generated_tokens'] for r in step['samples'])
            final_tokens+=sum(r['schema_final_generated_tokens'] for r in step['samples'])
            middle_tokens+=sum(c['generated_tokens'] for r in step['samples'] for c in r['candidates'] if c['pct']<100)
            if arm=='sd_only':
                assert step['GT_used'] is False
                assert all(r['GT_used'] is False and 'final_evaluation' not in r for r in step['samples'])
        training.append(dict(arm=arm,steps=steps,rollouts=steps*128,active=active,active_fraction=active/(steps*128),
            middle_readouts=teacher,natural_caps=cap,gpu_hours=complete['gpu_hours_both_devices'],GT_in_training=arm!='sd_only',
            natural_generated_tokens=natural_tokens,schema_final_generated_tokens=final_tokens,
            middle_generated_tokens=middle_tokens,total_generated_tokens=natural_tokens+final_tokens+middle_tokens))
        exposure_orders[arm]=exposures
    assert exposure_orders['grpo']==exposure_orders['joint']==exposure_orders['sd_only']
    summary=dict(n=500,baseline=json.loads((root/'baseline/baseline.json').read_text()),table=table,
        comparisons=comparisons,training=training,seed=0,checkpoint_selection=False,heldout_read=False,
        matched_prompt_exposure_order=True,prompt_exposures_per_arm=len(exposure_orders['grpo']))
    save_once(root/'comparison.json',summary);save_once(root/'per_sample_evaluations.json',per_sample)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline',action='store_true');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();(baseline if a.baseline else report)(a.output.resolve())
