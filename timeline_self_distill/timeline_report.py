"""Preregistered all500 matched deployment and fixed-fresh-base-history report."""
import argparse
import json
from pathlib import Path
import numpy as np
from rl_opd_v1.common import read_json
from .prepare import ROOT
from .full_chain_control import save_once
from .eval import evaluate,SCHEMA
from .control_report import comparison


def collect(directory,rows,base=None):
    cfg=read_json(directory/'run_config.json')
    assert cfg['limit']==500 and cfg['seed']==0 and cfg['fractions']==[100]
    assert cfg['thinking_temperature']==.7 and cfg['cut_answer_temperature']==0
    assert cfg['natural_answer_temperature']==.7 and cfg['max_generation_tokens']==8192 and cfg['cut_answer_max_tokens']==128
    assert cfg['schema']==SCHEMA and cfg['batch_size']==32 and cfg['shards']==1
    assert Path(cfg['model']['tokenizer_path']).resolve()==Path(read_json(ROOT/'model_receipt.json')['path']).resolve()
    source=cfg['model']['saved_prefix_source']
    assert (source is None) if base is None else Path(source).resolve()==base.resolve()
    assert len(list((directory/'raw').glob('*.json')))==500
    vals=[]
    for row in rows:
        r=read_json(directory/'raw'/f'{row["sample_id"]}.json');assert r['sample_id']==row['sample_id'] and not r['ground_truth_in_prompt']
        assert len(r['cuts'])==1 and r['cuts'][0]['trunc_pct']==100
        if base:
            b=read_json(base/'raw'/f'{row["sample_id"]}.json')
            assert all(r[k]==b[k] for k in ('reasoning_token_ids','base_unexpanded_token_ids','expanded_base_token_ids','reasoning_closed','length_capped'))
        c=r['cuts'][0];e=evaluate(c['text'],row);ne=evaluate(r['natural']['text'] if r['natural'] else '',row)
        vals.append({'sample_id':row['sample_id'],'final':e,'natural':ne,'length':r['reasoning_length'],
            'closed':r['reasoning_closed'],'cap':r['length_capped'],
            'affected':bool(r['length_capped'] or not r['reasoning_closed'] or c['finish_reason']=='length' or not e['parse_valid'])})
    return vals


def contrast(first,second,groups):
    out=comparison(np.array([float(v['final']['correct']) for v in first]),np.array([float(v['final']['correct']) for v in second]),
        groups,[a['affected'] or b['affected'] for a,b in zip(first,second)])
    out.pop('robust_pass') # Historical E3 gate must not be reused for this new protocol.
    out['new_utility_gate_pass']=bool(out['delta_pp']>=2 and all(out[k][0]>0 for k in
        ('paired_ci95_pp','image_cluster_ci95_pp','bootstrap_sensitivity_envelope_pp','cluster_sensitivity_envelope_pp')))
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();root=a.run.resolve()
    rows=[json.loads(s) for s in (ROOT/'data/val500.jsonl').read_text().splitlines()];assert len(rows)==500
    images=sorted({r['image_path'] for r in rows});groups=[np.array([i for i,r in enumerate(rows) if r['image_path']==im]) for im in images]
    train={};orders=[]
    for arm in ('late','zero'):
        config=read_json(root/arm/'run_config.json');done=read_json(root/arm/'completion.json')
        assert config['arm']==arm and config['mode']=='pilot' and config['updates']==32 and done['updates']==32 and done['exposures']==256
        assert done['state']=='complete' and done['adapter_delta_l2']>0
        orders.append(config['exposures']);assert len(orders[-1])==256
        assert all(read_json(root/arm/f'update_{i:03d}.json')['gradient_norm_before_clip']>0 for i in range(1,33))
        train[arm]={k:done[k] for k in ('updates','exposures','adapter_delta_l2','gpu_hours','peak_cuda_bytes')}
    assert orders[0]==orders[1]
    values={name:collect(root/name,rows,root/'base_final' if name.endswith('_fixed') else None)
        for name in ('base_final','late_final','zero_final','late_fixed','zero_fixed')}
    for arm in ('base','late','zero'):
        expected=Path('/root/rl_opd_v1_runtime/models/Qwen3.5-2B') if arm=='base' else root/arm/'checkpoint'
        for suffix in (['final'] if arm=='base' else ['final','fixed']):
            assert Path(read_json(root/f'{arm}_{suffix}'/'run_config.json')['model']['path']).resolve()==expected.resolve()
    stats={name:{'accuracy':sum(v['final']['correct'] for v in vs)/500,'natural_accuracy':sum(v['natural']['correct'] for v in vs)/500,
        'mean_thinking_tokens':float(np.mean([v['length'] for v in vs])),'closed_rate':sum(v['closed'] for v in vs)/500,
        'cap_rate':sum(v['cap'] for v in vs)/500,'affected_rows':sum(v['affected'] for v in vs)} for name,vs in values.items()}
    l,z,b=[values[n] for n in ('late_final','zero_final','base_final')]
    primary={'late_minus_zero':contrast(l,z,groups),'late_minus_base':contrast(l,b,groups)}
    ratio=stats['late_final']['mean_thinking_tokens']/stats['base_final']['mean_thinking_tokens']
    guard=bool(ratio>=.8 and stats['late_final']['closed_rate']>=stats['base_final']['closed_rate']-.01-1e-12
        and stats['late_final']['cap_rate']<=stats['base_final']['cap_rate']+.01+1e-12)
    summary={'n':500,'image_clusters':len(groups),'seed':0,'heldout_read':False,'stats':stats,'primary_deployment':primary,
        'secondary_fixed_fresh_base_history':{'late_minus_zero':contrast(values['late_fixed'],values['zero_fixed'],groups),
            'late_minus_base':contrast(values['late_fixed'],b,groups)},
        'late_base_thinking_length_ratio':ratio,'operational_reasoning_guard_pass':guard,
        'joint_utility_gate_pass':bool(guard and all(c['new_utility_gate_pass'] for c in primary.values())),
        'training':train,'paper_ready':False,'temporal_source_necessity_tested':False,
        'interpretation':'Single-seed exploratory utility only. Same-label history conditioning comparison does not isolate temporal teacher origin. Prior E0/E3R gates unchanged.'}
    save_once(root/'metrics.json',summary)
    for name,vs in values.items():save_once(root/(name+'_evaluations.json'),vs)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
