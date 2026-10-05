"""Engineering profile, then registered three-arm training and final val500."""
import argparse
import json
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from .prepare import ROOT
from .full_chain_control import save_once
from .timeline_workflow import run,CPU,INFER,TRAIN


def main():
    signal.signal(signal.SIGTERM,lambda signum,frame:sys.exit(128+signum))
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['profile','pilot'],required=True);p.add_argument('--profile-root',type=Path)
    p.add_argument('--steps',type=int,default=64);p.add_argument('--gate',choices=['middle_changed','lower_only'],default='middle_changed')
    p.add_argument('--reward',choices=['three_tier','continuous'],default='three_tier');p.add_argument('--coefficient',type=float,default=.05)
    a=p.parse_args();root=a.output.resolve();root.mkdir(parents=True,exist_ok=False)
    if a.mode=='pilot':
        assert a.profile_root is not None
        assert json.loads((a.profile_root/'completion.json').read_text())['mode']=='profile'
        for arm in ('grpo','joint','sd_only'):
            assert json.loads((a.profile_root/'training'/arm/'completion.json').read_text())['steps']==2
            assert json.loads((a.profile_root/f'{arm}_export_witness.json').read_text())['state']=='passed'
    else:
        assert a.steps==64,'Profile has exactly two engineering steps per arm regardless of default pilot steps'
    save_once(root/'registered.json',dict(mode=a.mode,command=sys.argv,steps=2 if a.mode=='profile' else a.steps,
        gate=a.gate,reward=a.reward,coefficient=a.coefficient,training_data=str(ROOT/'data/train2000.jsonl'),
        validation=str(ROOT/'data/val500.jsonl'),heldout_read=False,no_intermediate_validation=True))
    if a.mode=='pilot':
        run([CPU,'-m','timeline_self_distill.consensus_report','--baseline','--output',str(root/'baseline')],'baseline',root,600)
    run([TRAIN,'-m','timeline_self_distill.consensus_launch','--output',str(root/'training'),
        '--steps',str(2 if a.mode=='profile' else a.steps),'--gate',a.gate,'--reward',a.reward,
        '--coefficient',str(a.coefficient)],'training',root,7200 if a.mode=='profile' else 129600)
    if a.mode=='profile':
        for arm in ('grpo','joint','sd_only'):
            run([INFER,'-m','timeline_self_distill.e3_checkpoint_witness','--device','0',
                '--checkpoint',str(root/'training'/arm/'checkpoint'),'--output',str(root/f'{arm}_export_witness.json')],
                f'{arm}_export_witness',root,1200)
        for arm in ('grpo','joint','sd_only'):
            path=root/'training'/arm/'records'
            updates=[json.loads(p.read_text()) for p in path.glob('update_*.json')]
            assert len(updates)==2 and any(r['parameter_delta_l2']>0 for r in updates)
            if arm!='grpo':
                rolls=[json.loads(p.read_text()) for p in path.glob('rollout_step_*.json')]
                assert sum(r['active'] for r in rolls)>0
                assert all(r['teacher_generations']==1152 for r in rolls)
    else:
        for group in (('grpo','joint'),('sd_only',)):
            with ThreadPoolExecutor(max_workers=len(group)) as pool:
                jobs=[]
                for device,arm in enumerate(group):
                    cmd=[INFER,'-m','timeline_self_distill.e3_infer','--checkpoint',str(root/'training'/arm/'checkpoint'),
                         '--device',str(device),'--output',str(root/f'{arm}_eval')]
                    jobs.append(pool.submit(run,cmd,f'{arm}_eval',root,7200))
                for job in jobs:job.result()
        run([CPU,'-m','timeline_self_distill.consensus_report','--output',str(root)],'report',root,600)
    save_once(root/'completion.json',dict(state='complete',mode=a.mode,training_steps=2 if a.mode=='profile' else a.steps,
        arms=['grpo','joint','sd_only'],heldout_read=False))


if __name__=='__main__':main()
