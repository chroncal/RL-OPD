"""Owned sequential native-LoRA witness/pilot with bounded resources."""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from .prepare import ROOT
from .full_chain_control import save_once
from .phase_workflow import stop_owned_group
from rl_opd_v1.common import read_json
TRAIN='/root/timeline_grpo_train_env/bin/python'
INFER='/root/best_moment_vllm_env/bin/python'
CPU='/root/rl_opd_curve_plot_env/bin/python'


def run(command,name,root,timeout):
    log=Path('/root')/f'timeline_{root.name}_{name}.log';started=time.time();status=None;cleanup=None
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT.parent)
    with log.open('x') as handle:
        child=subprocess.Popen(command,cwd=ROOT.parent,env=env,stdin=subprocess.DEVNULL,
            stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        save_once(root/f'{name}_launch.json',{'pid':child.pid,'command':command,'log':str(log),'started_unix':started})
        try:
            status=child.wait(timeout=timeout)
            if status:raise RuntimeError(f'{name} exit{status}; inspect {log}')
        finally:
            cleanup=stop_owned_group(child,timeout=30)
            save_once(root/f'{name}_exit.json',{'exit_code':child.returncode,'elapsed_seconds':time.time()-started,
                'cleanup':cleanup,'timeout_seconds':timeout})


def main():
    signal.signal(signal.SIGTERM,lambda signum,frame:sys.exit(128+signum))
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['witness','pilot'],required=True)
    p.add_argument('--device',type=int,choices=[0,1],required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--witness-root',type=Path);a=p.parse_args()
    if a.mode=='pilot':
        assert a.witness_root and read_json(a.witness_root/'completion.json')['mode']=='witness'
        for arm in ('late','zero'):
            assert read_json(a.witness_root/arm/'completion.json')['updates']==2
            assert read_json(a.witness_root/f'{arm}_inference.json')['state']=='passed'
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=False)
    started=time.time();budget=10800 if a.mode=='witness' else 21600
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    save_once(root/'registered_inputs.json',{'command':sys.argv,'source_commit':source,'mode':a.mode,'device':a.device,
        'protocol':str(ROOT/'refine-logs/timeline_reformulation_20261004/EXPERIMENT_PLAN.md'),
        'witness_source':str(a.witness_root.resolve()) if a.witness_root else None,'hard_seconds':budget,'heldout_read':False})
    def stage(cmd,name,cap):
        remaining=budget-(time.time()-started)
        if remaining<=0:raise RuntimeError('Registered overall resource cap exhausted; preserve partial cohort')
        run(cmd,name,root,min(cap,remaining))
    for arm in ('late','zero'):
        stage([TRAIN,'-m','timeline_self_distill.timeline_train','--arm',arm,'--mode',a.mode,
            '--device',str(a.device),'--output',str(root/arm)],arm+'_training',7200)
        if a.mode=='witness':
            stage([INFER,'-m','timeline_self_distill.e3_checkpoint_witness','--checkpoint',str(root/arm/'checkpoint'),
                '--device',str(a.device),'--output',str(root/f'{arm}_inference.json')],arm+'_inference',1800)
    if a.mode=='pilot':
        for arm in ('base','late','zero'):
            checkpoint=Path('/root/rl_opd_v1_runtime/models/Qwen3.5-2B') if arm=='base' else root/arm/'checkpoint'
            stage([INFER,'-m','timeline_self_distill.e3_infer','--checkpoint',str(checkpoint),'--device',str(a.device),
                '--output',str(root/(arm+'_final'))],arm+'_final',5400)
        for arm in ('late','zero'):
            stage([INFER,'-m','timeline_self_distill.e3_infer','--checkpoint',str(root/arm/'checkpoint'),'--device',str(a.device),
                '--output',str(root/(arm+'_fixed')),'--saved-prefix-source',str(root/'base_final')],arm+'_fixed',5400)
        stage([CPU,'-m','timeline_self_distill.timeline_report','--run',str(root)],'report',600)
    save_once(root/'completion.json',{'state':'complete','mode':a.mode,'source_commit':source,'heldout_read':False,
        'gpu_hours_upper_bound_wall':(time.time()-started)/3600})

if __name__=='__main__':main()
