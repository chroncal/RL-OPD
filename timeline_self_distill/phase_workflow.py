"""One owned three-arm worker, bounded runtime, then complete-cohort-only GT report."""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from .prepare import ROOT,write_json
from .full_chain_control import save_once

def stop_owned_group(child,timeout=30):
    """Clean our session even if its leader already exited before descendants."""
    receipt={'owned_process_group':child.pid,'sent_sigterm':False,'sent_sigkill':False}
    try:
        os.killpg(child.pid,signal.SIGTERM);receipt['sent_sigterm']=True
    except ProcessLookupError:
        child.wait();return receipt
    try:child.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid,signal.SIGKILL);receipt['sent_sigkill']=True;child.wait()
    # Leader completion does not imply engine/descendant completion.
    until=time.monotonic()+2
    while time.monotonic()<until:
        try:os.killpg(child.pid,0)
        except ProcessLookupError:return receipt
        time.sleep(.1)
    try:
        os.killpg(child.pid,signal.SIGKILL);receipt['sent_sigkill']=True
    except ProcessLookupError:pass
    return receipt

def run_child(command,name,root,timeout):
    log=Path('/root')/f'timeline_{root.name}_{name}.log'
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT.parent)
    started=time.time()
    with log.open('x') as handle:
        child=subprocess.Popen(command,cwd=ROOT.parent,env=env,stdin=subprocess.DEVNULL,
            stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        save_once(root/f'{name}_launch.json',{'pid':child.pid,'command':command,'log':str(log)})
        try:
            status=child.wait(timeout=timeout)
            if status:raise RuntimeError(f'{name} exit{status}; primary log: {log}')
        finally:
            cleanup=stop_owned_group(child,timeout=30)
            save_once(root/f'{name}_cleanup.json',cleanup)
    save_once(root/f'{name}_exit.json',{'exit_code':status,'elapsed_seconds':time.time()-started,
        'exceeded_two_hour_estimate':time.time()-started>7200})

def main():
    signal.signal(signal.SIGTERM,lambda signum,frame:sys.exit(128+signum))
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',type=int,choices=[0,1],required=True)
    p.add_argument('--mode',choices=['pilot','synthetic'],required=True);a=p.parse_args()
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=False)
    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    command=['/root/best_moment_vllm_env/bin/python','-m','timeline_self_distill.phase_control',
        '--device',str(a.device),'--mode',a.mode,'--output',str(root/'worker')]
    save_once(root/'registered_inputs.json',{'source_commit':source_commit,'command':command,
        'protocol':str(ROOT/'refine-logs/phase_view_pilot_20261004/EXPERIMENT_PLAN.md'),
        'mode':a.mode,'seed':0,'heldout_read':False,'new_training':False,'timeout_seconds':10800})
    run_child(command,'worker',root,10800)
    if a.mode=='pilot':
        command=['/root/rl_opd_curve_plot_env/bin/python','-m','timeline_self_distill.phase_report','--run',str(root)]
        run_child(command,'comparison',root,600)
    write_json(root/'completion.json',{'state':'complete','mode':a.mode,'heldout_read':False,
        'command':sys.argv,'source_commit':source_commit})

if __name__=='__main__':main()
