"""One reused two-record sanity, then two owned inference workers; no training."""
import argparse
import json
import signal
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from .prepare import ROOT
from .full_chain_control import save_once
from .timeline_workflow import run, CPU, INFER


def main():
    signal.signal(signal.SIGTERM,lambda signum,frame:sys.exit(128+signum))
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=a.output.resolve()
    assert (root/'manifest300.json').exists()
    def command(device, stage='full'):
        return [INFER,'-m','timeline_self_distill.cross_trajectory_infer','--output',str(root/'sample'),
                '--manifest',str(root/'manifest300.json'),'--device',str(device),'--shard',str(device),
                '--shards','2','--stage',stage,'--seed','0','--batch-size','32']
    save_once(root/'launch_protocol.json',dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        commands=[command(0,'sanity')+['--max-tasks','2'],command(0),command(1)],
        training=False,heldout_read=False,hard_seconds_per_worker=7200))
    run(command(0,'sanity')+['--max-tasks','2'],'sanity',root,1200)
    sanity=list((root/'sample/raw').glob('*.json')); assert len(sanity)==2
    for path in sanity:
        raw=json.loads(path.read_text())
        assert len(raw['cuts'])==1 and raw['cuts'][0]['trunc_pct']==100
        assert raw['cuts'][0]['bbox_2d'] is not None
    save_once(root/'sanity_pass.json',dict(n=2,reused_in_full=True,training=False))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(run,command(device),f'gpu{device}',root,7200) for device in (0,1)]
        for future in futures: future.result()
    run([CPU,'-m','timeline_self_distill.cross_trajectory_report','--output',str(root)],'report',root,600)
    save_once(root/'completion.json',dict(state='complete',n=300,new_trajectories=2700,training_updates=0))


if __name__=='__main__':main()
