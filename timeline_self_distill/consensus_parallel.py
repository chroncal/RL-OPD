"""Resume the live pilot on three disjoint GPU pairs without resetting GRPO."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from .prepare import ROOT
from .full_chain_control import save_once
from .timeline_workflow import run, CPU, INFER, TRAIN
from .consensus_launch import ARMS, run_arm


def load(path):
    return json.loads(path.read_text())


def arm_command(root, arm, devices, port):
    return [TRAIN, '-m', 'timeline_self_distill.consensus_parallel', '--output', str(root),
            '--arm', arm, '--server-device', str(devices[0]), '--trainer-device', str(devices[1]),
            '--port', str(port)]


def main():
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128+signum))
    p=argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--arm', choices=['joint','sd_only'])
    p.add_argument('--server-device', type=int)
    p.add_argument('--trainer-device', type=int)
    p.add_argument('--port', type=int)
    a=p.parse_args();root=a.output.resolve();training=root/'training'
    if not a.arm:
        def terminate_owned(signum, frame):
            # ThreadPoolExecutor waits for workers on exit. Forward the signal
            # to our separate child sessions first so run_arm's finally blocks
            # release their own server/trainer sessions. Do not signal GRPO.
            for name in ('parallel_joint_training','parallel_sd_only_training',
                         'grpo_eval','joint_eval','sd_only_eval','report'):
                launch=root/f'{name}_launch.json'
                if launch.exists() and not (root/f'{name}_exit.json').exists():
                    try:os.killpg(load(launch)['pid'],signal.SIGTERM)
                    except ProcessLookupError:pass
            sys.exit(128+signum)
        signal.signal(signal.SIGTERM,terminate_owned)
    config=load(root/'registered.json')
    assert config['mode']=='pilot' and config['steps']==64
    assert (training/'swift_labeled.jsonl').is_file() and (training/'swift_unlabeled.jsonl').is_file()
    if a.arm:
        run_arm(training,a.arm,64,a.port,config['gate'],config['reward'],config['coefficient'],
                a.server_device,a.trainer_device)
        return
    assert not (root/'completion.json').exists()
    assert (training/'grpo'/'launch.json').is_file()
    assert not (root/'training_exit.json').exists(), 'Original launcher must still be running'
    assert all(not (training/arm).exists() for arm in ('joint','sd_only'))
    devices={'grpo':[0,1],'joint':[2,3],'sd_only':[4,5]}
    commands={arm:arm_command(root,arm,devices[arm],port)
              for arm,port in [('joint',8958),('sd_only',8968)]}
    save_once(root/'parallel_handoff_registered.json',dict(command=sys.argv,devices=devices,commands=commands,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        original_launcher=load(root/'training_launch.json'),scientific_config_unchanged=True,
        reason='User requested concurrent arms on idle GPUs; preserve live GRPO updates',
        duplicate_guard='New arms own their existing output directories; old serial launcher stops at joint mkdir',
        expected_old_launcher_exit=1,heldout_read=False))
    # run() owns each child's session; run_arm() owns its server/trainer sessions.
    # Reservation via mkdir(exist_ok=False) prevents the already-loaded serial
    # launcher from repeating joint/SD after it exports and cleans GRPO.
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(run,cmd,f'parallel_{arm}_training',root,45000)
              for arm,cmd in commands.items()]
        for job in jobs:job.result()
    deadline=time.monotonic()+43200
    while not (root/'training_exit.json').exists():
        if time.monotonic()>deadline:raise RuntimeError('Original GRPO handoff timeout')
        time.sleep(10)
    old_exit=load(root/'training_exit.json')
    old_log=Path(load(root/'training_launch.json')['log']).read_text()
    assert old_exit['exit_code']==1 and 'FileExistsError' in old_log and str(training/'joint') in old_log
    for arm in ARMS:
        out=training/arm
        assert load(out/'completion.json')['steps']==64
        assert (out/'cleanup.json').exists() and (out/'checkpoint'/'config.json').exists()
        assert not (out/'failure.json').exists()
        updates=[load(path) for path in (out/'records').glob('update_*.json')]
        assert len(updates)==64 and {r['step'] for r in updates}==set(range(1,65))
        assert all(r['finite'] for r in updates)
    save_once(root/'parallel_handoff_complete.json',dict(state='complete',original_exit=old_exit,
        original_exit_interpretation='Expected duplicate-launch guard after successful GRPO64 export and cleanup'))
    save_once(training/'completion.json',dict(state='complete',steps=64,training_arms=list(ARMS),parallel=True))
    for group in (('grpo','joint'),('sd_only',)):
        with ThreadPoolExecutor(max_workers=len(group)) as pool:
            jobs=[pool.submit(run,[INFER,'-m','timeline_self_distill.e3_infer','--checkpoint',
                  str(training/arm/'checkpoint'),'--device',str(device),'--output',str(root/f'{arm}_eval')],
                  f'{arm}_eval',root,7200) for device,arm in enumerate(group)]
            for job in jobs:job.result()
    run([CPU,'-m','timeline_self_distill.consensus_report','--output',str(root)],'report',root,600)
    save_once(root/'completion.json',dict(state='complete',mode='pilot',training_steps=64,
        arms=list(ARMS),heldout_read=False,parallel=True))


if __name__=='__main__':main()
