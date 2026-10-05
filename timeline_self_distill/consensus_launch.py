"""Three independent matched LoRA runs using the warm Swift rollout/training stack."""
import argparse
import datetime
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from .prepare import ROOT,write_json
from .e3_launch import commands as old_commands,local_checkpoint,SWIFT,WORK,ENV
from .phase_workflow import stop_owned_group

ARMS=('grpo','joint','sd_only')


def dataset(root):
    rows=[json.loads(line) for line in (ROOT/'data/train2000.jsonl').read_text().splitlines()]
    assert len(rows)==2000
    labeled=[];unlabeled=[]
    for r in rows:
        text=r['expression']+'\nReturn the bounding box of the referred object as {"bbox_2d": [x1, y1, x2, y2]}. Use four integers from 0 to 1000 in relative image coordinates. Output only this JSON in the final answer.'
        value={'messages':[{'role':'user','content':'<image>'+text}],'images':[r['image_path']],'sample_id':r['sample_id']}
        unlabeled.append(value)
        labeled.append({**value,'solution':json.dumps({k:r[k] for k in ('width','height','gt_xyxy')})})
    assert all(set(r)=={'messages','images','sample_id'} for r in unlabeled)
    for name,values in [('labeled',labeled),('unlabeled',unlabeled)]:
        path=root/f'swift_{name}.jsonl';assert not path.exists()
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in values))


def commands(root,arm,steps,port):
    server,trainer=old_commands(root,arm,profile=steps==2,port=port)
    def replace(cmd,key,value):cmd[cmd.index(key)+1]=str(value)
    for cmd in (server,trainer):replace(cmd,'--external_plugins',ROOT/'consensus_plugin.py')
    # Rollout serves merged current language weights; native Swift merges/unmerges
    # adapters on sync with vllm_enable_lora=False, avoiding a second LoRA backend.
    replace(server,'--tuner_type','full') if '--tuner_type' in server else None
    replace(trainer,'--tuner_type','lora')
    replace(trainer,'--dataset',root/f"swift_{'unlabeled' if arm=='sd_only' else 'labeled'}.jsonl")
    replace(trainer,'--reward_funcs','consensus_unlabeled' if arm=='sd_only' else 'consensus_iou')
    replace(trainer,'--max_steps',steps);replace(trainer,'--save_steps',steps)
    replace(trainer,'--learning_rate','1e-4');replace(trainer,'--beta','0' if arm=='sd_only' else '.04')
    replace(trainer,'--report_to','none');replace(trainer,'--run_name',f'Consensus-{arm}-{root.parent.name}')
    k=trainer.index('--deepspeed');del trainer[k:k+2]
    modules=[r['name'] for r in json.loads((ROOT/'refine-logs/timeline_reformulation_20261004/LORA_META_TARGETS.json').read_text())['modules']]
    assert len(modules)==186 and all('language_model.layers.' in n for n in modules)
    trainer+=['--lora_rank','8','--lora_alpha','16','--lora_dropout','0','--target_modules',*modules,
              '--vllm_enable_lora','false']
    return server,trainer


def run_arm(root,arm,steps,port,gate,reward,coefficient,server_device=0,trainer_device=1):
    assert server_device in range(6) and trainer_device in range(6) and server_device!=trainer_device
    out=root/arm;out.mkdir(exist_ok=False);records=out/'records';records.mkdir()
    local=local_checkpoint(root,arm);local.mkdir(parents=True,exist_ok=False)
    server_cmd,train_cmd=commands(root,arm,steps,port)
    # Swift otherwise uses the same NCCL weight-sync port (51216) for every
    # server. Keep the live GRPO default, isolate both concurrent new arms.
    group_port=51216 if (server_device,trainer_device)==(0,1) else port+42000
    if group_port!=51216:train_cmd+=['--vllm_server_group_port',str(group_port)]
    env=os.environ.copy()
    env.update(PYTHONPATH=str(WORK),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_NO_USAGE_STATS='1',OMP_NUM_THREADS='4',CONSENSUS_ARM=arm,
        CONSENSUS_RECORD_DIR=str(records),CONSENSUS_GATE=gate,CONSENSUS_REWARD=reward,CONSENSUS_LAMBDA=str(coefficient),
        MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port+100),NPROC_PER_NODE='1',NCCL_P2P_DISABLE='1',NCCL_DEBUG='INFO',
        NCCL_CUMEM_HOST_ENABLE='0',WANDB_MODE='disabled',CUDA_VISIBLE_DEVICES=str(server_device))
    # Actual capped-rollout backward OOM left 7.74 GiB of unused reserve.
    # Set both spellings: this Torch build prioritizes the legacy variable.
    env['PYTORCH_ALLOC_CONF']=env['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    started=time.time();server=None;trainer=None
    logbase=Path('/root')/f'consensus_{root.parent.name}_{arm}'
    with logbase.with_suffix('.server.log').open('x') as sl,logbase.with_suffix('.trainer.log').open('x') as tl:
        try:
            server=subprocess.Popen(server_cmd,cwd=WORK,env=env,stdin=subprocess.DEVNULL,stdout=sl,stderr=subprocess.STDOUT,start_new_session=True)
            launch={'server_pid':server.pid,'server_command':server_cmd,'trainer_command':train_cmd,'arm':arm,
                'server_gpu':server_device,'trainer_gpu':trainer_device,'port':port,'group_port':group_port,'steps':steps,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'server_log':str(logbase.with_suffix('.server.log')),
                'trainer_log':str(logbase.with_suffix('.trainer.log')),'gate':gate,'reward':reward,'lambda':coefficient,'GT_in_training':arm!='sd_only'}
            launch['allocator_config']={k:env[k] for k in ('PYTORCH_ALLOC_CONF','PYTORCH_CUDA_ALLOC_CONF')}
            write_json(out/'launch.json',launch)
            for attempt in range(240):
                if server.poll() is not None:raise RuntimeError('Server failed; inspect recorded log')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/health/',timeout=2) as response:
                        if response.status==200:break
                except Exception:pass
                time.sleep(3)
            else:raise RuntimeError('Server readiness timeout')
            env['CUDA_VISIBLE_DEVICES']=str(trainer_device)
            trainer=subprocess.Popen(train_cmd,cwd=WORK,env=env,stdin=subprocess.DEVNULL,stdout=tl,stderr=subprocess.STDOUT,start_new_session=True)
            launch['trainer_pid']=trainer.pid;write_json(out/'launch.json',launch)
            # The measured joint profile takes ~311 s/update (~5.5 h/64).
            # Allow up to twice that estimate; do not add optimizer updates.
            code=trainer.wait(timeout=43200)
            if code:raise RuntimeError(f'Trainer exit{code}; preserve logs and inspect')
            adapter=local/f'checkpoint-{steps}'
            assert (adapter/'adapter_model.safetensors').is_file()
            export_cmd=[str(ENV/'bin/python'),'-m','timeline_self_distill.consensus_export','--adapter',str(adapter),
                '--local',str(local/'merged'),'--output',str(out/'checkpoint')]
            with logbase.with_suffix('.export.log').open('x') as el:
                export=subprocess.run(export_cmd,cwd=WORK,env=env,stdout=el,stderr=subprocess.STDOUT,timeout=1200)
                assert export.returncode==0
            write_json(out/'completion.json',{'state':'complete','steps':steps,'seconds':time.time()-started,
                'gpu_hours_both_devices':2*(time.time()-started)/3600,'GT_in_training':arm!='sd_only','trainer_exit':code})
        except Exception as error:
            write_json(out/'failure.json',{'error':str(error),'seconds':time.time()-started})
            raise
        finally:
            cleanup={}
            if trainer is not None:cleanup['trainer']=stop_owned_group(trainer,timeout=30)
            if server is not None:cleanup['server']=stop_owned_group(server,timeout=30)
            write_json(out/'cleanup.json',cleanup)
            for suffix in ('server.log','trainer.log','export.log'):
                path=logbase.with_suffix('.'+suffix)
                if path.exists():shutil.copyfile(path,out/suffix)


def main():
    signal.signal(signal.SIGTERM,lambda signum,frame:sys.exit(128+signum))
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=64)
    p.add_argument('--arm',choices=[*ARMS,'all'],default='all');p.add_argument('--port',type=int,default=8948)
    p.add_argument('--gate',choices=['middle_changed','lower_only'],default='middle_changed')
    p.add_argument('--reward',choices=['three_tier','continuous'],default='three_tier');p.add_argument('--coefficient',type=float,default=.05)
    a=p.parse_args();root=a.output.resolve();root.mkdir(parents=True,exist_ok=False)
    dataset(root)
    write_json(root/'run_config.json',dict(command=sys.argv,seed=0,steps=a.steps,G=8,prompts_per_step=16,
        gate=a.gate,reward=a.reward,lambda_coefficient=a.coefficient,LoRA_rank=8,LoRA_alpha=16,lr=1e-4,
        visual_frozen=True,no_checkpoint_selection=True,GT_in_unlabeled_manifest=False,
        model=json.loads((ROOT/'model_receipt.json').read_text())))
    for which in ARMS if a.arm=='all' else [a.arm]:run_arm(root,which,a.steps,a.port,a.gate,a.reward,a.coefficient)
    write_json(root/'completion.json',{'state':'complete','steps':a.steps,'training_arms':list(ARMS) if a.arm=='all' else [a.arm]})


if __name__=='__main__':main()
