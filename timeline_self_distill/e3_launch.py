"""Persistent, isolated E3 profile or formal two-arm run; never reads heldout."""
import argparse
import datetime
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from .prepare import ROOT, write_json

ENV=Path('/root/timeline_grpo_train_env')
MODEL='/root/rl_opd_v1_runtime/models/Qwen3.5-2B'
WORK=ROOT.parent
SWIFT=ENV/'bin/swift'


def run_key(root):
    return root.parent.name+'_'+root.name


def local_checkpoint(root, arm):
    # OSS does not support safetensors' serialization syscall (observed OS95).
    # Workflow output is <unique formal root>/training; retain its parent name.
    return Path('/root/timeline_e3_checkpoints')/run_key(root)/arm/'checkpoint'


def add_eval_processor_metadata(checkpoint):
    # TF5 nests video settings in processor_config.json. The frozen TF4 eval
    # processor requires the base's separate video file even for image inputs.
    # Preserve every saved weight/config; add only the observed missing file.
    import shutil
    source = Path(MODEL)/'video_preprocessor_config.json'
    target = checkpoint/source.name
    added = not target.exists()
    if added:
        shutil.copyfile(source,target)
    return {'file':source.name,'added':added,'source':str(source),
            'native_checkpoint_weights_unchanged':True}


def dataset(dest):
    rows=[json.loads(s) for s in (ROOT/'data/train2000.jsonl').read_text().splitlines()]
    assert len(rows)==2000
    result=[]
    for r in rows:
        assert Path(r['image_path']).exists()
        text=r['expression']+'\nReturn the bounding box of the referred object as {"bbox_2d": [x1, y1, x2, y2]}. Use four integers from 0 to 1000 in relative image coordinates. Output only this JSON in the final answer.'
        result.append({'messages':[{'role':'user','content':'<image>'+text}], 'images':[r['image_path']],
            'solution':json.dumps({k:r[k] for k in ('width','height','gt_xyxy')}),
            'sample_id':r['sample_id']})
    dest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in result))
    return hashlib.sha256((ROOT/'data/train2000.jsonl').read_bytes()).hexdigest()


def commands(root, arm, profile=False, port=8934):
    common=['--model',MODEL,'--external_plugins',str(ROOT/'e3_plugin.py'),
        '--torch_dtype','bfloat16','--system','','--enable_thinking','true','--seed','0','--data_seed','0']
    server=[str(SWIFT),'rollout',*common,'--port',str(port),'--host','127.0.0.1',
        '--vllm_max_model_len','12288','--vllm_max_num_seqs','128',
        '--vllm_gpu_memory_utilization','0.7','--vllm_enforce_eager','true',
        '--vllm_enable_prefix_caching','false','--vllm_engine_kwargs',
        json.dumps({'attention_config':{'backend':'FLASH_ATTN'},'async_scheduling':False,
                    'max_num_batched_tokens':16384,'enable_chunked_prefill':True}),
        '--vllm_data_parallel_size','1']
    steps=2 if profile else 64
    trainer=[str(SWIFT),'rlhf',*common,'--rlhf_type','grpo','--reward_funcs','timeline_iou',
        '--dataset',str(root/'swift_train2000.jsonl'),'--split_dataset_ratio','0',
        '--remove_unused_columns','false','--use_vllm','true','--vllm_mode','server',
        '--vllm_server_host','127.0.0.1','--vllm_server_port',str(port),
        '--vllm_server_timeout','1800','--tuner_type','full','--freeze_vit','true','--freeze_aligner','true',
        '--attn_impl','sdpa','--torch_dtype','bfloat16','--padding_free','false','--packing','false',
        '--per_device_train_batch_size','1','--gradient_accumulation_steps','128',
        '--generation_batch_size','128','--num_generations','8','--num_iterations','1',
        '--max_completion_length','8192','--max_length','12288','--temperature','0.7',
        '--top_p','1','--top_k','-1','--beta','0.04','--loss_type','grpo',
        '--scale_rewards','group','--dynamic_sample','false','--overlong_filter','false',
        '--learning_rate','1e-6','--lr_scheduler_type','constant','--warmup_ratio','0',
        '--weight_decay','0','--max_grad_norm','1','--max_steps',str(steps),
        '--gradient_checkpointing','true','--gradient_checkpointing_kwargs','{"use_reentrant":false}',
        '--deepspeed',str(ROOT/'e3_zero2_cpu_optimizer.json'),'--logging_steps','1','--save_steps',str(steps),
        '--save_total_limit','1','--save_only_model','true','--eval_strategy','no',
        '--report_to','wandb','--log_completions','true','--dataset_num_proc','1',
        '--dataloader_num_workers','0','--run_name',f'E3-{arm}-'+run_key(root),
        '--output_dir',str(local_checkpoint(root,arm)),'--add_version','false']
    return server,trainer


def stop_owned(proc, timeout=30):
    if proc.poll() is None:
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL);proc.wait()


def run_arm(root,arm,profile,port):
    server_cmd,train_cmd=commands(root,arm,profile,port)
    out=root/arm;out.mkdir()
    records=out/'records';records.mkdir()
    saved=local_checkpoint(root,arm)
    saved.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy()
    env.update(PYTHONPATH=str(WORK),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false',VLLM_WORKER_MULTIPROC_METHOD='spawn',
        VLLM_NO_USAGE_STATS='1',OMP_NUM_THREADS='4',TIMELINE_ARM=arm,
        TIMELINE_RECORD_DIR=str(records),WANDB_PROJECT='timeline-self-distill',
        WANDB_ENTITY='meow2sakura-bupt',WANDB_DIR='/root/timeline_wandb',
        MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port+100),NPROC_PER_NODE='1',
        NCCL_P2P_DISABLE='1',NCCL_DEBUG='INFO',NCCL_CUMEM_HOST_ENABLE='0')
    key=Path('/root/.config/rl_opd_v1/wandb_api_key')
    if key.exists():env['WANDB_API_KEY']=key.read_text().strip()
    env['CUDA_VISIBLE_DEVICES']='0'
    logbase=Path('/root')/f'timeline_e3_{run_key(root)}_{arm}'
    start=time.time()
    with logbase.with_suffix('.server.log').open('w') as slog, logbase.with_suffix('.trainer.log').open('w') as tlog:
        server=subprocess.Popen(server_cmd,env=env,cwd=WORK,stdin=subprocess.DEVNULL,stdout=slog,stderr=subprocess.STDOUT,start_new_session=True)
        write_json(out/'launch.json',{'server_pid':server.pid,'server_command':server_cmd,'train_command':train_cmd,
            'server_gpu':0,'trainer_gpu':1,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'heldout_read':False,
            'local_checkpoint_directory':str(saved),'artifact_checkpoint_directory':str(out/'checkpoint'),
            'server_log':str(logbase.with_suffix('.server.log')),
            'trainer_log':str(logbase.with_suffix('.trainer.log')),
            'runtime_environment':{k:env[k] for k in ('NCCL_P2P_DISABLE','NCCL_DEBUG','NCCL_CUMEM_HOST_ENABLE')}})
        trainer=None
        try:
            for attempt in range(240):
                if server.poll() is not None:raise RuntimeError('Rollout server failed; inspect server log')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/health/',timeout=2) as response:
                        if response.status==200:break
                except Exception:pass
                time.sleep(3)
            else:raise RuntimeError('Rollout server readiness timeout')
            env['CUDA_VISIBLE_DEVICES']='1'
            trainer=subprocess.Popen(train_cmd,env=env,cwd=WORK,stdin=subprocess.DEVNULL,stdout=tlog,stderr=subprocess.STDOUT,start_new_session=True)
            receipt=json.loads((out/'launch.json').read_text());receipt['trainer_pid']=trainer.pid;write_json(out/'launch.json',receipt)
            code=trainer.wait()
            if code:raise RuntimeError(f'Trainer failed exit={code}; inspect trainer log')
            # Ordinary streaming copy supports this mount. Never copy a partial
            # failed save and never overwrite an older checkpoint artifact.
            import shutil
            shutil.copytree(saved,out/'checkpoint',copy_function=shutil.copyfile,
                ignore=shutil.ignore_patterns('last-checkpoint'))
            compatibility = add_eval_processor_metadata(out/'checkpoint'/f'checkpoint-{2 if profile else 64}')
            write_json(out/'checkpoint_copy_receipt.json',{'local':str(saved),
                'artifact':str(out/'checkpoint'),'trainer_exit':code,
                'ignored_alias':'last-checkpoint',
                'eval_processor_compatibility':compatibility,
                'files':[{'path':str(p.relative_to(saved)),'bytes':p.stat().st_size}
                         for p in saved.rglob('*') if p.is_file()]})
            write_json(out/'completion.json',{'state':'complete','steps':2 if profile else 64,
                'seconds':time.time()-start,'gpu_hours_both_devices':2*(time.time()-start)/3600,'heldout_read':False})
        except Exception as error:
            write_json(out/'failure.json',{'state':'failed','error':str(error),'seconds':time.time()-start,
                'gpu_hours_both_devices':2*(time.time()-start)/3600,'heldout_read':False})
            raise
        finally:
            if trainer is not None:stop_owned(trainer)
            stop_owned(server)
            for name, suffix in [('server','server.log'),('trainer','trainer.log')]:
                import shutil
                shutil.copyfile(logbase.with_suffix('.'+suffix),out/(name+'.log'))


def main():
    # The workflow may terminate this coordinator; enter run_arm's finally so
    # separately sessioned server/trainer processes are also cleaned up.
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128+signum))
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--profile',action='store_true');p.add_argument('--arm',choices=['control','method','both'],default='both')
    p.add_argument('--port',type=int,default=8934);args=p.parse_args()
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    digest=dataset(root/'swift_train2000.jsonl')
    write_json(root/'run_config.json',{'seed':0,'steps':2 if args.profile else 64,'G':8,'prompts_per_step':16,
        'train_manifest_sha256':digest,'profile':args.profile,'model':json.loads((ROOT/'model_receipt.json').read_text()),
        'protocol_sha256':hashlib.sha256((ROOT/'refine-logs/E3_PILOT_PROTOCOL.md').read_bytes()).hexdigest(),
        'command':sys.argv,'heldout_read':False})
    for arm in ['control','method'] if args.arm=='both' else [args.arm]:
        run_arm(root,arm,args.profile,args.port)
if __name__=='__main__':main()
