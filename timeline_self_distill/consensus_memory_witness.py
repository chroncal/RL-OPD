"""Saved failing-window memory witness: no generation, optimizer or validation."""
import argparse
import json
import os
import time
from pathlib import Path
from .prepare import ROOT,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['0','1'],default='1');a=p.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES=a.device,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',
        PYTORCH_ALLOC_CONF='expandable_segments:True',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    import torch
    from accelerate import Accelerator
    from swift.model import get_model_processor
    from peft import LoraConfig,get_peft_model
    from trl.trainer.utils import selective_log_softmax
    from .teacher_signal import extend_image_inputs,coordinate_indices
    from .consensus_objectives import coordinate_ce
    torch.set_num_threads(4);torch.manual_seed(0);started=time.time()
    source=ROOT/'results/consensus_pilot64_20261005/training/grpo/records/rollout_step_0003.json'
    records=json.loads(source.read_text())['samples'];assert len(records)==128
    target=max(range(128),key=lambda j:len(records[j]['response_ids']))
    assert len(records[target]['thinking_ids'])==8192
    rows={x['sample_id']:{k:x[k] for k in ('sample_id','image_path','expression')}
          for x in map(json.loads,(ROOT/'data/train2000.jsonl').read_text().splitlines())}
    modules=[x['name'] for x in json.loads((ROOT/'refine-logs/timeline_reformulation_20261004/LORA_META_TARGETS.json').read_text())['modules']]
    model_path='/root/rl_opd_v1_runtime/models/Qwen3.5-2B'
    # Use the actual native loader; it preserves linear-attention F32 states.
    model,processor=get_model_processor(model_path,torch_dtype=torch.bfloat16,
        device_map='cpu',attn_impl='sdpa',model_type='qwen3_5')
    f32_states=[(n,p) for n,p in model.named_parameters()
                if '.linear_attn.A_log' in n or '.linear_attn.norm.weight' in n]
    assert len(f32_states)==36 and all(p.dtype==torch.float32 for n,p in f32_states)
    model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=0,target_modules=modules,task_type='CAUSAL_LM'))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads();model.train()
    accelerator=Accelerator(mixed_precision='bf16',gradient_accumulation_steps=128)
    model=accelerator.prepare(model)
    trainable=[p for p in model.parameters() if p.requires_grad]
    assert sum(p.numel() for p in trainable)==8409600
    assert all(p.dtype==torch.float32 for p in trainable)
    # Match the native prepared window's distinct per-rollout image tensors.
    cache={};bases=[];inputs=[]
    for rec in records:
        sid=rec['sample_id'];row=rows[sid]
        if sid not in cache:
            text=row['expression']+'\nReturn the bounding box of the referred object as {"bbox_2d": [x1, y1, x2, y2]}. Use four integers from 0 to 1000 in relative image coordinates. Output only this JSON in the final answer.'
            messages=[{'role':'user','content':[{'type':'image','image':row['image_path']},{'type':'text','text':text}]}]
            cache[sid]=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
                enable_thinking=True,return_dict=True,return_tensors='pt')
        base={k:v.clone().to(accelerator.device) if torch.is_tensor(v) else v for k,v in cache[sid].items()}
        assert base['input_ids'][0].tolist()==rec['base_ids'],'Saved original image prefix must be exact'
        bases.append(base);inputs.append(extend_image_inputs(base,rec['base_ids']+rec['response_ids']))
    def logps(j):
        n=len(records[j]['response_ids'])
        logits=model(**inputs[j],use_cache=False,logits_to_keep=n+1).logits[:,:-1,:]
        assert logits.dtype==torch.float32 and logits.shape[1]==n
        logits.div_(.7)
        labels=inputs[j]['input_ids'][:,-n:]
        return selective_log_softmax(logits,labels)
    # Native preparation evaluates both current and reference before backward.
    raw_model=accelerator.unwrap_model(model)
    raw_model.gradient_checkpointing_disable()
    old=[];ref=[]
    with torch.no_grad():
        for j in range(128):
            old.append(logps(j))
            with raw_model.disable_adapter():ref.append(logps(j))
    raw_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    details=[]
    short=next(j for j,r in enumerate(records) if len(r['response_ids'])<256)
    # Exercise irregular allocation reuse; long joint pass retains both graphs.
    for j,auxiliary in [(short,False),(target,False),(short,True),(target,True),(short,False),(target,True)]:
        torch.cuda.reset_peak_memory_stats();model.zero_grad(set_to_none=True)
        current=logps(j);ratio=(current-old[j]).exp();delta=ref[j]-current
        loss=(-ratio*.5+.04*(delta.exp()-delta-1)).mean()
        if auxiliary:
            rec=records[j];closing=processor.tokenizer.encode('</think>',add_special_tokens=False)
            assert len(closing)==1
            # Own final box supplies only representative target shapes. No GT;
            # no teacher-quality or method-effect claim from this witness.
            from .eval import parse_bbox
            bbox=parse_bbox(rec['final_text']);assert bbox is not None
            target_ids=processor.tokenizer.encode(json.dumps({'bbox_2d':bbox},separators=(',',':')),add_special_tokens=False)+[processor.tokenizer.eos_token_id]
            target_inputs=extend_image_inputs(bases[j],rec['base_ids']+rec['thinking_ids']+closing+target_ids)
            ce=coordinate_ce(model,target_inputs,target_ids,coordinate_indices(processor.tokenizer,target_ids))
            loss=loss+.05*ce
        accelerator.backward(loss)
        assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in trainable)
        assert any(p.grad is not None and torch.count_nonzero(p.grad)>0 for p in trainable)
        stats=torch.cuda.memory_stats()
        torch.cuda.synchronize();free_bytes,total_bytes=torch.cuda.mem_get_info()
        details.append(dict(sample_id=records[j]['sample_id'],response_tokens=len(records[j]['response_ids']),auxiliary=auxiliary,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            inactive_split_bytes=stats.get('inactive_split_bytes.all.current',0),loss=float(loss.detach()),
            free_after_backward_bytes=free_bytes,device_total_bytes=total_bytes))
        print(json.dumps(details[-1]),flush=True)
        del loss,current,ratio,delta
        if auxiliary:del ce,target_inputs
    write_json(a.output,dict(state='passed',source=str(source),device=a.device,seconds=time.time()-started,
        original_prefix_matches=128,held_window_inputs=128,passes=details,GT_used=False,optimizer_updates=0,
        model_generation_calls=0,validation_read=False,allocator={k:os.environ[k] for k in ('PYTORCH_ALLOC_CONF','PYTORCH_CUDA_ALLOC_CONF')},
        scope='Actual saved8192token window; synthetic advantage0.5 and own-final coordinateCE test memory/gradient shapes only; no efficacy or exact native-loss-value claim'))


if __name__=='__main__':main()
