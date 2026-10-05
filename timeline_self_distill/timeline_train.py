"""Matched answer-only LM LoRA adaptation; no inference labels or sealed data."""
import argparse
import gc
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from .prepare import ROOT, write_json
from .full_chain_control import save_once

MODEL='/root/rl_opd_v1_runtime/models/Qwen3.5-2B'
PAIRS=ROOT/'results/timeline_pair_availability500_20261004/training_pairs.jsonl'
TARGETS=ROOT/'refine-logs/timeline_reformulation_20261004/LORA_META_TARGETS.json'
PROMPT='\nReturn the bounding box of the referred object as {"bbox_2d": [x1, y1, x2, y2]}. Use four integers from 0 to 1000 in relative image coordinates. Output only this JSON in the final answer.'


def exposure_order(rows,count):
    rng=random.Random(0);order=[]
    while len(order)<count:
        part=list(range(len(rows)));rng.shuffle(part);order.extend(part)
    return order[:count]


def encode(processor,row,device):
    # Explicit input whitelist; GT filter metadata and target are never prompt fields.
    messages=[{'role':'user','content':[{'type':'image','image':row['image_path']},
        {'type':'text','text':row['expression']+PROMPT}]}]
    base=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
        enable_thinking=True,return_dict=True,return_tensors='pt')
    assert base['input_ids'][0].tolist()==row['expanded_base_token_ids'], 'Original multimodal base IDs changed'
    return base.to(device)


def answer_inputs(base,row,arm,target):
    from .teacher_signal import extend_image_inputs
    prefix=row['expanded_base_token_ids']+(row['reasoning_token_ids'] if arm=='late' else [])+row['closing_token_ids']
    ids=prefix+target
    assert len(ids)<=12288 and target[-1]==248046
    return extend_image_inputs(base,ids)


def logps(model,base,row,arm,target):
    import torch
    from .timeline_objectives import answer_logps
    inputs=answer_inputs(base,row,arm,target)
    # All targets present: keep L+1 positions and discard final unused prediction.
    # First kept position is prefix[-1], which predicts target[0].
    logits=model(**inputs,use_cache=False,logits_to_keep=len(target)+1).logits[0,:-1]
    assert logits.shape[0]==len(target)
    return answer_logps(logits,torch.tensor(target,device=logits.device))


def diagnostic(model,processor,row,arm):
    import torch
    model.eval();base=encode(processor,row,model.device)
    with torch.no_grad():
        values={k:logps(model,base,row,arm,row[k]['target_ids_including_eos']).cpu().tolist()
            for k in ('positive','negative')}
    return values


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['late','zero'],required=True)
    p.add_argument('--mode',choices=['witness','pilot'],required=True)
    p.add_argument('--device',type=int,choices=[0,1],required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.output.resolve();root.mkdir(parents=True,exist_ok=False)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.device),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    import torch
    from transformers import AutoModelForImageTextToText,AutoProcessor
    from peft import LoraConfig,get_peft_model
    torch.set_num_threads(4);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    started=time.time();steps=2 if a.mode=='witness' else 32;accum=8
    rows=[json.loads(s) for s in PAIRS.read_text().splitlines()];assert len(rows)==70
    names=[m['name'] for m in json.loads(TARGETS.read_text())['modules']];assert len(names)==186
    order=exposure_order(rows,steps*accum)
    local=Path('/root/timeline_lora_checkpoints')/root.parent.name/root.name
    local.mkdir(parents=True,exist_ok=False)
    save_once(root/'run_config.json',{'command':sys.argv,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'model':MODEL,'pair_source':str(PAIRS),'arm':a.arm,'mode':a.mode,'seed':0,'updates':steps,'microbatch':1,'accumulation':accum,
        'exposures':[rows[i]['sample_id'] for i in order],'target_modules':names,'r':8,'alpha':16,'dropout':0,
        'lr':1e-4,'betas':[.9,.999],'eps':1e-8,'weight_decay':0,'clip':1,'local_export':str(local),
        'answer_only':True,'gt_in_prompt':False,'heldout_read':False,'validation_checkpoint_selection':False})
    processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
    assert processor.tokenizer.eos_token_id==248046
    model=AutoModelForImageTextToText.from_pretrained(MODEL,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda()
    actual=[n for n,m in model.named_modules() if '.language_model.layers.' in n and isinstance(m,torch.nn.Linear)]
    assert actual==names
    model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=0,target_modules=names,bias='none'))
    trainable=[(n,v) for n,v in model.named_parameters() if v.requires_grad]
    assert sum(v.numel() for _,v in trainable)==8409600
    assert all('lora_' in n and '.language_model.layers.' in n for n,_ in trainable)
    # LoRA tensors explicitly FP32, BF16 base unchanged; optimizer naturally FP32.
    for _,v in trainable:v.data=v.data.float()
    save_once(root/'parameter_receipt.json',{'trainable_count':sum(v.numel() for _,v in trainable),
        'trainable':[{'name':n,'shape':list(v.shape),'dtype':str(v.dtype)} for n,v in trainable],
        'frozen_base_parameter_count':sum(v.numel() for v in model.parameters() if not v.requires_grad)})
    before=diagnostic(model,processor,rows[0],a.arm)
    initial=[v.detach().cpu().clone() for _,v in trainable]
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.config.use_cache=False
    optimizer=torch.optim.AdamW([v for _,v in trainable],lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0)
    records=[];optimizer.zero_grad(set_to_none=True)
    for step in range(steps):
        model.train();window=[];step_started=time.time()
        for index in order[step*accum:(step+1)*accum]:
            row=rows[index];base=encode(processor,row,model.device)
            target=row['positive']['target_ids_including_eos']
            lp=logps(model,base,row,a.arm,target);loss=-lp.mean()
            assert torch.isfinite(loss), 'nonfinite CE'
            (loss/accum).backward();window.append({'sample_id':row['sample_id'],'loss':float(loss.detach()),'target_tokens':len(target)})
            del base,lp,loss
        norm=torch.nn.utils.clip_grad_norm_([v for _,v in trainable],1,error_if_nonfinite=True)
        assert float(norm)>0, 'No adapter gradient'
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        assert all(torch.isfinite(v).all() for _,v in trainable), 'nonfinite adapter'
        rec={'step':step+1,'examples':window,'gradient_norm_before_clip':float(norm),
            'elapsed_seconds':time.time()-step_started,'peak_cuda_bytes':torch.cuda.max_memory_allocated()}
        records.append(rec);save_once(root/f'update_{step+1:03d}.json',rec)
        print('LORA_UPDATE',step+1,'loss',sum(x['loss'] for x in window)/accum,'seconds',rec['elapsed_seconds'],flush=True)
    delta=sum(float((v.detach().cpu()-b).square().sum()) for (_,v),b in zip(trainable,initial))**.5
    assert delta>0
    state_dtypes=sorted({str(x.dtype) for s in optimizer.state.values() for x in s.values() if torch.is_tensor(x)})
    after=diagnostic(model,processor,rows[0],a.arm)
    model.save_pretrained(local/'adapter',safe_serialization=True)
    model=model.merge_and_unload(safe_merge=True)
    merged=diagnostic(model,processor,rows[0],a.arm)
    model.save_pretrained(local/'merged',safe_serialization=True,max_shard_size='2GB')
    # Evaluation uses official tokenizer/processor. Save their original metadata verbatim.
    for name in ('preprocessor_config.json','processor_config.json','video_preprocessor_config.json','chat_template.jinja',
                 'tokenizer.json','tokenizer_config.json','merges.txt','vocab.json'):
        source=Path(MODEL)/name
        if source.exists():shutil.copyfile(source,local/'merged'/name)
    shutil.copytree(local/'adapter',root/'adapter',copy_function=shutil.copyfile)
    shutil.copytree(local/'merged',root/'checkpoint',copy_function=shutil.copyfile)
    reloaded=None
    if a.mode=='witness':
        del model,optimizer,trainable;gc.collect();torch.cuda.empty_cache()
        model=AutoModelForImageTextToText.from_pretrained(root/'checkpoint',dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda()
        reloaded=diagnostic(model,processor,rows[0],a.arm)
        drift=max(abs(x-y) for k in merged for x,y in zip(merged[k],reloaded[k]))
        change=max(abs(x-y) for k in before for x,y in zip(before[k],reloaded[k]))
        assert drift<.05, 'Saved merged weight readout mismatch'
        assert change>1e-6, 'No detectable saved parameter effect'
    save_once(root/'completion.json',{'state':'complete','arm':a.arm,'mode':a.mode,'updates':steps,'exposures':steps*accum,
        'sample_id_diagnostic':rows[0]['sample_id'],'adapter_delta_l2':delta,'optimizer_state_dtypes':state_dtypes,
        'before_logps':before,'after_adapter_logps':after,'after_merge_logps':merged,'reloaded_logps':reloaded,
        'gpu_hours':(time.time()-started)/3600,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),
        'heldout_read':False,'scope':'Training/export receipt and train-pair logps, not deployment accuracy.'})
    print('LORA_COMPLETE',str(root),flush=True)

if __name__=='__main__':main()
