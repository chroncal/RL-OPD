"""Prospective GT-free three-arm input-view pilot with online token accounting."""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from .prepare import ROOT, write_json
from .eval import SCHEMA, parse_bbox
from .select import select_boxes
from .full_chain_control import save_once, split_natural
from .phase_views import (ARMS, CAP, ANSWER_MAX, render_view, view_index,
                         next_natural_limit, inverse_box, bbox_text, prompt_text)

INPUTS = ROOT/'results/phase_input_witness100_20261004/inputs.jsonl'

def config(mode):
    return {'model':json.loads((ROOT/'model_receipt.json').read_text()), 'mode':mode,
        'n':100 if mode=='pilot' else 1, 'arms':list(ARMS), 'seed':0,
        'batch_size':8, 'temperature':.7, 'top_p':1, 'top_k':-1,
        'token_cap_per_row_arm':CAP, 'natural_max_tokens':8192,
        'natural_minimum_request_cap':512, 'answer_max_tokens':ANSWER_MAX,
        'candidate_seed':'j for both source and readout', 'schema':SCHEMA,
        'max_model_len':12288, 'max_num_seqs':32, 'max_num_batched_tokens':16384,
        'memory_fraction':.60, 'prefix_cache':False, 'async_scheduling':False,
        'backend':'vllm0.18.1 BF16 FLASH_ATTN eager', 'heldout_read':False,
        'new_training':False, 'inputs':str(INPUTS) if mode=='pilot' else 'synthetic only',
        'selector':'frozen select.select_boxes; earliest tie; {} if no valid boxes'}

def make_synthetic(processor, directory):
    """Input/interface witness only; no data labels or accuracy calculation."""
    from PIL import Image, ImageDraw
    im=Image.new('RGB',(512,384),'white')
    ImageDraw.Draw(im).rectangle((160,110,330,270),fill='red')
    path=directory/'synthetic_red_rectangle.png';im.save(path)
    row={'sample_id':'synthetic_red_rectangle','image_path':str(path),
         'expression':'the red rectangle','width':512,'height':384}
    messages=[{'role':'user','content':[{'type':'image','image':str(path)},
        {'type':'text','text':prompt_text(row['expression'])}]}]
    prompt=processor.apply_chat_template(messages,add_generation_prompt=True,
        tokenize=False,enable_thinking=True)
    row['base_unexpanded_ids']=processor.tokenizer.encode(prompt,add_special_tokens=False)
    row['views']={}
    for key,arm,j in [('p0','phase',0),('p1','phase',1),('p2','phase',2),('original','original',0)]:
        view,meta=render_view(im,arm,j)
        processed=processor(text=[prompt],images=[view],return_tensors='pt',padding=False)
        g=processed['image_grid_thw'][0].tolist()
        row['views'][key]={**meta, 'expanded_base_ids':processed['input_ids'][0].tolist(),
            'image_grid_thw':g, 'visual_tokens':int(g[0]*g[1]*g[2]//4)}
        view.close()
    im.close()
    save_once(directory/'synthetic_input.json',row)
    return [row]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',type=int,choices=[0,1],required=True)
    parser.add_argument('--mode',choices=['pilot','synthetic'],required=True)
    args=parser.parse_args();output=args.output.resolve()
    assert not output.exists(),'preserve old/partial trials; no automatic restart'
    output.mkdir(parents=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.device),TOKENIZERS_PARALLELISM='false',
        VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_NO_USAGE_STATS='1',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    started=time.time();cfg=config(args.mode);save_once(output/'run_config.json',cfg)
    def status(state,**extra):
        write_json(output/'worker.json',{'state':state,'pid':os.getpid(),'command':sys.argv,
            'gpu':args.device,'elapsed_seconds':time.time()-started,
            'expected_samples':cfg['n'],'mode':args.mode,**extra})
    status('loading')
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    load_started=time.time()
    processor=AutoProcessor.from_pretrained(cfg['model']['path'],local_files_only=True)
    if args.mode=='pilot':
        rows=[json.loads(line) for line in INPUTS.read_text().splitlines()]
        allowed={'sample_id','image_path','expression','width','height','base_unexpanded_ids','views'}
        assert len(rows)==100 and all(set(row)==allowed for row in rows)
        assert len({row['sample_id'] for row in rows})==100
    else:
        rows=make_synthetic(processor,output)
    llm=LLM(model=cfg['model']['path'],tokenizer=cfg['model']['path'],dtype='bfloat16',
        tensor_parallel_size=1,max_model_len=12288,gpu_memory_utilization=.60,
        max_num_seqs=32,max_num_batched_tokens=16384,enforce_eager=True,
        enable_prefix_caching=False,enable_chunked_prefill=True,async_scheduling=False,
        attention_config={'backend':'FLASH_ATTN'},limit_mm_per_prompt={'image':1},
        seed=0,reasoning_parser='qwen3')
    tokenizer=llm.get_tokenizer();closing=tokenizer.encode('</think>',add_special_tokens=False)
    assert len(closing)==1
    structured=StructuredOutputsParams(json=SCHEMA)
    loading_seconds=time.time()-load_started;inference_started=time.time()
    events=[];render_seconds=0.;completed=0
    for begin in range(0,len(rows),8):
        chunk=rows[begin:begin+8];views={};states={}
        render_started=time.perf_counter()
        for row in chunk:
            sid=row['sample_id'];im=Image.open(row['image_path']).convert('RGB')
            assert list(im.size)==[row['width'],row['height']]
            for key,arm,v in [('p0','phase',0),('p1','phase',1),('p2','phase',2),('original','original',0)]:
                image,meta=render_view(im,arm,v);views[sid,key]=image
                expected=row['views'][key]
                assert all(meta[k]==expected[k] for k in ('original_wh','canvas_wh','offset_xy'))
            im.close()
            for arm in ARMS:
                states[sid,arm]={'sample_id':sid,'arm':arm,'candidates':[],
                    'total_accounted_tokens':0,'ground_truth_in_prompt':False}
        render_seconds+=time.perf_counter()-render_started
        def request(row,key,ids):
            assert len(ids)<12288
            return {'prompt_token_ids':ids,'multi_modal_data':{'image':views[row['sample_id'],key]},
                # Same actual pixels share a processor UUID; distinct views never collide.
                'multi_modal_uuids':{'image':f'{output.name}:{row["sample_id"]}:{key}'}}
        j=0
        while True:
            requests=[];parameters=[];mapping=[]
            for row in chunk:
                sid=row['sample_id']
                for arm in ARMS:
                    state=states[sid,arm]
                    if 'stop_remaining_tokens' in state:continue
                    key='original' if arm=='original' else f'p{view_index(arm,j)}'
                    base=row['views'][key]['expanded_base_ids'];b=len(base)
                    remaining=CAP-state['total_accounted_tokens']
                    limit=next_natural_limit(remaining,b,j)
                    if limit is None:
                        assert j>=3,'first3 prospective reserve violated'
                        state['stop_remaining_tokens']=remaining
                        state['next_request_limit']=None
                        continue
                    assert b+2*limit+1+ANSWER_MAX<=12288
                    requests.append(request(row,key,row['base_unexpanded_ids']))
                    parameters.append(SamplingParams(temperature=.7,top_p=1,top_k=-1,
                        seed=j,max_tokens=limit,skip_special_tokens=False,structured_outputs=structured))
                    mapping.append((row,arm,key,limit,remaining))
            if not requests:break
            batch_started=time.perf_counter()
            generated=llm.generate(requests,parameters,use_tqdm=False)
            assert len(generated)==len(mapping)
            events.append({'batch_begin':begin,'candidate_index':j,'kind':'natural',
                'requests':len(requests),'elapsed_seconds':time.perf_counter()-batch_started})
            naturals=[]
            for out,(row,arm,key,limit,remaining) in zip(generated,mapping):
                sid=row['sample_id'];base=row['views'][key]['expanded_base_ids']
                assert list(out.prompt_token_ids)==base,'actual vLLM expansion != frozen CPU inputs'
                first=out.outputs[0];ids=list(first.token_ids)
                assert len(ids)<=limit
                think,answer,closed=split_natural(ids,closing[0])
                nat={'sample_id':sid,'arm':arm,'candidate_index':j,'seed':j,'view_key':key,
                    'natural_limit':limit,'remaining_before_pair':remaining,
                    'natural_full_token_ids':ids,'reasoning_token_ids':think,
                    'natural_answer_token_ids_including_eos':answer,'reasoning_closed':closed,
                    'length_capped':first.finish_reason=='length','finish_reason':first.finish_reason,
                    'base_unexpanded_token_ids':row['base_unexpanded_ids'],
                    'expanded_base_token_ids':base,'ground_truth_in_prompt':False,
                    'natural_text':tokenizer.decode(ids,skip_special_tokens=False)}
                save_once(output/'natural'/arm/sid/f'{j}.json',nat)
                naturals.append(nat)
            requests=[];parameters=[]
            for (row,arm,key,limit,remaining),nat in zip(mapping,naturals):
                requests.append(request(row,key,row['base_unexpanded_ids']+nat['reasoning_token_ids']+closing))
                parameters.append(SamplingParams(temperature=.7,top_p=1,top_k=-1,seed=j,
                    max_tokens=ANSWER_MAX,skip_special_tokens=False,structured_outputs=structured))
            batch_started=time.perf_counter()
            generated=llm.generate(requests,parameters,use_tqdm=False)
            assert len(generated)==len(mapping)
            events.append({'batch_begin':begin,'candidate_index':j,'kind':'answer',
                'requests':len(requests),'elapsed_seconds':time.perf_counter()-batch_started})
            for out,(row,arm,key,limit,remaining),nat in zip(generated,mapping,naturals):
                sid=row['sample_id'];base=nat['expanded_base_token_ids']
                assert list(out.prompt_token_ids)==base+nat['reasoning_token_ids']+closing
                first=out.outputs[0];full=list(first.token_ids);assert len(full)<=ANSWER_MAX
                ids=full[:-1] if full and full[-1]==tokenizer.eos_token_id else full
                text=tokenizer.decode(ids,skip_special_tokens=False).strip()
                box=inverse_box(text,row['views'][key])
                ans={'sample_id':sid,'arm':arm,'candidate_index':j,'seed':j,'text':text,
                    'bbox_2d':parse_bbox(text),'mapped_bbox_2d':box,'mapped_text':bbox_text(box),
                    'token_ids':ids,'generated_token_ids_including_eos':full,
                    'finish_reason':first.finish_reason,'prefill_tokens':len(out.prompt_token_ids),
                    'expanded_prompt_token_ids':list(out.prompt_token_ids),'ground_truth_in_prompt':False}
                cost=len(base)+len(nat['natural_full_token_ids'])+ans['prefill_tokens']+len(full)
                assert cost<=2*len(base)+2*limit+1+ANSWER_MAX and cost<=remaining
                save_once(output/'answers'/arm/sid/f'{j}.json',ans)
                candidate={'candidate_index':j,'view_key':key,'view':row['views'][key],
                    'natural':nat,'answer':ans,'accounted_tokens':cost,
                    'affected':bool(nat['length_capped'] or not nat['reasoning_closed'] or
                                    ans['finish_reason']=='length' or box is None)}
                save_once(output/'pairs'/arm/sid/f'{j}.json',candidate)
                state=states[sid,arm];assert len(state['candidates'])==j
                state['candidates'].append(candidate);state['total_accounted_tokens']+=cost
                assert state['total_accounted_tokens']<=CAP
            j+=1
            status('running',completed_samples=completed,loading_seconds=loading_seconds,
                current_batch_begin=begin,current_candidate_index=j,
                completed_pairs=sum(e['requests'] for e in events if e['kind']=='answer'))
        for row in chunk:
            sid=row['sample_id']
            for arm in ARMS:
                state=states[sid,arm];assert len(state['candidates'])>=3
                texts=[c['answer']['mapped_text'] for c in state['candidates']]
                choice,support=select_boxes(texts)
                state['selection']={'sample_id':sid,'candidate_index':choice,'support':support,
                    'text':texts[choice] if choice is not None else '{}','fallback':choice is None,
                    'labels_used_for_selection':False}
                save_once(output/'raw'/arm/f'{sid}.json',state)
        for im in views.values():im.close()
        completed+=len(chunk)
        save_once(output/'batch_events'/f'{begin}.json',[e for e in events if e['batch_begin']==begin])
        status('running',completed_samples=completed,loading_seconds=loading_seconds)
        print('PHASE_BATCH',completed,len(rows),'seconds',round(time.time()-started,1),flush=True)
    save_once(output/'generation_events.json',events)
    status('complete',completed_samples=completed,loading_seconds=loading_seconds,
        inference_seconds=time.time()-inference_started,render_seconds=render_seconds,
        generation_batch_seconds=sum(e['elapsed_seconds'] for e in events),
        gpu_hours=(time.time()-started)/3600,completed_pairs=sum(e['requests'] for e in events if e['kind']=='answer'))
    if args.mode=='synthetic':
        save_once(output/'synthetic_receipt.json',{'state':'PASS_ACTUAL_VLLM_INPUT_BUDGET_SELECTION',
            'n':1,'arms':list(ARMS),'gt_scored':False,'heldout_read':False,
            'limits':'Interface witness only; no accuracy or semantic invariance evidence'})

if __name__=='__main__':main()
