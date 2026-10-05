"""Current-rollout consensus CE with matched native Swift GRPO controls."""
import copy
import json
import os
import time
from pathlib import Path
import torch
from swift.rewards import ORM, orms
from swift.template import Template
from swift.infer_engine import RequestConfig
from swift.rlhf_trainers.grpo_trainer import GRPOTrainer
from swift.utils import to_device
from transformers import TrainerCallback
from timeline_self_distill.eval import evaluate, parse_bbox
from timeline_self_distill.consensus_objectives import consensus_target, coordinate_ce
from timeline_self_distill.teacher_signal import coordinate_indices, extend_image_inputs
from timeline_self_distill.prepare import write_json

INTEGER = r'(0|[1-9][0-9]{0,2}|1000)'
BBOX_REGEX = r'\s*\{\s*"bbox_2d"\s*:\s*\[\s*'+r'\s*,\s*'.join([INTEGER]*4)+r'\s*\]\s*\}\s*'


def arm():
    return os.environ['CONSENSUS_ARM']


def raw_ids(value):
    if value and isinstance(value[0], list):
        assert len(value)==1
        value=value[0]
    assert all(type(v)==int for v in value)
    return list(value)


def answer_text(text):
    if '</think>' not in text:
        return ''
    text=text.split('</think>',1)[1].strip()
    for token in ('<|im_end|>','<|endoftext|>'):
        if text.endswith(token): text=text[:-len(token)].strip()
    return text


class IoUReward(ORM):
    def __call__(self,completions,solution,response_token_ids,**kwargs):
        assert arm() in ('grpo','joint')
        if not hasattr(self,'tokenizer'):
            from transformers import AutoTokenizer
            self.tokenizer=AutoTokenizer.from_pretrained('/root/rl_opd_v1_runtime/models/Qwen3.5-2B',local_files_only=True)
        values=[evaluate(answer_text(self.tokenizer.decode(raw_ids(ids),skip_special_tokens=False)),json.loads(gt))['iou']
                for ids,gt in zip(response_token_ids,solution)]
        if os.environ['CONSENSUS_REWARD']=='continuous':return values
        assert os.environ['CONSENSUS_REWARD']=='three_tier'
        return [0. if v<.5 else .5 if v<.75 else 1. for v in values]


class UnlabeledZero(ORM):
    def __call__(self,completions,**kwargs):
        assert arm()=='sd_only' and 'solution' not in kwargs
        return [0.]*len(completions)


orms['consensus_iou']=IoUReward
orms['consensus_unlabeled']=UnlabeledZero
_native_decode=Template.decode_generate_ids


def decode_prefix(self,generate_ids,*,template_inputs=None,**kwargs):
    prefix=self._get_response_prefix(template_inputs)
    if isinstance(prefix,list):
        template_inputs=copy.copy(template_inputs)
        template_inputs.chat_template_kwargs=dict(template_inputs.chat_template_kwargs)
        template_inputs.chat_template_kwargs['response_prefix']=self.tokenizer.decode(prefix,skip_special_tokens=False)
    return _native_decode(self,generate_ids,template_inputs=template_inputs,**kwargs)


Template.decode_generate_ids=decode_prefix
_native_generate=GRPOTrainer._generate_completions
_native_score=GRPOTrainer._score_completions
_native_prepare=GRPOTrainer._prepare_batch_inputs
_native_loss=GRPOTrainer._compute_loss_and_metrics


def run_record(self,name,value):
    path=Path(os.environ['CONSENSUS_RECORD_DIR'])/f'{name}_step_{self.state.global_step:04d}.json'
    assert not path.exists()
    write_json(path,value)


def original_base(self,sample):
    prompt=sample.to_template_dict()
    prompt['messages']=[m for m in prompt['messages'] if m['role']!='assistant']
    for key in ('response_token_ids','add_eos','chat_template_kwargs'):
        prompt.pop(key,None)
    with self._template_context(self.template,mode='transformers'):
        base=self.template.encode(copy.deepcopy(prompt))
    base.pop('labels',None);base.pop('loss_scale',None);base.pop('_extra_kwargs',None)
    return prompt,base


def generate(self,samples):
    samples=_native_generate(self,samples)
    assert len(samples)==128 and self.accelerator.num_processes==1
    assert self.rollout_importance_sampling_mode is None
    tok=self.template.tokenizer
    opening=tok.encode('<think>\n',add_special_tokens=False)
    closing=tok.encode('</think>',add_special_tokens=False)
    assert len(closing)==1
    requests=[];mapping=[]
    for sample in samples:
        ids=raw_ids(sample.response_token_ids)
        sample._consensus_natural_ids=ids
        sample._consensus_cap=sample.finish_reason=='length'
        sample._consensus_closed=closing[0] in ids
        think=ids[:ids.index(closing[0])] if closing[0] in ids else ids
        sample._consensus_think=think
        prompt,base=original_base(self,sample)
        assert base['input_ids'][-len(opening):]==opening
        sample._consensus_base=base
        sample._consensus_prompt=prompt
        req=sample.to_infer_request()
        req.messages=copy.deepcopy(prompt['messages'])
        req.chat_template_kwargs={'response_prefix':opening+think+closing}
        requests.append(req);mapping.append(sample)
    # Common exploratory final action, sampled at0.7 for native GRPO; all arms
    # share this schema replay. Evaluation uses the frozen greedy100% readout.
    cfg=RequestConfig(temperature=.7,max_tokens=128,return_details=True,structured_outputs_regex=BBOX_REGEX,
                      seed=0,top_p=1,top_k=-1)
    outputs=self._engine_infer(requests,cfg,use_tqdm=False)
    assert len(outputs)==len(samples)
    for out,sample in zip(outputs,mapping):
        choice=out.response.choices[0]
        prefix=sample._consensus_think+closing
        assert out.response.prompt_token_ids==sample._consensus_base['input_ids']+prefix
        sample.response_token_ids=prefix+list(choice.token_ids)
        sample.messages[-1]['content']=tok.decode(sample.response_token_ids,skip_special_tokens=False)
        sample.finish_reason='length' if sample._consensus_cap else choice.finish_reason
        sample.add_eos=False;sample.rollout_logprobs=[];sample.response_loss_mask=[];sample.encoded=None
        sample._consensus_final_ids=list(choice.token_ids)
    run_record(self,'natural',dict(arm=arm(),samples=[dict(sample_id=s.extra['sample_id'],
        natural_ids=s._consensus_natural_ids,natural_cap=s._consensus_cap,natural_closed=s._consensus_closed,
        sampled_final_ids=s._consensus_final_ids,full_schema_response_ids=raw_ids(s.response_token_ids),
        expanded_base_ids=s._consensus_base['input_ids']) for s in samples],GT_in_prompt=False))
    return samples


def score(self,samples):
    samples=_native_score(self,samples)
    assert self.num_generations==8 and self.args.generation_batch_size==128
    assert self.args.steps_per_generation==self.args.gradient_accumulation_steps==128
    assert not self.dynamic_sample and not self.dynamic_num_samples and self.num_iterations==1
    for start in range(0,len(samples),8):
        assert len({s.prompt_id for s in samples[start:start+8]})==1
    tok=self.template.tokenizer
    closing=tok.encode('</think>',add_special_tokens=False)
    opening=tok.encode('<think>\n',add_special_tokens=False)
    for sample in samples:
        text=answer_text(tok.decode(raw_ids(sample.response_token_ids),skip_special_tokens=False))
        final_ids=list(sample._consensus_final_ids)
        if final_ids and final_ids[-1]==tok.eos_token_id:final_ids=final_ids[:-1]
        rec={'sample_id':sample.extra['sample_id'],'request_id':sample.request_id,'prompt_id':sample.prompt_id,
             'thinking_ids':sample._consensus_think,'base_ids':sample._consensus_base['input_ids'],
             'response_ids':raw_ids(sample.response_token_ids),'natural_cap':sample._consensus_cap,
             'natural_closed':sample._consensus_closed,'candidates':[], 'active':False,'reason':'grpo_control',
             'final_text':text,'final_ids':final_ids,'GT_used':arm()!='sd_only'}
        rec['natural_generated_tokens']=len(sample._consensus_natural_ids)
        rec['schema_final_generated_tokens']=len(sample._consensus_final_ids)
        if arm()!='sd_only':
            rec['final_evaluation']=evaluate(text,json.loads(sample.extra['solution']))
        else:
            assert 'solution' not in sample.extra
        sample._consensus=rec
    if arm()=='grpo':return samples
    start=time.monotonic();requests=[];mapping=[]
    for sample in samples:
        for pct in range(10,100,10):
            k=len(sample._consensus_think)*pct//100
            req=sample.to_infer_request()
            req.messages=copy.deepcopy(sample._consensus_prompt['messages'])
            req.chat_template_kwargs={'response_prefix':opening+sample._consensus_think[:k]+closing}
            requests.append(req);mapping.append((sample,pct,k))
    cfg=RequestConfig(temperature=0,max_tokens=128,return_details=True,structured_outputs_regex=BBOX_REGEX,
                      seed=0,top_p=1,top_k=-1)
    for begin in range(0,len(requests),128):
        outputs=self._engine_infer(requests[begin:begin+128],cfg,use_tqdm=False)
        assert len(outputs)==len(mapping[begin:begin+128])
        for out,(sample,pct,k) in zip(outputs,mapping[begin:begin+128]):
            choice=out.response.choices[0];ids=list(choice.token_ids)
            generated_count=len(ids)
            if ids and ids[-1]==tok.eos_token_id:ids=ids[:-1]
            assert out.response.prompt_token_ids==sample._consensus_base['input_ids']+sample._consensus_think[:k]+closing
            sample._consensus['candidates'].append(dict(pct=pct,cut=k,ids=ids,
                text=tok.decode(ids,skip_special_tokens=False).strip(),finish_reason=choice.finish_reason,
                generated_tokens=generated_count,original_prefix_match=True))
    for sample in samples:
        rec=sample._consensus
        rec['candidates'].append(dict(pct=100,cut=len(sample._consensus_think),ids=rec['final_ids'],text=rec['final_text']))
        target=consensus_target(rec['candidates'],os.environ['CONSENSUS_GATE'])
        rec.update(target);rec['reason']='active' if rec['active'] else 'GT_free_gate_closed'
        if rec['active']:
            # Canonical teacher-forced answer under the student's own full chain.
            text=json.dumps({'bbox_2d':rec['teacher_bbox']},separators=(',',':'))
            ids=tok.encode(text,add_special_tokens=False)+[tok.eos_token_id]
            assert parse_bbox(tok.decode(ids[:-1],skip_special_tokens=False))==rec['teacher_bbox']
            mask=coordinate_indices(tok,ids);assert mask
            rec.update(target_ids=ids,coordinate_indices=mask)
    self._consensus_teacher_seconds=time.monotonic()-start
    return samples


def prepare(self,samples):
    batches=_native_prepare(self,samples)
    chunks=self.split_by_mini_batches(samples)
    assert len(batches)==len(chunks)==128
    records=[]
    closing=self.template.tokenizer.encode('</think>',add_special_tokens=False)
    for inp,chunk in zip(batches,chunks):
        assert len(chunk)==1
        sample=chunk[0];rec=sample._consensus
        if rec['active']:
            prefix=rec['base_ids']+rec['thinking_ids']+closing
            student=inp['model_inputs']['input_ids'][0].tolist()
            assert student[:len(prefix)]==prefix
            with self._template_context(self.template,mode='transformers'):
                base=self.template.data_collator([sample._consensus_base])
            base.pop('labels',None);base.pop('loss_scale',None)
            base=to_device(base,self.accelerator.device)
            target_inputs=extend_image_inputs(base,prefix+rec['target_ids'])
            assert len(prefix)+len(rec['target_ids'])<=12288
            inp['grpo_batch']._consensus_aux=dict(inputs=target_inputs,target_ids=rec['target_ids'],
                indices=rec['coordinate_indices'],request_id=sample.request_id)
        records.append(rec)
        # Do not retain duplicated image tensors after collating the target.
        del sample._consensus_base;del sample._consensus_prompt
    active=sum(r['active'] for r in records)
    run_record(self,'rollout',dict(step=self.state.global_step,arm=arm(),samples=records,active=active,
        teacher_generations=sum(len(r['candidates'])-1 for r in records) if arm()!='grpo' else 0,
        teacher_seconds=getattr(self,'_consensus_teacher_seconds',0),GT_used=arm()!='sd_only',
        full_batch_denominator=128))
    self._metrics['train']['consensus/active_fraction'].append(active/128)
    return batches


def loss(self,model,model_inputs,grpo_batch):
    native,metrics=_native_loss(self,model,model_inputs,grpo_batch)
    if arm()=='sd_only':
        assert self.beta==0 and torch.count_nonzero(grpo_batch.advantages)==0
        assert abs(float(native.detach()))<1e-10
        native=native*0
    aux=getattr(grpo_batch,'_consensus_aux',None)
    if aux is not None:
        # Same current full-chain conditioning, loss only on target coordinates.
        with self._template_context(self.template,mode='transformers'):
            ce=coordinate_ce(model,aux['inputs'],aux['target_ids'],aux['indices'])
        coefficient=float(os.environ['CONSENSUS_LAMBDA'])
        extra=coefficient*ce
        self._metrics[metrics['mode']]['consensus/CE_active'].append(float(ce.detach()))
        self._metrics[metrics['mode']]['consensus/weighted_CE_active'].append(float(extra.detach()))
        native=native+extra
    return native,metrics


GRPOTrainer._generate_completions=generate
GRPOTrainer._score_completions=score
GRPOTrainer._prepare_batch_inputs=prepare
GRPOTrainer._compute_loss_and_metrics=loss


class ParameterWitness(TrainerCallback):
    def __init__(self,trainer):self.trainer=trainer;self.previous=None
    def values(self):
        params=[(n,p) for n,p in self.trainer.model.named_parameters() if p.requires_grad]
        assert params and all('lora_' in n and 'language_model.layers.' in n for n,p in params)
        assert all(torch.isfinite(p).all() for n,p in params)
        values=torch.cat([p.detach().reshape(-1)[::max(1,p.numel()//64)].float().cpu() for n,p in params])
        return params,values
    def on_train_begin(self,args,state,control,**kwargs):
        params,self.previous=self.values()
        run_record(self.trainer,'parameter_contract',dict(trainable_count=sum(p.numel() for n,p in params),
            trainable=[dict(name=n,shape=list(p.shape),dtype=str(p.dtype)) for n,p in params],visual_frozen=True,arm=arm()))
    def on_step_end(self,args,state,control,**kwargs):
        params,current=self.values();delta=current-self.previous
        path=Path(os.environ['CONSENSUS_RECORD_DIR'])/f'update_{state.global_step:04d}.json'
        assert not path.exists()
        write_json(path,dict(step=state.global_step,arm=arm(),parameter_delta_max=float(delta.abs().max()),
            parameter_delta_l2=float(delta.norm()),finite=True,cuda_peak_bytes=torch.cuda.max_memory_allocated()))
        self.previous=current


_native_init=GRPOTrainer.__init__
def init(self,*args,**kwargs):
    _native_init(self,*args,**kwargs)
    self.add_callback(ParameterWitness(self))
GRPOTrainer.__init__=init
