"""E3 fixed final-only evaluation; derived from frozen E0 inference, no GT input."""
import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path
from rl_opd_v1.common import read_json

from .eval import SCHEMA, parse_bbox
from .prepare import ROOT, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=500)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-thinking-tokens', type=int, default=8192)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--memory-fraction', type=float, default=.60)
    parser.add_argument('--split', choices=['val', 'train'], default='val',
                        help='Train is a separately registered diagnostic cohort, never heldout.')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--saved-prefix-source', type=Path)
    args = parser.parse_args()
    assert args.split == 'val' and args.limit == 500, 'E3 evaluation cohort is fixed val500'
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.device), TOKENIZERS_PARALLELISM='false',
        VLLM_WORKER_MULTIPROC_METHOD='spawn', VLLM_NO_USAGE_STATS='1',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    from PIL import Image
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    started = time.time()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_path = output/f'worker_{args.shard}.json'
    def status(state, **kwargs):
        write_json(status_path, {'state': state, 'pid': os.getpid(), 'gpu': args.device,
            'command': sys.argv, 'elapsed_seconds': time.time()-started, **kwargs})
    status('loading')
    original_receipt = json.loads((ROOT/'model_receipt.json').read_text())
    receipt = dict(original_receipt)
    receipt.update(path=str(args.checkpoint.resolve()), tokenizer_path=original_receipt['path'], original_base=original_receipt,
        saved_prefix_source=str(args.saved_prefix_source.resolve()) if args.saved_prefix_source else None)
    config_path = output/'run_config.json'
    run_config = {'model': receipt, 'seed': args.seed, 'thinking_temperature': .7,
        'natural_answer_temperature': .7, 'cut_answer_temperature': 0,
        'max_generation_tokens': args.max_thinking_tokens, 'cut_answer_max_tokens': 128,
        'schema': SCHEMA, 'fractions': [100], 'limit': args.limit,
        'batch_size': args.batch_size, 'shards': args.shards,
        'backend': 'vllm0.18.1 BF16, FLASH_ATTN, eager, prefix_cache disabled'}
    if args.split == 'train':
        run_config['diagnostic_cohort'] = 'fixed first limit rows of train2000.jsonl'
    if config_path.exists():
        if read_json(config_path) != run_config:
            raise RuntimeError('Resume configuration mismatch; use a new output directory')
    else:
        write_json(config_path, run_config)
    # Heldout is never an inference input here. Discard labels before constructing prompts.
    cohort = 'val500.jsonl' if args.split == 'val' else 'train2000.jsonl'
    tasks = [{k: r[k] for k in ('sample_id', 'image_path', 'expression')}
             for r in [json.loads(line) for line in (ROOT/'data'/cohort).read_text().splitlines()]]
    tasks = [r for i, r in enumerate(tasks[:args.limit]) if i % args.shards == args.shard]
    tasks = [r for r in tasks if not (output/'raw'/f'{r["sample_id"]}.json').exists()]
    # TF5 checkpoint serialization changes tokenizer preprocessing/metadata.
    # Use the original E0 tokenizer while loading the trained native weights.
    llm = LLM(model=receipt['path'], tokenizer=original_receipt['path'], dtype='bfloat16', tensor_parallel_size=1,
        max_model_len=args.max_thinking_tokens+4096, gpu_memory_utilization=args.memory_fraction,
        max_num_seqs=args.batch_size, max_num_batched_tokens=16384, enforce_eager=True,
        enable_prefix_caching=False, enable_chunked_prefill=True, async_scheduling=False,
        attention_config={'backend': 'FLASH_ATTN'}, limit_mm_per_prompt={'image': 1},
        seed=args.seed, reasoning_parser='qwen3')
    tokenizer = llm.get_tokenizer()
    # Use the model's official template; the image placeholders are expanded by vLLM.
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(original_receipt['path'])
    closing = tokenizer.convert_tokens_to_ids('</think>')
    forced_suffix = tokenizer.encode('</think>', add_special_tokens=False)
    assert forced_suffix == [closing]
    constrained = StructuredOutputsParams(json=SCHEMA)
    bbox_params = SamplingParams(temperature=0, max_tokens=128, skip_special_tokens=False,
        structured_outputs=constrained)
    def decoded(out):
        o = out.outputs[0]
        ids = list(o.token_ids)
        if ids and ids[-1] == tokenizer.eos_token_id:
            ids = ids[:-1]
        text = tokenizer.decode(ids, skip_special_tokens=False).strip()
        return {'text': text, 'bbox_2d': parse_bbox(text), 'token_ids': ids,
                'finish_reason': o.finish_reason}
    for start in range(0, len(tasks), args.batch_size):
        chunk = tasks[start:start+args.batch_size]
        images = {r['sample_id']: Image.open(r['image_path']).convert('RGB') for r in chunk}
        def request(row, ids):
            return {'prompt_token_ids': ids, 'multi_modal_data': {'image': images[row['sample_id']]},
                'multi_modal_uuids': {'image': row['image_path']}}
        base = {}
        for row in chunk:
            text = (row['expression']+'\nReturn the bounding box of the referred object as '
                '{"bbox_2d": [x1, y1, x2, y2]}. Use four integers from 0 to 1000 '
                'in relative image coordinates. Output only this JSON in the final answer.')
            messages = [{'role': 'user', 'content': [{'type': 'image', 'image': row['image_path']},
                                                     {'type': 'text', 'text': text}]}]
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True,
                                                   tokenize=False, enable_thinking=True)
            base[row['sample_id']] = tokenizer.encode(prompt, add_special_tokens=False)
        sampled = [] if args.saved_prefix_source else llm.generate([request(r, base[r['sample_id']]) for r in chunk],
            [SamplingParams(temperature=.7, top_p=1, top_k=-1, seed=args.seed,
                max_tokens=args.max_thinking_tokens,
                skip_special_tokens=False, structured_outputs=constrained) for _ in chunk], use_tqdm=False)
        records, reqs, maps = {}, [], []
        if args.saved_prefix_source:
            for row in chunk:
                rec = read_json(args.saved_prefix_source/'raw'/f'{row["sample_id"]}.json')
                assert rec['base_unexpanded_token_ids'] == base[row['sample_id']]
                rec = {k:v for k,v in rec.items() if k not in ('cuts','natural')}
                rec.update(cuts=[], natural=None, frozen_original_full_prefix=True)
                records[row['sample_id']] = rec
                reqs.append(request(row, base[row['sample_id']]+rec['reasoning_token_ids']+forced_suffix))
                maps.append((row['sample_id'],100,len(rec['reasoning_token_ids'])))
        for row, out in zip(chunk, sampled):
            first = out.outputs[0]
            ids = list(first.token_ids)
            closed = closing in ids
            think = ids[:ids.index(closing)] if closing in ids else ids
            natural_ids = ids[ids.index(closing)+1:] if closed else []
            if natural_ids and natural_ids[-1] == tokenizer.eos_token_id:
                natural_ids = natural_ids[:-1]
            natural_text = tokenizer.decode(natural_ids, skip_special_tokens=False).strip()
            rec = {'sample_id': row['sample_id'], 'reasoning_token_ids': think,
                'reasoning_text': tokenizer.decode(think, skip_special_tokens=False),
                'reasoning_length': len(think), 'reasoning_closed': closed,
                'length_capped': first.finish_reason == 'length', 'seed': args.seed,
                'thinking_temperature': .7, 'natural_answer_temperature': .7,
                'cut_answer_temperature': 0, 'natural_full_token_ids': ids,
                'base_unexpanded_token_ids': base[row['sample_id']],
                'expanded_base_token_ids': list(out.prompt_token_ids), 'cuts': [],
                'natural': {'text': natural_text, 'bbox_2d': parse_bbox(natural_text),
                    'token_ids': natural_ids, 'finish_reason': first.finish_reason} if closed else None,
                'ground_truth_in_prompt': False}
            records[row['sample_id']] = rec
            for pct in (100,):
                k = len(think)*pct//100
                reqs.append(request(row, base[row['sample_id']]+think[:k]+forced_suffix))
                maps.append((row['sample_id'], pct, k))
        for begin in range(0, len(reqs), 128):
            answers = llm.generate(reqs[begin:begin+128], bbox_params, use_tqdm=False)
            for out, (sample, label, k) in zip(answers, maps[begin:begin+128]):
                rec = records[sample]
                expected = rec['expanded_base_token_ids']+rec['reasoning_token_ids'][:k]+[closing]
                if list(out.prompt_token_ids) != expected:
                    raise RuntimeError('Original multimodal prefix changed during fresh prefill')
                value = decoded(out)
                if label == 'natural':
                    rec['natural'] = value
                else:
                    rec['cuts'].append({'trunc_pct': label, 'cut_token': k, **value})
        for rec in records.values():
            write_json(output/'raw'/f'{rec["sample_id"]}.json', rec)
        for img in images.values():
            img.close()
        status('running', samples_completed=min(start+len(chunk), len(tasks)), total=len(tasks))
        print('BATCH_COMPLETE', min(start+len(chunk), len(tasks)), len(tasks),
              'seconds', round(time.time()-started, 1), flush=True)
    status('complete', total=len(tasks), gpu_hours=(time.time()-started)/3600)


if __name__ == '__main__':
    main()
