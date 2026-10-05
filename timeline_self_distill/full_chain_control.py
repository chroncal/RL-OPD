"""X5: fresh independent full chains, one sampled bbox per chain, no GT input."""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from .prepare import ROOT, write_json
from .eval import SCHEMA, parse_bbox

SOURCE = ROOT/'results/train_probe500_20261004'
EARLY = ROOT/'results/answer_control500_20261004'


def config(shard):
    return {'model': json.loads((ROOT/'model_receipt.json').read_text()), 'n': 500,
        'shards': 2, 'shard': shard, 'batch_size': 8, 'candidate_seeds': list(range(10)),
        'seed': 0, 'temperature': .7, 'top_p': 1, 'top_k': -1, 'natural_max_tokens': 8192,
        'answer_max_tokens': 128, 'schema': SCHEMA, 'max_model_len': 12288,
        'max_num_seqs': 32, 'max_num_batched_tokens': 16384, 'memory_fraction': .60,
        'source': str(SOURCE), 'early': str(EARLY), 'prefix_cache': False,
        'backend': 'vllm0.18.1 BF16 FLASH_ATTN eager, async scheduling disabled',
        'heldout_read': False, 'new_training': False}


def save_once(path, value):
    """Completed identities are immutable; atomic replace avoids partial JSON on recovery."""
    if path.exists():
        raise RuntimeError(f'Attempt to replace completed identity: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    os.replace(temporary, path)


def split_natural(ids, closing):
    closed = closing in ids
    at = ids.index(closing) if closed else len(ids)
    return ids[:at], ids[at+1:] if closed else [], closed


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', type=int, required=True); p.add_argument('--shard', type=int, required=True)
    args = p.parse_args(); assert args.device == args.shard and args.shard in (0, 1)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.device), TOKENIZERS_PARALLELISM='false',
        VLLM_WORKER_MULTIPROC_METHOD='spawn', VLLM_NO_USAGE_STATS='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    started = time.time(); output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    cfg = config(args.shard); cp = output/'run_config.json'
    if cp.exists():
        assert json.loads(cp.read_text()) == cfg, 'Fixed configuration mismatch on resume'
    else:
        save_once(cp, cfg)
    allrows = list(map(json.loads, (ROOT/'data/train2000.jsonl').read_text().splitlines()))[:500]
    rows = [{k: r[k] for k in ('sample_id', 'image_path', 'expression')} for i, r in enumerate(allrows) if i % 2 == args.shard]
    assert len(rows) == 250
    tasks = [r for r in rows if not (output/'raw'/f'{r["sample_id"]}.json').exists()]
    def status(state, **extra):
        write_json(output/'worker.json', {'state': state, 'pid': os.getpid(), 'command': sys.argv,
            'gpu': args.device, 'elapsed_seconds': time.time()-started, 'expected_samples': 250, **extra})
    status('loading')
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    load_started = time.time()
    llm = LLM(model=cfg['model']['path'], tokenizer=cfg['model']['path'], dtype='bfloat16', tensor_parallel_size=1,
        max_model_len=12288, gpu_memory_utilization=.60, max_num_seqs=32, max_num_batched_tokens=16384,
        enforce_eager=True, enable_prefix_caching=False, enable_chunked_prefill=True, async_scheduling=False,
        attention_config={'backend': 'FLASH_ATTN'}, limit_mm_per_prompt={'image': 1}, seed=0, reasoning_parser='qwen3')
    tokenizer = llm.get_tokenizer(); processor = AutoProcessor.from_pretrained(cfg['model']['path'])
    closing = tokenizer.encode('</think>', add_special_tokens=False); assert len(closing) == 1
    structured = StructuredOutputsParams(json=SCHEMA); loading_seconds = time.time()-load_started
    inference_started = time.time()
    for begin in range(0, len(tasks), 8):
        chunk = tasks[begin:begin+8]; images = {r['sample_id']: Image.open(r['image_path']).convert('RGB') for r in chunk}
        sources, bases = {}, {}
        for row in chunk:
            sid = row['sample_id']; original = json.loads((SOURCE/'raw'/f'{sid}.json').read_text()); sources[sid] = original
            text = row['expression']+'\nReturn the bounding box of the referred object as {"bbox_2d": [x1, y1, x2, y2]}. Use four integers from 0 to 1000 in relative image coordinates. Output only this JSON in the final answer.'
            messages = [{'role': 'user', 'content': [{'type': 'image', 'image': row['image_path']}, {'type': 'text', 'text': text}]}]
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=True)
            bases[sid] = tokenizer.encode(prompt, add_special_tokens=False)
            assert bases[sid] == original['base_unexpanded_token_ids']
        def request(row, ids):
            return {'prompt_token_ids': ids, 'multi_modal_data': {'image': images[row['sample_id']]},
                    'multi_modal_uuids': {'image': row['image_path']}}
        reqs, params, maps = [], [], []
        for row in chunk:
            for j in range(10):
                path = output/'natural'/row['sample_id']/f'{j}.json'
                if not path.exists():
                    reqs.append(request(row, bases[row['sample_id']]))
                    params.append(SamplingParams(temperature=.7, top_p=1, top_k=-1, seed=j,
                        max_tokens=8192, skip_special_tokens=False, structured_outputs=structured))
                    maps.append((row['sample_id'], j, path))
        if reqs:
            generated = llm.generate(reqs, params, use_tqdm=False); assert len(generated) == len(maps)
            for out, (sid, j, path) in zip(generated, maps):
                assert list(out.prompt_token_ids) == sources[sid]['expanded_base_token_ids']
                first = out.outputs[0]; ids = list(first.token_ids); think, answer, closed = split_natural(ids, closing[0])
                save_once(path, {'sample_id': sid, 'candidate_index': j, 'seed': j, 'natural_full_token_ids': ids,
                    'reasoning_token_ids': think, 'natural_answer_token_ids_including_eos': answer,
                    'reasoning_closed': closed, 'length_capped': first.finish_reason == 'length',
                    'finish_reason': first.finish_reason, 'base_unexpanded_token_ids': bases[sid],
                    'expanded_base_token_ids': list(out.prompt_token_ids), 'ground_truth_in_prompt': False})
        reqs, params, maps = [], [], []
        for row in chunk:
            for j in range(10):
                nat = json.loads((output/'natural'/row['sample_id']/f'{j}.json').read_text())
                assert nat['sample_id'] == row['sample_id'] and nat['candidate_index'] == nat['seed'] == j
                assert nat['base_unexpanded_token_ids'] == bases[row['sample_id']] and not nat['ground_truth_in_prompt']
                path = output/'answers'/row['sample_id']/f'{j}.json'
                if not path.exists():
                    reqs.append(request(row, bases[row['sample_id']]+nat['reasoning_token_ids']+closing))
                    params.append(SamplingParams(temperature=.7, top_p=1, top_k=-1, seed=j,
                        max_tokens=128, skip_special_tokens=False, structured_outputs=structured))
                    maps.append((nat, path))
        if reqs:
            generated = llm.generate(reqs, params, use_tqdm=False); assert len(generated) == len(maps)
            for out, (nat, path) in zip(generated, maps):
                assert list(out.prompt_token_ids) == nat['expanded_base_token_ids']+nat['reasoning_token_ids']+closing
                answer = out.outputs[0]; full = list(answer.token_ids); ids = full[:-1] if full and full[-1] == tokenizer.eos_token_id else full
                text = tokenizer.decode(ids, skip_special_tokens=False).strip()
                save_once(path, {'sample_id': nat['sample_id'], 'candidate_index': nat['candidate_index'], 'seed': nat['seed'],
                    'text': text, 'bbox_2d': parse_bbox(text), 'token_ids': ids, 'generated_token_ids_including_eos': full,
                    'finish_reason': answer.finish_reason, 'prefill_tokens': len(out.prompt_token_ids),
                    'expanded_prompt_token_ids': list(out.prompt_token_ids), 'ground_truth_in_prompt': False})
        for row in chunk:
            sid = row['sample_id']; candidates = []
            for j in range(10):
                nat = json.loads((output/'natural'/sid/f'{j}.json').read_text()); answer = json.loads((output/'answers'/sid/f'{j}.json').read_text())
                assert nat['sample_id'] == answer['sample_id'] == sid and nat['candidate_index'] == answer['candidate_index'] == j
                candidates.append({'candidate_index': j, 'natural': nat, 'answer': answer})
            save_once(output/'raw'/f'{sid}.json', {'sample_id': sid, 'arm': 'full_independent', 'candidates': candidates, 'ground_truth_in_prompt': False})
        for im in images.values(): im.close()
        completed = sum((output/'raw'/f'{r["sample_id"]}.json').exists() for r in rows)
        status('running', completed_samples=completed, loading_seconds=loading_seconds)
        print('FULL_CHAIN_BATCH', completed, 250, 'seconds', round(time.time()-started, 1), flush=True)
    assert all((output/'raw'/f'{r["sample_id"]}.json').exists() for r in rows)
    status('complete', completed_samples=250, loading_seconds=loading_seconds,
        inference_seconds=time.time()-inference_started, gpu_hours=(time.time()-started)/3600)


if __name__ == '__main__': main()
