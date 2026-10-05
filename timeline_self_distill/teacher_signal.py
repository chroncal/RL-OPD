"""Raw same-student-prefix teacher signals; no training and no GT input."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from rl_opd_v1.common import read_json
from .prepare import ROOT, write_json


def coordinate_indices(tokenizer, ids):
    text = tokenizer.decode(ids, skip_special_tokens=False)
    left, right = text.find('['), text.find(']')
    if left < 0 or right < left:
        return []
    indices = []
    for j in range(len(ids)):
        before = tokenizer.decode(ids[:j], skip_special_tokens=False)
        after = tokenizer.decode(ids[:j+1], skip_special_tokens=False)
        if any(text[k].isdigit() and left < k < right
               for k in range(len(before), min(len(after), len(text)))):
            indices.append(j)
    return indices


def numeric_fork(tokenizer, student, teacher):
    sc, tc = coordinate_indices(tokenizer, student), coordinate_indices(tokenizer, teacher)
    for j, (s, t) in enumerate(zip(student, teacher)):
        if s != t:
            if j in sc and j in tc and all(tokenizer.decode([v]).strip().isdigit() for v in (s, t)):
                return j, 'shared_numeric_fork'
            return None, 'first_fork_not_pure_coordinate'
    return None, 'identical_or_prefix_answers'


def extend_image_inputs(base, ids):
    """Reuse the original image tensors and preserve multimodal RoPE token types."""
    import torch
    n = base['input_ids'].shape[1]
    assert ids[:n] == base['input_ids'][0].tolist()
    inputs = {k: v for k, v in base.items() if k not in
              ('input_ids', 'attention_mask', 'token_type_ids', 'mm_token_type_ids', 'position_ids')}
    inputs['input_ids'] = torch.tensor([ids], device=base['input_ids'].device)
    inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
    mm = base['mm_token_type_ids']
    inputs['mm_token_type_ids'] = torch.cat([mm, torch.zeros((1, len(ids)-n),
        dtype=mm.dtype, device=mm.device)], dim=1)
    return inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--controls', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', type=int, required=True)
    parser.add_argument('--n', type=int, default=500)
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.device), HF_HUB_OFFLINE='1',
                      TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    torch.set_num_threads(4)
    started = time.time()
    receipt = read_json(ROOT/'model_receipt.json')
    config = {'source': str(args.source.resolve()), 'controls': str(args.controls.resolve()),
        'selection': str(args.selection.resolve()), 'model': receipt, 'n': args.n,
        'student': 'terminal sampled candidate seed0', 'teacher': 'frozen X1 greedy medoid cut',
        'backend': 'transformers5.17 BF16 SDPA raw full vocabulary',
        'grammar_renormalized': False, 'weight_updates': 0, 'heldout_read': False}
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output/'run_config.json'
    if path.exists():
        if read_json(path) != config:
            raise RuntimeError('Teacher-signal resume configuration mismatch')
    else:
        write_json(path, config)
    assert read_json(args.source/'run_config.json')['model'] == receipt
    assert read_json(args.controls/'run_config.json')['model'] == receipt
    rows = [{k: row[k] for k in ('sample_id', 'image_path', 'expression')} for row in
            [json.loads(line) for line in (ROOT/'data/train2000.jsonl').read_text().splitlines()][:args.n]]
    selected = {r['sample_id']: r for r in [json.loads(line) for line in
                (args.selection/'predictions.jsonl').read_text().splitlines()]}
    processor = AutoProcessor.from_pretrained(receipt['path'], local_files_only=True)
    tokenizer = processor.tokenizer
    closing = tokenizer.encode('</think>', add_special_tokens=False)
    assert len(closing) == 1
    model = Qwen3_5ForConditionalGeneration.from_pretrained(receipt['path'],
        dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True).cuda().eval()
    for i, row in enumerate(rows):
        dest = args.output/'raw'/f'{row["sample_id"]}.json'
        if dest.exists():
            continue
        rec = read_json(args.source/'raw'/f'{row["sample_id"]}.json')
        terminal = read_json(args.controls/'terminal/raw'/f'{row["sample_id"]}.json')
        student = terminal['candidates'][0]
        assert student['candidate_index'] == student['seed'] == 0 and student['source_trunc_pct'] == 100
        choice = selected[row['sample_id']]
        pct = choice['selected_pct']
        teacher = next(c for c in rec['cuts'] if c['trunc_pct'] == pct)
        assert teacher['text'] == choice['text']
        text = row['expression']+'\nReturn the bounding box of the referred object as '
        text += '{"bbox_2d": [x1, y1, x2, y2]}. Use four integers from 0 to 1000 '
        text += 'in relative image coordinates. Output only this JSON in the final answer.'
        messages = [{'role': 'user', 'content': [{'type': 'image', 'image': row['image_path']},
                                                 {'type': 'text', 'text': text}]}]
        base = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=True, return_dict=True, return_tensors='pt').to('cuda')
        assert base['input_ids'][0].tolist() == rec['expanded_base_token_ids'], 'Original image prompt changed'
        output_ids = student['token_ids']
        indices = coordinate_indices(tokenizer, output_ids)
        record = {'sample_id': row['sample_id'], 'teacher_pct': pct,
            'student_text': student['text'], 'teacher_text': teacher['text'],
            'coordinate_indices': indices, 'ground_truth_in_forward': False,
            'student_finish_reason': student['finish_reason'], 'teacher_fallback': choice['fallback']}
        if not output_ids or not indices:
            record.update(state='invalid_student_coordinates', coordinates=[], fork=None)
            write_json(dest, record)
            continue
        logs = []
        for k in [rec['reasoning_length'], rec['reasoning_length']*pct//100]:
            prefix = rec['expanded_base_token_ids']+rec['reasoning_token_ids'][:k]+closing
            inputs = extend_image_inputs(base, prefix+output_ids)
            with torch.inference_mode():
                logits = model(**inputs, use_cache=False, logits_to_keep=len(output_ids)+1).logits[0, :-1]
                assert logits.shape[0] == len(output_ids)
                logp = logits.float().log_softmax(-1)
                assert torch.isfinite(logp).all(), 'Non-finite raw probabilities'
                logs.append(logp)
        lp, lq = logs
        coords = []
        for j in indices:
            kl = (lp[j].exp()*(lp[j]-lq[j])).sum()
            coords.append({'index': j, 'token_id': output_ids[j],
                'student_logprob': float(lp[j, output_ids[j]]),
                'teacher_logprob': float(lq[j, output_ids[j]]),
                'sampled_advantage': float(lq[j, output_ids[j]]-lp[j, output_ids[j]]),
                'raw_reverse_kl': float(kl)})
        j, reason = numeric_fork(tokenizer, output_ids, teacher['token_ids'])
        fork = None
        if j is not None:
            vs, vt = output_ids[j], teacher['token_ids'][j]
            ps, pt = lp[j, vs].exp(), lp[j, vt].exp()
            adv = lq[j, vs]-lp[j, vs]
            kl = (lp[j].exp()*(lp[j]-lq[j])).sum()
            gs = ps*(lp[j, vs]-lq[j, vs]-kl)
            gt = pt*(lp[j, vt]-lq[j, vt]-kl)
            fork = {'index': j, 'student_token': vs, 'teacher_branch_token': vt,
                'student_branch_text': tokenizer.decode([vs]), 'teacher_branch_text': tokenizer.decode([vt]),
                'logodds_improvement': float((lq[j, vt]-lq[j, vs])-(lp[j, vt]-lp[j, vs])),
                'sampled_advantage': float(adv),
                'sampled_pg_margin_direction': float(-adv*(1-ps+pt)),
                'exact_kl_margin_direction': float(gs-gt), 'raw_reverse_kl': float(kl)}
        record.update(state='complete', coordinates=coords, fork=fork, fork_reason=reason,
                      original_prompt_tokens=len(rec['expanded_base_token_ids']))
        write_json(dest, record)
        write_json(args.output/'worker.json', {'pid': os.getpid(), 'state': 'running',
            'completed_through_row': i+1, 'n': args.n, 'seconds': time.time()-started})
        if (i+1) % 20 == 0 or args.n <= 5:
            print('TEACHER_SIGNAL', i+1, args.n, 'seconds', round(time.time()-started, 1), flush=True)
    write_json(args.output/'worker.json', {'pid': os.getpid(), 'state': 'complete',
        'command': sys.argv, 'n': args.n, 'gpu_hours': (time.time()-started)/3600})


if __name__ == '__main__':
    main()
