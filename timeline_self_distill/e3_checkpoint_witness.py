"""Zero-update saved-checkpoint inference witness on a synthetic image only."""
import argparse
import os
import time
from pathlib import Path

from .eval import SCHEMA, parse_bbox
from .e3_launch import MODEL
from .prepare import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    assert args.device in (0, 1) and not args.output.exists()
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.device), TOKENIZERS_PARALLELISM='false',
        VLLM_WORKER_MULTIPROC_METHOD='spawn', VLLM_NO_USAGE_STATS='1',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    from PIL import Image, ImageDraw
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    from transformers import AutoProcessor
    started = time.time()
    image = Image.new('RGB', (128, 128), 'white')
    ImageDraw.Draw(image).ellipse((40, 40, 88, 88), fill='blue')
    text = ('the blue circle\nReturn the bounding box of the referred object as '
        '{"bbox_2d": [x1, y1, x2, y2]}. Use four integers from 0 to 1000 '
        'in relative image coordinates. Output only this JSON in the final answer.')
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    prompt = processor.apply_chat_template([{'role':'user','content':[
        {'type':'image','image':'synthetic.png'},{'type':'text','text':text}]}],
        add_generation_prompt=True, tokenize=False, enable_thinking=True)
    llm = LLM(model=str(args.checkpoint.resolve()), tokenizer=MODEL, dtype='bfloat16',
        tensor_parallel_size=1, max_model_len=12288, gpu_memory_utilization=.6,
        max_num_seqs=1, max_num_batched_tokens=16384, enforce_eager=True,
        enable_prefix_caching=False, enable_chunked_prefill=True, async_scheduling=False,
        attention_config={'backend':'FLASH_ATTN'}, limit_mm_per_prompt={'image':1},
        seed=0, reasoning_parser='qwen3')
    tokenizer = llm.get_tokenizer()
    base = tokenizer.encode(prompt, add_special_tokens=False)
    closing = tokenizer.encode('</think>', add_special_tokens=False)
    assert len(closing) == 1
    schema = StructuredOutputsParams(json=SCHEMA)
    request = lambda ids: {'prompt_token_ids':ids,'multi_modal_data':{'image':image}}
    natural = llm.generate([request(base)], SamplingParams(temperature=.7, top_p=1, top_k=-1,
        seed=0, max_tokens=8192, skip_special_tokens=False, structured_outputs=schema), use_tqdm=False)[0]
    ids = list(natural.outputs[0].token_ids)
    assert closing[0] in ids, 'Synthetic witness did not close thinking'
    think = ids[:ids.index(closing[0])]
    final = llm.generate([request(base+think+closing)], SamplingParams(temperature=0,
        max_tokens=128, skip_special_tokens=False, structured_outputs=schema), use_tqdm=False)[0]
    assert list(final.prompt_token_ids) == list(natural.prompt_token_ids)+think+closing
    answer_ids = list(final.outputs[0].token_ids)
    if answer_ids and answer_ids[-1] == tokenizer.eos_token_id:
        answer_ids = answer_ids[:-1]
    answer = tokenizer.decode(answer_ids, skip_special_tokens=False).strip()
    assert parse_bbox(answer) is not None
    write_json(args.output, {'state':'passed','checkpoint':str(args.checkpoint.resolve()),
        'tokenizer':MODEL,'device':args.device,'seconds':time.time()-started,
        'natural_token_ids':ids,'natural_finish_reason':natural.outputs[0].finish_reason,
        'thinking_tokens':len(think),'final_text':answer,'final_token_ids':answer_ids,
        'fresh_original_expanded_prefix_match':True,'model_weights_loaded':True,
        'optimizer_updates':0,'dataset_examples_read':0,'ground_truth_used':False,
        'heldout_read':False,'scope':'Synthetic format/load/prefix witness only, no accuracy claim.'})
    print('CHECKPOINT_INFERENCE_WITNESS_PASSED', str(args.output), flush=True)


if __name__ == '__main__':
    main()
