"""Export a completed LoRA checkpoint into native frozen-base-compatible weights."""
import argparse
import json
import shutil
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--adapter',type=Path,required=True)
    p.add_argument('--local',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    import torch
    from swift.model import get_model_processor
    from peft import PeftModel
    torch.set_num_threads(4)
    base=Path('/root/rl_opd_v1_runtime/models/Qwen3.5-2B')
    model,_=get_model_processor(str(base),torch_dtype=torch.bfloat16,device_map='cpu',
                               attn_impl='sdpa',model_type='qwen3_5')
    frozen_f32={n:p.detach().clone() for n,p in model.named_parameters()
                if '.linear_attn.A_log' in n or '.linear_attn.norm.weight' in n}
    assert len(frozen_f32)==36 and all(p.dtype==torch.float32 for p in frozen_f32.values())
    model=PeftModel.from_pretrained(model,a.adapter,local_files_only=True)
    model=model.merge_and_unload(safe_merge=True)
    assert all(model.get_parameter(n).dtype==torch.float32 and torch.equal(model.get_parameter(n),p)
               for n,p in frozen_f32.items())
    a.local.mkdir(parents=True,exist_ok=False)
    model.save_pretrained(a.local,safe_serialization=True,max_shard_size='2GB')
    for name in ('preprocessor_config.json','processor_config.json','video_preprocessor_config.json','chat_template.jinja',
                 'tokenizer.json','tokenizer_config.json','merges.txt','vocab.json'):
        src=base/name
        if src.exists():shutil.copyfile(src,a.local/name)
    shutil.copytree(a.local,a.output,copy_function=shutil.copyfile)
    receipt={'adapter':str(a.adapter.resolve()),'local_merged':str(a.local.resolve()),'output':str(a.output.resolve()),
             'official_input_metadata_preserved':True,'checkpoint_weights_saved':True,'export_optimizer_updates':0,
             'frozen_native_F32_state_count':36,'frozen_native_F32_states_exact':True}
    (a.output/'export_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
