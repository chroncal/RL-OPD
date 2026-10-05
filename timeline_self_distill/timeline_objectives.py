"""Whole-answer temporal targets; raw-vocabulary objectives, not schema-policy KL.

The model caller must pass logits from the position immediately BEFORE each
target token. Prefix/image/thinking positions never enter these answer losses.
No model, trajectory sampling, or teacher selection is performed here.
"""
import torch
import torch.nn.functional as F

def answer_logps(aligned_logits,target_ids):
    assert aligned_logits.ndim==2 and target_ids.ndim==1
    assert aligned_logits.shape[0]==len(target_ids) and len(target_ids)>0
    return F.log_softmax(aligned_logits.float(),dim=-1).gather(1,target_ids[:,None]).squeeze(1)

def whole_answer_sft(aligned_logits,target_ids):
    return -answer_logps(aligned_logits,target_ids).mean()

def whole_answer_dpo(positive_logp,negative_logp,reference_positive_logp,reference_negative_logp,beta=.1):
    assert beta>0
    delta=(positive_logp-reference_positive_logp.detach())-(negative_logp-reference_negative_logp.detach())
    return F.softplus(-beta*delta).mean()
