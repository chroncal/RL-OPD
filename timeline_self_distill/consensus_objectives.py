"""GT-free consensus target and coordinate-only supervised objective."""
from .eval import parse_bbox
from .timeline_vote_statistics import vote, valid


def consensus_target(candidates, gate='middle_changed'):
    assert [c['pct'] for c in candidates] == list(range(10, 101, 10))
    boxes = [parse_bbox(c['text']) for c in candidates]
    answers, stability = vote(boxes)
    selected = answers['all']['index']
    teacher = boxes[selected]
    final = boxes[-1]
    active = valid(teacher) and stability >= .4
    if gate == 'middle_changed':
        active = active and stability < .9 and teacher != final
    elif gate != 'lower_only':
        raise ValueError(gate)
    return {'selected_index': selected, 'selected_pct': candidates[selected]['pct'],
            'teacher_bbox': teacher, 'final_bbox': final, 'stability': stability,
            'active': bool(active), 'gate': gate}


def coordinate_ce(model, inputs, target_ids, indices):
    import torch
    # The first retained logit belongs to the final prefix token, predicting
    # target[0]; the last logit is unused. Thinking has no loss-mask entries.
    kwargs = dict(inputs, use_cache=False, logits_to_keep=len(target_ids)+1)
    logits = model(**kwargs).logits[0, :-1]
    assert logits.shape[0] == len(target_ids)
    chosen = torch.tensor(indices, device=logits.device)
    selected = logits.index_select(0, chosen).float()
    target = torch.tensor([target_ids[j] for j in indices], device=logits.device)
    loss = torch.nn.functional.cross_entropy(selected, target, reduction='mean')
    assert torch.isfinite(loss)
    return loss
