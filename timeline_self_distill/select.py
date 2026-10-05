"""One frozen, label-free early-box medoid; exploratory, not an E0 gate."""
import argparse
import json
from pathlib import Path

import numpy as np

from rl_opd_v1.common import read_json
from .eval import evaluate, iou, parse_bbox
from .prepare import ROOT, write_json


def select_boxes(texts):
    """Return the earliest medoid index and support, using prediction texts only."""
    valid = []
    for index, text in enumerate(texts):
        box = parse_bbox(text)
        if box is not None and box[0] < box[2] and box[1] < box[3]:
            valid.append((index, box))
    if not valid:
        return None, []
    support = [float(np.mean([iou(box, other_box) for j, (_, other_box) in enumerate(valid)
                              if j != i])) if len(valid) > 1 else 0.
               for i, (_, box) in enumerate(valid)]
    best = int(np.argmax(support))
    return valid[best][0], [{'candidate_index': index, 'support': s}
                           for (index, _), s in zip(valid, support)]


def select_early(cuts):
    # Only early model outputs enter this function, no metadata or dataset labels.
    assert [c['trunc_pct'] for c in cuts] == list(range(0, 100, 10))
    index, support = select_boxes([cut['text'] for cut in cuts])
    return (cuts[index] if index is not None else None), [
        {'trunc_pct': cuts[s['candidate_index']]['trunc_pct'], 'support': s['support']}
        for s in support]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['val', 'train'], default='val')
    parser.add_argument('--n', type=int, default=500)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Exploration output already exists; do not overwrite a trial')
    args.output.mkdir(parents=True)
    cohort = 'val500.jsonl' if args.split == 'val' else 'train2000.jsonl'
    rows = [json.loads(line) for line in (ROOT/'data'/cohort).read_text().splitlines()][:args.n]
    records = [read_json(args.source/'raw'/f'{row["sample_id"]}.json') for row in rows]
    predictions = []
    for row, rec in zip(rows, records):
        assert row['sample_id'] == rec['sample_id']
        selected, support = select_early([{'trunc_pct': c['trunc_pct'], 'text': c['text']}
                                        for c in rec['cuts'][:10]])
        predictions.append({'sample_id': rec['sample_id'],
            'selected_pct': selected['trunc_pct'] if selected else 100,
            'text': selected['text'] if selected else rec['cuts'][-1]['text'],
            'support': support, 'fallback': selected is None,
            'labels_used_for_selection': False})
    # Selection is complete before dataset labels are evaluated.
    selected_scores = np.array([evaluate(p['text'], row)['correct']
                               for row, p in zip(rows, predictions)], dtype=float)
    final_scores = np.array([evaluate(rec['cuts'][-1]['text'], row)['correct']
                            for row, rec in zip(rows, records)], dtype=float)
    delta = selected_scores-final_scores
    affected = [rec['length_capped'] or not rec['reasoning_closed'] or any(
        parse_bbox(c['text']) is None or c['finish_reason'] == 'length' for c in rec['cuts'])
        for rec in records]
    # Each selected-minus-final outcome lies in [-1,1]. Replacing an arbitrary
    # affected pair can change its difference by up to TWO accuracy units.
    missingness_radius_pp = 200*sum(affected)/len(rows)
    rng = np.random.default_rng(0)
    indices = rng.integers(0, len(rows), size=(2000, len(rows)))
    draws = delta[indices].mean(axis=1)
    ci = np.quantile(draws, [.025, .975])*100
    lower_delta, upper_delta = delta.copy(), delta.copy()
    lower_delta[affected], upper_delta[affected] = -1, 1
    worst_ci = [float(np.quantile(lower_delta[indices].mean(axis=1), .025)*100),
                float(np.quantile(upper_delta[indices].mean(axis=1), .975)*100)]
    image_paths = sorted({row['image_path'] for row in rows})
    clusters = [np.array([i for i, row in enumerate(rows) if row['image_path'] == path])
                for path in image_paths]
    rng = np.random.default_rng(0)
    cluster_draws, cluster_lower, cluster_upper = [], [], []
    for _ in range(2000):
        ix = np.concatenate([clusters[i] for i in rng.integers(0, len(clusters), len(clusters))])
        cluster_draws.append(delta[ix].mean())
        cluster_lower.append(lower_delta[ix].mean())
        cluster_upper.append(upper_delta[ix].mean())
    cluster_ci = np.quantile(cluster_draws, [.025, .975])*100
    worst_cluster_ci = [float(np.quantile(cluster_lower, .025)*100),
                        float(np.quantile(cluster_upper, .975)*100)]
    summary = {'trial': 'E0-X1 early-only pairwise-IoU medoid', 'n': len(rows),
        'split': args.split,
        'selected_acc': float(selected_scores.mean()), 'endpoint_acc': float(final_scores.mean()),
        'delta_pp': float(delta.mean()*100), 'paired_ci95_pp': ci.tolist(),
        'image_cluster_ci95_pp': cluster_ci.tolist(),
        'selected_correct_final_wrong': int(((selected_scores == 1) & (final_scores == 0)).sum()),
        'selected_wrong_final_correct': int(((selected_scores == 0) & (final_scores == 1)).sum()),
        'selected_pct_counts': {str(k): sum(p['selected_pct'] == k for p in predictions)
                                for k in range(0, 101, 10)},
        'fallbacks': sum(p['fallback'] for p in predictions),
        'anomaly_affected_samples': sum(affected),
        'arbitrary_affected_pair_radius_pp': missingness_radius_pp,
        'worst_case_delta_interval_pp': [float(delta.mean()*100)-missingness_radius_pp,
                                       float(delta.mean()*100)+missingness_radius_pp],
        'worst_case_paired_ci95_pp': worst_ci,
        'worst_case_cluster_ci95_pp': worst_cluster_ci,
        'robust_to_arbitrary_affected_pairs': bool(
            len(rows) == 500 and delta.mean()*100-missingness_radius_pp > 2
            and worst_ci[0] > 0 and worst_cluster_ci[0] > 0),
        'mean_reasoning_tokens': float(np.mean([r['reasoning_length'] for r in records])),
        'bootstrap_resamples': 2000, 'seed': 0,
        'complete_cohort': len(rows) == 500,
        'exploration_pass': bool(len(rows) == 500 and delta.mean() > .02
                                 and ci[0] > 0 and cluster_ci[0] > 0),
        'original_E0_pass_changed': False, 'heldout_read': False,
        'caveats': ('Exploration designed after E0 aggregate results; capped chains retained; '
                    + ('same val cohort, not independent confirmation' if args.split == 'val'
                       else 'disjoint train diagnostic cohort; no RL training yet')
                    + '; not a distillation result.')}
    write_json(args.output/'metrics.json', summary)
    (args.output/'predictions.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in predictions))
    write_json(args.output/'run_config.json', {'source': str(args.source.resolve()),
        'split': args.split, 'n': args.n,
        'rule': 'max mean pairwise IoU among geometrically valid earlier 0..90% boxes; earliest tie',
        'no_tuning': True, 'no_gt_selection': True})
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
