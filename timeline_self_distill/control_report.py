"""Evaluate frozen same-candidate-count controls; selection never uses labels."""
import argparse
import json
from pathlib import Path

import numpy as np

from rl_opd_v1.common import read_json
from .eval import evaluate
from .prepare import ROOT, write_json
from .select import select_boxes


def comparison(first, second, groups, affected):
    delta = first-second
    low, high = delta.copy(), delta.copy()
    low[affected], high[affected] = -1, 1
    rng = np.random.default_rng(0)
    ix = rng.integers(0, len(delta), (2000, len(delta)))
    ci = np.quantile(delta[ix].mean(axis=1), [.025, .975])*100
    envelope = [float(np.quantile(low[ix].mean(axis=1), .025)*100),
                float(np.quantile(high[ix].mean(axis=1), .975)*100)]
    rng = np.random.default_rng(0)
    cluster, lower, upper = [], [], []
    for _ in range(2000):
        ix = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        cluster.append(delta[ix].mean()); lower.append(low[ix].mean()); upper.append(high[ix].mean())
    cluster_ci = np.quantile(cluster, [.025, .975])*100
    cluster_envelope = [float(np.quantile(lower, .025)*100), float(np.quantile(upper, .975)*100)]
    bound = 200*sum(affected)/len(delta)
    return {'delta_pp': float(delta.mean()*100), 'paired_ci95_pp': ci.tolist(),
        'image_cluster_ci95_pp': cluster_ci.tolist(),
        'first_correct_second_wrong': int(((first == 1) & (second == 0)).sum()),
        'first_wrong_second_correct': int(((first == 0) & (second == 1)).sum()),
        'affected_pairs': sum(affected), 'point_sensitivity_radius_pp': bound,
        'bootstrap_sensitivity_envelope_pp': envelope,
        'cluster_sensitivity_envelope_pp': cluster_envelope,
        'robust_pass': bool(len(delta) == 500 and delta.mean()*100-bound > 2
                            and envelope[0] > 0 and cluster_envelope[0] > 0)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--controls', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--frozen-selection', type=Path, required=True)
    parser.add_argument('--n', type=int, default=500)
    args = parser.parse_args()
    config = read_json(args.controls/'run_config.json')
    assert config['n'] == args.n and config['source'] == str(args.source.resolve())
    rows = [json.loads(line) for line in (ROOT/'data/train2000.jsonl').read_text().splitlines()][:args.n]
    frozen = {r['sample_id']: r for r in [json.loads(line) for line in
              (args.frozen_selection/'predictions.jsonl').read_text().splitlines()]}
    sources = [read_json(args.source/'raw'/f'{row["sample_id"]}.json') for row in rows]
    predictions, anomalies, costs = {}, {}, {}
    for arm in ('terminal', 'early_sampled'):
        predictions[arm], anomalies[arm] = [], []
        records = [read_json(args.controls/arm/'raw'/f'{row["sample_id"]}.json') for row in rows]
        for row, original, rec in zip(rows, sources, records):
            assert rec['sample_id'] == row['sample_id'] == original['sample_id']
            assert rec['arm'] == arm and [c['candidate_index'] for c in rec['candidates']] == list(range(10))
            texts = [c['text'] for c in rec['candidates']]
            choice, support = select_boxes(texts)
            predictions[arm].append({'sample_id': row['sample_id'], 'candidate_index': choice,
                'text': texts[choice] if choice is not None else original['cuts'][-1]['text'],
                'support': support, 'fallback': choice is None, 'labels_used_for_selection': False})
            anomalies[arm].append(any(c['bbox_2d'] is None or c['finish_reason'] == 'length'
                                      for c in rec['candidates']))
        costs[arm] = {'candidates': sum(len(r['candidates']) for r in records),
            'total_prefill_tokens': sum(c['prefill_tokens'] for r in records for c in r['candidates']),
            'answer_tokens_excluding_eos': sum(len(c['token_ids']) for r in records for c in r['candidates']),
            'fallbacks': sum(p['fallback'] for p in predictions[arm])}
    predictions['frozen_early_greedy'] = [frozen[row['sample_id']] for row in rows]
    # All candidates are selected before any ground-truth evaluation occurs.
    scores = {arm: np.array([evaluate(p['text'], row)['correct'] for row, p in zip(rows, selected)],
                           dtype=float) for arm, selected in predictions.items()}
    paths = sorted({row['image_path'] for row in rows})
    groups = [np.array([i for i, row in enumerate(rows) if row['image_path'] == path]) for path in paths]
    matched_bad = [a or b for a, b in zip(anomalies['early_sampled'], anomalies['terminal'])]
    greedy_bad = [a or rec['length_capped'] or not rec['reasoning_closed'] or any(
        c['bbox_2d'] is None or c['finish_reason'] == 'length' for c in rec['cuts'])
        for a, rec in zip(anomalies['terminal'], sources)]
    summary = {'n': len(rows), 'image_clusters': len(groups),
        'accuracies': {arm: float(value.mean()) for arm, value in scores.items()},
        'primary_matched_early_minus_terminal': comparison(scores['early_sampled'], scores['terminal'], groups, matched_bad),
        'secondary_frozen_greedy_minus_terminal': comparison(scores['frozen_early_greedy'], scores['terminal'], groups, greedy_bad),
        'costs': costs, 'seed': 0, 'candidate_seeds': list(range(10)), 'bootstrap_resamples': 2000,
        'heldout_read': False, 'original_E0_pass_changed': False,
        'interpretation': 'Same candidate count, same sampled temperature for primary contrast; not equal GPU time. No distillation/GRPO evidence.'}
    write_json(args.controls/'metrics.json', summary)
    for arm, selected in predictions.items():
        (args.controls/f'{arm}_predictions.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in selected))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
