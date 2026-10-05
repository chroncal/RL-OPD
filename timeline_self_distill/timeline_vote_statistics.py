"""CPU-only, GT-free IoU-medoid selection from saved fixed-position boxes."""
import argparse
import hashlib
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

from .eval import evaluate, parse_bbox
from .full_chain_control import save_once
from .prepare import ROOT

POSITIONS = list(range(10, 101, 10))
GROUPS = {"all": list(range(10)), "early": list(range(5)), "late": list(range(5, 10))}
BIN_EDGES = [0, .2, .4, .6, .8, .9, 1]


def valid(box):
    return box is not None and box[2] > box[0] and box[3] > box[1]


def exact_iou(a, b):
    if not valid(a) or not valid(b):
        return Fraction(0)
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return Fraction(intersection, union)


def vote(boxes):
    """No GT or correctness input. Exact rational scores avoid roundoff ties."""
    matrix = [[exact_iou(a, b) if i != j else Fraction(0)
               for j, b in enumerate(boxes)] for i, a in enumerate(boxes)]
    answers = {}
    for name, indices in GROUPS.items():
        scores = {i: sum((matrix[i][j] for j in indices if j != i), Fraction(0)) / (len(indices)-1)
                  for i in indices}
        best = max(scores.values())
        ties = [i for i in indices if scores[i] == best]
        chosen = ties[0]  # Earliest position, fixed before inspecting outcomes.
        answers[name] = {"index": chosen, "position_pct": POSITIONS[chosen],
                         "score": float(best), "tied_positions": [POSITIONS[i] for i in ties],
                         "distinct_tied_boxes": len({tuple(boxes[i]) if boxes[i] else None for i in ties})}
    mean_pair_iou = sum((matrix[i][j] for i in range(10) for j in range(i+1, 10)), Fraction(0)) / 45
    return answers, float(mean_pair_iou)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = ROOT / 'results/fixed_train2000_classification_20261004/sample/raw'
    cohort_file = ROOT / 'results/fixed_position_statistics_20261005/per_sample.json'
    cohort = json.loads(cohort_file.read_text())
    manifest = {r['sample_id']: r for r in map(json.loads, (ROOT/'data/train2000.jsonl').read_text().splitlines())}
    assert len(cohort) == len({r['sample_id'] for r in cohort}) == 1994
    records = []
    anomalies = []
    hashes = {}
    for old in cohort:
        sid = old['sample_id']
        raw_bytes = (source/f'{sid}.json').read_bytes()
        hashes[sid] = hashlib.sha256(raw_bytes).hexdigest()
        raw = json.loads(raw_bytes)
        assert not raw['length_capped'] and raw['reasoning_closed']
        cuts = {c['trunc_pct']: c for c in raw['cuts']}
        boxes = [parse_bbox(cuts[p]['text']) for p in POSITIONS]
        assert all(b == cuts[p]['bbox_2d'] for p, b in zip(POSITIONS, boxes))
        answers, pair_mean = vote(boxes)
        # Only now consult GT, after the choices have been fixed.
        final = evaluate(cuts[100]['text'], manifest[sid])
        assert final['correct'] == old['final_correct'] and final['iou'] == old['final_iou']
        for answer in answers.values():
            answer['evaluation'] = evaluate(cuts[answer['position_pct']]['text'], manifest[sid])
        invalid = [p for p, b in zip(POSITIONS, boxes) if not valid(b)]
        if invalid:
            anomalies.append({'sample_id': sid, 'invalid_positions': invalid})
        records.append({'sample_id': sid, 'boxes': boxes, 'final_correct': final['correct'],
                        'final_iou': final['iou'], 'votes': answers,
                        'mean_pair_iou': pair_mean, 'invalid_positions': invalid})
    n = len(records)
    final_correct = sum(r['final_correct'] for r in records)
    assert final_correct == 1299 and len(anomalies) == 4
    table = []
    for name in GROUPS:
        correct = sum(r['votes'][name]['evaluation']['correct'] for r in records)
        rescued = sum(not r['final_correct'] and r['votes'][name]['evaluation']['correct'] for r in records)
        harmed = sum(r['final_correct'] and not r['votes'][name]['evaluation']['correct'] for r in records)
        assert correct - final_correct == rescued - harmed
        table.append({'method': name, 'n': n, 'correct': correct, 'accuracy_pct': 100*correct/n,
                      'final_wrong_vote_correct': rescued, 'final_correct_vote_wrong': harmed,
                      'delta_pp': 100*(correct-final_correct)/n,
                      'ties': sum(len(r['votes'][name]['tied_positions']) > 1 for r in records),
                      'ties_with_distinct_boxes': sum(r['votes'][name]['distinct_tied_boxes'] > 1 for r in records),
                      'selected_position_counts': dict(Counter(r['votes'][name]['position_pct'] for r in records))})
    import numpy as np
    values = np.array([r['mean_pair_iou'] for r in records])
    counts, _ = np.histogram(values, bins=BIN_EDGES)
    assert int(counts.sum()) == n
    summary = {'n': n, 'excluded_capped': 6, 'source': str(source), 'cohort': str(cohort_file),
               'protocol': {'selection': 'Maximize mean IoU to other group positions; choose an existing box, not coordinate averaging; GT used only after selection',
                            'ties': 'Earliest position; exact rational comparison',
                            'invalid_boxes': 'Missing/nonpositive-area box pair IoU is zero; retain all trajectories and all fixed denominator pairs',
                            'pair_mean': '45 unordered pairs of all ten positions',
                            'correct': 'Frozen pixel-space GT evaluator IoU >= 0.5',
                            'final': 'Saved 100% schema readout', 'bin_rule': '[lower, upper), last bin includes 1'},
               'final': {'correct': final_correct, 'accuracy_pct': 100*final_correct/n}, 'voting': table,
               'pair_iou_distribution': [{'lower': lo, 'upper': hi, 'count': int(c), 'pct': 100*int(c)/n}
                                         for lo, hi, c in zip(BIN_EDGES[:-1], BIN_EDGES[1:], counts)],
               'pair_iou_summary': {'mean': float(values.mean()), **{f'p{p}': float(np.percentile(values, p)) for p in (0, 10, 25, 50, 75, 90, 100)}},
               'anomalies': anomalies, 'training_updates': 0, 'new_generation': False, 'new_readouts': False}
    save_once(output/'statistics.json', summary)
    save_once(output/'per_sample.json', records)
    save_once(output/'source_sha256.json', hashes)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
