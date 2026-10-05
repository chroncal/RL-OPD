"""Select official positive samples before inference; freeze local manifests."""
import argparse
import concurrent.futures
import datetime
import json
import random
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ASSETS = Path('/root/rl_opd_v1_runtime')


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    data_root = ROOT/'data'
    if data_root.exists():
        raise RuntimeError('Data already frozen; preparation cannot overwrite it')
    selections = {}
    for split in ('train', 'val', 'test'):
        raw = json.loads((ASSETS/'annotations'/f'{split}_coco.json').read_text())
        metas = {r['id']: r for r in raw['images']}
        positives = [a for a in raw['annotations'] if a.get('bbox') and
                     len(a['bbox']) == 4 and a['bbox'][2] > 0 and a['bbox'][3] > 0]
        rng = random.Random(args.seed)
        rng.shuffle(positives)
        if split == 'val':
            chosen = [a for level, n in ((1, 167), (2, 167), (3, 166))
                      for a in [r for r in positives if int(r['level']) == level][:n]]
            rng.shuffle(chosen)
        else:
            chosen = positives[:2000 if split == 'train' else 500]
        assert len(chosen) == (2000 if split == 'train' else 500)
        records = []
        for a in chosen:
            m = metas[a['image_id']]
            x, y, w, h = a['bbox']
            records.append({'sample_id': f'finecops-{split}-{a["id"]}',
                'source_annotation_id': a['id'], 'split': split,
                'image_path': str(ASSETS/'images'/Path(m['file_name']).name),
                'width': m['width'], 'height': m['height'],
                'expression': a.get('expression', m['caption']),
                'difficulty': int(a['level']), 'gt_xyxy': [x, y, x+w, y+h]})
        selections[split] = records
    write_json(ROOT/'selection_before_predictions.json', {
        'seed': args.seed, 'val_quotas': {'L1': 167, 'L2': 167, 'L3': 166},
        'sampling_unit': 'official positive expression/annotation; no outcome-based exclusion',
        'ids': {s: [r['source_annotation_id'] for r in rows] for s, rows in selections.items()},
        'heldout': 'metadata sealed; no image download or prediction/evaluation until final E2b',
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
    # Existing official byte-range image loader, without altering its cohorts or code.
    from rl_opd_v1.build_manifests import image
    from PIL import Image
    index = {Path(r['filename']).name: r for r in json.loads((ASSETS/'gqa_index.json').read_text()) if r['size']}
    # Training images are fetched only if E0 supports the hypothesis.
    unique = {Path(r['image_path']).name: r for r in selections['val']}
    def obtain(item):
        name, row = item
        details = image(index[name])
        if (details['width'], details['height']) != (row['width'], row['height']):
            raise RuntimeError(f'Dimension mismatch {name}')
        return name
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for i, _ in enumerate(pool.map(obtain, unique.items())):
            if (i+1) % 50 == 0:
                print('IMAGES_READY', i+1, len(unique), flush=True)
    (data_root/'heldout').mkdir(parents=True)
    for split, name in [('train', 'train2000.jsonl'), ('val', 'val500.jsonl'), ('test', 'heldout/test500.jsonl')]:
        path = data_root/name
        path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in selections[split]))
        path.chmod(0o444)
    # Images remain official shared cache; only manifest data live in the locked directory.
    (ROOT/'eval.py').chmod(0o444)
    write_json(ROOT/'data_receipt.json', {
        'counts': {s: len(r) for s, r in selections.items()},
        'difficulty': {s: dict(Counter(x['difficulty'] for x in r)) for s, r in selections.items()},
        'source': str(ASSETS/'annotations'), 'locked': ['eval.py', 'data/'],
        'heldout_read_for_experiments': False})
    print('DATA_FROZEN', flush=True)


if __name__ == '__main__':
    main()
