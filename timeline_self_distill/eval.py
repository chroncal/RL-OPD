"""Frozen ground-truth evaluator: normalized integer XYXY -> pixel IoU."""
import json
import math

SCHEMA = {
    "type": "object",
    "properties": {"bbox_2d": {"type": "array", "items": {
        "type": "integer", "minimum": 0, "maximum": 1000},
        "minItems": 4, "maxItems": 4}},
    "required": ["bbox_2d"], "additionalProperties": False,
}


def parse_bbox(text):
    try:
        obj = json.loads(text.strip())
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict) or set(obj) != {"bbox_2d"}:
        return None
    box = obj["bbox_2d"]
    if not isinstance(box, list) or len(box) != 4:
        return None
    if any(type(v) is not int or not 0 <= v <= 1000 for v in box):
        return None
    return box


def pixel_bbox(box, width, height):
    return [box[0] * width / 1000, box[1] * height / 1000,
            box[2] * width / 1000, box[3] * height / 1000]


def iou(a, b):
    if a is None or b is None or len(a) != 4 or len(b) != 4:
        return 0.0
    if any(not math.isfinite(v) for v in a + b):
        return 0.0
    aa = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    bb = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    inter = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(
        0, min(a[3], b[3])-max(a[1], b[1]))
    return inter / (aa + bb - inter) if aa + bb - inter > 0 else 0.0


def evaluate(text, row):
    box = parse_bbox(text)
    value = iou(pixel_bbox(box, row['width'], row['height']), row['gt_xyxy']) if box else 0.0
    return {"bbox_2d": box, "iou": value, "correct": value >= 0.5,
            "parse_valid": box is not None}
