"""GT-free fixed complete-image views, integer bbox mapping and token reserve."""
import json
import math
import numpy as np
from PIL import Image
from .eval import parse_bbox

ARMS = ('phase', 'canonical', 'original')
CAP = 12000
MIN_NATURAL = 512
NATURAL_MAX = 8192
ANSWER_MAX = 128

def canvas_size(width, height):
    w = math.ceil((width+64)/32)*32
    h = math.ceil((height+64)/32)*32
    while w*h < 65536:
        w += 32
        h += 32
    return w, h

def view_index(arm, candidate_index):
    assert arm in ARMS and candidate_index >= 0
    return candidate_index % 3 if arm == 'phase' else 0

def render_view(image, arm, index):
    assert arm in ARMS and index in (0,1,2)
    if arm == 'original':
        assert index == 0
        return image.copy(), {'canvas_wh':list(image.size),'offset_xy':[0,0],'original_wh':list(image.size)}
    w,h = canvas_size(*image.size)
    ox,oy = [(32,32),(40,32),(32,40)][index]
    result = Image.new('RGB',(w,h),(127,127,127))
    result.paste(image,(ox,oy))
    return result, {'canvas_wh':[w,h],'offset_xy':[ox,oy],'original_wh':list(image.size)}

def inverse_box(text, view):
    b = parse_bbox(text)
    if b is None or b[0] >= b[2] or b[1] >= b[3]:
        return None
    cw,ch = view['canvas_wh']; w,h = view['original_wh']; ox,oy = view['offset_xy']
    raw = [(b[0]*cw/1000-ox)*1000/w,(b[1]*ch/1000-oy)*1000/h,
           (b[2]*cw/1000-ox)*1000/w,(b[3]*ch/1000-oy)*1000/h]
    result = np.rint(np.clip(raw,0,1000)).astype(int).tolist()
    return result if result[0]<result[2] and result[1]<result[3] else None

def bbox_text(box):
    return json.dumps({'bbox_2d':box}) if box is not None else '{}'

def next_natural_limit(remaining, base_tokens, candidate_index):
    """Reserve first3 minimally512-token sources; then use all leftover budget.

    For each request pair: source B+N, answer B+T+1+A; T<=N<=M,
    A<=128, so worst2B+2M+129. Reserve512 for unstarted first3.
    EOS belongs to generated N/A; even an unclosed source has forced closing1.
    No outcome or GT enters this limit.
    """
    assert 0 <= remaining <= CAP and base_tokens>0 and candidate_index>=0
    future = max(0,2-candidate_index)
    future_min = future*(2*base_tokens+2*MIN_NATURAL+1+ANSWER_MAX)
    m = min(NATURAL_MAX,(remaining-2*base_tokens-1-ANSWER_MAX-future_min)//2)
    return m if m >= MIN_NATURAL else None

def prompt_text(expression):
    return expression+'\nReturn the bounding box of the referred object as {"bbox_2d": [x1, y1, x2, y2]}. Use four integers from 0 to 1000 in relative image coordinates. Output only this JSON in the final answer.'
