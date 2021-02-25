import torch
from PIL import ImageDraw
from torchvision.transforms import functional as VF
from torchvision.utils import make_grid


def normalize_boxes(img, boxes):
    _, H, W = img.size()
    x, y, w, h = boxes.unbind(-1)
    x1, x2 = (x-w/2)*W, (x+w/2)*W
    y1, y2 = (y-h/2)*H, (y+h/2)*H
    return torch.stack([x1, y1, x2, y2], -1)


def render_batch(samples, targets, outputs, classes, thresh=0.3):
    out1, out2 = [], []
    out_boxes = outputs['pred_boxes']
    out_scores, out_labels = torch.softmax(outputs['pred_logits'], -1).max(-1)
    outputs = list(zip(out_boxes, out_labels, out_scores))
    for img, tgt, out in zip(samples, targets, outputs):
        out1.append(draw_outputs(img, tgt['boxes'], tgt['labels'], classes, thresh=thresh))
        out2.append(draw_outputs(img, out[0], out[1], classes, out[2], thresh=thresh))
    out = out1+out2
    grid = make_grid(out, len(out1))
    return grid


def draw_outputs(samples, boxes, labels, classes, scores=None, thresh=0.3):
    boxes = normalize_boxes(samples, boxes)
    imx = VF.to_pil_image(samples)
    drw = ImageDraw.ImageDraw(imx)
    boxes, labels = boxes.tolist(), labels.tolist()
    if scores is not None:
        scores = scores.tolist()
    for i, (box, cls) in enumerate(zip(boxes, labels)):
        if cls == 0:
            continue
        if scores is not None and scores[i] < thresh:
            continue
        drw.rectangle(box, outline='red', width=2)
        txt = classes[cls]
        if scores is not None:
            txt += ' ({})'.format(round(scores[i], 2))
        drw.text(box[:2], txt, 'green')
    img = VF.to_tensor(imx)
    return img
