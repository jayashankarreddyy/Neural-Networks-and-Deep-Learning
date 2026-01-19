#object detecton 
import torch
import torchvision
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image

# COCO class names (fallback)
COCO_INSTANCE_CATEGORY_NAMES = [
    'background','person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
    'traffic light','fire hydrant','N/A','stop sign','parking meter','bench','bird','cat','dog','horse',
    'sheep','cow','elephant','bear','zebra','giraffe','N/A','backpack','umbrella','N/A','N/A','handbag',
    'tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove',
    'skateboard','surfboard','tennis racket','bottle','N/A','wine glass','cup','fork','knife','spoon',
    'bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake',
    'chair','couch','potted plant','bed','N/A','dining table','N/A','N/A','toilet','N/A','tv','laptop',
    'mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','N/A',
    'book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
]

# Try to create model using the modern weights API so we can get categories from weights.meta
try:
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    categories = weights.meta.get("categories", COCO_INSTANCE_CATEGORY_NAMES)
except Exception:
    # Fallback: older torchvision, create model with pretrained=True and use fallback categories
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    categories = COCO_INSTANCE_CATEGORY_NAMES

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
model.eval()

def detect_and_draw(img_path, score_thresh=0.6, save_path="output.jpg"):
    img = Image.open(img_path).convert("RGB")
    transform = transforms.ToTensor()
    img_tensor = transform(img).to(device)

    with torch.no_grad():
        outputs = model([img_tensor])[0]

    boxes = outputs['boxes'].cpu().numpy()
    labels = outputs['labels'].cpu().numpy()
    scores = outputs['scores'].cpu().numpy()

    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    for box, label, score in zip(boxes, labels, scores):
        if score < score_thresh:
            continue
        x1, y1, x2, y2 = box.astype(int)
        cls_name = categories[label] if label < len(categories) else str(label)
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img_cv, f"{cls_name} {score:.2f}", (x1, max(10,y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imwrite(save_path, img_cv)
    print(f"Saved: {save_path}")
    return boxes, labels, scores

img_path = input("Enter image path: ")
boxes, labels, scores = detect_and_draw(img_path)
for b, l, s in zip(boxes, labels, scores):
    if s >= 0.6:
        name = categories[l] if l < len(categories) else str(l)
        print(f"Detected {name} ({l}) with confidence {s:.3f} at {b.astype(int).tolist()}")
