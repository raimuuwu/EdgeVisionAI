import cv2
import numpy as np
import onnxruntime as ort

CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush"
]

CONF_THRESH = 0.4
IOU_THRESH = 0.45

session = ort.InferenceSession("yolov5s.onnx")
input_name = session.get_inputs()[0].name

cap = cv2.VideoCapture(0)

def nms(boxes, scores, iou_thresh):
    indices = np.argsort(scores)[::-1]
    keep = []

    while indices.size > 0:
        i = indices[0]
        keep.append(i)

        xx1 = np.maximum(boxes[i][0], boxes[indices[1:]][:, 0])
        yy1 = np.maximum(boxes[i][1], boxes[indices[1:]][:, 1])
        xx2 = np.minimum(boxes[i][2], boxes[indices[1:]][:, 2])
        yy2 = np.minimum(boxes[i][3], boxes[indices[1:]][:, 3])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h

        area_i = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
        area_j = (boxes[indices[1:]][:, 2] - boxes[indices[1:]][:, 0]) * \
                 (boxes[indices[1:]][:, 3] - boxes[indices[1:]][:, 1])

        iou = inter / (area_i + area_j - inter)
        indices = indices[1:][iou < iou_thresh]

    return keep


def preprocess(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    return img

def postprocess(preds, frame_shape):
    frame_h, frame_w = frame_shape
    input_size = 640  # rozmiar wejścia YOLO

    scale_x = frame_w / input_size
    scale_y = frame_h / input_size

    boxes, scores, class_ids = [], [], []

    for det in preds:
        obj_conf = det[4]
        if obj_conf < CONF_THRESH:
            continue

        class_id = np.argmax(det[5:])
        class_conf = det[5 + class_id]
        score = obj_conf * class_conf

        if score < CONF_THRESH:
            continue

        cx, cy, bw, bh = det[:4]

        x1 = int((cx - bw / 2) * scale_x)
        y1 = int((cy - bh / 2) * scale_y)
        x2 = int((cx + bw / 2) * scale_x)
        y2 = int((cy + bh / 2) * scale_y)

        boxes.append([x1, y1, x2, y2])
        scores.append(score)
        class_ids.append(class_id)

    if not boxes:
        return [], [], []

    keep = nms(np.array(boxes), np.array(scores), IOU_THRESH)
    return [boxes[i] for i in keep], [scores[i] for i in keep], [class_ids[i] for i in keep]

def draw_boxes(frame, boxes, scores, class_ids):
    for box, score, cid in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box
        label = f"{CLASSES[cid]} {score:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    inp = preprocess(frame)
    preds = session.run(None, {input_name: inp})[0][0]

    boxes, scores, class_ids = postprocess(preds, frame.shape[:2])
    draw_boxes(frame, boxes, scores, class_ids)

    cv2.imshow("YOLOv5 ONNX Runtime", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
