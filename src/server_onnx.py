import zmq
import cv2
import numpy as np
import onnxruntime as ort
import json

# --- KONFIGURACJA ---
MODEL_PATH = "best.onnx"
providers = [('CUDAExecutionProvider', {'device_id': 0}), 'CPUExecutionProvider']

print("[*] Inicjalizacja sesji ONNX (Dla architektury YOLOv8)...")
session = ort.InferenceSession(MODEL_PATH, providers=providers)

current_provider = session.get_providers()[0]
print(f"[+] AKTYWNY SILNIK: {current_provider}")
if current_provider == 'CPUExecutionProvider':
    print(" [!] UWAGA: Model działa na procesorze (CPU)!")
else:
    print(f" [OK] Model śmiga na GPU: {ort.get_device()}")

input_name = session.get_inputs()[0].name

# --- ZMQ ---
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://0.0.0.0:5555")


def preprocess(img):
    img = cv2.resize(img, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)


print("[+] Serwer gotowy. Czekam na dane od klienta...")

while True:
    msg = socket.recv()
    nparr = np.frombuffer(msg, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is not None:
        input_tensor = preprocess(frame)
        outputs = session.run(None, {input_name: input_tensor})

        # --- POPRAWNY PARSER DLA YOLOv8 [1, 6, 8400] ---
        # Usuwamy wymiar batch -> otrzymujemy [6, 8400]
        raw_output = outputs[0][0]
        # Transponujemy macierz, aby uzyskać kształt [8400, 6] (8400 obiektów, każdy ma 6 cech)
        raw_output = np.transpose(raw_output, (1, 0))

        # Podział na współrzędne [cx, cy, w, h] oraz kolumny klas [score_klasa_0, score_klasa_1]
        bboxes = raw_output[:, :4]
        class_scores = raw_output[:, 4:]

        # Matematyczne wyciągnięcie najlepszych klas i ich pewności
        class_ids = np.argmax(class_scores, axis=1)
        confs = np.max(class_scores, axis=1)

        # Odfiltrowanie śmieci na podstawie progu pewności (0.6)
        conf_threshold = 0.6
        mask = confs > conf_threshold

        filtered_boxes = bboxes[mask]
        filtered_confs = confs[mask]
        filtered_class_ids = class_ids[mask]

        # Konwersja formatu YOLO [cx, cy, w, h] do formatu OpenCV [x, y, w, h] potrzebnego do NMS
        nms_boxes = []
        for box in filtered_boxes:
            cx, cy, w, h = box
            x = cx - w / 2
            y = cy - h / 2
            nms_boxes.append([int(x), int(y), int(w), int(h)])

        # Uruchomienie Non-Maximum Suppression (NMS) zapobiegającego powielaniu ramek
        indices = cv2.dnn.NMSBoxes(nms_boxes, filtered_confs.tolist(), conf_threshold, 0.45)

        predictions = []
        if len(indices) > 0:
            for i in indices.flatten():
                box = filtered_boxes[i]
                conf = filtered_confs[i]
                class_id = filtered_class_ids[i]

                # Konwersja do znormalizowanego formatu narożników [x1, y1, x2, y2] (zakres 0.0 - 1.0)
                # Gwarantuje to pełną stabilność i odporność na błędy rysowania w backend.py
                cx, cy, nw, nh = box
                x1 = (cx - nw / 2) / 640.0
                y1 = (cy - nh / 2) / 640.0
                x2 = (cx + nw / 2) / 640.0
                y2 = (cy + nh / 2) / 640.0

                predictions.append({
                    "box": [float(x1), float(y1), float(x2), float(y2)],
                    "conf": float(conf),
                    "class": int(class_id)
                })

        socket.send_json(predictions)
    else:
        socket.send_json([])