import zmq
import cv2
import numpy as np
import onnxruntime as ort
import json

# --- KONFIGURACJA GPU ---
MODEL_PATH = "yolov5s.onnx"
providers = [('CUDAExecutionProvider', {'device_id': 0}), 'CPUExecutionProvider']

print("[*] Inicjalizacja sesji ONNX...")
session = ort.InferenceSession(MODEL_PATH, providers=providers)

# TEST GPU: To ostatecznie potwierdzi, czego używamy
current_provider = session.get_providers()[0]
print(f"[+] AKTYWNY SILNIK: {current_provider}")
if current_provider == 'CPUExecutionProvider':
    print(" [!] UWAGA: Model działa na procesorze (CPU)!")
else:
    print(f" [OK] Model śmiga na: {ort.get_device()}")

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
        raw_output = outputs[0]

        # --- FILTROWANIE (Zmieniony próg na 0.6 i wyższa precyzja) ---
        conf_threshold = 0.6  # Podnieśliśmy z 0.4, żeby było mniej błędnych detekcji
        mask = raw_output[0, :, 4] > conf_threshold
        hits = raw_output[0, mask]

        predictions = []
        for hit in hits:
            conf = float(hit[4])
            class_id = int(np.argmax(hit[5:]))

            # Dodatkowy filtr: pewność klasy musi być też wysoka
            if hit[5 + class_id] * conf < 0.5:
                continue

            box = hit[:4].tolist()
            predictions.append({
                "box": [box[0] / 640, box[1] / 640, box[2] / 640, box[3] / 640],
                "conf": conf,
                "class": class_id
            })

        socket.send_json(predictions)
    else:
        socket.send_json([])