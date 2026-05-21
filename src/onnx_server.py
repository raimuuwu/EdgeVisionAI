import zmq
import cv2
import numpy as np
import onnxruntime as ort
import json

# --- KONFIGURACJA ---
MODEL_PATH = "yolov5s.onnx"
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

print("[*] Inicjalizacja sesji ONNX...")
try:
    session = ort.InferenceSession(MODEL_PATH, providers=providers)
    active_providers = session.get_providers()
    print(f"[+] Aktywne providery: {active_providers}")
except Exception as e:
    print(f"[-] KRYTYCZNY BŁĄD ładowania: {e}")
    exit()

input_name = session.get_inputs()[0].name

# --- ZMQ SETUP ---
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://0.0.0.0:5555")

def preprocess(img):
    img = cv2.resize(img, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)

print("[+] Serwer ONNX gotowy do pracy i czeka na porcie 5555!")

while True:
    try:
        message = socket.recv()
        nparr = np.frombuffer(message, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is not None:
            input_tensor = preprocess(frame)
            outputs = session.run(None, {input_name: input_tensor})
            raw_output = outputs[0]

            mask = raw_output[0, :, 4] > 0.6 
            hits = raw_output[0, mask]
            
            formatted_results = []
            for hit in hits:
                conf = float(hit[4])
                class_id = int(np.argmax(hit[5:]))
                if hit[5 + class_id] * conf < 0.5: 
                    continue
                
                box = hit[:4].tolist() 
                formatted_results.append({
                    "box": [box[0]/640, box[1]/640, box[2]/640, box[3]/640],
                    "conf": conf,
                    "class": class_id
                })
            
            socket.send_string(json.dumps(formatted_results))
        else:
            socket.send_string(json.dumps([]))
            
    except Exception as e:
        print(f"[-] Błąd pętli: {e}")
        socket.send_string(json.dumps([]))