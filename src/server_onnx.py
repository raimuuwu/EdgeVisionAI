import zmq
import numpy as np
import onnxruntime as ort

# --- KONFIGURACJA ---
MODEL_PATH = "yolov5s.onnx"
PORT = "5555"

print("[*] Ładowanie modelu na GPU...")
# Używamy CUDAExecutionProvider dla przyspieszenia na Jetsonie
session = ort.InferenceSession(MODEL_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

context = zmq.Context()
socket = context.socket(zmq.REP)  # REP - Reply
socket.bind(f"tcp://*:{PORT}")

print(f"[+] Serwer AI gotowy! Słucham na porcie {PORT}...")

while True:
    # 1. Odbierz klatkę (jako surowe bajty)
    message = socket.recv()

    # 2. Rekonstrukcja tablicy numpy (model oczekuje 1x3x640x640)
    data = np.frombuffer(message, dtype=np.float32).reshape(1, 3, 640, 640)

    # 3. Inferencja na GPU
    outputs = session.run(None, {input_name: data})

    # 4. Wysyłamy surowe wyniki (output0) z powrotem
    # Wynik to zazwyczaj macierz [1, 25200, 85]
    socket.send(outputs[0].tobytes())