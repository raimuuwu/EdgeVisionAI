import cv2
import zmq
import numpy as np
import time

# --- KONFIGURACJA ---
SERVER_IP = "10.8.0.5"  # IP Twojego Jetsona w VPN
PORT = "5555"
CONF_THRESH = 0.25

context = zmq.Context()
print(f"[*] Łączenie z serwerem AI: {SERVER_IP}:{PORT}...")
socket = context.socket(zmq.REQ)  # REQ - Request
socket.connect(f"tcp://{SERVER_IP}:{PORT}")

cap = cv2.VideoCapture(0)


def preprocess(frame):
    # Resize do formatu YOLO i konwersja na float32
    img = cv2.resize(frame, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    return np.expand_dims(img, axis=0)  # Add batch dim


while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. Przygotowanie klatki
    input_tensor = preprocess(frame)

    # 2. Wysyłka do serwera
    start_t = time.time()
    socket.send(input_tensor.tobytes())

    # 3. Oczekiwanie na odpowiedź (predykcje)
    raw_res = socket.recv()
    predictions = np.frombuffer(raw_res, dtype=np.float32).reshape(1, 25200, 85)

    latency = (time.time() - start_t) * 1000

    # 4. Prosty Post-processing i Rysowanie
    # (Dla czytelności kodu uproszczony - bierzemy tylko klatki z wysokim confidence)
    for det in predictions[0]:
        if det[4] > CONF_THRESH:
            # Skalowanie współrzędnych z 640 na rozmiar okna
            h, w = frame.shape[:2]
            cx, cy, nw, nh = det[:4]
            x1 = int((cx - nw / 2) * (w / 640))
            y1 = int((cy - nh / 2) * (h / 640))
            cv2.rectangle(frame, (x1, y1), (x1 + int(nw * w / 640), y1 + int(nh * h / 640)), (0, 255, 0), 2)

    cv2.putText(frame, f"RTT: {latency:.0f}ms", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Custom AI Cloud", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()