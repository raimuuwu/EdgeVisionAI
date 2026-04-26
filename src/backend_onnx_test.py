import zmq
import cv2
import numpy as np
import time

# --- KONFIGURACJA ---
SERVER_IP = "10.141.6.24"  # Zmień jeśli serwer jest na innym kompie
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect(f"tcp://{SERVER_IP}:5555")

cap = cv2.VideoCapture(0)  # 0 to domyślna kamerka

print("[+] Klient uruchomiony. Naciśnij 'q' aby wyjść.")

while True:
    ret, frame = cap.read()
    if not ret: break

    h, w, _ = frame.shape

    # --- POMIAR CZASU START ---
    start_time = time.perf_counter()

    # 1. Wysyłka
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    socket.send(buffer)

    # 2. Odbiór
    predictions = socket.recv_json()

    # --- POMIAR CZASU KONIEC ---
    latency = (time.perf_counter() - start_time) * 1000  # Wynik w milisekundach

    # 3. Rysowanie statystyk połączenia
    stats_text = f"IP: {SERVER_IP} | Latency: {latency:.1f}ms | Objects: {len(predictions)}"
    # Czarny pasek tła dla czytelności statystyk
    cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.putText(frame, stats_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 4. Rysowanie ramek
    for pred in predictions:
        nx, ny, nw, nh = pred['box']

        x1 = int((nx - nw / 2) * w)
        y1 = int((ny - nh / 2) * h)
        x2 = int((nx + nw / 2) * w)
        y2 = int((ny + nh / 2) * h)

        label = f"Obj: {pred['class']} {int(pred['conf'] * 100)}%"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLO Engine - NVIDIA GPU", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()