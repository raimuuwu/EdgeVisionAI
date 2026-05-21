import zmq
import cv2
import numpy as np
import json

# --- KONFIGURACJA ---
BACKEND_IP = "127.0.0.1" # Zmień na IP maszyny z backendem, jeśli to inny komputer

def start_frontend():
    print(f"[*] Łączenie z backendem {BACKEND_IP}...")
    
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(f"tcp://{BACKEND_IP}:5556")
    
    # Subskrybujemy wiadomości wideo
    socket.setsockopt(zmq.SUBSCRIBE, b"ai_stream")
    
    print("[+] Połączono! Oczekiwanie na strumień wideo... (Naciśnij 'q' aby wyjść)")
    cv2.namedWindow("AI Dashboard - CamOverIP", cv2.WINDOW_NORMAL)

    while True:
        try:
            # Odbiór paczki z backendu
            topic, metadata_bytes, frame_bytes = socket.recv_multipart()
            
            # Dekodowanie
            metadata = json.loads(metadata_bytes.decode('utf-8'))
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            h, w = frame.shape[:2]

            # Rysowanie paska statystyk
            stats = metadata.get("stats", {})
            onnx_lat = stats.get("onnx", {}).get("latency", 0)
            triton_lat = stats.get("triton", {}).get("latency", 0)
            
            stats_text = f"ONNX: {onnx_lat:.0f}ms | Triton: {triton_lat:.0f}ms | Opoznienie Sieci: ZMQ"
            cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
            cv2.putText(frame, stats_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Rysowanie ONNX (Zielone)
            for p in metadata.get("onnx", []):
                nx, ny, nw, nh = p['box']
                x1, y1 = int((nx-nw/2)*w), int((ny-nh/2)*h)
                x2, y2 = int((nx+nw/2)*w), int((ny+nh/2)*h)
                
                label = f"ONNX {int(p['conf']*100)}%"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Rysowanie Triton (Niebieskie)
            for p in metadata.get("triton", []):
                x, y, w_box, h_box = p['box']
                
                label = f"TRITON {int(p['conf']*100)}%"
                cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (255, 0, 0), 2)
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Wyświetlanie
            cv2.imshow("AI Dashboard - CamOverIP", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except KeyboardInterrupt:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_frontend()