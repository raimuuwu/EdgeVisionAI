import cv2
import numpy as np
import zmq
import json
from threading import Thread
import time

# --- KONFIGURACJA POŁĄCZENIA ---
# Wpisz adres IP komputera, na którym uruchomiony jest plik backend.py.
# Jeśli uruchamiasz backend i frontend na tej samej maszynie, zostaw "127.0.0.1".
BACKEND_IP = "10.141.6.29" 

print("[*] Inicjalizacja Frontendu (Klienta AI)...")
ctx = zmq.Context()

# 1. Konfiguracja gniazda odbiorczego (SUB) dla strumienia wideo i metadanych
sub_sock = ctx.socket(zmq.SUB)
sub_sock.connect(f"tcp://{BACKEND_IP}:5556")
# KLUCZOWE: Musimy zasubskrybować konkretny temat (Topic) wysyłany przez backend
sub_sock.setsockopt(zmq.SUBSCRIBE, b"ai_stream")

# 2. Konfiguracja gniazda testowego (REQ) do wysyłania pingów do backendu
ping_sock = ctx.socket(zmq.REQ)
ping_sock.setsockopt(zmq.RCVTIMEO, 2000)  # Timeout 2 sekundy na odpowiedź
ping_sock.connect(f"tcp://{BACKEND_IP}:5557")


def ping_tester_worker():
    """Wątek wysyłający automatyczny test komunikacji (PING) co 10 sekund"""
    print("[*] Wątek testowy PING uruchomiony.")
    while True:
        try:
            time.sleep(10)
            print("\n[*] [Test Łącza] Wysyłam PING do backendu...")
            ping_sock.send_string("Cześć! Słyszysz mój frontend?")
            
            # Oczekiwanie na odpowiedź PONG od backendu
            reply = ping_sock.recv_string()
            print(f"[+] [Test Łącza] ODPOWIEDŹ Z BACKENDU: {reply}")
        except zmq.error.Again:
            print("[-] [Test Łącza] BŁĄD: Brak odpowiedzi od backendu na porcie 5557 (Timeout).")
        except Exception as e:
            print(f"[-] [Test Łącza] Niespodziewany błąd komunikacji: {e}")


# Uruchomienie wątku ping w tle, żeby nie blokował wyświetlania wideo
Thread(target=ping_tester_worker, daemon=True).start()

print(f"[+] Pomyślnie połączono z backendem pod adresem sieciowym: {BACKEND_IP}")
print("[*] Oczekiwanie na pierwszy pakiet danych... (Wciśnij 'q' w oknie, aby zamknąć)")

while True:
    try:
        # Odbieranie wieloczęściowego pakietu ZMQ PUB/SUB
        topic, metadata_json, frame_bytes = sub_sock.recv_multipart()
        
        # Dekodowanie przesłanych metadanych tekstowych JSON
        metadata = json.loads(metadata_json.decode('utf-8'))
        results_onnx = metadata.get("onnx", [])
        results_triton = metadata.get("triton", [])
        stats = metadata.get("stats", {})
        
        # Rekonstrukcja klatki obrazu z surowych bajtów skompresowanych do JPEG
        nparr = np.frombuffer(frame_bytes, dtype=np.uint8)
        display_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if display_frame is None:
            continue
            
        h, w = display_frame.shape[:2]
        
        # ====================================================================
        # 1. RYSOWANIE WYNIKÓW DETEKCJI Z MODELU ONNX (Zielone ramki)
        # ====================================================================
        for p in results_onnx:
            try:
                if isinstance(p, dict) and 'box' in p:
                    box = p['box']
                    class_id = p.get('class', 0)
                    conf = p.get('conf', 0.0)
                    
                    # Sprawdzenie czy współrzędne są znormalizowane (ułamki 0.0 - 1.0)
                    is_normalized = all(float(x) <= 1.05 for x in box)
                    
                    # Format narożników: [x1, y1, x2, y2]
                    if box[2] >= box[0] and box[3] >= box[1]:
                        x1 = int(box[0] * w if is_normalized else box[0])
                        y1 = int(box[1] * h if is_normalized else box[1])
                        x2 = int(box[2] * w if is_normalized else box[2])
                        y2 = int(box[3] * h if is_normalized else box[3])
                    else:
                        # Format środka: [cx, cy, nw, nh]
                        cx, cy, nw, nh = box
                        if is_normalized:
                            x1 = int((cx - nw / 2) * w)
                            y1 = int((cy - nh / 2) * h)
                            x2 = int((cx + nw / 2) * w)
                            y2 = int((cy + nh / 2) * h)
                        else:
                            x1 = int(cx - nw / 2)
                            y1 = int(cy - nh / 2)
                            x2 = int(cx + nw / 2)
                            y2 = int(cx + nh / 2)
                            
                    # Rysowanie obiektów ONNX na zielono
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"ONNX ID {class_id} ({conf:.2f})"
                    cv2.putText(display_frame, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            except Exception as e:
                print(f"[-] Błąd rysowania danych ONNX: {e}")

        # ====================================================================
        # 2. RYSOWANIE WYNIKÓW DETEKCJI Z SERWERA TRITON (Niebieskie ramki)
        # ====================================================================
        for p in results_triton:
            try:
                # Triton przesyła bezpośrednie piksele w formacie [x, y, szerokość, wysokość]
                x, y, w_box, h_box = map(int, p['box'][:4])
                conf = p.get('conf', 0.0)
                class_id = p.get('class', 0)
                
                # Rysowanie obiektów Triton na niebiesko
                cv2.rectangle(display_frame, (x, y), (x + w_box, y + h_box), (255, 0, 0), 2)
                label = f"Triton ID {class_id} ({conf:.2f})"
                cv2.putText(display_frame, label, (x, y - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            except Exception:
                pass

        # ====================================================================
        # 3. WYŚWIETLANIE STATYSTYK WYDAJNOŚCI (HUD na obrazie)
        # ====================================================================
        o = stats.get("onnx", {"status": "Disconnected", "latency": 0})
        t = stats.get("triton", {"status": "Disconnected", "latency": 0})
        
        cv2.putText(display_frame, f"ONNX: {o['status']} ({o['latency']:.0f}ms)", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(display_frame, f"TRITON: {t['status']} ({t['latency']:.0f}ms)", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Wywołanie graficznego okna OpenCV systemu operacyjnego
        cv2.imshow("System Monitoringu AI - Widok Klienta", display_frame)
        
        # Odświeżanie okna; klawisz 'q' przerywa aplikację
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[*] Zamykanie aplikacji przez użytkownika.")
            break
            
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"[-] Błąd pętli głównej frontendu: {e}")
        time.sleep(0.1)

cv2.destroyAllWindows()
print("[+] Frontend wyłączony.")