import cv2
import numpy as np
import zmq
import tritonclient.grpc as grpcclient
from threading import Thread, Lock
import time
import json
import imagezmq

# --- KONFIGURACJA ADRESÓW ---
ONNX_IP = "10.141.6.26"
TRITON_URL = "10.140.123.226:8001"
TRITON_MODEL = "ensemble_model"  # Zmieniono z 'boundary_detection' na działający 'ensemble_model'
RPI_VPN_IP = "10.141.6.25"


class UnifiedBackend:
    def __init__(self):
        self.running = True
        self.frame = None
        self.lock = Lock()

        # Dane statystyczne
        self.stats = {
            "onnx": {"status": "Disconnected", "latency": 0, "objects": 0},
            "triton": {"status": "Disconnected", "latency": 0, "objects": 0}
        }

        # Wyniki detekcji
        self.results_onnx = []
        self.results_triton = []

        # Inicjalizacja ZMQ (ONNX)
        self.ctx = zmq.Context()
        self.onnx_sock = self.ctx.socket(zmq.REQ)
        self.onnx_sock.setsockopt(zmq.RCVTIMEO, 2000)  # Timeout 2s
        self.onnx_sock.connect(f"tcp://{ONNX_IP}:5555")

        # Inicjalizacja ZMQ (Frontend - PUB)
        # Zakładam, że pub_sock był inicjalizowany gdzieś w Twoim kodzie, dodaję bezpieczną definicję:
        self.pub_sock = self.ctx.socket(zmq.PUB)
        self.pub_sock.bind("tcp://0.0.0.0:5556")  # Port dla frontendu

        # Inicjalizacja Triton
        try:
            self.triton_client = grpcclient.InferenceServerClient(url=TRITON_URL)
            self.stats["triton"]["status"] = "Connected" if self.triton_client.is_server_live() else "Error"
        except Exception:
            self.stats["triton"]["status"] = "Offline"

    def camera_worker_local(self):
        #--- WERSJA Z LOKALNĄ KAMERKĄ LAPTOPA (DO TESTÓW) ---
        print("[*] Inicjalizacja lokalnej kamerki laptopa...")
        cap = cv2.VideoCapture(0)  # 0 to domyślna wbudowana kamera

        if not cap.isOpened():
            print("[-] BŁĄD: Nie można otworzyć lokalnej kamerki.")
            self.stats["onnx"]["status"] = "Cam Error"
            self.stats["triton"]["status"] = "Cam Error"
            return

        while self.running:
            try:
                ret, frame = cap.read()
                if not ret:
                    print("[-] Błąd odczytu klatki z kamerki laptopa.")
                    time.sleep(0.1)
                    continue

                # Zapisujemy klatkę do współdzielonej zmiennej
                with self.lock:
                    self.frame = frame

                # Mały sleep, żeby nie zajechać procesora (ok. 60 FPS max z kamerki)
                time.sleep(0.01)

            except Exception as e:
                print(f"[-] Niespodziewany błąd kamerki: {e}")
                time.sleep(1)

        cap.release()
        print("[*] Zatrzymano camera_worker (Lokalna kamerka).")

    def camera_worker(self):
        # !!! WPISZ TUTAJ ADRES IP MALINKI Z OPENVPN (z interfejsu tun0 Malinki) !!!
        # Na podstawie Twojego zrzutu, jeśli laptop to .45, Malinka pewnie ma coś blisko w klasie 10.141.6.x

        port = 5555

        print(f"[*] [OpenVPN Mode] Łączę się ze strumieniem RPi pod adresem: tcp://{RPI_VPN_IP}:{port}")

        try:
            # Używamy connect_to i podajemy IP Malinki. Backend wykona operację 'connect' do działającej Malinki.
            image_hub = imagezmq.ImageHub(open_port=f'tcp://{RPI_VPN_IP}:{port}', REQ_REP=False)
        except Exception as e:
            print(f"[-] BŁĄD inicjalizacji ImageHub: {e}")
            return

        while self.running:
            try:
                rpi_name, jpg_buffer = image_hub.recv_jpg()
                frame = cv2.imdecode(np.frombuffer(jpg_buffer, dtype='uint8'), cv2.IMREAD_COLOR)

                if frame is not None:
                    with self.lock:
                        self.frame = frame

            except Exception as e:
                print(f"[-] Błąd odbierania klatki z RPi: {e}")
                time.sleep(1)

        print("[*] Zatrzymano camera_worker.")

    def onnx_worker(self):
        print_tracker = 0
        while self.running:
            local_frame = None
            with self.lock:
                if self.frame is not None:
                    local_frame = self.frame.copy()

            if local_frame is not None:
                start = time.perf_counter()
                try:
                    # Kompresja i konwersja na czysty strumień bajtów (bezpieczniejsze dla serwerów ZMQ)
                    _, buf = cv2.imencode('.jpg', local_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    self.onnx_sock.send(buf.tobytes())
                    res = self.onnx_sock.recv_json()

                    # --- DEBUG LOG (Wypisze surowy JSON w konsoli raz na 30 klatek, żeby zweryfikować klucze) ---
                    print_tracker += 1
                    if print_tracker % 30 == 0:
                        print(f"\n[DEBUG ONNX RAW JSON]: {res}")

                    # --- INTELIGENTNE ROZPAKOWYWANIE SŁOWNIKA ---
                    predictions_list = []
                    if isinstance(res, dict):
                        # Szukamy klucza, pod którym serwer ukrywa listę detekcji
                        for key in ["predictions", "boxes", "results", "detections", "output"]:
                            if key in res and isinstance(res[key], list):
                                predictions_list = res[key]
                                break
                        else:
                            # Awaryjnie: bierzemy pierwszą napotkaną listę, pomijając listę nazw klas 'names'
                            for k, val in res.items():
                                if k != "names" and isinstance(val, list):
                                    predictions_list = val
                                    break
                    elif isinstance(res, list):
                        predictions_list = res

                    # Odsiewamy opisy tekstowe klas, zostawiamy wyłącznie czyste dane obiektów
                    clean_predictions = []
                    for item in predictions_list:
                        if isinstance(item, str) and ("names" in item or "classes" in item):
                            continue
                        clean_predictions.append(item)

                    # Zapisujemy wyczyszczoną listę do rysowania i aktualizujemy HUD
                    self.results_onnx = clean_predictions
                    self.stats["onnx"]["latency"] = (time.perf_counter() - start) * 1000
                    self.stats["onnx"]["status"] = "Online"
                    self.stats["onnx"]["objects"] = len(clean_predictions)  # Teraz to prawdziwa liczba obiektów!

                except Exception as e:
                    self.stats["onnx"]["status"] = "Timeout/Err"
            time.sleep(0.01)

    def post_process_triton(self, predictions, orig_shape):
        h_orig, w_orig = orig_shape
        detections = predictions[0]
        boxes, confs, class_ids = [], [], []

        sw, sh = w_orig / 640, h_orig / 640

        for det in detections:
            obj_conf = det[4]
            if obj_conf > 0.15:  # Threshold detekcji
                class_scores = det[5:]
                class_id = np.argmax(class_scores)
                final_conf = obj_conf * class_scores[class_id]

                if final_conf > 0.15:
                    cx, cy, w, h = det[:4]
                    boxes.append([
                        int((cx - w / 2) * sw),
                        int((cy - h / 2) * sh),
                        int(w * sw),
                        int(h * sh)
                    ])
                    confs.append(float(final_conf))
                    class_ids.append(int(class_id))

        indices = cv2.dnn.NMSBoxes(boxes, confs, 0.15, 0.45)
        final = []
        if len(indices) > 0:
            for i in indices.flatten():
                final.append({"box": boxes[i], "conf": confs[i], "class": class_ids[i]})
        return final

    def triton_worker(self):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]

        while self.running:
            local_frame = None
            with self.lock:
                if self.frame is not None:
                    local_frame = self.frame.copy()

            if local_frame is not None:
                start = time.perf_counter()
                try:
                    # 1. Zmiana rozmiaru do 640x640 (tak jak w działającym teście)
                    if local_frame.shape[:2] != (640, 640):
                        local_frame = cv2.resize(local_frame, (640, 640))

                    # 2. Kompresja do JPEG (Triton Ensemble oczekuje surowych bajtów pliku)
                    ret, buffer = cv2.imencode('.jpg', local_frame, encode_param)
                    if not ret:
                        continue

                    jpeg_bytes = buffer.tobytes()

                    # 3. Przygotowanie wejścia gRPC zgodnie z konfiguracją modelu ensemble
                    infer_input = grpcclient.InferInput("input_image", [1], "BYTES")
                    infer_input.set_data_from_numpy(np.array([jpeg_bytes], dtype=object))

                    # 4. Wywołanie inferencji na serwerze Triton
                    res = self.triton_client.infer(
                        model_name=TRITON_MODEL,
                        inputs=[infer_input],
                        outputs=[grpcclient.InferRequestedOutput("object_boundaries")]
                    )

                    # 5. Odbiór i postprocessing danych wyjściowych
                    raw_preds = res.as_numpy("object_boundaries")
                    processed_results = self.post_process_triton(raw_preds, (640, 640))

                    # 6. Aktualizacja struktur danych i statystyk
                    self.results_triton = processed_results
                    self.stats["triton"]["latency"] = (time.perf_counter() - start) * 1000
                    self.stats["triton"]["status"] = "Online"
                    self.stats["triton"]["objects"] = len(processed_results)

                except Exception as e:
                    self.stats["triton"]["status"] = "Err"
                    # Opcjonalnie: odkomentuj poniższe, jeśli chcesz debugować konkretny błąd w konsoli:
                    # print(f"[Triton Worker Error]: {e}")

            time.sleep(0.01)

    def stats_printer(self):
        while self.running:
            o = self.stats["onnx"]
            t = self.stats["triton"]
            line = (f"\rONNX: [{o['status']}] {o['latency']:4.0f}ms, Obj: {o['objects']} | "
                    f"TRITON: [{t['status']}] {t['latency']:4.0f}ms, Obj: {t['objects']}   ")
            print(line, end="", flush=True)
            time.sleep(0.1)

    def run(self):
        Thread(target=self.camera_worker, daemon=True).start()
        Thread(target=self.onnx_worker, daemon=True).start()
        Thread(target=self.triton_worker, daemon=True).start()
        Thread(target=self.stats_printer, daemon=True).start()

        print("\n[+] Backend uruchomiony w trybie HEADLESS. Czekam na połączenie z frontendu...")

        while self.running:
            try:
                with self.lock:
                    if self.frame is None:
                        time.sleep(0.01)
                        continue
                    clean_frame = self.frame.copy()

                # Kompresja klatki do wysyłki na frontend
                _, buffer = cv2.imencode('.jpg', clean_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

                # Pakiet metadanych
                metadata = {
                    "onnx": self.results_onnx,
                    "triton": self.results_triton,
                    "stats": self.stats
                }

                # Wysyłka streamu przez gniazdo PUB
                self.pub_sock.send_multipart([
                    b"ai_stream",
                    json.dumps(metadata).encode('utf-8'),
                    buffer.tobytes()
                ])

                time.sleep(0.03)  # Limit ok. 30 FPS
            except KeyboardInterrupt:
                self.running = False
                break

    def run_local(self):
        """Tryb DEBUG: Uruchamia wątki i otwiera lokalne okno z podglądem wideo i metadanymi."""
        Thread(target=self.camera_worker_local, daemon=True).start()
        Thread(target=self.onnx_worker, daemon=True).start()
        Thread(target=self.triton_worker, daemon=True).start()
        Thread(target=self.stats_printer, daemon=True).start()

        print("\n[+] Backend uruchomiony w trybie LOKALNEGO PODGLĄDU (Debug).")
        print("[*] Wciśnij 'q' w oknie wideo, aby zamknąć aplikację.")

        while self.running:
            with self.lock:
                if self.frame is None:
                    time.sleep(0.01)
                    continue
                display_frame = self.frame.copy()

            h, w = display_frame.shape[:2]

            # 1. RYSOWANIE WYNIKÓW Z ONNX (Zielone ramki)
            current_onnx = list(self.results_onnx)
            for p in current_onnx:
                try:
                    # OBSŁUGA FORMATU: Słownik {'box': [x1,y1,x2,y2], 'conf':..., 'class':...}
                    if isinstance(p, dict) and 'box' in p:
                        box = p['box']
                        class_id = p.get('class', 0)
                        conf = p.get('conf', 0.0)

                        # Sprawdzamy czy dane to ułamki 0.0 - 1.0 (znormalizowane)
                        is_normalized = all(float(x) <= 1.05 for x in box)

                        # Sprawdzamy czy to format narożników [x1, y1, x2, y2] czy środka [cx, cy, nw, nh]
                        if box[2] >= box[0] and box[3] >= box[1]:
                            x1 = int(box[0] * w if is_normalized else box[0])
                            y1 = int(box[1] * h if is_normalized else box[1])
                            x2 = int(box[2] * w if is_normalized else box[2])
                            y2 = int(box[3] * h if is_normalized else box[3])
                        else:
                            # Format YOLO: [cx, cy, nw, nh]
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
                                y2 = int(cy + nh / 2)

                        # Rysowanie na ekranie
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(display_frame, f"ONNX ID {class_id} ({conf:.2f})", (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        continue

                    # Alternatywny fallback dla czystych list/stringów (na wszelki wypadek)
                    if isinstance(p, str):
                        parts = [float(x) for x in p.strip().split()]
                    elif isinstance(p, (list, tuple)):
                        parts = [float(x) for x in p]
                    else:
                        continue

                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        cx, cy, nw, nh = parts[1:5]
                        x1 = int((cx - nw / 2) * w) if cx <= 1.05 else int(cx - nw / 2)
                        y1 = int((cy - nh / 2) * h) if cy <= 1.05 else int(cy - nh / 2)
                        x2 = int((cx + nw / 2) * w) if nw <= 1.05 else int(cx + nw / 2)
                        y2 = int((cy + nh / 2) * h) if nh <= 1.05 else int(cy + nh / 2)

                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(display_frame, f"ONNX ID: {class_id}", (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                except Exception as e:
                    print(f"\n[-] Błąd rysowania ONNX: {e} | Dane: {p}")

            # 2. RYSOWANIE WYNIKÓW Z TRITONA (Niebieskie ramki)
            current_triton = list(self.results_triton)
            for p in current_triton:
                try:
                    x, y, w_box, h_box = map(int, p['box'][:4])
                    label = f"Triton: {p['conf']:.2f}"
                    cv2.rectangle(display_frame, (x, y), (x + w_box, y + h_box), (255, 0, 0), 2)
                    cv2.putText(display_frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                except Exception:
                    pass

            # 3. NAKŁADANIE STATYSTYK (HUD)
            o = self.stats["onnx"]
            t = self.stats["triton"]
            cv2.putText(display_frame, f"ONNX: {o['status']} ({o['latency']:.0f}ms)", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(display_frame, f"TRITON: {t['status']} ({t['latency']:.0f}ms)", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            cv2.imshow("Multi-Backend Debug View", display_frame)

            if cv2.waitKey(15) & 0xFF == ord('q'):
                self.running = False
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    hub = UnifiedBackend()
    hub.run_local()