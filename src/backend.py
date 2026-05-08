import cv2
import numpy as np
import zmq
import tritonclient.grpc as grpcclient
from threading import Thread, Lock
import time
import json
import os
import imagezmq

# --- KONFIGURACJA ADRESÓW ---
ONNX_IP = "10.141.6.34"
TRITON_URL = "10.140.123.226:8001"
TRITON_MODEL = "boundary_detection"
RASSBERY_IP = "malinkaedgevision"


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

        # Inicjalizacja Triton
        try:
            self.triton_client = grpcclient.InferenceServerClient(url=TRITON_URL)
            self.stats["triton"]["status"] = "Connected" if self.triton_client.is_server_live() else "Error"
        except:
            self.stats["triton"]["status"] = "Offline"

    def _send_to_frontend_placeholder(self, frame, data):
        pass

    def camera_worker(self):
        rpi_ip = RASSBERY_IP
        port = 5555

        print(f"[*] Łączenie ze strumieniem RPi: tcp://{rpi_ip}:{port}")

        image_hub = imagezmq.ImageHub(open_port=f'tcp://{rpi_ip}:{port}', REQ_REP=False)

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
        while self.running:
            local_frame = None
            with self.lock:
                if self.frame is not None:
                    local_frame = self.frame.copy()

            if local_frame is not None:
                start = time.perf_counter()
                try:
                    _, buf = cv2.imencode('.jpg', local_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    self.onnx_sock.send(buf)
                    res = self.onnx_sock.recv_json()

                    self.results_onnx = res
                    self.stats["onnx"]["latency"] = (time.perf_counter() - start) * 1000
                    self.stats["onnx"]["status"] = "Online"
                    self.stats["onnx"]["objects"] = len(res)
                except Exception:
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
        while self.running:
            local_frame = None
            with self.lock:
                if self.frame is not None:
                    local_frame = self.frame.copy()

            if local_frame is not None:
                start = time.perf_counter()
                try:
                    #Preprocessing
                    img_rgb = cv2.cvtColor(local_frame, cv2.COLOR_BGR2RGB)
                    img_resized = cv2.resize(img_rgb, (640, 640))
                    img_input = np.transpose(img_resized.astype(np.float32) / 255.0, (2, 0, 1))[np.newaxis, :]

                    inputs = [grpcclient.InferInput("images", img_input.shape, "FP32")]
                    inputs[0].set_data_from_numpy(img_input)
                    outputs = [grpcclient.InferRequestedOutput("output0")]

                    res = self.triton_client.infer(model_name=TRITON_MODEL, inputs=inputs, outputs=outputs)
                    raw_preds = res.as_numpy("output0")

                    #Postprocessing
                    processed_results = self.post_process_triton(raw_preds, local_frame.shape[:2])

                    self.results_triton = processed_results
                    self.stats["triton"]["latency"] = (time.perf_counter() - start) * 1000
                    self.stats["triton"]["status"] = "Online"
                    self.stats["triton"]["objects"] = len(processed_results)
                except Exception as e:
                    self.stats["triton"]["status"] = f"Err: {str(e)[:10]}"
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
        # Start wątków
        Thread(target=self.camera_worker, daemon=True).start()
        Thread(target=self.onnx_worker, daemon=True).start()
        Thread(target=self.triton_worker, daemon=True).start()
        Thread(target=self.stats_printer, daemon=True).start()

        print("[+] Statystyki połączenia:")

        while True:
            with self.lock:
                if self.frame is None: continue
                display_frame = self.frame.copy()

            #RYSOWANIE ONNX (Zielone)
            for p in self.results_onnx:
                h, w = display_frame.shape[:2]
                nx, ny, nw, nh = p['box']
                x1, y1 = int((nx - nw / 2) * w), int((ny - nh / 2) * h)
                x2, y2 = int((nx + nw / 2) * w), int((ny + nh / 2) * h)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            #RYSOWANIE TRITON (Niebieskie)
            for p in self.results_triton:
                x, y, w_box, h_box = p['box']
                label = f"Triton: {p['conf']:.2f}"
                cv2.rectangle(display_frame, (x, y), (x + w_box, y + h_box), (255, 0, 0), 2)
                cv2.putText(display_frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            self._send_to_frontend_placeholder(display_frame, {"onnx": self.results_onnx})

            cv2.imshow("Multi-Backend Unified View", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break


if __name__ == "__main__":
    hub = UnifiedBackend()
    hub.run()