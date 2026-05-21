import cv2
import numpy as np
import zmq
import tritonclient.grpc as grpcclient
from threading import Thread, Lock
import time
import json

# --- KONFIGURACJA ADRESÓW ---
ONNX_IP = "10.141.6.24"
TRITON_URL = "10.140.123.226:8001"
TRITON_MODEL = "boundary_detection"

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

        # --- ZMQ: ONNX CLIENT ---
        self.ctx = zmq.Context()
        self.onnx_sock = self.ctx.socket(zmq.REQ)
        self.onnx_sock.setsockopt(zmq.RCVTIMEO, 2000) # Timeout 2s
        self.onnx_sock.connect(f"tcp://{ONNX_IP}:5555")

        # --- ZMQ: FRONTEND PUBLISHER ---
        self.pub_sock = self.ctx.socket(zmq.PUB)
        self.pub_sock.bind("tcp://0.0.0.0:5556")
        print("[*] Gniazdo streamingu (PUB) otwarte na porcie 5556")

        # --- TRITON CLIENT ---
        try:
            self.triton_client = grpcclient.InferenceServerClient(url=TRITON_URL)
            self.stats["triton"]["status"] = "Connected" if self.triton_client.is_server_live() else "Error"
        except Exception as e:
            self.stats["triton"]["status"] = "Offline"
            print(f"[-] Błąd Triton: {e}")

    def camera_worker(self):
        cap = cv2.VideoCapture(0)
        while self.running:
            ret, f = cap.read()
            if ret:
                with self.lock:
                    self.frame = f
            time.sleep(0.01)
        cap.release()

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
            if obj_conf > 0.15:
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
                    img_rgb = cv2.cvtColor(local_frame, cv2.COLOR_BGR2RGB)
                    img_resized = cv2.resize(img_rgb, (640, 640))
                    img_input = np.transpose(img_resized.astype(np.float32) / 255.0, (2, 0, 1))[np.newaxis, :]
                    
                    inputs = [grpcclient.InferInput("images", img_input.shape, "FP32")]
                    inputs[0].set_data_from_numpy(img_input)
                    outputs = [grpcclient.InferRequestedOutput("output0")]
                    
                    res = self.triton_client.infer(model_name=TRITON_MODEL, inputs=inputs, outputs=outputs)
                    raw_preds = res.as_numpy("output0")
                    
                    processed_results = self.post_process_triton(raw_preds, local_frame.shape[:2])
                    
                    self.results_triton = processed_results
                    self.stats["triton"]["latency"] = (time.perf_counter() - start) * 1000
                    self.stats["triton"]["status"] = "Online"
                    self.stats["triton"]["objects"] = len(processed_results)
                except Exception as e:
                    self.stats["triton"]["status"] = "Offline/Err"
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

                # Kompresja klatki
                _, buffer = cv2.imencode('.jpg', clean_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                
                # Pakiet metadanych
                metadata = {
                    "onnx": self.results_onnx,
                    "triton": self.results_triton,
                    "stats": self.stats
                }
                
                # Wysyłka streamu
                self.pub_sock.send_multipart([
                    b"ai_stream", 
                    json.dumps(metadata).encode('utf-8'), 
                    buffer.tobytes()
                ])
                
                time.sleep(0.03) # Limit ok. 30 FPS
            except KeyboardInterrupt:
                self.running = False
                break

if __name__ == "__main__":
    hub = UnifiedBackend()
    hub.run()