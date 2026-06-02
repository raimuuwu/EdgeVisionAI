import cv2
import numpy as np
import tritonclient.grpc as grpcclient
from threading import Thread, Lock
import time

# --- KONFIGURACJA ---
URL          = "10.140.123.226:8001"
MODEL_NAME   = "ensemble_model"
CONF_THRESH  = 0.25  # Podniosłem z 0.05 na 0.25 (optymalne dla YOLOv8)
NMS_THRESH   = 0.45

TARGET_INFERENCE_FPS = 30
JPEG_QUALITY     = 80
SEND_WIDTH       = 640
SEND_HEIGHT      = 640
# =====================================================

class TritonStreamer:
    def __init__(self):
        print(f"[*] Ladowanie klienta Triton: {URL}...")
        try:
            self.client = grpcclient.InferenceServerClient(url=URL)
            if not self.client.is_server_live():
                print("[-] BLAD: Serwer nieosiagalny.")
            if not self.client.is_model_ready(MODEL_NAME):
                print(f"[-] BLAD: Model '{MODEL_NAME}' nie jest zaladowany.")
            else:
                print(f"[+] SUKCES: Model '{MODEL_NAME}' gotowy.")
        except Exception as e:
            print(f"[-] BLAD krytyczny: {e}")

        self.cap     = cv2.VideoCapture(0)
        self.frame   = None
        self.results = []
        self.running = True
        self.lock    = Lock()

        self.inference_busy   = False
        self.last_latency     = 0

        self._min_interval = 1.0 / TARGET_INFERENCE_FPS
        self._last_sent    = 0.0

    def camera_thread(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            with self.lock:
                self.frame = frame
        self.cap.release()

    def inference_thread(self):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]

        while self.running:
            now = time.time()
            local_frame = None

            with self.lock:
                too_soon = (now - self._last_sent) < self._min_interval
                if self.inference_busy or too_soon:
                    pass
                elif self.frame is not None:
                    local_frame        = self.frame.copy()
                    self.frame         = None
                    self.inference_busy = True
                    self._last_sent    = now

            if local_frame is None:
                time.sleep(0.002)
                continue

            start = time.time()
            try:
                orig_shape = local_frame.shape[:2]

                if local_frame.shape[:2] != (SEND_HEIGHT, SEND_WIDTH):
                    local_frame = cv2.resize(local_frame, (SEND_WIDTH, SEND_HEIGHT))

                ret, buffer = cv2.imencode('.jpg', local_frame, encode_param)
                if not ret:
                    continue
                
                img_encoded = np.array(buffer, dtype=np.uint8).flatten()

                infer_input = grpcclient.InferInput("input_image", img_encoded.shape, "UINT8")
                infer_input.set_data_from_numpy(img_encoded)

                response = self.client.infer(
                    model_name=MODEL_NAME,
                    inputs=[infer_input],
                    outputs=[grpcclient.InferRequestedOutput("object_boundaries")]
                )
                
                raw_preds  = response.as_numpy("object_boundaries")
                new_results = self.post_process(raw_preds, orig_shape)

                latency = (time.time() - start) * 1000
                with self.lock:
                    self.results      = new_results
                    self.last_latency = latency

            except Exception as e:
                print(f"[!] Triton Error: {e}")
            finally:
                with self.lock:
                    self.inference_busy = False

    def post_process(self, predictions, orig_shape):
        h_orig, w_orig = orig_shape
        detections = np.transpose(predictions[0])
        
        boxes, confs, class_ids = [], [], []
        sw, sh = w_orig / 640, h_orig / 640

        for det in detections:
            scores = det[4:]
            class_id = np.argmax(scores)
            max_score = scores[class_id]

            if max_score > CONF_THRESH:
                cx, cy, w, h = det[:4]
                boxes.append([
                    int((cx - w / 2) * sw), int((cy - h / 2) * sh),
                    int(w * sw), int(h * sh)
                ])
                confs.append(float(max_score))
                class_ids.append(int(class_id))

        indices = cv2.dnn.NMSBoxes(boxes, confs, CONF_THRESH, NMS_THRESH)
        final = []
        if len(indices) > 0:
            for i in indices.flatten():
                final.append({"box": boxes[i], "conf": confs[i], "class": class_ids[i]})
        return final

    def run(self):
        Thread(target=self.camera_thread,   daemon=True).start()
        Thread(target=self.inference_thread, daemon=True).start()

        print("[*] Interfejs wideo otwarty. Q = Wyjscie.")

        while True:
            with self.lock:
                if self.frame is not None:
                    self._last_display = self.frame.copy()
                if not hasattr(self, '_last_display'):
                    time.sleep(0.005)
                    continue
                display_frame   = self._last_display.copy()
                current_results = list(self.results)
                current_latency = self.last_latency

            for det in current_results:
                x, y, w, h = det['box']
                label = f"ID:{det['class']} {det['conf']:.2f}"
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(display_frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.putText(display_frame,
                        f"Latency E2E: {current_latency:.0f}ms",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Stream", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    streamer = TritonStreamer()
    streamer.run()