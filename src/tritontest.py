import cv2
import numpy as np
import tritonclient.grpc as grpcclient
from threading import Thread, Lock
import time

# --- KONFIGURACJA ---
URL = "10.140.123.226:8001"
MODEL_NAME = "boundary_detection"
CONF_THRESH = 0.15
NMS_THRESH = 0.45


# --------------------

class TritonStreamer:
    def __init__(self):
        print(f"[*] Inicjalizacja połączenia z Tritonem: {URL}...")
        try:
            self.client = grpcclient.InferenceServerClient(url=URL)
            # Sprawdzenie czy serwer i model są gotowe
            if not self.client.is_server_live():
                print("[-] BŁĄD: Serwer Triton jest nieosiągalny. Sprawdź VPN/IP.")
            if not self.client.is_model_ready(MODEL_NAME):
                print(f"[-] BŁĄD: Model '{MODEL_NAME}' nie jest załadowany na serwerze.")
            else:
                print(f"[+] SUKCES: Połączono. Model '{MODEL_NAME}' jest gotowy.")
        except Exception as e:
            print(f"[-] BŁĄD krytyczny połączenia: {e}")

        self.cap = cv2.VideoCapture(0)
        self.frame = None
        self.results = []
        self.running = True
        self.lock = Lock()

        # Statystyki
        self.inference_count = 0
        self.last_latency = 0

    def camera_thread(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret: break
            with self.lock:
                self.frame = frame
        self.cap.release()

    def inference_thread(self):
        print("[*] Wątek inferencji uruchomiony.")
        while self.running:
            local_frame = None
            with self.lock:
                if self.frame is not None:
                    local_frame = self.frame.copy()

            if local_frame is not None:
                start_time = time.time()

                # Preprocessing
                img = cv2.cvtColor(local_frame, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (640, 640))
                img = img.astype(np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1))
                img = np.expand_dims(img, axis=0)

                try:
                    inputs = [grpcclient.InferInput("images", img.shape, "FP32")]
                    inputs[0].set_data_from_numpy(img)
                    outputs = [grpcclient.InferRequestedOutput("output0")]

                    response = self.client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)
                    raw_preds = response.as_numpy("output0")

                    new_results = self.post_process(raw_preds, local_frame.shape[:2])

                    with self.lock:
                        self.results = new_results
                        self.inference_count += 1
                        self.last_latency = (time.time() - start_time) * 1000  # ms

                    # Feedback co 30 klatek
                    if self.inference_count % 30 == 0:
                        det_count = len(new_results)
                        print(f"[Log] Przetworzono {self.inference_count} klatek. "
                              f"Latency: {self.last_latency:.0f}ms. Wykryto obiektów: {det_count}")

                except Exception as e:
                    # Wyświetlamy błąd tylko raz na jakiś czas, żeby nie spamować
                    if self.inference_count % 30 == 0:
                        print(f"[!] Triton Error: {e}")

            time.sleep(0.01)

    def post_process(self, predictions, orig_shape):
        h_orig, w_orig = orig_shape
        detections = predictions[0]
        boxes, confs, class_ids = [], [], []
        sw, sh = w_orig / 640, h_orig / 640

        for det in detections:
            # YOLOv5: [x, y, w, h, obj, c1, c2...]
            obj_conf = det[4]
            if obj_conf > CONF_THRESH:
                class_scores = det[5:]
                class_id = np.argmax(class_scores)
                final_conf = obj_conf * class_scores[class_id]

                if final_conf > CONF_THRESH:
                    cx, cy, w, h = det[:4]
                    boxes.append([int((cx - w / 2) * sw), int((cy - h / 2) * sh), int(w * sw), int(h * sh)])
                    confs.append(float(final_conf))
                    class_ids.append(int(class_id))

        indices = cv2.dnn.NMSBoxes(boxes, confs, CONF_THRESH, NMS_THRESH)
        final = []
        if len(indices) > 0:
            for i in indices.flatten():
                final.append({"box": boxes[i], "conf": confs[i], "class": class_ids[i]})
        return final

    def run(self):
        t1 = Thread(target=self.camera_thread, daemon=True)
        t2 = Thread(target=self.inference_thread, daemon=True)
        t1.start()
        t2.start()

        print("[*] Interfejs wideo otwarty. Q = Wyjście.")

        while True:
            with self.lock:
                if self.frame is None: continue
                display_frame = self.frame.copy()
                current_results = self.results
                current_latency = self.last_latency

            # Nakładanie ramek
            for det in current_results:
                x, y, w, h = det['box']
                label = f"ID:{det['class']} {det['conf']:.2f}"
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(display_frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Informacja o opóźnieniu na ekranie
            cv2.putText(display_frame, f"Triton Latency: {current_latency:.0f}ms",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Stream", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    streamer = TritonStreamer()
    streamer.run()