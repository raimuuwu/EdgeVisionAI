import cv2
import numpy as np
import tritonclient.grpc as grpcclient
import urllib.request
from threading import Thread, Lock
import time

# --- KONFIGURACJA ---
URL          = "10.140.123.226:8001"
METRICS_URL  = "http://10.140.123.226:8002/metrics"
MODEL_NAME   = "ensemble_model"
CONF_THRESH  = 0.25
NMS_THRESH   = 0.45

TARGET_INFERENCE_FPS = 30
JPEG_QUALITY         = 80
SEND_RESOLUTION      = (640, 640)
MONITOR_INTERVAL     = 10

class TritonStreamer:
    def __init__(self):
        print(f"[*] Ladowanie klienta Triton: {URL}...")
        try:
            self.client = grpcclient.InferenceServerClient(url=URL)
            if self.client.is_server_live() and self.client.is_model_ready(MODEL_NAME):
                print(f"[+] SUKCES: Polaczono z serwerem. Model '{MODEL_NAME}' jest gotowy.")
            else:
                print("[-] BLAD: Problem z serwerem lub modelem.")
        except Exception as e:
            print(f"[-] BLAD krytyczny polaczenia: {e}")

        self.cap = cv2.VideoCapture(0)
        self.frame = None
        self.results = []
        self.running = True
        self.lock = Lock()

        self.inference_busy = False
        self.last_latency = 0
        self.inference_count = 0
        self.skipped_count = 0
        self.last_payload_size = 0

        self._min_interval = 1.0 / TARGET_INFERENCE_FPS
        self._last_sent = 0.0

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
                    if self.inference_busy:
                        self.skipped_count += 1
                elif self.frame is not None:
                    local_frame = self.frame.copy()
                    self.frame = None
                    self.inference_busy = True
                    self._last_sent = now

            if local_frame is None:
                time.sleep(0.005)
                continue

            start = time.time()
            try:
                orig_shape = local_frame.shape[:2]
                
                if orig_shape != SEND_RESOLUTION:
                    local_frame = cv2.resize(local_frame, SEND_RESOLUTION)

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
                
                raw_preds = response.as_numpy("object_boundaries")
                new_results = self.post_process(raw_preds, orig_shape)

                latency = (time.time() - start) * 1000
                with self.lock:
                    self.results = new_results
                    self.last_latency = latency
                    self.inference_count += 1
                    self.last_payload_size = len(img_encoded) / 1024.0 # w KB

            except Exception as e:
                print(f"[!] Triton Error: {e}")
            finally:
                with self.lock:
                    self.inference_busy = False

    def stats_thread(self):
        prev_stats = {}
        last_time = time.time()

        while self.running:
            time.sleep(MONITOR_INTERVAL)
            if not self.running:
                break
            
            now = time.time()
            elapsed = now - last_time
            last_time = now

            with self.lock:
                frames = self.inference_count
                skipped = self.skipped_count
                e2e = self.last_latency
                payload = self.last_payload_size

            fps = frames / elapsed if elapsed > 0 else 0

            print("\n" + "=" * 60)
            print(f"  PEŁNY RAPORT TRITON STATS  [{time.strftime('%H:%M:%S')}]")
            print("=" * 60)
            print(" [1] STATYSTYKI KLIENTA (LOKALNE)")
            print(f"  * Przetworzono klatek  : {frames}")
            print(f"  * Pominięto klatek     : {skipped} (GPU nie wyrabia/za wysoki FPS)")
            print(f"  * Klient FPS           : {fps:.1f} klatek/s")
            print(f"  * End-to-End Latency   : {e2e:.0f} ms")
            print(f"  * Rozmiar paczki (sieć): {payload:.1f} KB")

            try:
                stats = self.client.get_inference_statistics(model_name="", as_json=False)
                if stats and len(stats.model_stats) > 0:
                    print("\n [2] STATYSTYKI SERWERA")
                    for ms in stats.model_stats:
                        name = ms.name
                        inf = ms.inference_stats
                        
                        count = inf.success.count
                        queue_ns = inf.queue.ns
                        compute_ns = inf.compute_infer.ns
                        input_ns = inf.compute_input.ns
                        output_ns = inf.compute_output.ns

                        prev = prev_stats.get(name, {"count": count})
                        delta_count = count - prev.get("count", count)

                        if delta_count > 0:
                            def avg(cur, key): return (cur - prev.get(key, cur)) / delta_count / 1e6
                            a_in = avg(input_ns, "input_ns")
                            a_q = avg(queue_ns, "queue_ns")
                            a_gpu = avg(compute_ns, "compute_ns")
                            a_out = avg(output_ns, "output_ns")
                        else:
                            a_in = a_q = a_gpu = a_out = 0.0

                        prev_stats[name] = {
                            "count": count, "input_ns": input_ns,
                            "queue_ns": queue_ns, "compute_ns": compute_ns,
                            "output_ns": output_ns
                        }

                        total_server_time = a_in + a_q + a_gpu + a_out
                        print(f"  --- {name.upper()} ---")
                        print(f"   > Obsługa wejścia : {a_in:.2f} ms")
                        print(f"   > Czas w kolejce  : {a_q:.2f} ms")
                        print(f"   > Czyste GPU/CPU  : {a_gpu:.2f} ms")
                        print(f"   > Obsługa wyjścia : {a_out:.2f} ms")
                        print(f"   * TOTAL Serwer    : {total_server_time:.2f} ms")
            except Exception as e:
                print(f"  [!] Nie udało się pobrać statystyk gRPC: {e}")
                
            print("=" * 60 + "\n")
            
            with self.lock:
                self.inference_count = 0
                self.skipped_count = 0


    def post_process(self, predictions, orig_shape):
        h_orig, w_orig = orig_shape
        detections = np.transpose(predictions[0])
        
        boxes, confs, class_ids = [], [], []
        sw, sh = w_orig / SEND_RESOLUTION[0], h_orig / SEND_RESOLUTION[1]

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
        Thread(target=self.camera_thread, daemon=True).start()
        Thread(target=self.inference_thread, daemon=True).start()
        Thread(target=self.stats_thread, daemon=True).start()

        print("[*] Strumieniowanie wideo aktywne. Wcisnij 'Q' w oknie zeby wyjsc.")

        while self.running:
            with self.lock:
                if self.frame is not None:
                    self._last_display = self.frame.copy()
                if not hasattr(self, '_last_display'):
                    time.sleep(0.005)
                    continue
                display_frame = self._last_display.copy()
                current_results = list(self.results)
                current_latency = self.last_latency

            for det in current_results:
                x, y, w, h = det['box']
                label = f"ID:{det['class']} {det['conf']:.2f}"
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(display_frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.putText(display_frame, f"Latency: {current_latency:.0f} ms",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Stream", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    streamer = TritonStreamer()
    streamer.run()