import cv2
import numpy as np
import tritonclient.grpc as grpcclient
import urllib.request
from threading import Thread, Lock
import time

#  KONFIGURACJA
URL          = "10.140.123.226:8001"
METRICS_URL  = "http://10.140.123.226:8002/metrics"
MODEL_NAME   = "ensemble_model"
CONF_THRESH  = 0.15
NMS_THRESH   = 0.45

# Docelowa liczba klatek wysyłanych do Tritona na sekundę.
TARGET_INFERENCE_FPS = 30

MONITOR_INTERVAL = 10   # co ile sekund drukować statystyki [s]
JPEG_QUALITY     = 80   # jakość JPEG [1-100], niżej = mniejszy transfer
SEND_WIDTH       = 640  # resize przed enkodowaniem (640 = brak straty precyzji)
SEND_HEIGHT      = 640
# =====================================================

class TritonMonitor:
    def __init__(self, client):
        self.client  = client
        self.running = True
        self._prev   = {}

    def _fetch_metrics(self):
        try:
            with urllib.request.urlopen(METRICS_URL, timeout=3) as r:
                return r.read().decode()
        except Exception:
            return None

    def _fetch_stats(self):
        try:
            return self.client.get_inference_statistics(model_name="", as_json=False)
        except Exception as e:
            print(f"[Monitor] Blad statystyk gRPC: {e}")
            return None

    def _report(self):
        stats        = self._fetch_stats()
        metrics_text = self._fetch_metrics()

        print("\n" + "=" * 55)
        print(f"  TRITON STATS  [{time.strftime('%H:%M:%S')}]")
        print("=" * 55)

        if stats:
            for ms in stats.model_stats:
                name = ms.name
                inf  = ms.inference_stats
                count      = inf.success.count
                queue_ns   = inf.queue.ns
                compute_ns = inf.compute_infer.ns
                input_ns   = inf.compute_input.ns
                output_ns  = inf.compute_output.ns

                prev        = self._prev.get(name, {})
                delta_count = count - prev.get("count", count)

                if delta_count > 0:
                    def avg(cur, key):
                        return (cur - prev.get(key, cur)) / delta_count / 1e6
                    a_in  = avg(input_ns,   "input_ns")
                    a_q   = avg(queue_ns,   "queue_ns")
                    a_gpu = avg(compute_ns, "compute_ns")
                    a_out = avg(output_ns,  "output_ns")
                else:
                    a_in = a_q = a_gpu = a_out = 0.0

                self._prev[name] = {
                    "count": count, "input_ns": input_ns,
                    "queue_ns": queue_ns, "compute_ns": compute_ns,
                    "output_ns": output_ns
                }

                total = a_in + a_q + a_gpu + a_out
                print(f"\n  Model: {name}")
                print(f"    Laczne wywolania : {count}")
                print(f"    Ostatnie {MONITOR_INTERVAL}s ({delta_count} inferencji):")
                print(f"      Wejscie (CPU)  : {a_in:.1f} ms")
                print(f"      Kolejka        : {a_q:.1f} ms")
                print(f"      Inferencja GPU : {a_gpu:.1f} ms")
                print(f"      Wyjscie (CPU)  : {a_out:.1f} ms")
                print(f"      SUMA serwer    : {total:.1f} ms")

        if metrics_text:
            print("\n  GPU / System:")
            for keyword, label, unit in [
                ("nv_gpu_utilization",      "GPU utilization", "%"),
                ("nv_gpu_memory_used_bytes","GPU memory used ", "MB"),
                ("nv_gpu_power_usage",      "GPU power       ", "W"),
            ]:
                for line in metrics_text.splitlines():
                    if line.startswith("#") or keyword not in line:
                        continue
                    try:
                        val = float(line.split()[-1])
                        if unit == "%":
                            print(f"    {label}: {val*100:.1f}%")
                        elif unit == "MB":
                            print(f"    {label}: {val/1024/1024:.0f} MB")
                        else:
                            print(f"    {label}: {val:.1f} W")
                        break
                    except ValueError:
                        pass

        print("=" * 55 + "\n")

    def start(self):
        def loop():
            while self.running:
                time.sleep(MONITOR_INTERVAL)
                self._report()
        Thread(target=loop, daemon=True).start()

    def stop(self):
        self.running = False


class TritonStreamer:
    def __init__(self):
        print(f"[*] Inicjalizacja polaczenia z Tritonem: {URL}...")
        print(f"[*] Target inference FPS : {TARGET_INFERENCE_FPS}")
        print(f"[*] JPEG quality         : {JPEG_QUALITY}")
        print(f"[*] Send resolution      : {SEND_WIDTH}x{SEND_HEIGHT}")
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
        self.inference_count  = 0
        self.skipped_count    = 0
        self.last_latency     = 0

        # Minimalny odstęp między kolejnymi inferencjami [s]
        self._min_interval = 1.0 / TARGET_INFERENCE_FPS
        self._last_sent    = 0.0

        self.monitor = TritonMonitor(self.client)

    def camera_thread(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            with self.lock:
                self.frame = frame   # zawsze najświeższa klatka
        self.cap.release()

    def inference_thread(self):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        print("[*] Watek inferencji uruchomiony.")

        while self.running:
            now        = time.time()
            local_frame = None

            with self.lock:
                too_soon = (now - self._last_sent) < self._min_interval
                if self.inference_busy or too_soon:
                    self.skipped_count += 1
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
                # Resize po stronie klienta — mniejszy payload przez sieć
                if local_frame.shape[:2] != (SEND_HEIGHT, SEND_WIDTH):
                    local_frame = cv2.resize(local_frame, (SEND_WIDTH, SEND_HEIGHT))

                ret, buffer = cv2.imencode('.jpg', local_frame, encode_param)
                if not ret:
                    continue

                jpeg_bytes  = buffer.tobytes()
                infer_input = grpcclient.InferInput("input_image", [1], "BYTES")
                infer_input.set_data_from_numpy(np.array([jpeg_bytes], dtype=object))

                response  = self.client.infer(
                    model_name=MODEL_NAME,
                    inputs=[infer_input],
                    outputs=[grpcclient.InferRequestedOutput("object_boundaries")]
                )
                raw_preds  = response.as_numpy("object_boundaries")
                new_results = self.post_process(raw_preds, (SEND_HEIGHT, SEND_WIDTH))

                latency = (time.time() - start) * 1000
                with self.lock:
                    self.results         = new_results
                    self.inference_count += 1
                    self.last_latency    = latency

                if self.inference_count % 30 == 0:
                    print(f"[Log] Klatek: {self.inference_count} | "
                          f"Pominieto: {self.skipped_count} | "
                          f"Latency E2E: {latency:.0f}ms | "
                          f"Payload: {len(jpeg_bytes)/1024:.1f} KB")

            except Exception as e:
                print(f"[!] Triton Error: {e}")
            finally:
                with self.lock:
                    self.inference_busy = False

    def post_process(self, predictions, orig_shape):
        h_orig, w_orig = orig_shape
        detections     = predictions[0]
        boxes, confs, class_ids = [], [], []
        sw, sh = w_orig / 640, h_orig / 640

        for det in detections:
            obj_conf = det[4]
            if obj_conf > CONF_THRESH:
                class_scores = det[5:]
                class_id     = np.argmax(class_scores)
                final_conf   = obj_conf * class_scores[class_id]
                if final_conf > CONF_THRESH:
                    cx, cy, w, h = det[:4]
                    boxes.append([
                        int((cx - w / 2) * sw), int((cy - h / 2) * sh),
                        int(w * sw),             int(h * sh)
                    ])
                    confs.append(float(final_conf))
                    class_ids.append(int(class_id))

        indices = cv2.dnn.NMSBoxes(boxes, confs, CONF_THRESH, NMS_THRESH)
        final   = []
        if len(indices) > 0:
            for i in indices.flatten():
                final.append({"box": boxes[i], "conf": confs[i], "class": class_ids[i]})
        return final

    def run(self):
        Thread(target=self.camera_thread,   daemon=True).start()
        Thread(target=self.inference_thread, daemon=True).start()
        self.monitor.start()

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
                        f"Latency: {current_latency:.0f}ms  |  FPS: {TARGET_INFERENCE_FPS}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Stream", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                self.monitor.stop()
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    streamer = TritonStreamer()
    streamer.run()