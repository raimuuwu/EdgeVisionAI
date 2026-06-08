import triton_python_backend_utils as pb_utils
import numpy as np
import cv2
import traceback

class TritonPythonModel:
    def initialize(self, args):
        pass

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                # 1. Pobranie z Tritona
                in_tensor = pb_utils.get_input_tensor_by_name(request, "raw_bytes")
                if in_tensor is None:
                    raise ValueError("Triton przekazal puste wejscie")

                raw = in_tensor.as_numpy()
                jpeg_bytes = raw.flatten()[0]

                img_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)

                if img_array.size == 0:
                    raise ValueError(f"Otrzymano pusty bufor bajtow")


                # 2. Dekodowanie
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if img is None:
                    # Zaloguj rozmiar bufora zeby pomoc w diagnostyce
                    raise ValueError(
                        f"cv2.imdecode zwrocilo None. "
                        f"Rozmiar bufora: {img_array.size} bajtow, "
                        f"pierwsze bajty: {img_array[:16].tolist()}"
                    )                

                # 3. Preprocessing do YOLO
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (640, 640))
                img = img.astype(np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1))
                img = np.expand_dims(img, axis=0)

                img = np.ascontiguousarray(img, dtype=np.float32)

                # 4. Output
                out_tensor = pb_utils.Tensor("tensors", img)
                responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))

            except Exception as e:
                error_msg = f"Triton Python Error: {traceback.format_exc()}"
                print(error_msg)
                # Zwracamy fallback zamiast TritonError zeby ensemble nie dostawal pustego outputu.
                # Czarna klatka pozwoli modelowi ONNX wykonac sie i zwrocic wynik (bez wykryc).
                fallback = np.zeros((1, 3, 640, 640), dtype=np.float32)
                out_tensor = pb_utils.Tensor("tensors", fallback)
                responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))


        return responses

    def finalize(self):
        pass
