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
                # odczyt wychwytywanych bajtow
                in_tensor = pb_utils.get_input_tensor_by_name(request, "raw_bytes")
                if in_tensor is None:
                    raise ValueError("Triton przekazal puste wejscie")

                img_array = in_tensor.as_numpy()
                
                if img_array.size < 100:
                    raise ValueError(f"Otrzymano {img_array.size} bajtow, to za malo.")

                # Dekodowanie
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if img is None:
                    raise ValueError(f"cv2.imdecode zwrocilo None.")

                # Preprocessing pod YOLOv8
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (640, 640))
                img = img.astype(np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1))
                img = np.expand_dims(img, axis=0)

                img = np.ascontiguousarray(img, dtype=np.float32)

                out_tensor = pb_utils.Tensor("tensors", img)
                responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))

            except Exception as e:
                error_msg = f"Triton Python Error: {traceback.format_exc()}"
                print(error_msg)
                
                # Zwracamy czarną klatkę jako Fallback
                fallback = np.zeros((1, 3, 640, 640), dtype=np.float32)
                out_tensor = pb_utils.Tensor("tensors", fallback)
                responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))

        return responses

    def finalize(self):
        pass
