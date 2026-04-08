import cv2
import numpy as np
import tritonclient.grpc as grpcclient
import sys

# Lista klas COCO (standard dla YOLOv5, kot jest pod indeksem 15)
# Możesz to podmienić na swoje klasy jeśli model był customowy.
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]


def prepare_image(img_path, width=640, height=640):
    """Przygotowuje obraz i zwraca wersję przygotowaną oraz oryginał z wymiarami."""
    original_img = cv2.imread(img_path)
    if original_img is None:
        print(f"Błąd: Nie znaleziono pliku {img_path}")
        sys.exit(1)

    orig_h, orig_w = original_img.shape[:2]

    # Preprocessing dla modelu: BGR2RGB -> Resize -> Normalizacja -> NCHW
    img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (width, height))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    return img, original_img, (orig_w, orig_h)


def post_process(raw_predictions, orig_dims, model_dims=(640, 640), conf_threshold=0.20, nms_threshold=0.45):
    """
    Wykonuje post-processing na surowym wyjściu YOLOv5.
    (Kształt: [1, 25200, 85])
    """
    detections = raw_predictions[0]  # [25200, 85]
    orig_w, orig_h = orig_dims
    model_w, model_h = model_dims

    # Współczynniki skalowania
    scale_w = orig_w / model_w
    scale_h = orig_h / model_h

    boxes = []
    confidences = []
    class_ids = []

    # 1. Filtrowanie po współczynniku pewności (Confidence Thresholding)
    for det in detections:
        # det format: [cx, cy, w, h, objectness, c1, ..., c80]
        objectness = det[4]
        if objectness < conf_threshold:
            continue

        # Obliczanie najlepszej klasy i jej wyniku
        class_scores = det[5:]
        class_id = np.argmax(class_scores)
        confidence = objectness * class_scores[class_id]  # Finalny score

        if confidence < conf_threshold:
            continue

        # Konwersja współrzędnych modelu (cx, cy, w, h) na absolutne x1, y1
        # i skalowanie do oryginalnego obrazu.
        # cv2.dnn.NMSBoxes oczekuje formatu [top_left_x, top_left_y, w, h]
        cx, cy, w, h = det[:4]

        top_left_x = int((cx - (w / 2)) * scale_w)
        top_left_y = int((cy - (h / 2)) * scale_h)
        width = int(w * scale_w)
        height = int(h * scale_h)

        boxes.append([top_left_x, top_left_y, width, height])
        confidences.append(float(confidence))
        class_ids.append(int(class_id))

    # 2. Non-Maximum Suppression (NMS) - OpenCV ma do tego świetną funkcję
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    final_detections = []
    if len(indices) > 0:
        # NMSBoxes zwraca indeksy w formacie [[idx1], [idx2]...] w starszych wersjach,
        # lub [idx1, idx2...] w nowszych. Spłaszczamy dla pewności.
        for i in indices.flatten():
            final_detections.append({
                'box': boxes[i],
                'conf': confidences[i],
                'class_id': class_ids[i]
            })

    return final_detections


def draw_and_show(image, detections):
    """Rysuje ramki na obrazie i wyświetla go."""
    img_copy = image.copy()

    print(f"Liczba wykrytych obiektów po NMS: {len(detections)}")

    for det in detections:
        x, y, w, h = det['box']
        conf = det['conf']
        class_id = det['class_id']

        # Nazwa klasy i pewność
        label = f"{COCO_CLASSES[class_id]}: {conf:.2f}"

        # Kolor ramki (zielony)
        color = (0, 255, 0)

        # Rysowanie ramki
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, 2)

        # Rysowanie tła pod tekst
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(img_copy, (x, y - 20), (x + text_size[0], y), color, -1)

        # Rysowanie tekstu
        cv2.putText(img_copy, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Wyświetlenie obrazu w oknie
    cv2.imshow("Wynik Boundary Detection", img_copy)
    print("Obraz wyświetlony. Naciśnij dowolny klawisz w oknie obrazu, aby zamknąć.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    # --- KONFIGURACJA ---
    URL = "10.140.123.226:8001"  # IP Jetsona w sieci VPN
    MODEL_NAME = "boundary_detection"  # Nazwa Twojego modelu
    IMAGE_PATH = "test1.jpg"  # Obrazek
    # --------------------

    try:
        # 1. Tworzymy klienta
        client = grpcclient.InferenceServerClient(url=URL)

        # 2. Przygotowanie danych i zapamiętanie oryginału
        input_data, original_img, original_dims = prepare_image(IMAGE_PATH)

        # 3. Definicja wejścia/wyjścia
        inputs = [grpcclient.InferInput("images", input_data.shape, "FP32")]
        inputs[0].set_data_from_numpy(input_data)
        outputs = [grpcclient.InferRequestedOutput("output0")]

        print(f"Wysyłanie klatki do modelu {MODEL_NAME}...")

        # 4. Inferencja
        results = client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)

        # 5. Pobranie surowych wyników
        raw_predictions = results.as_numpy("output0")
        print("Udało się otrzymać dane z Tritona.")

        # 6. Post-processing (Kluczowy etap)
        # Obniżyłem próg pewności do 0.15, bo 0.30 to mało.
        detections = post_process(raw_predictions, original_dims, conf_threshold=0.15)

        # 7. Wizualizacja
        draw_and_show(original_img, detections)

    except Exception as e:
        print(f"Wystąpił błąd: {e}")


if __name__ == "__main__":
    main()