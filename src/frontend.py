import zmq
import cv2
import numpy as np
import json

# --- KONFIGURACJA ---
BACKEND_IP = "127.0.0.1" 
WINDOW_NAME = "AI Dashboard - CamOverIP"

# Zmienne globalne do obsługi interfejsu (myszki i ekranu)
ui_state = {
    "fullscreen": False,
    "hover_fs": False,
    "width": 640 
}

def mouse_callback(event, x, y, flags, param):
    """Funkcja obsługująca zdarzenia myszki w oknie OpenCV"""
    w = param["width"]
    
    in_button = (w - 160 <= x <= w - 15) and (15 <= y <= 45)

    if event == cv2.EVENT_MOUSEMOVE:
        param["hover_fs"] = in_button
        
    elif event == cv2.EVENT_LBUTTONDOWN:
        if in_button:
            param["fullscreen"] = not param["fullscreen"]
            if param["fullscreen"]:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

def draw_shadow_text(img, text, pos, color=(255, 255, 255), scale=0.6, thickness=2):
    """Rysuje wygładzony tekst z cieniem"""
    x, y = pos
    cv2.putText(img, text, (x + 2, y + 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (15, 15, 15), thickness, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def start_frontend():
    print(f"[*] Łączenie z backendem {BACKEND_IP}...")
    context = zmq.Context()
    
    video_sock = context.socket(zmq.SUB)
    video_sock.connect(f"tcp://{BACKEND_IP}:5556")
    video_sock.setsockopt(zmq.SUBSCRIBE, b"ai_stream")
    
    ping_sock = context.socket(zmq.REQ)
    ping_sock.connect(f"tcp://{BACKEND_IP}:5557")
    ping_sock.setsockopt(zmq.RCVTIMEO, 2000) 
    ping_sock.setsockopt(zmq.SNDTIMEO, 2000)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback, ui_state)

    print("[+] Połączono! Oczekiwanie na strumień wideo...")

    while True:
        try:
            try:
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    print("[*] Zamknięto okno. Wychodzenie z aplikacji...")
                    break
            except cv2.error:
                break

            if video_sock.poll(10): 
                topic, metadata_bytes, frame_bytes = video_sock.recv_multipart()
                
                metadata = json.loads(metadata_bytes.decode('utf-8'))
                nparr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is not None:
                    h, w = frame.shape[:2]
                    ui_state["width"] = w 
                    
                    # 1. RYSOWANIE HUDU (Overlay)
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 0), (w, 85), (20, 20, 20), -1) 
                    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

                    onnx_data = metadata.get("onnx", [])
                    triton_data = metadata.get("triton", [])
                    stats = metadata.get("stats", {})

                    o = stats.get("onnx", {"status": "N/A", "latency": 0})
                    t = stats.get("triton", {"status": "N/A", "latency": 0})

                    # Wypisywanie statystyk po lewej
                    draw_shadow_text(frame, f"ONNX: {o['status']} ({o['latency']:.0f}ms)", (15, 25), (100, 255, 100), 0.65, 2)
                    draw_shadow_text(frame, f"TRITON: {t['status']} ({t['latency']:.0f}ms)", (15, 50), (255, 150, 50), 0.65, 2)
                    draw_shadow_text(frame, f"Detekcje -> ONNX: {len(onnx_data)} | TRITON: {len(triton_data)}", (15, 75), (255, 255, 255), 0.6, 2)

                    btn_bg_color = (100, 100, 100) if ui_state["hover_fs"] else (40, 40, 40)
                    cv2.rectangle(frame, (w - 160, 15), (w - 15, 45), btn_bg_color, -1)
                    cv2.rectangle(frame, (w - 160, 15), (w - 15, 45), (255, 255, 255), 1) # Biała ramka
                    
                    btn_text = "[ ] W OKNIE" if ui_state["fullscreen"] else "[ ] FULLSCREEN"
                    draw_shadow_text(frame, btn_text, (w - 145, 33), (255, 255, 255), 0.45, 1)

                    # 2. RYSOWANIE ONNX
                    for p in onnx_data:
                        try:
                            if isinstance(p, dict) and 'box' in p:
                                box = p['box']
                                is_norm = all(float(x) <= 1.05 for x in box)
                                cx, cy, nw, nh = box if len(box) == 4 else [0,0,0,0]
                                x1 = int((cx - nw / 2) * w) if is_norm else int(cx - nw / 2)
                                y1 = int((cy - nh / 2) * h) if is_norm else int(cy - nh / 2)
                                x2 = int((cx + nw / 2) * w) if is_norm else int(cx + nw / 2)
                                y2 = int((cy + nh / 2) * h) if is_norm else int(cy + nh / 2)
                                
                                conf = p.get('conf', 0.0)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 255, 50), 2, cv2.LINE_AA)
                                draw_shadow_text(frame, f"ONNX {int(conf*100)}%", (x1, y1 - 8), (50, 255, 50), 0.5, 1)
                        except Exception: pass

                    # 3. RYSOWANIE TRITON
                    for p in triton_data:
                        try:
                            if isinstance(p, dict) and 'box' in p:
                                x, y, w_box, h_box = map(int, p['box'][:4])
                                conf = p.get('conf', 0.0)
                                
                                cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (255, 150, 50), 2, cv2.LINE_AA)
                                draw_shadow_text(frame, f"TRITON {int(conf*100)}%", (x, y - 8), (255, 150, 50), 0.5, 1)
                        except Exception: pass

                    cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(10) & 0xFF
            
            if key == ord('q') or key == 27: 
                break
            elif key == ord('t'):
                print("\n[*] Wysyłam testowy PING do backendu...")
                try:
                    ping_sock.send_string("HALO! CZY MNIE SŁYCHAĆ?!")
                    reply = ping_sock.recv_string()
                    print(f"[+] SUKCES! Odpowiedź z backendu: {reply}")
                except zmq.error.Again:
                    print("[-] BŁĄD PINGA: Brak odpowiedzi! Sprawdź połączenie.")

        except KeyboardInterrupt:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_frontend()