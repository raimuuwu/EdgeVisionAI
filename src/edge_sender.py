import cv2
import imagezmq
import socket
import time
from picamera2 import Picamera2

sender = imagezmq.ImageSender(connect_to='tcp://*:5555', REQ_REP=False)
rpi_name = socket.gethostname()

picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

jpeg_quality = 80

print(f"[{rpi_name}] Rozpoczęto nadawanie strumienia w tle (PUB)...")

try:
    while True:
        start_time = time.time()
        
        try:
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Błąd odczytu z kamery: {e}. Próbuję ponownie...")
            time.sleep(1)
            continue

        ret_code, jpg_buffer = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        )
        
        sender.send_jpg(rpi_name, jpg_buffer)

        elapsed_time = time.time() - start_time
        sleep_time = max(0.0, 0.2 - elapsed_time)
        time.sleep(sleep_time)

except KeyboardInterrupt:
    print("\nZatrzymano skrypt nadawczy.")
finally:
    picam2.stop()
    print("Zwolniono kamerę i zamknięto Picamera2.")