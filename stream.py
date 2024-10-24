import cv2
import numpy as np
import socket
import postprocessor
import time
import onnxruntime
onnxruntime.set_default_logger_severity(3)
import threading

ip = "192.168.4.1"
port = 8080

latest_data = None
to_process = None
lock = threading.Lock()
object_model = onnxruntime.InferenceSession("decoder.onnx", providers=['CPUExecutionProvider']) 

def read_frames():
    global latest_data
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((ip, port))
    request = "GET / HTTP/1.1\r\n\r\n"
    client_socket.sendall(request.encode())
    stream_bytes = b''
    while True:
        stream_bytes += client_socket.recv(1024)
        start = stream_bytes.find(b'\xff\xd8')  
        end = stream_bytes.find(b'\xff\xd9') 
        if start != -1 and end != -1:
            sb = stream_bytes[start:end+2]
            stream_bytes = stream_bytes[end+2:]
            input_array = np.frombuffer(sb, dtype=np.uint8)
            in_bytes_decoded = cv2.imdecode(input_array, cv2.IMREAD_COLOR)
            with lock:
                latest_data = in_bytes_decoded

thread = threading.Thread(target=read_frames, daemon=True)
thread.start()

cv2.namedWindow('Livestream', cv2.WINDOW_NORMAL)

while True:
    with lock:
        if latest_data is not None:
            to_process = latest_data.copy()

    if to_process is not None:
        img = cv2.resize(to_process, (320, 256))
        time.sleep(0.1)
        input_image = np.expand_dims(img.astype(np.float32).transpose(2, 0, 1) / 255.0, axis=0)
        ort_inputs = {object_model.get_inputs()[0].name: input_image}
        results = object_model.run(None, ort_inputs)[0]
        postprocessor.plot_boxes(results, img)
        annotated_frame = cv2.resize(img, (640, 480))
        cv2.imshow('Latest Frame', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
