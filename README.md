# EdgeVisionAI

**EdgeVisionAI** is a university project focused on creating a high-performance, single-purpose AI inference gateway for edge devices. 

---

## Core Concept & Single-Responsibility

The main philosophy of this project is **strict single-task focus**. The core application behaves as a dedicated, lightweight **Central AI Backend Gateway**. 

Instead of heavy processing on a single machine, its sole responsibility is to:
1. **Ingest** live video streams (either from a local camera or a remote Raspberry Pi via VPN).
2. **Distribute** frames concurrently to dedicated inference servers (**ONNX** and **Triton Inference Server**).
3. **Stream** the unified results (synchronized video frames + combined AI metadata) via ZeroMQ to any connected frontend.

By decoupling ingestion, inference, and visualization, the system achieves maximum frame rates and minimal latency on edge hardware.

---

## Features (What's Done)

* **Multi-Source Video Ingestion**: Supports local webcam testing (`cv2.VideoCapture`) as well as production-ready streaming from a remote Raspberry Pi via `imagezmq`.
* **Dual Inference Pipelines**: 
    * **ONNX Worker**: Connects to a custom ONNX server via ZeroMQ (`zmq.REQ`) with automated fail-safes and timeout handling.
    * **Triton Worker**: Communicates via gRPC (`grpcclient`) with a Triton ensemble model, featuring custom YOLOv8 output tensor parsing and Non-Maximum Suppression (NMS).
* **Concurrent Multi-Threading**: Fully asynchronous architecture utilizing Python `threading` and thread-safe `Lock` structures to prevent performance bottlenecks.
* **Live Pub/Sub Streaming**: Broadcasts a multipart ZMQ stream (`zmq.PUB`) containing raw JPEG buffers and standardized JSON metadata (`onnx` results, `triton` results, and performance metrics).
* **Built-in Diagnostics**: Includes a CLI status printer, an active Frontend Ping-Pong monitoring thread, and a local OpenCV HUD debug view.

---

## Prerequisites

To run this backend, you need **Python 3.8+** and the following core dependencies:

* `opencv-python` (Image processing & local display)
* `numpy` (Tensor manipulation)
* `pyzmq` (Network communication layers)
* `tritonclient[grpc]` (Triton Inference Server communication)
* `imagezmq` (Raspberry Pi video streaming)

---

## How to Run

### 1. Clone & Setup Environment
```bash
git clone [https://github.com/raimuuwu/EdgeVisionAI.git](https://github.com/raimuuwu/EdgeVisionAI.git)
cd EdgeVisionAI

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install requirements (make sure to generate requirements.txt based on prerequisites)
pip install -r requirements.txt
```
### 2. Configuration
Before running, open src/backend.py (or your equivalent path) and configure the network endpoints to match your lab environment setup:
```python
ONNX_IP = "YOUR_ONNX_SERVER_IP"
TRITON_URL = "YOUR_TRITON_SERVER_IP:8001"
RPI_VPN_IP = "YOUR_RASPBERRY_PI_IP"
```
### 3. Execution Modes

* Production / Gateway Mode (Default)
    Runs the multi-threaded network pipeline, waiting for Raspberry Pi stream and broadcasting data on port 5556 for the frontend (can be used with parameters to use local camera or remote one):
```bash
    python src/backend.py
```
```python
    if __name__ == "__main__":
        hub = UnifiedBackend()
        hub.run() # True/Empty for remote camera; False for local device camera
```
* Local Debug / Verification Mode
    If you want to test the system locally without the full network infrastructure, you can invoke the run_local() method (e.g., by modifying the script's __main__ block). This utilizes your laptop webcam and renders a live window with an ONNX (Green) and Triton (Blue) HUD overlay:
```python
    if __name__ == "__main__":
        hub = UnifiedBackend()
        hub.run_local() # For local testing
```
## Developed as a university project for an Group Project course in Warsaw University of Technology.
