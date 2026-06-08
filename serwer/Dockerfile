FROM nvcr.io/nvidia/tritonserver:24.08-py3-igpu

USER root
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir opencv-python-headless numpy

RUN pip3 install --no-cache-dir numpy==1.25.1

CMD ["tritonserver", "--model-repository=/models", "--backend-config=python,shm-default-byte-size=33554432"]
