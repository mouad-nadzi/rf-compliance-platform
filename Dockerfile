FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# 1. System dependencies (poppler-utils, postgres client headers, gcc)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    poppler-utils \
    libpq-dev \
    gcc \
    g++ \
    make \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. PyTorch + torchvision (CUDA 11.8 / 12 build)
RUN pip3 install --no-cache-dir -U \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu118

# 3. Pre-compiled llama-cpp-python CUDA wheel (v0.3.34-cu122)
RUN pip3 install --no-cache-dir \
    "https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.34-cu122/llama_cpp_python-0.3.34-py3-none-manylinux_2_35_x86_64.whl"

# 4. Transformers >= 5.0 (Required for GLM-OCR)
RUN pip3 install --no-cache-dir transformers==5.15.1

# 5. Standard Python Requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 6. Apply ABI/Numpy fix
RUN pip3 install --no-cache-dir "numpy==2.0.2" && \
    pip3 uninstall -y torchcodec || true

# 7. Copy Application Code & Setup Entrypoint
COPY . .
RUN chmod +x entrypoint.sh

ENTRYPOINT ["/bin/bash", "entrypoint.sh"]