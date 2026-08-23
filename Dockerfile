# 1. Use CUDA Devel image so nvcc/C++ compilers are available for llama-cpp
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# 2. Install ALL system dependencies (poppler-utils, postgresql-client, build-essential)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    poppler-utils \
    postgresql-client \
    libpq-dev \
    gcc \
    g++ \
    make \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. Install PyTorch + torchvision (CUDA Index)
RUN pip3 install --no-cache-dir -U \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu118

# 4. Install llama-cpp-python with full CUDA hardware acceleration
# Install pre-compiled llama-cpp-python CUDA wheel
RUN pip3 install --no-cache-dir \
    "https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.34-cu122/llama_cpp_python-0.3.34-py3-none-manylinux_2_35_x86_64.whl"

# 5. Install Transformers & rest of requirements.txt
RUN pip3 install --no-cache-dir transformers==5.15.1
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 6. Apply Numpy fix & clean up ABI conflicts
RUN pip3 install --no-cache-dir "numpy==2.0.2" && \
    pip3 uninstall -y torchcodec || true

# 7. Copy project code and set entrypoint
COPY . .
RUN chmod +x entrypoint.sh

ENTRYPOINT ["/bin/bash", "entrypoint.sh"]