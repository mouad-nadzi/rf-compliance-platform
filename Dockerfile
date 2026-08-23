FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Prevent interactive prompts during apt installations
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10 and required system packages
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    postgresql-client \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set python3 as default python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# Upgrade pip
RUN python -m pip install --upgrade pip

# Copy requirements and install dependencies
COPY requirements.txt .

# Pre-install llama-cpp-python specific wheel as required for GPU on Ubuntu 22.04
RUN CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install llama-cpp-python

# Install the rest of the dependencies
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Ensure entrypoint is executable
RUN chmod +x entrypoint.sh

# Expose FastAPI and Streamlit ports
EXPOSE 8000 8501

ENTRYPOINT ["./entrypoint.sh"]