FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_ETAG_TIMEOUT=60 \
    MODEL_DIR=/workspace/models \
    OUTPUT_DIR=/workspace/outputs

WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y --no-install-recommends git ffmpeg git-lfs libmagic1 && \
    rm -rf /var/lib/apt/lists/*

COPY . /workspace

RUN sed -i "s/from inspect import ArgSpec/# from inspect import ArgSpec  # Removed for Python 3.11 compatibility/" infinitetalk/wan/multitalk.py

RUN pip install --upgrade pip && \
    pip install --no-cache-dir runpod misaki[en] ninja psutil packaging flash_attn==2.7.4.post1 pydantic python-magic huggingface_hub soundfile librosa xformers==0.0.28 && \
    pip install --no-cache-dir -r infinitetalk/requirements.txt && \
    pip install --no-cache-dir .

COPY start.sh /start.sh
RUN chmod +x /start.sh

ENV WAN_GPU_COUNT=0

CMD ["/start.sh"]
