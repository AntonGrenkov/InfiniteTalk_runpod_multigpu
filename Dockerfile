FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_ETAG_TIMEOUT=60 \
    MODEL_DIR=/workspace/models \
    OUTPUT_DIR=/workspace/outputs \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y --no-install-recommends git ffmpeg git-lfs libmagic1 && \
    rm -rf /var/lib/apt/lists/*

COPY . /workspace

RUN sed -i "s/from inspect import ArgSpec/# from inspect import ArgSpec  # Removed for Python 3.11 compatibility/" infinitetalk/wan/multitalk.py

RUN pip install --upgrade pip && \
    pip install --no-cache-dir runpod misaki[en] ninja psutil packaging flash_attn==2.7.4.post1 pydantic python-magic "huggingface_hub[cli]" soundfile librosa xformers==0.0.28 && \
    pip install --no-cache-dir -r infinitetalk/requirements.txt

# Pre-download model weights into the image so runtime containers are ready to infer immediately.
RUN set -eu; \
    mkdir -p \
        /workspace/models/Wan2.1-I2V-14B-480P \
        /workspace/models/chinese-wav2vec2-base \
        /workspace/models/InfiniteTalk/quant_models \
        /workspace/models/FusionX_LoRa; \
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P \
        --local-dir /workspace/models/Wan2.1-I2V-14B-480P \
        --local-dir-use-symlinks False; \
    huggingface-cli download TencentGameMate/chinese-wav2vec2-base \
        --local-dir /workspace/models/chinese-wav2vec2-base \
        --local-dir-use-symlinks False; \
    huggingface-cli download TencentGameMate/chinese-wav2vec2-base model.safetensors \
        --revision refs/pr/1 \
        --local-dir /workspace/models/chinese-wav2vec2-base \
        --local-dir-use-symlinks False; \
    huggingface-cli download MeiGen-AI/InfiniteTalk single/infinitalk.safetensors \
        --local-dir /workspace/models/InfiniteTalk \
        --local-dir-use-symlinks False; \
    huggingface-cli download vrgamedevgirl84/Wan14BT2VFusioniX FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors \
        --local-dir /workspace/models/FusionX_LoRa \
        --local-dir-use-symlinks False; \
    if [ -f /workspace/models/FusionX_LoRa/FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors ]; then \
        mv /workspace/models/FusionX_LoRa/FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors /workspace/models/FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors; \
        rmdir /workspace/models/FusionX_LoRa/FusionX_LoRa; \
    fi

COPY start.sh /start.sh
RUN chmod +x /start.sh

ENV WAN_GPU_COUNT=0 \
    PYTHONPATH=/workspace:/workspace/infinitetalk

CMD ["/start.sh"]
