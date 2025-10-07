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
RUN python - <<'PY'
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download

root = Path("/workspace/models")
root.mkdir(parents=True, exist_ok=True)

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

snapshot_download(
    repo_id="Wan-AI/Wan2.1-I2V-14B-480P",
    local_dir=str(ensure_dir(root / "Wan2.1-I2V-14B-480P")),
    local_dir_use_symlinks=False,
)

snapshot_download(
    repo_id="TencentGameMate/chinese-wav2vec2-base",
    local_dir=str(ensure_dir(root / "chinese-wav2vec2-base")),
    local_dir_use_symlinks=False,
)

snapshot_download(
    repo_id="TencentGameMate/chinese-wav2vec2-base",
    allow_patterns=["model.safetensors"],
    local_dir=str(root / "chinese-wav2vec2-base"),
    local_dir_use_symlinks=False,
    revision="refs/pr/1",
)

snapshot_download(
    repo_id="MeiGen-AI/InfiniteTalk",
    allow_patterns=["single/infinitalk.safetensors"],
    local_dir=str(ensure_dir(root / "InfiniteTalk")),
    local_dir_use_symlinks=False,
)

try:
    snapshot_download(
        repo_id="MeiGen-AI/InfiniteTalk",
        allow_patterns=["quant_models/infinitalk_single_fp8.safetensors"],
        local_dir=str(ensure_dir(root / "InfiniteTalk" / "quant_models")),
        local_dir_use_symlinks=False,
    )
except Exception as exc:
    print(f"⚠️  fp8 quant weights not available: {exc}")

snapshot_download(
    repo_id="vrgamedevgirl84/Wan14BT2VFusioniX",
    allow_patterns=["FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors"],
    local_dir=str(ensure_dir(root / "FusionX_LoRa")),
    local_dir_use_symlinks=False,
)

nested = root / "FusionX_LoRa" / "FusionX_LoRa" / "Wan2.1_I2V_14B_FusionX_LoRA.safetensors"
target = root / "FusionX_LoRa" / "Wan2.1_I2V_14B_FusionX_LoRA.safetensors"
if nested.exists() and not target.exists():
    shutil.move(str(nested), str(target))
    try:
        nested.parent.rmdir()
    except OSError:
        pass
PY

COPY start.sh /start.sh
RUN chmod +x /start.sh

ENV WAN_GPU_COUNT=0 \
    PYTHONPATH=/workspace:/workspace/infinitetalk

CMD ["/start.sh"]
