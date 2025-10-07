import base64
import json
import os
import shutil
import sys
import tempfile
import urllib.request
import uuid
from pathlib import Path
from types import SimpleNamespace

import runpod
from huggingface_hub import snapshot_download, hf_hub_download

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
infinitetalk_dir = ROOT_DIR / "infinitetalk"
if infinitetalk_dir.exists() and str(infinitetalk_dir) not in sys.path:
    sys.path.append(str(infinitetalk_dir))

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/workspace/models"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/workspace/outputs"))
MASTER_ADDR = os.environ.get("MASTER_ADDR", "127.0.0.1")
MASTER_PORT = os.environ.get("MASTER_PORT", "29500")

def _spawn_generate_worker(local_rank: int, args_namespace, world_size: int) -> None:
    """Entry point for torch.multiprocessing to run distributed generation."""
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ.setdefault("MASTER_ADDR", MASTER_ADDR)
    os.environ.setdefault("MASTER_PORT", MASTER_PORT)

    try:
        from infinitetalk.generate_infinitetalk import generate as _generate
    except ImportError:
        from generate_infinitetalk import generate as _generate

    try:
        _generate(args_namespace)
    finally:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()


def _decode_reference(ref: str) -> bytes:
    """Resolve input reference as URL, base64 data URI, raw base64, or local path."""
    if not ref:
        raise ValueError("Empty reference")

    if ref.startswith(("http://", "https://")):
        with urllib.request.urlopen(ref) as response:
            return response.read()

    if ref.startswith("data:"):
        _, encoded = ref.split(",", 1)
        return base64.b64decode(encoded)

    candidate_path = None
    try:
        candidate_path = Path(ref)
        if candidate_path.exists():
            return candidate_path.read_bytes()
    except OSError:
        # Often triggered when a base64 string starts with '/' and Path tries to treat it as an absolute path.
        candidate_path = None

    # Fallback: treat as raw base64 string
    try:
        return base64.b64decode(ref, validate=True)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Unsupported reference format: {exc}") from exc


OPTION_KEYS = {
    "size": str,
    "sample_steps": int,
    "sample_text_guide_scale": float,
    "sample_audio_guide_scale": float,
    "color_correction_strength": float,
    "mode": str,
    "quant": str,
    "quant_dir": str,
    "use_teacache": lambda v: v if isinstance(v, bool) else str(v).lower() in {"1", "true", "yes", "on"},
}


class InfiniteTalkRunner:
    def __init__(self) -> None:
        self.model_dir = MODEL_DIR
        self.output_dir = OUTPUT_DIR
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._initialized = False

    def _download_and_validate(self, content: bytes, expected_types: list[str]) -> bytes:
        import magic

        mime = magic.Magic(mime=True)
        detected = mime.from_buffer(content)
        if detected not in expected_types:
            raise ValueError(f"Invalid file type {detected}; expected {expected_types}")
        return content

    def _ensure_models(self) -> None:
        if self._initialized:
            return

        model_root = self.model_dir
        print("--- RunPod: Ensuring model assets ---")

        # download Wan base
        ensure_dir = lambda p: p.mkdir(parents=True, exist_ok=True) or p
        ensure_dir(model_root / "Wan2.1-I2V-14B-480P")
        snapshot_download(
            repo_id="Wan-AI/Wan2.1-I2V-14B-480P",
            local_dir=str(model_root / "Wan2.1-I2V-14B-480P"),
            local_dir_use_symlinks=False,
        )

        # wav2vec2 base + safetensors
        ensure_dir(model_root / "chinese-wav2vec2-base")
        snapshot_download(
            repo_id="TencentGameMate/chinese-wav2vec2-base",
            local_dir=str(model_root / "chinese-wav2vec2-base"),
            local_dir_use_symlinks=False,
        )
        hf_hub_download(
            repo_id="TencentGameMate/chinese-wav2vec2-base",
            filename="model.safetensors",
            revision="refs/pr/1",
            local_dir=str(model_root / "chinese-wav2vec2-base"),
            local_dir_use_symlinks=False,
        )

        # InfiniteTalk weights
        ensure_dir(model_root / "InfiniteTalk")
        hf_hub_download(
            repo_id="MeiGen-AI/InfiniteTalk",
            filename="single/infinitetalk.safetensors",
            local_dir=str(model_root / "InfiniteTalk"),
            local_dir_use_symlinks=False,
        )

        # optional fp8
        if overrides.get("quant") and str(overrides["quant"]).lower() == "fp8":
            try:
                ensure_dir(model_root / "InfiniteTalk" / "quant_models")
                hf_hub_download(
                    repo_id="MeiGen-AI/InfiniteTalk",
                    filename="quant_models/infinitalk_single_fp8.safetensors",
                    local_dir=str(model_root / "InfiniteTalk" / "quant_models"),
                    local_dir_use_symlinks=False,
                )
            except Exception as exc:
                print(f"⚠️  fp8 weights unavailable: {exc}")

        # LoRA and clean up nested folder
        ensure_dir(model_root / "FusionX_LoRa")
        hf_hub_download(
            repo_id="vrgamedevgirl84/Wan14BT2VFusioniX",
            filename="FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors",
            local_dir=str(model_root / "FusionX_LoRa"),
            local_dir_use_symlinks=False,
        )
        nested = model_root / "FusionX_LoRa" / "FusionX_LoRa" / "Wan2.1_I2V_14B_FusionX_LoRA.safetensors"
        if nested.exists():
            target = model_root / "FusionX_LoRa" / "Wan2.1_I2V_14B_FusionX_LoRA.safetensors"
            shutil.move(str(nested), str(target))
            try:
                nested.parent.rmdir()
            except OSError:
                pass

        self._initialized = True

    def _build_args(
        self,
        image_path: str,
        audio_path: str,
        prompt: str | None,
        input_json_path: str,
        mode: str,
        chunk_frame_num: int,
        max_frame_num: int,
        overrides: dict[str, object],
    ) -> SimpleNamespace:
        size = str(overrides.get("size", "infinitetalk-720"))
        sample_steps = int(overrides.get("sample_steps", 8))
        sample_text_guide_scale = float(overrides.get("sample_text_guide_scale", 1.0))
        sample_audio_guide_scale = float(overrides.get("sample_audio_guide_scale", 6.0))
        color_correction_strength = float(overrides.get("color_correction_strength", 0.2))
        quant = overrides.get("quant")
        quant_dir = overrides.get("quant_dir")
        use_teacache = overrides.get("use_teacache")
        if use_teacache is None:
            use_teacache = True
        if quant and not quant_dir:
            if str(quant).lower() == "fp8":
                default_fp8 = self.model_dir / "InfiniteTalk" / "quant_models" / "infinitetalk_single_fp8.safetensors"
                if default_fp8.exists():
                    quant_dir = str(default_fp8)
                else:
                    raise FileNotFoundError(
                        "fp8 quant weights are not baked into the image. "
                        "Either rebuild the image with these assets or provide 'quant_dir'."
                    )
            else:
                raise ValueError(f"Unsupported quant '{quant}'. Please supply 'quant_dir'.")
        if isinstance(overrides.get("mode"), str):
            mode = overrides["mode"]

        return SimpleNamespace(
            task="infinitetalk-14B",
            size=size,
            frame_num=chunk_frame_num,
            max_frame_num=max_frame_num,
            ckpt_dir=str(self.model_dir / "Wan2.1-I2V-14B-480P"),
            infinitetalk_dir=str(self.model_dir / "InfiniteTalk" / "single" / "infinitetalk.safetensors"),
            quant=quant,
            quant_dir=quant_dir,
            wav2vec_dir=str(self.model_dir / "chinese-wav2vec2-base"),
            dit_path=None,
            lora_dir=[str(self.model_dir / "FusionX_LoRa" / "Wan2.1_I2V_14B_FusionX_LoRA.safetensors")],
            lora_scale=[1.0],
            offload_model=False,
            ulysses_size=1,
            ring_size=1,
            t5_fsdp=False,
            t5_cpu=False,
            dit_fsdp=False,
            save_file=str(self.output_dir / Path(image_path).stem),
            audio_save_dir=str(self.output_dir / "temp_audio"),
            base_seed=42,
            input_json=input_json_path,
            motion_frame=25,
            mode=mode,
            sample_steps=sample_steps,
            sample_shift=3.0,
            sample_text_guide_scale=sample_text_guide_scale,
            sample_audio_guide_scale=sample_audio_guide_scale,
            num_persistent_param_in_dit=0,
            audio_mode="localfile",
            use_teacache=use_teacache,
            teacache_thresh=0.3,
            use_apg=True,
            apg_momentum=-0.75,
            apg_norm_threshold=55,
            color_correction_strength=color_correction_strength,
            scene_seg=False,
            prompt=prompt,
        )

    def generate(
        self,
        image_ref: str,
        audio_ref: str,
        prompt: str | None,
        overrides: dict[str, object] | None = None,
    ) -> tuple[Path, SimpleNamespace]:
        import io
        import librosa
        import magic
        import os
        import torch
        import torch.multiprocessing as mp
        from PIL import Image as PILImage

        self._ensure_models()

        critical_assets = [
            self.model_dir / "FusionX_LoRa" / "Wan2.1_I2V_14B_FusionX_LoRA.safetensors",
            self.model_dir / "InfiniteTalk" / "single" / "infinitetalk.safetensors",
            self.model_dir / "Wan2.1-I2V-14B-480P" / "config.json",
        ]
        missing_assets = [asset for asset in critical_assets if not asset.exists()]
        if missing_assets:
            print(f"--- RunPod: Missing critical assets detected {missing_assets}. Redownloading... ---")
            self._initialized = False
            self._ensure_models()
            critical_assets = [asset for asset in critical_assets]
            still_missing = [asset for asset in critical_assets if not asset.exists()]
            if still_missing:
                raise FileNotFoundError(f"Critical assets still missing after reinitialization: {still_missing}")

        overrides = overrides or {}

        quant = overrides.get("quant")
        quant_dir = overrides.get("quant_dir")
        if quant and not quant_dir:
            default_fp8 = self.model_dir / "InfiniteTalk" / "quant_models" / "infinitetalk_single_fp8.safetensors"
            if default_fp8.exists():
                quant_dir = str(default_fp8)
            else:
                raise FileNotFoundError(
                    "Quantization requested but fp8 weights are not present in the image. "
                    "Provide 'quant_dir' in the request or rebuild the image with these assets."
                )
        if quant and quant_dir:
            quant_path = Path(quant_dir)
            if not quant_path.exists():
                raise FileNotFoundError(f"Quantization file not found at {quant_path}")

        image_bytes = self._download_and_validate(
            _decode_reference(image_ref),
            [
                "image/jpeg",
                "image/png",
                "image/gif",
                "image/bmp",
                "image/tiff",
                "video/mp4",
                "video/avi",
                "video/quicktime",
                "video/x-msvideo",
                "video/webm",
                "video/x-ms-wmv",
                "video/x-flv",
            ],
        )
        audio_bytes = self._download_and_validate(
            _decode_reference(audio_ref),
            ["audio/mpeg", "audio/wav", "audio/x-wav"],
        )

        print("--- RunPod: Preparing inputs ---")
        mime = magic.Magic(mime=True)
        detected_mime = mime.from_buffer(image_bytes)

        with tempfile.NamedTemporaryFile(suffix=".mp4" if detected_mime.startswith("video/") else ".jpg", delete=False) as tmp_image:
            if detected_mime.startswith("video/"):
                tmp_image.write(image_bytes)
            else:
                source_image = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
                source_image.save(tmp_image.name, "JPEG")
            image_path = tmp_image.name

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
            tmp_audio.write(audio_bytes)
            audio_path = tmp_audio.name

        total_audio_duration = librosa.get_duration(path=audio_path)
        audio_embedding_frames = int(total_audio_duration * 25)
        max_possible_frames = max(5, audio_embedding_frames - 5)
        calculated_frame_num = min(1000, max_possible_frames)
        n = (calculated_frame_num - 1) // 4
        frame_num = 4 * n + 1
        if frame_num >= audio_embedding_frames:
            safe_frames = audio_embedding_frames - 10
            n = max(1, (safe_frames - 1) // 4)
            frame_num = 4 * n + 1

        if calculated_frame_num > 81:
            mode = "streaming"
            chunk_frame_num = 81
            max_frame_num = frame_num
        else:
            mode = "clip"
            chunk_frame_num = frame_num
            max_frame_num = frame_num

        requested_mode = overrides.get("mode")
        if requested_mode is not None:
            requested_mode = str(requested_mode).lower()
            if requested_mode not in {"clip", "streaming"}:
                raise ValueError("mode must be either 'clip' or 'streaming'")
            if requested_mode == "clip":
                mode = "clip"
                chunk_frame_num = frame_num
                max_frame_num = frame_num
            else:
                mode = "streaming"
                chunk_frame_num = min(chunk_frame_num, 81)
                max_frame_num = frame_num
            overrides["mode"] = requested_mode

        print(
            f"--- RunPod: Audio {total_audio_duration:.2f}s, frames {frame_num}, chunk {chunk_frame_num}, mode {mode} ---"
        )

        input_json_data = {
            "prompt": prompt or "a person is talking",
            "cond_video": image_path,
            "cond_audio": {"person1": audio_path},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp_json:
            json.dump(input_json_data, tmp_json)
            input_json_path = tmp_json.name

        args = self._build_args(
            image_path,
            audio_path,
            input_json_data["prompt"],
            input_json_path,
            mode,
            chunk_frame_num,
            max_frame_num,
            overrides,
        )

        temp_output_name = f"{uuid.uuid4()}"
        args.save_file = str(self.output_dir / temp_output_name)

        audio_save_dir = Path(args.audio_save_dir)
        audio_save_dir.mkdir(parents=True, exist_ok=True)

        available_gpus = torch.cuda.device_count()
        requested_gpus = int(os.environ.get("WAN_GPU_COUNT", "0") or 0)
        if requested_gpus <= 0 or requested_gpus > available_gpus:
            requested_gpus = available_gpus
        world_size = max(1, requested_gpus)
        print(f"--- RunPod: Using {world_size} GPU(s) ---")

        if world_size > 1:
            try:
                try:
                    from wan.configs import WAN_CONFIGS
                except ImportError:
                    from infinitetalk.wan.configs import WAN_CONFIGS

                config = WAN_CONFIGS.get(args.task)
                num_heads = getattr(config, "num_heads", None) if config else None
            except Exception as config_error:  # noqa: BLE001
                print(f"--- RunPod: Failed to read config for parallelism: {config_error} ---")
                num_heads = None

            ulysses_size = 1
            ring_size = 1
            if num_heads is not None:
                divisors = [d for d in range(world_size, 0, -1) if world_size % d == 0]
                for candidate in divisors:
                    if candidate > 1 and num_heads % candidate == 0:
                        ulysses_size = candidate
                        ring_size = world_size // candidate
                        break

            if ulysses_size > 1 or ring_size > 1:
                args.ulysses_size = ulysses_size
                args.ring_size = ring_size
                args.t5_fsdp = True
                args.dit_fsdp = True
                print(
                    "--- RunPod: Parallelism "
                    f"ulysses={args.ulysses_size}, ring={args.ring_size}, FSDP=True ---"
                )
            else:
                print("--- RunPod: Parallelism configuration not compatible; falling back to duplicated compute ---")
                world_size = 1

        if world_size > 1:
            try:
                mp.set_start_method("spawn", force=False)
            except RuntimeError:
                pass
            mp.spawn(_spawn_generate_worker, args=(args, world_size), nprocs=world_size, join=True)
        else:
            _spawn_generate_worker(0, args, 1)

        generated_file = Path(f"{args.save_file}.mp4")
        final_output_path = self.output_dir / f"{temp_output_name}.mp4"
        if generated_file.exists():
            shutil.move(str(generated_file), final_output_path)
        else:
            raise RuntimeError("Generation finished but output video not found")

        for path in (input_json_path, audio_path, image_path):
            if path and os.path.exists(path):
                os.unlink(path)

        temp_audio_dir = self.output_dir / "temp_audio"
        if temp_audio_dir.exists():
            shutil.rmtree(temp_audio_dir, ignore_errors=True)

        return final_output_path, args


RUNNER = InfiniteTalkRunner()


def handler(event):
    """RunPod serverless handler."""
    job_input = event.get("input", {})
    image_ref = job_input.get("image") or job_input.get("image1")
    audio_ref = job_input.get("audio1") or job_input.get("audio")
    prompt = job_input.get("prompt")

    if not image_ref or not audio_ref:
        return {"error": "Both 'image' and 'audio1' inputs are required."}

    overrides: dict[str, object] = {}
    for key, caster in OPTION_KEYS.items():
        if key in job_input and job_input[key] is not None:
            try:
                overrides[key] = caster(job_input[key])
            except Exception as exc:  # noqa: BLE001
                return {"error": f"Invalid value for {key}: {exc}"}

    try:
        video_path, used_args = RUNNER.generate(image_ref, audio_ref, prompt, overrides)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}

    response: dict[str, str] = {"video_path": str(video_path)}
    response["generation_params"] = {
        "size": used_args.size,
        "sample_steps": used_args.sample_steps,
        "sample_text_guide_scale": used_args.sample_text_guide_scale,
        "sample_audio_guide_scale": used_args.sample_audio_guide_scale,
        "color_correction_strength": used_args.color_correction_strength,
        "mode": used_args.mode,
        "quant": used_args.quant,
        "quant_dir": used_args.quant_dir,
        "use_teacache": used_args.use_teacache,
    }

    try:
        from runpod.serverless.utils import rp_upload

        uploaded_url = rp_upload.upload_output(str(video_path))
        if uploaded_url:
            response["video_url"] = uploaded_url
    except Exception as upload_error:  # noqa: BLE001
        print(f"--- RunPod: Upload failed: {upload_error} ---")
        with open(video_path, "rb") as f:
            response["video_base64"] = base64.b64encode(f.read()).decode("utf-8")

    return response


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
