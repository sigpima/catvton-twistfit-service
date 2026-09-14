import io
import os
import sys
from dataclasses import dataclass

from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vendor", "CatVTON"))

from model.cloth_masker import AutoMasker  # noqa: E402
from model.pipeline import CatVTONPipeline  # noqa: E402
from utils import init_weight_dtype, resize_and_crop, resize_and_padding  # noqa: E402

BASE_MODEL_PATH = "booksforcharlie/stable-diffusion-inpainting"
RESUME_REPO_ID = "zhengchong/CatVTON"
WIDTH = 768
HEIGHT = 1024


@dataclass
class PipelineBundle:
    pipeline: CatVTONPipeline
    automasker: AutoMasker
    mask_processor: VaeImageProcessor


def load_pipeline() -> PipelineBundle:
    repo_path = snapshot_download(repo_id=RESUME_REPO_ID)

    pipeline = CatVTONPipeline(
        base_ckpt=BASE_MODEL_PATH,
        attn_ckpt=repo_path,
        attn_ckpt_version="mix",
        weight_dtype=init_weight_dtype("bf16"),
        use_tf32=True,
        device="cuda",
    )

    mask_processor = VaeImageProcessor(
        vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True
    )

    automasker = AutoMasker(
        densepose_ckpt=os.path.join(repo_path, "DensePose"),
        schp_ckpt=os.path.join(repo_path, "SCHP"),
        device="cuda",
    )

    return PipelineBundle(pipeline=pipeline, automasker=automasker, mask_processor=mask_processor)


def generate_with_pipeline(
    bundle: PipelineBundle, person_bytes: bytes, garment_bytes: bytes, cloth_type: str
) -> bytes:
    person_image = Image.open(io.BytesIO(person_bytes)).convert("RGB")
    garment_image = Image.open(io.BytesIO(garment_bytes)).convert("RGB")

    person_image = resize_and_crop(person_image, (WIDTH, HEIGHT))
    garment_image = resize_and_padding(garment_image, (WIDTH, HEIGHT))

    mask = bundle.automasker(person_image, cloth_type)["mask"]
    mask = bundle.mask_processor.blur(mask, blur_factor=9)

    result_image = bundle.pipeline(
        image=person_image,
        condition_image=garment_image,
        mask=mask,
        num_inference_steps=50,
        guidance_scale=2.5,
        generator=None,
    )[0]

    output = io.BytesIO()
    result_image.save(output, format="PNG")
    return output.getvalue()
