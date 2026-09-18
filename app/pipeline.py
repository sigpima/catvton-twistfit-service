import io
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vendor", "CatVTON"))

from model.cloth_masker import AutoMasker  # noqa: E402
from model.pipeline import CatVTONPipeline  # noqa: E402
from utils import (  # noqa: E402
    init_weight_dtype,
    prepare_image,
    prepare_mask_image,
    resize_and_crop,
    resize_and_padding,
)

BASE_MODEL_PATH = "booksforcharlie/stable-diffusion-inpainting"
RESUME_REPO_ID = "zhengchong/CatVTON"
WIDTH = 768
HEIGHT = 1024

# Matches the CatVTON authors' own Gradio demo default (app.py's `seed`
# slider defaults to 42, not -1/random). A fixed seed makes generations
# reproducible instead of varying wildly run to run on the same inputs.
FIXED_SEED = 42


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


def _fixed_generator(device: str) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(FIXED_SEED)


def _repaint(person_image: Image.Image, mask: Image.Image, result_image: Image.Image) -> Image.Image:
    """Composite the generated result back over the original photo outside
    the mask, so anything the model altered outside the intended garment
    region (hair, background, skin) snaps back to the real pixels instead
    of bleeding/artifacting. Mirrors CatVTON's own `inference.py:repaint`.
    """
    person_np = np.array(person_image).astype(np.float32)
    result_np = np.array(result_image).astype(np.float32)
    mask_np = np.array(mask.convert("L")).astype(np.float32)[:, :, None] / 255.0
    repainted = person_np * (1 - mask_np) + result_np * mask_np
    return Image.fromarray(repainted.astype(np.uint8))


def _to_png_bytes(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def generate_with_pipeline(
    bundle: PipelineBundle, person_bytes: bytes, garment_bytes: bytes, cloth_type: str
) -> bytes:
    return generate_batch_with_pipeline(bundle, [person_bytes], garment_bytes, cloth_type)[0]


def generate_batch_with_pipeline(
    bundle: PipelineBundle, person_bytes_list: list[bytes], garment_bytes: bytes, cloth_type: str
) -> list[bytes]:
    """Run every person image through the same garment/cloth_type in one
    batched pipeline call (e.g. the front and side angles of one job)
    instead of one call per image — halves the number of GPU forward
    passes for a multi-angle job.
    """
    person_images = [Image.open(io.BytesIO(pb)).convert("RGB") for pb in person_bytes_list]
    garment_image = Image.open(io.BytesIO(garment_bytes)).convert("RGB")

    person_images = [resize_and_crop(p, (WIDTH, HEIGHT)) for p in person_images]
    garment_image = resize_and_padding(garment_image, (WIDTH, HEIGHT))

    masks = [bundle.automasker(p, cloth_type)["mask"] for p in person_images]
    masks = [bundle.mask_processor.blur(m, blur_factor=9) for m in masks]

    device = bundle.pipeline.device
    weight_dtype = bundle.pipeline.weight_dtype

    image_batch = prepare_image(person_images).to(device, dtype=weight_dtype)
    condition_batch = prepare_image([garment_image] * len(person_images)).to(device, dtype=weight_dtype)
    mask_batch = prepare_mask_image(masks).to(device, dtype=weight_dtype)

    result_images = bundle.pipeline(
        image=image_batch,
        condition_image=condition_batch,
        mask=mask_batch,
        num_inference_steps=50,
        guidance_scale=2.5,
        generator=_fixed_generator(device),
    )

    return [
        _to_png_bytes(_repaint(person_image, mask, result_image))
        for person_image, mask, result_image in zip(person_images, masks, result_images)
    ]
