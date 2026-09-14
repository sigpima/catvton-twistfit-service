from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, HTTPException, Response, UploadFile, status

from app.auth import require_api_key
from app.generator import CLOTH_TYPES, default_generate_fn


@asynccontextmanager
async def lifespan(app: FastAPI):
    import os

    if os.environ.get("CATVTON_SKIP_MODEL_LOAD") != "1":
        from app.pipeline import generate_with_pipeline, load_pipeline

        bundle = load_pipeline()
        app.state.generate_fn = lambda p, g, c: generate_with_pipeline(bundle, p, g, c)
    yield


app = FastAPI(title="CatVTON Service", lifespan=lifespan)
app.state.generate_fn = default_generate_fn


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/generate", dependencies=[Depends(require_api_key)])
async def generate(
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
    cloth_type: str = Form(...),
) -> Response:
    if cloth_type not in CLOTH_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"cloth_type must be one of {CLOTH_TYPES}",
        )
    person_bytes = await person_image.read()
    garment_bytes = await garment_image.read()
    result_bytes = app.state.generate_fn(person_bytes, garment_bytes, cloth_type)
    return Response(content=result_bytes, media_type="image/png")
