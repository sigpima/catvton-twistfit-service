from fastapi import FastAPI

app = FastAPI(title="CatVTON Service")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
