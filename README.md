# CatVTON Service

Wraps CatVTON (CC BY-NC-SA 4.0 — non-commercial only) behind an
authenticated FastAPI endpoint. See
`frontend/docs/superpowers/specs/2026-09-15-virtual-tryon-design.md`
for the full design.

## Deploying on Vast.ai

1. Rent an NVIDIA GPU instance, ≥8GB VRAM (RTX 3060/4060/3090 all
   work), using a template with CUDA + PyTorch preinstalled. When
   creating the instance, map a container port (e.g. 8000) to a public
   port.
2. SSH into the instance:
   ```bash
   git clone <this-catvton-service-repo-url>
   cd catvton-service
   git clone https://github.com/Zheng-Chong/CatVTON vendor/CatVTON
   python3 -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   export CATVTON_API_KEY="<choose a long random secret>"
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```
   The first startup downloads the CatVTON checkpoints from Hugging
   Face (several GB) — this can take a while.
3. Note the instance's public IP and the port Vast.ai mapped to 8000 —
   this is `CATVTON_SERVICE_URL` for the main backend
   (`http://<public-ip>:<mapped-port>`), and `CATVTON_API_KEY` is the
   secret you exported above.

## Lifecycle

- `stop` the instance when not in use — this halts GPU billing (a
  small storage charge continues; see the Vast.ai billing notes in the
  design spec).
- `delete` the instance once done with that phase of work, to stop
  storage billing entirely.

## Local dev (no GPU)

Run with `CATVTON_SKIP_MODEL_LOAD=1 uvicorn app.main:app --reload` to
skip loading the real model — `/generate` will raise
`NotImplementedError` until something overrides `app.state.generate_fn`,
but `/health` and the auth/contract tests all work normally.

Run the test suite (no GPU or vendored repo needed for this — those
tests never import `app.pipeline`):

```bash
python3 -m venv venv && source venv/bin/activate
pip install fastapi uvicorn[standard] python-multipart pillow httpx pytest
pytest -v
```
