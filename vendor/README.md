# vendor/CatVTON

Cloned from https://github.com/Zheng-Chong/CatVTON (CC BY-NC-SA 4.0,
non-commercial use only — see the root README's Global Constraints).
Not modified. `app/pipeline.py` imports `model.pipeline.CatVTONPipeline`
and `model.cloth_masker.AutoMasker` from here — this directory must stay
on `sys.path` (handled by `app/pipeline.py`).

The base checkpoint is `booksforcharlie/stable-diffusion-inpainting` —
a copy of `runwayml/stable-diffusion-inpainting`, which CatVTON's own
`app.py` notes was deleted by runwayml upstream. If
`booksforcharlie/stable-diffusion-inpainting` is ever unavailable too,
check `vendor/CatVTON/app.py`'s `--base_model_path` default for the
currently recommended value and update `BASE_MODEL_PATH` in
`app/pipeline.py` to match.
