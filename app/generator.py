from typing import Callable

GenerateFn = Callable[[bytes, bytes, str], bytes]

CLOTH_TYPES = ("upper", "lower", "overall")


def default_generate_fn(person_bytes: bytes, garment_bytes: bytes, cloth_type: str) -> bytes:
    raise NotImplementedError("Real CatVTON pipeline not wired in — see app/pipeline.py")
