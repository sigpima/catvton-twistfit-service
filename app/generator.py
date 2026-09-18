from typing import Callable

GenerateFn = Callable[[bytes, bytes, str], bytes]
GenerateBatchFn = Callable[[list[bytes], bytes, str], list[bytes]]

CLOTH_TYPES = ("upper", "lower", "overall")


def default_generate_fn(person_bytes: bytes, garment_bytes: bytes, cloth_type: str) -> bytes:
    raise NotImplementedError("Real CatVTON pipeline not wired in — see app/pipeline.py")


def default_generate_batch_fn(person_bytes_list: list[bytes], garment_bytes: bytes, cloth_type: str) -> list[bytes]:
    raise NotImplementedError("Real CatVTON pipeline not wired in — see app/pipeline.py")
