import hashlib


def hash_string(input_string: str) -> str:
    return hashlib.blake2b(input_string.encode(), usedforsecurity=False).hexdigest()
