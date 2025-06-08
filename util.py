import numpy as np                                                # type: ignore


class RandomSeed:

    def __init__(self, seed: int):
        self.seed = seed
        self.rng: np.random.Generator | None = None

    def __enter__(self) -> np.random.Generator:
        self.rng = np.random.default_rng(self.seed)
        return self.rng

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.rng = None
