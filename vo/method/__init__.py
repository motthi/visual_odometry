import numpy as np


class VoEstimator():
    def estimate(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def save_results(self, *args, **kwargs):
        raise NotImplementedError
