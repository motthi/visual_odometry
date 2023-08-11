import cv2
import numpy as np


class HarrisCornerDetector():
    def __init__(self, blocksize: int = 2, ksize: int = 3, k: float = 0.04, bordertype: int = cv2.BORDER_DEFAULT, thd: float = 0.01):
        self.blocksize = blocksize  # Window size for corner detection
        self.ksize = ksize  # Aperture parameter for Sobel operator
        self.k = k  # Harris detector free parameter
        self.bordertype = bordertype
        self.thd = thd

    def detect(self, img: np.ndarray, mask: np.ndarray = None) -> list[cv2.KeyPoint]:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img = np.float32(gray_img)
        dst = cv2.cornerHarris(gray_img, self.blocksize, self.ksize, self.k, self.bordertype)
        dst[np.where(mask == 0)] = 0
        kpts = np.argwhere(dst > self.thd * dst.max())
        kpts = [cv2.KeyPoint(float(x[1]), float(x[0]), 13) for x in kpts]
        return kpts


class ShiTomashiCornerDetector():
    def __init__(self, max_corners: int = 1000, quality_level: float = 0.01, min_distance: int = 10, blocksize: int = 3):
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.blocksize = blocksize

    def detect(self, img: np.ndarray, mask: np.ndarray = None) -> list[cv2.KeyPoint]:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kpts = cv2.goodFeaturesToTrack(gray_img, self.max_corners, self.quality_level, self.min_distance, blockSize=self.blocksize, mask=mask)
        if kpts is None:
            return []
        kpts = [cv2.KeyPoint(float(x[0][0]), float(x[0][1]), 13) for x in kpts]
        return kpts
