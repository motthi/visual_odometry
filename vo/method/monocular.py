from __future__ import annotations
import cv2
import numpy as np
from vo.utils import form_transf
from vo.method import VoEstimator


class MonocularVoEstimator(VoEstimator):
    def __init__(self, K_l):
        self.K_l = K_l

    def estimate(self, pkpts: list(cv2.KeyPoint), ckpts: list(cv2.KeyPoint)):
        """
        Calculates the transformation matrix
        Parameters
        ----------
        pkpts (ndarray): The good keypoints matches position in i-1'th image
        ckpts (ndarray): The good keypoints matches position in i'th image
        Returns
        -------
        transformation_matrix (ndarray): The transformation matrix
        """
        pkpts = np.float32([kpt.pt for kpt in pkpts])
        ckpts = np.float32([kpt.pt for kpt in ckpts])
        # pkpts = cv2.undistortPoints(np.expand_dims(pkpts, axis=1), cameraMatrix=self.K_l, distCoeffs=None)
        # ckpts = cv2.undistortPoints(np.expand_dims(ckpts, axis=1), cameraMatrix=self.K_l, distCoeffs=None)

        E, mask = cv2.findEssentialMat(pkpts, ckpts, cameraMatrix=self.K_l, threshold=1)
        if E.shape != (3, 3):
            return None
        _, R, t, _ = cv2.recoverPose(E, pkpts, ckpts, cameraMatrix=self.K_l, mask=mask)

        transf = form_transf(R, np.squeeze(t))
        return transf
