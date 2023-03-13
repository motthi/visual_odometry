import cv2
import numpy as np
from scipy.optimize import least_squares
from vo.utils import form_transf
from vo.method import VoEstimator


class StereoVoEstimator(VoEstimator):
    def __init__(self, P_l):
        self.P_l = P_l

    def reprojection_residuals(
        self,
        transf: np.ndarray,
        prev_pixes: np.ndarray, curr_pixes: np.ndarray,
        prev_3d_pts: np.ndarray, curr_3d_pts: np.ndarray
    ) -> np.ndarray:
        """Calculate residuals for reprojection

        Args:
            transf (np.ndarray): Transformation matrix in the homogeneous form
            prev_pixes (np.ndarray): Pixel points in image 1
            curr_pixes (np.ndarray): Pixel points in image 2
            prev_3d_pts (np.ndarray): 3D points in image 1
            curr_3d_pts (np.ndarray): 3D points in image 2

        Returns:
            np.ndarray: Reprojection residuals (Flattened)
        """
        # Create the projection matrix for the i-1'th image and i'th image
        f_projection = self.P_l @ transf
        b_projection = self.P_l @ np.linalg.inv(transf)

        q1_pred = curr_3d_pts.T @ f_projection.T        # Project 3D points from i'th image to i-1'th image
        q1_pred = q1_pred[:, :2].T / q1_pred[:, 2]      # Un-homogenize
        q2_pred = prev_3d_pts.T @ b_projection.T        # Project 3D points from i-1'th image to i'th image
        q2_pred = q2_pred[:, :2].T / q2_pred[:, 2]      # Un-homogenize
        residuals = np.vstack([q1_pred - prev_pixes.T, q2_pred - curr_pixes.T]).flatten()  # Calculate the residuals
        return residuals


class LmBasedEstimator(StereoVoEstimator):
    def __init__(self, P_l, max_iter=100):
        super().__init__(P_l)
        self.max_iter = max_iter

    def estimate(
        self,
        prev_pixes: np.ndarray, curr_pixes: np.ndarray,
        prev_3d_pts: np.ndarray, curr_3d_pts: np.ndarray
    ) -> np.ndarray:
        # Initialize the min_error and early_termination counter
        min_error = float('inf')
        early_termination = 0
        early_termination_thd = 10
        for _ in range(self.max_iter):
            sample_idx = np.random.choice(range(prev_pixes.shape[0]), 20)

            sample_q1 = prev_pixes[sample_idx]
            sample_q2 = curr_pixes[sample_idx]
            sample_Q1 = prev_3d_pts[sample_idx]
            sample_Q2 = curr_3d_pts[sample_idx]

            # Perform least squares optimization
            in_guess = np.zeros(6)  # Make the start guess
            opt_res = least_squares(
                self.optimize_function,                             # Function to minimize
                in_guess,                                           # Initial guess
                method='lm',                                        # Levenberg-Marquardt algorithm
                max_nfev=200,                                       # Max number of function evaluations
                args=(sample_q1, sample_q2, sample_Q1, sample_Q2)   # Additional arguments to pass to the function
            )

            # Calculate the error for the optimized transformation
            res = self.optimize_function(opt_res.x, prev_pixes, curr_pixes, prev_3d_pts, curr_3d_pts)
            res = res.reshape((prev_3d_pts.shape[0] * 2, 2))
            err = np.sum(np.linalg.norm(res, axis=1))

            # Check if the error is less the the current min error. Save the result if it is
            if err < min_error:
                min_error = err
                out_pose = opt_res.x
                early_termination = 0
            else:
                early_termination += 1
            if early_termination == early_termination_thd:
                # If we have not fund any better result in early_termination_threshold iterations
                break
        r = out_pose[:3]             # Get the rotation vector
        R, _ = cv2.Rodrigues(r)      # Make the rotation matrix
        t = -out_pose[3:]            # Get the translation vector
        T = form_transf(R, t)
        return T

    def optimize_function(
            self,
            dof: np.ndarray,
            p_pixes: np.ndarray, c_pixes: np.ndarray,
            p3d_pts: np.ndarray, c3d_pts: np.ndarray
    ) -> np.ndarray:
        """Optimize function: Calculates the reprojection residuals from the rotation and translation vector

        Args:
            dof (np.ndarray): Rotation vector and translation vector
            prev_pixes (np.ndarray): Pixel points in image 1
            curr_pixes (np.ndarray): Pixel points in image 2
            prev_3d_pts (np.ndarray): 3D points in image 1
            curr_3d_pts (np.ndarray): 3D points in image 2

        Returns:
            np.ndarray: Reprojection residuals (Flattened)
        """
        r = dof[:3]                   # Get the rotation vector
        R, _ = cv2.Rodrigues(r)       # Create the rotation matrix from the rotation vector
        t = dof[3:]                   # Get the translation vector
        transf = form_transf(R, t)    # Create the transformation matrix from the rotation matrix and translation vector
        ones = np.ones((p_pixes.shape[0], 1))
        p3d_pts = np.hstack([p3d_pts, ones]).T
        c3d_pts = np.hstack([c3d_pts, ones]).T
        return self.reprojection_residuals(transf, p_pixes, c_pixes, p3d_pts, c3d_pts)


class SvdBasedEstimator(StereoVoEstimator):
    def estimate(
        self,
        prev_pixes: np.ndarray, curr_pixes: np.ndarray,
        prev_3d_pts: np.ndarray, curr_3d_pts: np.ndarray,
    ) -> np.ndarray:
        prev_3d_pts = np.vstack((prev_3d_pts.T, np.ones((1, prev_3d_pts.shape[0]))))
        curr_3d_pts = np.vstack((curr_3d_pts.T, np.ones((1, curr_3d_pts.shape[0]))))
        T = self.svd_based_estimate(prev_3d_pts, curr_3d_pts)
        return T

    def svd_based_estimate(
        self,
        prev_3d_pts: np.ndarray, curr_3d_pts: np.ndarray,
    ) -> np.ndarray:
        """Estime transformation matrix using SVD

        Args:
            prev_3d_pts (np.ndarray): The homogeneous 3D points in the previous frame
            curr_3d_pts (np.ndarray): The homogeneous 3D points in the current frame

        Returns:
            np.ndarray: Transformation matrix in homogeneous coordinates
        """
        avg_prev_3d_pts = np.mean(prev_3d_pts, axis=1).reshape((4, -1))
        avg_curr_3d_pts = np.mean(curr_3d_pts, axis=1).reshape((4, -1))

        R = self.rotation_estimate(prev_3d_pts - avg_prev_3d_pts, curr_3d_pts - avg_curr_3d_pts)
        t = avg_curr_3d_pts - R @ avg_prev_3d_pts
        T = np.eye(4)
        T[: 3, : 3] = R[: 3, : 3].T
        T[: 3, 3] = t[: 3, 0]
        return T

    def rotation_estimate(self, pts1: np.ndarray, pts2: np.ndarray):
        """Estimate rotation matrix by SVD

        Args:
            pts1 (np.ndarray): Points1 such as previous points
            pts2 (np.ndarray): Points2 such as current points
        """
        U, _, V = np.linalg.svd(pts1 @ pts2.T)
        S = np.eye(4)
        S[3, 3] = np.linalg.det(U @ V)  # For cope with reflection
        R = V.T @ S @ U.T
        return R


class RansacSvdBasedEstimator(SvdBasedEstimator):
    def __init__(self, P_l, max_trial: int = 100, early_termination_thd: int = 20, inlier_thd: float = 1.5):
        super().__init__(P_l)
        self.max_trial = max_trial
        self.early_termination_thd = early_termination_thd
        self.inlier_thd = inlier_thd

    def estimate(
        self,
        prev_pixes: np.ndarray, curr_pixes: np.ndarray,
        prev_3d_pts: np.ndarray, curr_3d_pts: np.ndarray,
    ) -> np.ndarray:
        prev_3d_pts = np.vstack((prev_3d_pts.T, np.ones((1, prev_3d_pts.shape[0]))))
        curr_3d_pts = np.vstack((curr_3d_pts.T, np.ones((1, curr_3d_pts.shape[0]))))

        min_error = 1e10
        early_termination = 0
        T = None
        sample_num = 3
        for _ in range(self.max_trial):
            sample_idx = np.random.choice(range(prev_3d_pts.shape[1]), sample_num)
            sample_prev_3d_pts = prev_3d_pts[:, sample_idx]
            sample_curr_3d_pts = curr_3d_pts[:, sample_idx]
            sample_T = self.svd_based_estimate(sample_prev_3d_pts, sample_curr_3d_pts)
            if sample_T is None or np.linalg.det(sample_T[:3, :3]) < 0:
                continue

            # Error estimation
            res = self.reprojection_residuals(sample_T, prev_pixes, curr_pixes, prev_3d_pts, curr_3d_pts)
            res = res.reshape((prev_3d_pts.shape[1] * 2, 2))
            error_pred = res[:prev_3d_pts.shape[1], :]  # Reprojection error against i to i-1
            error_curr = res[prev_3d_pts.shape[1]:, :]  # Reprojection error against i-1 to i

            # Find inliner and re-estimate
            inlier_idx = np.where(np.logical_and(error_pred < self.inlier_thd, error_curr < self.inlier_thd))[0]
            if len(inlier_idx) < 10:
                continue
            inliner_prev_3d_pts = prev_3d_pts[:, inlier_idx]
            inliner_curr_3d_pts = curr_3d_pts[:, inlier_idx]
            inliner_T = self.svd_based_estimate(inliner_prev_3d_pts, inliner_curr_3d_pts)
            if inliner_T is None or np.linalg.det(inliner_T[:3, :3]) < 0:
                continue

            res = self.reprojection_residuals(sample_T, prev_pixes[inlier_idx], curr_pixes[inlier_idx], prev_3d_pts[:, inlier_idx], curr_3d_pts[:, inlier_idx])
            res = res.reshape((len(inlier_idx) * 2, 2))
            error = np.mean(np.linalg.norm(res, axis=1))

            if error < min_error:
                min_error = error
                early_termination = 0
                T = inliner_T
            else:
                early_termination += 1
            if early_termination > self.early_termination_thd:
                break
        return T
