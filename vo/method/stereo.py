import cv2
import warnings
import numpy as np
from liegroups import SE3
from scipy.optimize import least_squares
from vo.utils import form_transf
from vo.method import VoEstimator


class StereoVoEstimator(VoEstimator):
    def __init__(self, P_l):
        self.P_l = P_l

    def image_reprojection_residuals(
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
        transf_inv = np.linalg.inv(transf)
        f_projection = self.P_l @ transf
        b_projection = self.P_l @ transf_inv

        q1_pred = curr_3d_pts.T @ f_projection.T        # Project 3D points from i'th image to i-1'th image
        q1_pred = q1_pred[:, :2].T / q1_pred[:, 2]      # Un-homogenize
        q2_pred = prev_3d_pts.T @ b_projection.T        # Project 3D points from i-1'th image to i'th image
        q2_pred = q2_pred[:, :2].T / q2_pred[:, 2]      # Un-homogenize
        residuals = np.vstack([q1_pred - prev_pixes.T, q2_pred - curr_pixes.T]).flatten()  # Calculate the residuals
        return residuals

    def point_reprojection_residuals(
        self,
        transf: np.ndarray,
        prev_pixes: np.ndarray, curr_pixes: np.ndarray,
        prev_pts: np.ndarray, curr_pts: np.ndarray,
    ) -> np.ndarray:
        transf_inv = np.linalg.inv(transf)
        f_reprojection = transf @ prev_pts
        b_reprojection = transf_inv @ curr_pts
        e1 = curr_pts - f_reprojection
        e2 = prev_pts - b_reprojection
        residuals = np.vstack([e1[:3], e2[:3]]).flatten()
        return residuals


class LmBasedEstimator(StereoVoEstimator):
    def __init__(self, P_l, max_iter=100, manifold='rpy'):
        super().__init__(P_l)
        self.max_iter = max_iter
        self.iter_cnts = []
        self.min_erros = []
        self.manifold = manifold
        if manifold == 'rpy':
            self.optimize_function = self.rpy_optimize_cost
        elif manifold == 'se3':
            self.optimize_function = self.se3_optimize_cost
        else:
            warnings.warn('Invalid manifold type. Using rpy as default')
            self.optimize_function = self.rpy_optimize_cost
            self.manifold = 'rpy'

    def estimate(
        self,
        prev_pixes: np.ndarray, curr_pixes: np.ndarray,
        prev_pts: np.ndarray, curr_pts: np.ndarray
    ) -> np.ndarray:
        if prev_pixes.shape[0] < 3:
            return None

        # Initialize the min_error and early_termination counter
        min_error = float('inf')
        early_termination = 0
        early_termination_thd = 10

        xi_init = np.zeros(6)
        for cnt in range(self.max_iter):
            sample_idx = np.random.choice(range(prev_pixes.shape[0]), 20)
            sample_prev_pixes = prev_pixes[sample_idx]
            sample_curr_pixes = curr_pixes[sample_idx]
            sample_prev_pts = prev_pts[sample_idx]
            sample_curr_pts = curr_pts[sample_idx]

            # Perform least squares optimization
            opt_res = least_squares(
                self.optimize_function,                             # Function to minimize
                xi_init,                                           # Initial guess
                method='lm',                                        # Levenberg-Marquardt algorithm
                max_nfev=200,                                       # Max number of function evaluations
                args=(sample_prev_pixes, sample_curr_pixes, sample_prev_pts, sample_curr_pts)   # Additional arguments to pass to the function
            )

            # Calculate the error for the optimized transformation
            res = self.optimize_function(opt_res.x, prev_pixes, curr_pixes, prev_pts, curr_pts)
            res = res.reshape((prev_pts.shape[0] * 2, -1))
            err = np.sum(np.linalg.norm(res, axis=1))

            # Check if the error is less the the current min error. Save the result if it is
            if err < min_error:
                min_error = err
                out_pose = opt_res.x
                xi_init = out_pose
                early_termination = 0
            else:
                early_termination += 1
            if early_termination == early_termination_thd:
                # If we have not fund any better result in early_termination_threshold iterations
                break
        self.iter_cnts.append(cnt)
        self.min_erros.append(min_error)

        if self.manifold == 'rpy':
            r = out_pose[:3]             # Get the rotation vector
            R, _ = cv2.Rodrigues(r)      # Make the rotation matrix
            t = out_pose[3:]            # Get the translation vector
        elif self.manifold == 'se3':
            t_SE3 = SE3.exp(out_pose)
            R = t_SE3.rot.mat
            t = t_SE3.trans
        T = form_transf(R, t)
        return T

    def se3_optimize_cost(
        self,
        v: np.ndarray,
        prev_pixes: np.ndarray, curr_pixes: np.ndarray,
        prev_pts: np.ndarray, curr_pts: np.ndarray
    ) -> np.ndarray:
        """Optimization cost function for SE3 manifold

        Args:
            v (np.ndarray): Tangent vector
            prev_pixes (np.ndarray): Pixes in the previous frame
            curr_pixes (np.ndarray): Pixes in the current frame
            prev_3d_pts (np.ndarray): Corresponding 3D points in the previous frame
            curr_3d_pts (np.ndarray): Corresponding 3D points in the current frame

        Returns:
            np.ndarray: Loss vector
        """
        t_SE3 = SE3.exp(v)
        transf = form_transf(t_SE3.rot.mat, t_SE3.trans)
        ones = np.ones((prev_pixes.shape[0], 1))
        prev_pts = np.hstack([prev_pts, ones]).T
        curr_pts = np.hstack([curr_pts, ones]).T
        return self.image_reprojection_residuals(transf, prev_pixes, curr_pixes, prev_pts, curr_pts)

    def rpy_optimize_cost(
        self,
        dof: np.ndarray,
        prev_pixes: np.ndarray, curr_pixes: np.ndarray,
        prev_pts: np.ndarray, curr_pts: np.ndarray
    ) -> np.ndarray:
        """Optimization cost function for RPY manifold

        Args:
            dof (np.ndarray): Rotation vector and translation vector
            p_pixes (np.ndarray): Pixes in the previous frame
            c_pixes (np.ndarray): Pixes in the current frame
            p3d_pts (np.ndarray): Corresponding 3D points in the previous frame
            c3d_pts (np.ndarray): Corresponding 3D points in the current frame

        Returns:
            np.ndarray: Loss vector
        """
        r = dof[:3]                   # Get the rotation vector
        R, _ = cv2.Rodrigues(r)       # Create the rotation matrix from the rotation vector
        t = dof[3:]                   # Get the translation vector
        transf = form_transf(R, t)    # Create the transformation matrix from the rotation matrix and translation vector
        ones = np.ones((prev_pixes.shape[0], 1))
        prev_pts = np.hstack([prev_pts, ones]).T
        curr_pts = np.hstack([curr_pts, ones]).T
        return self.image_reprojection_residuals(transf, prev_pixes, curr_pixes, prev_pts, curr_pts)

    def save_results(self, src: str):
        self.iter_cnts = np.array(self.iter_cnts)
        self.min_erros = np.array(self.min_erros)
        np.savez(src, manifold=self.manifold, iter_cnts=self.iter_cnts, min_erros=self.min_erros)


class SvdBasedEstimator(StereoVoEstimator):
    def estimate(
        self,
        prev_pixes: np.ndarray, curr_pixes: np.ndarray,
        prev_pts: np.ndarray, curr_pts: np.ndarray,
    ) -> np.ndarray:
        prev_pts = np.vstack((prev_pts.T, np.ones((1, prev_pts.shape[0]))))
        curr_pts = np.vstack((curr_pts.T, np.ones((1, curr_pts.shape[0]))))
        T = self.svd_based_estimate(prev_pts, curr_pts)
        return T

    def svd_based_estimate(
        self,
        prev_pts: np.ndarray, curr_pts: np.ndarray,
    ) -> np.ndarray:
        """Estime transformation matrix using SVD

        Args:
            prev_3d_pts (np.ndarray): The homogeneous 3D points in the previous frame
            curr_3d_pts (np.ndarray): The homogeneous 3D points in the current frame

        Returns:
            np.ndarray: Transformation matrix in homogeneous coordinates
        """
        prev_pts = prev_pts[: 3, :]
        curr_pts = curr_pts[: 3, :]
        avg_prev_pts = np.mean(prev_pts, axis=1).reshape((3, -1))
        avg_curr_pts = np.mean(curr_pts, axis=1).reshape((3, -1))

        R = self.rotation_estimate(prev_pts - avg_prev_pts, curr_pts - avg_curr_pts)
        t = avg_curr_pts - R @ avg_prev_pts
        T = np.eye(4)
        T[: 3, : 3] = R
        T[: 3, 3] = t[: 3, 0]
        return T

    def rotation_estimate(self, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
        """Estimate rotation matrix using SVD

        Args:
            pts1 (np.ndarray): Points1 such as previous points
            pts2 (np.ndarray): Points2 such as current points

        Returns:
            np.ndarray: Rotation matrix in shape 3x3
        """
        U, _, V = np.linalg.svd(pts1 @ pts2.T)
        S = np.eye(3)
        S[2, 2] = np.linalg.det(V.T @ U.T)  # To cope with reflection
        R = V.T @ S @ U.T
        return R

    def save_results(self, *args, **kwargs):
        print("SvdBasedEstimator : No results to save")


class RansacSvdBasedEstimator(SvdBasedEstimator):
    def __init__(self, P_l, max_iter: int = 20, inlier_thd: float = 0.01):
        super().__init__(P_l)
        self.max_iter = max_iter
        self.inlier_thd = inlier_thd

    def estimate(
        self,
        prev_pixes: np.ndarray, curr_pixes: np.ndarray,
        prev_pts: np.ndarray, curr_pts: np.ndarray,
    ) -> np.ndarray:
        # Add ones to the end of the points to make them homogeneous
        prev_pts = np.vstack((prev_pts.T, np.ones((1, prev_pts.shape[0]))))
        curr_pts = np.vstack((curr_pts.T, np.ones((1, curr_pts.shape[0]))))

        max_inlier_num = 0
        sample_num = 3
        T = None
        if prev_pts.shape[1] < sample_num:
            return T
        for _ in range(self.max_iter):
            # Sample 3 points and estimate
            sample_idx = np.random.choice(range(prev_pts.shape[1]), sample_num, replace=False)
            sample_T = self.svd_based_estimate(prev_pts[:, sample_idx], curr_pts[:, sample_idx])

            # Error estimation
            res = self.point_reprojection_residuals(sample_T, prev_pixes, curr_pixes, prev_pts, curr_pts)
            res = res.reshape((prev_pts.shape[1] * 2, -1))
            error_pred_flag = np.all(res[:prev_pts.shape[1], :] < self.inlier_thd, axis=1)
            error_curr_flag = np.all(res[prev_pts.shape[1]:, :] < self.inlier_thd, axis=1)

            # Find inliner and re-estimate
            inlier_idx = np.where(np.logical_and(error_pred_flag, error_curr_flag))[0]
            if len(inlier_idx) < 10:
                continue
            inlier_T = self.svd_based_estimate(prev_pts[:, inlier_idx], curr_pts[:, inlier_idx])

            # Count up the number of inliers
            res = self.point_reprojection_residuals(inlier_T, prev_pixes[inlier_idx], curr_pixes[inlier_idx], prev_pts[:, inlier_idx], curr_pts[:, inlier_idx])
            res = res.reshape((len(inlier_idx) * 2, -1))
            error_pred = np.linalg.norm(res[:len(inlier_idx), :], axis=1)  # Reprojection error against i to i-1
            error_curr = np.linalg.norm(res[len(inlier_idx):, :], axis=1)  # Reprojection error against i-1 to i
            inlier_idx_ = np.where(np.logical_and(error_pred < self.inlier_thd, error_curr < self.inlier_thd))[0]
            if len(inlier_idx_) > max_inlier_num:
                max_inlier_num = len(inlier_idx_)
                T = inlier_T

        # Inversion is needed because T will be multiplied from the right
        if T is not None:
            T = np.linalg.inv(T)

        return T

    def save_results(self, src: str):
        np.savez(
            src,
            max_iter=self.max_iter,
            inlier_thd=self.inlier_thd
        )
