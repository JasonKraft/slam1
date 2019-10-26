import cv2
import numpy as np

class KalmanFilter:
    def __init__(self, F_0, P_0, Q_0, B_0, U_0, H_0, R_0, X_hat_0):
        """
        Initializes a new instance of KalmanFilter.

        Parameters
        ----------
        F_0 : matrix
            The transition matrix.
        P_0 : matrix
            The initial state covariance matrix. Measures the certainty of our prediction.
        """
        self.F_t = F_0
        self.P_t = P_0
        self.Q_t = Q_0
        self.B_t = B_0
        self.U_t = U_0
        self.H_t = H_0
        self.R_t = R_0
        self.X_hat_t = X_hat_0

    def predict(self, X_hat_t_1, P_t_1, F_t = None, B_t = None, U_t = None, Q_t = None):
        """
        Predicts the state of our world at the current timestep based on
        the prediction and measurements from the previous timestep.
        """
        if F_t is None:
            F_t = self.F_t
        
        if B_t is None:
            B_t = self.B_t
        
        if U_t is None:
            U_t = self.U_t

        if Q_t is None:
            Q_t = self.Q_t

        X_hat_t = F_t.dot(X_hat_t_1) + (B_t.dot(U_t).reshape(B_t.shape[0], -1))
        self.P_t = np.diag(np.diag(F_t.dot(P_t_1).dot(F_t.transpose()))) + Q_t
        return X_hat_t, self.P_t

    def update(self, X_hat_t, P_t, Z_t, R_t, H_t):
        """
        Refines our prediction of the world state at the current timestep
        using new measurements.
        """
        K_prime = P_t.dot(H_t.transpose()).dot(np.linalg.inv(H_t.dot(P_t).dot(H_t.transpose()) + R_t))
        print("K:\n", K_prime)
        
        X_t = X_hat_t + K_prime.dot(Z_t - H_t.dot(X_hat_t))
        self.P_t = self.P_t - K_prime.dot(H_t).dog(P_t)

        return X_t, P_t

