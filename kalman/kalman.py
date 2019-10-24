import cv2
import numpy as np

def prediction(X_hat_t_1, P_t_1, F_t, B_t, U_t, Q_t):
    """
    Predicts the next 
    """
    X_hat_t = F_t.dot(X_hat_t_1) + (B_t.dot(U_t).reshape(B_t.shape[0], -1))
    P_t = np.diag(np.diag(F_t.dot(P_t_1).dot(F_t.transpose()))) + Q_t
    return X_hat_t, P_t

def update(X_hat_t, P_t, Z_t, R_t, H_t):
    