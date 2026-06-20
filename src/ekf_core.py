"""Core Extended Kalman Filter localization utilities.

The EKF equations in this module are adapted from the PythonRobotics
``extended_kalman_filter.py`` implementation by Atsushi Sakai and contributors.
This version keeps only the standalone estimation and simulation primitives
needed for benchmarking and validation workflows.
"""

from __future__ import annotations

import math

import numpy as np


# Covariance used during EKF prediction.
Q = np.diag(
    [
        0.1,
        0.1,
        np.deg2rad(1.0),
        1.0,
    ]
) ** 2

# Observation covariance for x-y position measurements.
R = np.diag([1.0, 1.0]) ** 2

# Simulation noise applied to control input and GPS-like observations.
INPUT_NOISE = np.diag([1.0, np.deg2rad(30.0)]) ** 2
GPS_NOISE = np.diag([0.5, 0.5]) ** 2

# Simulation timestep in seconds.
DT = 0.1


def calc_input() -> np.ndarray:
    """Return the nominal control input vector ``[velocity, yaw_rate]``."""

    velocity = 1.0
    yaw_rate = 0.1
    return np.array([[velocity], [yaw_rate]])


def motion_model(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Propagate the state forward using the nonlinear motion model."""

    state_transition = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    control_matrix = np.array(
        [
            [DT * math.cos(x[2, 0]), 0.0],
            [DT * math.sin(x[2, 0]), 0.0],
            [0.0, DT],
            [1.0, 0.0],
        ]
    )
    return state_transition @ x + control_matrix @ u


def observation_model(x: np.ndarray) -> np.ndarray:
    """Project the state into the observed x-y measurement space."""

    observation_matrix = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )
    return observation_matrix @ x


def jacob_f(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Compute the Jacobian of the nonlinear motion model.

    The underlying equations are adapted from PythonRobotics by Atsushi Sakai.
    """

    yaw = x[2, 0]
    velocity = u[0, 0]
    return np.array(
        [
            [1.0, 0.0, -DT * velocity * math.sin(yaw), DT * math.cos(yaw)],
            [0.0, 1.0, DT * velocity * math.cos(yaw), DT * math.sin(yaw)],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def jacob_h() -> np.ndarray:
    """Return the Jacobian of the observation model."""

    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )


def observation(
    x_true: np.ndarray,
    x_dead_reckoning: np.ndarray,
    u: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Advance the simulation and generate noisy measurement data.

    Returns:
        A tuple containing the updated true state, noisy observation,
        dead-reckoned state, and noisy control input.
    """

    x_true = motion_model(x_true, u)
    z = observation_model(x_true) + GPS_NOISE @ np.random.randn(2, 1)
    u_noisy = u + INPUT_NOISE @ np.random.randn(2, 1)
    x_dead_reckoning = motion_model(x_dead_reckoning, u_noisy)
    return x_true, z, x_dead_reckoning, u_noisy


def ekf_estimation(
    x_est: np.ndarray,
    p_est: np.ndarray,
    z: np.ndarray,
    u: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one prediction-update step of the Extended Kalman Filter."""

    x_pred = motion_model(x_est, u)
    j_f = jacob_f(x_est, u)
    p_pred = j_f @ p_est @ j_f.T + Q

    j_h = jacob_h()
    z_pred = observation_model(x_pred)
    innovation = z - z_pred
    innovation_covariance = j_h @ p_pred @ j_h.T + R
    kalman_gain = p_pred @ j_h.T @ np.linalg.inv(innovation_covariance)

    x_est = x_pred + kalman_gain @ innovation
    p_est = (np.eye(len(x_est)) - kalman_gain @ j_h) @ p_pred
    return x_est, p_est


def compute_position_error(x_est: np.ndarray, x_true: np.ndarray) -> float:
    """Return Euclidean position error between estimated and true x-y state."""

    delta_x = float(x_est[0, 0] - x_true[0, 0])
    delta_y = float(x_est[1, 0] - x_true[1, 0])
    return math.hypot(delta_x, delta_y)


def covariance_trace(p_est: np.ndarray) -> float:
    """Return the trace of the EKF covariance matrix."""

    return float(np.trace(p_est))
