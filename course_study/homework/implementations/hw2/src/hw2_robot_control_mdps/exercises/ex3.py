import numpy as np
from hw2_robot_control_mdps.scripts.utils import (
    quat_conjugate,
    quat_mul,
    quat_normalize,
    rot_mat_to_quat,
)


def reset_robot(default_qpos: np.ndarray, rng=None) -> np.ndarray:
    rng = np.random if rng is None else rng
    return np.asarray(default_qpos) + rng.uniform(
        -0.5, 0.5, size=np.asarray(default_qpos).shape
    )


def reset_target_position(base_pos: np.ndarray, rng=None) -> np.ndarray:
    rng = np.random if rng is None else rng
    offset = np.array(
        [rng.uniform(0.2, 0.4), rng.uniform(-0.2, 0.2), rng.uniform(0.1, 0.4)]
    )
    return np.asarray(base_pos) + offset


def process_action(action: np.ndarray, jnt_range: np.ndarray) -> np.ndarray:
    action = np.clip(np.asarray(action), -1.0, 1.0)
    joint_range = np.asarray(jnt_range)
    low = joint_range[:, 0]
    high = joint_range[:, 1]
    return low + 0.5 * (action + 1.0) * (high - low)


def compute_reward(ee_tracking_error: float) -> float:
    dense_reward = np.exp(-2.0 * ee_tracking_error)
    sparse_reward = 1.0 if ee_tracking_error < 0.005 else 0.0
    return float(dense_reward + sparse_reward)


def get_obs(
    qpos: np.ndarray,
    ee_pos_w: np.ndarray,
    ee_rot_w: np.ndarray,
    base_pos_w: np.ndarray,
    base_rot_w: np.ndarray,
    target_pos_w: np.ndarray,
) -> np.ndarray:
    base_rot_inv = np.asarray(base_rot_w).T
    ee_pos_base = base_rot_inv @ (np.asarray(ee_pos_w) - np.asarray(base_pos_w))
    target_pos_base = base_rot_inv @ (np.asarray(target_pos_w) - np.asarray(base_pos_w))
    ee_quat_w = rot_mat_to_quat(np.asarray(ee_rot_w))
    base_quat_w = rot_mat_to_quat(np.asarray(base_rot_w))
    ee_quat_base = quat_normalize(quat_mul(quat_conjugate(base_quat_w), ee_quat_w))
    return np.concatenate(
        (np.asarray(qpos), ee_pos_base, ee_quat_base, target_pos_base)
    ).astype(np.float32)
