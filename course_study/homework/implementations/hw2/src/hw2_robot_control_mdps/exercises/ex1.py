import mujoco
import numpy as np


def get_lemniscate_keypoint(t, a=0.2):
    denominator = 1.0 + np.sin(t) ** 2
    y = a * np.cos(t) / denominator
    z = a * np.cos(t) * np.sin(t) / denominator
    return (y, z)


def build_keypoints(count=16, width=0.25, x_offset=0.3, z_offset=0.25):
    t = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    y, z = get_lemniscate_keypoint(t, a=width)
    return np.column_stack((np.full(count, x_offset, dtype=float), y, z + z_offset))


def ik_track(
    model,
    data,
    site_name,
    target_pos,
    damping=1e-3,
    pos_gain=2.0,
    dt=0.1,
    max_iters=2000,
):
    position, velocity = data.qpos.copy(), data.qvel.copy()
    site_id = model.site(site_name).id
    limited = model.jnt_limited.astype(bool)
    bounds = model.jnt_range[limited]
    addresses = model.jnt_qposadr[limited]
    try:
        for _ in range(max_iters):
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            error = np.asarray(target_pos) - data.site(site_name).xpos
            if np.linalg.norm(error) < 0.001:
                break
            jacobian = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jacobian, None, site_id)
            system = jacobian @ jacobian.T + damping * np.eye(3)
            delta = jacobian.T @ np.linalg.solve(system, pos_gain * error)
            data.qpos[:] += dt * delta.clip(-2.0, 2.0)
            data.qpos[addresses] = data.qpos[addresses].clip(bounds[:, 0], bounds[:, 1])
        return data.qpos.copy()
    finally:
        data.qpos[:] = position
        data.qvel[:] = velocity
        mujoco.mj_forward(model, data)
