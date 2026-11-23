from __future__ import annotations

import importlib
import sys

import mujoco
import numpy as np
from stable_baselines3 import PPO

from conftest import add_homework_to_path, requested_device


HW_ROOT = add_homework_to_path("hw2_robot_control_mdps")
ex1 = importlib.import_module("hw2_robot_control_mdps.exercises.ex1")
ex2 = importlib.import_module("hw2_robot_control_mdps.exercises.ex2")
ex3 = importlib.import_module("hw2_robot_control_mdps.exercises.ex3")


def test_lemniscate_and_keypoints() -> None:
    y, z = ex1.get_lemniscate_keypoint(np.array([0.0, np.pi / 2]))
    np.testing.assert_allclose(y, [0.2, 0.0], atol=1e-8)
    np.testing.assert_allclose(z, [0.0, 0.0], atol=1e-8)
    points = ex1.build_keypoints(count=8)
    assert points.shape == (8, 3)
    assert np.unique(points, axis=0).shape[0] == 8


def test_damped_least_squares_ik_reaches_reachable_offset() -> None:
    model = mujoco.MjModel.from_xml_path(
        str(HW_ROOT / "so101_gym/assets/so100_pos_ctrl.xml")
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    site_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, index)
        for index in range(model.nsite)
    ]
    site_name = "gripperframe" if "gripperframe" in site_names else site_names[-1]
    start = data.site(site_name).xpos.copy()
    target = start + np.array([0.0, 0.01, 0.01])
    q_target = ex1.ik_track(model, data, site_name, target, max_iters=1000)
    original = data.qpos.copy()
    data.qpos[:] = q_target
    mujoco.mj_forward(model, data)
    residual = np.linalg.norm(data.site(site_name).xpos - target)
    data.qpos[:] = original
    assert residual < 1e-3


def test_quintic_boundary_conditions_and_pid() -> None:
    start = np.array([0.0, -1.0])
    end = np.array([2.0, 3.0])
    points = ex2.generate_quintic_spline_waypoints(start, end, 101)
    np.testing.assert_allclose(points[0], start)
    np.testing.assert_allclose(points[-1], end)
    np.testing.assert_allclose(points[1] - points[0], 0.0, atol=1e-4)
    np.testing.assert_allclose(points[-1] - points[-2], 0.0, atol=1e-4)
    signal = ex2.pid_control(np.array([[1.0], [0.5]]), 0.1, Kp=2, Ki=1, Kd=0.5)
    np.testing.assert_allclose(signal, [-1.35])

    state = np.array([0.0])
    target = np.array([1.0])
    history = []
    initial_error = float(np.linalg.norm(target - state))
    for _ in range(20):
        history.append(target - state)
        control = ex2.pid_control(np.asarray(history), 0.05, Kp=2.0, Ki=0.0, Kd=0.0)
        state += 0.05 * control
    assert np.linalg.norm(target - state) < initial_error * 0.15


def test_mdp_helpers_and_base_frame_observation() -> None:
    np.random.seed(7)
    default = np.zeros(5)
    reset = ex3.reset_robot(default)
    assert reset.shape == (5,)
    assert np.all((-0.5 <= reset) & (reset <= 0.5))
    target = ex3.reset_target_position(np.array([1.0, 2.0, 3.0]))
    assert 1.2 <= target[0] <= 1.4
    assert 1.8 <= target[1] <= 2.2
    assert 3.1 <= target[2] <= 3.4
    mapped = ex3.process_action(
        np.array([-1.0, 0.0, 1.0]),
        np.array([[0.0, 2.0], [-2.0, 2.0], [4.0, 8.0]]),
    )
    np.testing.assert_allclose(mapped, [0.0, 0.0, 8.0])
    assert ex3.compute_reward(0.0) == 2.0

    theta = np.pi / 2
    rot_z = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    obs = ex3.get_obs(
        qpos=np.array([0.1, 0.2]),
        ee_pos_w=np.array([1.0, 1.0, 0.0]),
        ee_rot_w=rot_z,
        base_pos_w=np.array([1.0, 0.0, 0.0]),
        base_rot_w=rot_z,
        target_pos_w=np.array([2.0, 0.0, 0.0]),
    )
    assert obs.shape == (12,)
    np.testing.assert_allclose(obs[2:5], [1.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(obs[5:9]), 1.0, atol=1e-6)
    np.testing.assert_allclose(obs[9:12], [0.0, -1.0, 0.0], atol=1e-6)

    environment_module = importlib.import_module("env.so100_tracking_env")
    env = environment_module.SO100TrackEnv(
        xml_path=HW_ROOT / "so101_gym/assets/so100_pos_ctrl.xml",
        render_mode=None,
    )
    reset_obs, _ = env.reset(seed=3)
    assert env.observation_space.contains(reset_obs)
    for _ in range(4):
        reset_obs, reward, terminated, truncated, info = env.step(
            env.action_space.sample()
        )
        assert np.isfinite(reward)
        assert np.isfinite(info["ee_tracking_error"])

    device = requested_device()
    learner = PPO(
        "MlpPolicy",
        env,
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        policy_kwargs={"net_arch": [16]},
        device=device,
        verbose=0,
    )
    learner.learn(total_timesteps=16)
    env.close()
