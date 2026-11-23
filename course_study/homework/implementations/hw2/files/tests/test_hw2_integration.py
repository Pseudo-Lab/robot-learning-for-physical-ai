from pathlib import Path
import importlib
import os
import runpy
import subprocess
import sys

import mujoco
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from conftest import add_homework_to_path

HW = add_homework_to_path("hw2_robot_control_mdps")
Env = importlib.import_module(
    "hw2_robot_control_mdps.env.so100_tracking_env"
).SO100TrackEnv
XML = HW / "so101_gym/assets/so100_pos_ctrl.xml"


def test_environment_contract_and_seed_isolation():
    env = Env(XML)
    try:
        check_env(env, skip_render_check=True)
        a, _ = env.reset(seed=42)
        for _ in range(20):
            np.random.uniform(size=10)
        b, _ = env.reset(seed=42)
        np.testing.assert_array_equal(a, b)
        assert a.dtype == env.observation_space.dtype == np.float32
        assert np.all(env.data.qpos >= env.model.jnt_range[:, 0])
        assert np.all(env.data.qpos <= env.model.jnt_range[:, 1])
        for i in range(env.max_episode_length):
            obs, reward, terminated, truncated, info = env.step(np.zeros(6))
            assert np.isfinite(obs).all() and np.isfinite(reward)
            assert truncated == (i == env.max_episode_length - 1)
        assert not terminated
    finally:
        env.close()


@pytest.mark.parametrize(
    "script", ["train.py", "evaluate_rand_targets.py", "evaluate_trajectory.py"]
)
def test_entrypoint_from_outside_repository(script, tmp_path):
    result = subprocess.run(
        [sys.executable, str(HW / "scripts" / script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--headless" in result.stdout


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="No GUI/OpenGL display")
def test_rgb_array_render_and_close():
    env = Env(XML, render_mode="rgb_array")
    try:
        env.reset(seed=0)
        image = env.render()
        assert image.shape == (480, 640, 3)
        assert image.dtype == np.uint8 and image.std() > 1
    finally:
        env.close()
    assert env.renderer is None


def test_pid_script_tracks_waypoints_without_gui(monkeypatch):
    import time
    import mujoco.viewer

    ex1 = importlib.import_module("hw2_robot_control_mdps.exercises.ex1")
    ex2 = importlib.import_module("hw2_robot_control_mdps.exercises.ex2")
    keypoints = ex1.build_keypoints()
    expected = {
        tuple(np.round(p, 6))
        for a, b in zip(keypoints, np.roll(keypoints, -1, axis=0))
        for p in ex2.generate_quintic_spline_waypoints(a, b, 5)
    }

    class Viewer:
        def __init__(self, model, data):
            self.user_scn = mujoco.MjvScene(model, maxgeom=1000)
            self.steps = 0
            self.data = data
            self.reached = set()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def is_running(self):
            self.steps += 1
            return self.steps <= 60000 and self.reached != expected

        def sync(self):
            target = self.data.mocap_pos[0]
            if np.linalg.norm(self.data.site("ee_site").xpos - target) < 0.03:
                self.reached.add(tuple(np.round(target, 6)))

    monkeypatch.syspath_prepend(str(HW / "scripts"))
    monkeypatch.setattr(mujoco.viewer, "launch_passive", Viewer)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    state = runpy.run_path(str(HW / "scripts/pid_control.py"), run_name="__main__")
    assert state["viewer"].reached == expected, expected - state["viewer"].reached
    assert np.isfinite(state["data"].qpos).all()
    assert not any(w.number for w in state["data"].warning)

    assert mujoco.get_mjcb_control() is None


def test_ik_preserves_joint_positions_and_velocities():
    ex1 = importlib.import_module("hw2_robot_control_mdps.exercises.ex1")
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    data.qvel[:] = np.arange(model.nv) * 0.02
    mujoco.mj_forward(model, data)
    q, v = data.qpos.copy(), data.qvel.copy()
    result = ex1.ik_track(model, data, "ee_site", ex1.build_keypoints()[0])
    np.testing.assert_array_equal(data.qpos, q)
    np.testing.assert_array_equal(data.qvel, v)
    assert np.isfinite(result).all()


def test_ik_full_keypoint_path_respects_joint_limits():
    ex1 = importlib.import_module("hw2_robot_control_mdps.exercises.ex1")
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    for target in ex1.build_keypoints():
        q = ex1.ik_track(model, data, "ee_site", target)
        assert np.all(q >= model.jnt_range[:, 0] - 1e-9)
        assert np.all(q <= model.jnt_range[:, 1] + 1e-9)
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        assert np.linalg.norm(data.site("ee_site").xpos - target) < 0.005
