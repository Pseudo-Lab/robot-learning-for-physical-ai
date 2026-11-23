from pathlib import Path
import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from hw2_robot_control_mdps.exercises import ex3


class SO100TrackEnv(gym.Env):
    metadata = dict(render_modes=["human", "rgb_array"], render_fps=30)

    def __init__(self, xml_path, render_mode=None):
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError(f"Unsupported render mode: {render_mode}")
        self.xml_path = Path(xml_path).resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.viewer = self.renderer = None
        self.ctrl_decimation = 50
        self.sim_timestep = self.model.opt.timestep
        self.ctrl_timestep = self.ctrl_decimation * self.sim_timestep
        self.max_episode_length_s = 10
        self.max_episode_length = int(self.max_episode_length_s / self.ctrl_timestep)
        self.default_qpos = np.array([0, -1.57, 1, 1, 0, 0.02239])
        self.current_step = 0
        self.ee_tracking_error = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, self._get_obs().shape, np.float32
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, (self.model.nu,), np.float32)

    def _get_obs(self):
        tip, base = self.data.site("ee_site"), self.data.body("Base")
        return ex3.get_obs(
            self.data.qpos,
            tip.xpos,
            tip.xmat.reshape(3, 3),
            base.xpos,
            base.xmat.reshape(3, 3),
            self.data.mocap_pos[0],
        )

    def _distance(self):
        return float(
            np.linalg.norm(self.data.site("ee_site").xpos - self.data.mocap_pos[0])
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        mujoco.mj_resetData(self.model, self.data)
        low, high = self.model.jnt_range.T
        self.data.qpos[:] = ex3.reset_robot(self.default_qpos, self.np_random).clip(
            low, high
        )
        self.data.ctrl[:] = self.data.qpos
        mujoco.mj_forward(self.model, self.data)
        self.data.mocap_pos[0] = ex3.reset_target_position(
            self.data.body("Base").xpos, self.np_random
        )
        self.ee_tracking_error = self._distance()
        self.current_step = 0
        return self._get_obs(), {}

    def _process_action(self, action):
        return ex3.process_action(action, self.model.jnt_range)

    def compute_reward(self):
        return ex3.compute_reward(self.ee_tracking_error)

    def step(self, action):
        self.data.ctrl[:] = self._process_action(action)
        mujoco.mj_step(self.model, self.data, nstep=self.ctrl_decimation)
        self.ee_tracking_error = self._distance()
        self.current_step += 1
        if self.render_mode == "human":
            self.render()
        return (
            self._get_obs(),
            self.compute_reward(),
            False,
            self.current_step >= self.max_episode_length,
            {"ee_tracking_error": self.ee_tracking_error},
        )

    def render(self):
        if self.render_mode == "rgb_array":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            self.renderer.update_scene(self.data)
            return self.renderer.render()
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()

    def close(self):
        for name in ("viewer", "renderer"):
            window = getattr(self, name)
            if window is not None:
                window.close()
                setattr(self, name, None)
