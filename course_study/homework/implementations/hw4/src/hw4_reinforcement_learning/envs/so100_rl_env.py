import mujoco
import numpy as np
from ethz_starter.hw4_reinforcement_learning.envs.so100_rl_env import (
    SO100RLEnv as CourseEnvironment,
)


class SO100RLEnv(CourseEnvironment):
    def step(self, action):
        self.data.ctrl[:] = self._process_action(np.asarray(action, dtype=np.float32))
        mujoco.mj_step(self.model, self.data, nstep=self.ctrl_decimation)
        error = self.data.site("ee_site").xpos - self.data.mocap_pos[0]
        self.ee_tracking_error = float(np.linalg.norm(error))
        self.current_step += 1
        if self.render_mode == "human":
            self.render()
        return (
            self._get_obs(),
            self._compute_reward(),
            False,
            self.current_step >= self.max_episode_length,
            {"ee_tracking_error": self.ee_tracking_error},
        )
