from collections import deque
from pathlib import Path
import time
import mujoco
import mujoco.viewer
import numpy as np
from hw2_robot_control_mdps.exercises.ex1 import build_keypoints, ik_track
from hw2_robot_control_mdps.exercises.ex2 import (
    generate_quintic_spline_waypoints,
    pid_control,
)
from hw2_robot_control_mdps.scripts.utils import refresh_markers


def track():
    assets = Path(__file__).resolve().parents[1] / "so101_gym/assets"
    model = mujoco.MjModel.from_xml_path(str(assets / "so100_torque_ctrl.xml"))
    data = mujoco.MjData(model)
    points = build_keypoints()
    route = np.vstack(
        [
            generate_quintic_spline_waypoints(a, b, 5)
            for a, b in zip(points, np.roll(points, -1, axis=0))
        ]
    )
    planner = mujoco.MjData(model)
    planner.qpos[:] = model.jnt_range.mean(axis=1)
    planner.qpos[-1] = data.qpos[-1]
    commands = []
    for target in route:
        q = ik_track(model, planner, "ee_site", target)
        commands.append(q)
        planner.qpos[:] = q
    data.qpos[:] = commands[0]
    mujoco.mj_forward(model, data)
    idx = 0
    history = deque(maxlen=10)

    def control(model, data):
        history.append(commands[idx] - data.qpos)
        data.ctrl[:] = pid_control(np.array(history), model.opt.timestep)

    mujoco.set_mjcb_control(control)
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            refresh_markers(viewer, points)
            refresh_markers(
                viewer, route, radius=0.003, rgba=(0, 1, 1, 1), ngeom_start=len(points)
            )
            while viewer.is_running():
                data.mocap_pos[0] = route[idx]
                mujoco.mj_step(model, data)
                viewer.sync()
                if np.linalg.norm(data.site("ee_site").xpos - route[idx]) < 0.03:
                    history.clear()
                    idx = (idx + 1) % len(route)
                time.sleep(model.opt.timestep)
        return viewer, data
    finally:
        mujoco.set_mjcb_control(None)


if __name__ == "__main__":
    viewer, data = track()
