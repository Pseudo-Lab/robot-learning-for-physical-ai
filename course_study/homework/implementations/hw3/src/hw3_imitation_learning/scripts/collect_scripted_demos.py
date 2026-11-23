"""Collect successful pick-and-place demonstrations in MuJoCo.

The scripted teacher controls the mocap target and jaw actuator. With a checkpoint,
the learner acts first and the single-cube environment uses adversarial obstacle
placement. The Zarr dataset pairs each teacher command with the preceding observation.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import zarr

from hw3.eval_utils import (
    ZARR_KEY_TO_OBS,
    apply_action,
    check_success,
    check_wrong_cube_in_bin,
    infer_action_chunk,
    load_checkpoint,
)
from hw3.sim_env import CUBE_COLORS, SO100MulticubeSimEnv, SO100SimEnv
from so101_gym.constants import ASSETS_DIR

ACTIONS = ["action_ee_xyz", "action_gripper"]
OPEN, CLOSED = 1.1, -0.14


def waypoints(env):
    cube = env.get_cube_state()[:3]
    x, y = cube[:2]
    x -= 0.012  # The fixed finger is offset from the end-effector site.
    gx, gy = env.get_goal_pos()[:2]
    single = isinstance(env, SO100SimEnv)
    if not single:
        x -= 0.010  # Leave clearance for the fixed finger during descent.
        yield (x, y, 0.135), OPEN, 1
        yield (x, y, 0.062), OPEN, 2
        yield (x, y, 0.062), CLOSED, 7
        yield (x, y, 0.150), CLOSED, 2
        # Shift the drop point by the measured gripper-to-cube offset.
        offset = env.get_ee_pos()[:2] - env.get_cube_state()[:2]
        drop = env.get_goal_pos()[:2] + np.clip(offset, -0.045, 0.045)
        yield (*drop, 0.135), CLOSED, 2
        yield (*drop, 0.100), CLOSED, 2
        yield (*drop, 0.100), OPEN, 6
        yield (*drop, 0.165), OPEN, 4
        return
    ox, oy, _ = env.get_obstacle_pos()
    bypass = -0.315 if ox > -0.255 else -0.20
    carrying = cube[2] > 0.068 and env.get_gripper_angle() < 0.8
    if not carrying:
        if env.get_ee_pos()[1] > oy:
            yield (bypass, oy + 0.060, 0.150), OPEN, 1
            yield (bypass, y, 0.150), OPEN, 1
        yield (x, y, 0.135), OPEN, 1
        yield (x, y, 0.050), OPEN, 1
        yield (x, y, 0.050), CLOSED, 4
    yield (x, y, 0.150), CLOSED, 1
    if y < oy + 0.025:
        yield (bypass, y, 0.150), CLOSED, 1
        yield (bypass, oy + 0.075, 0.150), CLOSED, 1
    yield (gx - 0.012, gy, 0.135), CLOSED, 1
    yield (gx - 0.012, gy, 0.135), OPEN, 5
    yield (gx - 0.012, gy, 0.170), OPEN, 5


def demonstrate(env):
    records = []
    for target, gripper, dwell in waypoints(env):
        stable_steps = 0
        for _ in range(50):
            # Feedback compensates for the weld's compliance during transport.
            error = np.asarray(target) - env.get_ee_pos()
            delta = np.clip(0.65 * error, -0.010, 0.010)
            mocap = env.data.mocap_pos[env.mocap_id]
            delta = (
                np.clip(
                    mocap + delta, np.subtract(target, 0.045), np.add(target, 0.045)
                )
                - mocap
            )
            action = np.append(delta, gripper)
            records.append((env.get_obs(), action))
            apply_action(env, action, ACTIONS)
            env.step()
            stable_steps = stable_steps + 1 if np.max(np.abs(error)) < 0.005 else 0
            if stable_steps >= dwell:
                break
    return records


def learner_prefix(env, checkpoint, length, device):
    model, normalizer, _, keys, action_keys = checkpoint
    remaining = length
    while remaining > 0:
        chunk = infer_action_chunk(model, normalizer, env.get_obs(), keys, device)
        for action in chunk[:remaining]:
            apply_action(env, action, action_keys)
            env.step()
            remaining -= 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--seed", type=int, default=100)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--multicube", action="store_true")
    mode.add_argument("--dagger-checkpoint", type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.episodes < 1:
        parser.error("Choose a new output path and a positive episode count.")
    if args.multicube:
        env = SO100MulticubeSimEnv(
            ASSETS_DIR / "so100_multicube_ee.xml", seed=args.seed
        )
    else:
        env = SO100SimEnv(
            ASSETS_DIR / "so100_transfer_cube_obstacle_ee.xml",
            seed=args.seed,
            obstacle_mode="adversarial" if args.dagger_checkpoint else "train",
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = (
        load_checkpoint(args.dagger_checkpoint, device)
        if args.dagger_checkpoint
        else None
    )
    rng = np.random.default_rng(args.seed)
    keys = ["state_ee_xyz", "state_gripper", "state_cube", "state_obstacle", "goal_pos"]
    if args.multicube:
        keys += ["original_pos_cube_" + color for color in CUBE_COLORS] + ["state_goal"]
    arrays = {k: [] for k in keys + ACTIONS}
    ends, prefix_lengths = [], []
    attempts = 0
    try:
        while len(ends) < args.episodes and attempts < 8 * args.episodes:
            if args.multicube:
                env.set_goal(CUBE_COLORS[len(ends) % 3])
            env.reset()
            attempts += 1
            length = 0
            if checkpoint:
                length = int(rng.integers(3, 52))
                learner_prefix(env, checkpoint, length, device)
                if check_success(env):
                    continue
            records = demonstrate(env)
            if not check_success(env) or (
                args.multicube and check_wrong_cube_in_bin(env)
            ):
                continue
            if (
                env.get_gripper_angle() < 0.85
                or env.get_ee_pos()[2] - env.get_cube_state()[2] < 0.075
            ):
                continue  # Keep an episode only after the gripper opens and clears the cube.
            for obs, action in records:
                for k in keys:
                    arrays[k].append(ZARR_KEY_TO_OBS[k](obs))
                arrays["action_ee_xyz"].append(action[:3])
                arrays["action_gripper"].append(action[3:])
            ends.append(len(arrays["action_gripper"]))
            prefix_lengths.append(length)
            if len(ends) % 10 == 0:
                print(
                    f"Saved in memory: {len(ends)}/{args.episodes} from {attempts} attempts",
                    flush=True,
                )
    finally:
        env.renderer.close()
    if len(ends) != args.episodes:
        raise RuntimeError(
            f"Collected only {len(ends)} successful episodes from {attempts} attempts"
        )
    root = zarr.open_group(str(args.output), mode="w-")
    data, meta = root.create_group("data"), root.create_group("meta")
    for k, values in arrays.items():
        data.create_array(k, data=np.asarray(values, dtype=np.float32))
    meta.create_array("episode_ends", data=np.asarray(ends, dtype=np.int64))
    meta.create_array("learner_prefix_steps", data=np.asarray(prefix_lengths))
    root.attrs.update(
        {
            "source": "scripted MuJoCo expert",
            "seed": args.seed,
            "attempts": attempts,
            "num_dagger_episodes": len(ends) if checkpoint else 0,
            "learner_checkpoint": str(args.dagger_checkpoint or ""),
        }
    )
    print(
        f"Wrote {args.output}: {len(ends)} successes from {attempts} attempts, {ends[-1]} states"
    )


if __name__ == "__main__":
    main()
