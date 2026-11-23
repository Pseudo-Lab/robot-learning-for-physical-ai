import numpy as np
import zarr
from ethz_starter.hw3_imitation_learning.hw3.dataset import (
    Normalizer,
    SO100ChunkDataset,
    build_valid_indices,
    _parse_key_spec,
)


def target_offset(cubes_xyz, goal, ee_xyz):
    return (cubes_xyz * goal[..., :, None]).sum(axis=-2) - ee_xyz


def load_zarr(zarr_path, state_keys=None, action_keys=None):
    group = zarr.open_group(str(zarr_path), mode="r")
    data = group["data"]

    def columns(specs):
        parts = []
        for spec in specs:
            name, selection = _parse_key_spec(spec)
            if name == "state_target_offset":
                cubes = np.stack(
                    [
                        data[f"original_pos_cube_{c}"][:, :3]
                        for c in ("red", "green", "blue")
                    ],
                    axis=1,
                )
                values = target_offset(
                    cubes, data["state_goal"][:], data["state_ee_xyz"][:]
                )
            elif name == "state_bin_offset":
                values = data["goal_pos"][:, :3] - data["state_ee_xyz"][:]
            else:
                values = data[name][:]
            values = np.asarray(values, dtype=np.float32)
            parts.append(values if selection == slice(None) else values[:, selection])
        return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1)

    return (
        columns(
            [group.attrs.get("state_key", "state")]
            if state_keys is None
            else state_keys
        ),
        columns(
            [group.attrs.get("action_key", "action")]
            if action_keys is None
            else action_keys
        ),
        np.asarray(group["meta"]["episode_ends"][:], dtype=np.int64),
    )


def load_and_merge_zarrs(zarr_paths, state_keys=None, action_keys=None):
    states, actions, boundaries = [], [], []
    length = 0
    for path in zarr_paths:
        x, y, ends = load_zarr(path, state_keys, action_keys)
        states.append(x)
        actions.append(y)
        boundaries.append(ends + length)
        length += len(x)
    return tuple(np.concatenate(items) for items in (states, actions, boundaries))
