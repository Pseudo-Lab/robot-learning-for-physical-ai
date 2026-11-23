from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import numpy as np
import torch
import zarr
from torch.utils.data import DataLoader

from conftest import add_homework_to_path, requested_device


add_homework_to_path("hw3_imitation_learning")
dataset_module = importlib.import_module("hw3.dataset")
model_module = importlib.import_module("hw3.model")
train_path = (
    Path(__file__).resolve().parents[1]
    / "hw3_imitation_learning"
    / "scripts"
    / "train.py"
)
train_spec = importlib.util.spec_from_file_location("hw3_training_script", train_path)
assert train_spec is not None and train_spec.loader is not None
train_module = importlib.util.module_from_spec(train_spec)
train_spec.loader.exec_module(train_module)


def synthetic_dataset(chunk_size: int = 4):
    rng = np.random.default_rng(4)
    states = rng.normal(size=(64, 6)).astype(np.float32)
    weight = rng.normal(size=(6, 3)).astype(np.float32)
    actions = (states @ weight).astype(np.float32)
    episode_ends = np.array([32, 64])
    normalizer = dataset_module.Normalizer.from_data(states, actions)
    return dataset_module.SO100ChunkDataset(
        states, actions, episode_ends, chunk_size, normalizer
    )


def test_dataset_never_crosses_episode_boundary() -> None:
    dataset = synthetic_dataset(chunk_size=4)
    assert len(dataset) == 58
    assert 29 not in dataset.indices
    assert 30 not in dataset.indices
    assert 31 not in dataset.indices
    state, action_chunk = dataset[0]
    assert state.shape == (6,)
    assert action_chunk.shape == (4, 3)


def train_policy(policy_type: str, tmp_path: Path) -> tuple[float, float]:
    device = requested_device()
    dataset = synthetic_dataset()
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    policy = model_module.build_policy(
        policy_type, state_dim=6, action_dim=3, chunk_size=4, d_model=32
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
    initial = None
    for _ in range(12):
        total = train_module.train_one_epoch(policy, loader, optimizer, device)
        initial = total if initial is None else initial
    assert total < initial
    validation_loss = train_module.evaluate(policy, loader, device)
    assert np.isfinite(validation_loss)
    sample = policy.sample_actions(torch.zeros(2, 6, device=device))
    assert sample.shape == (2, 4, 3)
    checkpoint = tmp_path / f"{policy_type}.pt"
    torch.save(policy.state_dict(), checkpoint)
    clone = model_module.build_policy(
        policy_type, state_dim=6, action_dim=3, chunk_size=4, d_model=32
    ).to(device)
    clone.load_state_dict(torch.load(checkpoint, map_location=device))
    torch.testing.assert_close(
        clone.sample_actions(torch.zeros(2, 6, device=device)), sample
    )
    return initial, total


def test_obstacle_policy_mse_training_and_checkpoint(tmp_path: Path) -> None:
    train_policy("obstacle", tmp_path)


def test_dagger_aggregation_preserves_episode_boundaries(tmp_path: Path) -> None:
    first_states = np.zeros((8, 3), dtype=np.float32)
    first_actions = np.ones((8, 2), dtype=np.float32)
    second_states = np.full((6, 3), 2.0, dtype=np.float32)
    second_actions = np.full((6, 2), 3.0, dtype=np.float32)
    first_ends = np.array([4, 8])
    second_ends = np.array([3, 6])
    paths = []
    for name, states, actions, ends in (
        ("teleop", first_states, first_actions, first_ends),
        ("dagger", second_states, second_actions, second_ends),
    ):
        path = tmp_path / f"{name}.zarr"
        root = zarr.open_group(str(path), mode="w")
        data = root.create_group("data")
        data.create_array("state", data=states)
        data.create_array("action", data=actions)
        meta = root.create_group("meta")
        meta.create_array("episode_ends", data=ends)
        root.attrs["state_key"] = "state"
        root.attrs["action_key"] = "action"
        paths.append(path)
    merged_states, merged_actions, merged_ends = dataset_module.load_and_merge_zarrs(
        paths
    )
    np.testing.assert_allclose(merged_states[:8], first_states)
    np.testing.assert_allclose(merged_states[8:], second_states)
    np.testing.assert_allclose(merged_actions[:8], first_actions)
    np.testing.assert_allclose(merged_actions[8:], second_actions)
    assert merged_ends.tolist() == [4, 8, 11, 14]
    dataset = dataset_module.SO100ChunkDataset(
        merged_states, merged_actions, merged_ends, chunk_size=3
    )
    assert dataset.indices.tolist() == [0, 1, 4, 5, 8, 11]
    for start in dataset.indices:
        assert not any(start < boundary <= start + 2 for boundary in merged_ends)


def test_goal_conditioned_multitask_policy(tmp_path: Path) -> None:
    initial, final = train_policy("multitask", tmp_path)
    assert final < initial


def test_goal_features_match_live_observations_and_change_with_colour(
    tmp_path: Path,
) -> None:
    from hw3.eval_utils import obs_to_state

    poses = np.arange(3 * 7, dtype=np.float32).reshape(3, 7)
    hands = np.array([[1.0, 2.0, 3.0]] * 3, dtype=np.float32)
    bins = hands + 5
    goals = np.eye(3, dtype=np.float32)
    keys = ["state_target_offset", "state_bin_offset"]
    store = tmp_path / "goals.zarr"
    root = zarr.open_group(str(store), mode="w")
    data = root.create_group("data")
    for color, pose in zip(("red", "green", "blue"), poses, strict=True):
        data.create_array(f"original_pos_cube_{color}", data=np.tile(pose, (3, 1)))
    for key, value in (
        ("state_ee_xyz", hands),
        ("goal_pos", bins),
        ("state_goal", goals),
    ):
        data.create_array(key, data=value)
    data.create_array("action", data=hands)
    root.create_group("meta").create_array("episode_ends", data=np.array([3]))
    states, _, _ = dataset_module.load_zarr(store, state_keys=keys)
    for row, goal in enumerate(goals):
        obs = {
            "ee_pos": hands[row],
            "goal_pos": bins[row],
            "cubes_xyz": poses[:, :3].ravel(),
            "goal": goal,
        }
        np.testing.assert_allclose(states[row], obs_to_state(obs, keys))
        np.testing.assert_allclose(states[row, :3], poses[row, :3] - hands[row])
        np.testing.assert_allclose(states[row, 3:], [5.0, 5.0, 5.0])
