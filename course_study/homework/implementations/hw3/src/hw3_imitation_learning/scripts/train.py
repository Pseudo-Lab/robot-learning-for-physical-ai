import argparse
from pathlib import Path
import numpy as np
import torch
import zarr
from torch.utils.data import DataLoader, Subset, random_split
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hw3.dataset import Normalizer, SO100ChunkDataset, load_and_merge_zarrs
from hw3.model import BasePolicy, build_policy


def train_one_epoch(
    model: BasePolicy,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        states, action_chunks = batch
        states = states.to(device)
        action_chunks = action_chunks.to(device)
        optimizer.zero_grad()
        loss = model.compute_loss(states, action_chunks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model: BasePolicy, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        states, action_chunks = batch
        states = states.to(device)
        action_chunks = action_chunks.to(device)
        loss = model.compute_loss(states, action_chunks)
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser(
        description="Learn action chunks from recorded demonstrations."
    )
    parser.add_argument("--zarr", required=True, type=Path)
    parser.add_argument("--extra-zarr", nargs="*", type=Path, default=[])
    parser.add_argument(
        "--policy", choices=("obstacle", "multitask"), default="obstacle"
    )
    parser.add_argument("--state-keys", nargs="+")
    parser.add_argument("--action-keys", nargs="+")
    parser.add_argument("--episode-split", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--lr", type=float, default=3e-4)
    for name, default in [
        ("seed", 42),
        ("epochs", 100),
        ("batch-size", 256),
        ("d-model", 128),
        ("depth", 2),
        ("chunk-size", 16),
    ]:
        parser.add_argument("--" + name, type=int, default=default)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    paths = [args.zarr, *args.extra_zarr]
    attrs = zarr.open_group(str(args.zarr), mode="r").attrs
    state_keys = args.state_keys or [attrs.get("state_key", "state")]
    action_keys = args.action_keys or [attrs.get("action_key", "action")]
    x, y, ends = load_and_merge_zarrs(paths, state_keys, action_keys)
    generator = torch.Generator().manual_seed(args.seed)
    validation = np.zeros(len(x), dtype=bool)
    if args.episode_split:
        lengths = np.diff(ends, prepend=0)
        usable = np.flatnonzero(lengths >= args.chunk_size)
        if len(usable) < 2:
            raise ValueError(
                "Episode validation needs at least two episodes long enough for one chunk."
            )
        held = usable[
            torch.randperm(len(usable), generator=generator).numpy()[
                : max(1, len(usable) // 10)
            ]
        ]
        validation = np.repeat(np.isin(np.arange(len(lengths)), held), lengths)
    norm = Normalizer.from_data(x[~validation], y[~validation])
    ds = SO100ChunkDataset(x, y, ends, chunk_size=args.chunk_size, normalizer=norm)
    if args.episode_split:
        mask = validation[ds.indices]
        train, valid = (
            Subset(ds, np.flatnonzero(~mask)),
            Subset(ds, np.flatnonzero(mask)),
        )
    else:
        n = max(1, len(ds) // 10)
        if n >= len(ds):
            raise ValueError("At least two samples are required.")
        train, valid = random_split(ds, [len(ds) - n, n], generator=generator)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(valid, batch_size=args.batch_size)
    config = dict(
        state_dim=x.shape[1],
        action_dim=y.shape[1],
        chunk_size=args.chunk_size,
        d_model=args.d_model,
        depth=args.depth,
    )
    model = build_policy(args.policy, **config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    task = "multi_cube" if "multi_cube" in str(args.zarr) else "single_cube"
    space = next(
        (
            k.split("[")[0].removeprefix("action_")
            for k in args.action_keys or []
            if k.split("[")[0] != "action_gripper"
        ),
        "unknown",
    )
    count = sum(
        zarr.open_group(str(p), mode="r").attrs.get("num_dagger_episodes", 0)
        for p in paths
    )
    suffix = f"_dagger{count}ep" if count else ""
    output = Path("checkpoints") / task / f"best_model_{space}_{args.policy}{suffix}.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        score = evaluate(model, val_loader, device)
        schedule.step()
        if score < best:
            best = score
            torch.save(
                dict(
                    config,
                    epoch=epoch,
                    model_state_dict=model.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    normalizer=vars(norm),
                    policy_type=args.policy,
                    state_keys=state_keys,
                    action_keys=action_keys,
                    val_loss=score,
                ),
                output,
            )
        print(f"Epoch {epoch}: train {loss:.6f}, validation {score:.6f}")
    print(output)


if __name__ == "__main__":
    main()
