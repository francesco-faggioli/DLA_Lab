from __future__ import annotations

import argparse
from pathlib import Path

import torch


def summarize_checkpoint(path: Path) -> dict:
    """Load a checkpoint and return a lightweight summary.

    Args:
        path: Path to a `.pt` or `.pth` file.

    What it does:
        Loads the checkpoint on CPU and reports whether it looks like a raw
        state dict or a dictionary with metadata.

    Outputs:
        Dictionary with checkpoint path, top-level keys and tensor count.
    """

    checkpoint = torch.load(path, map_location="cpu")
    summary = {
        "path": str(path),
        "type": type(checkpoint).__name__,
        "top_level_keys": [],
        "tensor_count": 0,
    }

    if isinstance(checkpoint, dict):
        summary["top_level_keys"] = list(checkpoint.keys())[:20]
        state = checkpoint.get("model_state_dict", checkpoint)
        if isinstance(state, dict):
            summary["tensor_count"] = sum(torch.is_tensor(value) for value in state.values())
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect old DLA Lab 3 checkpoints.")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("/home/francescofaggioli/DRL_lab"),
        help="Directory containing old checkpoints.",
    )
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir.expanduser()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    checkpoint_paths = sorted(
        list(checkpoint_dir.rglob("*.pt")) + list(checkpoint_dir.rglob("*.pth"))
    )
    if not checkpoint_paths:
        print(f"No .pt or .pth files found under {checkpoint_dir}")
        return

    print(f"Found {len(checkpoint_paths)} checkpoint files under {checkpoint_dir}")
    for path in checkpoint_paths:
        try:
            summary = summarize_checkpoint(path)
            print()
            print(summary["path"])
            print(f"  type: {summary['type']}")
            print(f"  keys: {summary['top_level_keys']}")
            print(f"  tensor_count: {summary['tensor_count']}")
        except Exception as exc:
            print()
            print(path)
            print(f"  ERROR: {exc}")


if __name__ == "__main__":
    main()
