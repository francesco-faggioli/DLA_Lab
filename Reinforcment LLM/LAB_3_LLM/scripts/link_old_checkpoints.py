from __future__ import annotations

import argparse
from pathlib import Path


def project_root() -> Path:
    """Return the LAB_3_LLM project root.

    Args:
        None.

    What it does:
        Resolves the parent directory of `scripts`.

    Outputs:
        Path pointing to the working lab folder.
    """

    return Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Link old checkpoints into this lab folder.")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/home/francescofaggioli/DRL_lab"),
        help="Directory containing old checkpoints.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="old",
        help="Symlink name created under the local checkpoints folder.",
    )
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Source checkpoint directory not found: {source}")

    checkpoints = project_root() / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)

    target = checkpoints / args.name
    if target.exists() or target.is_symlink():
        print(f"Checkpoint link already exists: {target}")
        print(f"Points to: {target.resolve()}")
        return

    target.symlink_to(source, target_is_directory=True)
    print(f"Created checkpoint link: {target}")
    print(f"Points to: {source}")


if __name__ == "__main__":
    main()
