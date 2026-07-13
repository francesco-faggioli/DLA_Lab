from __future__ import annotations

import argparse
from pathlib import Path

import torch


def summarize_checkpoint(path: Path) -> dict:
    """Carica un checkpoint e ne restituisce un riepilogo leggero.

    Argomenti:
        path: Percorso di un file `.pt` o `.pth`.

    Operazione:
        Carica il checkpoint su CPU e distingue uno state dict grezzo da un
        dizionario con metadati. Il file viene soltanto letto.

    Output:
        Dizionario con percorso, chiavi di primo livello e numero di tensori.
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
    parser = argparse.ArgumentParser(description="Ispeziona i vecchi checkpoint di DLA Lab 3.")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "checkpoints",
        help="Directory contenente i checkpoint da ispezionare.",
    )
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir.expanduser()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Directory dei checkpoint non trovata: {checkpoint_dir}")

    checkpoint_paths = sorted(
        list(checkpoint_dir.rglob("*.pt")) + list(checkpoint_dir.rglob("*.pth"))
    )
    if not checkpoint_paths:
        print(f"Nessun file .pt o .pth trovato in {checkpoint_dir}")
        return

    print(f"Found {len(checkpoint_paths)} checkpoint files under {checkpoint_dir}")
    for path in checkpoint_paths:
        try:
            summary = summarize_checkpoint(path)
            print()
            print(summary["path"])
            print(f"  tipo: {summary['type']}")
            print(f"  chiavi: {summary['top_level_keys']}")
            print(f"  numero di tensori: {summary['tensor_count']}")
        except Exception as exc:
            print()
            print(path)
            print(f"  ERROR: {exc}")


if __name__ == "__main__":
    main()
