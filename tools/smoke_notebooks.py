"""Esegue i notebook tecnici su copie temporanee con i flag costosi disattivati."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    "DLA_1/notebooks/01_eda_and_feature_baseline.ipynb",
    "DLA_1/notebooks/02_finetuning_pipeline.ipynb",
    "DLA_1/notebooks/02b_pipeline_consolidation.ipynb",
    "DLA_1/notebooks/03_improvements_and_retrieval.ipynb",
    "DLA_1/notebooks/03b_retrieval_training_free_classification.ipynb",
    "DLA_2/notebooks/01_sentiment_dataset_tokenizer_baseline.ipynb",
    "DLA_2/notebooks/02_distilbert_full_finetuning.ipynb",
    "DLA_2/notebooks/03_efficient_finetuning_sentiment.ipynb",
    "DLA_2/notebooks/04_clip_adapter_imagenet_sketch.ipynb",
    "DLA_3/notebooks/01_cartpole_reinforce_evaluation.ipynb",
    "DLA_3/notebooks/02_cartpole_value_baseline.ipynb",
    "DLA_3/notebooks/03_a2c_cartpole_lunarlander.ipynb",
]
IGNORED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".smoke_deps",
    "data",
    "artifacts",
    "checkpoints",
    "wandb",
    "__pycache__",
    ".venv",
    "logs",
    "runs",
}
EXPENSIVE_FLAG = re.compile(r"(?m)^(\s*(?:RUN_[A-Z0-9_]+|ENABLE_WANDB)\s*=\s*)True\b")


def public_snapshot() -> dict[str, str]:
    """Calcola hash dei file pubblici per rilevare scritture accidentali."""

    snapshot: dict[str, str] = {}
    for directory, child_dirs, names in os.walk(ROOT, topdown=True, onerror=lambda _: None):
        child_dirs[:] = [
            name
            for name in child_dirs
            if name not in IGNORED_PARTS and not name.startswith(("output", "checkpoint-"))
        ]
        base = Path(directory)
        for name in names:
            path = base / name
            if not path.is_file():
                continue
            snapshot[path.relative_to(ROOT).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return snapshot


def quick_copy(notebook: nbformat.NotebookNode) -> nbformat.NotebookNode:
    """Crea una copia senza output e forza a False i soli flag costosi."""

    result = copy.deepcopy(notebook)
    for cell in result.cells:
        if cell.cell_type != "code":
            continue
        cell.source = EXPENSIVE_FLAG.sub(r"\1False", cell.source)
        cell.outputs = []
        cell.execution_count = None
    return result


def execute_one(relative: str, output_root: Path, kernel_name: str, timeout: int) -> dict:
    """Esegue una singola copia e restituisce un record sintetico."""

    source_path = ROOT / relative
    output_path = output_root / relative
    output_path.parent.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.read(source_path, as_version=4)
    executable = quick_copy(notebook)
    started = time.perf_counter()
    record = {
        "notebook": relative,
        "stato": "OK",
        "secondi": 0.0,
        "copia": str(output_path),
        "errore": None,
    }
    try:
        client = NotebookClient(
            executable,
            timeout=timeout,
            kernel_name=kernel_name,
            allow_errors=False,
            resources={"metadata": {"path": str(source_path.parent)}},
        )
        client.execute()
    except Exception as exc:
        record["stato"] = "ERRORE"
        record["errore"] = f"{type(exc).__name__}: {exc}"
    finally:
        record["secondi"] = round(time.perf_counter() - started, 2)
        nbformat.write(executable, output_path)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Esegue copie temporanee dei notebook senza training o rete."
    )
    parser.add_argument("--kernel", default="drl", help="Nome del kernelspec da usare.")
    parser.add_argument("--timeout", type=int, default=240, help="Timeout per cella in secondi.")
    parser.add_argument(
        "--only",
        action="append",
        choices=NOTEBOOKS,
        help="Notebook tecnico da eseguire; ripetere l'opzione per più file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/dla_lab_smoke"),
        help="Directory temporanea delle copie eseguite.",
    )
    args = parser.parse_args()

    output_root = args.output.resolve()
    if output_root.exists():
        raise FileExistsError(
            f"La directory temporanea esiste già; indicarne una nuova con --output: {output_root}"
        )
    output_root.mkdir(parents=True)
    runtime_root = output_root / f".runtime_{os.getpid()}"
    runtime_root.mkdir(parents=True)
    source_roots = [str(ROOT / lab / "src") for lab in ("DLA_1", "DLA_2", "DLA_3")]
    existing_pythonpath = os.environ.get("PYTHONPATH")
    if existing_pythonpath:
        source_roots.append(existing_pythonpath)
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "WANDB_MODE": "disabled",
            "MPLBACKEND": "Agg",
            "SDL_VIDEODRIVER": "dummy",
            "PYTHONPYCACHEPREFIX": str(runtime_root / "pycache"),
            "MPLCONFIGDIR": str(runtime_root / "matplotlib"),
            "JUPYTER_RUNTIME_DIR": str(runtime_root / "jupyter_runtime"),
            "IPYTHONDIR": str(runtime_root / "ipython"),
            "PYTHONPATH": os.pathsep.join(source_roots),
        }
    )

    before = public_snapshot()
    records = []
    selected_notebooks = args.only or NOTEBOOKS
    for relative in selected_notebooks:
        print(f"[AVVIO] {relative}", flush=True)
        record = execute_one(relative, output_root, args.kernel, args.timeout)
        records.append(record)
        print(
            f"[{record['stato']}] {relative} ({record['secondi']:.2f} s)",
            flush=True,
        )
        if record["errore"]:
            print(record["errore"], flush=True)

    after = public_snapshot()
    common = set(before) & set(after)
    changed = sorted(
        set(before) ^ set(after) | {key for key in common if before[key] != after[key]}
    )
    report = {
        "kernel": args.kernel,
        "output_root": str(output_root),
        "notebook": records,
        "file_pubblici_modificati": changed,
    }
    report_path = output_root / "smoke_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Report: {report_path}")
    print(f"File pubblici modificati: {len(changed)}")
    for relative in changed:
        print(f"- {relative}")
    return 1 if changed or any(record["stato"] != "OK" for record in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
