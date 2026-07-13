"""Controlli statici riproducibili per la repository di consegna."""

from __future__ import annotations

import ast
import csv
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
IGNORED_PARTS = {
    ".git",
    "data",
    "wandb",
    "artifacts",
    "checkpoints",
    "__pycache__",
    ".venv",
    "logs",
    "runs",
}
TEXT_SUFFIXES = {".md", ".py", ".yaml", ".yml", ".toml", ".txt", ".csv", ".json", ".ipynb", ".svg"}
IGNORED_SUFFIXES = {".pt", ".pth", ".ckpt", ".bin", ".safetensors", ".pyc"}
LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def public_files() -> list[Path]:
    files: list[Path] = []
    for directory, child_dirs, names in os.walk(ROOT, topdown=True, onerror=lambda _: None):
        child_dirs[:] = [
            name
            for name in child_dirs
            if name not in IGNORED_PARTS and not name.startswith(("output", "checkpoint-"))
        ]
        base = Path(directory)
        for name in names:
            path = base / name
            if path.suffix.lower() in IGNORED_SUFFIXES:
                continue
            try:
                if path.is_file():
                    files.append(path)
            except OSError:
                continue
    return files


def notebook_checks(files: list[Path]) -> tuple[list[str], int]:
    errors: list[str] = []
    count = 0
    for path in files:
        if path.suffix != ".ipynb":
            continue
        count += 1
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"JSON non valido: {path.relative_to(ROOT)}: {exc}")
            continue
        if notebook.get("nbformat") != 4 or not isinstance(notebook.get("cells"), list):
            errors.append(f"Struttura notebook non valida: {path.relative_to(ROOT)}")
        cell_ids = [cell.get("id") for cell in notebook.get("cells", [])]
        if any(not value for value in cell_ids) or len(cell_ids) != len(set(cell_ids)):
            errors.append(f"ID di cella mancanti o duplicati: {path.relative_to(ROOT)}")
        markdown = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook.get("cells", [])
            if cell.get("cell_type") == "markdown"
        )
        expected_header = (
            "| Funzione o classe | Tipo | File di definizione | Scopo | "
            "Input principali | Output principali | Sezione |"
        )
        if (
            "## Funzioni, classi e moduli locali richiamati" not in markdown
            or expected_header not in markdown
        ):
            errors.append(
                f"Tabella finale delle funzioni mancante o non standard: {path.relative_to(ROOT)}"
            )
        for index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            if "sys.path" in source:
                errors.append(f"Bootstrap sys.path non portabile: {path.relative_to(ROOT)}:{index}")
            if re.search(r"(?im)^\s*(?:%pip|!.*pip|pip\s+install)", source):
                errors.append(
                    f"Installazione pip dentro il notebook: {path.relative_to(ROOT)}:{index}"
                )
            for output in cell.get("outputs", []):
                if output.get("output_type") == "error":
                    errors.append(
                        f"Output di errore salvato: {path.relative_to(ROOT)}:{index}: "
                        f"{output.get('ename', 'errore sconosciuto')}"
                    )
                html_value = output.get("data", {}).get("text/html", "")
                html_text = "".join(html_value) if isinstance(html_value, list) else str(html_value)
                if (
                    len(html_text) > 100_000
                    or "animation.FuncAnimation" in html_text
                    or "<canvas" in html_text
                ):
                    errors.append(
                        f"Output HTML pesante incorporato: {path.relative_to(ROOT)}:{index}"
                    )
            # Le magic Jupyter non appartengono alla grammatica Python standard.
            filtered = "\n".join(
                line for line in source.splitlines() if not line.lstrip().startswith(("%", "!"))
            )
            try:
                tree = ast.parse(filtered)
            except SyntaxError as exc:
                errors.append(
                    f"Sintassi cella: {path.relative_to(ROOT)}:{index}: {exc.msg} riga {exc.lineno}"
                )
                continue
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if not any(
                    isinstance(target, ast.Name)
                    and (target.id.startswith("RUN_") or target.id == "ENABLE_WANDB")
                    for target in targets
                ):
                    continue
                try:
                    value = ast.literal_eval(node.value)
                except Exception:
                    continue
                if value is True:
                    names = [target.id for target in targets if isinstance(target, ast.Name)]
                    errors.append(
                        f"Flag costoso attivo {', '.join(names)}: {path.relative_to(ROOT)}:{index}"
                    )
    return errors, count


def configuration_checks() -> list[str]:
    errors: list[str] = []
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    for token in [
        "jupyterlab",
        "ipykernel",
        "numpy",
        "pandas",
        "matplotlib",
        "torch",
        "gymnasium[box2d]",
        "pygame",
        "-e .",
    ]:
        if token not in requirements:
            errors.append(f"Dipendenza canonica mancante in requirements.txt: {token}")
    environment = (ROOT / "environment.yml").read_text(encoding="utf-8").lower()
    if "python=3.12" not in environment or "-r requirements.txt" not in environment:
        errors.append("environment.yml non delega a requirements.txt con Python 3.12")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for token in ["DLA_1/src", "DLA_2/src", "DLA_3/src", 'requires-python = ">=3.11,<3.14"']:
        if token not in pyproject:
            errors.append(f"Configurazione di packaging mancante in pyproject.toml: {token}")
    return errors


def _csv_rows(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _expected_csv_value(
    errors: list[str],
    relative: str,
    filters: dict[str, str],
    column: str,
    expected: float,
) -> None:
    rows = [
        row
        for row in _csv_rows(relative)
        if all(row.get(key) == value for key, value in filters.items())
    ]
    label = ", ".join(f"{key}={value}" for key, value in filters.items())
    if len(rows) != 1:
        errors.append(f"Riga numerica non univoca in {relative}: {label}")
        return
    try:
        actual = float(rows[0][column])
    except (KeyError, TypeError, ValueError):
        errors.append(f"Valore numerico non valido in {relative}: {label}, colonna {column}")
        return
    if abs(actual - expected) > 1e-9 * max(1.0, abs(expected)):
        errors.append(f"Valore incoerente in {relative}: {label}, {actual} != {expected}")


def numeric_checks() -> list[str]:
    errors: list[str] = []
    _expected_csv_value(
        errors,
        "DLA_1/results/test_metrics.csv",
        {"experiment": "ResNet-18 features + SVM", "metric": "test_accuracy"},
        "value",
        0.6412,
    )
    _expected_csv_value(
        errors,
        "DLA_1/results/test_metrics.csv",
        {"experiment": "Head-only ResNet-18 fine-tuning", "metric": "test_accuracy"},
        "value",
        0.5038,
    )
    _expected_csv_value(
        errors,
        "DLA_1/results/test_metrics.csv",
        {"experiment": "Improved ResNet-18 fine-tuning", "metric": "test_accuracy"},
        "value",
        0.8025,
    )
    _expected_csv_value(
        errors,
        "DLA_1/results/test_metrics.csv",
        {"experiment": "Cosine retrieval", "metric": "precision_at_1"},
        "value",
        0.481235,
    )
    _expected_csv_value(
        errors,
        "DLA_1/results/test_metrics.csv",
        {"experiment": "Nearest-Mean Classifier", "metric": "test_accuracy"},
        "value",
        0.418527,
    )
    _expected_csv_value(
        errors,
        "DLA_2/results/sentiment_results.csv",
        {"method": "Full DistilBERT fine-tuning", "split": "test"},
        "accuracy",
        0.844278,
    )
    _expected_csv_value(
        errors,
        "DLA_2/results/sentiment_results.csv",
        {"method": "LoRA", "split": "test"},
        "accuracy",
        0.8386491557,
    )
    _expected_csv_value(
        errors,
        "DLA_2/results/sentiment_results.csv",
        {"method": "Partial freezing", "split": "test"},
        "accuracy",
        0.8377110694,
    )
    _expected_csv_value(
        errors,
        "DLA_2/results/clip_results.csv",
        {"method": "CLIP-Adapter bottleneck 128"},
        "accuracy",
        0.524072,
    )
    _expected_csv_value(
        errors,
        "DLA_3/results/method_summary.csv",
        {
            "environment": "CartPole-v1",
            "method": "REINFORCE standardized returns",
            "metric": "best_evaluation_return",
        },
        "value",
        489.35,
    )
    _expected_csv_value(
        errors,
        "DLA_3/results/method_summary.csv",
        {
            "environment": "CartPole-v1",
            "method": "REINFORCE with value baseline",
            "metric": "best_evaluation_return",
        },
        "value",
        500.0,
    )
    _expected_csv_value(
        errors,
        "DLA_3/results/method_summary.csv",
        {"environment": "CartPole-v1", "method": "A2C", "metric": "greedy_average_return"},
        "value",
        494.51,
    )
    final = json.loads(
        (ROOT / "DLA_3/results/lunarlander_final_evaluation.json").read_text(encoding="utf-8")
    )
    for key, expected in {
        "evaluation_episodes": 200,
        "average_return": 165.76132762441972,
        "standard_deviation_return": 100.56278526980033,
        "success_rate_percent": 56.0,
    }.items():
        if abs(float(final.get(key, float("nan"))) - expected) > 1e-9 * max(1.0, abs(expected)):
            errors.append(f"Valore incoerente in lunarlander_final_evaluation.json: {key}")
    summary = _csv_rows("DLA_3/results/method_summary.csv")
    lunar = [row for row in summary if row["environment"] == "LunarLander-v3"]
    summary_values = {row["metric"]: float(row["value"]) for row in lunar}
    if abs(summary_values.get("average_return", float("nan")) - final["average_return"]) > 1e-9:
        errors.append("Return LunarLander incoerente tra method_summary.csv e JSON finale")
    if (
        abs(
            summary_values.get("success_rate_percent", float("nan")) - final["success_rate_percent"]
        )
        > 1e-9
    ):
        errors.append("Success rate LunarLander incoerente tra method_summary.csv e JSON finale")
    expected_docs = {
        "DLA_1/README.md": ["0.6412", "0.5038", "0.8025", "0.4812", "0.4185"],
        "DLA_2/README.md": ["0.8443", "0.8386", "0.8377", "0.5241"],
        "DLA_3/README.md": ["489.35", "500.00", "494.51", "165.76", "100.56", "56.0%"],
    }
    for relative, tokens in expected_docs.items():
        text = (ROOT / relative).read_text(encoding="utf-8")
        for token in tokens:
            if token not in text:
                errors.append(f"Valore documentale mancante in {relative}: {token}")
    expected_notebooks = {
        "DLA_1/notebooks/01_eda_and_feature_baseline.ipynb": ["0.6412"],
        "DLA_1/notebooks/02_finetuning_pipeline.ipynb": ["0.5038"],
        "DLA_1/notebooks/03_improvements_and_retrieval.ipynb": ["0.8025"],
        "DLA_1/notebooks/03b_retrieval_training_free_classification.ipynb": [
            "0.481235",
            "0.418527",
        ],
        "DLA_2/notebooks/02_distilbert_full_finetuning.ipynb": ["0.844278"],
        "DLA_2/notebooks/03_efficient_finetuning_sentiment.ipynb": [
            "0.838649",
            "1.092533",
            "0.837711",
        ],
        "DLA_2/notebooks/04_clip_adapter_imagenet_sketch.ipynb": ["0.524072"],
        "DLA_3/notebooks/01_cartpole_reinforce_evaluation.ipynb": ["489.35"],
        "DLA_3/notebooks/02_cartpole_value_baseline.ipynb": ["500.0"],
        "DLA_3/notebooks/03_a2c_cartpole_lunarlander.ipynb": [
            "494.51",
            "165.76",
            "100.56",
            "56.0%",
        ],
    }
    for relative, tokens in expected_notebooks.items():
        text = (ROOT / relative).read_text(encoding="utf-8")
        for token in tokens:
            if token not in text:
                errors.append(f"Evidenza numerica mancante nel notebook {relative}: {token}")
    expected_figures = {
        "DLA_1/figures/gtsrb_test_accuracy_comparison.svg": [
            "0.6412",
            "0.5038",
            "0.8025",
            "0.4185",
        ],
        "DLA_1/figures/gtsrb_retrieval_metrics.svg": ["0.4812"],
        "DLA_2/figures/sentiment_test_accuracy.svg": ["0.8443", "0.8386", "0.8377"],
        "DLA_2/figures/sentiment_trainable_parameters.svg": ["1.0925"],
        "DLA_2/figures/clip_accuracy_comparison.svg": ["0.5241"],
        "DLA_3/figures/cartpole_primary_returns.svg": ["489.3500", "500.0000", "494.5100"],
        "DLA_3/figures/lunarlander_final_evaluation.svg": ["165.76", "100.56", "56.0%"],
    }
    for relative, tokens in expected_figures.items():
        text = (ROOT / relative).read_text(encoding="utf-8")
        for token in tokens:
            if token not in text:
                errors.append(f"Evidenza numerica mancante nella figura {relative}: {token}")
    return errors


def markdown_link_checks(files: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in files:
        if path.suffix == ".md":
            text = path.read_text(encoding="utf-8")
        elif path.suffix == ".ipynb":
            notebook = json.loads(path.read_text(encoding="utf-8"))
            text = "\n".join(
                "".join(cell.get("source", []))
                for cell in notebook.get("cells", [])
                if cell.get("cell_type") == "markdown"
            )
        else:
            continue
        for raw_target in LINK_PATTERN.findall(text):
            target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
            if not target or target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            relative = unquote(target.split("#", 1)[0])
            resolved = (path.parent / relative).resolve()
            try:
                resolved.relative_to(ROOT)
            except ValueError:
                errors.append(f"Link fuori repository: {path.relative_to(ROOT)} -> {target}")
                continue
            if not resolved.exists():
                errors.append(f"Link mancante: {path.relative_to(ROOT)} -> {target}")
    return errors


def content_checks(files: list[Path]) -> list[str]:
    errors: list[str] = []
    forbidden = {
        "Lab_" + "3.ipynb": "vecchio nome notebook",
        "ASSIGNMENT.ipynb": "duplicato traccia eliminato",
        "CODE_OF_CONDUCT.ipynb": "duplicato Code of Conduct eliminato",
        "Reinforcment" + " LLM": "vecchio nome cartella",
        "DLA_" + "Lab2": "vecchio nome cartella",
        "C:\\Users\\" + "checc": "percorso Windows locale",
        "/mnt/c/Users/" + "checc": "percorso WSL locale",
        "/home/" + "francescofaggioli": "percorso Linux locale",
        "_archive" + "/": "archivio pubblico obsoleto",
        "Gianni" + "Moretti": "riferimento esterno vietato",
        "DeepLearning" + "ApplicationLAB": "riferimento esterno vietato",
    }
    secret_patterns = {
        "OpenAI key": re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
        "W&B key": re.compile(r"(?i)WANDB_API_KEY\s*[=:]\s*[A-Za-z0-9]{20,}"),
        "generic bearer": re.compile(r"(?i)Authorization\s*:\s*Bearer\s+[A-Za-z0-9._-]{20,}"),
    }
    for path in files:
        if path.name == "audit_submission.py":
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for token, reason in forbidden.items():
            if token in text:
                errors.append(f"{reason}: {path.relative_to(ROOT)} contiene {token!r}")
        for reason, pattern in secret_patterns.items():
            if pattern.search(text):
                errors.append(f"Possibile credenziale {reason}: {path.relative_to(ROOT)}")
    return errors


def required_checks() -> list[str]:
    errors: list[str] = []
    required = [
        "README.md",
        "AI_USAGE.md",
        "CODE_OF_CONDUCT.md",
        "DLA_1/DLA_1.ipynb",
        "DLA_2/DLA_2.ipynb",
        "DLA_3/DLA_3.ipynb",
        "DLA_1/ASSIGNMENT.md",
        "DLA_2/ASSIGNMENT.md",
        "DLA_3/ASSIGNMENT.md",
        "DLA_1/results/test_metrics.csv",
        "DLA_2/results/sentiment_results.csv",
        "DLA_3/results/lunarlander_final_evaluation.json",
    ]
    for relative in required:
        if not (ROOT / relative).exists():
            errors.append(f"File richiesto mancante: {relative}")
    for forbidden in [
        "CODE_OF_CONDUCT.ipynb",
        "DLA_1/ASSIGNMENT.ipynb",
        "DLA_2/ASSIGNMENT.ipynb",
        "DLA_3/ASSIGNMENT.ipynb",
        "DLA_3/Lab_" + "3.ipynb",
        "_archive",
    ]:
        if (ROOT / forbidden).exists():
            errors.append(f"Duplicato/obsoleto ancora presente: {forbidden}")
    return errors


def image_checks(files: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in files:
        try:
            if path.suffix.lower() == ".svg":
                root = ET.fromstring(path.read_text(encoding="utf-8"))
                if not root.tag.endswith("svg"):
                    errors.append(f"Radice SVG non valida: {path.relative_to(ROOT)}")
            elif path.suffix.lower() == ".png":
                if path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
                    errors.append(f"Firma PNG non valida: {path.relative_to(ROOT)}")
        except Exception as exc:
            errors.append(f"Immagine non valida: {path.relative_to(ROOT)}: {exc}")
    return errors


def main() -> int:
    files = public_files()
    notebook_errors, notebook_count = notebook_checks(files)
    errors = (
        required_checks()
        + configuration_checks()
        + numeric_checks()
        + notebook_errors
        + markdown_link_checks(files)
        + content_checks(files)
        + image_checks(files)
    )
    large_public = sorted(
        (
            (path.stat().st_size, path.relative_to(ROOT))
            for path in files
            if path.stat().st_size > 25 * 1024 * 1024
        ),
        reverse=True,
    )
    if large_public:
        errors.extend(
            f"File pubblico oltre 25 MiB: {path} ({size / 1024 / 1024:.1f} MiB)"
            for size, path in large_public
        )
    print(f"File pubblici controllati: {len(files)}")
    print(f"Notebook JSON/sintassi controllati: {notebook_count}")
    print(f"Errori: {len(errors)}")
    for error in errors:
        print(f"- {error}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
