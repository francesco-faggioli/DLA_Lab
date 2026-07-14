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
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
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


def _markdown_text(path: Path) -> str:
    if path.suffix == ".md":
        return path.read_text(encoding="utf-8")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "markdown"
    )


def markdown_latex_checks(files: list[Path]) -> list[str]:
    """Controlla i blocchi math GitHub e i delimitatori Jupyter."""

    errors: list[str] = []
    markdown_files = [path for path in files if path.suffix in {".md", ".ipynb"}]
    expected_readme_blocks = {
        "DLA_1/README.md": 4,
        "DLA_2/README.md": 4,
        "DLA_3/README.md": 5,
    }

    def check_formula(formula: str, relative: Path) -> None:
        if not formula.strip():
            errors.append(f"Formula display vuota in {relative}")
        if formula.count("{") != formula.count("}"):
            errors.append(f"Parentesi graffe LaTeX sbilanciate in {relative}")
        if formula.count("\\left") != formula.count("\\right"):
            errors.append(f"Coppie \\left/\\right sbilanciate in {relative}")
        if re.search(r"(?m)^\s*={2,}\s*$", formula):
            errors.append(f"Separatore Markdown incorporato in una formula: {relative}")

    for path in markdown_files:
        text = _markdown_text(path)
        relative = path.relative_to(ROOT)
        relative_posix = relative.as_posix()
        if "\\[" in text or "\\]" in text:
            errors.append(f"Delimitatore LaTeX non supportato in {relative}")

        if relative_posix in expected_readme_blocks:
            if "$$" in text:
                errors.append(f"Delimitatore $$ residuo nel README GitHub: {relative}")
            openings = len(re.findall(r"(?m)^```math[ \t]*$", text))
            matches = list(
                re.finditer(
                    r"(?ms)^```math[ \t]*\n(.*?)^```[ \t]*$",
                    text,
                )
            )
            if openings != len(matches):
                errors.append(f"Blocco math non chiuso correttamente in {relative}")
            expected = expected_readme_blocks[relative_posix]
            if len(matches) != expected:
                errors.append(
                    f"Numero inatteso di blocchi math in {relative}: "
                    f"attesi {expected}, trovati {len(matches)}"
                )
            for match in matches:
                if match.start() and not text[: match.start()].endswith("\n\n"):
                    errors.append(f"Riga vuota mancante prima di un blocco math: {relative}")
                if match.end() < len(text) and not text[match.end() :].startswith("\n\n"):
                    errors.append(f"Riga vuota mancante dopo un blocco math: {relative}")
                check_formula(match.group(1), relative)
        else:
            if path.suffix == ".ipynb":
                delimiters = len(re.findall(r"\$\$", text))
                standalone = len(re.findall(r"(?m)^[ \t]*\$\$[ \t]*$", text))
                if delimiters != standalone:
                    errors.append(f"Delimitatore $$ non isolato su una riga in {relative}")
            if text.count("$$") % 2:
                errors.append(f"Numero dispari di delimitatori $$ in {relative}")
                continue
            for formula in re.findall(r"\$\$(.*?)\$\$", text, flags=re.DOTALL):
                check_formula(formula, relative)

        prose = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
        prose = re.sub(r"`[^`\n]*`", "", prose)
        prose = re.sub(r"\$\$.*?\$\$", "", prose, flags=re.DOTALL)
        inline_dollars = len(re.findall(r"(?<!\\)\$", prose))
        if inline_dollars % 2:
            errors.append(f"Delimitatori LaTeX inline sbilanciati in {relative}")
    return errors


def language_and_formula_checks() -> list[str]:
    errors: list[str] = []
    root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    ai_usage = (ROOT / "AI_USAGE.md").read_text(encoding="utf-8")
    conduct = (ROOT / "CODE_OF_CONDUCT.md").read_text(encoding="utf-8")

    for token in [
        "## Panoramica del portfolio",
        "## Risultati principali",
        "## Struttura della repository",
        "## Ordine di esecuzione consigliato",
        "## Modalità di consultazione rapida",
        "## Modalità di esperimento completo",
        "## Politica di conservazione degli output",
        "## Politica per dataset e checkpoint",
        "## Riproducibilità",
        "## Uso dell'IA e integrità accademica",
        "## Codice di condotta",
    ]:
        if token not in root_readme:
            errors.append(f"Sezione italiana mancante in README.md: {token}")
    if "## Portfolio overview" in root_readme or "## Main results" in root_readme:
        errors.append("README.md contiene ancora intestazioni inglesi")

    for token in [
        "# Uso di strumenti assistiti dall'IA",
        "chiarimento concettuale",
        "tracce ufficiali",
        "debug",
        "commenti e delle docstring",
        "struttura della repository",
        "presentazione matematica",
        "ogni formula è stata verificata",
        "Verifica umana e responsabilità",
        "Nessuna metrica, figura, formula, iperparametro, fonte o esercizio completato è stato inventato",
    ]:
        if token.lower() not in ai_usage.lower():
            errors.append(f"Dichiarazione italiana mancante in AI_USAGE.md: {token}")
    if "Code of Conduct" not in conduct.splitlines()[0]:
        errors.append("CODE_OF_CONDUCT.md non risulta in inglese o ha titolo inatteso")

    for relative in ["DLA_1/README.md", "DLA_2/README.md", "DLA_3/README.md"]:
        text = (ROOT / relative).read_text(encoding="utf-8")
        for token in ["## Panoramica", "## Obiettivi", "## Conclusione"]:
            if token not in text:
                errors.append(f"Sezione italiana mancante in {relative}: {token}")

    required_formula_tokens = {
        "DLA_1/README.md": [
            r"\mathcal{L}_{\mathrm{CE}}",
            r"\mathrm{cos}(q,x)",
            "P@K",
            r"\mu_c",
            "class_weights_from_labels",
            "FocalLoss",
            "nearest_mean_classifier",
        ],
        "DLA_2/README.md": [
            r"\frac{\alpha}{r}BA",
            "Parametri addestrabili",
            r"\mathrm{cos}",
            r"\sigma(\alpha)",
            "CLIPAdapter.forward",
        ],
        "DLA_3/README.md": [
            r"G_t=\sum",
            r"\mathcal{L}_{\mathrm{REINFORCE}}",
            r"A_t=G_t-V_\phi",
            r"\mathcal{L}_{\mathrm{A2C}}",
            "SmoothL1",
            r"\pi_T(a\mid s)",
            "numpy.std",
        ],
        "DLA_1/notebooks/03b_retrieval_training_free_classification.ipynb": [
            "math-dla1-retrieval",
            "math-dla1-nmc",
        ],
        "DLA_2/notebooks/03_efficient_finetuning_sentiment.ipynb": [
            "math-dla2-lora",
            "math-dla2-partial-freezing",
        ],
        "DLA_2/notebooks/04_clip_adapter_imagenet_sketch.ipynb": [
            "math-dla2-clip-similarity",
            "math-dla2-adapter",
        ],
        "DLA_3/notebooks/01_cartpole_reinforce_evaluation.ipynb": ["math-dla3-reinforce"],
        "DLA_3/notebooks/02_cartpole_value_baseline.ipynb": ["math-dla3-value-baseline"],
        "DLA_3/notebooks/03_a2c_cartpole_lunarlander.ipynb": [
            "math-dla3-a2c-gae",
            "math-dla3-temperature",
            "math-dla3-evaluation",
        ],
    }
    for relative, tokens in required_formula_tokens.items():
        text = (ROOT / relative).read_text(encoding="utf-8")
        for token in tokens:
            if token not in text:
                errors.append(
                    f"Formula o collegamento formula-codice mancante in {relative}: {token}"
                )
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
    local_user = "che" + "cc"
    linux_user = "francesco" + "faggioli"
    exploratory_name = "explor" + "atory"
    forbidden_patterns = {
        "percorso personale Windows": re.compile(
            r"(?i)c:[\\/]+users[\\/]+" + re.escape(local_user)
        ),
        "percorso personale WSL": re.compile(r"(?i)/mnt/c/users/" + re.escape(local_user)),
        "riferimento cloud personale": re.compile("(?i)one" + "drive"),
        "account Linux personale": re.compile("(?i)" + linux_user + r"@|/home/" + linux_user),
        "notebook exploratory rimosso": re.compile(
            "(?i)Esperimenti_"
            + "di_prova|00_esperimenti_"
            + "di_prova_a2c|"
            + exploratory_name
            + "/"
        ),
        "riferimento vietato": re.compile(
            "(?i)Gianni" + "Moretti|DeepLearning" + r"ApplicationLAB|github\.com/Gianni" + "Moretti"
        ),
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
        for reason, pattern in forbidden_patterns.items():
            if pattern.search(text):
                errors.append(f"{reason}: {path.relative_to(ROOT)}")
        for reason, pattern in secret_patterns.items():
            if pattern.search(text):
                errors.append(f"Possibile credenziale {reason}: {path.relative_to(ROOT)}")
        task_marker = r"(?i)\b(?:" + "TO" + "DO|FIX" + "ME)\b"
        if path.name not in {"ASSIGNMENT.md", "CODE_OF_CONDUCT.md"} and re.search(
            task_marker, text
        ):
            errors.append(f"Segnaposto di lavoro residuo: {path.relative_to(ROOT)}")
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
        "DLA_1/" + "explor" + "atory",
        "DLA_3/" + "explor" + "atory",
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
        + markdown_latex_checks(files)
        + language_and_formula_checks()
        + content_checks(files)
        + image_checks(files)
    )
    if notebook_count != 15:
        errors.append(f"Numero di notebook pubblici inatteso: {notebook_count} != 15")
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
