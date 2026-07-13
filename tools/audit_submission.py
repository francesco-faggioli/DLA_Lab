"""Controlli statici riproducibili per la repository di consegna."""

from __future__ import annotations

import ast
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
IGNORED_PARTS = {".git", "data", "wandb", "artifacts", "checkpoints", "__pycache__", ".venv", "logs", "runs"}
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
        for index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            # Le magic Jupyter non appartengono alla grammatica Python standard.
            filtered = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("%", "!")))
            try:
                ast.parse(filtered)
            except SyntaxError as exc:
                errors.append(f"Sintassi cella: {path.relative_to(ROOT)}:{index}: {exc.msg} riga {exc.lineno}")
    return errors, count


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
    errors = required_checks() + notebook_errors + markdown_link_checks(files) + content_checks(files) + image_checks(files)
    large_public = sorted(
        ((path.stat().st_size, path.relative_to(ROOT)) for path in files if path.stat().st_size > 25 * 1024 * 1024),
        reverse=True,
    )
    if large_public:
        errors.extend(f"File pubblico oltre 25 MiB: {path} ({size / 1024 / 1024:.1f} MiB)" for size, path in large_public)
    print(f"File pubblici controllati: {len(files)}")
    print(f"Notebook JSON/sintassi controllati: {notebook_count}")
    print(f"Errori: {len(errors)}")
    for error in errors:
        print(f"- {error}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
