from __future__ import annotations

from pathlib import Path
from typing import Any


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return None
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if inner == "":
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    try:
        if any(marker in value.lower() for marker in [".", "e"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _next_container(lines: list[str], start_index: int, current_indent: int) -> Any:
    for line in lines[start_index + 1 :]:
        stripped = line.strip()
        if stripped == "" or stripped.startswith("#"):
            continue
        next_indent = len(line) - len(line.lstrip(" "))
        if next_indent <= current_indent:
            return None
        return [] if stripped.startswith("- ") else {}
    return None


def load_simple_yaml(path: str | Path) -> dict[str, Any]:
    """Carica il sottoinsieme YAML usato dalla configurazione del laboratorio.

    Argomenti:
        path: Percorso del file YAML da leggere.

    Operazione:
        Analizza dizionari annidati, liste semplici o inline e valori scalari
        usando soltanto la libreria standard. Il file viene letto senza scritture.

    Output:
        Dizionario contenente la configurazione analizzata.
    """

    lines = Path(path).read_text(encoding="utf-8").splitlines()
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]

    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "" or stripped.startswith("#"):
            continue

        indent = len(line) - len(line.lstrip(" "))
        while stack and indent <= stack[-1][0]:
            stack.pop()
        container = stack[-1][1]

        if stripped.startswith("- "):
            if not isinstance(container, list):
                raise ValueError(f"List item found outside a list at line {index + 1}: {line}")
            container.append(_parse_scalar(stripped[2:]))
            continue

        if ":" not in stripped:
            raise ValueError(f"Unsupported YAML line {index + 1}: {line}")

        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()

        if raw_value == "":
            value = _next_container(lines, index, indent)
            container[key] = value
            if isinstance(value, (dict, list)):
                stack.append((indent, value))
        else:
            container[key] = _parse_scalar(raw_value)

    return root


def load_lab_config(path: str | Path) -> dict[str, Any]:
    """Carica il file di configurazione del laboratorio.

    Argomenti:
        path: Percorso di `lab3_defaults.yaml`.

    Operazione:
        Usa PyYAML quando disponibile; in caso contrario usa il parser minimale
        locale. Propaga errori di lettura o sintassi non supportata.

    Output:
        Dizionario con la configurazione del laboratorio.
    """

    try:
        import yaml  # type: ignore

        with Path(path).open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ModuleNotFoundError:
        return load_simple_yaml(path)
