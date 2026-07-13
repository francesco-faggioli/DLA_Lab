"""Rimuove soltanto le animazioni HTML pesanti, mantenendo gli output scientifici."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "DLA_3" / "notebooks" / "03_a2c_cartpole_lunarlander.ipynb"


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    removed = 0
    for cell in notebook["cells"]:
        filtered_outputs = []
        for output in cell.get("outputs", []):
            html_payload = output.get("data", {}).get("text/html")
            if html_payload is None:
                filtered_outputs.append(output)
                continue
            payload_size = len("".join(html_payload) if isinstance(html_payload, list) else html_payload)
            if payload_size < 1_000_000:
                filtered_outputs.append(output)
                continue
            removed += 1
            filtered_outputs.append(
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        "Animazione HTML rimossa dalla versione GitHub per contenere la dimensione del notebook.\n",
                        "Il return, la lunghezza e il successo dell'episodio restano nell'output precedente.\n",
                    ],
                }
            )
        cell["outputs"] = filtered_outputs
    NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"Animazioni HTML rimosse: {removed}")


if __name__ == "__main__":
    main()
