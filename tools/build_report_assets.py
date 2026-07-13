"""Rigenera le figure dei README da evidenze versionate e output dei notebook."""

from __future__ import annotations

import base64
import csv
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLORS = ["#176B87", "#D1495B", "#2E7D32", "#7A5195", "#E07A1F", "#536878"]


ITALIAN_LABELS = {
    "ResNet-18 features + SVM": "Feature ResNet-18 + SVM",
    "Head-only ResNet-18 fine-tuning": "Fine-tuning ResNet-18 della sola testa",
    "Improved ResNet-18 fine-tuning": "Fine-tuning ResNet-18 migliorato",
    "Nearest-Mean Classifier": "Classificatore del centroide più vicino",
    "Full DistilBERT fine-tuning": "Fine-tuning completo DistilBERT",
    "Partial freezing": "Congelamento parziale",
    "Zero-shot CLIP": "CLIP zero-shot",
    "Zero-shot CLIP + prompt ensemble": "CLIP zero-shot + insieme di prompt",
}


def italian_label(value: str) -> str:
    """Traduce le etichette descrittive, preservando gli identificatori tecnici."""

    return ITALIAN_LABELS.get(value, value)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_svg(path: Path, title: str, body: str, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
<title id="title">{html.escape(title)}</title>
<desc id="desc">Figura generata da dati sperimentali versionati nella cartella results.</desc>
<rect width="100%" height="100%" fill="#ffffff"/>
<style>text{{font-family:Arial,sans-serif;fill:#182026;letter-spacing:0}} .grid{{stroke:#d8dee4;stroke-width:1}} .axis{{stroke:#59636e;stroke-width:1.2}}</style>
<text x="{width / 2}" y="32" text-anchor="middle" font-size="20" font-weight="700">{html.escape(title)}</text>
{body}
</svg>
"""
    path.write_text(document, encoding="utf-8", newline="\n")


def horizontal_bars(
    path: Path,
    title: str,
    rows: list[tuple[str, float]],
    x_label: str,
    maximum: float | None = None,
) -> None:
    width = 980
    left, right, top, bottom = 330, 55, 65, 68
    row_height = 44
    height = top + bottom + row_height * len(rows)
    plot_width = width - left - right
    limit = maximum or max(value for _, value in rows) * 1.08
    body: list[str] = []
    for tick in range(6):
        value = limit * tick / 5
        x = left + plot_width * tick / 5
        body.append(
            f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}"/>'
        )
        body.append(
            f'<text x="{x:.1f}" y="{height-bottom+22}" text-anchor="middle" font-size="12">{value:.2f}</text>'
        )
    for index, (label, value) in enumerate(rows):
        y = top + index * row_height + 8
        bar_width = max(1.0, plot_width * value / limit)
        body.append(
            f'<text x="{left-12}" y="{y+18}" text-anchor="end" font-size="13">{html.escape(label)}</text>'
        )
        body.append(
            f'<rect x="{left}" y="{y}" width="{bar_width:.1f}" height="25" rx="3" fill="{COLORS[index % len(COLORS)]}"/>'
        )
        body.append(
            f'<text x="{left+bar_width+8:.1f}" y="{y+18}" font-size="13" font-weight="700">{value:.4f}</text>'
        )
    body.append(
        f'<line class="axis" x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}"/>'
    )
    body.append(
        f'<text x="{left+plot_width/2}" y="{height-15}" text-anchor="middle" font-size="14">{html.escape(x_label)}</text>'
    )
    write_svg(path, title, "\n".join(body), width, height)


def line_chart(
    path: Path,
    title: str,
    x_values: list[float],
    series: list[tuple],
    x_label: str,
    y_label: str,
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    width, height = 980, 520
    left, right, top, bottom = 82, 40, 65, 72
    plot_width, plot_height = width - left - right, height - top - bottom
    normalized_series = []
    for item in series:
        if len(item) == 3:
            label, values, color = item
            normalized_series.append((label, x_values, values, color))
        else:
            label, own_x_values, values, color = item
            normalized_series.append((label, own_x_values, values, color))
    all_values = [value for _, _, values, _ in normalized_series for value in values]
    lower = min(all_values) if y_min is None else y_min
    upper = max(all_values) if y_max is None else y_max
    if upper == lower:
        upper = lower + 1
    x_low, x_high = min(x_values), max(x_values)

    def sx(value: float) -> float:
        return left + (value - x_low) * plot_width / (x_high - x_low or 1)

    def sy(value: float) -> float:
        return top + (upper - value) * plot_height / (upper - lower)

    body: list[str] = []
    for tick in range(6):
        value = lower + (upper - lower) * tick / 5
        y = sy(value)
        body.append(
            f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}"/>'
        )
        body.append(
            f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-size="12">{value:.2f}</text>'
        )
    for tick in range(6):
        value = x_low + (x_high - x_low) * tick / 5
        x = sx(value)
        body.append(
            f'<text x="{x:.1f}" y="{height-bottom+22}" text-anchor="middle" font-size="12">{value:g}</text>'
        )
    for _label, own_x_values, values, color in normalized_series:
        points = " ".join(
            f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(own_x_values, values, strict=True)
        )
        body.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round"/>'
        )
        for x, y in zip(own_x_values, values, strict=True):
            body.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3.5" fill="{color}"/>')
    legend_x = left
    for index, (label, _, _, color) in enumerate(normalized_series):
        x = legend_x + index * 245
        body.append(
            f'<line x1="{x}" y1="{top-18}" x2="{x+24}" y2="{top-18}" stroke="{color}" stroke-width="4"/>'
        )
        body.append(f'<text x="{x+31}" y="{top-13}" font-size="13">{html.escape(label)}</text>')
    body.append(
        f'<line class="axis" x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}"/>'
    )
    body.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}"/>')
    body.append(
        f'<text x="{left+plot_width/2}" y="{height-16}" text-anchor="middle" font-size="14">{html.escape(x_label)}</text>'
    )
    body.append(
        f'<text x="18" y="{top+plot_height/2}" text-anchor="middle" font-size="14" transform="rotate(-90 18 {top+plot_height/2})">{html.escape(y_label)}</text>'
    )
    write_svg(path, title, "\n".join(body), width, height)


def temperature_chart(path: Path, rows: list[dict[str, str]]) -> None:
    width, height = 980, 600
    left, right = 82, 42
    x_values = [float(row["temperature"]) for row in rows]
    means = [float(row["average_return"]) for row in rows]
    stds = [float(row["standard_deviation_return"]) for row in rows]
    successes = [float(row["success_rate_percent"]) for row in rows]
    x_low, x_high = min(x_values), max(x_values)

    def sx(value: float) -> float:
        return left + (value - x_low) * (width - left - right) / (x_high - x_low)

    def panel(
        top: int, panel_height: int, values: list[float], low: float, high: float, color: str
    ) -> tuple[list[str], callable]:
        def sy(value: float) -> float:
            return top + (high - value) * panel_height / (high - low)

        parts: list[str] = []
        for tick in range(5):
            value = low + (high - low) * tick / 4
            y = sy(value)
            parts.append(
                f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}"/>'
            )
            parts.append(
                f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-size="12">{value:.0f}</text>'
            )
        points = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(x_values, values, strict=True))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        return parts, sy

    body, sy_return = panel(72, 210, means, -70, 230, COLORS[0])
    for x, mean, std in zip(x_values, means, stds, strict=True):
        y1, y2 = sy_return(min(230, mean + std)), sy_return(max(-70, mean - std))
        body.append(
            f'<line x1="{sx(x):.1f}" y1="{y1:.1f}" x2="{sx(x):.1f}" y2="{y2:.1f}" stroke="{COLORS[0]}" stroke-width="1.5"/>'
        )
        body.append(
            f'<circle cx="{sx(x):.1f}" cy="{sy_return(mean):.1f}" r="4" fill="{COLORS[0]}"/>'
        )
    body.append(
        '<text x="18" y="177" text-anchor="middle" font-size="14" transform="rotate(-90 18 177)">Return medio (media +/- DS)</text>'
    )
    lower_body, sy_success = panel(340, 150, successes, 0, 100, COLORS[1])
    body.extend(lower_body)
    for x, value in zip(x_values, successes, strict=True):
        body.append(
            f'<circle cx="{sx(x):.1f}" cy="{sy_success(value):.1f}" r="4" fill="{COLORS[1]}"/>'
        )
    for value in x_values:
        body.append(
            f'<text x="{sx(value):.1f}" y="512" text-anchor="middle" font-size="11">{value:.2f}</text>'
        )
    body.append(
        '<text x="18" y="415" text-anchor="middle" font-size="14" transform="rotate(-90 18 415)">Percentuale di successo (%)</text>'
    )
    body.append(
        f'<text x="{(left+width-right)/2}" y="548" text-anchor="middle" font-size="14">Temperatura di campionamento</text>'
    )
    body.append(
        '<text x="490" y="578" text-anchor="middle" font-size="12">Checkpoint selezionato; 100 episodi per temperatura; barre pari a una deviazione standard.</text>'
    )
    write_svg(
        path, "Sweep della temperatura della policy LunarLander", "\n".join(body), width, height
    )


def extract_png(notebook: Path, source_marker: str, output_index: int, destination: Path) -> None:
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    matching_cells = [
        cell for cell in payload["cells"] if source_marker in "".join(cell.get("source", []))
    ]
    if len(matching_cells) != 1:
        raise RuntimeError(
            f"Attesa una cella con il marcatore {source_marker!r} in {notebook}, "
            f"trovate {len(matching_cells)}."
        )
    image_outputs = [
        output["data"]["image/png"]
        for output in matching_cells[0].get("outputs", [])
        if "image/png" in output.get("data", {})
    ]
    if output_index >= len(image_outputs):
        if destination.is_file():
            print(
                f"Figura conservata perché l'output PNG non è incorporato nel notebook: {destination}"
            )
            return
        raise RuntimeError(
            f"Output PNG {output_index} assente nella cella {source_marker!r} di {notebook}."
        )
    encoded = image_outputs[output_index]
    if isinstance(encoded, list):
        encoded = "".join(encoded)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(base64.b64decode(encoded))


def build_lab1() -> None:
    results = ROOT / "DLA_1" / "results"
    figures = ROOT / "DLA_1" / "figures"
    metrics = read_csv(results / "test_metrics.csv")
    accuracy_rows = [
        (italian_label(row["experiment"]), float(row["value"]))
        for row in metrics
        if row["metric"] == "test_accuracy"
    ]
    horizontal_bars(
        figures / "gtsrb_test_accuracy_comparison.svg",
        "Accuracy di test GTSRB",
        accuracy_rows,
        "Accuracy",
        1.0,
    )
    retrieval_rows = [
        (
            row["metric"]
            .replace("precision_at", "precisione a")
            .replace("mean_average_precision", "precisione media")
            .replace("_", " "),
            float(row["value"]),
        )
        for row in metrics
        if row["task"] == "GTSRB retrieval"
    ]
    horizontal_bars(
        figures / "gtsrb_retrieval_metrics.svg",
        "Metriche di retrieval coseno GTSRB",
        retrieval_rows,
        "Valore della metrica",
        0.55,
    )
    runs = read_csv(results / "run_validation_summary.csv")
    horizontal_bars(
        figures / "gtsrb_validation_accuracy_summary.svg",
        "Migliore accuracy di validazione delle run salvate",
        [
            (row["experiment"].replace("ex3_1_", ""), float(row["best_validation_accuracy"]))
            for row in runs
        ],
        "Accuracy di validazione",
        1.0,
    )
    baseline = read_csv(results / "baseline_history.csv")
    improved = read_csv(results / "improved_history.csv")
    line_chart(
        figures / "gtsrb_training_curves.svg",
        "Accuracy di validazione durante il fine-tuning",
        [float(row["epoch"]) for row in improved],
        [
            (
                "Fine-tuning migliorato",
                [float(row["epoch"]) for row in improved],
                [float(row["val_acc"]) for row in improved],
                COLORS[0],
            ),
            (
                "Baseline della sola testa",
                [float(row["epoch"]) for row in baseline],
                [float(row["val_acc"]) for row in baseline],
                COLORS[1],
            ),
        ],
        "Epoca",
        "Accuracy di validazione",
        0.35,
        0.82,
    )
    source = ROOT / "DLA_1" / "notebooks" / "01_eda_and_feature_baseline.ipynb"
    extract_png(source, 'plt.suptitle("Campioni grezzi', 0, figures / "gtsrb_examples.png")
    extract_png(source, "aspect_ratio.hist", 0, figures / "gtsrb_image_geometry.png")
    extract_png(
        source, "plot_class_distribution(train_meta", 0, figures / "gtsrb_class_distribution.png"
    )


def build_lab2() -> None:
    results = ROOT / "DLA_2" / "results"
    figures = ROOT / "DLA_2" / "figures"
    sentiment = [
        row for row in read_csv(results / "sentiment_results.csv") if row["split"] == "test"
    ]
    horizontal_bars(
        figures / "sentiment_test_accuracy.svg",
        "Accuracy di test Rotten Tomatoes",
        [(italian_label(row["method"]), float(row["accuracy"])) for row in sentiment],
        "Accuracy",
        1.0,
    )
    horizontal_bars(
        figures / "sentiment_trainable_parameters.svg",
        "Quota di parametri addestrabili",
        [
            (italian_label(row["method"]), float(row["trainable_percent"]))
            for row in sentiment
            if row["trainable_percent"]
        ],
        "Parametri addestrabili (%)",
        105.0,
    )
    history = read_csv(results / "full_finetuning_history.csv")
    line_chart(
        figures / "distilbert_validation_history.svg",
        "Cronologia di validazione del fine-tuning completo DistilBERT",
        [float(row["epoch"]) for row in history],
        [
            ("Accuracy", [float(row["validation_accuracy"]) for row in history], COLORS[0]),
            ("F1", [float(row["validation_f1"]) for row in history], COLORS[1]),
        ],
        "Epoca",
        "Metrica",
        0.80,
        0.87,
    )
    clip = read_csv(results / "clip_results.csv")
    horizontal_bars(
        figures / "clip_accuracy_comparison.svg",
        "Accuracy ImageNet-Sketch",
        [(italian_label(row["method"]), float(row["accuracy"])) for row in clip],
        "Accuracy",
        0.60,
    )
    losses = read_csv(results / "adapter_loss_tail.csv")
    epochs = sorted({float(row["epoch"]) for row in losses})
    series = []
    for adapter, color in [("bottleneck 64", COLORS[0]), ("bottleneck 128", COLORS[1])]:
        rows = [row for row in losses if row["adapter"] == adapter]
        series.append((adapter + " training", [float(row["train_loss"]) for row in rows], color))
    line_chart(
        figures / "clip_adapter_training_loss_tail.svg",
        "Coda della loss di training CLIP-Adapter",
        epochs,
        series,
        "Epoca",
        "Loss di training",
        0.05,
        0.34,
    )


def build_lab3() -> None:
    results = ROOT / "DLA_3" / "results"
    figures = ROOT / "DLA_3" / "figures"
    method_summary = read_csv(results / "method_summary.csv")
    primary_rows = []
    primary_methods = {
        ("REINFORCE standardized returns", "best_evaluation_return"): "REINFORCE",
        ("REINFORCE with value baseline", "best_evaluation_return"): "Baseline di valore",
        ("A2C", "greedy_average_return"): "A2C CartPole",
    }
    for row in method_summary:
        label = primary_methods.get((row["method"], row["metric"]))
        if row["environment"] == "CartPole-v1" and label:
            primary_rows.append((label, float(row["value"])))
    horizontal_bars(
        figures / "cartpole_primary_returns.svg",
        "Return principali CartPole",
        primary_rows,
        "Return medio di valutazione",
        520.0,
    )
    cartpole = read_csv(results / "cartpole_evaluation.csv")
    line_chart(
        figures / "cartpole_evaluation_curves.svg",
        "Valutazione greedy periodica CartPole",
        [float(row["episode"]) for row in cartpole],
        [
            (
                "Return standardizzati",
                [float(row["standardized_return"]) for row in cartpole],
                COLORS[0],
            ),
            ("Return grezzi", [float(row["raw_return"]) for row in cartpole], COLORS[1]),
            (
                "Baseline di valore",
                [float(row["value_baseline_return"]) for row in cartpole],
                COLORS[2],
            ),
        ],
        "Episodio di training",
        "Return medio (20 episodi di valutazione)",
        0,
        520,
    )
    temperature_chart(
        figures / "lunarlander_temperature_sweep.svg",
        read_csv(results / "lunarlander_temperature_sweep.csv"),
    )
    policy_modes = read_csv(results / "lunarlander_policy_mode_comparison.csv")
    horizontal_bars(
        figures / "lunarlander_policy_mode_comparison.svg",
        "LunarLander: policy greedy e stocastica selezionata",
        [
            (f'{row["mode"]} (T={row["temperature"]})', float(row["average_return"]))
            for row in policy_modes
        ],
        "Return medio su 100 episodi",
        210.0,
    )
    final = json.loads((results / "lunarlander_final_evaluation.json").read_text(encoding="utf-8"))
    horizontal_bars(
        figures / "lunarlander_final_evaluation.svg",
        "Valutazione finale LunarLander su 200 episodi",
        [
            (f"Return medio {final['average_return']:.2f} / 300", final["average_return"] / 300),
            (
                f"Deviazione standard {final['standard_deviation_return']:.2f} / 300",
                final["standard_deviation_return"] / 300,
            ),
            (
                f"Percentuale di successo {final['success_rate_percent']:.1f}%",
                final["success_rate_percent"] / 100,
            ),
            (
                "Percentuale di terminazione",
                final["terminated_episodes"] / final["evaluation_episodes"],
            ),
            ("Percentuale senza troncamento", 1 - final["truncation_rate_percent"] / 100),
        ],
        "Valore normalizzato (unità nel README)",
        1.05,
    )
    labels = ["Nessuna azione", "Motore sinistro", "Motore principale", "Motore destro"]
    horizontal_bars(
        figures / "lunarlander_action_frequencies.svg",
        "Frequenze delle azioni LunarLander",
        [
            (label + " - episodio completo", value)
            for label, value in zip(labels, final["action_frequencies"], strict=True)
        ]
        + [
            (label + " - ultimo quarto", value)
            for label, value in zip(labels, final["last_quarter_action_frequencies"], strict=True)
        ],
        "Frequenza dell'azione",
        0.65,
    )
    extract_png(
        ROOT / "DLA_3" / "notebooks" / "01_cartpole_reinforce_evaluation.ipynb",
        "plot_training_returns(",
        0,
        figures / "reinforce_training_returns.png",
    )
    extract_png(
        ROOT / "DLA_3" / "notebooks" / "01_cartpole_reinforce_evaluation.ipynb",
        "plot_training_returns(",
        1,
        figures / "reinforce_periodic_evaluation.png",
    )
    extract_png(
        ROOT / "DLA_3" / "notebooks" / "02_cartpole_value_baseline.ipynb",
        "CartPole-v1 - confronto delle baseline",
        0,
        figures / "cartpole_value_baseline_comparison.png",
    )
    extract_png(
        ROOT / "DLA_3" / "notebooks" / "03_a2c_cartpole_lunarlander.ipynb",
        "plot_lunar_selection(selection_results",
        0,
        figures / "lunarlander_checkpoint_selection.png",
    )


def main() -> None:
    build_lab1()
    build_lab2()
    build_lab3()
    print("Figure dei report rigenerate da evidenze versionate e output dei notebook.")


if __name__ == "__main__":
    main()
