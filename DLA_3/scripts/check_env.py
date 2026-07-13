from __future__ import annotations

import importlib
import platform
import sys

REQUIRED_MODULES = [
    "numpy",
    "pandas",
    "matplotlib",
    "torch",
    "gymnasium",
    "pygame",
    "Box2D",
    "ipykernel",
]


def module_version(name: str) -> str:
    """Restituisce la versione di un modulo installato, quando disponibile.

    Argomenti:
        name: Nome del modulo importabile.

    Operazione:
        Importa il modulo e legge `__version__`, se presente.

    Output:
        Stringa di versione oppure `installed`.
    """

    module = importlib.import_module(name)
    return getattr(module, "__version__", "installato")


def main() -> None:
    print("Interprete:", sys.executable)
    print("Python:", sys.version)
    print("Piattaforma:", platform.platform())
    print()

    missing = []
    for module_name in REQUIRED_MODULES:
        try:
            version = module_version(module_name)
            print(f"[OK]      {module_name}: {version}")
        except Exception as exc:
            missing.append(module_name)
            print(f"[MANCANTE] {module_name}: {exc}")

    try:
        import dla_lab3

        print(f"[OK]      dla_lab3: {dla_lab3.__file__}")
    except Exception as exc:
        missing.append("dla_lab3")
        print(f"[MANCANTE] dla_lab3: {exc}")

    print()
    try:
        import gymnasium as gym

        for env_id in ["CartPole-v1", "LunarLander-v3"]:
            env = gym.make(env_id)
            obs, info = env.reset(seed=2112)
            print(
                f"[AMBIENTE OK] {env_id}: forma_osservazioni={env.observation_space.shape}, "
                f"spazio_azioni={env.action_space}, tipo_prima_osservazione={getattr(obs, 'dtype', type(obs))}"
            )
            env.close()
    except Exception as exc:
        missing.append("ambienti gymnasium")
        print(f"[ERRORE AMBIENTE] {exc}")

    if missing:
        print()
        print("Requisiti mancanti o non funzionanti:", ", ".join(missing))
        print("Installazione conda suggerita:")
        print(
            "  conda create -n DRL -c conda-forge python=3.12 torch gymnasium[box2d] matplotlib pygame jupyterlab jupyter ipykernel"
        )
        raise SystemExit(1)

    print()
    print("Controllo dell'ambiente superato.")


if __name__ == "__main__":
    main()
