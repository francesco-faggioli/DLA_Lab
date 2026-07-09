from __future__ import annotations

import importlib
import platform
import sys


REQUIRED_MODULES = [
    "numpy",
    "matplotlib",
    "torch",
    "gymnasium",
    "pygame",
    "Box2D",
    "yaml",
    "ipykernel",
]


def module_version(name: str) -> str:
    """Return an installed module version when available.

    Args:
        name: Importable module name.

    What it does:
        Imports the module and reads `__version__` if present.

    Outputs:
        Version string or `installed`.
    """

    module = importlib.import_module(name)
    return getattr(module, "__version__", "installed")


def main() -> None:
    print("Python:", sys.version)
    print("Platform:", platform.platform())
    print()

    missing = []
    for module_name in REQUIRED_MODULES:
        try:
            version = module_version(module_name)
            print(f"[OK]      {module_name}: {version}")
        except Exception as exc:
            missing.append(module_name)
            print(f"[MISSING] {module_name}: {exc}")

    print()
    try:
        import gymnasium as gym

        for env_id in ["CartPole-v1", "LunarLander-v3"]:
            env = gym.make(env_id)
            obs, info = env.reset(seed=2112)
            print(
                f"[ENV OK] {env_id}: obs_shape={env.observation_space.shape}, "
                f"action_space={env.action_space}, first_obs_dtype={getattr(obs, 'dtype', type(obs))}"
            )
            env.close()
    except Exception as exc:
        missing.append("gymnasium environments")
        print(f"[ENV ERROR] {exc}")

    if missing:
        print()
        print("Missing or failing requirements:", ", ".join(missing))
        print("Suggested conda install:")
        print("  conda create -n DRL -c conda-forge python=3.12 torch gymnasium[box2d] matplotlib pygame pyyaml jupyterlab jupyter ipykernel")
        raise SystemExit(1)

    print()
    print("Environment check passed.")


if __name__ == "__main__":
    main()
