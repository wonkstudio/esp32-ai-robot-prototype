"""LeRobot EnvHub entry point for DESKRO V1.

LeRobot only requires this module to expose ``make_env``.  It may return a
single Gymnasium environment or a vector environment.  We return a vector
environment so the same repository can later train several simulations in
parallel.
"""

from __future__ import annotations

from pathlib import Path
import sys

import gymnasium as gym

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from deskro_env import DeskroV1Env  # noqa: E402


def make_env(
    n_envs: int = 1,
    use_async_envs: bool = False,
    cfg=None,
):
    if n_envs < 1:
        raise ValueError("n_envs must be at least 1")

    episode_length = int(getattr(cfg, "episode_length", 240)) if cfg is not None else 240

    def make_one():
        return DeskroV1Env(
            episode_length=episode_length,
            domain_randomization=True,
        )

    vector_cls = gym.vector.AsyncVectorEnv if use_async_envs else gym.vector.SyncVectorEnv
    return vector_cls([make_one for _ in range(n_envs)])
