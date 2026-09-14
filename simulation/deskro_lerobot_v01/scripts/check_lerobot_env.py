"""Load the local EnvHub file through the same helper documented by LeRobot."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / "envhub" / "env.py"

from lerobot.envs.utils import _call_make_env, _load_module_from_path, _normalize_hub_result

module = _load_module_from_path(str(ENV_FILE))
result = _call_make_env(module, n_envs=2, use_async_envs=False, cfg=None)
normalized = _normalize_hub_result(result)
suite_name = next(iter(normalized))
env = normalized[suite_name][0]

obs, info = env.reset(seed=7)
print("LeRobot local EnvHub load: OK")
print("suite:", suite_name)
print("observation keys:", list(obs))
print("agent_pos shape:", obs["agent_pos"].shape)
print("action space:", env.action_space)

for step in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(
        f"step={step:02d} reward_mean={float(np.mean(reward)):.3f} "
        f"done={bool(np.any(terminated | truncated))}"
    )

env.close()
