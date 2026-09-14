from pathlib import Path
import sys, time
import numpy as np
import mujoco.viewer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "envhub"))
from gaze_env import GazeTrackingDeskroEnv

env = GazeTrackingDeskroEnv(
    episode_length=260,
    domain_randomization=False,
    model_filename="model.xml",
)

seed = 1700
obs, info = env.reset(seed=seed)

print("No PPO is used.")
print("Orange sphere = raw target; the direct baseline follows the filtered target.")

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    while viewer.is_running():
        state = obs["agent_pos"]
        seen_yaw = float(state[4])
        seen_pitch = float(state[5])

        # Direct baseline action only to inspect reaction-delay + target smoothing
        action = np.array([
            seen_yaw / 60.0,
            seen_pitch / 25.0,
            0.0,
        ], dtype=np.float32)

        obs, _, term, trunc, info = env.step(action)
        viewer.sync()

        time.sleep(env.model.opt.timestep * env.frame_skip)

        if term or trunc:
            seed += 1
            time.sleep(0.4)
            obs, info = env.reset(seed=seed)
