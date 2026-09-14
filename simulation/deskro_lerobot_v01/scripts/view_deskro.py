from pathlib import Path
import math
import sys
import time

import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "envhub"))
from deskro_env import DeskroV1Env


def main() -> None:
    env = DeskroV1Env(
        episode_length=100000,
        frame_skip=4,
        domain_randomization=False,
    )
    env.reset(seed=3)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.distance = 0.55
        viewer.cam.azimuth = 145
        viewer.cam.elevation = -15

        start = time.time()
        while viewer.is_running():
            t = time.time() - start

            # Gentle demo: normalized action -> about ±18° yaw and ±5° pitch.
            # One full yaw cycle takes roughly 25 seconds.
            yaw = 0.30 * math.sin(t * 0.25)
            pitch = 0.20 * math.sin(t * 0.18)
            face = 0.0

            env.step(np.array([yaw, pitch, face], dtype=np.float32))
            viewer.sync()
            time.sleep(env.model.opt.timestep * env.frame_skip)

    env.close()


if __name__ == "__main__":
    main()
