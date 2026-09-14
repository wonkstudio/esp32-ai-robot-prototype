from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

import numpy as np
from lerobot.datasets import LeRobotDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "envhub"))
from deskro_env import DeskroV1Env


STATE_NAMES = [
    "head_yaw_deg",
    "head_pitch_deg",
    "yaw_velocity_deg_s",
    "pitch_velocity_deg_s",
    "target_yaw_deg",
    "target_pitch_deg",
    "wake_word",
    "face_state_normalized",
]
ACTION_NAMES = ["target_yaw_normalized", "target_pitch_normalized", "face_normalized"]


def heuristic_action(state: np.ndarray) -> np.ndarray:
    wake_word = state[6] >= 0.5
    if wake_word:
        return np.array(
            [
                np.clip(state[4] / 60.0, -1.0, 1.0),
                np.clip(state[5] / 25.0, -1.0, 1.0),
                0.0,  # face state 2 / big_smile
            ],
            dtype=np.float32,
        )
    return np.array([0.0, 0.0, -1.0], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output", type=Path, default=ROOT / "datasets" / "deskro_sim_v01")
    parser.add_argument("--repo-id", default="wonkstudio/deskro-sim-v01")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists():
        if not args.overwrite:
            raise SystemExit(f"Dataset path already exists: {args.output}. Add --overwrite to replace it.")
        shutil.rmtree(args.output)

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": STATE_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (3,),
            "names": ACTION_NAMES,
        },
        "next.reward": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["reward"],
        },
        "next.done": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["done"],
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=args.output,
        fps=20,
        robot_type="deskro_v1_sim",
        features=features,
        use_videos=False,
    )

    env = DeskroV1Env(domain_randomization=True)
    try:
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=1000 + episode)
            done = False
            frames = 0
            while not done:
                state = obs["agent_pos"]
                action = heuristic_action(state)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)

                dataset.add_frame(
                    {
                        "observation.state": state.astype(np.float32),
                        "action": action.astype(np.float32),
                        "next.reward": np.array([reward], dtype=np.float32),
                        "next.done": np.array([float(done)], dtype=np.float32),
                        "task": env.task_description,
                    }
                )
                obs = next_obs
                frames += 1

            dataset.save_episode()
            print(f"episode {episode + 1}/{args.episodes}: {frames} frames, success={info['is_success']}")
    finally:
        env.close()
        dataset.finalize()

    print("LeRobotDataset created:", args.output)


if __name__ == "__main__":
    main()
