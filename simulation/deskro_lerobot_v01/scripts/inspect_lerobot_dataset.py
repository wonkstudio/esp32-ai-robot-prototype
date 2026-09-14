import argparse
from pathlib import Path

from lerobot.datasets import LeRobotDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-id", default="wonkstudio/deskro-sim-v01")
    args = parser.parse_args()

    dataset = LeRobotDataset(args.repo_id, root=args.root)
    print("episodes:", dataset.num_episodes)
    print("frames:", dataset.num_frames)
    print("fps:", dataset.fps)
    if dataset.num_frames:
        item = dataset[0]
        print("sample keys:", sorted(item.keys()))
        print("observation.state:", item["observation.state"])
        print("action:", item["action"])


if __name__ == "__main__":
    main()
