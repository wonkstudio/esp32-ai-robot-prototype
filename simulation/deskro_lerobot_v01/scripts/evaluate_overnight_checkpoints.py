from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import numpy as np
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "envhub"))
from overnight_settling_env import OvernightSettlingGazeEnv


def run_model(model, randomized, episodes, seed_base):
    env = OvernightSettlingGazeEnv(
        domain_randomization=randomized,
        model_filename="model_train.xml",
    )
    rows = []

    for i in range(episodes):
        obs, _ = env.reset(seed=seed_base + i)
        done = False
        info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            done = bool(term or trunc)

        rows.append({
            "success": float(info.get("is_success", False)),
            "yaw_error": float(info["raw_yaw_error_deg"]),
            "pitch_error": float(info["raw_pitch_error_deg"]),
            "yaw_speed": abs(float(info["yaw_velocity_deg_s"])),
            "pitch_speed": abs(float(info["pitch_velocity_deg_s"])),
        })

    env.close()
    return {
        "episodes": int(episodes),
        "success_rate": float(np.mean([x["success"] for x in rows])),
        "final_yaw_error_mean_deg": float(
            np.mean([x["yaw_error"] for x in rows])
        ),
        "final_pitch_error_mean_deg": float(
            np.mean([x["pitch_error"] for x in rows])
        ),
        "final_yaw_speed_mean_deg_s": float(
            np.mean([x["yaw_speed"] for x in rows])
        ),
        "final_pitch_speed_mean_deg_s": float(
            np.mean([x["pitch_speed"] for x in rows])
        ),
    }


def selection_cost(nominal, robust):
    """Lower is better. Focus on settling without accepting large error."""
    return float(
        nominal["final_yaw_error_mean_deg"] / 4.0
        + nominal["final_pitch_error_mean_deg"] / 2.0
        + nominal["final_yaw_speed_mean_deg_s"] / 10.0
        + nominal["final_pitch_speed_mean_deg_s"] / 7.0
        + 0.60 * robust["final_yaw_error_mean_deg"] / 6.0
        + 0.60 * robust["final_pitch_error_mean_deg"] / 3.0
        + 0.60 * robust["final_yaw_speed_mean_deg_s"] / 15.0
        + 0.60 * robust["final_pitch_speed_mean_deg_s"] / 10.0
        - 2.0 * nominal["success_rate"]
        - 1.0 * robust["success_rate"]
    )


def candidates():
    folder = ROOT / "outputs" / "overnight_settling_6h"
    found = []

    for hour in range(1, 7):
        path = folder / f"deskro_overnight_hour_{hour:02d}.zip"
        if path.exists():
            found.append((f"hour_{hour:02d}", path))

    final = ROOT / "outputs" / "deskro_overnight_settling_6h.zip"
    if final.exists():
        found.append(("final", final))

    return found


def main():
    found = candidates()
    if not found:
        raise SystemExit(
            "No overnight checkpoints found.\n"
            "Run .\\41_train_overnight_settling_6h.bat first."
        )

    print("Quick checkpoint comparison")
    print("50 nominal + 50 robustness episodes per checkpoint.")
    print("Lower selection_cost is better.")
    print()

    results = []
    for idx, (name, path) in enumerate(found):
        print(f"Evaluating {name}: {path}")
        model = PPO.load(path)

        nominal = run_model(
            model, randomized=False, episodes=50,
            seed_base=50000 + idx * 1000,
        )
        robust = run_model(
            model, randomized=True, episodes=50,
            seed_base=60000 + idx * 1000,
        )
        cost = selection_cost(nominal, robust)

        row = {
            "name": name,
            "path": str(path),
            "selection_cost": cost,
            "nominal": nominal,
            "robustness": robust,
        }
        results.append(row)

        print(" NOMINAL  ", nominal)
        print(" ROBUST   ", robust)
        print(f" COST     {cost:.4f}")
        print()

    best = min(results, key=lambda x: x["selection_cost"])
    best_src = Path(best["path"])
    best_dst = ROOT / "outputs" / "deskro_overnight_best.zip"
    shutil.copy2(best_src, best_dst)

    report_dir = ROOT / "outputs" / "overnight_settling_6h"
    report_path = report_dir / "checkpoint_evaluation.json"
    report_path.write_text(
        json.dumps(
            {
                "selection_note": (
                    "Automatic candidate selection prioritizes low residual "
                    "motion while penalizing loss of yaw/pitch accuracy. "
                    "Final visual judgement still matters."
                ),
                "best": best,
                "all_results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 68)
    print("AUTO-SELECTED CANDIDATE:", best["name"])
    print("Copied to:", best_dst)
    print("Report:", report_path)
    print()
    print("Next:")
    print("  .\\43_watch_overnight_best.bat")
    print("  .\\44_eval_overnight_best.bat")


if __name__ == "__main__":
    main()
