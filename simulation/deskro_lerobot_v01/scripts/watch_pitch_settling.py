from pathlib import Path
import sys, time
import mujoco.viewer
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "envhub"))
from pitch_settling_env import PitchSettlingDeskroEnv

model_path = ROOT / "outputs" / "deskro_pitch_settling_15m.zip"
if not model_path.exists():
    raise SystemExit(f"Model not found: {model_path}")

model = PPO.load(model_path)
env = PitchSettlingDeskroEnv(domain_randomization=False)
seed = 900
obs, _ = env.reset(seed=seed)

print("Watching final settling model.")
print("Check whether it reaches the target and actually stops.")

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    step = 0
    while viewer.is_running():
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        viewer.sync()

        if step % 10 == 0:
            print(
                f"pitch={info['head_pitch_deg']:+5.1f} "
                f"target={info['target_pitch_deg']:+5.1f} "
                f"error={info['pitch_error_deg']:4.1f} "
                f"speed={info['pitch_velocity_deg_s']:+6.1f} "
                f"streak={info['settled_streak']:02d}"
            )

        time.sleep(env.model.opt.timestep * env.frame_skip)
        step += 1

        if term or trunc:
            seed += 1
            obs, _ = env.reset(seed=seed)
            step = 0
