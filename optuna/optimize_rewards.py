import optuna
import subprocess
import re
import os
import shutil

# === Configuration ===
TRAIN_SCRIPT = "<path/to/your/IsaacLab>/scripts/reinforcement_learning/rsl_rl/train.py"      
TASK_NAME = "Isaac-G1locomotion-v0"                                                     #registered task name
LOG_DIR = "logs/optuna_rewards"

os.makedirs(LOG_DIR, exist_ok=True)


def run_training(params, trial_id):
    """Run IsaacLab training with given reward weights."""
    cmd = [
        "python", TRAIN_SCRIPT,
        f"--task={TASK_NAME}",
        "--headless",
        "--max_iterations=1500",        # keep small for tuning
        "--num_envs=8192"
    ]

    # === Inject reward weight overrides ===
    cmd += [
        f"env.rewards.track_lin_vel_xy_exp.weight={params['track_lin_vel_xy_exp']}",
        f"env.rewards.track_ang_vel_z_exp.weight={params['track_ang_vel_z_exp']}",
        f"env.rewards.symmetry_joint_motion.weight={params['symmetry_joint_motion']}",
        f"env.rewards.feet_air_time.weight={params['feet_air_time']}",
        f"env.rewards.feet_slide.weight={params['feet_slide']}",
        f"env.rewards.dof_pos_limits.weight={params['dof_pos_limits']}",
        f"env.rewards.joint_deviation_hip.weight={params['joint_deviation_hip']}",
        f"env.rewards.joint_deviation_arms.weight={params['joint_deviation_arms']}",
        f"env.rewards.joint_deviation_torso.weight={params['joint_deviation_torso']}",
        f"env.rewards.dof_torques_l2.weight={params['dof_torques_l2']}",
        f"env.rewards.dof_acc_l2.weight={params['dof_acc_l2']}",
        f"env.rewards.action_rate_l2.weight={params['action_rate_l2']}",
        f"env.rewards.flat_orientation_l2.weight={params['flat_orientation_l2']}",
      ]

    trial_dir = os.path.join(LOG_DIR, f"trial_{trial_id}")
    os.makedirs(trial_dir, exist_ok=True)
    print(f"\n[Optuna] Trial {trial_id}: running {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        stdout = result.stdout

        # Optional: save stdout for inspection
        with open(os.path.join(trial_dir, "stdout.txt"), "w") as f:
            f.write(stdout)

        # Try to extract final mean reward (adjust regex to your print output)
        match = re.search(r"Mean reward[:=]\s*([-+]?\d*\.\d+|\d+)", stdout)
        if match:
            reward = float(match.group(1))
        else:
            reward = -999.0  # fallback for failed runs
        return reward

    except Exception as e:
        print(f"[Optuna] Trial {trial_id} failed: {e}")
        return -999.0
    finally:
        shutil.rmtree(trial_dir, ignore_errors=True)


def objective(trial: optuna.Trial):
    """Optuna objective function defining the search space."""
    params = {
        "track_lin_vel_xy_exp": trial.suggest_float("track_lin_vel_xy_exp", 0.5, 3.0),
        "track_ang_vel_z_exp": trial.suggest_float("track_ang_vel_z_exp", 0.1, 1.0),
        "symmetry_joint_motion": trial.suggest_float("symmetry_joint_motion", 0.0, 0.2),
        "feet_air_time": trial.suggest_float("feet_air_time", 0.0, 0.3),
        "feet_slide": trial.suggest_float("feet_slide", -0.5, -0.01),
        "dof_pos_limits": trial.suggest_float("dof_pos_limits", -2, -0.001),
        "joint_deviation_hip": trial.suggest_float("joint_deviation_hip", -0.5, -0.001),
        "joint_deviation_arms": trial.suggest_float("joint_deviation_arms", -0.5, -0.001),
        "joint_deviation_torso": trial.suggest_float("joint_deviation_torso", -0.5, -0.001),
        "action_rate_l2": trial.suggest_float("action_rate_l2", -2e-2, -1e-6),
        "dof_torques_l2": trial.suggest_float("dof_torques_l2", -1e-4, -1e-6),
        "dof_acc_l2": trial.suggest_float("dof_acc_l2", -1e-4, -2.5e-7),
        "flat_orientation_l2": trial.suggest_float("flat_orientation_l2", -0.5, 0.),
    }
    return run_training(params, trial.number)


if __name__ == "__main__":
    study = optuna.create_study(
        study_name="isaac_g1_reward_tuning_study",
        direction="maximize",
        storage="sqlite:///isaac_g1_optuna.db",
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=10, n_jobs=1)

    print("\n=== Optimization Complete ===")
    print(f"Best Trial: {study.best_trial.number}")
    print(f"Best Params: {study.best_trial.params}")
    
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")

