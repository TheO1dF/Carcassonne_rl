"""
train.py
========

GPU-accelerated, generation-based **self-play** training for ``CarcassonneEnv``
using ``sb3_contrib.MaskablePPO``.

Built for a many-core CPU + a single CUDA GPU:

* **24 parallel ``SubprocVecEnv`` workers** collect experience (configurable).
* The **learner runs on the GPU** (``device="cuda"``); large ``batch_size`` and
  ``n_epochs=10`` keep the GPU busy on each update.
* The **opponent in every worker stays on CPU** -- loading a CUDA model into all
  24 worker processes would create 24 CUDA contexts and exhaust GPU memory.
  Each worker also caps torch to 1 thread to avoid oversubscribing the CPU.
* **Resumable**: ``--start-gen`` / ``--resume-model`` continue a stopped run, and
  a ``CheckpointCallback`` saves every 100k steps so Ctrl+C never loses much.

Required packages:
    torch (CUDA build)  gymnasium  stable-baselines3  sb3-contrib  numpy  tensorboard

Usage
-----
    python train.py                                   # 3 gens, 2M steps, 24 envs, cuda
    python train.py --generations 4 --timesteps 3000000

    # Resume gen 2 from a checkpoint after Ctrl+C:
    python train.py --start-gen 2 \
        --resume-model models/checkpoints/ckpt_gen2_600000_steps.zip

    python train.py --eval-only models/model_gen3.zip
"""

from __future__ import annotations

import argparse
import os

import torch

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    BaseCallback, CallbackList, CheckpointCallback)
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.utils import get_action_masks

from env import CarcassonneEnv
from rl_model import CarcassonneFeaturesExtractor, SelfPlayWrapper, mask_fn


# Observation/action WINDOW side. The board itself is large & dict-backed; this
# window recenters on the played area each step, so it only needs to exceed a
# game's *span* (~20-24 for a 2p base game), NOT 2x the drift from a fixed centre
# like the old grid did. MUST be identical across all generations (it fixes the
# observation shape and action space) AND match the window used to translate any
# Behavioural-Cloning kifus, or the action ids won't line up.
BOARD_SIZE = 32

# Scale defaults (override on the CLI).
# 8 workers keeps Windows 'spawn' RAM use sane (each worker re-imports the CUDA
# PyTorch build); the GPU update is the bottleneck, so more envs add little.
# Raise with --n-envs if you have RAM headroom.
N_ENVS = 8
N_STEPS = 1024          # per env -> rollout = N_STEPS * N_ENVS
BATCH_SIZE = 8192       # must divide the rollout buffer size
N_EPOCHS = 10

# Fine-tuning defaults: a very low LR, no entropy bonus, and a KL trust region
# keep self-play from destroying the cloned BC policy (catastrophic forgetting).
LR_DEFAULT = 1e-5
ENT_COEF_DEFAULT = 0.0
TARGET_KL_DEFAULT = 0.015
GAMMA = 0.997             # discount; must match VecNormalize's gamma
# Reward returns are large enough (win_bonus, point-margins, meeple terms) to
# dominate the value loss, so VecNormalize scales rewards by the running
# return-std to keep value targets near unit scale.
NORMALIZE_REWARD = True
VECNORM_FILE = "vecnormalize.pkl"   # saved next to each model for resume
WIN_BONUS = 10.0
# Reward is the zero-sum score margin plus a potential-based shaping term
# (Ng, Harada & Russell 1999) on available meeples:
#   F = gamma*Phi(s') - Phi(s),  Phi = MEEPLE_POTENTIAL * available_meeples,
#   Phi(terminal) = 0.
# This speeds up learning the place->complete->recover loop without changing the
# optimal policy. Set to 0.0 for pure zero-sum.
MEEPLE_POTENTIAL = 0.5
CHECKPOINT_FREQ = 100_000

MODELS_DIR = "models"
CKPT_DIR = os.path.join(MODELS_DIR, "checkpoints")
TB_DIR = "tb_logs"


# --------------------------------------------------------------------------- #
# Device helpers
# --------------------------------------------------------------------------- #
def print_device_info() -> None:
    """Confirm CUDA availability + GPU name (and flag CPU-only torch builds)."""
    print("=" * 64)
    print(f"PyTorch {torch.__version__}")
    cuda = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {cuda}")
    if cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"VRAM: {total:.1f} GiB")
    elif "+cpu" in torch.__version__:
        print("[!] This PyTorch is a CPU-ONLY build (note the '+cpu' tag).")
        print("    To use your GPU, install a CUDA build, e.g. (CUDA 12.4):")
        print("      pip uninstall -y torch")
        print("      pip install torch --index-url https://download.pytorch.org/whl/cu124")
    else:
        print("[!] CUDA not available -- check your NVIDIA driver / torch install.")
    print("=" * 64)


def resolve_device(requested: str) -> str:
    """Return a usable device, falling back to CPU if CUDA was asked for in vain."""
    if requested == "cuda" and not torch.cuda.is_available():
        print("[warn] device='cuda' requested but CUDA is unavailable -> using CPU.")
        return "cpu"
    return requested


def linear_schedule(initial: float, final: float):
    """Linear schedule for SB3.

    SB3 calls the schedule with ``progress_remaining`` going 1.0 -> 0.0 over
    training, so this returns ``initial`` at the start and decays to ``final``
    at the end. A decaying LR stops the weights from thrashing late in training.
    """
    def schedule(progress_remaining: float) -> float:
        return final + progress_remaining * (initial - final)
    return schedule


# --------------------------------------------------------------------------- #
# Environment construction
# --------------------------------------------------------------------------- #
def _make_single_env(opponent_path, seed, win_bonus=WIN_BONUS):
    """Build ONE fully-wrapped environment.

    ``opponent_path`` is a path string (or None for a random opponent). The
    opponent model is loaded *here* -- inside the worker process -- and always
    on CPU, so we never pickle a live model across the process boundary and
    never create a CUDA context per worker.
    """
    env = CarcassonneEnv(board_size=BOARD_SIZE)

    opponent = None
    if opponent_path is not None:
        # custom_objects forces SB3 to rebuild the opponent's rollout buffer for
        # a SINGLE env (n_envs=1, n_steps=1) instead of inheriting the saved
        # model's n_envs (e.g. 24). Otherwise EACH of the 24 worker processes
        # would allocate a 24-wide buffer -> 24x24 buffers -> instant RAM OOM.
        # The opponent is inference-only here, so a 1-step buffer is plenty.
        opponent = MaskablePPO.load(
            opponent_path, device="cpu",
            custom_objects={"n_envs": 1, "n_steps": 1})
        opponent.policy.set_training_mode(False)  # inference only

    env = SelfPlayWrapper(env, opponent_policy=opponent, win_bonus=win_bonus,
                          meeple_potential=MEEPLE_POTENTIAL,
                          gamma=GAMMA)
    env = Monitor(env)
    env = ActionMasker(env, mask_fn)
    return env


def _env_thunk(opponent_path, seed):
    """Return a zero-arg factory for one worker env.

    ``SubprocVecEnv`` serialises these factories with cloudpickle, which can
    pickle this closure because it only captures a path string and an int.
    """
    def _init():
        # Each worker is one process; keep its CPU torch single-threaded so 24
        # workers don't oversubscribe the cores (the GPU learner is elsewhere).
        torch.set_num_threads(1)
        return _make_single_env(opponent_path, seed)
    return _init


def build_vec_env(opponent_path=None, n_envs=N_ENVS, seed=0):
    """``n_envs`` parallel self-play environments, with reward normalisation.

    The ``VecNormalize`` wrapper (``norm_reward=True``) divides rewards by the
    running std of the discounted return, so the critic's targets are unit-scale
    instead of ~15-20. ``norm_obs=False``: our observation is already scaled and
    contains discrete/mask planes that must not be shifted.
    """
    venv = SubprocVecEnv(
        [_env_thunk(opponent_path, seed + i) for i in range(n_envs)])
    if NORMALIZE_REWARD:
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True,
                            gamma=GAMMA, clip_reward=10.0)
    return venv


def _vecnorm_path(model_path: str) -> str:
    """Reward-normalisation stats file that pairs with a model ``.zip``."""
    base = model_path[:-4] if model_path.endswith(".zip") else model_path
    return base + "_vecnormalize.pkl"


def build_single_env(opponent_path=None, seed=0):
    """A single (non-parallel) env, used for evaluation."""
    return _make_single_env(opponent_path, seed)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def _tb_log():
    """Tensorboard dir, or None if tensorboard isn't installed."""
    try:
        import tensorboard  # noqa: F401
        return TB_DIR
    except ImportError:
        print("[warn] tensorboard not installed -> no TB logging "
              "(pip install tensorboard to enable).")
        return None


def make_model(env, device="cuda", n_steps=N_STEPS, batch_size=BATCH_SIZE,
               n_epochs=N_EPOCHS, lr: float = LR_DEFAULT,
               ent_coef: float = ENT_COEF_DEFAULT,
               target_kl: float = TARGET_KL_DEFAULT, seed: int = 0) -> MaskablePPO:
    """Fresh MaskablePPO with the custom CNN+MLP feature extractor.

    ``lr`` is kept low (fine-tuning) and ``target_kl`` hard-stops a rollout's
    update once the policy has moved ``target_kl`` nats away from the data-
    collecting policy -- together they stop self-play from tearing apart the
    cloned BC weights (catastrophic forgetting). ``ent_coef=0`` lets the already
    confident BC policy stay sharp instead of being pushed back toward uniform.
    """
    policy_kwargs = dict(
        features_extractor_class=CarcassonneFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=256, cnn_out_dim=256),
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
        normalize_images=False,
    )
    return MaskablePPO(
        policy="MultiInputPolicy",
        env=env,
        device=device,          # GPU learner
        learning_rate=lr,       # low, constant: safe fine-tuning of BC weights
        n_steps=n_steps,
        batch_size=batch_size,  # big batch to saturate the GPU
        n_epochs=n_epochs,      # passes per rollout
        gamma=GAMMA,
        gae_lambda=0.95,
        ent_coef=ent_coef,      # 0.0: don't push the confident BC policy apart
        clip_range=0.2,
        vf_coef=0.8,            # stronger critic to learn board value faster
        max_grad_norm=0.5,
        target_kl=target_kl,    # early-stop a rollout if the policy drifts too far
        policy_kwargs=policy_kwargs,
        tensorboard_log=_tb_log(),
        seed=seed,
        verbose=1,
    )


def _checkpoint_cb(tag: str, n_envs: int, checkpoint_freq: int) -> CheckpointCallback:
    """Save a checkpoint every ``checkpoint_freq`` *environment* steps.

    CheckpointCallback counts *calls* (one per vec-env step = ``n_envs``
    timesteps), so we divide the desired timestep frequency by ``n_envs``.
    """
    os.makedirs(CKPT_DIR, exist_ok=True)
    return CheckpointCallback(
        save_freq=max(checkpoint_freq // n_envs, 1),
        save_path=CKPT_DIR,
        name_prefix=f"ckpt_{tag}",
        verbose=1,
    )


class MeepleStatsCallback(BaseCallback):
    """Log the agent's per-game meeple economy to TensorBoard.

    ``SelfPlayWrapper`` stamps per-episode counts into the terminal ``info``
    dict; here we collect them across each rollout and record the means under
    the ``meeples/`` namespace. The headline metric is
    ``meeples/recovered_per_game`` -- recovery throughput, the place->complete->
    retrieve loop made visible (expert play ~4/game, BC ~1.25/game). Watch it
    climb; flat-low means the agent is still dumping meeples without recovering.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._rec: list[int] = []
        self._pla: list[int] = []
        self._str: list[int] = []
        self._pro: list[int] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "ep_meeples_recovered" in info:
                self._rec.append(info["ep_meeples_recovered"])
                self._pla.append(info["ep_meeples_placed"])
                self._str.append(info["ep_meeples_stranded"])
                self._pro.append(info["ep_meeples_productive"])
        return True

    def _on_rollout_end(self) -> None:
        if not self._rec:
            return
        import numpy as np
        self.logger.record("meeples/recovered_per_game", float(np.mean(self._rec)))
        self.logger.record("meeples/placed_per_game", float(np.mean(self._pla)))
        self.logger.record("meeples/stranded_per_game", float(np.mean(self._str)))
        self.logger.record("meeples/productive_stranded_per_game",
                           float(np.mean(self._pro)))
        self._rec.clear(); self._pla.clear(); self._str.clear(); self._pro.clear()


# --------------------------------------------------------------------------- #
# Evaluation (model as Player 0 vs the wrapper's opponent)
# --------------------------------------------------------------------------- #
def evaluate(model, opponent_path=None, n_games: int = 20, seed: int = 1_000_000):
    env = build_single_env(opponent_path, seed)
    wins = draws = losses = 0
    diff_sum = 0
    for g in range(n_games):
        obs, _ = env.reset(seed=seed + g)
        terminated = truncated = False
        while not (terminated or truncated):
            masks = get_action_masks(env)
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            obs, _r, terminated, truncated, _info = env.step(int(action))
        s0, s1 = env.unwrapped.scores
        diff_sum += s0 - s1
        wins += s0 > s1
        losses += s1 > s0
        draws += s0 == s1
    env.close()
    print(f"  vs {'random' if opponent_path is None else os.path.basename(opponent_path)}: "
          f"win={wins} draw={draws} loss={losses} "
          f"(win-rate {wins / n_games:.0%}, avg score diff {diff_sum / n_games:+.1f})")
    return wins / n_games


# --------------------------------------------------------------------------- #
# Generational self-play league (resumable)
# --------------------------------------------------------------------------- #
def train_league(generations: int = 3, timesteps: int = 2_000_000,
                 n_envs: int = N_ENVS, n_steps: int = N_STEPS,
                 batch_size: int = BATCH_SIZE, n_epochs: int = N_EPOCHS,
                 lr: float = LR_DEFAULT, ent_coef: float = ENT_COEF_DEFAULT,
                 target_kl: float = TARGET_KL_DEFAULT,
                 device: str = "cuda", seed: int = 0,
                 start_gen: int = 1, resume_model: str | None = None,
                 checkpoint_freq: int = CHECKPOINT_FREQ):
    os.makedirs(MODELS_DIR, exist_ok=True)
    device = resolve_device(device)

    # The opponent for the first generation we train is the previous gen's model.
    if start_gen > 1:
        opponent_path = os.path.join(MODELS_DIR, f"model_gen{start_gen - 1}")
        if not os.path.exists(opponent_path + ".zip"):
            raise SystemExit(
                f"--start-gen {start_gen} needs {opponent_path}.zip (the gen "
                f"{start_gen - 1} model) as the opponent, but it was not found.")
    else:
        # Gen 1: face the BC warm-start model (a competent opponent), not random.
        # A random opponent can't punish bad play, so over-deployment / camping is
        # never corrected; pressure to play well only comes from a strong opponent
        # (cf. AlphaZero/ZeusAI: always self-play, never random). Falls back to
        # random when we're not warm-starting from a checkpoint.
        opponent_path = resume_model

    for gen in range(start_gen, generations + 1):
        tag = f"gen{gen}"
        opp_desc = ("random" if opponent_path is None
                    else (f"gen{gen - 1}" if gen > start_gen else os.path.basename(opponent_path)))
        print(f"\n========== Training {tag} vs {opp_desc} "
              f"({n_envs} envs x {timesteps:,} steps, device={device}) ==========")

        venv = build_vec_env(opponent_path, n_envs=n_envs, seed=seed + gen * 100)

        # Resume the starting generation from a checkpoint, or start it fresh.
        if resume_model is not None and gen == start_gen:
            print(f"Resuming {tag} from {resume_model}")
            # Restore reward-normalisation stats if the checkpoint has them (a
            # self-play checkpoint will; the BC warm-start won't -- VecNormalize
            # then self-calibrates within the first rollout, which is fine).
            stats_path = _vecnorm_path(resume_model)
            if NORMALIZE_REWARD and os.path.exists(stats_path):
                venv = VecNormalize.load(stats_path, venv.venv)
                print(f"  restored reward-normalisation stats from {stats_path}")
            model = MaskablePPO.load(resume_model, env=venv, device=device)
            # MaskablePPO.load restores the saved model's optimiser
            # hyper-parameters, so override them with the CLI fine-tuning regime
            # (a low LR and a KL trust region) to avoid destroying the BC weights.
            model.batch_size, model.n_epochs = batch_size, n_epochs
            model.learning_rate = lr
            model.lr_schedule = lambda _progress_remaining, _lr=lr: _lr  # constant
            model.ent_coef = ent_coef
            model.target_kl = target_kl
            remaining = max(0, timesteps - model.num_timesteps)
            reset_counter = False
            print(f"  already trained {model.num_timesteps:,} steps; "
                  f"{remaining:,} remaining this generation.")
        else:
            model = make_model(venv, device=device, n_steps=n_steps,
                               batch_size=batch_size, n_epochs=n_epochs,
                               lr=lr, ent_coef=ent_coef, target_kl=target_kl,
                               seed=seed + gen)
            remaining = timesteps
            reset_counter = True

        callbacks = CallbackList([_checkpoint_cb(tag, n_envs, checkpoint_freq),
                                  MeepleStatsCallback()])

        try:
            if remaining > 0:
                model.learn(total_timesteps=remaining, callback=callbacks,
                            tb_log_name=tag, reset_num_timesteps=reset_counter,
                            progress_bar=False)
        except KeyboardInterrupt:
            # Save a recovery point and tell the user exactly how to resume.
            recovery = os.path.join(MODELS_DIR, f"model_{tag}_interrupted")
            model.save(recovery)
            if NORMALIZE_REWARD and isinstance(venv, VecNormalize):
                venv.save(_vecnorm_path(recovery))
            venv.close()
            print(f"\n[interrupted] saved {recovery}.zip")
            print("Resume with:")
            print(f"  python train.py --start-gen {gen} "
                  f"--resume-model {recovery}.zip --generations {generations}")
            return

        path = os.path.join(MODELS_DIR, f"model_{tag}")
        model.save(path)
        if NORMALIZE_REWARD and isinstance(venv, VecNormalize):
            venv.save(_vecnorm_path(path))
        venv.close()
        print(f"Saved {path}.zip")

        # Progress check vs random and vs the model just faced.
        evaluate(model, opponent_path=None, n_games=20)
        if opponent_path is not None:
            evaluate(model, opponent_path=opponent_path, n_games=20)

        opponent_path = path        # next generation faces this frozen model
        resume_model = None         # resume only applies to the starting gen

    print("\nLeague training complete. Evaluate with: python eval_league.py")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Carcassonne self-play training")
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--timesteps", type=int, default=2_000_000,
                        help="training timesteps per generation")
    parser.add_argument("--n-envs", type=int, default=N_ENVS,
                        help="parallel SubprocVecEnv workers")
    parser.add_argument("--n-steps", type=int, default=N_STEPS,
                        help="rollout steps per env (rollout = n_steps * n_envs)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--n-epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--lr", type=float, default=LR_DEFAULT,
                        help="learning rate (default: very low for safe BC fine-tuning)")
    parser.add_argument("--ent-coef", type=float, default=ENT_COEF_DEFAULT,
                        help="entropy bonus (default 0: keep the confident BC policy)")
    parser.add_argument("--target-kl", type=float, default=TARGET_KL_DEFAULT,
                        help="KL trust region; early-stops a rollout's updates "
                             "to prevent catastrophic forgetting")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=("cuda", "cpu"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-gen", type=int, default=1,
                        help="generation to begin at (for resuming a league)")
    parser.add_argument("--resume-model", type=str, default=None,
                        help="checkpoint/model .zip to continue the start gen from")
    parser.add_argument("--checkpoint-freq", type=int, default=CHECKPOINT_FREQ,
                        help="save a checkpoint every N timesteps")
    parser.add_argument("--eval-only", type=str, default=None,
                        help="evaluate a saved model vs random and exit")
    args = parser.parse_args()

    print_device_info()

    if args.eval_only:
        device = resolve_device(args.device)
        model = MaskablePPO.load(args.eval_only, device=device)
        evaluate(model, opponent_path=None, n_games=50)
        return

    train_league(
        generations=args.generations, timesteps=args.timesteps,
        n_envs=args.n_envs, n_steps=args.n_steps, batch_size=args.batch_size,
        n_epochs=args.n_epochs, lr=args.lr, ent_coef=args.ent_coef,
        target_kl=args.target_kl, device=args.device, seed=args.seed,
        start_gen=args.start_gen, resume_model=args.resume_model,
        checkpoint_freq=args.checkpoint_freq,
    )


if __name__ == "__main__":
    main()
