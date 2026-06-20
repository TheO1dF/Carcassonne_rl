"""Behavioural cloning (supervised pre-training) for the Carcassonne policy.

Clones the winner of each recorded BGA game: for every move by the winning seat
we record (observation, action-mask, expert-action) and a value target from the
final margin, then train the same CNN+MLP MaskablePPO policy to imitate the moves
(illegal actions are masked out of the logits) and regress the value head. The
resulting models/model_bc.zip is the starting point for train.py / alphazero.py.

Usage:
    python train_bc.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader

from replay_env import CarcassonneReplayEnv
from train import make_model, resolve_device, BOARD_SIZE

try:                                   # live progress bar if available
    from tqdm import tqdm
except ImportError:                    # fall back to periodic prints
    tqdm = None

KIFU_DIR = "kifus"
MODEL_OUT = "models/model_bc.zip"        # final model (last epoch)
MODEL_BEST = "models/model_bc_best.zip"  # best model (lowest avg-loss epoch)

# Defaults tuned for proper convergence: many epochs + a higher LR with cosine
# annealing, and a small batch so each epoch does many gradient updates.
EPOCHS_DEFAULT = 150
BATCH_DEFAULT = 128
LR_DEFAULT = 5e-4


# --------------------------------------------------------------------------- #
# 1. Transition extraction (winner-only)
# --------------------------------------------------------------------------- #
def extract_transitions(env, files=None):
    """Replay every kifu and collect the WINNER's (obs, mask, action, v_target).

    For each move we step the env exactly as recorded; we only *record* the
    transition when it is the winning seat's turn, so the dataset is pure
    "expert" play.

    ``v_target`` is a *critic warm-start* label: the winner's final score margin
    (normalised by 50) attached to every state on that winning trajectory. It
    lets BC pretrain the value head alongside the policy head, so PPO self-play
    starts with a critic that already ranks winning board states highly --
    instead of a random critic whose garbage advantages destroy the cloned
    policy (the "critic burn-in" pitfall: ep_rew_mean falls, explained_variance
    sits near 0).
    """
    if files is None:
        files = sorted(glob.glob(os.path.join(KIFU_DIR, "*.json")))

    obs_list, masks_list, expert_actions, v_targets = [], [], [], []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            kifu = json.load(f)
        winner = kifu["winner_seat"]
        if winner is None:                         # ties were filtered out already
            continue

        # Final score margin from the winner's perspective, kept at a sane scale
        # for the critic (margins run to ~50+, so /50 lands roughly in [-1, 1]+).
        margin = (kifu["final_scores"][winner]
                  - kifu["final_scores"][1 - winner])
        v_target = float(margin) / 50.0

        env.reset(tile_sequence=kifu["tile_sequence"],
                  start_rotation=kifu["start_rotation"])
        for action in kifu["actions"]:
            action = int(action)
            if env.current_player == winner:
                obs_list.append(env._build_obs())
                masks_list.append(env.get_valid_actions())
                expert_actions.append(action)
                v_targets.append(v_target)
            env.step(action)

    return obs_list, masks_list, expert_actions, v_targets


# --------------------------------------------------------------------------- #
# 2. Dataset + collate
# --------------------------------------------------------------------------- #
class TransitionDataset(Dataset):
    def __init__(self, obs_list, masks_list, expert_actions, v_targets):
        self.obs = obs_list
        self.masks = masks_list
        self.actions = expert_actions
        self.v_targets = v_targets

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, i):
        return self.obs[i], self.masks[i], self.actions[i], self.v_targets[i]


def collate_fn(batch):
    """Batch a list of (obs-dict, mask, action, v_target) into tensors.

    Each observation key is stacked into one tensor; masks become a bool tensor,
    actions a long tensor, and value targets a float tensor.
    """
    obs_keys = batch[0][0].keys()
    obs = {k: torch.as_tensor(np.stack([b[0][k] for b in batch]))
           for k in obs_keys}
    masks = torch.as_tensor(np.stack([b[1] for b in batch])).bool()
    actions = torch.as_tensor([b[2] for b in batch], dtype=torch.long)
    v_targets = torch.as_tensor([b[3] for b in batch], dtype=torch.float32)
    return obs, masks, actions, v_targets


# --------------------------------------------------------------------------- #
# 4. Supervised training loop
# --------------------------------------------------------------------------- #
def train(model, loader, epochs, lr, log_every=20):
    policy = model.policy
    device = model.device
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    # Cosine annealing: LR sweeps from `lr` down to ~0 over the run, which helps
    # the policy settle into a sharp imitation minimum instead of bouncing.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    os.makedirs(os.path.dirname(MODEL_BEST), exist_ok=True)
    best_loss = float("inf")
    n_batches = len(loader)

    for epoch in range(epochs):
        policy.train()
        running, run_pi, run_v, count = 0.0, 0.0, 0.0, 0
        lr_now = optimizer.param_groups[0]["lr"]    # LR used this epoch

        iterator = loader
        bar = None
        if tqdm is not None:
            bar = tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}",
                       unit="batch", leave=False)
            iterator = bar

        for bi, (obs, masks, expert, v_targets) in enumerate(iterator):
            obs = {k: v.to(device) for k, v in obs.items()}
            masks = masks.to(device)
            expert = expert.to(device)
            v_targets = v_targets.to(device).float()

            # --- Policy head: action logits (same path SB3 uses internally) ---
            features = policy.extract_features(obs)
            latent_pi, _ = policy.mlp_extractor(features)
            logits = policy.action_net(latent_pi)
            # Hard-mask illegal actions so the policy can't clone an illegal move.
            logits = logits.masked_fill(~masks, -1e9)
            pi_loss = F.cross_entropy(logits, expert)

            # --- Value head: pretrain the critic on the trajectory's outcome ---
            # so PPO self-play starts with a useful critic, not a random one.
            values = policy.predict_values(obs).flatten()
            value_loss = F.mse_loss(values, v_targets)

            loss = pi_loss + 0.5 * value_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n = expert.size(0)
            running += loss.item() * n
            run_pi += pi_loss.item() * n
            run_v += value_loss.item() * n
            count += n
            avg = running / max(count, 1)
            avg_pi = run_pi / max(count, 1)
            avg_v = run_v / max(count, 1)
            if bar is not None:
                bar.set_postfix(pi=f"{avg_pi:.4f}", v=f"{avg_v:.4f}",
                                lr=f"{lr_now:.1e}")
            elif (bi + 1) % log_every == 0:
                print(f"  epoch {epoch + 1} batch {bi + 1}/{n_batches} "
                      f"pi_loss {avg_pi:.4f}  v_loss {avg_v:.4f}")

        scheduler.step()                            # advance LR for the next epoch
        avg = running / max(count, 1)
        avg_pi = run_pi / max(count, 1)
        avg_v = run_v / max(count, 1)

        # Track best on the POLICY loss -- the imitation quality we ultimately
        # care about (the value head is a warm-start, not the objective).
        marker = ""
        if avg_pi < best_loss:
            best_loss = avg_pi
            model.save(MODEL_BEST)
            marker = "  <- best (saved)"
        print(f"epoch {epoch + 1:3d}/{epochs}  pi_loss {avg_pi:.4f}  "
              f"v_loss {avg_v:.4f}  lr {lr_now:.2e}{marker}")

    return best_loss


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Behavioural Cloning of the Carcassonne policy on winner kifus.")
    parser.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT,
                        help=f"training epochs (default: {EPOCHS_DEFAULT})")
    parser.add_argument("--batch-size", type=int, default=BATCH_DEFAULT,
                        help=f"minibatch size (default: {BATCH_DEFAULT})")
    parser.add_argument("--lr", type=float, default=LR_DEFAULT,
                        help=f"initial Adam learning rate (default: {LR_DEFAULT})")
    args = parser.parse_args()

    device = resolve_device("cuda")
    # BOARD_SIZE is the observation/action WINDOW (32); it MUST match the kifu
    # board_size and the self-play env so the action ids line up.
    env = CarcassonneReplayEnv(board_size=BOARD_SIZE)

    files = sorted(glob.glob(os.path.join(KIFU_DIR, "*.json")))
    print(f"Extracting winner-only transitions from {len(files)} kifus ...")
    obs_list, masks_list, expert_actions, v_targets = extract_transitions(env, files)
    print(f"  {len(expert_actions)} expert transitions.")
    if not expert_actions:
        sys.exit(f"No transitions found in '{KIFU_DIR}/'. Run batch_translate.py first.")

    dataset = TransitionDataset(obs_list, masks_list, expert_actions, v_targets)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate_fn)

    # Fresh policy with the EXACT self-play architecture (CNN + MLP heads).
    model = make_model(env, device=device)

    print(f"Behavioural cloning for {args.epochs} epochs on {device} "
          f"({len(dataset)} samples, batch {args.batch_size}, lr {args.lr}) ...")
    best_loss = train(model, loader, epochs=args.epochs, lr=args.lr)

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    model.save(MODEL_OUT)
    print(f"\nBest avg loss {best_loss:.4f} -> '{MODEL_BEST}'")
    print(f"Final (last-epoch) model     -> '{MODEL_OUT}'")


if __name__ == "__main__":
    main()
