"""AlphaZero training loop for Carcassonne.

Each iteration plays ``games_per_iter`` self-play games with MCTS on both sides
(via mcts.selfplay_game), then fits the policy head with cross-entropy to the
MCTS visit distribution and the value head with MSE to the game outcome
(+1/-1/0). The improved network becomes the evaluator for the next iteration.
This is supervised training on MCTS targets, so unlike train.py there is no
VecNormalize or environment reward.

Usage:
    # bootstrap from the behavioural-cloning net (focused policy prior):
    python alphazero.py --init-model models/model_bc_best.zip --iterations 20

    # or from scratch:
    python alphazero.py --iterations 20
"""

from __future__ import annotations

import argparse
import os
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset

from mcts import NeuralEvaluator, selfplay_game
from train import BOARD_SIZE, make_model, resolve_device

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

MODELS_DIR = "models"

# Defaults (small-ish; scale games_per_iter / n_sims up with compute).
ITERATIONS = 20
GAMES_PER_ITER = 30
N_SIMS = 160
EPOCHS = 4
LR = 3e-4
BATCH_SIZE = 64
BUFFER_GAMES = 120          # sliding replay window (most-recent games)


# --------------------------------------------------------------------------- #
# 1. Self-play data generation
# --------------------------------------------------------------------------- #
def generate_selfplay_data(model, n_games, n_sims, rng, device=None):
    """Play ``n_games`` self-play games and return a flat list of training
    tuples ``(obs, visit_policy, value)``."""
    evaluator = NeuralEvaluator(model, device=device)
    data = []
    games = range(n_games)
    if tqdm is not None:
        games = tqdm(games, desc="self-play", unit="game", leave=False)
    for _ in games:
        data.extend(selfplay_game(evaluator, n_sims, board_size=BOARD_SIZE, rng=rng))
    return data


# --------------------------------------------------------------------------- #
# Parallel self-play: many worker processes, CPU inference (frees the GPU and
# uses all cores). Self-play is CPU-bound -- this is the main speedup.
# --------------------------------------------------------------------------- #
_WORKER = {}   # per-process state, populated by _worker_init


def _worker_init(model_path, n_sims, mcts_batch):
    import torch
    torch.set_num_threads(1)        # avoid N workers x M threads oversubscription
    from sb3_contrib import MaskablePPO
    from train import build_single_env
    model = MaskablePPO.load(model_path, env=build_single_env(seed=0), device="cpu")
    _WORKER.update(ev=NeuralEvaluator(model, device="cpu"),
                   n_sims=n_sims, mcts_batch=mcts_batch)


def _worker_play(seed):
    rng = np.random.default_rng(seed)
    return selfplay_game(_WORKER["ev"], _WORKER["n_sims"], board_size=BOARD_SIZE,
                         rng=rng, return_info=True, batch_size=_WORKER["mcts_batch"])


def generate_selfplay_parallel(model, n_games, n_sims, n_workers, mcts_batch, seed):
    """Generate ``n_games`` self-play games across ``n_workers`` processes.

    Returns a list of ``(game_records, info)``. Workers load the current net from
    a temp checkpoint and run inference on CPU, so they parallelise across cores
    without contending for the GPU.
    """
    import multiprocessing as mp
    tmp = os.path.join(MODELS_DIR, "_az_selfplay_tmp.zip")
    model.save(tmp)                                   # workers load the current net
    seeds = [int(seed) + i for i in range(n_games)]
    ctx = mp.get_context("spawn")
    results = []
    with ctx.Pool(n_workers, initializer=_worker_init,
                  initargs=(tmp, n_sims, mcts_batch)) as pool:
        stream = pool.imap_unordered(_worker_play, seeds)
        if tqdm is not None:
            stream = tqdm(stream, total=n_games, desc="self-play", unit="game")
        for r in stream:
            results.append(r)
    return results


# --------------------------------------------------------------------------- #
# 2. Dataset + collate
# --------------------------------------------------------------------------- #
class AlphaZeroDataset(Dataset):
    def __init__(self, records):
        self.records = list(records)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        return self.records[i]      # (obs-dict, visit_policy-dict, value-float)


def collate_fn(batch):
    """Stack a batch into (obs tensors, action mask, soft policy target, value)."""
    obs_keys = batch[0][0].keys()
    obs = {k: torch.as_tensor(np.stack([b[0][k] for b in batch])) for k in obs_keys}

    n_actions = batch[0][0]["action_mask"].shape[0]
    target = torch.zeros((len(batch), n_actions), dtype=torch.float32)
    for i, (_o, pi, _v) in enumerate(batch):
        for a, p in pi.items():
            target[i, int(a)] = p

    mask = torch.as_tensor(np.stack([b[0]["action_mask"] for b in batch])).bool()
    value = torch.as_tensor([b[2] for b in batch], dtype=torch.float32)
    return obs, mask, target, value


# --------------------------------------------------------------------------- #
# 3. Network training on the MCTS targets
# --------------------------------------------------------------------------- #
def train_on_buffer(model, records, epochs, lr, batch_size, device):
    policy = model.policy
    policy.set_training_mode(True)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    loader = DataLoader(AlphaZeroDataset(records), batch_size=batch_size,
                        shuffle=True, collate_fn=collate_fn)

    last = (0.0, 0.0)
    for epoch in range(epochs):
        run_pi = run_v = count = 0.0
        it = loader
        if tqdm is not None:
            it = tqdm(loader, desc=f"train {epoch + 1}/{epochs}", unit="batch", leave=False)
        for obs, mask, target, value in it:
            obs = {k: v.to(device) for k, v in obs.items()}
            mask = mask.to(device)
            target = target.to(device)
            value = value.to(device)

            features = policy.extract_features(obs)
            latent_pi, _ = policy.mlp_extractor(features)
            logits = policy.action_net(latent_pi).masked_fill(~mask, -1e9)
            logp = F.log_softmax(logits, dim=1)
            pi_loss = -(target * logp).sum(dim=1).mean()   # soft cross-entropy

            values = policy.predict_values(obs).flatten()
            value_loss = F.mse_loss(values, value)

            loss = pi_loss + value_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n = value.size(0)
            run_pi += pi_loss.item() * n
            run_v += value_loss.item() * n
            count += n
        last = (run_pi / max(count, 1), run_v / max(count, 1))
        print(f"  epoch {epoch + 1}/{epochs}  pi_loss {last[0]:.4f}  v_loss {last[1]:.4f}")
    return last


# --------------------------------------------------------------------------- #
# 4. The AlphaZero loop
# --------------------------------------------------------------------------- #
def alphazero_loop(model, iterations, games_per_iter, n_sims, epochs, lr,
                   batch_size, device, rng, buffer_games=BUFFER_GAMES,
                   mcts_batch=1, n_workers=1, save_prefix="model_az",
                   tb_log_dir=os.path.join("tb_logs", "alphazero")):
    os.makedirs(MODELS_DIR, exist_ok=True)
    buffer = deque(maxlen=buffer_games)     # sliding window of per-game records

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=tb_log_dir)
    print(f"TensorBoard logging to {tb_log_dir}  (run:  tensorboard --logdir tb_logs)")

    for it in range(1, iterations + 1):
        print(f"\n===== AlphaZero iteration {it}/{iterations} "
              f"({games_per_iter} games x {n_sims} sims, {n_workers} workers, "
              f"device={device}) =====")

        # Self-play with the current net. Self-play is CPU-bound, so run it across
        # worker processes when n_workers > 1 (the big speedup on a multi-core CPU).
        t_sp = time.perf_counter()
        if n_workers > 1:
            games_info = generate_selfplay_parallel(
                model, games_per_iter, n_sims, n_workers, mcts_batch,
                seed=int(rng.integers(1 << 30)))
        else:
            evaluator = NeuralEvaluator(model, device=device)
            games_info = []
            bar = (tqdm(range(games_per_iter), desc=f"iter {it} self-play", unit="game")
                   if tqdm else range(games_per_iter))
            for _g in bar:
                games_info.append(selfplay_game(
                    evaluator, n_sims, board_size=BOARD_SIZE, rng=rng,
                    return_info=True, batch_size=mcts_batch))

        plies, abs_margins, draws = [], [], 0
        for game, info in games_info:
            buffer.append(game)
            plies.append(info["plies"])
            abs_margins.append(abs(info["margin"]))
            draws += int(info["margin"] == 0)
        records = [t for game in buffer for t in game]
        print(f"  self-play {time.perf_counter() - t_sp:.0f}s  "
              f"({(time.perf_counter() - t_sp) / games_per_iter:.1f}s/game)")
        print(f"  buffer: {len(buffer)} games / {len(records)} positions  "
              f"| avg len {np.mean(plies):.0f}  avg |margin| {np.mean(abs_margins):.1f}  "
              f"draws {draws}/{games_per_iter}")

        # 2. train on the (search-improved) targets
        pi_loss, v_loss = train_on_buffer(model, records, epochs, lr, batch_size, device)

        # 3. log + checkpoint
        writer.add_scalar("loss/policy", pi_loss, it)
        writer.add_scalar("loss/value", v_loss, it)
        writer.add_scalar("selfplay/avg_game_len", float(np.mean(plies)), it)
        writer.add_scalar("selfplay/avg_abs_margin", float(np.mean(abs_margins)), it)
        writer.add_scalar("selfplay/draw_rate", draws / games_per_iter, it)
        writer.add_scalar("buffer/positions", len(records), it)
        writer.flush()

        path = os.path.join(MODELS_DIR, f"{save_prefix}_iter{it}")
        model.save(path)
        print(f"  saved {path}.zip")

    writer.close()
    print("\nAlphaZero training complete.")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="AlphaZero training for Carcassonne")
    p.add_argument("--iterations", type=int, default=ITERATIONS)
    p.add_argument("--games-per-iter", type=int, default=GAMES_PER_ITER)
    p.add_argument("--n-sims", type=int, default=N_SIMS)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--buffer-games", type=int, default=BUFFER_GAMES)
    p.add_argument("--mcts-batch", type=int, default=8,
                   help="leaves per batched network eval during self-play "
                        "(>1 amortises GPU latency; speed/accuracy trade-off)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                   help="parallel self-play worker processes (CPU inference); "
                        "self-play is CPU-bound, so this is the main speedup")
    p.add_argument("--init-model", type=str, default=None,
                   help="warm-start the network from this .zip (e.g. the BC net)")
    p.add_argument("--device", type=str, default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = resolve_device(args.device)
    rng = np.random.default_rng(args.seed)

    # The network lives inside a MaskablePPO container (same architecture as
    # everywhere else); we train its policy/value heads directly on MCTS targets.
    env = make_model_env()
    if args.init_model:
        from sb3_contrib import MaskablePPO
        print(f"Warm-starting network from {args.init_model}")
        model = MaskablePPO.load(args.init_model, env=env, device=device)
    else:
        model = make_model(env, device=device)

    alphazero_loop(model, args.iterations, args.games_per_iter, args.n_sims,
                   args.epochs, args.lr, args.batch_size, device, rng,
                   buffer_games=args.buffer_games, mcts_batch=args.mcts_batch,
                   n_workers=args.workers)


def make_model_env():
    """A single env just to define the observation/action spaces for the net."""
    from train import build_single_env
    return build_single_env(seed=0)


if __name__ == "__main__":
    main()
