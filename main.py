from __future__ import annotations

import argparse
import csv
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Suppress gym deprecation noise
warnings.filterwarnings("ignore", category=DeprecationWarning)

import gymnasium as gym
import evogym.envs  # noqa: F401 — registers Walker-v0
from evogym.utils import get_full_connectivity

# ──────────────────────────────────────────────────────────────────────────────
# DEFAULT HYPER-PARAMETERS
# ──────────────────────────────────────────────────────────────────────────────

DEFAULTS = dict(
    env_name    = "Walker-v0",
    generations = 50,
    lam         = 40,       # λ  — offspring population size
    mu          = 10,       # μ  — number of parents used for the update
    sigma       = 0.1,      # initial mutation std-dev
    lr          = 1.0,      # ES learning rate (step multiplier)
    max_steps   = 500,      # episode length
    h_size      = 32,       # hidden units per layer
    workers     = max(1, (os.cpu_count() or 2) - 1),  # parallel workers
    sigma_adapt = True,     # enable adaptive σ
    sigma_tau   = 0.95,     # σ decay when success_rate < 0.2
    sigma_tau_p = 1.05,     # σ growth when success_rate > 0.2
    seed        = 42,
    log_csv     = "training_log.csv",
    checkpoint  = "checkpoint.npz",
    render      = True,
)

# ──────────────────────────────────────────────────────────────────────────────
# ROBOT MORPHOLOGY
# ──────────────────────────────────────────────────────────────────────────────

WALKER = np.array([
    [3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3],
    [4, 4, 0, 4, 4],
    [4, 4, 0, 4, 4],
    [4, 4, 0, 4, 4],
], dtype=int)


# ──────────────────────────────────────────────────────────────────────────────
# NEURAL NETWORK
# ──────────────────────────────────────────────────────────────────────────────

class Network(nn.Module):
    """Three-layer MLP with ReLU activations."""

    def __init__(self, n_in: int, h_size: int, n_out: int):
        super().__init__()
        self.fc1 = nn.Linear(n_in, h_size)
        self.fc2 = nn.Linear(h_size, h_size)
        self.fc3 = nn.Linear(h_size, n_out)
        self.n_out = n_out

    def reset(self):
        pass  # kept for API compatibility

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ──────────────────────────────────────────────────────────────────────────────
# AGENT
# ──────────────────────────────────────────────────────────────────────────────

class Agent:
    """Wraps a Network and exposes a flat gene vector for ES."""

    def __init__(self, cfg: dict, genes: Optional[np.ndarray] = None):
        self.cfg     = cfg
        self.fitness: Optional[float] = None
        self.device  = torch.device("cpu")  # workers always run on CPU
        self.model   = Network(cfg["n_in"], cfg["h_size"], cfg["n_out"]) \
                           .to(self.device).double()
        if genes is not None:
            self.genes = genes

    # ── gene vector property ──────────────────────────────────────────────

    @property
    def genes(self) -> np.ndarray:
        with torch.no_grad():
            vec = torch.nn.utils.parameters_to_vector(self.model.parameters())
        return vec.cpu().double().numpy()

    @genes.setter
    def genes(self, params: np.ndarray):
        if np.isnan(params).any():
            raise ValueError("NaN detected in gene vector.")
        t = torch.tensor(params, device=self.device, dtype=torch.float64)
        torch.nn.utils.vector_to_parameters(t, self.model.parameters())
        self.fitness = None

    # ── action ────────────────────────────────────────────────────────────

    def act(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.tensor(obs).double().unsqueeze(0).to(self.device)
            return self.model(x).cpu().numpy()


# ──────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def make_env(env_name: str, robot: np.ndarray) -> gym.Env:
    connections = get_full_connectivity(robot)
    env = gym.make(env_name, body=robot, connections=connections)
    env.robot = robot
    return env


def get_network_dims(env_name: str, robot: np.ndarray) -> Tuple[int, int]:
    env = make_env(env_name, robot)
    n_in  = env.observation_space.shape[0]
    n_out = env.action_space.shape[0]
    env.close()
    return n_in, n_out


def evaluate(
    agent: Agent,
    env_name: str,
    robot: np.ndarray,
    max_steps: int = 500,
    render: bool = False,
) -> float:
    """Run one episode and return total reward."""
    env = make_env(env_name, robot)
    obs, _ = env.reset()
    agent.model.reset()
    total_reward = 0.0
    done = False
    steps = 0
    while not done and steps < max_steps:
        if render:
            env.render()
        action = agent.act(obs)
        obs, r, done, trunc, _ = env.step(action)
        total_reward += r
        done = done or trunc
        steps += 1
    env.close()
    return total_reward


# ──────────────────────────────────────────────────────────────────────────────
# WORKER FUNCTION  (must be top-level for multiprocessing pickling)
# ──────────────────────────────────────────────────────────────────────────────

def _worker_eval(
    genes: np.ndarray,
    cfg: dict,
) -> float:
    """Evaluate one individual in a subprocess."""
    agent = Agent(cfg, genes=genes)
    return evaluate(agent, cfg["env_name"], cfg["robot"], cfg["max_steps"])


# ──────────────────────────────────────────────────────────────────────────────
# RANK-BASED FITNESS SHAPING
# ──────────────────────────────────────────────────────────────────────────────

def rank_fitness(fitnesses: List[float]) -> np.ndarray:
    """
    Map raw fitness values to normalised rank-based weights ∈ (0, 1].
    The best individual gets weight 1, the worst gets weight 1/n.
    This removes the effect of fitness scale and outliers.
    """
    n   = len(fitnesses)
    idx = np.argsort(fitnesses)[::-1]   # best → worst
    w   = np.empty(n)
    for rank, i in enumerate(idx):
        w[i] = (n - rank) / n
    w /= w.sum()
    return w


# ──────────────────────────────────────────────────────────────────────────────
# CMA-LIKE WEIGHTED RECOMBINATION WEIGHTS  (used alongside rank shaping)
# ──────────────────────────────────────────────────────────────────────────────

def _recombination_weights(mu: int) -> np.ndarray:
    w = np.array([np.log(mu + 0.5) - np.log(i) for i in range(1, mu + 1)])
    w /= w.sum()
    return w


# ──────────────────────────────────────────────────────────────────────────────
# MAIN ES LOOP
# ──────────────────────────────────────────────────────────────────────────────

def ES(config: dict) -> Agent:
    """
    (μ/μ_w, λ)-ES with:
      - parallel evaluation via ProcessPoolExecutor
      - adaptive σ
      - rank-based fitness shaping
      - hall-of-fame (all-time best)
      - CSV logging
    """
    cfg = deepcopy(config)
    n_in, n_out = get_network_dims(cfg["env_name"], cfg["robot"])
    cfg.update(n_in=n_in, n_out=n_out)

    lam          = cfg["lam"]
    mu           = cfg["mu"]
    sigma        = cfg["sigma"]
    lr           = cfg["lr"]
    workers      = cfg["workers"]
    generations  = cfg["generations"]
    sigma_adapt  = cfg["sigma_adapt"]
    sigma_tau    = cfg["sigma_tau"]
    sigma_tau_p  = cfg["sigma_tau_p"]
    log_csv_path = cfg.get("log_csv", "training_log.csv")
    ckpt_path    = cfg.get("checkpoint", "checkpoint.npz")

    rec_w = _recombination_weights(mu)   # CMA-like recombination weights

    # ── initialise ────────────────────────────────────────────────────────
    rng = np.random.default_rng(cfg["seed"])

    # Load checkpoint if present
    if Path(ckpt_path).exists() and cfg.get("_resume", False):
        data  = np.load(ckpt_path, allow_pickle=True)
        theta = data["theta"]
        best_fitness = float(data["best_fitness"])
        best_genes   = data["best_genes"]
        start_gen    = int(data.get("generation", 0))
        sigma        = float(data.get("sigma", sigma))
        print(f"\n  ✔  Resumed from {ckpt_path}  "
              f"(gen={start_gen}, best={best_fitness:.3f})\n")
    else:
        elite   = Agent(cfg)
        theta   = elite.genes
        best_fitness = -np.inf
        best_genes   = theta.copy()
        start_gen    = 0

    d = len(theta)

    # ── CSV logger ────────────────────────────────────────────────────────
    csv_file = open(log_csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        ["generation", "best", "mean", "std", "sigma", "elapsed_s"]
    )

    print(f"\n  Workers : {workers}")
    print(f"  λ={lam}  μ={mu}  σ={sigma:.4f}  lr={lr}")
    print(f"  Genome  : {d} parameters")
    print(f"  Log     : {log_csv_path}")
    print(f"  Ckpt    : {ckpt_path}\n")

    bar = tqdm(
        range(start_gen, start_gen + generations),
        desc="Evolving",
        unit="gen",
        dynamic_ncols=True,
    )

    t0 = time.perf_counter()

    for gen in bar:
        # ── sample offspring ──────────────────────────────────────────────
        noise      = rng.standard_normal((lam, d))
        candidates = [theta + sigma * noise[i] for i in range(lam)]

        # ── parallel evaluation ───────────────────────────────────────────
        fitnesses = [None] * lam
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_to_idx = {
                pool.submit(_worker_eval, candidates[i], cfg): i
                for i in range(lam)
            }
            for future in as_completed(future_to_idx):
                i = future_to_idx[future]
                try:
                    fitnesses[i] = future.result()
                except Exception as exc:
                    tqdm.write(f"  Worker {i} raised: {exc}")
                    fitnesses[i] = -1e9

        fitnesses = np.array(fitnesses, dtype=float)

        # ── fitness shaping + update ──────────────────────────────────────
        shaped_w   = rank_fitness(fitnesses.tolist())  # per-individual
        sorted_idx = np.argsort(fitnesses)[::-1]       # best → worst

        step = np.zeros(d)
        for k in range(mu):
            i     = sorted_idx[k]
            step += rec_w[k] * noise[i]
        theta += lr * sigma * step

        # ── adaptive σ ────────────────────────────────────────────────────
        if sigma_adapt:
            success_rate = np.mean(fitnesses > best_fitness)
            sigma *= sigma_tau_p if success_rate > 0.2 else sigma_tau
            sigma  = float(np.clip(sigma, 1e-4, 10.0))

        # ── hall of fame ──────────────────────────────────────────────────
        gen_best_idx = sorted_idx[0]
        if fitnesses[gen_best_idx] > best_fitness:
            best_fitness = float(fitnesses[gen_best_idx])
            best_genes   = candidates[gen_best_idx].copy()

        # ── logging ───────────────────────────────────────────────────────
        elapsed = time.perf_counter() - t0
        csv_writer.writerow([
            gen + 1,
            f"{best_fitness:.4f}",
            f"{fitnesses.mean():.4f}",
            f"{fitnesses.std():.4f}",
            f"{sigma:.6f}",
            f"{elapsed:.1f}",
        ])
        csv_file.flush()

        bar.set_postfix(
            best  = f"{best_fitness:.2f}",
            mean  = f"{fitnesses.mean():.2f}",
            sigma = f"{sigma:.4f}",
        )

        # ── checkpoint ────────────────────────────────────────────────────
        np.savez(
            ckpt_path,
            theta        = theta,
            best_genes   = best_genes,
            best_fitness = np.array([best_fitness]),
            generation   = np.array([gen + 1]),
            sigma        = np.array([sigma]),
        )

    csv_file.close()

    # ── return hall-of-fame agent ─────────────────────────────────────────
    hof = Agent(cfg, genes=best_genes)
    hof.fitness = best_fitness
    return hof


# ──────────────────────────────────────────────────────────────────────────────
# PLOT TRAINING CURVE
# ──────────────────────────────────────────────────────────────────────────────

def plot_log(csv_path: str):
    try:
        import matplotlib.pyplot as plt

        gens, bests, means = [], [], []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                gens.append(int(row["generation"]))
                bests.append(float(row["best"]))
                means.append(float(row["mean"]))

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(gens, bests, label="Best (hall-of-fame)", linewidth=2)
        ax.plot(gens, means, label="Mean fitness",
                linewidth=1.5, linestyle="--", alpha=0.7)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness (cumulative reward)")
        ax.set_title("Walker-v0 — ES Training Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("training_curve.png", dpi=150)
        plt.show()
        print("  ✔  Curve saved → training_curve.png")
    except ImportError:
        print("  matplotlib not available — skipping plot.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EvoGym Walker — Multithreaded ES",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--generations", type=int,  default=DEFAULTS["generations"])
    p.add_argument("--lam",         type=int,  default=DEFAULTS["lam"],
                   help="Offspring population size (λ)")
    p.add_argument("--mu",          type=int,  default=DEFAULTS["mu"],
                   help="Parent count (μ)")
    p.add_argument("--sigma",       type=float,default=DEFAULTS["sigma"])
    p.add_argument("--lr",          type=float,default=DEFAULTS["lr"])
    p.add_argument("--max-steps",   type=int,  default=DEFAULTS["max_steps"])
    p.add_argument("--h-size",      type=int,  default=DEFAULTS["h_size"])
    p.add_argument("--workers",     type=int,  default=DEFAULTS["workers"],
                   help="Parallel worker processes (default: CPU count − 1)")
    p.add_argument("--no-sigma-adapt", action="store_true",
                   help="Disable adaptive σ schedule")
    p.add_argument("--seed",        type=int,  default=DEFAULTS["seed"])
    p.add_argument("--log-csv",     type=str,  default=DEFAULTS["log_csv"])
    p.add_argument("--checkpoint",  type=str,  default=DEFAULTS["checkpoint"])
    p.add_argument("--load",        action="store_true",
                   help="Resume from --checkpoint if it exists")
    p.add_argument("--no-render",   action="store_true",
                   help="Skip final visual playback")
    p.add_argument("--no-plot",     action="store_true",
                   help="Skip training curve plot")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    config = {
        "env_name"    : "Walker-v0",
        "robot"       : WALKER,
        "generations" : args.generations,
        "lam"         : args.lam,
        "mu"          : args.mu,
        "sigma"       : args.sigma,
        "lr"          : args.lr,
        "max_steps"   : args.max_steps,
        "h_size"      : args.h_size,
        "workers"     : args.workers,
        "sigma_adapt" : not args.no_sigma_adapt,
        "sigma_tau"   : DEFAULTS["sigma_tau"],
        "sigma_tau_p" : DEFAULTS["sigma_tau_p"],
        "seed"        : args.seed,
        "log_csv"     : args.log_csv,
        "checkpoint"  : args.checkpoint,
        "_resume"     : args.load,
    }

    t_start = time.perf_counter()
    best_agent = ES(config)
    elapsed    = time.perf_counter() - t_start

    print(f"\n  ✔  Training done in {elapsed:.1f}s")
    print(f"  ✔  Hall-of-fame fitness : {best_agent.fitness:.3f}")

    # ── plot ──────────────────────────────────────────────────────────────
    if not args.no_plot:
        plot_log(args.log_csv)

    # ── render ────────────────────────────────────────────────────────────
    if not args.no_render:
        print("\n  Rendering best walker … (close window to exit)")
        score = evaluate(
            best_agent,
            config["env_name"],
            config["robot"],
            max_steps=config["max_steps"],
            render=True,
        )
        print(f"  Rendered episode reward : {score:.3f}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()