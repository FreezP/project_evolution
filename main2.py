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
import evogym.envs  # noqa: F401
from evogym.utils import get_full_connectivity

# ──────────────────────────────────────────────────────────────────────────────
# DEFAULT HYPER-PARAMETERS
# ──────────────────────────────────────────────────────────────────────────────

DEFAULTS = dict(
    env_name="Walker-v0",
    generations=100,  # Increased for convergence
    pop_size=50,  # Population size (N)
    sigma=0.02,  # Noise standard deviation (lower is better for NN)
    lr=0.01,  # Learning rate for Adam
    max_steps=500,
    h_size=64,  # Slightly larger capacity
    workers=max(1, (os.cpu_count() or 2) - 1),
    seed=42,
    log_csv="training_log.csv",
    checkpoint="checkpoint.npz",
    render=True,
    sigma_decay=0.999,  # Gentle decay
    sigma_min=0.001,
)

WALKER = np.array([
    [3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3],
    [4, 4, 0, 4, 4],
    [4, 4, 0, 4, 4],
    [4, 4, 0, 4, 4],
], dtype=int)


# ──────────────────────────────────────────────────────────────────────────────
# NEURAL NETWORK (FIXED: TANH OUTPUT)
# ──────────────────────────────────────────────────────────────────────────────

class Network(nn.Module):
    """Three-layer MLP with Tanh output to bound actions in [-1, 1]."""

    def __init__(self, n_in: int, h_size: int, n_out: int):
        super().__init__()
        self.fc1 = nn.Linear(n_in, h_size)
        self.fc2 = nn.Linear(h_size, h_size)
        self.fc3 = nn.Linear(h_size, n_out)
        self.n_out = n_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # CRITICAL FIX: Tanh bounds output to [-1, 1] for valid torques
        return torch.tanh(self.fc3(x))


# ──────────────────────────────────────────────────────────────────────────────
# AGENT
# ──────────────────────────────────────────────────────────────────────────────

class Agent:
    """Wraps a Network and exposes a flat gene vector."""

    def __init__(self, cfg: dict, genes: Optional[np.ndarray] = None):
        self.cfg = cfg
        self.fitness = None
        self.device = torch.device("cpu")
        self.model = Network(cfg["n_in"], cfg["h_size"], cfg["n_out"]).to(self.device).double()
        if genes is not None:
            self.genes = genes

    @property
    def genes(self) -> np.ndarray:
        with torch.no_grad():
            vec = torch.nn.utils.parameters_to_vector(self.model.parameters())
        return vec.cpu().double().numpy()

    @genes.setter
    def genes(self, params: np.ndarray):
        t = torch.tensor(params, device=self.device, dtype=torch.float64)
        torch.nn.utils.vector_to_parameters(t, self.model.parameters())
        self.fitness = None

    def act(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.tensor(obs).double().unsqueeze(0).to(self.device)
            return self.model(x).cpu().numpy()[0]


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def make_env(env_name: str, robot: np.ndarray) -> gym.Env:
    connections = get_full_connectivity(robot)
    env = gym.make(env_name, body=robot, connections=connections)
    env.robot = robot
    return env


def get_network_dims(env_name: str, robot: np.ndarray) -> Tuple[int, int]:
    env = make_env(env_name, robot)
    n_in = env.observation_space.shape[0]
    n_out = env.action_space.shape[0]
    env.close()
    return n_in, n_out


def evaluate(genes: np.ndarray, cfg: dict) -> float:
    """Evaluate one individual."""
    agent = Agent(cfg, genes=genes)
    env = make_env(cfg["env_name"], cfg["robot"])
    obs, _ = env.reset()
    total_reward = 0.0
    done = False
    steps = 0
    while not done and steps < cfg["max_steps"]:
        action = agent.act(obs)
        obs, r, done, trunc, _ = env.step(action)
        total_reward += r
        done = done or trunc
        steps += 1
    env.close()
    return total_reward


# ──────────────────────────────────────────────────────────────────────────────
# MAIN ES LOOP (UPGRADED: OPENAI-ES STYLE)
# ──────────────────────────────────────────────────────────────────────────────

def ES(config: dict) -> Agent:
    cfg = deepcopy(config)
    n_in, n_out = get_network_dims(cfg["env_name"], cfg["robot"])
    cfg.update(n_in=n_in, n_out=n_out)

    # Hyperparams
    N = cfg["pop_size"]
    sigma = cfg["sigma"]
    lr = cfg["lr"]
    generations = cfg["generations"]
    workers = cfg["workers"]

    # Initialize
    rng = np.random.default_rng(cfg["seed"])
    elite = Agent(cfg)
    theta = elite.genes
    d = len(theta)

    # Adam Optimizer State
    m = np.zeros(d)  # 1st moment
    v = np.zeros(d)  # 2nd moment
    adam_beta1, adam_beta2, adam_eps = 0.9, 0.999, 1e-8

    best_fitness = -np.inf
    best_genes = theta.copy()

    # Logging
    csv_file = open(cfg["log_csv"], "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["generation", "best", "mean", "std", "sigma", "elapsed_s"])

    print(f"\n  Workers : {workers}")
    print(f"  Pop Size: {N}")
    print(f"  Sigma   : {sigma:.4f}")
    print(f"  Genome  : {d} parameters")
    print(f"  Algo    : OpenAI-ES (Adam + Antithetic)\n")

    bar = tqdm(range(generations), desc="Evolving", unit="gen")
    t0 = time.perf_counter()

    for gen in bar:
        # 1. Sample Noise (Antithetic / Mirrored)
        # We generate N/2 noise vectors, and evaluate +noise and -noise
        half_n = N // 2
        noise = rng.standard_normal((half_n, d))

        # Create population: [theta + sigma*eps, theta - sigma*eps, ...]
        candidates = []
        for i in range(half_n):
            candidates.append(theta + sigma * noise[i])
            candidates.append(theta - sigma * noise[i])

        # Pad if N is odd
        if len(candidates) < N:
            candidates.append(theta + sigma * rng.standard_normal(d))

        # 2. Parallel Evaluation
        fitnesses = [None] * N
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_to_idx = {
                pool.submit(evaluate, candidates[i], cfg): i
                for i in range(N)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    fitnesses[idx] = future.result()
                except Exception as exc:
                    tqdm.write(f"  Worker {idx} raised: {exc}")
                    fitnesses[idx] = -1e9

        fitnesses = np.array(fitnesses, dtype=float)

        # 3. Fitness Shaping (Rank Normalization)
        # Maps fitnesses to a normal distribution N(0, 1) roughly.
        # Crucial for robustness against outliers.
        ranks = np.empty(N)
        ranks[np.argsort(fitnesses)] = np.linspace(-1, 1, N)  # Linear rank scaling
        ranks = (ranks - ranks.mean()) / (ranks.std() + 1e-8)

        # 4. Gradient Estimation (Black-box gradient)
        # grad = (1/N*sigma) * sum(F_i * epsilon_i)
        # We reconstruct the 'full' noise array to match the ranks
        full_noise = np.concatenate([noise, -noise], axis=0)
        if full_noise.shape[0] < N:
            # Handle the odd padding case
            last_noise = (candidates[-1] - theta) / sigma
            full_noise = np.vstack([full_noise, last_noise])

        # Weighted sum of noise
        grad = np.dot(ranks, full_noise) / (N * sigma)

        # 5. Adam Update Step
        t = gen + 1
        m = adam_beta1 * m + (1 - adam_beta1) * grad
        v = adam_beta2 * v + (1 - adam_beta2) * (grad ** 2)
        m_hat = m / (1 - adam_beta1 ** t)
        v_hat = v / (1 - adam_beta2 ** t)

        theta += lr * m_hat / (np.sqrt(v_hat) + adam_eps)

        # 6. Sigma Decay
        if sigma > cfg["sigma_min"]:
            sigma *= cfg["sigma_decay"]

        # 7. Logging & HOF
        gen_best = fitnesses.max()
        if gen_best > best_fitness:
            best_fitness = gen_best
            # In OpenAI-ES, usually theta is the best, but we can save the actual evaluated best if needed
            best_genes = candidates[fitnesses.argmax()].copy()

        elapsed = time.perf_counter() - t0
        csv_writer.writerow([gen + 1, best_fitness, fitnesses.mean(), fitnesses.std(), sigma, f"{elapsed:.1f}"])
        csv_file.flush()

        bar.set_postfix(best=f"{best_fitness:.1f}", mean=f"{fitnesses.mean():.1f}", sig=f"{sigma:.4f}")

    csv_file.close()

    hof = Agent(cfg, genes=best_genes)
    hof.fitness = best_fitness
    return hof


# ──────────────────────────────────────────────────────────────────────────────
# PLOTTING & RENDERING (Kept mostly same)
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

        plt.figure(figsize=(10, 5))
        plt.plot(gens, bests, label="Best")
        plt.plot(gens, means, label="Mean", alpha=0.5)
        plt.legend()
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("ES Training Progress")
        plt.savefig("training_curve.png")
        plt.close()
        print("  ✔  Plot saved to training_curve.png")
    except Exception as e:
        print(f"Plotting failed: {e}")


def render_gif(agent, env_name, robot, output="best.gif", max_steps=500, fps=30):
    try:
        from PIL import Image
    except ImportError:
        return

    env = gym.make(env_name, body=robot, connections=get_full_connectivity(robot), render_mode="rgb_array")
    obs, _ = env.reset()
    frames = []
    total_r = 0

    for _ in range(max_steps):
        frame = env.render()
        if frame is not None:
            frames.append(Image.fromarray(frame))

        action = agent.act(obs)
        obs, r, done, trunc, _ = env.step(action)
        total_r += r
        if done or trunc: break

    env.close()

    if frames:
        frames[0].save(output, save_all=True, append_images=frames[1:], loop=0, duration=1000 // fps)
        print(f"  ✔  GIF saved to {output} (Score: {total_r:.1f})")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--generations", type=int, default=DEFAULTS["generations"])
    p.add_argument("--pop_size", type=int, default=DEFAULTS["pop_size"])
    p.add_argument("--sigma", type=float, default=DEFAULTS["sigma"])
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--max_steps", type=int, default=DEFAULTS["max_steps"])
    p.add_argument("--workers", type=int, default=DEFAULTS["workers"])
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    p.add_argument("--log_csv", type=str, default=DEFAULTS["log_csv"])
    return p.parse_args()


def main():
    args = parse_args()

    config = {
        "env_name": "Walker-v0",
        "robot": WALKER,
        "generations": args.generations,
        "pop_size": args.pop_size,
        "sigma": args.sigma,
        "lr": args.lr,
        "max_steps": args.max_steps,
        "h_size": DEFAULTS["h_size"],
        "workers": args.workers,
        "seed": args.seed,
        "log_csv": args.log_csv,
        "sigma_decay": DEFAULTS["sigma_decay"],
        "sigma_min": DEFAULTS["sigma_min"],
    }

    best_agent = ES(config)
    print(f"\n  ✔  Training finished. Best Fitness: {best_agent.fitness:.2f}")

    plot_log(config["log_csv"])
    render_gif(best_agent, config["env_name"], config["robot"])


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()