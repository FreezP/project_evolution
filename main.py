"""
Evolution Strategies on Evogym environments.
Optimizes a neural network controller for a given robot morphology.
"""
from evogym import sample_robot
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib import animation
import gymnasium as gym
import evogym.envs
from evogym.utils import get_full_connectivity
from tqdm import tqdm
import imageio
import multiprocessing as mp


class Network(nn.Module):
    """Feedforward network with optional LayerNorm, Xavier init."""
    def __init__(self, n_in, h_size, n_out, use_layernorm=True):
        super().__init__()
        self.use_layernorm = use_layernorm
        self.fc1 = nn.Linear(n_in, h_size)
        self.fc2 = nn.Linear(h_size, h_size)
        self.fc3 = nn.Linear(h_size, n_out)
        if use_layernorm:
            self.ln1 = nn.LayerNorm(h_size)
            self.ln2 = nn.LayerNorm(h_size)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def reset(self):
        pass

    def forward(self, x):
        x = self.fc1(x)
        if self.use_layernorm:
            x = self.ln1(x)
        x = F.relu(x)
        x = self.fc2(x)
        if self.use_layernorm:
            x = self.ln2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class Agent:
    """Encapsulates a neural network controller, its parameters, and fitness."""
    def __init__(self, Net, config, genes=None):
        self.config = config
        self.Net = Net
        self.model = None
        self.fitness = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.make_network()
        if genes is not None:
            self.genes = genes

    def __repr__(self):
        return f"Agent {self.model} > fitness={self.fitness}"

    def __str__(self):
        return self.__repr__()

    def make_network(self):
        n_in = self.config["n_in"]
        h_size = self.config["h_size"]
        n_out = self.config["n_out"]
        use_ln = self.config.get("use_layernorm", True)
        self.model = self.Net(n_in, h_size, n_out, use_layernorm=use_ln).to(self.device).float()
        return self

    @property
    def genes(self):
        if self.model is None:
            return None
        with torch.no_grad():
            params = self.model.parameters()
            vec = torch.nn.utils.parameters_to_vector(params)
        return vec.cpu().float().numpy()

    @genes.setter
    def genes(self, params):
        if self.model is None:
            self.make_network()
        params = np.clip(params, -10.0, 10.0)
        a = torch.tensor(params, device=self.device)
        torch.nn.utils.vector_to_parameters(a, self.model.parameters())
        self.model = self.model.to(self.device).float()
        self.fitness = None
        return self

    def act(self, obs):
        with torch.no_grad():
            x = torch.tensor(obs).float().unsqueeze(0).to(self.device)
            actions = self.model(x).cpu().detach().numpy()
        return actions


robot_config = np.array([
    [1, 3, 3, 3, 1],
    [4, 4, 1, 4, 4],
    [0, 4, 2, 4, 0],
    [4, 4, 2, 4, 4],
    [4, 3, 3, 3, 4]
])

'Walker:'
''' np.array([
    [3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3],
    [3, 3, 0, 3, 3],
    [3, 3, 0, 3, 3],
    [1, 1, 0, 1, 1]
])'''

'Thrower:'
''' np.array([
    [0, 0, 1, 1, 1],
    [1, 0, 4, 4, 4],
    [4, 0, 4, 4, 4],
    [4, 0, 4, 4, 4],
    [3, 3, 3, 1, 1]
])'''


def make_env(env_name, seed=None, robot=None, **kwargs):
    """Create an Evolution Gym environment, optionally with a custom robot body."""
    if robot is None:
        env = gym.make(env_name, **kwargs)
    else:
        env = gym.make(env_name, body=robot, **kwargs)
    env.robot = robot
    if seed is not None:
        env.reset(seed=seed)
    return env


def evaluate(agent, env, max_steps=500, render=False):
    """Run one episode and return total reward (and optionally frames for a GIF)."""
    obs, _ = env.reset()
    agent.model.reset()
    total_reward = 0
    steps = 0
    done = False
    imgs = [] if render else None
    while not done and steps < max_steps:
        if render:
            img = env.render()
            imgs.append(img)
        action = agent.act(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
    if render:
        return total_reward, imgs
    return total_reward


def get_cfg(env_name, robot=None):
    """Derive controller configuration from environment observation/action spaces."""
    env = make_env(env_name, robot=robot_config)
    cfg = {
        "n_in": env.observation_space.shape[0],
        "h_size": 32,
        "n_out": env.action_space.shape[0],
    }
    env.close()
    return cfg


def mp_eval(agent_genes, cfg, device_str):
    """Worker function for parallel evaluation of one agent."""
    if device_str == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(0)
    agent = Agent(Network, cfg, genes=agent_genes)
    env = make_env(cfg["env_name"], robot=cfg["robot"])
    fitness = evaluate(agent, env, max_steps=cfg["max_steps"])
    env.close()
    return fitness


def ES(config):
    """Run (μ,λ)-Evolution Strategies with momentum and adaptive sigma."""
    cfg = get_cfg(config["env_name"], robot=config["robot"])
    cfg = {**config, **cfg}
    cfg.setdefault("use_layernorm", True)

    mu = cfg["mu"]
    lambda_ = cfg["lambda"]
    sigma = cfg["sigma"]
    lr = cfg.get("lr", 1.0)
    momentum = 0.9
    sigma_lr = 0.01
    target_success_rate = 0.1
    param_clip = 5.0

    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w /= np.sum(w)

    elite = Agent(Network, cfg)
    elite.fitness = -np.inf
    theta = elite.genes
    d = len(theta)
    velocity = np.zeros(d)

    fits = []
    total_evals = []
    sigma_history = []

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = mp.cpu_count() - 1
    pool = mp.Pool(processes=num_workers)

    bar = tqdm(range(cfg["generations"]))
    for gen in bar:
        half_lambda = lambda_ // 2
        population_genes = []
        perturbations = []
        for _ in range(half_lambda):
            eps = np.random.randn(d)
            perturbations.append(eps)
            population_genes.append(theta + sigma * eps)
            population_genes.append(theta - sigma * eps)
        if lambda_ % 2 == 1:
            population_genes.append(theta + sigma * np.random.randn(d))

        args = [(genes, cfg, device_str) for genes in population_genes]
        pop_fitness = pool.starmap(mp_eval, args)

        best_idx = np.argmax(pop_fitness)
        if pop_fitness[best_idx] > elite.fitness:
            elite.genes = population_genes[best_idx]
            elite.fitness = pop_fitness[best_idx]

        ranks = np.argsort(np.argsort(pop_fitness)[::-1]) + 1
        shaped_fitness = 1.0 / ranks

        sorted_indices = np.argsort(pop_fitness)[::-1]
        step = np.zeros(d)
        for i in range(mu):
            idx = sorted_indices[i]
            if idx < 2 * half_lambda:
                pair_idx = idx // 2
                eps = perturbations[pair_idx]
                direction = eps if idx % 2 == 0 else -eps
            else:
                direction = (population_genes[idx] - theta) / sigma
            step += w[i] * direction * shaped_fitness[idx]
        step /= np.sum(w[:mu] * shaped_fitness[sorted_indices[:mu]])

        velocity = momentum * velocity + lr * step
        velocity = np.clip(velocity, -param_clip, param_clip)
        theta = theta + sigma * velocity

        baseline = elite.fitness
        success_count = sum(f > baseline for f in pop_fitness)
        success_rate = success_count / lambda_

        if success_rate > target_success_rate:
            sigma *= np.exp(sigma_lr)
        else:
            sigma /= np.exp(sigma_lr)
        sigma = np.clip(sigma, 0.01, 1.0)

        fits.append(elite.fitness)
        total_evals.append(lambda_ * (gen + 1))
        sigma_history.append(sigma)
        bar.set_description(f"Best: {elite.fitness:.2f} | σ: {sigma:.3f} | succ: {success_rate:.2f}")

    pool.close()
    pool.join()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    ax1.plot(total_evals, fits)
    ax1.set_xlabel("Evaluations")
    ax1.set_ylabel("Fitness")
    ax2.plot(total_evals, sigma_history)
    ax2.set_xlabel("Evaluations")
    ax2.set_ylabel("Mutation σ")
    plt.tight_layout()
    plt.show()

    return elite


if __name__ == "__main__":
    config = {
        "env_name": "Climber-v2",
        "robot": robot_config,
        "generations": 100,
        "lambda": 100,
        "mu": 50,
        "sigma": 0.4,
        "lr": 0.1,
        "max_steps": 500,
        "use_layernorm": False,
    }

    elite_agent = ES(config)

    env = make_env(config["env_name"], robot=config["robot"], render_mode="rgb_array")
    env.metadata['render_fps'] = 30
    fitness, imgs = evaluate(elite_agent, env, max_steps=500, render=True)
    env.close()
    print(f"Final fitness: {fitness}")
    imageio.mimsave('Thrower.gif', imgs, duration=(1/30.0))