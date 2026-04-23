import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
import evogym.envs
from evogym.utils import get_full_connectivity
import imageio
import matplotlib.pyplot as plt
from tqdm import tqdm

# ----------------------------------------------------------------------
# 1. Reused network backbone (unchanged)
# ----------------------------------------------------------------------
class Network(nn.Module):
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

# ----------------------------------------------------------------------
# 2. Actor-Critic network for PPO (continuous actions)
# ----------------------------------------------------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=128, use_layernorm=True):
        super().__init__()
        self.base = Network(obs_dim, hidden_size, hidden_size, use_layernorm)  # shared trunk
        self.actor = nn.Linear(hidden_size, act_dim)
        self.critic = nn.Linear(hidden_size, 1)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        features = self.base(x)
        action_mean = self.actor(features)
        value = self.critic(features)
        return action_mean, value

    def act(self, obs):
        """Return action and log probability given a numpy observation (batched)."""
        obs_t = torch.FloatTensor(obs)
        with torch.no_grad():
            mean, value = self.forward(obs_t)
            std = self.log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        return action.numpy(), log_prob.numpy(), value.squeeze(-1).numpy()

    def evaluate_actions(self, obs, actions):
        """Evaluate log probs, values, entropy for a batch."""
        mean, value = self.forward(obs)
        std = self.log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_probs, value.squeeze(-1), entropy

# ----------------------------------------------------------------------
# 3. Environment helpers
# ----------------------------------------------------------------------
robot_config = np.array([
    [1, 3, 3, 3, 1],
    [4, 4, 1, 4, 4],
    [0, 4, 2, 4, 0],
    [4, 4, 2, 4, 4],
    [4, 3, 3, 3, 4]
])

def make_env(env_name, seed=None, robot=None, **kwargs):
    """Create a single environment."""
    if robot is None:
        env = gym.make(env_name, **kwargs)
    else:
        env = gym.make(env_name, body=robot, **kwargs)
    if seed is not None:
        env.reset(seed=seed)
    return env

def evaluate(agent, env, max_steps=500, render=False):
    """Run one episode, optionally collect frames."""
    obs, _ = env.reset()
    if hasattr(agent.base, 'reset'): agent.base.reset()
    total_reward = 0
    steps = 0
    done = False
    imgs = []
    while not done and steps < max_steps:
        if render:
            imgs.append(env.render())
        action, _, _ = agent.act(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
    if render:
        return total_reward, imgs
    return total_reward

# ----------------------------------------------------------------------
# 4. PPO Trainer with multithreaded environment sampling
# ----------------------------------------------------------------------
class PPO:
    def __init__(self, env_name, robot, num_envs=4, hidden_size=128, lr=3e-4,
                 gamma=0.99, lam=0.95, clip_epsilon=0.2, epochs=10,
                 batch_size=64, n_steps=512, max_episode_steps=500,
                 use_layernorm=True):
        self.env_name = env_name
        self.robot = robot
        self.num_envs = num_envs
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_steps = n_steps                # steps per environment per rollout
        self.total_steps_per_update = num_envs * n_steps
        self.max_episode_steps = max_episode_steps

        # Create vectorized environment
        def _make_env(rank):
            def _init():
                env = make_env(env_name, seed=rank, robot=robot)
                return env
            return _init
        self.venv = AsyncVectorEnv([_make_env(i) for i in range(num_envs)],
                                   context='spawn')  # spawn for safety

        obs_dim = self.venv.single_observation_space.shape[0]
        act_dim = self.venv.single_action_space.shape[0]
        self.model = ActorCritic(obs_dim, act_dim, hidden_size, use_layernorm)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Logging
        self.episodic_rewards = []   # stores episode total rewards

    def _collect_rollout(self):
        """Run n_steps in each of the num_envs environments in parallel."""
        # Storage: lists of arrays, each of shape (n_steps, num_envs, ...)
        obs_buf = np.zeros((self.n_steps, self.num_envs, self.venv.single_observation_space.shape[0]), dtype=np.float32)
        act_buf = np.zeros((self.n_steps, self.num_envs, self.venv.single_action_space.shape[0]), dtype=np.float32)
        rew_buf = np.zeros((self.n_steps, self.num_envs), dtype=np.float32)
        next_obs_buf = np.zeros_like(obs_buf)
        done_buf = np.zeros((self.n_steps, self.num_envs), dtype=np.float32)
        logp_buf = np.zeros((self.n_steps, self.num_envs), dtype=np.float32)
        val_buf = np.zeros((self.n_steps, self.num_envs), dtype=np.float32)

        # Initial observations
        obs, _ = self.venv.reset()
        episode_rewards = np.zeros(self.num_envs)

        for step in range(self.n_steps):
            # Get actions and values for whole batch
            action, log_prob, value = self.model.act(obs)

            next_obs, reward, terminated, truncated, _ = self.venv.step(action)
            done = np.logical_or(terminated, truncated).astype(np.float32)

            # Store
            obs_buf[step] = obs
            act_buf[step] = action
            rew_buf[step] = reward
            next_obs_buf[step] = next_obs
            done_buf[step] = done
            logp_buf[step] = log_prob
            val_buf[step] = value

            # Update episode rewards and log complete episodes
            episode_rewards += reward
            for i in range(self.num_envs):
                if done[i]:
                    self.episodic_rewards.append(episode_rewards[i])
                    episode_rewards[i] = 0.0

            obs = next_obs

        # Bootstrapping value for the last state (if not done)
        with torch.no_grad():
            _, last_vals = self.model(torch.FloatTensor(obs))
            last_vals = last_vals.squeeze(-1).numpy()

        # Compute GAE and returns separately for each environment
        all_obs, all_acts, all_advs, all_rets, all_logps, all_vals = [], [], [], [], [], []
        for i in range(self.num_envs):
            env_rews = rew_buf[:, i]
            env_dones = done_buf[:, i]
            env_vals = val_buf[:, i]
            last_val = last_vals[i] if not done_buf[-1, i] else 0.0

            advs, rets = self._compute_gae(env_rews, env_vals, env_dones, last_val)

            # Append entire trajectory for this environment
            all_obs.append(obs_buf[:, i])
            all_acts.append(act_buf[:, i])
            all_advs.append(advs)
            all_rets.append(rets)
            all_logps.append(logp_buf[:, i])
            all_vals.append(env_vals)

        # Concatenate into flat arrays (across all environments)
        flat_obs = np.concatenate(all_obs, axis=0)
        flat_acts = np.concatenate(all_acts, axis=0)
        flat_advs = np.concatenate(all_advs, axis=0)
        flat_rets = np.concatenate(all_rets, axis=0)
        flat_logps = np.concatenate(all_logps, axis=0)
        flat_vals = np.concatenate(all_vals, axis=0)

        # Normalize advantages
        flat_advs = (flat_advs - flat_advs.mean()) / (flat_advs.std() + 1e-8)

        return (torch.FloatTensor(flat_obs),
                torch.FloatTensor(flat_acts),
                torch.FloatTensor(flat_advs),
                torch.FloatTensor(flat_rets),
                torch.FloatTensor(flat_logps),
                torch.FloatTensor(flat_vals))

    def _compute_gae(self, rewards, values, dones, last_val):
        """Compute Generalized Advantage Estimation for a single trajectory."""
        advantages = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = last_val
            else:
                next_val = values[t + 1]
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        return advantages, returns

    def update(self):
        """Perform one PPO update using collected rollout."""
        obs, acts, advs, rets, old_logps, old_vals = self._collect_rollout()

        # PPO epochs with minibatching
        dataset_size = obs.shape[0]
        for _ in range(self.epochs):
            indices = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, self.batch_size):
                idx = indices[start:start+self.batch_size]
                batch_obs = obs[idx]
                batch_acts = acts[idx]
                batch_advs = advs[idx]
                batch_rets = rets[idx]
                batch_old_logps = old_logps[idx]

                new_logps, vals, entropy = self.model.evaluate_actions(batch_obs, batch_acts)

                # Clipped surrogate objective
                ratio = torch.exp(new_logps - batch_old_logps)
                surr1 = ratio * batch_advs
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advs
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(vals, batch_rets)

                # Entropy bonus
                entropy_bonus = entropy.mean()

                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

    def train(self, num_updates=200):
        """Run the full training loop."""
        progress = tqdm(range(num_updates))
        for i in progress:
            self.update()
            if len(self.episodic_rewards) > 0:
                recent_avg = np.mean(self.episodic_rewards[-5:]) if len(self.episodic_rewards) >= 5 else np.mean(self.episodic_rewards)
                progress.set_description(f"Recent avg reward: {recent_avg:.2f}")

    def close(self):
        self.venv.close()

# ----------------------------------------------------------------------
# 5. Main training and final render
# ----------------------------------------------------------------------
if __name__ == "__main__":
    ENV_NAME = "Climber-v2"          # can be any Climber variant
    ROBOT = robot_config

    ppo = PPO(
        env_name=ENV_NAME,
        robot=ROBOT,
        num_envs=20,                   # run 4 environments in parallel
        hidden_size=256,
        lr=3e-4,
        gamma=0.99,
        lam=0.95,
        clip_epsilon=0.2,
        epochs=10,
        batch_size=128,
        n_steps=512,                  # steps per environment per rollout → total 4*128=512
        max_episode_steps=500,
        use_layernorm=True
    )

    ppo.train(num_updates=200)

    # Render final policy and save GIF
    env = make_env(ENV_NAME, robot=ROBOT, render_mode="rgb_array")
    fitness, imgs = evaluate(ppo.model, env, max_steps=500, render=True)
    print(f"Final fitness: {fitness}")
    imageio.mimsave('Climber.gif', imgs, duration=1/30.0)
    env.close()
    ppo.close()

    # Show training progress
    plt.plot(ppo.episodic_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Episode reward")
    plt.title(f"PPO on {ENV_NAME} (multithreaded)")
    plt.show()