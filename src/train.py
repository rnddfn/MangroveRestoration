import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from src.env import RestorasiEnv
from src.config import GAMMA, ENT_COEF, NET_ARCH, N_ENVS, REWARD_SCALE, TIMESTEPS


class ProgressLatih(BaseCallback):
    def __init__(self, total_timesteps, label, reward_scale=REWARD_SCALE, update_every=256):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.label = label
        self.reward_scale = reward_scale
        self.update_every = update_every
        self.episode_rewards = []
        self._cur = None
        self._t0 = None

    def _on_training_start(self):
        self._t0 = time.time()
        self._cur = np.zeros(self.training_env.num_envs, dtype=np.float64)

    def _on_step(self):
        self._cur += np.asarray(self.locals["rewards"], dtype=np.float64)
        for i in np.flatnonzero(np.asarray(self.locals["dones"])):
            self.episode_rewards.append(self._cur[i] / self.reward_scale)
            self._cur[i] = 0.0
        if self.num_timesteps % self.update_every == 0 or self.num_timesteps >= self.total_timesteps:
            elapsed = time.time() - self._t0
            sps = self.num_timesteps / elapsed if elapsed > 0 else 0.0
            eta = max(self.total_timesteps - self.num_timesteps, 0) / sps if sps > 0 else float("inf")
            frac = min(self.num_timesteps / self.total_timesteps, 1.0)
            bar = "#" * int(frac * 24) + "-" * (24 - int(frac * 24))
            mean_r = np.mean(self.episode_rewards[-20:]) if self.episode_rewards else float("nan")
            print(
                f"\r[{self.label}] |{bar}| {self.num_timesteps:>6}/{self.total_timesteps} "
                f"({100 * frac:5.1f}%)  {sps:5.1f} step/s  ep={len(self.episode_rewards):>4}  "
                f"R20={mean_r:8.4f}  ETA={eta:5.0f}s",
                end="", flush=True,
            )
            if self.num_timesteps >= self.total_timesteps:
                print()
        return True


def _env_ppo():
    return RestorasiEnv(pakai_penalti=True, reward_scale=REWARD_SCALE)


def _env_mask():
    env = RestorasiEnv(pakai_penalti=False, reward_scale=REWARD_SCALE)
    return ActionMasker(env, lambda e: e.mask_aksi())


def latih(jenis, seed):
    Algo, factory, nama = (PPO, _env_ppo, "PPO") if jenis == "ppo" else (MaskablePPO, _env_mask, "MaskablePPO")
    vec = DummyVecEnv([factory for _ in range(N_ENVS)])
    model = Algo(
        "MlpPolicy", vec, seed=seed, verbose=0, n_steps=256,
        gamma=GAMMA, ent_coef=ENT_COEF, policy_kwargs=dict(net_arch=NET_ARCH),
    )
    t0 = time.time()
    model.learn(total_timesteps=TIMESTEPS, callback=ProgressLatih(TIMESTEPS, f"{nama} seed={seed}"))
    print()
    return model, time.time() - t0