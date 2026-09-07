import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from pathlib import Path

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


class SimpanIICterbaik(BaseCallback):
    def __init__(self, env_kw, model_path, eval_every=10_000):
        super().__init__()
        self.env_kw = dict(env_kw or {})
        self.model_path = Path(model_path)
        self.eval_every = int(eval_every)
        self.berikutnya = self.eval_every
        self.iic_terbaik = -np.inf

    def _evaluasi(self):
        env = RestorasiEnv(pakai_penalti=False, reward_scale=1.0, **self.env_kw)
        obs, info = env.reset(seed=10_000)
        selesai = False
        while not selesai:
            action = self.model.predict(
                obs,
                deterministic=True,
                action_masks=env.mask_aksi(),
            )[0]
            obs, _, term, trunc, info = env.step(action)
            selesai = term or trunc
        return float(info["iic"])

    def _on_step(self):
        if self.num_timesteps < self.berikutnya:
            return True
        iic = self._evaluasi()
        if iic > self.iic_terbaik:
            self.iic_terbaik = iic
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(self.model_path)
        print(
            f"\n[eval] step={self.num_timesteps}  iic={iic:.8f}  "
            f"best={self.iic_terbaik:.8f}"
        )
        while self.berikutnya <= self.num_timesteps:
            self.berikutnya += self.eval_every
        return True


def _factory(jenis, env_kw, reward_scale):
    def buat():
        kw = dict(env_kw or {})
        if jenis == "ppo":
            return RestorasiEnv(pakai_penalti=True, reward_scale=reward_scale, **kw)
        env = RestorasiEnv(pakai_penalti=False, reward_scale=reward_scale, **kw)
        return ActionMasker(env, lambda e: e.mask_aksi())

    return buat


def latih(
    jenis,
    seed,
    env_kw=None,
    timesteps=None,
    reward_scale=REWARD_SCALE,
    model_path=None,
    run_name=None,
    algo_kw=None,
    select_best=False,
    eval_every=10_000,
):
    steps = TIMESTEPS if timesteps is None else timesteps
    Algo, nama = (PPO, "PPO") if jenis == "ppo" else (MaskablePPO, "MaskablePPO")
    vec = DummyVecEnv([_factory(jenis, env_kw, reward_scale) for _ in range(N_ENVS)])
    log_dir = Path(__file__).resolve().parents[1] / "outputs" / "tb"
    log_dir.mkdir(parents=True, exist_ok=True)

    model_kw = {
        "seed": seed,
        "verbose": 0,
        "n_steps": 256,
        "gamma": GAMMA,
        "ent_coef": ENT_COEF,
        "policy_kwargs": dict(net_arch=NET_ARCH),
        "tensorboard_log": str(log_dir),
    }
    model_kw.update(algo_kw or {})
    model = Algo("MlpPolicy", vec, **model_kw)
    t0 = time.time()
    progress = ProgressLatih(steps, f"{nama} seed={seed}")
    pilih_terbaik = bool(select_best and jenis == "mask" and model_path is not None)
    if pilih_terbaik:
        callback = CallbackList(
            [progress, SimpanIICterbaik(env_kw, model_path, eval_every)]
        )
    else:
        callback = progress
    model.learn(
        total_timesteps=steps,
        callback=callback,
        tb_log_name=run_name or f"{nama}_seed{seed}",
    )
    print()
    if model_path is not None:
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        if pilih_terbaik and model_path.with_suffix(".zip").exists():
            model = Algo.load(model_path, env=vec)
        else:
            model.save(model_path)
        print(f"Model tersimpan: {model_path}")
    return model, time.time() - t0
