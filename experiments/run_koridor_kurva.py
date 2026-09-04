"""Satu seed MaskablePPO di peta koridor + kurva reward.

Tidak menimpa iic_frontier_skenario.csv.
Default: 200k step, ent_coef 0.05, seed 0.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import NET_ARCH, N_ENVS, REWARD_SCALE, GAMMA
from src.env import RestorasiEnv, cek_raster
from src.evaluation import evaluasi_model, dari_df

RASTER = ROOT / "data" / "processed" / "koridor_32x64.npy"
BIAYA = {13: 1, 21: 3}
ANGGARAN = 8


class KurvaCallback(BaseCallback):
    def __init__(self, path_csv, total, reward_scale=REWARD_SCALE, update_every=256):
        super().__init__()
        self.path_csv = Path(path_csv)
        self.total = total
        self.reward_scale = reward_scale
        self.update_every = update_every
        self.episode_rewards = []
        self._cur = None
        self._t0 = None
        self.path_csv.parent.mkdir(parents=True, exist_ok=True)
        with self.path_csv.open("w", newline="") as f:
            csv.writer(f).writerow(["timestep", "episode", "R20", "R_terakhir"])

    def _on_training_start(self):
        self._t0 = time.time()
        self._cur = np.zeros(self.training_env.num_envs, dtype=np.float64)

    def _on_step(self):
        self._cur += np.asarray(self.locals["rewards"], dtype=np.float64)
        for i in np.flatnonzero(np.asarray(self.locals["dones"])):
            self.episode_rewards.append(self._cur[i] / self.reward_scale)
            self._cur[i] = 0.0
        if self.num_timesteps % self.update_every == 0 or self.num_timesteps >= self.total:
            mean_r = float(np.mean(self.episode_rewards[-20:])) if self.episode_rewards else float("nan")
            last_r = float(self.episode_rewards[-1]) if self.episode_rewards else float("nan")
            with self.path_csv.open("a", newline="") as f:
                csv.writer(f).writerow(
                    [self.num_timesteps, len(self.episode_rewards), mean_r, last_r]
                )
            elapsed = time.time() - self._t0
            sps = self.num_timesteps / elapsed if elapsed > 0 else 0.0
            eta = max(self.total - self.num_timesteps, 0) / sps if sps > 0 else float("inf")
            frac = min(self.num_timesteps / self.total, 1.0)
            bar = "#" * int(frac * 24) + "-" * (24 - int(frac * 24))
            print(
                f"\r[kurva] |{bar}| {self.num_timesteps:>7}/{self.total} "
                f"R20={mean_r:8.4f}  ep={len(self.episode_rewards):>4}  ETA={eta:5.0f}s",
                end="",
                flush=True,
            )
            if self.num_timesteps >= self.total:
                print()
        return True


def _buat_env(reward_akhir, reward_scale):
    def buat():
        kw = dict(path=str(RASTER), biaya=BIAYA, budget=ANGGARAN, pakai_penalti=False, reward_scale=reward_scale)
        try:
            env = RestorasiEnv(reward_akhir=reward_akhir, **kw)
        except TypeError:
            env = RestorasiEnv(**kw)
        return ActionMasker(env, lambda e: e.mask_aksi())

    return buat


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--ent-coef", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reward-akhir", action="store_true")
    args = p.parse_args()

    if not RASTER.exists():
        raise SystemExit("Jalankan dulu: python tools/buat_koridor.py")

    cek_raster(path=RASTER, budget=ANGGARAN, biaya=BIAYA)
    out_dir = ROOT / "outputs" / "csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    path_kurva = out_dir / "kurva_koridor.csv"

    vec = DummyVecEnv([_buat_env(args.reward_akhir, REWARD_SCALE) for _ in range(N_ENVS)])
    kw_model = dict(
        seed=args.seed,
        verbose=0,
        n_steps=256,
        gamma=GAMMA,
        ent_coef=args.ent_coef,
        policy_kwargs=dict(net_arch=NET_ARCH),
    )
    tb = ROOT / "outputs" / "tb"
    try:
        tb.mkdir(parents=True, exist_ok=True)
        model = MaskablePPO("MlpPolicy", vec, tensorboard_log=str(tb), **kw_model)
        tb_name = f"koridor_ent{args.ent_coef}_seed{args.seed}"
    except ImportError:
        model = MaskablePPO("MlpPolicy", vec, **kw_model)
        tb_name = None
        print("TensorBoard tidak terpasang; kurva tetap disimpan ke CSV.")

    t0 = time.time()
    learn_kw = dict(
        total_timesteps=args.timesteps,
        callback=KurvaCallback(path_kurva, args.timesteps),
    )
    if tb_name:
        learn_kw["tb_log_name"] = tb_name
    try:
        model.learn(**learn_kw)
    except ImportError:
        model.learn(total_timesteps=args.timesteps, callback=KurvaCallback(path_kurva, args.timesteps))
    waktu = time.time() - t0

    env_kw = dict(path=str(RASTER), biaya=BIAYA, budget=ANGGARAN)
    df_ep = evaluasi_model(model, True, env_kw=env_kw)
    ringkas = dari_df("MaskablePPO kurva", df_ep, waktu)
    print(pd.DataFrame([ringkas]).round(4).to_string(index=False))

    out_hasil = out_dir / "iic_koridor_kurva.csv"
    pd.DataFrame([ringkas]).to_csv(out_hasil, index=False)
    print(f"Kurva: {path_kurva}")
    print(f"Hasil: {out_hasil}")
    print("Buka kurva: tensorboard --logdir outputs/tb")
    print("Atau buka CSV timestep,R20 di Excel.")
