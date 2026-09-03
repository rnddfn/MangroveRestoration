import numpy as np
import pandas as pd

from src.env import RestorasiEnv
from src.iic import hitung_iic
from src.config import BUDGET, N_EVAL


def aksi_acak(env, obs=None):
    return int(np.random.choice(np.flatnonzero(env.mask_aksi())))

def aksi_greedy(env, obs=None):
    pilihan = np.flatnonzero(env.mask_aksi())
    terbaik, dmax = int(pilihan[0]), -1e18
    for a in pilihan:
        r, c = divmod(int(a), env.W)
        coba = env.habitat.copy()
        coba[r, c] = True
        d = hitung_iic(coba, env.luas_lanskap) - env.iic
        if d > dmax:
            dmax, terbaik = d, int(a)
    return terbaik

def jalankan_episode(env, pilih_aksi, seed=None):
    obs, info = env.reset(seed=seed)
    selesai = False
    while not selesai:
        obs, _, term, trunc, info = env.step(pilih_aksi(env, obs))
        selesai = term or trunc
    return info["iic_awal"], info["iic"], env.n_invalid

def jalankan(env, pilih_aksi):
    return jalankan_episode(env, pilih_aksi)


def dari_df(nama, df, waktu=0.0):
    d = df["akhir"] - df["awal"]
    return {
        "metode": nama,
        "iic_awal": df["awal"].mean(),
        "iic_akhir": df["akhir"].mean(),
        "peningkatan_iic": d.mean(),
        "std_peningkatan_iic": d.std(),
        "invalid_action_rate": df["invalid"].mean() / BUDGET,
        "training_time_detik": waktu,
    }


def evaluasi_baseline(nama, pilih_aksi, n=N_EVAL):
    baris = []
    for i in range(n):
        print(f"\r  [{nama}] {i + 1}/{n}", end="", flush=True)
        np.random.seed(i)
        baris.append(jalankan_episode(RestorasiEnv(), pilih_aksi, seed=i))
    print()
    return dari_df(nama, pd.DataFrame(baris, columns=["awal", "akhir", "invalid"]))


def prediksi(model, env, obs, pakai_mask):
    if pakai_mask:
        return model.predict(obs, deterministic=True, action_masks=env.mask_aksi())[0]
    return model.predict(obs, deterministic=True)[0]


def evaluasi_model(model, pakai_mask, n=N_EVAL):
    def pilih(env, obs):
        return prediksi(model, env, obs, pakai_mask)

    baris = []
    for i in range(n):
        print(f"\r  [evaluasi] {i + 1}/{n}", end="", flush=True)
        env = RestorasiEnv(pakai_penalti=not pakai_mask, reward_scale=1.0)
        baris.append(jalankan_episode(env, pilih, seed=1000 + i))
    print()
    return pd.DataFrame(baris, columns=["awal", "akhir", "invalid"])


def ringkas(nama, daftar_model_waktu, pakai_mask):
    metrik = []
    for model, waktu in daftar_model_waktu:
        df = evaluasi_model(model, pakai_mask)
        metrik.append(dari_df(nama, df, waktu))
    dfm = pd.DataFrame(metrik)
    out = dfm.mean(numeric_only=True).to_dict()
    out["metode"] = nama
    out["std_peningkatan_iic"] = dfm["peningkatan_iic"].std()
    return out
