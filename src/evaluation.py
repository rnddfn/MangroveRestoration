import numpy as np
import pandas as pd

from src.env import RestorasiEnv
from src.iic import hitung_iic
from src.config import N_EVAL


def aksi_acak(env, obs=None):
    pilihan = np.flatnonzero(env.mask_aksi())
    if pilihan.size == 0:
        return 0
    return int(np.random.choice(pilihan))


def aksi_greedy(env, obs=None):
    pilihan = np.flatnonzero(env.mask_aksi())
    if pilihan.size == 0:
        return 0
    terbaik, dmax = int(pilihan[0]), -1e18
    for a in pilihan:
        r, c = divmod(int(a), env.W)
        coba = env.habitat.copy()
        coba[r, c] = True
        d = hitung_iic(coba, env.luas_lanskap) - env.iic
        if d > dmax:
            dmax, terbaik = d, int(a)
    return terbaik


def aksi_greedy_biaya(env, obs=None):
    pilihan = np.flatnonzero(env.mask_aksi())
    if pilihan.size == 0:
        return 0
    terbaik, rasio_max = int(pilihan[0]), -1e18
    for a in pilihan:
        r, c = divmod(int(a), env.W)
        cost = max(float(env.biaya_peta[r, c]), 1e-9)
        coba = env.habitat.copy()
        coba[r, c] = True
        d = hitung_iic(coba, env.luas_lanskap) - env.iic
        rasio = d / cost
        if rasio > rasio_max:
            rasio_max, terbaik = rasio, int(a)
    return terbaik


def jalankan_episode(env, pilih_aksi, seed=None):
    obs, info = env.reset(seed=seed)
    selesai = False
    while not selesai:
        obs, _, term, trunc, info = env.step(pilih_aksi(env, obs))
        selesai = term or trunc
    langkah = max(env.langkah_ke, 1)
    return {
        "awal": info["iic_awal"],
        "akhir": info["iic"],
        "invalid": env.n_invalid,
        "langkah": langkah,
        "biaya": env.biaya_terpakai,
        "n_tanam": env.n_tanam,
        "n_petak": info["n_petak"],
        "petak_max": info["petak_max"],
        "urutan": list(env.urutan),
    }


def dari_df(nama, df, waktu=0.0):
    d = df["akhir"] - df["awal"]
    biaya = df["biaya"].replace(0, np.nan)
    return {
        "metode": nama,
        "iic_awal": df["awal"].mean(),
        "iic_akhir": df["akhir"].mean(),
        "peningkatan_iic": d.mean(),
        "std_peningkatan_iic": d.std(),
        "biaya_terpakai": df["biaya"].mean(),
        "iic_per_unit": (d / biaya).mean(),
        "rata_sel_ditanam": df["n_tanam"].mean(),
        "rata_n_petak": df["n_petak"].mean(),
        "rata_petak_max": df["petak_max"].mean(),
        "invalid_action_rate": (df["invalid"] / df["langkah"]).mean(),
        "training_time_detik": waktu,
    }


def _env(pakai_penalti=False, reward_scale=1.0, env_kw=None):
    kw = dict(env_kw or {})
    return RestorasiEnv(pakai_penalti=pakai_penalti, reward_scale=reward_scale, **kw)


def evaluasi_baseline(nama, pilih_aksi, n=N_EVAL, env_kw=None):
    baris = []
    for i in range(n):
        print(f"\r  [{nama}] {i + 1}/{n}", end="", flush=True)
        np.random.seed(i)
        baris.append(jalankan_episode(_env(env_kw=env_kw), pilih_aksi, seed=i))
    print()
    return dari_df(nama, pd.DataFrame(baris))


def prediksi(model, env, obs, pakai_mask):
    if pakai_mask:
        return model.predict(obs, deterministic=True, action_masks=env.mask_aksi())[0]
    return model.predict(obs, deterministic=True)[0]


def evaluasi_model(model, pakai_mask, n=N_EVAL, env_kw=None):
    def pilih(env, obs):
        return prediksi(model, env, obs, pakai_mask)

    baris = []
    for i in range(n):
        print(f"\r  [evaluasi] {i + 1}/{n}", end="", flush=True)
        env = _env(pakai_penalti=not pakai_mask, reward_scale=1.0, env_kw=env_kw)
        baris.append(jalankan_episode(env, pilih, seed=1000 + i))
    print()
    return pd.DataFrame(baris)


def ringkas(nama, daftar_model_waktu, pakai_mask, env_kw=None):
    metrik = []
    for model, waktu in daftar_model_waktu:
        df = evaluasi_model(model, pakai_mask, env_kw=env_kw)
        metrik.append(dari_df(nama, df, waktu))
    dfm = pd.DataFrame(metrik)
    out = dfm.mean(numeric_only=True).to_dict()
    out["metode"] = nama
    out["std_peningkatan_iic"] = dfm["peningkatan_iic"].std()
    return out
