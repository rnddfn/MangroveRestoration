from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RASTER_PATH = ROOT / "data" / "processed" / "kelas_clip2.npy"

RESOLUTION_CONFIG = {
    "32x64": {
        "raster_path": ROOT
        / "data"
        / "processed"
        / "multires"
        / "kelas_kualalupak_32x64.npy",
        "dispersal_distance": 5.0,
        "budget": 20,
    },
    "48x96": {
        "raster_path": ROOT
        / "data"
        / "processed"
        / "multires"
        / "kelas_kualalupak_48x96.npy",
        "dispersal_distance": 7.5,
        "budget": 45,
    },
    "64x128": {
        "raster_path": ROOT
        / "data"
        / "processed"
        / "multires"
        / "kelas_kualalupak_64x128.npy",
        "dispersal_distance": 10.0,
        "budget": 80,
    },
}

BUDGET = 20
DISPERSAL_DISTANCE = 5.0
TIMESTEPS = 50_000
SEEDS = [0]
N_EVAL = 20
INVALID_PENALTY = 0.05
REPEAT_INVALID_LIMIT = 3
MIN_STEPS = 300
STEPS_PER_BUDGET = 30
GAMMA = 1.0
ENT_COEF = 0.01
NET_ARCH = [256, 256]
REWARD_SCALE = 100.0
N_ENVS = 4

MANGROVE_CODES = [5, 76]
RESTORABLE_CODES = [13, 21]
LOCKED_CODES = [31, 40, 33, 0]

SKENARIO_LAHAN = {
    "utama": {
        "restorable_codes": [13, 21],
        "biaya_tambahan": {},
    },
    "s2": {
        "restorable_codes": [13, 21, 31, 40],
        "biaya_tambahan": {31: 3, 40: 3},
    },
}

PRESET_RL = {
    "default": {
        "env": {},
        "algo": {},
    },
    "s2-tuned": {
        "env": {"reward_per_biaya": True},
        "select_best": True,
        "eval_every": 10_000,
        "algo": {
            "learning_rate": 1e-4,
            "n_steps": 512,
            "batch_size": 128,
            "n_epochs": 5,
            "gae_lambda": 1.0,
            "ent_coef": 0.003,
            "target_kl": 0.02,
        },
    },
}

# Kelas biaya restorasi
KODE_MURAH = 13
KODE_MAHAL = 21
BIAYA_MURAH = 1
BIAYA_MAHAL = 2

BIAYA_DEFAULT = {KODE_MURAH: BIAYA_MURAH, KODE_MAHAL: BIAYA_MAHAL}

# Skenario biaya
SKENARIO_BIAYA = {
    "1:1": {13: 1, 21: 1},
    "1:2": {13: 1, 21: 2},
    "1:3": {13: 1, 21: 3},
}
SKENARIO_ANGGARAN = [10, 20, 30]
