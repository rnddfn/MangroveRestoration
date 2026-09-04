from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RASTER_PATH = ROOT / "data" / "processed" / "kelas_clip2.npy"

BUDGET = 8
DISPERSAL_DISTANCE = 5.0
TIMESTEPS = 50_000
SEEDS = [0, 1, 2]
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

# 13 = tumbuhan non-hutan lain, 21 = pertanian lain
KODE_MURAH = 13
KODE_MAHAL = 21
BIAYA_MURAH = 1
BIAYA_MAHAL = 1

BIAYA_DEFAULT = {KODE_MURAH: BIAYA_MURAH, KODE_MAHAL: BIAYA_MAHAL}

# skenario uji
SKENARIO_BIAYA = {
    "1:1": {13: 1, 21: 1},
    "1:2": {13: 1, 21: 2},
    "1:3": {13: 1, 21: 3},
}
SKENARIO_ANGGARAN = [10, 20, 30]
