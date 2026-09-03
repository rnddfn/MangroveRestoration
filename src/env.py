import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.iic import hitung_iic
from src.config import (
    RASTER_PATH,
    BUDGET,
    MANGROVE_CODES,
    RESTORABLE_CODES,
    LOCKED_CODES,
    INVALID_PENALTY,
    REPEAT_INVALID_LIMIT,
    MIN_STEPS,
    STEPS_PER_BUDGET,
)

class RestorasiEnv(gym.Env):
    def __init__(self, path=RASTER_PATH, budget=BUDGET, pakai_penalti=False, reward_scale=1.0):
        super().__init__()
        kelas = np.load(path)
        self.H, self.W = kelas.shape
        self.n_sel = self.H * self.W
        self.mangrove = np.isin(kelas, MANGROVE_CODES)
        self.restorable = np.isin(kelas, RESTORABLE_CODES)
        self.dikunci = np.isin(kelas, LOCKED_CODES)
        self.budget_awal = budget
        if int(self.restorable.sum()) < budget:
            raise ValueError(f"Budget {budget} > restorable {int(self.restorable.sum())}")
        self.pakai_penalti = pakai_penalti
        self.reward_scale = reward_scale
        self.luas_lanskap = float((self.mangrove | self.restorable).sum())
        self.batas_langkah = max(MIN_STEPS, budget * STEPS_PER_BUDGET)
        self.action_space = spaces.Discrete(self.n_sel)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(3, self.H, self.W), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sudah_restore = np.zeros((self.H, self.W), dtype=np.uint8)
        self.habitat = self.mangrove.copy()
        self.sisa_budget = self.budget_awal
        self.iic_awal = hitung_iic(self.habitat, self.luas_lanskap)
        self.iic = self.iic_awal
        self.n_invalid = 0
        self.langkah_ke = 0
        self._aksi_invalid_terakhir = None
        self._aksi_invalid_beruntun = 0
        return self._obs(), {"iic": self.iic, "iic_awal": self.iic_awal}

    def _obs(self):
        masih = (self.restorable & (self.sudah_restore == 0)).astype(np.float32)
        bud = np.full((self.H, self.W), self.sisa_budget / max(self.budget_awal, 1), dtype=np.float32)
        return np.stack([self.habitat.astype(np.float32), masih, bud])

    def mask_aksi(self):
        return self.restorable.flatten() & (self.sudah_restore.flatten() == 0)

    def step(self, action):
        r, c = divmod(int(action), self.W)
        valid = bool(self.restorable[r, c] and self.sudah_restore[r, c] == 0 and self.sisa_budget > 0)
        if valid:
            # tanam satu sel
            self.sudah_restore[r, c] = 1
            self.habitat[r, c] = True
            self.sisa_budget -= 1
            iic_baru = hitung_iic(self.habitat, self.luas_lanskap)
            reward = iic_baru - self.iic
            self.iic = iic_baru
            self._aksi_invalid_beruntun = 0
            self._aksi_invalid_terakhir = None
        else:
            # aksi tidak sah
            self.n_invalid += 1
            reward = -INVALID_PENALTY if self.pakai_penalti else 0.0
            sama = action == self._aksi_invalid_terakhir
            self._aksi_invalid_beruntun = self._aksi_invalid_beruntun + 1 if sama else 1
            self._aksi_invalid_terakhir = action
        self.langkah_ke += 1
        selesai = (self.sisa_budget <= 0) or (int(self.mask_aksi().sum()) == 0)
        trunc = self._aksi_invalid_beruntun >= REPEAT_INVALID_LIMIT or self.langkah_ke >= self.batas_langkah
        info = {"iic": self.iic, "iic_awal": self.iic_awal, "invalid": not valid}
        return self._obs(), reward * self.reward_scale, selesai, trunc, info

def cek_raster(path=RASTER_PATH):
    kelas = np.load(path)
    print(f"Raster {path}: {kelas.shape}, n={kelas.size}")
    nama = {
        5: "mangrove", 76: "rawa gambut", 13: "restorable", 21: "restorable",
        31: "tambak", 40: "sawah", 33: "air", 0: "NoData",
    }
    for kode, n in zip(*np.unique(kelas, return_counts=True)):
        print(f"  {int(kode):>3}  {nama.get(int(kode), '?'):16s}  {n}")
    print(f"  tanam={int(np.isin(kelas, RESTORABLE_CODES).sum())}  budget={BUDGET}")
    lain = [int(k) for k in np.unique(kelas) if int(k) not in nama]
    if lain:
        print("  kode lain:", lain)
