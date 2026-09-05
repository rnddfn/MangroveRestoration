import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.iic import hitung_iic, statistik_petak
from src.config import (
    RASTER_PATH,
    BUDGET,
    MANGROVE_CODES,
    RESTORABLE_CODES,
    LOCKED_CODES,
    BIAYA_DEFAULT,
    INVALID_PENALTY,
    REPEAT_INVALID_LIMIT,
    MIN_STEPS,
    STEPS_PER_BUDGET,
)

TETANGGA = ((-1, 0), (1, 0), (0, -1), (0, 1))


class RestorasiEnv(gym.Env):
    def __init__(
        self,
        path=RASTER_PATH,
        budget=BUDGET,
        biaya=None,
        pakai_penalti=False,
        reward_scale=1.0,
        jarak_max=None,
        reward_akhir=False,
        ukuran_blok=1,
    ):
        super().__init__()
        from src.config import DISPERSAL_DISTANCE
        kelas = np.load(path)
        self.kelas = kelas.astype(np.int16)
        self.H, self.W = kelas.shape
        self.n_sel = self.H * self.W
        self.mangrove = np.isin(kelas, MANGROVE_CODES)
        self.restorable = np.isin(kelas, RESTORABLE_CODES)
        self.dikunci = np.isin(kelas, LOCKED_CODES)
        self.biaya_peta = np.zeros((self.H, self.W), dtype=np.float32)
        for kode, hrg in (biaya or BIAYA_DEFAULT).items():
            self.biaya_peta[self.kelas == kode] = float(hrg)
        self.biaya_max = float(max(self.biaya_peta.max(), 1.0))
        self.budget_awal = float(budget)
        self.pakai_penalti = pakai_penalti
        self.reward_scale = reward_scale
        self.jarak_max = DISPERSAL_DISTANCE if jarak_max is None else float(jarak_max)
        self.reward_akhir = bool(reward_akhir)
        self.ukuran_blok = max(1, int(ukuran_blok))
        self.luas_lanskap = float((self.mangrove | self.restorable).sum())
        self.batas_langkah = max(MIN_STEPS, int(budget) * STEPS_PER_BUDGET)
        self.action_space = spaces.Discrete(self.n_sel)
        self.observation_space = spaces.Box(
            0.0, 1.0, shape=(4, self.H, self.W), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sudah_restore = np.zeros((self.H, self.W), dtype=np.uint8)
        self.habitat = self.mangrove.copy()
        self.frontier = self._frontier_awal()
        self.sisa_budget = self.budget_awal
        self.biaya_terpakai = 0.0
        self.n_tanam = 0
        self.urutan = []
        self.iic_awal = hitung_iic(self.habitat, self.luas_lanskap, self.jarak_max)
        self.iic = self.iic_awal
        self.n_invalid = 0
        self.langkah_ke = 0
        self._aksi_invalid_terakhir = None
        self._aksi_invalid_beruntun = 0
        return self._obs(), self._info(valid=True)

    def _frontier_awal(self):
        buka = np.zeros((self.H, self.W), dtype=bool)
        ys, xs = np.nonzero(self.mangrove)
        for y, x in zip(ys, xs):
            for dy, dx in TETANGGA:
                r, c = y + dy, x + dx
                if 0 <= r < self.H and 0 <= c < self.W and self.restorable[r, c]:
                    buka[r, c] = True
        return buka

    def _buka_tetangga(self, r, c):
        for dy, dx in TETANGGA:
            y, x = r + dy, c + dx
            if 0 <= y < self.H and 0 <= x < self.W:
                if self.restorable[y, x] and self.sudah_restore[y, x] == 0:
                    self.frontier[y, x] = True
        self.frontier[r, c] = False

    def _obs(self):
        front = (self.frontier & (self.sudah_restore == 0)).astype(np.float32)
        biaya = self.biaya_peta / self.biaya_max
        bud = np.full(
            (self.H, self.W),
            self.sisa_budget / max(self.budget_awal, 1.0),
            dtype=np.float32,
        )
        return np.stack(
            [self.habitat.astype(np.float32), front, biaya.astype(np.float32), bud]
        )

    def mask_aksi(self):
        cukup = self.biaya_peta <= self.sisa_budget
        m = self.restorable & (self.sudah_restore == 0) & self.frontier & cukup
        return m.flatten()

    def _sah(self, r, c):
        return bool(
            self.restorable[r, c]
            and self.sudah_restore[r, c] == 0
            and self.frontier[r, c]
            and self.biaya_peta[r, c] <= self.sisa_budget
        )

    def blok_dari(self, r, c):
        """Sel bersebelahan yang ditanam dari jangkar (r, c), terikat budget."""
        if not self._sah(r, c):
            return []
        sisa = self.sisa_budget
        ambil = []
        dilihat = set()
        antrian = [(r, c)]
        dilihat.add((r, c))
        while antrian and len(ambil) < self.ukuran_blok:
            y, x = antrian.pop(0)
            hrg = float(self.biaya_peta[y, x])
            if hrg > sisa:
                continue
            ambil.append((y, x))
            sisa -= hrg
            for dy, dx in TETANGGA:
                ny, nx = y + dy, x + dx
                if not (0 <= ny < self.H and 0 <= nx < self.W):
                    continue
                if (ny, nx) in dilihat:
                    continue
                dilihat.add((ny, nx))
                if (
                    self.restorable[ny, nx]
                    and self.sudah_restore[ny, nx] == 0
                    and (self.frontier[ny, nx] or (y, x) in ambil)
                    and float(self.biaya_peta[ny, nx]) <= sisa
                ):
                    antrian.append((ny, nx))
        return ambil

    def step(self, action):
        r, c = divmod(int(action), self.W)
        blok = self.blok_dari(r, c)
        valid = len(blok) > 0
        if valid:
            cost = 0.0
            for y, x in blok:
                hrg = float(self.biaya_peta[y, x])
                self.sudah_restore[y, x] = 1
                self.habitat[y, x] = True
                cost += hrg
                self.n_tanam += 1
                self.urutan.append(y * self.W + x)
                self._buka_tetangga(y, x)
            self.sisa_budget -= cost
            self.biaya_terpakai += cost
            iic_baru = hitung_iic(self.habitat, self.luas_lanskap, self.jarak_max)
            reward = iic_baru - self.iic
            self.iic = iic_baru
            if self.reward_akhir:
                reward = 0.0
            self._aksi_invalid_beruntun = 0
            self._aksi_invalid_terakhir = None
        else:
            self.n_invalid += 1
            reward = -INVALID_PENALTY if self.pakai_penalti else 0.0
            sama = action == self._aksi_invalid_terakhir
            self._aksi_invalid_beruntun = self._aksi_invalid_beruntun + 1 if sama else 1
            self._aksi_invalid_terakhir = action
        self.langkah_ke += 1
        selesai = (self.sisa_budget <= 0) or (int(self.mask_aksi().sum()) == 0)
        trunc = (
            self._aksi_invalid_beruntun >= REPEAT_INVALID_LIMIT
            or self.langkah_ke >= self.batas_langkah
        )
        if self.reward_akhir and (selesai or trunc):
            reward = self.iic - self.iic_awal
        return self._obs(), reward * self.reward_scale, selesai, trunc, self._info(valid)

    def _info(self, valid):
        n_petak, petak_max = statistik_petak(self.habitat)
        return {
            "iic": self.iic,
            "iic_awal": self.iic_awal,
            "invalid": not valid,
            "biaya_terpakai": self.biaya_terpakai,
            "n_tanam": self.n_tanam,
            "n_petak": n_petak,
            "petak_max": petak_max,
            "sisa_budget": self.sisa_budget,
        }


def cek_raster(path=RASTER_PATH, budget=BUDGET, biaya=None):
    kelas = np.load(path)
    print(f"Raster {path}: {kelas.shape}, n={kelas.size}")
    nama = {
        5: "mangrove",
        76: "rawa gambut",
        13: "non-hutan lain",
        21: "pertanian lain",
        31: "tambak",
        40: "sawah",
        33: "air",
        0: "NoData",
    }
    for kode, n in zip(*np.unique(kelas, return_counts=True)):
        print(f"  {int(kode):>3}  {nama.get(int(kode), '?'):16s}  {n}")
    rest = np.isin(kelas, RESTORABLE_CODES)
    mang = np.isin(kelas, MANGROVE_CODES)
    front = np.zeros_like(rest)
    ys, xs = np.nonzero(mang)
    for y, x in zip(ys, xs):
        for dy, dx in TETANGGA:
            r, c = y + dy, x + dx
            if 0 <= r < kelas.shape[0] and 0 <= c < kelas.shape[1] and rest[r, c]:
                front[r, c] = True
    print(f"  restorable={int(rest.sum())}  frontier awal={int(front.sum())}")
    print(f"  budget unit={budget}  biaya={biaya or BIAYA_DEFAULT}")
