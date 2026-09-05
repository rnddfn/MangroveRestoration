import numpy as np
from scipy.ndimage import label
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path

from src.config import DISPERSAL_DISTANCE


def _patch_edges(lbl, jarak_max):
    # tautan jarak
    ys, xs = np.nonzero(lbl)
    id_patch = lbl[ys, xs] - 1
    tree = cKDTree(np.column_stack((ys, xs)).astype(np.float64))
    pasangan = tree.query_pairs(r=jarak_max, output_type="ndarray")
    if pasangan.size == 0:
        return np.empty((0, 2), dtype=int)
    a, b = id_patch[pasangan[:, 0]], id_patch[pasangan[:, 1]]
    beda = a != b
    a, b = a[beda], b[beda]
    if a.size == 0:
        return np.empty((0, 2), dtype=int)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    return np.unique(np.column_stack((lo, hi)), axis=0)


def hitung_iic(habitat, luas_lanskap, jarak_max=DISPERSAL_DISTANCE):
    # label petak
    lbl, n_patch = label(habitat, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    if n_patch == 0:
        return 0.0
    # luas tiap petak
    luas = np.bincount(lbl.ravel())[1:].astype(np.float64)
    if n_patch == 1:
        return float(luas[0] ** 2) / (luas_lanskap ** 2)
    # graf hop
    edges = _patch_edges(lbl, jarak_max)
    if len(edges) == 0:
        graph = csr_matrix((n_patch, n_patch))
    else:
        graph = csr_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n_patch, n_patch))
    hop = shortest_path(graph, method="D", directed=False, unweighted=True)
    # rumus IIC
    nilai = np.outer(luas, luas) / (1.0 + hop)
    nilai[np.isinf(hop)] = 0.0
    return float(nilai.sum()) / (luas_lanskap ** 2)


def statistik_petak(habitat):
    lbl, n_patch = label(habitat, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    if n_patch == 0:
        return 0, 0
    luas = np.bincount(lbl.ravel())[1:]
    return int(n_patch), int(luas.max())


def jumlah_komponen_konektivitas(habitat, jarak_max=DISPERSAL_DISTANCE):
    """Jumlah grup patch yang terhubung pada ambang dispersal tertentu."""
    _, n_komponen = label_komponen_konektivitas(habitat, jarak_max)
    return n_komponen


def label_komponen_konektivitas(habitat, jarak_max=DISPERSAL_DISTANCE):
    """Label komponen konektivitas per sel habitat; nonhabitat bernilai 0."""
    lbl, n_patch = label(
        habitat,
        structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
    )
    if n_patch == 0:
        return np.zeros_like(habitat, dtype=np.int32), 0
    edges = _patch_edges(lbl, jarak_max)
    if len(edges) == 0:
        label_patch = np.arange(n_patch, dtype=np.int32)
        n_komponen = n_patch
    else:
        graph = csr_matrix(
            (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
            shape=(n_patch, n_patch),
        )
        n_komponen, label_patch = connected_components(graph, directed=False)
    label_sel = np.zeros_like(lbl, dtype=np.int32)
    mask = lbl > 0
    label_sel[mask] = label_patch[lbl[mask] - 1] + 1
    return label_sel, int(n_komponen)
