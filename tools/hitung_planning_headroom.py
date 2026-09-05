"""Hitung planning headroom Bridge terhadap baseline greedy multiresolusi."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs" / "csv" / "iic_frontier_multiresolusi.csv"
OUTPUT = ROOT / "outputs" / "csv" / "planning_headroom_multiresolusi.csv"

NAMA_METODE = {
    "naif": "Greedy Naif (Delta IIC)",
    "cost": "Greedy Cost (Delta IIC/Cost)",
    "bridge": "Bridge Planner",
}


def persen(selisih: float, pembanding: float) -> float:
    return 100.0 * selisih / pembanding if pembanding != 0 else float("nan")


def main() -> None:
    if not INPUT.exists():
        raise SystemExit(f"Hasil eksperimen tidak ditemukan: {INPUT}")

    df = pd.read_csv(INPUT)
    rows = []
    for resolusi, grup in df.groupby("resolusi", sort=False):
        nilai = {}
        for nama, label in NAMA_METODE.items():
            cocok = grup.loc[grup["metode"] == label, "peningkatan_iic"]
            if len(cocok) != 1:
                raise SystemExit(
                    f"{resolusi}: dibutuhkan tepat satu hasil untuk {label}, "
                    f"ditemukan {len(cocok)}"
                )
            nilai[nama] = float(cocok.iloc[0])

        best_greedy = max(nilai["naif"], nilai["cost"])
        vs_naif = nilai["bridge"] - nilai["naif"]
        vs_cost = nilai["bridge"] - nilai["cost"]
        vs_best = nilai["bridge"] - best_greedy
        rows.append(
            {
                "resolusi": resolusi,
                "peningkatan_iic_naif": nilai["naif"],
                "peningkatan_iic_cost": nilai["cost"],
                "peningkatan_iic_bridge": nilai["bridge"],
                "headroom_vs_naif_absolut": vs_naif,
                "headroom_vs_naif_persen": persen(vs_naif, nilai["naif"]),
                "headroom_vs_cost_absolut": vs_cost,
                "headroom_vs_cost_persen": persen(vs_cost, nilai["cost"]),
                "headroom_vs_best_greedy_absolut": vs_best,
                "headroom_vs_best_greedy_persen": persen(vs_best, best_greedy),
                "planning_headroom_positif_persen": max(
                    0.0,
                    persen(vs_best, best_greedy),
                ),
            }
        )

    hasil = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    hasil.to_csv(OUTPUT, index=False)
    print(hasil.round(6).to_string(index=False))
    print(f"\nCSV: {OUTPUT}")


if __name__ == "__main__":
    main()
