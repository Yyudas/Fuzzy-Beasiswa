"""
fuzzy_beasiswa: logika Fuzzy Mamdani untuk SPK kelayakan penerima beasiswa.

Berkas ini TIDAK bergantung pada Streamlit, jadi bisa diimpor dan diuji sendiri:
    from fuzzy_beasiswa import mamdani, susun_aturan, ATURAN_BAWAAN
"""
import io
import numpy as np
import pandas as pd

# =====================================================================
# 1. FUNGSI KEANGGOTAAN
# =====================================================================
def trapesium(x, a, b, c, d):
    """µ(x) trapesium (a,b,c,d). Segitiga: b == c. Bahu kiri: a == b. Bahu kanan: c == d."""
    x = np.asarray(x, dtype=float)
    naik  = np.where(x >= b, 1.0, np.where(x <= a, 0.0, (x - a) / np.where(b > a, b - a, 1.0)))
    turun = np.where(x <= c, 1.0, np.where(x >= d, 0.0, (d - x) / np.where(d > c, d - c, 1.0)))
    return np.minimum(naik, turun)

VAR = {
    "IPK":         {"u": (0, 4),   "s": {"rendah": (0, 0, 2.5, 3.0), "sedang": (2.75, 3.25, 3.25, 3.75), "tinggi": (3.25, 3.75, 4, 4)}},
    "Penghasilan": {"u": (0, 15),  "s": {"rendah": (0, 0, 2.5, 3.9), "sedang": (2.5, 5.5, 5.5, 9),       "tinggi": (7.5, 11, 15, 15)}},
    "Tanggungan":  {"u": (0, 8),   "s": {"sedikit": (0, 0, 1, 3),    "sedang": (2, 4, 4, 6),            "banyak": (5, 7, 8, 8)}},
    "Prestasi":    {"u": (0, 100), "s": {"rendah": (0, 0, 30, 50),   "sedang": (30, 50, 50, 75),        "tinggi": (60, 80, 100, 100)}},
    "Kelayakan":   {"u": (0, 100), "s": {"tidak layak": (0, 0, 20, 40), "cukup layak": (25, 45, 45, 65),
                                          "layak": (50, 70, 70, 85),   "sangat layak": (70, 85, 100, 100)}},
}
INPUT = ["IPK", "Penghasilan", "Tanggungan", "Prestasi"]

# =====================================================================
# 2. BASIS ATURAN BAWAAN (81 aturan = 3 x 3 x 3 x 3, dari Basis_Aturan2.csv)
#    Tanda "-" = variabel tidak dipakai pada aturan itu
# =====================================================================
ATURAN_BAWAAN = [
    ("R1", "rendah", "tinggi", "sedikit", "rendah", "tidak layak"),
    ("R2", "rendah", "tinggi", "sedikit", "sedang", "tidak layak"),
    ("R3", "rendah", "tinggi", "sedikit", "tinggi", "tidak layak"),
    ("R4", "rendah", "tinggi", "sedang", "rendah", "tidak layak"),
    ("R5", "rendah", "tinggi", "sedang", "sedang", "tidak layak"),
    ("R6", "rendah", "tinggi", "sedang", "tinggi", "tidak layak"),
    ("R7", "rendah", "tinggi", "banyak", "rendah", "tidak layak"),
    ("R8", "rendah", "tinggi", "banyak", "sedang", "tidak layak"),
    ("R9", "rendah", "tinggi", "banyak", "tinggi", "cukup layak"),
    ("R10", "rendah", "sedang", "sedikit", "rendah", "tidak layak"),
    ("R11", "rendah", "sedang", "sedikit", "sedang", "tidak layak"),
    ("R12", "rendah", "sedang", "sedikit", "tinggi", "tidak layak"),
    ("R13", "rendah", "sedang", "sedang", "rendah", "tidak layak"),
    ("R14", "rendah", "sedang", "sedang", "sedang", "tidak layak"),
    ("R15", "rendah", "sedang", "sedang", "tinggi", "tidak layak"),
    ("R16", "rendah", "sedang", "banyak", "rendah", "tidak layak"),
    ("R17", "rendah", "sedang", "banyak", "sedang", "tidak layak"),
    ("R18", "rendah", "sedang", "banyak", "tinggi", "cukup layak"),
    ("R19", "rendah", "rendah", "sedikit", "rendah", "tidak layak"),
    ("R20", "rendah", "rendah", "sedikit", "sedang", "cukup layak"),
    ("R21", "rendah", "rendah", "sedikit", "tinggi", "cukup layak"),
    ("R22", "rendah", "rendah", "sedang", "rendah", "cukup layak"),
    ("R23", "rendah", "rendah", "sedang", "sedang", "cukup layak"),
    ("R24", "rendah", "rendah", "sedang", "tinggi", "cukup layak"),
    ("R25", "rendah", "rendah", "banyak", "rendah", "cukup layak"),
    ("R26", "rendah", "rendah", "banyak", "sedang", "cukup layak"),
    ("R27", "rendah", "rendah", "banyak", "tinggi", "cukup layak"),
    ("R28", "sedang", "tinggi", "sedikit", "rendah", "tidak layak"),
    ("R29", "sedang", "tinggi", "sedikit", "sedang", "tidak layak"),
    ("R30", "sedang", "tinggi", "sedikit", "tinggi", "tidak layak"),
    ("R31", "sedang", "tinggi", "sedang", "rendah", "tidak layak"),
    ("R32", "sedang", "tinggi", "sedang", "sedang", "tidak layak"),
    ("R33", "sedang", "tinggi", "sedang", "tinggi", "tidak layak"),
    ("R34", "sedang", "tinggi", "banyak", "rendah", "tidak layak"),
    ("R35", "sedang", "tinggi", "banyak", "sedang", "tidak layak"),
    ("R36", "sedang", "tinggi", "banyak", "tinggi", "cukup layak"),
    ("R37", "sedang", "sedang", "sedikit", "rendah", "tidak layak"),
    ("R38", "sedang", "sedang", "sedikit", "sedang", "cukup layak"),
    ("R39", "sedang", "sedang", "sedikit", "tinggi", "cukup layak"),
    ("R40", "sedang", "sedang", "sedang", "rendah", "cukup layak"),
    ("R41", "sedang", "sedang", "sedang", "sedang", "cukup layak"),
    ("R42", "sedang", "sedang", "sedang", "tinggi", "cukup layak"),
    ("R43", "sedang", "sedang", "banyak", "rendah", "cukup layak"),
    ("R44", "sedang", "sedang", "banyak", "sedang", "cukup layak"),
    ("R45", "sedang", "sedang", "banyak", "tinggi", "layak"),
    ("R46", "sedang", "rendah", "sedikit", "rendah", "cukup layak"),
    ("R47", "sedang", "rendah", "sedikit", "sedang", "layak"),
    ("R48", "sedang", "rendah", "sedikit", "tinggi", "layak"),
    ("R49", "sedang", "rendah", "sedang", "rendah", "layak"),
    ("R50", "sedang", "rendah", "sedang", "sedang", "layak"),
    ("R51", "sedang", "rendah", "sedang", "tinggi", "layak"),
    ("R52", "sedang", "rendah", "banyak", "rendah", "layak"),
    ("R53", "sedang", "rendah", "banyak", "sedang", "layak"),
    ("R54", "sedang", "rendah", "banyak", "tinggi", "sangat layak"),
    ("R55", "tinggi", "tinggi", "sedikit", "rendah", "tidak layak"),
    ("R56", "tinggi", "tinggi", "sedikit", "sedang", "cukup layak"),
    ("R57", "tinggi", "tinggi", "sedikit", "tinggi", "cukup layak"),
    ("R58", "tinggi", "tinggi", "sedang", "rendah", "cukup layak"),
    ("R59", "tinggi", "tinggi", "sedang", "sedang", "cukup layak"),
    ("R60", "tinggi", "tinggi", "sedang", "tinggi", "cukup layak"),
    ("R61", "tinggi", "tinggi", "banyak", "rendah", "cukup layak"),
    ("R62", "tinggi", "tinggi", "banyak", "sedang", "cukup layak"),
    ("R63", "tinggi", "tinggi", "banyak", "tinggi", "layak"),
    ("R64", "tinggi", "sedang", "sedikit", "rendah", "cukup layak"),
    ("R65", "tinggi", "sedang", "sedikit", "sedang", "layak"),
    ("R66", "tinggi", "sedang", "sedikit", "tinggi", "layak"),
    ("R67", "tinggi", "sedang", "sedang", "rendah", "layak"),
    ("R68", "tinggi", "sedang", "sedang", "sedang", "layak"),
    ("R69", "tinggi", "sedang", "sedang", "tinggi", "layak"),
    ("R70", "tinggi", "sedang", "banyak", "rendah", "layak"),
    ("R71", "tinggi", "sedang", "banyak", "sedang", "layak"),
    ("R72", "tinggi", "sedang", "banyak", "tinggi", "sangat layak"),
    ("R73", "tinggi", "rendah", "sedikit", "rendah", "layak"),
    ("R74", "tinggi", "rendah", "sedikit", "sedang", "sangat layak"),
    ("R75", "tinggi", "rendah", "sedikit", "tinggi", "sangat layak"),
    ("R76", "tinggi", "rendah", "sedang", "rendah", "sangat layak"),
    ("R77", "tinggi", "rendah", "sedang", "sedang", "sangat layak"),
    ("R78", "tinggi", "rendah", "sedang", "tinggi", "sangat layak"),
    ("R79", "tinggi", "rendah", "banyak", "rendah", "sangat layak"),
    ("R80", "tinggi", "rendah", "banyak", "sedang", "sangat layak"),
    ("R81", "tinggi", "rendah", "banyak", "tinggi", "sangat layak"),
]
KOLOM_ATURAN = ["No", "IPK", "Penghasilan", "Tanggungan", "Prestasi", "Kelayakan"]
TANDA_BEBAS = {"–", "-", "—", "", "nan"}

# Data contoh: ID, IPK, Penghasilan, Tanggungan, Prestasi
DATA_CONTOH = [
    ("M1", 4.00, 0.0, 8, 100), ("M2", 0.00, 15.0, 0, 0), ("M3", 4.00, 15.0, 0, 100),
    ("M4", 0.00, 0.0, 8, 0), ("M5", 3.00, 2.5, 1, 30), ("M6", 3.25, 5.5, 4, 50),
    ("M7", 3.75, 11.0, 7, 80), ("M8", 2.75, 9.0, 2, 75), ("M9", 3.50, 3.9, 6, 60),
    ("M10", 2.50, 7.5, 3, 40), ("M11", 2.90, 3.2, 2, 40), ("M12", 3.50, 8.0, 6, 67.5),
    ("M13", 3.40, 6.5, 4, 55), ("M14", 3.30, 3.5, 6, 85), ("M15", 2.90, 5.5, 6, 70),
    ("M16", 3.60, 10.0, 7, 70), ("M17", 3.90, 12.0, 1, 90), ("M18", 2.20, 1.5, 7, 20),
    ("M19", 3.80, 1.0, 1, 25), ("M20", 3.20, 6.0, 8, 90), ("M21", 3.85, 5.5, 0, 95),
    ("M22", 2.60, 14.0, 7, 85), ("M23", 3.80, 12.0, 0, 15), ("M24", 3.70, 5.5, 1, 20),
]

# =====================================================================
# 3. MESIN INFERENSI MAMDANI
# =====================================================================
INTERVAL = 5                                     # interval grid output (sama dengan perhitungan manual)
Z = np.arange(0, 100 + INTERVAL / 2, INTERVAL)   # 0, 5, 10, ..., 100 (21 titik)
MU_OUT = {n: trapesium(Z, *p) for n, p in VAR["Kelayakan"]["s"].items()}

def fuzzifikasi(var, x):
    return {nama: float(trapesium(x, *p)) for nama, p in VAR[var]["s"].items()}

def susun_aturan(df):
    """Ubah tabel aturan menjadi daftar aturan IF-THEN (sekaligus memvalidasi nama himpunan)."""
    hasil = []
    for _, r in df.iterrows():
        anteseden = {}
        for v in INPUT:
            nilai = str(r[v]).strip().lower()
            if nilai not in TANDA_BEBAS:
                if nilai not in VAR[v]["s"]:
                    raise ValueError(f"Aturan {r['No']}: himpunan '{nilai}' tidak ada pada variabel {v}")
                anteseden[v] = nilai
        if not anteseden:
            raise ValueError(f"Aturan {r['No']}: tidak ada satu pun syarat yang dipakai")
        konsekuen = str(r["Kelayakan"]).strip().lower()
        if konsekuen not in VAR["Kelayakan"]["s"]:
            raise ValueError(f"Aturan {r['No']}: kategori '{konsekuen}' tidak dikenal")
        hasil.append({"No": str(r["No"]), "if": anteseden, "then": konsekuen})
    return hasil

def mamdani(ipk, penghasilan, tanggungan, prestasi, aturan):
    # 1. Fuzzifikasi
    nilai = dict(zip(INPUT, [ipk, penghasilan, tanggungan, prestasi]))
    mu = {v: fuzzifikasi(v, nilai[v]) for v in INPUT}

    agregasi = np.zeros_like(Z)
    aktif = []
    for a in aturan:
        # 2. alpha-predikat: AND = MIN (variabel bertanda "-" tidak ikut)
        derajat = {v: mu[v][h] for v, h in a["if"].items()}
        alpha = min(derajat.values())
        if alpha > 0:
            # 3. Implikasi = MIN (potong himpunan output setinggi alpha)
            potongan = np.minimum(alpha, MU_OUT[a["then"]])
            # 4. Agregasi = MAX
            agregasi = np.maximum(agregasi, potongan)
            aktif.append({"No": a["No"], "if": a["if"], "Output": a["then"], "alpha": alpha, "derajat": derajat})

    if agregasi.sum() == 0:
        raise ValueError("Tidak ada aturan yang aktif untuk input ini (ada celah pada basis aturan).")

    # 5. Defuzzifikasi centroid diskret
    pembilang = float((agregasi * Z).sum())
    penyebut = float(agregasi.sum())
    return {"skor": pembilang / penyebut, "pembilang": pembilang, "penyebut": penyebut,
            "mu": mu, "aktif": aktif, "agregasi": agregasi}

def kategori(skor):
    m = fuzzifikasi("Kelayakan", skor)
    return max(m, key=m.get)

# =====================================================================
# 4. FUNGSI BANTU (BACA CSV DAN TEKS)
# =====================================================================
def baca_csv(isi_bytes):
    for enc in ("utf-8-sig", "cp1252"):
        try:
            teks = isi_bytes.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("Encoding file tidak dikenali")
    baris1 = teks.splitlines()[0]
    sep = ";" if baris1.count(";") >= baris1.count(",") else ","
    df = pd.read_csv(io.StringIO(teks), sep=sep, dtype=str)
    df.columns = df.columns.str.strip()
    return df

def ke_angka(df, kolom):
    for k in kolom:
        df[k] = pd.to_numeric(df[k].astype(str).str.strip().str.replace(",", ".", regex=False))
    return df

def kondisi_teks(anteseden):
    return " DAN ".join(f"{v} {h}" for v, h in anteseden.items())