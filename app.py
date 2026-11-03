import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from fuzzy_beasiswa import (
    VAR, INPUT, ATURAN_BAWAAN, KOLOM_ATURAN, DATA_CONTOH, INTERVAL, Z, MU_OUT,
    trapesium, fuzzifikasi, susun_aturan, mamdani, kategori,
    baca_csv, ke_angka, kondisi_teks,
)

st.set_page_config(page_title="SPK Beasiswa - Fuzzy Mamdani", layout="wide")

# =====================================================================
# 1. PENGATURAN INPUT SLIDER
# =====================================================================
# Pengaturan input slider: (label, min, max, nilai awal, langkah, format)
SLIDER = {
    "IPK":         ("IPK (0–4)",                                  0.0, 4.0,   3.25, 0.01, "%.2f"),
    "Penghasilan": ("Penghasilan orang tua (juta Rp/bulan, 0–15)", 0.0, 15.0,  8.0,  0.1,  "%.1f"),
    "Tanggungan":  ("Jumlah tanggungan (orang, 0–8)",             0,   8,     5,    1,    "%d"),
    "Prestasi":    ("Skor prestasi (0–100)",                      0.0, 100.0, 70.0, 0.5,  "%.1f"),
}

# =====================================================================
# 2. FUNGSI BANTU TAMPILAN (grafik dan tabel)
# =====================================================================
def grafik_agregasi(hasil, judul="Hasil agregasi dan centroid"):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for n, m in MU_OUT.items():
        ax.plot(Z, m, "--", alpha=.45, label=n)
    ax.fill_between(Z, hasil["agregasi"], alpha=.5, color="tab:orange", label="Agregasi")
    ax.axvline(hasil["skor"], color="red", label=f"Centroid = {hasil['skor']:.2f}")
    ax.set_xlabel("Skor kelayakan"); ax.set_ylabel("µ"); ax.set_title(judul)
    ax.grid(alpha=.3); ax.legend(fontsize=7)
    fig.tight_layout()
    return fig

NAMA = {"IPK": "IPK", "Penghasilan": "Penghasilan orang tua", "Tanggungan": "Jumlah tanggungan",
        "Prestasi": "Skor prestasi"}
JUDUL = {"IPK": "IPK", "Penghasilan": "Penghasilan orang tua", "Tanggungan": "Jumlah tanggungan",
         "Prestasi": "Prestasi mahasiswa", "Kelayakan": "Kelayakan beasiswa (keluaran)"}
SUMBU_X = {"IPK": "IPK (skala 0-4)", "Penghasilan": "Penghasilan (juta Rp/bulan)",
           "Tanggungan": "Tanggungan (orang)", "Prestasi": "Prestasi (skor 0-100)",
           "Kelayakan": "Skor kelayakan (0-100)"}
GAYA = ["-", "--", "-.", ":"]
WARNA = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

def grafik_keanggotaan(var, titik=None):
    """Grafik fungsi keanggotaan; 'titik' = nilai input saat ini (garis vertikal hitam)."""
    d = VAR[var]
    xs = np.linspace(d["u"][0], d["u"][1], 1001)
    lebar = var == "Kelayakan"
    fig, ax = plt.subplots(figsize=(13, 3.6) if lebar else (6.2, 3.6))
    for (n, p), g, w in zip(d["s"].items(), GAYA, WARNA):
        ax.plot(xs, trapesium(xs, *p), g, color=w, lw=2.2, label=n.capitalize())
    if titik is not None:
        ax.axvline(titik, color="black", lw=1.6)
    ax.set_xlim(*d["u"]); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(SUMBU_X[var]); ax.set_ylabel("Derajat keanggotaan")
    ax.grid(alpha=.3)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(d["s"]), frameon=False)
    fig.tight_layout()
    return fig

def tampil_grafik(fig):
    st.pyplot(fig)
    plt.close(fig)          # cegah penumpukan figure setiap kali slider digeser

def tabel_fuzzifikasi(h, nilai):
    return pd.DataFrame([{"Variabel": NAMA[v], "Nilai input": nilai[v], "Himpunan": himp.title(),
                          "Derajat keanggotaan (µ)": round(mu, 4)}
                         for v in INPUT for himp, mu in h["mu"][v].items()])

def tabel_alpha(h):
    baris = []
    for a in h["aktif"]:
        syarat = [f"µ {v} {k} = {a['derajat'][v]:.4f}" for v, k in a["if"].items()]
        angka = ", ".join(f"{a['derajat'][v]:.4f}" for v in a["if"])
        baris.append({"Aturan": a["No"], "Kondisi (JIKA)": kondisi_teks(a["if"]),
                      "Derajat tiap syarat": "; ".join(syarat),
                      "α = MIN": f"min({angka}) = {a['alpha']:.4f}",
                      "Kesimpulan (MAKA)": a["Output"].title()})
    return pd.DataFrame(baris)

def tabel_implikasi(h):
    return pd.DataFrame([{"Aturan": a["No"], "Himpunan output": a["Output"].title(),
                          "Dipotong setinggi α": round(a["alpha"], 4)} for a in h["aktif"]])

def tabel_agregasi(h):
    baris = []
    for n in VAR["Kelayakan"]["s"]:
        cocok = [a for a in h["aktif"] if a["Output"] == n]
        baris.append({"Himpunan output": n.title(),
                      "α tertinggi (MAX)": round(max((a["alpha"] for a in cocok), default=0.0), 4),
                      "Berasal dari aturan": ", ".join(a["No"] for a in cocok) if cocok else "-"})
    return pd.DataFrame(baris)

# ---- sinkronisasi slider <-> kotak angka ----
def dari_slider(k):
    st.session_state[f"{k}_num"] = st.session_state[f"{k}_sld"]

def dari_kotak(k):
    st.session_state[f"{k}_sld"] = st.session_state[f"{k}_num"]

def input_geser(k):
    """Slider + kotak angka yang saling terhubung. Mengembalikan nilai saat ini."""
    label, lo, hi, awal, langkah, fmt = SLIDER[k]
    if f"{k}_sld" not in st.session_state:
        st.session_state[f"{k}_sld"] = awal
        st.session_state[f"{k}_num"] = awal
    st.sidebar.slider(label, lo, hi, step=langkah, format=fmt, key=f"{k}_sld",
                      on_change=dari_slider, args=(k,))
    st.sidebar.number_input(label, lo, hi, step=langkah, format=fmt, key=f"{k}_num",
                            on_change=dari_kotak, args=(k,), label_visibility="collapsed")
    return st.session_state[f"{k}_sld"]

# =====================================================================
# 3. ANTARMUKA
# =====================================================================
st.title("SPK Kelayakan Penerima Beasiswa")
st.caption("Sistem pendukung keputusan berbasis Logika Fuzzy metode Mamdani. Hasilnya berupa rekomendasi, bukan keputusan final.")

# ---- Sidebar: input pendaftar (slider di samping) ----
st.sidebar.header("Masukkan data pendaftar")
st.sidebar.caption("Geser slider atau ketik nilai langsung pada kotak di bawahnya.")
ipk = input_geser("IPK")
peng = input_geser("Penghasilan")
tang = input_geser("Tanggungan")
pres = input_geser("Prestasi")

# ---- Sidebar: pilihan basis aturan ----
st.sidebar.divider()
st.sidebar.header("Basis Aturan")
sumber_aturan = st.sidebar.radio("Sumber aturan", ["Bawaan (81 aturan)", "Upload CSV sendiri"])
df_aturan = pd.DataFrame(ATURAN_BAWAAN, columns=KOLOM_ATURAN)
if sumber_aturan == "Upload CSV sendiri":
    f_aturan = st.sidebar.file_uploader("File basis aturan (.csv)", type="csv", key="aturan")
    st.sidebar.caption("Kolom: No, IPK, Penghasilan, Tanggungan, Prestasi, Kelayakan. Tanda '-' = variabel tidak dipakai.")
    if f_aturan is not None:
        try:
            tmp = baca_csv(f_aturan.getvalue())
            assert set(KOLOM_ATURAN) <= set(tmp.columns), "Kolom file aturan tidak lengkap"
            df_aturan = tmp[KOLOM_ATURAN]
        except Exception as e:
            st.sidebar.error(f"File aturan tidak bisa dibaca: {e}")

try:
    ATURAN = susun_aturan(df_aturan)
except Exception as e:
    st.error(f"Basis aturan tidak valid: {e}")
    st.stop()
st.sidebar.success(f"{len(ATURAN)} aturan dimuat")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["1. Hitung Satu Mahasiswa", "2. Hitung Banyak Data", "3. Fungsi Keanggotaan", "4. Basis Aturan", "5. Tentang Metode"])

# ---------------- TAB 1: satu mahasiswa (otomatis mengikuti slider) ----------------
with tab1:
    st.subheader("Hasil perhitungan")
    nilai = dict(zip(INPUT, [ipk, peng, tang, pres]))
    try:
        h = mamdani(ipk, peng, tang, pres, ATURAN)
    except ValueError as e:
        st.error(str(e))
    else:
        kat = kategori(h["skor"])
        m1, m2, m3 = st.columns(3)
        m1.metric("Skor kelayakan", f"{h['skor']:.2f}")
        m2.metric("Rekomendasi", kat.title())
        m3.metric("Aturan aktif", f"{len(h['aktif'])} dari {len(ATURAN)}")

        st.markdown("## Rincian perhitungan")

        # ---- 1. Fuzzifikasi ----
        st.markdown("### 1. Fuzzifikasi")
        st.markdown(
            "Fuzzifikasi mengubah nilai tegas (angka yang Anda masukkan) menjadi **derajat keanggotaan (µ)** "
            "antara 0 sampai 1 pada setiap himpunan fuzzy. Nilai µ dihitung dari fungsi keanggotaan trapesium "
            "pada tab *3. Fungsi Keanggotaan*. Nilai µ = 0 berarti nilai input tidak termasuk himpunan tersebut, "
            "sedangkan µ = 1 berarti termasuk sepenuhnya.")
        st.dataframe(tabel_fuzzifikasi(h, nilai), hide_index=True)

        # ---- 2. Alpha-predikat ----
        st.markdown("### 2. Penentuan aturan aktif dan α-predikat")
        st.markdown(
            "Setiap aturan berbentuk *JIKA IPK ... DAN Penghasilan ... DAN Tanggungan ... DAN Prestasi ... "
            "MAKA Kelayakan ...*. Operator **DAN** dihitung dengan **MIN**, sehingga kekuatan aturan "
            "(**α-predikat**) sama dengan derajat keanggotaan terkecil di antara keempat syaratnya. "
            "Aturan disebut **aktif** bila α > 0. Dari "
            f"{len(ATURAN)} aturan, **{len(h['aktif'])} aturan aktif**; aturan lainnya tidak aktif karena "
            "sedikitnya satu syaratnya bernilai µ = 0.")
        st.dataframe(tabel_alpha(h), hide_index=True)

        # ---- 3. Implikasi ----
        st.markdown("### 3. Implikasi (MIN)")
        st.markdown(
            "Implikasi menentukan seberapa besar himpunan output pada kesimpulan aturan dipakai. "
            "Dengan **MIN**, himpunan output **dipotong** (dipangkas) setinggi nilai α aturan tersebut. "
            "Semakin besar α, semakin tinggi bagian himpunan output yang dipertahankan.")
        st.dataframe(tabel_implikasi(h), hide_index=True)

        # ---- 4. Agregasi ----
        st.markdown("### 4. Agregasi (MAX)")
        st.markdown(
            "Agregasi menggabungkan semua hasil implikasi menjadi **satu daerah fuzzy** dengan operator "
            "**MAX**: pada setiap titik skor, diambil nilai µ tertinggi dari semua himpunan yang sudah "
            "dipotong. Untuk beberapa aturan yang berkesimpulan sama, yang berlaku adalah α terbesar.")
        st.dataframe(tabel_agregasi(h), hide_index=True)

        # ---- 5. Defuzzifikasi ----
        st.markdown("### 5. Defuzzifikasi (centroid)")
        st.markdown(
            "Daerah hasil agregasi diubah kembali menjadi **satu angka tegas** dengan metode **centroid**, "
            "yaitu titik keseimbangan daerah tersebut. Perhitungan dilakukan secara diskret pada grid skor "
            f"0 sampai 100 dengan interval {INTERVAL}, jadi z* = jumlah (µ × z) dibagi jumlah µ.")
        st.latex(r"z^{*}=\frac{\sum \mu_{agr}(z_i)\,z_i}{\sum \mu_{agr}(z_i)}"
                 rf"=\frac{{{h['pembilang']:.4f}}}{{{h['penyebut']:.4f}}}={h['skor']:.4f}")
        tampil_grafik(grafik_agregasi(h))

        # ---- 6. Kesimpulan ----
        st.markdown("### 6. Kesimpulan")
        mu_z = fuzzifikasi("Kelayakan", h["skor"])
        st.markdown(
            f"Skor kelayakan akhir adalah **{h['skor']:.2f}**. Skor ini dimasukkan kembali ke himpunan output "
            "untuk menentukan kategori; kategori dengan derajat keanggotaan tertinggi dipilih sebagai rekomendasi.")
        st.dataframe(pd.DataFrame([{"Himpunan output": n.title(), "µ pada skor akhir": round(m, 4)}
                                   for n, m in mu_z.items()]), hide_index=True)
        st.success(f"Rekomendasi: {kat.title()} (skor {h['skor']:.2f}). "
                   "Hasil ini hanya rekomendasi pendukung keputusan, bukan keputusan final.")

# ---------------- TAB 2: banyak data ----------------
with tab2:
    st.subheader("Hitung banyak mahasiswa sekaligus")
    pilihan = st.radio("Sumber data", ["Data contoh (24 mahasiswa)", "Upload CSV sendiri"], horizontal=True)
    df_data = None
    if pilihan.startswith("Data contoh"):
        df_data = pd.DataFrame(DATA_CONTOH, columns=["ID"] + INPUT)
    else:
        st.caption("Kolom file: ID, IPK, Penghasilan, Tanggungan, Prestasi. Pemisah `;` atau `,` dan koma desimal dikenali otomatis.")
        f_data = st.file_uploader("File data mahasiswa (.csv)", type="csv", key="data")
        if f_data is not None:
            try:
                tmp = baca_csv(f_data.getvalue())
                assert {"ID"} | set(INPUT) <= set(tmp.columns), "Kolom file data tidak lengkap (ID, IPK, Penghasilan, Tanggungan, Prestasi)"
                df_data = ke_angka(tmp[["ID"] + INPUT].copy(), INPUT)
            except Exception as e:
                st.error(f"File data tidak bisa dibaca: {e}")

    if df_data is not None:
        baris, gagal = [], []
        for r in df_data.itertuples(index=False):
            try:
                h = mamdani(r.IPK, r.Penghasilan, r.Tanggungan, r.Prestasi, ATURAN)
                baris.append({"ID": r.ID, "IPK": r.IPK, "Penghasilan": r.Penghasilan, "Tanggungan": r.Tanggungan,
                              "Prestasi": r.Prestasi, "Aturan aktif": len(h["aktif"]),
                              "No aturan": ", ".join(a["No"] for a in h["aktif"]),
                              "Skor": round(h["skor"], 2), "Kategori": kategori(h["skor"])})
            except ValueError:
                gagal.append(str(r.ID))
        if gagal:
            st.warning("Tidak ada aturan aktif untuk: " + ", ".join(gagal))
        if baris:
            df_hasil = pd.DataFrame(baris).sort_values("Skor", ascending=False)
            st.dataframe(df_hasil, hide_index=True)
            ringkas = df_hasil["Kategori"].value_counts().rename_axis("Kategori").reset_index(name="Jumlah")
            st.markdown("**Ringkasan kategori**")
            st.dataframe(ringkas, hide_index=True)
            st.download_button("Unduh hasil (CSV)",
                               df_hasil.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                               file_name="hasil_kelayakan_beasiswa.csv", mime="text/csv")

# ---------------- TAB 3: fungsi keanggotaan ----------------
with tab3:
    st.subheader("Fungsi keanggotaan")
    st.caption("Semua himpunan berbentuk trapesium (a, b, c, d). Segitiga adalah kasus b = c. "
               "Garis hitam menunjukkan nilai input saat ini.")
    nilai_kini = dict(zip(INPUT, [ipk, peng, tang, pres]))
    kolom = st.columns(2)
    for i, v in enumerate(INPUT):
        with kolom[i % 2]:
            st.markdown(f"**{JUDUL[v]}**")
            tampil_grafik(grafik_keanggotaan(v, nilai_kini[v]))
    st.markdown(f"**{JUDUL['Kelayakan']}**")
    tampil_grafik(grafik_keanggotaan("Kelayakan"))
    param = [{"Variabel": v, "Himpunan": n, "Parameter (a, b, c, d)": str(p)}
             for v, d in VAR.items() for n, p in d["s"].items()]
    st.dataframe(pd.DataFrame(param), hide_index=True)

# ---------------- TAB 4: basis aturan ----------------
with tab4:
    st.subheader("Basis aturan")
    st.caption("Tanda '-' berarti variabel itu tidak dipakai pada aturan tersebut.")
    st.dataframe(df_aturan, hide_index=True)

# ---------------- TAB 5: tentang metode ----------------
with tab5:
    st.subheader("Tentang metode")
    st.markdown("""
Sistem ini memakai **Logika Fuzzy metode Mamdani** dengan empat input dan satu output.

| Tahap | Operasi | Penjelasan sederhana |
|---|---|---|
| 1 | Fuzzifikasi | angka diubah menjadi "seberapa" masuk tiap kategori (0 sampai 1) |
| 2 | α-predikat | kekuatan sebuah aturan = syarat terlemahnya (**MIN**) |
| 3 | Implikasi | kategori hasil dipotong setinggi kekuatan aturan (**MIN**) |
| 4 | Agregasi | semua potongan digabung, ambil yang tertinggi (**MAX**) |
| 5 | Defuzzifikasi | titik seimbang bidang gabungan (**centroid**) menjadi skor 0 sampai 100 |

**Variabel input:** IPK, penghasilan orang tua, jumlah tanggungan, dan prestasi.
**Output:** skor kelayakan dengan kategori tidak layak, cukup layak, layak, dan sangat layak.

**Basis aturan:** 81 aturan, yaitu seluruh kombinasi 3 × 3 × 3 × 3 himpunan input, sehingga setiap
input valid pasti mengaktifkan minimal satu aturan.

**Keterbatasan:** parameter fungsi keanggotaan dan aturan ditentukan manual (subjektif), skor tidak
pernah mencapai tepat 0 atau 100, dan hasilnya hanya rekomendasi pendukung keputusan.
""")