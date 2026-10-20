"""
Studi Kasus : Make Sense of Data - Prediksi Keterlambatan Pengiriman Paket
Kerangka    : CRISP-DM (6 tahap) + 4 tingkat analitik + data traps
Materi      : Informatics Practitioner Lecture Series
              "Make Sense of Data with Analytics and AI"

Struktur kode mengikuti CRISP-DM:
  1. Business Understanding   4. Modeling (Predictive)
  2. Data Understanding        5. Evaluation
  3. Data Preparation          6. Deployment (+ Prescriptive, Human-in-the-loop)
Analitik Deskriptif & Diagnostik dikerjakan di tahap 3-4 (eksplorasi).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

os.makedirs("output", exist_ok=True)
RNG = np.random.default_rng(2025)
garis = lambda t: print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


# ======================================================================
# SIMULASI DATA INDUSTRI (berantakan, seperti dunia nyata)
# ======================================================================
def simulasi_data(n=3000):
    kota_info = {"Jakarta": 15, "Bandung": 150, "Surabaya": 780,
                 "Medan": 1400, "Makassar": 1300, "Padang": 950}
    kota = RNG.choice(list(kota_info), n, p=[.30, .18, .18, .12, .12, .10])
    tanggal = pd.to_datetime("2025-01-01") + pd.to_timedelta(RNG.integers(0, 181, n), "D")
    jarak = np.array([kota_info[k] for k in kota]) * RNG.uniform(.9, 1.1, n)
    berat = RNG.gamma(2, 1.5, n).round(1) + 0.2
    layanan = RNG.choice(["Express", "Reguler"], n, p=[.35, .65])
    jam_order = RNG.integers(7, 22, n)
    kurir_id = RNG.integers(1, 41, n)

    # tabel kurir (ada duplikat entri -> jebakan join)
    kurir = pd.DataFrame({"kurir_id": np.arange(1, 41),
                          "pengalaman_bulan": RNG.integers(1, 49, 40),
                          "rating": RNG.uniform(3, 5, 40).round(2)})
    kurir.loc[[3, 11, 25], "rating"] = np.nan
    kurir = pd.concat([kurir, kurir.iloc[[5, 17, 30]]], ignore_index=True)

    # tabel cuaca (3 pembacaan per kota per hari -> jebakan join)
    hari = pd.date_range("2025-01-01", periods=181)
    cuaca = pd.DataFrame([(h, k, RNG.exponential(6)) for h in hari
                          for k in kota_info for _ in range(3)],
                         columns=["tanggal", "kota", "curah_hujan_mm"])
    maks_hujan = cuaca.groupby(["tanggal", "kota"])["curah_hujan_mm"].max()
    hujan_nyata = np.array([maks_hujan[(t, k)] >= 20 for t, k in zip(tanggal, kota)])

    # proses pembangkit label (kebenaran tersembunyi)
    peng = kurir.drop_duplicates("kurir_id").set_index("kurir_id")["pengalaman_bulan"]
    pengalaman = np.array([peng[i] for i in kurir_id])
    z = (-3.4 + 0.0011 * jarak + 1.3 * hujan_nyata + 0.8 * (layanan == "Reguler")
         + 0.7 * (jam_order >= 15) - 0.03 * pengalaman + 0.12 * berat)
    terlambat = (RNG.random(n) < 1 / (1 + np.exp(-z))).astype(int)

    sla = np.where(layanan == "Express", 24, 72)
    lama = np.where(terlambat == 1, sla * RNG.uniform(1.05, 1.6, n),
                    sla * RNG.uniform(.5, .95, n)).round(1)

    orders = pd.DataFrame({"order_id": np.arange(1, n + 1), "tanggal": tanggal,
                           "kota": kota, "jarak_km": jarak.round(0), "berat_kg": berat,
                           "layanan": layanan, "jam_order": jam_order,
                           "kurir_id": kurir_id, "lama_kirim_jam": lama,
                           "terlambat": terlambat})

    # KOTOR 1: penulisan kota tidak konsisten
    idx = orders.sample(frac=.10, random_state=1).index
    orders.loc[idx, "kota"] = orders.loc[idx, "kota"].str.upper() + " "
    # KOTOR 2: nilai hilang
    orders.loc[orders.sample(frac=.04, random_state=2).index, "berat_kg"] = np.nan
    # KOTOR 3: baris duplikat
    orders = pd.concat([orders, orders.sample(40, random_state=3)], ignore_index=True)
    # KOTOR 4: ANOMALI SISTEM - migrasi server 10-11 Maret 2025
    # semua paket tercatat "tepat waktu" dengan lama kirim 0 jam (bukan kenyataan)
    anom = orders["tanggal"].between("2025-03-10", "2025-03-11")
    orders.loc[anom, ["lama_kirim_jam", "terlambat"]] = [0, 0]
    return orders, kurir, cuaca


# ======================================================================
# 1. BUSINESS UNDERSTANDING
# ======================================================================
garis("1. BUSINESS UNDERSTANDING")
print("Pernyataan manajemen : 'Pelanggan kami sering komplain karena paket telat.'")
print("Rumusan analitik     :")
print("  - Deskriptif   : Seberapa sering paket terlambat? Di mana?")
print("  - Diagnostik   : Faktor apa yang menyebabkan keterlambatan?")
print("  - Prediktif    : Paket mana yang berisiko terlambat SEBELUM dikirim?")
print("  - Preskriptif  : Intervensi apa & untuk paket mana agar biaya paling hemat?")
print("Metrik sukses        : Recall & F1 kelas 'terlambat' (BUKAN akurasi), lalu total biaya.")

orders, kurir, cuaca = simulasi_data()

# ======================================================================
# 2. DATA UNDERSTANDING
# ======================================================================
garis("2. DATA UNDERSTANDING")
print("orders:", orders.shape, "| kurir:", kurir.shape, "| cuaca:", cuaca.shape)
print("\nNilai hilang (orders):\n", orders.isna().sum()[lambda s: s > 0])
print("\nEjaan kota apa adanya:", sorted(orders["kota"].unique()))
print("Baris duplikat order_id:", orders.duplicated("order_id").sum())
nol = orders[orders["lama_kirim_jam"] == 0].groupby(orders["tanggal"].dt.date).size()
print("\nTanggal dengan lama_kirim_jam = 0 (mencurigakan!):\n", nol)

# ======================================================================
# 3. DATA PREPARATION  (porsi terbesar kerja analis: 80-90% waktu)
# ======================================================================
garis("3. DATA PREPARATION")
awal = len(orders)

# 3a. standarisasi teks
orders["kota"] = orders["kota"].str.strip().str.title()
print(f"[a] Ejaan kota distandarkan -> {orders['kota'].nunique()} kota unik")

# 3b. hapus duplikat
orders = orders.drop_duplicates("order_id").reset_index(drop=True)
print(f"[b] Duplikat dihapus: {awal} -> {len(orders)} baris")

# 3c. DATA TRAP #1 - join berlebihan memicu duplikasi
cuaca["tanggal"] = pd.to_datetime(cuaca["tanggal"])
join_naif = orders.merge(cuaca, on=["tanggal", "kota"], how="left")
print(f"[c] JEBAKAN JOIN: join langsung cuaca -> {len(join_naif)} baris "
      f"(seharusnya {len(orders)})  => data membengkak {len(join_naif)/len(orders):.1f}x!")
cuaca_harian = (cuaca.groupby(["tanggal", "kota"], as_index=False)
                .agg(hujan_maks_mm=("curah_hujan_mm", "max")))
kurir_bersih = kurir.drop_duplicates("kurir_id")
df = (orders.merge(cuaca_harian, on=["tanggal", "kota"], how="left")
            .merge(kurir_bersih, on="kurir_id", how="left"))
assert len(df) == len(orders), "Join menambah baris!"
print(f"    Solusi: agregasi dulu sebelum join -> {len(df)} baris (aman)")

# 3d. DATA TRAP #2 - anomali sistem, butuh pengetahuan domain
mask_anom = df["tanggal"].between("2025-03-10", "2025-03-11")
print(f"[d] ANOMALI DOMAIN: {mask_anom.sum()} paket pada 10-11 Maret dicatat "
      "'tepat waktu, 0 jam' akibat migrasi server -> dibuang (bukan kenyataan).")
df = df[~mask_anom].reset_index(drop=True)

# 3e. nilai hilang
df["berat_kg"] = df["berat_kg"].fillna(df["berat_kg"].median())
df["rating"] = df["rating"].fillna(df["rating"].median())

# 3f. rekayasa fitur
df["hujan_deras"] = (df["hujan_maks_mm"] >= 20).astype(int)
df["akhir_pekan"] = (df["tanggal"].dt.dayofweek >= 5).astype(int)
df["lewat_cutoff"] = (df["jam_order"] >= 15).astype(int)
print(f"[e-f] Nilai hilang ditangani, fitur baru dibuat. Data siap: {df.shape}")

# 3g. DATA TRAP #3 - ketidakseimbangan kelas
rasio = df["terlambat"].mean()
print(f"[g] JEBAKAN IMBALANCE: paket terlambat hanya {rasio:.1%}. "
      f"Model yang selalu menebak 'tepat waktu' akurasinya {1-rasio:.1%} tapi TIDAK berguna.")

# 3h. DATA TRAP #4 - kebocoran data
print("[h] KEBOCORAN: kolom 'lama_kirim_jam' baru diketahui SETELAH paket sampai "
      "-> tidak boleh jadi fitur.")

# ---------- ANALITIK DESKRIPTIF ----------
garis("ANALITIK 1 - DESKRIPTIF (apa yang terjadi?)")
print(f"Tingkat keterlambatan keseluruhan: {rasio:.1%}")
print("\nPer layanan:\n", (df.groupby("layanan")["terlambat"].mean() * 100).round(1))
per_kota = (df.groupby("kota")["terlambat"].mean() * 100).sort_values(ascending=False)
print("\nPer kota (%):\n", per_kota.round(1))

# ---------- ANALITIK DIAGNOSTIK ----------
garis("ANALITIK 2 - DIAGNOSTIK (mengapa terjadi?)")
df["kelompok_pengalaman"] = pd.cut(df["pengalaman_bulan"], [0, 12, 24, 48],
                                   labels=["<=12 bln", "13-24 bln", "25-48 bln"])
diag = {
    "Hujan deras": df.groupby("hujan_deras")["terlambat"].mean() * 100,
    "Order lewat cut-off (>=15.00)": df.groupby("lewat_cutoff")["terlambat"].mean() * 100,
    "Pengalaman kurir": df.groupby("kelompok_pengalaman", observed=True)["terlambat"].mean() * 100,
}
for nama, s in diag.items():
    print(f"\n{nama} (% terlambat):\n{s.round(1)}")

fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
per_kota.plot.bar(ax=ax[0], color="#4C78A8"); ax[0].set_title("Deskriptif: per Kota")
for a, (nama, s) in zip(ax[1:], diag.items()):
    s.plot.bar(ax=a, color="#F58518"); a.set_title("Diagnostik: " + nama, fontsize=9)
for a in ax: a.set_ylabel("% terlambat"); a.tick_params(axis="x", rotation=30)
plt.tight_layout(); plt.savefig("output/1_deskriptif_diagnostik.png", dpi=120); plt.close()

# ======================================================================
# 4. MODELING - ANALITIK 3 PREDIKTIF
# ======================================================================
garis("ANALITIK 3 - PREDIKTIF (apa yang akan terjadi?)")
num = ["jarak_km", "berat_kg", "jam_order", "pengalaman_bulan", "rating",
       "hujan_maks_mm", "hujan_deras", "akhir_pekan", "lewat_cutoff"]
kat = ["layanan", "kota"]
X, y = df[num + kat], df["terlambat"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
print(f"Train={len(X_tr)}  Test={len(X_te)}  (stratified: rasio terlambat dijaga sama)")

def pipa(model):
    pre = ColumnTransformer([("n", StandardScaler(), num),
                             ("k", OneHotEncoder(handle_unknown="ignore"), kat)])
    return Pipeline([("pre", pre), ("model", model)])

kandidat = {
    "Baseline (tebak mayoritas)": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                            class_weight="balanced", random_state=42),
}
hasil, model_fit = [], {}
for nama, m in kandidat.items():
    p = pipa(m).fit(X_tr, y_tr)
    model_fit[nama] = p
    pred, proba = p.predict(X_te), p.predict_proba(X_te)[:, 1]
    hasil.append({"Model": nama,
                  "Akurasi": accuracy_score(y_te, pred),
                  "Precision": precision_score(y_te, pred, zero_division=0),
                  "Recall": recall_score(y_te, pred),
                  "F1": f1_score(y_te, pred),
                  "ROC-AUC": roc_auc_score(y_te, proba)})
tabel = pd.DataFrame(hasil).set_index("Model").round(3)

# ======================================================================
# 5. EVALUATION
# ======================================================================
garis("5. EVALUATION (perbandingan model, mirip workflow KNIME)")
print(tabel)
print("\nPerhatikan: Baseline akurasinya tinggi tapi Recall = 0 -> akurasi menyesatkan!")
terbaik = tabel.drop("Baseline (tebak mayoritas)")["F1"].idxmax()
print(f"Model terbaik berdasarkan F1 kelas 'terlambat': {terbaik}")

tabel.drop("Baseline (tebak mayoritas)")[["Precision", "Recall", "F1", "ROC-AUC"]].plot.bar(
    figsize=(7, 4), rot=15, title="Perbandingan Model (data uji)")
plt.ylim(0, 1); plt.tight_layout(); plt.savefig("output/2_perbandingan_model.png", dpi=120); plt.close()

# ======================================================================
# 6. DEPLOYMENT - ANALITIK 4 PRESKRIPTIF + HUMAN-IN-THE-LOOP
# ======================================================================
garis("ANALITIK 4 - PRESKRIPTIF (apa yang harus dilakukan?)")
BIAYA_KOMPENSASI = 50_000   # biaya per paket terlambat (voucher + CS)
BIAYA_INTERVENSI = 8_000    # prioritas sortir / kurir tambahan per paket
EFEKTIF = 0.60              # intervensi menyelamatkan 60% paket berisiko

proba = model_fit[terbaik].predict_proba(X_te)[:, 1]
y_arr = y_te.values
tanpa = y_arr.sum() * BIAYA_KOMPENSASI
batas_uji = np.arange(.05, .96, .05)
biaya = []
for t in batas_uji:
    flag = proba >= t
    tp = ((flag) & (y_arr == 1)).sum()
    fn = ((~flag) & (y_arr == 1)).sum()
    biaya.append(flag.sum() * BIAYA_INTERVENSI
                 + (fn + tp * (1 - EFEKTIF)) * BIAYA_KOMPENSASI)
biaya = np.array(biaya)
t_opt = batas_uji[biaya.argmin()]
hemat = tanpa - biaya.min()
print(f"Biaya tanpa intervensi         : Rp {tanpa:,.0f}")
print(f"Biaya dengan intervensi optimal: Rp {biaya.min():,.0f} (ambang risiko {t_opt:.2f})")
print(f"Estimasi penghematan (data uji): Rp {hemat:,.0f} ({hemat/tanpa:.1%})")

plt.figure(figsize=(6, 4))
plt.plot(batas_uji, biaya / 1e6, marker="o")
plt.axhline(tanpa / 1e6, color="r", ls="--", label="Tanpa intervensi")
plt.axvline(t_opt, color="g", ls=":", label=f"Ambang optimal {t_opt:.2f}")
plt.xlabel("Ambang probabilitas"); plt.ylabel("Total biaya (juta Rp)")
plt.title("Preskriptif: memilih ambang intervensi"); plt.legend(); plt.tight_layout()
plt.savefig("output/3_preskriptif_biaya.png", dpi=120); plt.close()

# Faktor penting (untuk dijelaskan ke manajemen)
if terbaik == "Random Forest":
    nama_fitur = model_fit[terbaik]["pre"].get_feature_names_out()
    imp = pd.Series(model_fit[terbaik]["model"].feature_importances_, nama_fitur)
    print("\n5 faktor paling berpengaruh:\n", imp.sort_values(ascending=False).head(5).round(3))

# Keluaran untuk ditinjau MANUSIA (keputusan akhir tetap di tangan manusia)
rek = X_te.copy()
rek["risiko_terlambat"] = proba.round(3)
rek["rekomendasi"] = np.where(proba >= t_opt, "TINJAU: prioritaskan sortir / kurir senior",
                              "Proses normal")
rek = rek.sort_values("risiko_terlambat", ascending=False)
rek.to_csv("output/rekomendasi_untuk_ditinjau.csv")
print(f"\n{(rek['rekomendasi'] != 'Proses normal').sum()} dari {len(rek)} paket uji diberi tanda TINJAU.")
print("Daftar disimpan ke output/rekomendasi_untuk_ditinjau.csv")

garis("CATATAN PENUTUP (peran manusia)")
print("- Model hanya memberi probabilitas; manajer operasional yang memutuskan intervensi.")
print("- Model tidak tahu variabel di luar data (mis. demo/jalan ditutup) -> perlu pengetahuan domain.")
print("- Asumsi biaya & efektivitas intervensi harus divalidasi bisnis, lalu diuji (A/B test).")
