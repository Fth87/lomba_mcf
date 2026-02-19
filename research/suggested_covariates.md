# Rekomendasi Covariates untuk Dataset Asuransi Kesehatan

Berdasarkan analisis dataset (`Data_Polis.csv`, `Data_Klaim.csv`) dan riset industri asuransi kesehatan di Indonesia, berikut adalah _covariates_ (fitur tambahan) yang paling cocok untuk meningkatkan akurasi forecasting Anda.

## 1. Internal Covariates (Bisa dibuat langsung dari data Anda)

Ini adalah fitur yang paling "murah" dan efektif karena datanya sudah tersedia. Anda hanya perlu melakukan rekayasa fitur (_feature engineering_).

### A. Demografi Resiko (Dari Data Polis)

- **Kelompok Usia (Age Groups)**: Risiko kesehatan sangat berkorelasi dengan usia.
  - _Saran:_ Buat binning usia (misal: 0-5, 6-18, 19-35, 36-50, 50+).
- **Jenis Kelamin (Gender)**: Pola klaim wanita (misal: terkait kehamilan/melahirkan) berbeda dengan pria.
- **Lokasi (Domisili/Wilayah)**: Biaya medis di Jakarta (Urban) pasti beda dengan di daerah lain. Anda bisa map `Domisili` ke "Tier Kota" (Tier 1, 2, 3).
- **Lama Kepesertaan (Policy Duration)**:
  - _Rumus:_ `Tanggal Klaim` - `Tanggal Efektif Polis`.
  - _Insight:_ Polis baru mungkin punya masa tunggu (_waiting period_), polis lama mungkin klaim lebih stabil.

### B. Karakteristik Klaim (Dari Data Klaim)

- **Tipe Perawatan (Inpatient vs Outpatient)**: Ini adalah pembeda terbesar untuk _Severity_ (biaya). Rawat inap (IP) jauh lebih mahal dari rawat jalan (OP).
- **Jenis Penyakit (ICD groupings)**:
  - Gunakan huruf pertama kode ICD (misal `A00`, `C50`) untuk mengelompokkan penyakit (Infeksi, Kanker, dll). Penyakit kronis vs akut punya pola berulang yang berbeda.
- **Channel Klaim (Cashless vs Reimburse)**:
  - _Cashless_ biasanya lebih mahal dan tercatat real-time (lebih rapi).
  - _Reimburse_ bisa jadi ada _delay_ pelaporan (_IBNR - Incurred But Not Reported_) yang mempengaruhi tren bulanan.

## 2. External Covariates (Data Tambahan)

Jika Anda ingin model lebih canggih, tambahkan data eksternal ini (bisa dicari di internet/BPS):

### A. Kalender & Musiman (Sangat Penting di Indonesia)

- **Hari Libur Nasional (Hari Raya Idul Fitri/Natal)**:
  - Klaim _elektif_ (tidak darurat) biasanya turun drastis saat libur panjang.
  - Klaim kecelakaan/pencernaan mungkin naik pasca liburan.
- **Musim (Hujan vs Kemarau)**:
  - Indonesia punya pola penyakit musiman, misal **DBD (Demam Berdarah)** sering naik di musim hujan (Januari-Maret).

### B. Indikator Ekonomi (Makro)

- **Inflasi Medis (Medical Inflation)**: Biaya RS naik lebih cepat dari inflasi umum. Jika ada data inflasi kesehatan tahunan (biasanya 10-14% di Indo), ini sangat membantu prediksi _Severity_ di masa depan.
- **Kurs USD/IDR**: Banyak obat/alkes impor. Kenaikan Dollar bisa menaikkan rata-rata biaya klaim.

## 3. Rekomendasi Prioritas Implementasi

Untuk kompetisi/proyek ini, urutan prioritas terbaik adalah:

1.  **Time Features**: Bulan, Tahun, Jumlah Hari dalam Bulan, Libur Nasional.
2.  **Lag Features**: Rata-rata klaim 1-3 bulan lalu (Autoregressive).
3.  **Cross Features**: Rata-rata klaim per `Plan Code` atau `Age Group` di masa lalu.
